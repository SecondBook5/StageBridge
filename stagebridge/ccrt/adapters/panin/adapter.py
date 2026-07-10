"""PanIN edge-specific model-ready views and the top-level adapter.

Builds per-(transition_edge, fold, platform) partitions: a ``CCRTBatch`` of
source-state receivers with their typed sender context, plus a target semantic
population drawn from target-state receivers in the same fold/platform. Target
populations are population-level (no one-to-one pairing). Everything stays in
memory; nothing is written to the repository, and no training/HPO occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ...contracts.errors import CCRTValidationError
from ...data.batch import CCRTBatch
from ...representations import FeatureSpaceRegistry
from .config import PanINAdapterConfig
from .features import (
    PanINFeatureBlock,
    load_panin_feature_block,
    register_panin_feature_spaces,
)
from .neighborhoods import (
    PanINNeighborRecord,
    PanINSpatialObservation,
    build_continuous_sender_neighborhoods,
)
from .ontology import PanINOntology, build_panin_ontology
from .records import PanINRecordBundle, build_panin_record_bundle
from .source_audit import (
    PanINSourceAudit,
    audit_panin_source,
    validate_reference_source_audit,
)
from .splits import (
    PanINFoldAssignment,
    PanINGroupRecord,
    build_grouped_panin_folds,
    validate_no_panin_group_leakage,
)

__all__ = [
    "PanINEdgePartition",
    "PanINAdapterOutput",
    "PanINSpatialLoader",
    "adapt_reference_panin",
]


@dataclass(frozen=True)
class PanINEdgePartition:
    transition_edge_id: str
    source_receiver_state_id: str
    target_receiver_state_id: str
    fold_index: int
    platform: str
    source_receiver_ids: tuple[str, ...]
    target_receiver_ids: tuple[str, ...]
    source_batch: CCRTBatch
    target_semantic_features: torch.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PanINAdapterOutput:
    config: PanINAdapterConfig
    audit: PanINSourceAudit
    ontology: PanINOntology
    records: PanINRecordBundle
    edge_partitions: tuple[PanINEdgePartition, ...]
    feature_registry: FeatureSpaceRegistry
    validation_report: Any  # PanINValidationReport (avoid import cycle)


class PanINSpatialLoader:
    """Protocol-ish base: real sources subclass to yield spatial observations.

    In Milestone 8 the local environment has no biological data, so the concrete
    reader is exercised via fixtures in tests. The adapter accepts a loader so
    the real Xenium reader can be plugged in later without changing the pipeline.
    """

    def load(self, config: PanINAdapterConfig, audit: PanINSourceAudit) -> Sequence[PanINSpatialObservation]:
        raise NotImplementedError(
            "no concrete PanIN spatial loader is available in this environment; "
            "supply a loader once Xenium data are downloaded (see SOURCE_AUDIT.md)"
        )


def _build_edge_partitions(
    *,
    config: PanINAdapterConfig,
    ontology: PanINOntology,
    receiver_features: PanINFeatureBlock,
    sender_features: PanINFeatureBlock,
    semantic_features: PanINFeatureBlock,
    observations: Sequence[PanINSpatialObservation],
    neighbors: Sequence[PanINNeighborRecord],
    fold_by_group: Mapping[str, int],
    group_of: Mapping[str, str],
) -> list[PanINEdgePartition]:
    obs_by_id = {o.observation_id: o for o in observations}
    neighbors_by_receiver: dict[str, list[PanINNeighborRecord]] = {}
    for n in neighbors:
        neighbors_by_receiver.setdefault(n.receiver_id, []).append(n)

    recv_index = {oid: i for i, oid in enumerate(receiver_features.observation_ids)}
    sem_index = {oid: i for i, oid in enumerate(semantic_features.observation_ids)}
    send_index = {oid: i for i, oid in enumerate(sender_features.observation_ids)}
    d_s = sender_features.values.shape[1]

    partitions: list[PanINEdgePartition] = []
    for eid, src_state, tgt_state in config.transition_edges:
        # group receivers by (fold, platform)
        combos: dict[tuple[int, str], dict[str, list[PanINSpatialObservation]]] = {}
        for o in observations:
            if o.canonical_receiver_state_id not in (src_state, tgt_state):
                continue
            grp = group_of.get(o.observation_id)
            if grp is None or grp not in fold_by_group:
                continue
            key = (fold_by_group[grp], o.platform)
            combos.setdefault(key, {"source": [], "target": []})
            if o.canonical_receiver_state_id == src_state:
                combos[key]["source"].append(o)
            else:
                combos[key]["target"].append(o)

        for (fold_index, platform), pop in combos.items():
            sources = pop["source"]
            targets = pop["target"]
            if len(sources) < config.minimum_receivers_per_edge:
                continue
            if len(targets) < config.minimum_targets_per_edge:
                continue

            source_ids = [o.observation_id for o in sources]
            target_ids = [o.observation_id for o in targets]

            # receiver feature matrix [B, D_R]
            receiver_mat = torch.stack(
                [receiver_features.values[recv_index[i]] for i in source_ids], dim=0
            )
            # source semantic [B, D_Z]
            source_sem = torch.stack(
                [semantic_features.values[sem_index[i]] for i in source_ids], dim=0
            )
            # target semantic [M, D_Z]
            target_sem = torch.stack(
                [semantic_features.values[sem_index[i]] for i in target_ids], dim=0
            )

            # padded sender context [B, K, D_S] + parallel metadata
            b = len(source_ids)
            k = max(
                (len(neighbors_by_receiver.get(i, [])) for i in source_ids), default=0
            )
            k = max(k, 1)
            sender_feat = torch.zeros((b, k, d_s), dtype=receiver_features.values.dtype)
            sender_mask = [[0] * k for _ in range(b)]
            distance = [[0.0] * k for _ in range(b)]
            type_ids: list[list[Any]] = [[None] * k for _ in range(b)]
            for bi, rid in enumerate(source_ids):
                ns = sorted(
                    neighbors_by_receiver.get(rid, []),
                    key=lambda n: (n.rank, n.sender_id),
                )
                for j, n in enumerate(ns[:k]):
                    if n.sender_id in send_index:
                        sender_feat[bi, j] = sender_features.values[send_index[n.sender_id]]
                    sender_mask[bi][j] = 1
                    distance[bi][j] = float(n.distance_to_receiver)
                    type_ids[bi][j] = n.sender_context_type_id

            batch = CCRTBatch(
                receiver_features=receiver_mat.tolist(),
                sender_features=sender_feat.tolist(),
                sender_mask=sender_mask,
                distance_to_receiver=distance,
                biological_system_id=[config.biological_system_id] * b,
                transition_edge_id=[eid] * b,
                receiver_state_id=[src_state] * b,
                semantic_features=source_sem.tolist(),
                sender_context_type_ids=type_ids,
            )
            batch.validate()

            partitions.append(
                PanINEdgePartition(
                    transition_edge_id=eid,
                    source_receiver_state_id=src_state,
                    target_receiver_state_id=tgt_state,
                    fold_index=fold_index,
                    platform=platform,
                    source_receiver_ids=tuple(source_ids),
                    target_receiver_ids=tuple(target_ids),
                    source_batch=batch,
                    target_semantic_features=target_sem,
                    provenance={
                        "num_source": b,
                        "num_target": len(target_ids),
                        "max_sender_context": k,
                    },
                )
            )
    return partitions


def adapt_reference_panin(
    config: PanINAdapterConfig,
    *,
    spatial_loader: PanINSpatialLoader | None = None,
) -> PanINAdapterOutput:
    """Run the PanIN adapter end-to-end (in memory)."""
    from .validation import validate_panin_adapter_output  # local import: avoid cycle

    # 1-2) audit + validate
    audit = audit_panin_source(config.source_root)
    validate_reference_source_audit(audit, config)

    # 3) ontology
    ontology = build_panin_ontology(config)

    # 4) feature blocks
    receiver_block = load_panin_feature_block(config.receiver_feature_block, source_root=config.source_root)
    sender_block = load_panin_feature_block(config.sender_feature_block, source_root=config.source_root)
    semantic_block = load_panin_feature_block(config.semantic_feature_block, source_root=config.source_root)
    regulatory_block = (
        load_panin_feature_block(config.regulatory_feature_block, source_root=config.source_root)
        if config.regulatory_feature_block is not None
        else None
    )
    registry = register_panin_feature_spaces(
        [receiver_block, sender_block, semantic_block]
        + ([regulatory_block] if regulatory_block else [])
    )

    # 5) spatial observations (via a source loader)
    if spatial_loader is None:
        raise CCRTValidationError(
            "adapt_reference_panin requires a spatial_loader; no concrete PanIN "
            "loader is available in this environment (see SOURCE_AUDIT.md)"
        )
    observations = list(spatial_loader.load(config, audit))

    # 6-7) receivers / candidate senders are the typed observations
    receivers = [o for o in observations if o.canonical_receiver_state_id is not None]
    senders = [o for o in observations if o.canonical_context_type_id is not None]

    # 8) continuous neighborhoods
    neighbors = build_continuous_sender_neighborhoods(
        receivers=receivers, candidate_senders=senders, config=config.neighborhood
    )

    # 9) grouped folds
    groups, group_of = _build_groups(config, observations)
    fold_assignments = build_grouped_panin_folds(groups=groups, config=config.splits)
    validate_no_panin_group_leakage(fold_assignments, observations)
    fold_by_group = {a.group_id: a.fold_index for a in fold_assignments}

    # 10) canonical records
    records = build_panin_record_bundle(
        config=config, audit=audit, ontology=ontology,
        receiver_features=receiver_block, sender_features=sender_block,
        semantic_features=semantic_block, regulatory_features=regulatory_block,
        spatial_observations=observations, neighbors=neighbors,
        fold_assignments=fold_assignments,
    )

    # 11) edge/fold/platform partitions
    partitions = _build_edge_partitions(
        config=config, ontology=ontology, receiver_features=receiver_block,
        sender_features=sender_block, semantic_features=semantic_block,
        observations=observations, neighbors=neighbors,
        fold_by_group=fold_by_group, group_of=group_of,
    )

    output = PanINAdapterOutput(
        config=config, audit=audit, ontology=ontology, records=records,
        edge_partitions=tuple(partitions), feature_registry=registry,
        validation_report=None,
    )
    # 12) validate partitions + attach report
    report = validate_panin_adapter_output(output)
    return PanINAdapterOutput(
        config=config, audit=audit, ontology=ontology, records=records,
        edge_partitions=tuple(partitions), feature_registry=registry,
        validation_report=report,
    )


def _build_groups(
    config: PanINAdapterConfig, observations: Sequence[PanINSpatialObservation]
) -> tuple[list[PanINGroupRecord], dict[str, str]]:
    """Group observations by donor (or sample) and map each observation to its group."""
    level = config.splits.grouping_level
    group_members: dict[str, dict[str, set]] = {}
    group_of: dict[str, str] = {}
    for o in observations:
        gid = (o.donor_id if level == "donor" else o.sample_id) or ""
        if level == "donor" and not gid:
            raise CCRTValidationError(
                f"observation '{o.observation_id}' lacks donor_id for donor grouping"
            )
        group_of[o.observation_id] = gid
        gm = group_members.setdefault(
            gid, {"samples": set(), "sections": set(), "platforms": set(), "stages": set()}
        )
        gm["samples"].add(o.sample_id)
        gm["sections"].add(o.section_id)
        gm["platforms"].add(o.platform)
        if o.canonical_receiver_state_id is not None:
            gm["stages"].add(o.canonical_receiver_state_id)

    groups = [
        PanINGroupRecord(
            group_id=gid,
            donor_id=gid if level == "donor" else None,
            sample_ids=tuple(sorted(gm["samples"])),
            section_ids=tuple(sorted(gm["sections"])),
            platforms=tuple(sorted(gm["platforms"])),
            canonical_stage_ids=tuple(sorted(gm["stages"])),
        )
        for gid, gm in sorted(group_members.items())
    ]
    return groups, group_of
