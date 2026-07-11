"""LUAD edge-specific model-ready views and the top-level adapter.

Builds per-(transition_edge, fold, platform, backend) partitions: a ``CCRTBatch``
of source-state Visium receivers with their typed deconvolved context (from one
preserved backend), plus a target semantic population drawn from target-state
receivers in the same fold/platform. Target populations are population-level (no
one-to-one pairing). A batch never mixes backends. Everything stays in memory;
nothing is written to the repository, and no training/HPO occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ...contracts.errors import CCRTValidationError
from ...data.batch import CCRTBatch
from ...representations import FeatureSpaceRegistry
from .config import LUADAdapterConfig
from .context_components import (
    LUADContextComponent,
    load_luad_context_components,
)
from .features import (
    LUADFeatureBlock,
    load_luad_feature_block,
    register_luad_feature_spaces,
)
from .manifests import (
    LUADModalityRecord,
    LUADModalityRelationship,
    build_luad_modality_manifest,
    validate_luad_modality_manifest,
)
from .neighborhoods import (
    LUADReceiverContextRecord,
    LUADSpatialSpot,
    build_luad_context_neighborhoods,
)
from .ontology import LUADOntology, build_luad_ontology
from .records import LUADRecordBundle, build_luad_record_bundle
from .source_audit import (
    LUADSourceAudit,
    audit_luad_source,
    validate_reference_source_audit,
)
from .splits import (
    LUADGroupRecord,
    build_grouped_luad_folds,
    validate_no_luad_group_leakage,
)

__all__ = [
    "LUADEdgePartition",
    "LUADAdapterOutput",
    "LUADSpatialLoader",
    "adapt_reference_luad",
]


@dataclass(frozen=True)
class LUADEdgePartition:
    transition_edge_id: str
    source_receiver_state_id: str
    target_receiver_state_id: str
    fold_index: int
    platform: str
    context_backend_id: str
    source_receiver_ids: tuple[str, ...]
    target_receiver_ids: tuple[str, ...]
    source_batch: CCRTBatch
    target_semantic_features: torch.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class LUADAdapterOutput:
    config: LUADAdapterConfig
    audit: LUADSourceAudit
    ontology: LUADOntology
    records: LUADRecordBundle
    edge_partitions: tuple[LUADEdgePartition, ...]
    feature_registry: FeatureSpaceRegistry
    modalities: tuple[LUADModalityRecord, ...]
    modality_relationships: tuple[LUADModalityRelationship, ...]
    context_components: tuple[LUADContextComponent, ...]
    validation_report: Any  # LUADValidationReport (avoid import cycle)


class LUADSpatialLoader:
    """Protocol-ish base: real sources subclass to yield spatial spots + modalities.

    The local environment does not load the giant h5ad objects; the concrete
    reader (Space Ranger coordinates -> microns, niche/lesion association) is
    exercised via fixtures in tests. The adapter accepts a loader so the real
    Visium reader can be plugged in later without changing the pipeline.
    """

    def load_spots(
        self, config: LUADAdapterConfig, audit: LUADSourceAudit
    ) -> Sequence[LUADSpatialSpot]:
        raise NotImplementedError(
            "no concrete LUAD spatial loader is available in this environment; "
            "supply a loader once Visium coordinates are resolved (see SOURCE_AUDIT.md)"
        )

    def load_modalities(
        self, config: LUADAdapterConfig, audit: LUADSourceAudit
    ) -> Sequence[LUADModalityRecord]:
        raise NotImplementedError(
            "no concrete LUAD modality loader is available in this environment"
        )


def _build_groups(
    config: LUADAdapterConfig, receivers: Sequence[LUADSpatialSpot]
) -> tuple[list[LUADGroupRecord], dict[str, str]]:
    """Group receivers by donor (or sample); map each receiver to its group."""
    level = config.splits.grouping_level
    group_members: dict[str, dict[str, set]] = {}
    group_of: dict[str, str] = {}
    for r in receivers:
        gid = (r.donor_id if level == "donor" else r.sample_id) or ""
        if level == "donor" and not gid:
            raise CCRTValidationError(
                f"receiver '{r.spot_id}' lacks donor_id for donor grouping"
            )
        group_of[r.spot_id] = gid
        gm = group_members.setdefault(
            gid,
            {"samples": set(), "sections": set(), "platforms": set(), "stages": set()},
        )
        gm["samples"].add(r.sample_id)
        gm["sections"].add(r.section_id)
        gm["platforms"].add(r.platform)
        if r.canonical_receiver_state_id is not None:
            gm["stages"].add(r.canonical_receiver_state_id)

    backend_id = config.context_backend.backend_id
    groups = [
        LUADGroupRecord(
            group_id=gid,
            donor_id=gid if level == "donor" else None,
            sample_ids=tuple(sorted(gm["samples"])),
            section_ids=tuple(sorted(gm["sections"])),
            platforms=tuple(sorted(gm["platforms"])),
            backend_ids=(backend_id,),
            canonical_stage_ids=tuple(sorted(gm["stages"])),
        )
        for gid, gm in sorted(group_members.items())
    ]
    return groups, group_of


def _build_edge_partitions(
    *,
    config: LUADAdapterConfig,
    receivers: Sequence[LUADSpatialSpot],
    receiver_features: LUADFeatureBlock,
    semantic_features: LUADFeatureBlock,
    context_records: Sequence[LUADReceiverContextRecord],
    component_profiles: Mapping[str, tuple[float, ...]],
    context_feature_dim: int,
    fold_by_group: Mapping[str, int],
    group_of: Mapping[str, str],
) -> list[LUADEdgePartition]:
    recv_niche = {r.spot_id: r.niche_id for r in receivers}
    recv_index = {oid: i for i, oid in enumerate(receiver_features.observation_ids)}
    sem_index = {oid: i for i, oid in enumerate(semantic_features.observation_ids)}

    # context grouped by (receiver_id, backend_id)
    ctx_by_receiver_backend: dict[tuple[str, str], list[LUADReceiverContextRecord]] = {}
    backends: set[str] = set()
    for c in context_records:
        ctx_by_receiver_backend.setdefault((c.receiver_id, c.backend_id), []).append(c)
        backends.add(c.backend_id)
    # ensure at least the configured backend is represented for partitioning
    backends.add(config.context_backend.backend_id)

    dtype = receiver_features.values.dtype
    partitions: list[LUADEdgePartition] = []

    for eid, src_state, tgt_state in config.transition_edges:
        # group receivers by (fold, platform)
        combos: dict[tuple[int, str], dict[str, list[LUADSpatialSpot]]] = {}
        for r in receivers:
            if r.canonical_receiver_state_id not in (src_state, tgt_state):
                continue
            grp = group_of.get(r.spot_id)
            if grp is None or grp not in fold_by_group:
                continue
            key = (fold_by_group[grp], r.platform)
            combos.setdefault(key, {"source": [], "target": []})
            if r.canonical_receiver_state_id == src_state:
                combos[key]["source"].append(r)
            else:
                combos[key]["target"].append(r)

        for (fold_index, platform), pop in combos.items():
            sources = pop["source"]
            targets = pop["target"]
            if len(sources) < config.minimum_receivers_per_edge:
                continue
            if len(targets) < config.minimum_targets_per_edge:
                continue

            source_ids = [r.spot_id for r in sources]
            target_ids = [r.spot_id for r in targets]

            # target semantic [M, D_Z] (population-level; backend-independent)
            target_sem = torch.stack(
                [semantic_features.values[sem_index[recv_niche[i]]] for i in target_ids],
                dim=0,
            )

            for backend_id in sorted(backends):
                # receiver feature matrix [B, D_R]
                receiver_mat = torch.stack(
                    [receiver_features.values[recv_index[recv_niche[i]]] for i in source_ids],
                    dim=0,
                )
                # source semantic [B, D_Z]
                source_sem = torch.stack(
                    [semantic_features.values[sem_index[recv_niche[i]]] for i in source_ids],
                    dim=0,
                )

                b = len(source_ids)
                k = max(
                    (
                        len(ctx_by_receiver_backend.get((i, backend_id), []))
                        for i in source_ids
                    ),
                    default=0,
                )
                k = max(k, 1)
                sender_feat = torch.zeros((b, k, context_feature_dim), dtype=dtype)
                sender_mask = [[0] * k for _ in range(b)]
                distance = [[0.0] * k for _ in range(b)]
                uncertainty = [[0.0] * k for _ in range(b)]
                type_ids: list[list[Any]] = [[None] * k for _ in range(b)]

                for bi, rid in enumerate(source_ids):
                    ctx = sorted(
                        ctx_by_receiver_backend.get((rid, backend_id), []),
                        key=lambda c: (c.rank, c.component_id),
                    )
                    for j, c in enumerate(ctx[:k]):
                        sender_mask[bi][j] = 1
                        distance[bi][j] = float(c.distance_to_receiver)
                        uncertainty[bi][j] = float(c.uncertainty)
                        type_ids[bi][j] = c.sender_context_type_id
                    # feature vectors are attached from the component profiles
                    for j, c in enumerate(ctx[:k]):
                        comp_vec = component_profiles.get(c.component_id)
                        if comp_vec is not None:
                            sender_feat[bi, j] = torch.tensor(comp_vec, dtype=dtype)

                batch = CCRTBatch(
                    receiver_features=receiver_mat.tolist(),
                    sender_features=sender_feat.tolist(),
                    sender_mask=sender_mask,
                    distance_to_receiver=distance,
                    uncertainty=uncertainty,
                    biological_system_id=[config.biological_system_id] * b,
                    transition_edge_id=[eid] * b,
                    receiver_state_id=[src_state] * b,
                    semantic_features=source_sem.tolist(),
                    sender_context_type_ids=type_ids,
                )
                batch.validate()

                partitions.append(
                    LUADEdgePartition(
                        transition_edge_id=eid,
                        source_receiver_state_id=src_state,
                        target_receiver_state_id=tgt_state,
                        fold_index=fold_index,
                        platform=platform,
                        context_backend_id=backend_id,
                        source_receiver_ids=tuple(source_ids),
                        target_receiver_ids=tuple(target_ids),
                        source_batch=batch,
                        target_semantic_features=target_sem,
                        provenance={
                            "num_source": b,
                            "num_target": len(target_ids),
                            "max_sender_context": k,
                            "context_backend_id": backend_id,
                        },
                    )
                )
    return partitions


def adapt_reference_luad(
    config: LUADAdapterConfig,
    *,
    spatial_loader: LUADSpatialLoader | None = None,
) -> LUADAdapterOutput:
    """Run the LUAD multimodal adapter end-to-end (in memory)."""
    from .validation import validate_luad_adapter_output  # local import: avoid cycle

    # 1-2) audit + validate
    audit = audit_luad_source(config.source_root)
    validate_reference_source_audit(audit, config)

    # 3) ontology
    ontology = build_luad_ontology(config)

    # 4) feature blocks
    receiver_block = load_luad_feature_block(
        config.receiver_feature_block, source_root=config.source_root
    )
    semantic_block = load_luad_feature_block(
        config.semantic_feature_block, source_root=config.source_root
    )
    regulatory_block = (
        load_luad_feature_block(
            config.regulatory_feature_block, source_root=config.source_root
        )
        if config.regulatory_feature_block is not None
        else None
    )
    registry = register_luad_feature_spaces(
        [receiver_block, semantic_block] + ([regulatory_block] if regulatory_block else [])
    )

    # 5) spatial spots + modality manifest (via a source loader)
    if spatial_loader is None:
        raise CCRTValidationError(
            "adapt_reference_luad requires a spatial_loader; no concrete LUAD loader "
            "is available in this environment (see SOURCE_AUDIT.md)"
        )
    spots = list(spatial_loader.load_spots(config, audit))
    spots_by_id = {s.spot_id: s for s in spots}
    receivers = [s for s in spots if s.canonical_receiver_state_id is not None]

    modalities = list(spatial_loader.load_modalities(config, audit))
    modalities, relationships = build_luad_modality_manifest(modalities)
    validate_luad_modality_manifest(modalities, relationships)

    # 6) deconvolved typed-context components (one preserved backend)
    context_components = load_luad_context_components(
        config.context_backend,
        source_root=config.source_root,
        sender_context_annotation_map=config.sender_context_annotation_map,
        strict_unknown_annotations=config.strict_unknown_annotations,
        excluded_annotations=config.excluded_annotations,
    )

    # 7) continuous, same-donor/sample/section/platform/backend neighborhoods
    context_records = build_luad_context_neighborhoods(
        receivers=receivers,
        context_components=context_components,
        spots_by_id=spots_by_id,
        config=config.neighborhood,
    )

    # component profiles for partition assembly (keeps neighbor records lean)
    component_profiles = {
        comp.component_id: comp.feature_vector for comp in context_components
    }

    # 8) grouped folds (donor = patient; all backends of a donor stay together)
    groups, group_of = _build_groups(config, receivers)
    fold_assignments = build_grouped_luad_folds(groups=groups, config=config.splits)
    validate_no_luad_group_leakage(fold_assignments)
    fold_by_group = {a.group_id: a.fold_index for a in fold_assignments}

    # 9) canonical records
    records = build_luad_record_bundle(
        config=config,
        audit=audit,
        ontology=ontology,
        receiver_features=receiver_block,
        semantic_features=semantic_block,
        regulatory_features=regulatory_block,
        receivers=receivers,
        context_records=context_records,
        fold_assignments=fold_assignments,
        modalities=modalities,
        modality_relationships=relationships,
    )

    # 10) edge/fold/platform/backend partitions
    partitions = _build_edge_partitions(
        config=config,
        receivers=receivers,
        receiver_features=receiver_block,
        semantic_features=semantic_block,
        context_records=context_records,
        component_profiles=component_profiles,
        context_feature_dim=len(config.context_backend.feature_ids),
        fold_by_group=fold_by_group,
        group_of=group_of,
    )

    output = LUADAdapterOutput(
        config=config,
        audit=audit,
        ontology=ontology,
        records=records,
        edge_partitions=tuple(partitions),
        feature_registry=registry,
        modalities=tuple(modalities),
        modality_relationships=tuple(relationships),
        context_components=tuple(context_components),
        validation_report=None,
    )
    # 11) validate + attach report
    report = validate_luad_adapter_output(output)
    return LUADAdapterOutput(
        config=config,
        audit=audit,
        ontology=ontology,
        records=records,
        edge_partitions=tuple(partitions),
        feature_registry=registry,
        modalities=tuple(modalities),
        modality_relationships=tuple(relationships),
        context_components=tuple(context_components),
        validation_report=report,
    )
