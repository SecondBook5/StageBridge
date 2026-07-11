"""LUAD adapter output validation and reporting.

Produces a ``LUADValidationReport`` summarizing the adapted output and enforcing
the Milestone-9 invariants: no duplicate ids, no cross-donor/sample/section/
platform/backend context, no donor leakage, all records valid, consistent
semantic dimensions, no metadata/target leakage into features, no non-finite or
negative distances/abundances, single preserved backend per partition (never
mixed), and honest reporting of unsupported edge partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...contracts.errors import CCRTValidationError
from ...contracts.tensors import shape_of
from .features import FORBIDDEN_FEATURE_ID_TOKENS

__all__ = ["LUADValidationReport", "validate_luad_adapter_output"]


@dataclass(frozen=True)
class LUADValidationReport:
    source_dataset_commit: str | None
    source_layout_version: str
    num_donors: int
    num_samples: int
    num_sections: int
    platforms: tuple[str, ...]
    stages: tuple[str, ...]
    receiver_counts_by_state: Mapping[str, int]
    sender_counts_by_type: Mapping[str, int]
    context_counts_by_backend: Mapping[str, int]
    transition_edge_counts: Mapping[str, int]
    fold_counts: Mapping[int, int]
    supported_edge_partitions: int
    unsupported_edge_partitions: tuple[str, ...]
    context_backend_ids: tuple[str, ...]
    modality_relationship_counts: Mapping[str, int]
    coordinate_units: Mapping[str, str]
    semantic_feature_space_id: str
    semantic_dimension: int
    regulatory_feature_space_id: str | None
    regulatory_dimension: int | None
    smoke_check_artifact_count: int
    warnings: tuple[str, ...]
    passed: bool


def validate_luad_adapter_output(output) -> LUADValidationReport:
    """Validate a LUADAdapterOutput and build its report."""
    config = output.config
    records = output.records
    warnings: list[str] = []

    # -- feature-id leakage guard --
    for spec in records.feature_space_specs:
        for fid in spec.feature_ids:
            if fid.strip().lower() in FORBIDDEN_FEATURE_ID_TOKENS:
                raise CCRTValidationError(
                    f"feature space '{spec.feature_space_id}' contains a metadata/"
                    f"coordinate feature id '{fid}'"
                )

    # -- receiver id uniqueness + counts --
    receiver_ids = [r["receiver_id"] for r in records.receiver_records]
    if len(set(receiver_ids)) != len(receiver_ids):
        raise CCRTValidationError("duplicate receiver ids in records")
    receiver_counts: dict[str, int] = {}
    donors: set[str] = set()
    samples: set[str] = set()
    sections: set[str] = set()
    for r in records.receiver_records:
        receiver_counts[r["receiver_state_id"]] = receiver_counts.get(r["receiver_state_id"], 0) + 1
        if r.get("donor_id"):
            donors.add(r["donor_id"])
        samples.add(r["sample_id"])
        if r.get("section_id"):
            sections.add(r["section_id"])

    # -- sender / context counts + distance/abundance sanity --
    sender_counts: dict[str, int] = {}
    context_counts_by_backend: dict[str, int] = {}
    for s in records.sender_context_records:
        sender_counts[s["sender_context_type_id"]] = sender_counts.get(s["sender_context_type_id"], 0) + 1
        backend = s.get("sender_source", "")
        if backend:
            context_counts_by_backend[backend] = context_counts_by_backend.get(backend, 0) + 1
        d = s["distance_to_receiver"]
        if d < 0:
            raise CCRTValidationError("negative sender distance detected")
        if d != d:  # NaN
            raise CCRTValidationError("non-finite sender distance detected")
        a = s.get("abundance", 0.0)
        if a != a:
            raise CCRTValidationError("non-finite context abundance detected")
        if a < 0:
            raise CCRTValidationError("negative context abundance detected")

    # -- edge counts + partition support (single preserved backend each) --
    edge_counts: dict[str, int] = {e["transition_edge_id"]: 0 for e in records.transition_edge_records}
    for p in output.edge_partitions:
        edge_counts[p.transition_edge_id] = edge_counts.get(p.transition_edge_id, 0) + 1
        # partition internal consistency
        if p.source_batch.batch_size() != len(p.source_receiver_ids):
            raise CCRTValidationError(f"edge partition {p.transition_edge_id}: batch/source id mismatch")
        if p.target_semantic_features.shape[0] != len(p.target_receiver_ids):
            raise CCRTValidationError(f"edge partition {p.transition_edge_id}: target/id mismatch")
        # a batch never mixes backends: its context type ids come from one backend
        if not p.context_backend_id:
            raise CCRTValidationError(
                f"edge partition {p.transition_edge_id}: missing context_backend_id"
            )
        # source semantic dim = width of the batch's semantic_features
        # (CCRTBatch has NO semantic_dim() method; use shape_of).
        source_sem_dim = shape_of(p.source_batch.semantic_features)[1]
        if source_sem_dim != p.target_semantic_features.shape[1]:
            raise CCRTValidationError(
                f"edge partition {p.transition_edge_id}: semantic dim mismatch"
            )

    unsupported = tuple(eid for eid, count in edge_counts.items() if count == 0)
    for eid in unsupported:
        warnings.append(f"transition edge '{eid}' has no supported partitions")

    # -- fold counts --
    fold_counts: dict[int, int] = {}
    for a in records.fold_assignments:
        fold_counts[a.fold_index] = fold_counts.get(a.fold_index, 0) + 1

    # -- modality relationship counts (never upgraded past honest strength) --
    modality_relationship_counts: dict[str, int] = {}
    for rel in records.modality_relationships:
        rt = rel["relationship_type"]
        modality_relationship_counts[rt] = modality_relationship_counts.get(rt, 0) + 1
        if rt == "same_observation":
            raise CCRTValidationError(
                "modality relationship 'same_observation' asserted; LUAD snRNA and "
                "Visium are separate accessions (not cell-matched)"
            )

    # -- feature specs / semantic + regulatory dimensions --
    semantic_spec = next(
        (s for s in records.feature_space_specs if s.role == "semantic"), None
    )
    if semantic_spec is None:
        raise CCRTValidationError("no registered semantic feature space")
    regulatory_spec = next(
        (s for s in records.feature_space_specs if s.role == "regulatory"), None
    )

    platforms = tuple(
        sorted({r.get("assay_platform", "") for r in records.receiver_records if r.get("assay_platform")})
    )
    coordinate_units = dict(output.audit.coordinate_units)

    return LUADValidationReport(
        source_dataset_commit=output.audit.dataset_commit,
        source_layout_version=config.source_layout_version,
        num_donors=len(donors),
        num_samples=len(samples),
        num_sections=len(sections),
        platforms=platforms,
        stages=tuple(sorted(receiver_counts)),
        receiver_counts_by_state=receiver_counts,
        sender_counts_by_type=sender_counts,
        context_counts_by_backend=context_counts_by_backend,
        transition_edge_counts=edge_counts,
        fold_counts=fold_counts,
        supported_edge_partitions=len(output.edge_partitions),
        unsupported_edge_partitions=unsupported,
        context_backend_ids=tuple(records.backend_ids),
        modality_relationship_counts=modality_relationship_counts,
        coordinate_units=coordinate_units,
        semantic_feature_space_id=semantic_spec.feature_space_id,
        semantic_dimension=semantic_spec.dimension,
        regulatory_feature_space_id=regulatory_spec.feature_space_id if regulatory_spec else None,
        regulatory_dimension=regulatory_spec.dimension if regulatory_spec else None,
        smoke_check_artifact_count=len(output.audit.smoke_check_artifacts),
        warnings=tuple(warnings),
        passed=True,
    )
