"""Preprocessing pipeline: normalization, QC, harmonisation, and stage ontology."""

from .stage_ontology import (
    CANONICAL_STAGE_ORDER,
    StageTransition,
    apply_stage_ontology,
    infer_stage_pair_id,
    normalize_stage_label,
    ordered_transitions,
    stage_to_index,
)

__all__ = [
    "CANONICAL_STAGE_ORDER",
    "StageTransition",
    "apply_stage_ontology",
    "infer_stage_pair_id",
    "normalize_stage_label",
    "ordered_transitions",
    "stage_to_index",
]
