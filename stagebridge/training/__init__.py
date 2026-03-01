"""Training loop, losses, and evaluation for StageBridge."""

from .eval import evaluate_transition, summarize_fold_metrics
from .losses import build_sinkhorn_coupling, flow_matching_loss, sinkhorn_distance
from .trainer import (
    DonorSplit,
    StageBridgeTrainer,
    build_donor_holdout_splits,
    build_samplers_from_anndata,
    donors_with_min_stage_coverage,
)

__all__ = [
    "DonorSplit",
    "StageBridgeTrainer",
    "build_donor_holdout_splits",
    "build_samplers_from_anndata",
    "donors_with_min_stage_coverage",
    "build_sinkhorn_coupling",
    "flow_matching_loss",
    "sinkhorn_distance",
    "evaluate_transition",
    "summarize_fold_metrics",
]
