"""Evaluation and tissue-level interpretation."""

from stagebridge.evaluation.metrics import (
    compute_all_metrics,
    compute_ari,
    compute_batch_integration_metrics,
    compute_kbet,
    compute_lisi,
    compute_nmi,
    compute_representation_metrics,
    compute_silhouette,
    expected_calibration_error,
    maximum_mean_discrepancy,
    MetricsTracker,
    wasserstein_nd_distance,
)

__all__ = [
    "compute_all_metrics",
    "compute_ari",
    "compute_batch_integration_metrics",
    "compute_kbet",
    "compute_lisi",
    "compute_nmi",
    "compute_representation_metrics",
    "compute_silhouette",
    "expected_calibration_error",
    "maximum_mean_discrepancy",
    "MetricsTracker",
    "wasserstein_nd_distance",
]
