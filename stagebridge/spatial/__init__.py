"""Spatial deconvolution backend comparison and selection."""

from stagebridge.spatial.backend_comparison import (
    METRICS_CONFIG,
    FALLBACK_METRICS,
    compute_composite_score,
    load_backend_metrics,
    compare_backends,
    select_canonical_backend,
)

__all__ = [
    "METRICS_CONFIG",
    "FALLBACK_METRICS",
    "compute_composite_score",
    "load_backend_metrics",
    "compare_backends",
    "select_canonical_backend",
]
