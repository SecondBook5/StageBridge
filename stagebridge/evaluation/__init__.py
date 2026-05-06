"""Evaluation metrics and results aggregation for StageBridge."""

from stagebridge.evaluation.metrics import (
    compute_wasserstein,
    compute_mmd,
    compute_displacement,
    compute_stage_accuracy,
    evaluate_predictions,
)

from stagebridge.evaluation.results_summary import (
    generate_summary,
    summary_to_dict,
    summary_to_dataframe,
    summary_to_markdown,
    ResultsSummary,
    ModelResult,
    MetricSummary,
)

__all__ = [
    # Metrics
    "compute_wasserstein",
    "compute_mmd",
    "compute_displacement",
    "compute_stage_accuracy",
    "evaluate_predictions",
    # Results summary
    "generate_summary",
    "summary_to_dict",
    "summary_to_dataframe",
    "summary_to_markdown",
    "ResultsSummary",
    "ModelResult",
    "MetricSummary",
]
