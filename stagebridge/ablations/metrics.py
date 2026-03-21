"""
Ablation Metrics - compute and compare ablation results.
"""

import numpy as np
from typing import Any


def compute_ablation_metrics(results: dict[str, Any]) -> dict[str, float]:
    """
    Compute standard metrics from ablation results.

    Parameters
    ----------
    results : dict
        Raw results from ablation run

    Returns
    -------
    dict
        Computed metrics
    """
    metrics = results.get("metrics", {})

    return {
        "transition_loss": metrics.get("transition_loss", np.nan),
        "donor_held_out_accuracy": metrics.get("donor_held_out_accuracy", np.nan),
        "calibration_error": metrics.get("calibration_error", np.nan),
        "nll": metrics.get("nll", np.nan),
        "uncertainty_quality": metrics.get("uncertainty_quality", np.nan),
    }


def compute_degradation(
    baseline_metrics: dict[str, float],
    ablation_metrics: dict[str, float],
) -> dict[str, float]:
    """
    Compute degradation from baseline for each metric.

    Parameters
    ----------
    baseline_metrics : dict
        Metrics from full model
    ablation_metrics : dict
        Metrics from ablation

    Returns
    -------
    dict
        Degradation values (positive = ablation is worse)
    """
    degradation = {}

    for key in baseline_metrics:
        if key not in ablation_metrics:
            continue

        base_val = baseline_metrics[key]
        abl_val = ablation_metrics[key]

        if np.isnan(base_val) or np.isnan(abl_val):
            degradation[key] = np.nan
            continue

        # For loss metrics, higher is worse
        if "loss" in key or "error" in key or "nll" in key:
            degradation[key] = abl_val - base_val
        # For accuracy/quality metrics, lower is worse
        else:
            degradation[key] = base_val - abl_val

    return degradation


def compute_effect_size(
    baseline_values: list[float],
    ablation_values: list[float],
) -> float:
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    baseline_values : list
        Values from baseline (across folds/seeds)
    ablation_values : list
        Values from ablation

    Returns
    -------
    float
        Cohen's d effect size
    """
    baseline = np.array(baseline_values)
    ablation = np.array(ablation_values)

    pooled_std = np.sqrt((baseline.std() ** 2 + ablation.std() ** 2) / 2)

    if pooled_std == 0:
        return 0.0

    return (baseline.mean() - ablation.mean()) / pooled_std


def summarize_degradation(degradation: dict[str, float]) -> str:
    """
    Summarize degradation level as human-readable category.

    Returns one of: "significant", "moderate", "minimal", "none"
    """
    values = [v for v in degradation.values() if not np.isnan(v)]

    if not values:
        return "unknown"

    avg_degradation = np.mean(np.abs(values))

    if avg_degradation > 0.2:
        return "significant"
    elif avg_degradation > 0.1:
        return "moderate"
    elif avg_degradation > 0.01:
        return "minimal"
    else:
        return "none"
