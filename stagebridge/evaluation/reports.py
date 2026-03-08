"""Evaluation report assembly for Mission 3 edge runs."""
from __future__ import annotations

from typing import Any


def build_evaluation_report(
    *,
    edge_label: str,
    mode: str,
    heldout_metrics: dict[str, float],
    calibration: dict[str, float],
    context_sensitivity: dict[str, float] | None,
    trajectory: dict[str, float],
    fixed_points: dict[str, float],
    niche_regimes: dict[str, float] | None,
    pseudotime_structure: dict[str, float],
    split_summary: dict[str, Any],
    biology_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge": edge_label,
        "mode": mode,
        "heldout_metrics": heldout_metrics,
        "calibration": calibration,
        "context_sensitivity": context_sensitivity,
        "trajectory": trajectory,
        "fixed_points": fixed_points,
        "niche_regimes": niche_regimes,
        "pseudotime_structure": pseudotime_structure,
        "split_summary": split_summary,
        "biology_summary": biology_summary,
    }
