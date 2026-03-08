"""Result-card rendering helpers for scratch and milestone runs."""
from __future__ import annotations

from typing import Sequence

from stagebridge.results.manifest import RunMetadata, RunMetrics


def build_result_card(
    metadata: RunMetadata,
    metrics: RunMetrics,
    *,
    worked: Sequence[str] | None = None,
    failed: Sequence[str] | None = None,
    milestone_candidate: bool = False,
    next_recommended_step: str = "Review the scratch outputs before promotion.",
) -> str:
    """Render a compact markdown result card."""
    worked = list(worked or [])
    failed = list(failed or [])
    stage_edges = ", ".join(metadata.stage_edges) if metadata.stage_edges else "none specified"
    primary_metric = metrics.primary_metric if metrics.primary_metric is not None else "n/a"
    secondary_metrics = metrics.secondary_metrics or {}
    calibration = metrics.calibration or {}
    notes = metrics.notes or "None."
    worked_text = "; ".join(worked) if worked else "Nothing beyond infrastructure setup."
    failed_text = "; ".join(failed) if failed else "No explicit failures recorded."
    candidate_text = "yes" if milestone_candidate else "no"

    lines = [
        "# Result Card",
        "",
        "## Run Attempt",
        f"- Experiment: {metadata.experiment_name}",
        f"- Mode: {metadata.mode}",
        f"- Stage edge(s): {stage_edges}",
        f"- Split: {metadata.split_name}",
        f"- Status: {metadata.status}",
        "",
        "## Outcome",
        f"- What worked: {worked_text}",
        f"- What failed: {failed_text}",
        "",
        "## Metrics",
        f"- Primary metric: {primary_metric}",
        f"- Secondary metrics: {secondary_metrics}",
        f"- Calibration: {calibration}",
        f"- Notes: {notes}",
        "",
        "## Promotion",
        f"- Milestone candidate: {candidate_text}",
        f"- Next recommended step: {next_recommended_step}",
        "",
    ]
    return "\n".join(lines)
