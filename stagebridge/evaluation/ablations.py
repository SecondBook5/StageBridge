"""Ablation summaries for context and regularization comparisons."""
from __future__ import annotations


def summarize_ablation(mode: str, metrics: dict[str, float], *, wes_enabled: bool) -> dict[str, object]:
    return {
        "mode": mode,
        "wes_enabled": bool(wes_enabled),
        "primary_metric": metrics.get("sinkhorn"),
        "sinkhorn_delta": metrics.get("sinkhorn_delta"),
        "classifier_auc": metrics.get("classifier_auc"),
    }
