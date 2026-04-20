"""Spatial backend comparison and selection.

Compares deconvolution backends (Tangram, DestVI, TACCO, Cell2location)
and selects the canonical backend based on comprehensive metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS_CONFIG = {
    # Higher is better
    "types_per_spot_mean": {"weight": 0.30, "higher_better": True},
    "effective_coverage": {"weight": 0.25, "higher_better": True},
    "global_type_coverage": {"weight": 0.20, "higher_better": True},
    "mean_entropy": {"weight": 0.15, "higher_better": True},
    # Lower is better
    "gini_coefficient_mean": {"weight": 0.10, "higher_better": False},
}

FALLBACK_METRICS = {
    "coverage": {"weight": 0.4, "higher_better": True},
    "mean_entropy": {"weight": 0.3, "higher_better": True},
    "sparsity": {"weight": 0.3, "higher_better": False},
}


def compute_composite_score(
    metrics: dict[str, float],
    config: dict[str, dict] | None = None,
) -> float:
    """Compute weighted composite score from metrics.

    Args:
        metrics: Dictionary of metric name -> value
        config: Metrics configuration (uses METRICS_CONFIG by default)

    Returns:
        Composite score in [0, 1]
    """
    if config is None:
        config = METRICS_CONFIG

    available = [m for m in config if m in metrics]

    if len(available) < 2:
        config = FALLBACK_METRICS
        available = [m for m in config if m in metrics]

    if not available:
        return 0.0

    score = 0.0
    total_weight = 0.0

    for metric_name in available:
        cfg = config[metric_name]
        val = metrics.get(metric_name)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        if metric_name == "types_per_spot_mean":
            norm_val = min(val / 15.0, 1.0)
        else:
            norm_val = val

        if not cfg["higher_better"]:
            norm_val = 1.0 - norm_val

        score += norm_val * cfg["weight"]
        total_weight += cfg["weight"]

    return score / max(total_weight, 1e-6)


def load_backend_metrics(
    spatial_dir: Path,
    label_sources: list[str],
    backends: list[str],
) -> dict[str, dict[str, dict]]:
    """Load metrics for all backend/label-source combinations.

    Args:
        spatial_dir: Root spatial benchmark directory
        label_sources: List of label sources (e.g., ['hlca', 'luca'])
        backends: List of backends (e.g., ['tangram', 'destvi', 'tacco'])

    Returns:
        Nested dict: results[label_source][backend] = metrics
    """
    results = {}

    for label_source in label_sources:
        results[label_source] = {}
        for backend in backends:
            metrics_path = spatial_dir / label_source / backend / "upstream_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    results[label_source][backend] = json.load(f)

    return results


def compare_backends(
    results: dict[str, dict[str, dict]],
    force_backend: str | None = None,
) -> dict[str, Any]:
    """Compare all backends and select canonical.

    Args:
        results: Nested dict from load_backend_metrics
        force_backend: Force selection of specific backend

    Returns:
        Dictionary with comparison results and canonical selection
    """
    scores = {}
    for label_source, backend_results in results.items():
        for backend, metrics in backend_results.items():
            key = (label_source, backend)
            scores[key] = compute_composite_score(metrics)

    if not scores:
        return {
            "error": "No backend metrics available",
            "canonical": None,
        }

    if force_backend:
        forced_scores = {k: v for k, v in scores.items() if k[1] == force_backend}
        if forced_scores:
            best_key = max(forced_scores.keys(), key=lambda x: forced_scores[x])
        else:
            best_key = max(scores.keys(), key=lambda x: scores[x])
    else:
        best_key = max(scores.keys(), key=lambda x: scores[x])

    canonical_label_source, canonical_backend = best_key

    label_sources = list(results.keys())
    backends = list(set(be for lr in results.values() for be in lr.keys()))

    hlca_scores = [scores[(ls, be)] for ls, be in scores if ls == "hlca"]
    luca_scores = [scores[(ls, be)] for ls, be in scores if ls == "luca"]

    return {
        "label_sources": label_sources,
        "backends": backends,
        "composite_scores": {f"{ls}/{be}": float(s) for (ls, be), s in scores.items()},
        "canonical": {
            "label_source": canonical_label_source,
            "backend": canonical_backend,
            "score": float(scores[best_key]),
            "forced": bool(force_backend),
        },
        "label_ablation": {
            "hlca_mean_score": float(np.mean(hlca_scores)) if hlca_scores else 0.0,
            "luca_mean_score": float(np.mean(luca_scores)) if luca_scores else 0.0,
            "better_label_source": "hlca" if np.mean(hlca_scores or [0]) >= np.mean(luca_scores or [0]) else "luca",
        },
        "metrics_config": {
            k: {"weight": v["weight"], "higher_better": v["higher_better"]}
            for k, v in METRICS_CONFIG.items()
        },
    }


def select_canonical_backend(
    spatial_dir: Path,
    label_sources: list[str] | None = None,
    backends: list[str] | None = None,
    force_backend: str | None = None,
) -> dict[str, Any]:
    """Load metrics and select canonical backend.

    Convenience function combining load and compare.

    Args:
        spatial_dir: Root spatial benchmark directory
        label_sources: List of label sources (default: ['hlca', 'luca'])
        backends: List of backends (default: ['tangram', 'destvi', 'tacco', 'cell2location'])
        force_backend: Force selection of specific backend

    Returns:
        Dictionary with canonical selection
    """
    if label_sources is None:
        label_sources = ["hlca", "luca"]
    if backends is None:
        backends = ["tangram", "destvi", "tacco", "cell2location"]

    results = load_backend_metrics(spatial_dir, label_sources, backends)
    return compare_backends(results, force_backend=force_backend)
