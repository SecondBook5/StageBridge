"""Interpretable lesion-level progression-risk scoring for label repair."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config value from OmegaConf or dict payloads.

    Args:
        cfg: Config tree.
        dotted: Dotted key path.
        default: Fallback when the key is missing.
    """
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, dotted)
        return default if value is None else value
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _normalize_series(values: pd.Series) -> pd.Series:
    """Map numeric values into `[0, 1]` with stable handling of flat inputs.

    Args:
        values: Numeric series.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    scaled = (numeric - low) / max(high - low, 1e-8)
    return scaled.fillna(0.0).clip(0.0, 1.0)


def _series_or_default(frame: pd.DataFrame, column: str, default: Any = 0.0) -> pd.Series:
    """Return a column or a constant-length default series.

    Args:
        frame: Source table.
        column: Requested column.
        default: Scalar fallback value.
    """
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * frame.shape[0], index=frame.index)


def build_risk_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive normalized scoring features from merged lesion evidence tables.

    Args:
        frame: Merged lesion evidence table.
    """
    features = frame.copy()
    mutation_columns = [
        column
        for column in [
            "kras_mut",
            "egfr_mut",
            "tp53_mut",
            "stk11_mut",
            "keap1_mut",
            "smad4_mut",
            "braf_mut",
        ]
        if column in features.columns
    ]
    features["driver_burden"] = (
        features[mutation_columns].fillna(0.0).sum(axis=1) if mutation_columns else 0.0
    )
    features["tmb_norm"] = _normalize_series(_series_or_default(features, "tmb", 0.0))
    features["driver_burden_norm"] = _normalize_series(features["driver_burden"])
    features["cna_burden_norm"] = _normalize_series(
        _series_or_default(features, "cna_burden", 0.0)
    )
    features["clone_sharing_norm"] = _normalize_series(
        _series_or_default(features, "shared_cluster_count_with_later_lesions", 0.0)
    )
    features["descendant_sharing_norm"] = _normalize_series(
        _series_or_default(features, "descendant_sharing_score", 0.0)
    )
    features["pathology_risk_norm"] = _normalize_series(
        _series_or_default(features, "pathology_risk_score", 0.0)
    )
    features["later_stage_support"] = (
        pd.to_numeric(_series_or_default(features, "has_later_stage", 0.0), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    features["curated_positive_support"] = (
        _series_or_default(features, "original_label_source", "")
        .astype(str)
        .str.startswith("peng_")
        & pd.to_numeric(_series_or_default(features, "original_label", 0.0), errors="coerce")
        .fillna(0.0)
        .eq(1.0)
    ).astype(float)
    features["curated_negative_support"] = (
        _series_or_default(features, "original_label_source", "")
        .astype(str)
        .str.startswith("peng_")
        & pd.to_numeric(_series_or_default(features, "original_label", 0.0), errors="coerce")
        .fillna(0.0)
        .eq(0.0)
    ).astype(float)
    features["heuristic_label_support"] = (
        _series_or_default(features, "original_label_source", "")
        .astype(str)
        .eq("heuristic_edge_expansion")
        .astype(float)
    )
    features["non_proxy_evidence_count"] = (
        features[
            [
                "cna_burden_norm",
                "clone_sharing_norm",
                "descendant_sharing_norm",
                "pathology_risk_norm",
            ]
        ]
        .gt(0.0)
        .sum(axis=1)
        .astype(float)
    )
    return features


def score_lesions(
    frame: pd.DataFrame, cfg: DictConfig | dict[str, Any]
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute interpretable progression-risk scores and contribution terms.

    Args:
        frame: Merged lesion evidence table.
        cfg: Active config tree.
    """
    features = build_risk_feature_table(frame)
    weights = _cfg_select(cfg, "labels.scoring.weights", {}) or {}
    defaults = {
        "curated_positive": 0.30,
        "curated_negative": -0.45,
        "later_stage_presence": 0.15,
        "tmb": 0.10,
        "driver_burden": 0.10,
        "cna_burden": 0.10,
        "clone_sharing": 0.12,
        "descendant_sharing": 0.08,
        "pathology_risk": 0.05,
        "heuristic_label": 0.05,
    }
    resolved = {key: float(weights.get(key, value)) for key, value in defaults.items()}
    contribution_frame = pd.DataFrame(
        {
            "curated_positive": features["curated_positive_support"]
            * resolved["curated_positive"],
            "curated_negative": features["curated_negative_support"]
            * resolved["curated_negative"],
            "later_stage_presence": features["later_stage_support"]
            * resolved["later_stage_presence"],
            "tmb": features["tmb_norm"] * resolved["tmb"],
            "driver_burden": features["driver_burden_norm"] * resolved["driver_burden"],
            "cna_burden": features["cna_burden_norm"] * resolved["cna_burden"],
            "clone_sharing": features["clone_sharing_norm"] * resolved["clone_sharing"],
            "descendant_sharing": features["descendant_sharing_norm"]
            * resolved["descendant_sharing"],
            "pathology_risk": features["pathology_risk_norm"] * resolved["pathology_risk"],
            "heuristic_label": features["heuristic_label_support"] * resolved["heuristic_label"],
        },
        index=frame.index,
    )
    linear = 0.5 + contribution_frame.sum(axis=1)
    score = (1.0 / (1.0 + np.exp(-4.0 * (linear - 0.5)))).clip(0.0, 1.0)
    return score.astype(float), contribution_frame


def summarize_contributions(contributions: pd.Series, *, positive: bool) -> str:
    """Return the top evidence reasons from one contribution vector.

    Args:
        contributions: Named contribution vector for one lesion.
        positive: Whether to summarize positive or negative evidence.
    """
    if positive:
        selected = contributions[contributions > 0].sort_values(ascending=False)
    else:
        selected = contributions[contributions < 0].sort_values()
    if selected.empty:
        return ""
    return "; ".join(f"{name}={value:.3f}" for name, value in selected.head(3).items())
