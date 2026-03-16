"""Typed niche and edge-level biological insight summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER, normalize_stage_label


def _ordered_stages(values: list[str]) -> list[str]:
    present = [str(value) for value in values]
    return [stage for stage in CANONICAL_STAGE_ORDER if stage in present] + sorted(
        {stage for stage in present if stage not in CANONICAL_STAGE_ORDER}
    )


def _safe_dict(series: pd.Series) -> dict[str, float]:
    return {str(idx): float(val) for idx, val in series.items()}


def summarize_edge_biology(
    typed_tokens: Any,
    *,
    edge_label: str,
    context_sensitivity: dict[str, Any] | None = None,
    split_summary: dict[str, Any] | None = None,
    wes_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize typed niche biology for one transition edge."""
    groups = list(typed_tokens.schema.typed_feature_names)
    table = pd.DataFrame(np.asarray(typed_tokens.tokens, dtype=np.float32), columns=groups)
    table["stage"] = typed_tokens.obs["stage"].astype(str).map(normalize_stage_label)
    table["donor_id"] = typed_tokens.obs["donor_id"].astype(str)
    ordered = _ordered_stages(table["stage"].tolist())

    stage_means = table.groupby("stage")[groups].mean().reindex(ordered).fillna(0.0)
    donor_stage_means = table.groupby(["stage", "donor_id"])[groups].mean()
    donor_stage_std = donor_stage_means.groupby("stage")[groups].std().reindex(ordered).fillna(0.0)

    stage_src, stage_tgt = [
        normalize_stage_label(part.strip()) for part in str(edge_label).split("->", 1)
    ]
    src_profile = (
        stage_means.loc[stage_src]
        if stage_src in stage_means.index
        else pd.Series(0.0, index=groups)
    )
    tgt_profile = (
        stage_means.loc[stage_tgt]
        if stage_tgt in stage_means.index
        else pd.Series(0.0, index=groups)
    )
    delta = (tgt_profile - src_profile).sort_values(ascending=False)

    dominant_increase = str(delta.index[0]) if not delta.empty else "n/a"
    dominant_decrease = str(delta.index[-1]) if not delta.empty else "n/a"
    context_delta = (
        None
        if context_sensitivity is None
        else context_sensitivity.get("context_sensitivity_delta")
    )
    interpretation = [
        f"Across mapped typed niches, {dominant_increase} increases most from {stage_src} to {stage_tgt}."
    ]
    if dominant_decrease != dominant_increase:
        interpretation.append(f"{dominant_decrease} decreases most across the same edge.")
    if context_delta is not None:
        interpretation.append(
            "Context shuffling "
            + ("worsens" if float(context_delta) > 0 else "does not improve")
            + f" this edge by {float(context_delta):.3f} Sinkhorn units."
        )
    if wes_diagnostics and wes_diagnostics.get("enabled"):
        interpretation.append(
            f"WES regularization mean penalty for the training pairs is {float(wes_diagnostics.get('regularizer_mean_penalty', 0.0)):.3f}."
        )

    return {
        "edge": edge_label,
        "stage_order": ordered,
        "typed_groups": groups,
        "stage_mean_profiles": {
            stage: _safe_dict(stage_means.loc[stage]) for stage in stage_means.index
        },
        "stage_donor_std": {
            stage: _safe_dict(donor_stage_std.loc[stage]) for stage in donor_stage_std.index
        },
        "edge_delta_by_group": _safe_dict(delta),
        "dominant_increase_group": dominant_increase,
        "dominant_decrease_group": dominant_decrease,
        "context_sensitivity_delta": None if context_delta is None else float(context_delta),
        "split_strategy": None if split_summary is None else split_summary.get("split_strategy"),
        "overlap_donors": []
        if split_summary is None
        else list(split_summary.get("overlap_donors", [])),
        "interpretation": interpretation,
    }
