"""Hybrid provider benchmarking for Tangram, TACCO, and DestVI."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _rank(values: pd.Series, *, ascending: bool) -> pd.Series:
    return values.rank(method="dense", ascending=ascending).astype(float)


def _mode_pair(value: pd.Series) -> tuple[str, float]:
    values = value.dropna().astype(str).tolist()
    if not values:
        return "n/a", float("nan")
    counts = Counter(values)
    winner, count = counts.most_common(1)[0]
    return winner, float(count / max(len(values), 1))


def _provider_agreement_summary(agreement_table: pd.DataFrame) -> pd.DataFrame:
    if agreement_table.empty:
        return pd.DataFrame(columns=["method", "winner_agreement_mean", "cosine_similarity_mean"])
    rows: list[dict[str, Any]] = []
    for row in agreement_table.itertuples(index=False):
        left = {
            "method": str(row.left_method),
            "winner_agreement": float(getattr(row, "winner_agreement", np.nan)),
            "cosine_similarity": float(getattr(row, "mean_cosine_similarity", np.nan)),
        }
        right = {
            "method": str(row.right_method),
            "winner_agreement": float(getattr(row, "winner_agreement", np.nan)),
            "cosine_similarity": float(getattr(row, "mean_cosine_similarity", np.nan)),
        }
        rows.extend([left, right])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("method", as_index=False)
        .agg(
            winner_agreement_mean=("winner_agreement", "mean"),
            cosine_similarity_mean=("cosine_similarity", "mean"),
        )
    )


def summarize_provider_benchmark(
    *,
    provider_metric_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
    agreement_table: pd.DataFrame,
    reference_gate: Mapping[str, Any] | None = None,
    decisive_margin: float = 0.25,
) -> dict[str, Any]:
    """Aggregate provider QC, downstream performance, and biological stability.

    Lower hybrid scores are better. Selection can still be marked ``inconclusive``
    if the leading provider does not separate cleanly from the runner-up.
    """
    methods = sorted(
        {
            *provider_metric_table.get("method", pd.Series(dtype=object)).astype(str).tolist(),
            *downstream_table.get("method", pd.Series(dtype=object)).astype(str).tolist(),
        }
    )
    if not methods:
        return {
            "selected_provider": None,
            "selection_status": "inconclusive",
            "selection_reason": "No provider results were available for benchmarking.",
            "recommended_action": "needs_more_data",
            "provider_scores": [],
        }

    qc = provider_metric_table.copy()
    if qc.empty:
        qc = pd.DataFrame({"method": methods})
    qc["method"] = qc["method"].astype(str)
    qc["complete_flag"] = qc.get("status", pd.Series(["failed"] * qc.shape[0])).eq("complete").astype(float)
    qc["row_sum_deviation"] = (qc.get("mean_row_sum", pd.Series([np.nan] * qc.shape[0])).astype(float) - 1.0).abs()
    qc_agg = (
        qc.groupby("method", as_index=False)
        .agg(
            mean_row_sum=("mean_row_sum", "mean"),
            rows_close_to_one_frac=("rows_close_to_one_frac", "mean"),
            mean_max_assignment=("mean_max_assignment", "mean"),
            mean_normalized_entropy=("mean_normalized_entropy", "mean"),
            complete_fraction=("complete_flag", "mean"),
            row_sum_deviation=("row_sum_deviation", "mean"),
        )
    )

    qc_agg["mapping_rank"] = (
        _rank(qc_agg["row_sum_deviation"], ascending=True)
        + _rank(qc_agg["mean_max_assignment"], ascending=False)
        + _rank(qc_agg["mean_normalized_entropy"], ascending=True)
        + _rank(qc_agg["complete_fraction"], ascending=False)
    ) / 4.0

    perf = downstream_table.copy()
    if perf.empty:
        perf = pd.DataFrame({"method": methods})
    perf["method"] = perf["method"].astype(str)
    perf_agg = (
        perf.groupby("method", as_index=False)
        .agg(
            sinkhorn_mean=("sinkhorn", "mean"),
            sinkhorn_std=("sinkhorn", "std"),
            calibration_mean=("calibration_error", "mean"),
            calibration_std=("calibration_error", "std"),
        )
    )
    perf_agg["performance_rank"] = (
        _rank(perf_agg["sinkhorn_mean"], ascending=True)
        + _rank(perf_agg["calibration_mean"], ascending=True)
    ) / 2.0

    stability_rows: list[dict[str, Any]] = []
    if not perf.empty and {"dominant_increase_group", "dominant_decrease_group", "edge", "mode"}.issubset(perf.columns):
        perf = perf.copy()
        perf["biology_pair"] = (
            perf["dominant_increase_group"].fillna("n/a").astype(str)
            + " | "
            + perf["dominant_decrease_group"].fillna("n/a").astype(str)
        )
        combo_rows: list[dict[str, Any]] = []
        for (method, edge, mode), group in perf.groupby(["method", "edge", "mode"]):
            pair, frac = _mode_pair(group["biology_pair"])
            combo_rows.append(
                {
                    "method": str(method),
                    "edge": str(edge),
                    "mode": str(mode),
                    "majority_pair": pair,
                    "majority_fraction": frac,
                }
            )
        combo = pd.DataFrame(combo_rows)
        if not combo.empty:
            for method, group in combo.groupby("method"):
                majority_frac = float(group["majority_fraction"].mean())
                edge_distinct = []
                for mode, mode_group in group.groupby("mode"):
                    by_edge = mode_group.set_index("edge")["majority_pair"].to_dict()
                    if len(by_edge) >= 2:
                        edge_distinct.append(float(len(set(by_edge.values())) > 1))
                stability_rows.append(
                    {
                        "method": str(method),
                        "dominant_pair_consistency": majority_frac,
                        "edge_interpretation_distinctiveness": float(np.mean(edge_distinct)) if edge_distinct else float("nan"),
                    }
                )
    stability = pd.DataFrame(stability_rows)
    if stability.empty:
        stability = pd.DataFrame({"method": methods})
    agreement_summary = _provider_agreement_summary(agreement_table)
    stability = stability.merge(agreement_summary, on="method", how="outer")
    stability["stability_rank"] = (
        _rank(stability["dominant_pair_consistency"].fillna(0.0), ascending=False)
        + _rank(stability["winner_agreement_mean"].fillna(0.0), ascending=False)
        + _rank(stability["edge_interpretation_distinctiveness"].fillna(0.0), ascending=False)
    ) / 3.0

    provider_scores = (
        pd.DataFrame({"method": methods})
        .merge(qc_agg, on="method", how="left")
        .merge(perf_agg, on="method", how="left")
        .merge(stability, on="method", how="left")
    )
    provider_scores["mapping_rank"] = provider_scores["mapping_rank"].fillna(float(len(methods)))
    provider_scores["performance_rank"] = provider_scores["performance_rank"].fillna(float(len(methods)))
    provider_scores["stability_rank"] = provider_scores["stability_rank"].fillna(float(len(methods)))
    provider_scores["hybrid_rank_score"] = (
        0.25 * provider_scores["mapping_rank"]
        + 0.50 * provider_scores["performance_rank"]
        + 0.25 * provider_scores["stability_rank"]
    )
    provider_scores = provider_scores.sort_values(
        ["hybrid_rank_score", "performance_rank", "mapping_rank"],
        kind="mergesort",
    ).reset_index(drop=True)

    top_method = None if provider_scores.empty else str(provider_scores.iloc[0]["method"])
    second_score = float(provider_scores.iloc[1]["hybrid_rank_score"]) if provider_scores.shape[0] > 1 else float("inf")
    top_score = float(provider_scores.iloc[0]["hybrid_rank_score"]) if provider_scores.shape[0] > 0 else float("inf")
    winner_margin = float(second_score - top_score) if np.isfinite(second_score) and np.isfinite(top_score) else float("inf")

    selection_status = "pass"
    selection_reason = f"{top_method} achieved the best weighted provider rank."
    recommended_action = "keep"
    selected_provider = top_method

    gate_status = str((reference_gate or {}).get("status", "pass"))
    if gate_status == "fail":
        selection_status = "inconclusive"
        selection_reason = "HLCA alignment gate failed, so provider selection cannot be trusted as a default."
        recommended_action = "needs_more_data"
    elif provider_scores.shape[0] < 2:
        selection_status = "inconclusive"
        selection_reason = "Only one provider completed credibly, so selection remains inconclusive."
        recommended_action = "needs_more_data"
    elif winner_margin < decisive_margin:
        selection_status = "inconclusive"
        selection_reason = (
            f"{top_method} ranked first, but the margin to the runner-up ({winner_margin:.3f}) "
            "is too small to call the provider winner decisive."
        )
        recommended_action = "keep_as_optional"
    elif provider_scores.iloc[0]["performance_rank"] > provider_scores.iloc[0]["mapping_rank"] + 1.0:
        selection_status = "weak_pass"
        selection_reason = (
            f"{top_method} ranked first overall, but the QC/performance split is uneven. "
            "Treat the provider as a provisional downstream default."
        )
        recommended_action = "keep_as_optional"

    return {
        "selected_provider": selected_provider,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "recommended_action": recommended_action,
        "winner_margin": winner_margin,
        "reference_gate_status": gate_status,
        "provider_scores": provider_scores.to_dict(orient="records"),
    }


def render_provider_benchmark_md(benchmark_payload: Mapping[str, Any]) -> str:
    lines = [
        "# Spatial Provider Benchmark",
        "",
        f"- Selected provider: {benchmark_payload.get('selected_provider', 'n/a')}",
        f"- Selection status: {benchmark_payload.get('selection_status', 'n/a')}",
        f"- Recommended action: {benchmark_payload.get('recommended_action', 'n/a')}",
        f"- Winner margin: {benchmark_payload.get('winner_margin', 'n/a')}",
        f"- Interpretation: {benchmark_payload.get('selection_reason', 'n/a')}",
        "",
    ]
    for row in benchmark_payload.get("provider_scores", []):
        lines.extend(
            [
                f"## {row.get('method', 'n/a')}",
                f"- Hybrid score: {row.get('hybrid_rank_score', 'n/a')}",
                f"- Mapping rank: {row.get('mapping_rank', 'n/a')}",
                f"- Performance rank: {row.get('performance_rank', 'n/a')}",
                f"- Stability rank: {row.get('stability_rank', 'n/a')}",
                f"- Sinkhorn mean: {row.get('sinkhorn_mean', 'n/a')}",
                f"- Calibration mean: {row.get('calibration_mean', 'n/a')}",
                f"- Mean max assignment: {row.get('mean_max_assignment', 'n/a')}",
                f"- Mean normalized entropy: {row.get('mean_normalized_entropy', 'n/a')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"

