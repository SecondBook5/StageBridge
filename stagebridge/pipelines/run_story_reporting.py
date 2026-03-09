"""Generate benchmark summaries and poster-ready figures for the active StageBridge story."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from omegaconf import DictConfig

from stagebridge.viz.story_figures import (
    plot_communication_metric_panels,
    plot_context_shuffle_deltas,
    plot_label_balance,
    plot_transition_vs_communication,
)

MODEL_DISPLAY = {
    "focal_only": "focal_only",
    "pooled": "pooled",
    "deep_sets": "deep_sets",
    "graphsage": "graphsage",
    "graph_transformer": "graph_transformer",
    "transformer_no_priors": "transformer_no_priors",
    "transformer_no_relay": "transformer_no_relay",
    "stagebridge": "stagebridge",
    "rna_only": "rna_only",
    "set_only": "set_only",
    "graph_of_sets": "graph_of_sets",
}


def _cfg(section: DictConfig | dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_table(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    _ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    try:
        markdown = df.to_markdown(index=False)
    except Exception:
        markdown = df.to_string(index=False)
    md_path.write_text(markdown + "\n", encoding="utf-8")


def _summarize_transition_core(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    plot_df = df.copy()
    plot_df["mode"] = plot_df["mode"].map(lambda x: MODEL_DISPLAY.get(str(x), str(x)))
    winners = (
        plot_df.sort_values(["edge", "primary_metric"], ascending=[True, True])
        .groupby("edge", as_index=False)
        .first()[["edge", "mode", "primary_metric", "course_interpretation"]]
        .rename(columns={"mode": "winning_mode", "primary_metric": "winning_metric"})
    )
    return plot_df, winners


def _summarize_communication(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = (
        df.groupby("model_name", as_index=False)
        .agg(
            n_runs=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            context_shuffle_auroc_delta_mean=("context_shuffle_auroc_delta", "mean"),
            context_shuffle_auroc_delta_std=("context_shuffle_auroc_delta", "std"),
            context_shuffle_auprc_delta_mean=("context_shuffle_auprc_delta", "mean"),
            context_shuffle_auprc_delta_std=("context_shuffle_auprc_delta", "std"),
        )
        .sort_values(["auroc_mean", "auprc_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary["model_name"] = summary["model_name"].map(lambda x: MODEL_DISPLAY.get(str(x), str(x)))
    shuffle = summary[
        [
            "model_name",
            "context_shuffle_auroc_delta_mean",
            "context_shuffle_auroc_delta_std",
            "context_shuffle_auprc_delta_mean",
            "context_shuffle_auprc_delta_std",
        ]
    ].copy()
    return summary, shuffle


def _label_balance(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        return pd.DataFrame(columns=["edge_label", "negative_bags", "positive_bags", "total_bags"])
    df = pd.read_csv(manifest_path)
    if df.empty or "progression_competent_label" not in df.columns:
        return pd.DataFrame(columns=["edge_label", "negative_bags", "positive_bags", "total_bags"])
    usable = df.loc[df["progression_competent_label"].isin([0, 1])].copy()
    if usable.empty:
        return pd.DataFrame(columns=["edge_label", "negative_bags", "positive_bags", "total_bags"])
    summary = (
        usable.groupby(["edge_label", "progression_competent_label"], as_index=False)
        .size()
        .pivot(index="edge_label", columns="progression_competent_label", values="size")
        .fillna(0.0)
        .rename(columns={0: "negative_bags", 1: "positive_bags"})
        .reset_index()
    )
    for column in ["negative_bags", "positive_bags"]:
        if column not in summary.columns:
            summary[column] = 0
    summary["total_bags"] = summary["negative_bags"] + summary["positive_bags"]
    return summary[["edge_label", "negative_bags", "positive_bags", "total_bags"]]


def _story_table(transition_df: pd.DataFrame, communication_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not transition_df.empty:
        ais_transition = transition_df.loc[transition_df["edge"] == "AIS->MIA"].copy()
        for _, row in ais_transition.iterrows():
            rows.append(
                {
                    "benchmark_family": "transition_model",
                    "target": "AIS->MIA",
                    "model_name": row["mode"],
                    "metric_name": "sinkhorn_distance",
                    "metric_value": float(row["primary_metric"]),
                    "direction": "lower_better",
                }
            )
    if not communication_df.empty:
        for _, row in communication_df.iterrows():
            rows.append(
                {
                    "benchmark_family": "communication_relay",
                    "target": "AIS proxy",
                    "model_name": row["model_name"],
                    "metric_name": "auroc_mean",
                    "metric_value": float(row["auroc_mean"]),
                    "direction": "higher_better",
                }
            )
    return pd.DataFrame(rows)


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_story_reporting(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    report_cfg = cfg.get("story_report", {}) if isinstance(cfg, DictConfig) else cfg.get("story_report", {})
    reports_root = Path(_cfg(report_cfg, "reports_root", "reports"))
    transition_source = Path(
        _cfg(
            report_cfg,
            "transition_source",
            "reports/archive/course_project/tables/core_mode_comparison.csv",
        )
    )
    if not transition_source.exists():
        transition_source = Path("reports/course_project/tables/core_mode_comparison.csv")
    communication_ais_sources = list(
        _cfg(
            report_cfg,
            "communication_ais_sources",
            [
                "outputs/scratch/communication_relay_ais_progression_benchmark/communication_relay/benchmark_summary.csv",
                "outputs/scratch/communication_relay_ais_graph_transformer_benchmark/communication_relay/benchmark_summary.csv",
            ],
        )
    )
    communication_combined_sources = list(
        _cfg(
            report_cfg,
            "communication_combined_sources",
            [
                "outputs/scratch/communication_relay_combined_progression_benchmark/communication_relay/benchmark_summary.csv",
            ],
        )
    )
    manifest_path = Path(
        _cfg(
            report_cfg,
            "curated_manifest_path",
            "stagebridge/data/luad_evo/curated_progression_labels.csv",
        )
    )

    transition_raw = pd.read_csv(transition_source) if transition_source.exists() else pd.DataFrame()
    communication_ais_raw = _read_many(communication_ais_sources)
    communication_combined_raw = _read_many(communication_combined_sources)

    transition_plot, transition_winners = _summarize_transition_core(transition_raw)
    communication_ais_summary, communication_ais_shuffle = _summarize_communication(communication_ais_raw)
    communication_combined_summary, _ = _summarize_communication(communication_combined_raw)
    label_balance = _label_balance(manifest_path)
    story_df = _story_table(transition_plot, communication_ais_summary)

    benchmarks_root = _ensure_dir(reports_root / "benchmarks")
    transition_root = _ensure_dir(benchmarks_root / "transition_model")
    communication_root = _ensure_dir(benchmarks_root / "communication_relay")
    story_root = _ensure_dir(benchmarks_root / "story")
    poster_root = _ensure_dir(reports_root / "poster" / "hca_general_meeting")
    poster_fig_root = _ensure_dir(poster_root / "figures")

    if not transition_plot.empty:
        _write_table(transition_plot, transition_root / "core_mode_comparison.csv", transition_root / "core_mode_comparison.md")
        _write_table(transition_winners, transition_root / "winning_modes_by_edge.csv", transition_root / "winning_modes_by_edge.md")
    if not communication_ais_summary.empty:
        _write_table(communication_ais_summary, communication_root / "ais_model_family_summary.csv", communication_root / "ais_model_family_summary.md")
        _write_table(communication_ais_shuffle, communication_root / "ais_context_shuffle_summary.csv", communication_root / "ais_context_shuffle_summary.md")
    if not communication_combined_summary.empty:
        _write_table(
            communication_combined_summary,
            communication_root / "combined_model_family_summary.csv",
            communication_root / "combined_model_family_summary.md",
        )
    if not label_balance.empty:
        _write_table(label_balance, communication_root / "label_balance_summary.csv", communication_root / "label_balance_summary.md")
    if not story_df.empty:
        _write_table(story_df, story_root / "transition_vs_communication_story.csv", story_root / "transition_vs_communication_story.md")

    if not transition_plot.empty and not communication_ais_summary.empty:
        transition_ais = transition_plot.loc[transition_plot["edge"] == "AIS->MIA", ["mode", "primary_metric"]].copy()
        plot_transition_vs_communication(transition_ais, communication_ais_summary, poster_fig_root / "figure_transition_vs_communication_story.png")
    if not communication_ais_summary.empty:
        plot_communication_metric_panels(communication_ais_summary, poster_fig_root / "figure_communication_benchmark_metrics.png")
        plot_context_shuffle_deltas(communication_ais_shuffle, poster_fig_root / "figure_context_shuffle_delta.png")
    if not label_balance.empty:
        plot_label_balance(label_balance, poster_fig_root / "figure_label_balance.png")

    if not communication_ais_summary.empty:
        top_row = communication_ais_summary.iloc[0]
        stagebridge_row = communication_ais_summary.loc[communication_ais_summary["model_name"] == "stagebridge"]
        stagebridge_row = stagebridge_row.iloc[0] if not stagebridge_row.empty else None
        abstract_text = f"""# HCA General Meeting Poster Abstract Draft

## Title
Task-dependent transformer benefit in early LUAD: compact niche attention helps transition modeling while richer communication-relay attention does not yet beat pooled summaries

## Abstract
We studied early lung adenocarcinoma progression as a donor-held-out, niche-conditioned learning problem on matched snRNA-seq, Visium spatial transcriptomics, and WES from the precursor ladder. In the original StageBridge transition benchmark, compact Set Transformer context gave the best active transformer result on the clinically important AIS->MIA edge, improving Sinkhorn transition fidelity over pooled and graph-augmented context encoders. We then extended StageBridge into a focal-receiver communication-relay transformer that reasons over sender cells, ligand-receptor proposals, receiver-response programs, and relay-memory tokens to predict progression-competent precursor niches. Under paper-derived clonal-proxy supervision for the AIS proxy task, however, pooled communication summaries were the strongest model family (mean AUROC {top_row['auroc_mean']:.3f}, mean AUPRC {top_row['auprc_mean']:.3f}), while the full communication-relay transformer underperformed (mean AUROC {float(stagebridge_row['auroc_mean']) if stagebridge_row is not None else float('nan'):.3f}). These results show that transformer benefit in early LUAD is task-dependent: attention helps when the target is edge-specific transition transport with compact typed context, but richer relation-heavy communication transformers likely require denser supervision or larger cohorts. The benchmark contributes a practical boundary for transformer use in spatially conditioned cancer progression modeling.
"""
        _write_text(poster_root / "ABSTRACT.md", abstract_text)

        narrative_text = """# Poster Narrative

## Message
The repo supports a stronger story than “the biggest transformer won.” The benchmark now shows a boundary condition:

- compact set attention helps the AIS->MIA transition benchmark
- richer communication-relay attention does not beat pooled CCC summaries on sparse clonal-proxy supervision
- graph-style complexity also does not rescue the CCC benchmark if local supervision is thin

## Poster flow
1. Introduce StageBridge as a transformer-centered framework for edge-specific LUAD progression.
2. Show the positive transition benchmark on AIS->MIA.
3. Show the communication-relay extension and its richer tokenization.
4. Show the donor-held-out communication benchmark where pooled wins.
5. Conclude with the task-dependent transformer lesson and the next biological steps.
"""
        _write_text(poster_root / "NARRATIVE.md", narrative_text)

        figure_plan_text = """# Figure Plan

## Required figures
- `figures/figure_transition_vs_communication_story.png`
  Side-by-side comparison of the positive transition benchmark and the communication benchmark.
- `figures/figure_communication_benchmark_metrics.png`
  AUROC and AUPRC comparison across communication model families.
- `figures/figure_context_shuffle_delta.png`
  Context-shuffle degradation across communication models.
- `figures/figure_label_balance.png`
  Positive/negative bag counts by curated communication target.

## Recommended legacy figures
- `reports/archive/course_project/figures/figure_1_stagebridge_block_diagram.png`
- `reports/archive/course_project/figures/figure_2_mode_comparison_by_edge.png`
- `reports/archive/course_project/figures/figure_14_spatial_niche_map.png`
"""
        _write_text(poster_root / "FIGURE_PLAN.md", figure_plan_text)

    payload = {
        "ok": True,
        "transition_source": str(transition_source),
        "communication_ais_sources": [str(Path(path)) for path in communication_ais_sources],
        "communication_combined_sources": [str(Path(path)) for path in communication_combined_sources],
        "reports_root": str(reports_root),
        "poster_root": str(poster_root),
    }
    _write_text(story_root / "story_report_manifest.json", json.dumps(payload, indent=2))
    return payload


__all__ = ["run_story_reporting"]
