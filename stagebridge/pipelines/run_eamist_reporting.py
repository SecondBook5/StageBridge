"""Reporting pipeline for EA-MIST benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from stagebridge.data.luad_evo.neighborhood_builder import build_lesion_bags_from_config
from stagebridge.viz.eamist_figures import (
    save_ablation_figure,
    save_benchmark_comparison_figure,
    save_embedding_diagnostics_figure,
    save_method_overview_figure,
    save_prototype_interpretation_figure,
)


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    if isinstance(cfg, DictConfig):
        current = cfg
        for part in dotted.split("."):
            current = current.get(part, None)
            if current is None:
                return default
        return current
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _collect_auxiliary_edge_tables(benchmark_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_path in benchmark_root.glob("*/*/fold_*/seed_*/auxiliary_edge_metrics.json"):
        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        artifact_dir = metric_path.parent
        split_summary = (
            json.loads((artifact_dir / "split_summary.json").read_text(encoding="utf-8"))
            if (artifact_dir / "split_summary.json").exists()
            else {}
        )
        for metric_name, metric_value in payload.items():
            rows.append(
                {
                    "artifact_dir": str(artifact_dir),
                    "reference_feature_mode": split_summary.get("reference_feature_mode"),
                    "model_family": split_summary.get("model_family"),
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )
    return pd.DataFrame(rows)


def run_eamist_reporting(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Generate active EA-MIST tables and figures from saved run outputs."""
    reports_root = _ensure_dir(
        Path(str(_cfg_select(cfg, "eamist_report.reports_root", "reports")))
    )
    benchmark_root = Path(
        str(
            _cfg_select(
                cfg,
                "eamist_report.benchmark_root",
                Path(str(_cfg_select(cfg, "output_dir", "outputs/scratch")))
                / str(_cfg_select(cfg, "run_name", "stagebridge_v1"))
                / "eamist_benchmark",
            )
        )
    )
    benchmark_summary_path = benchmark_root / "benchmark_summary.csv"
    model_family_summary_path = benchmark_root / "model_family_summary.csv"
    if not benchmark_summary_path.exists():
        raise FileNotFoundError(f"EA-MIST benchmark summary not found: {benchmark_summary_path}")

    benchmark_summary = pd.read_csv(benchmark_summary_path)
    model_family_summary = (
        pd.read_csv(model_family_summary_path)
        if model_family_summary_path.exists()
        else pd.DataFrame()
    )
    build_result = build_lesion_bags_from_config(cfg)

    tables_root = _ensure_dir(reports_root / "tables" / "eamist")
    figures_root = _ensure_dir(reports_root / "figures" / "eamist")
    benchmarks_root = _ensure_dir(reports_root / "benchmarks")

    dataset_table = build_result.summary.copy()
    stage_support = (
        dataset_table.groupby("stage", as_index=False)
        .agg(
            n_lesions=("lesion_id", "nunique"),
            n_donors=("donor_id", "nunique"),
            mean_niches=("num_neighborhoods", "mean"),
        )
        .sort_values("stage")
        .reset_index(drop=True)
    )
    stage_support["interpretation_note"] = np.where(
        stage_support["stage"].isin(["Normal", "MIA"]), "exploratory_low_support", "core_stage"
    )
    dataset_table.to_csv(tables_root / "table1_dataset_composition.csv", index=False)
    stage_support.to_csv(tables_root / "table1b_stage_support.csv", index=False)
    benchmark_summary.to_csv(tables_root / "table2_benchmark_results.csv", index=False)
    model_family_summary.to_csv(tables_root / "table3_model_family_summary.csv", index=False)

    prototype_frames: list[pd.DataFrame] = []
    for prototype_path in benchmark_root.glob("*/*/fold_*/seed_*/prototype_composition.parquet"):
        frame = pd.read_parquet(prototype_path).copy()
        frame["artifact_dir"] = str(prototype_path.parent)
        prototype_frames.append(frame)
    prototype_table = (
        pd.concat(prototype_frames, ignore_index=True) if prototype_frames else pd.DataFrame()
    )
    if not prototype_table.empty:
        prototype_enrichment = (
            prototype_table.groupby(["stage", "prototype"], as_index=False)["occupancy"]
            .mean()
            .sort_values(["stage", "occupancy"], ascending=[True, False])
        )
        prototype_enrichment.to_csv(tables_root / "table4_prototype_enrichment.csv", index=False)

    auxiliary_edge_table = _collect_auxiliary_edge_tables(benchmark_root)
    if not auxiliary_edge_table.empty:
        auxiliary_edge_table.to_csv(tables_root / "table5_auxiliary_edge_metrics.csv", index=False)

    acceptance_rows: list[dict[str, object]] = []
    if not benchmark_summary.empty:
        grouped = benchmark_summary.groupby(
            ["reference_feature_mode", "model_family"], as_index=False
        ).agg(
            stage_macro_f1_mean=("stage_macro_f1", "mean"),
            displacement_spearman_mean=("displacement_spearman", "mean"),
        )
        for reference_mode in sorted(
            grouped["reference_feature_mode"].dropna().astype(str).unique().tolist()
        ):
            mode_frame = grouped[grouped["reference_feature_mode"] == reference_mode]
            pooled_f1 = (
                float(
                    mode_frame.loc[
                        mode_frame["model_family"] == "pooled", "stage_macro_f1_mean"
                    ].iloc[0]
                )
                if (mode_frame["model_family"] == "pooled").any()
                else float("nan")
            )
            eamist_f1 = (
                float(
                    mode_frame.loc[
                        mode_frame["model_family"] == "eamist", "stage_macro_f1_mean"
                    ].iloc[0]
                )
                if (mode_frame["model_family"] == "eamist").any()
                else float("nan")
            )
            acceptance_rows.append(
                {
                    "reference_feature_mode": reference_mode,
                    "pooled_stage_macro_f1": pooled_f1,
                    "eamist_stage_macro_f1": eamist_f1,
                    "eamist_beats_pooled": bool(
                        np.isfinite(pooled_f1) and np.isfinite(eamist_f1) and eamist_f1 > pooled_f1
                    ),
                }
            )
        acceptance_frame = pd.DataFrame(acceptance_rows)
        acceptance_frame.to_csv(tables_root / "table6_acceptance_checks.csv", index=False)

    save_method_overview_figure(figures_root / "figure1_method_overview.png")

    embedding_candidates = sorted(
        (benchmark_root.parent / "eamist_pretrain").glob("neighborhood_embeddings.parquet")
    )
    if embedding_candidates:
        embeddings = pd.read_parquet(embedding_candidates[0])
        save_embedding_diagnostics_figure(
            embeddings, figures_root / "figure2_embedding_diagnostics.png", color_column="stage"
        )

    save_benchmark_comparison_figure(
        benchmark_summary, figures_root / "figure3_benchmark_comparison.png"
    )

    if not prototype_table.empty:
        save_prototype_interpretation_figure(
            prototype_table, figures_root / "figure4_prototypes_attention.png"
        )

    ablation_rows: list[dict[str, object]] = []
    mode_frame = (
        benchmark_summary.groupby(["reference_feature_mode", "model_family"], as_index=False)[
            "stage_macro_f1"
        ]
        .mean()
        .sort_values(["reference_feature_mode", "stage_macro_f1"], ascending=[True, False])
    )
    for reference_mode in sorted(
        mode_frame["reference_feature_mode"].dropna().astype(str).unique().tolist()
    ):
        pooled_value = (
            float(
                mode_frame.loc[
                    (mode_frame["reference_feature_mode"] == reference_mode)
                    & (mode_frame["model_family"] == "pooled"),
                    "stage_macro_f1",
                ].iloc[0]
            )
            if (
                (mode_frame["reference_feature_mode"] == reference_mode)
                & (mode_frame["model_family"] == "pooled")
            ).any()
            else float("nan")
        )
        eamist_value = (
            float(
                mode_frame.loc[
                    (mode_frame["reference_feature_mode"] == reference_mode)
                    & (mode_frame["model_family"] == "eamist"),
                    "stage_macro_f1",
                ].iloc[0]
            )
            if (
                (mode_frame["reference_feature_mode"] == reference_mode)
                & (mode_frame["model_family"] == "eamist")
            ).any()
            else float("nan")
        )
        if np.isfinite(pooled_value) and np.isfinite(eamist_value):
            ablation_rows.append(
                {
                    "ablation": f"{reference_mode}: EA-MIST - pooled",
                    "delta_stage_macro_f1": eamist_value - pooled_value,
                }
            )
    if ablation_rows:
        save_ablation_figure(pd.DataFrame(ablation_rows), figures_root / "figure5_ablations.png")

    readme = (
        "# EA-MIST Benchmarks\n\n"
        "Active tables and figures for the lesion-level dual-reference Set Transformer benchmark.\n\n"
        f"- Benchmark summary: `{benchmark_summary_path}`\n"
        f"- Tables: `{tables_root}`\n"
        f"- Figures: `{figures_root}`\n"
        "- Note: the displacement target is weak stage-ordered supervision, not independent biological progression truth.\n"
    )
    (benchmarks_root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "benchmark_root": str(benchmark_root),
        "tables_root": str(tables_root),
        "figures_root": str(figures_root),
        "dataset_rows": int(dataset_table.shape[0]),
        "benchmark_rows": int(benchmark_summary.shape[0]),
    }
    (reports_root / "eamist_report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "ok": True,
        "pipeline": "run_eamist_reporting",
        "status": "complete",
        "reports_root": str(reports_root),
        "tables_root": str(tables_root),
        "figures_root": str(figures_root),
    }


__all__ = ["run_eamist_reporting"]
