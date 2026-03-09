"""Reporting pipeline for EA-MIST benchmark outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def run_eamist_reporting(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Generate active EA-MIST tables and figures from saved run outputs."""
    reports_root = _ensure_dir(Path(str(_cfg_select(cfg, "eamist_report.reports_root", "reports"))))
    benchmark_root = Path(
        str(
            _cfg_select(
                cfg,
                "eamist_report.benchmark_root",
                Path(str(_cfg_select(cfg, "output_dir", "outputs/scratch"))) / str(_cfg_select(cfg, "run_name", "stagebridge_v1")) / "eamist_benchmark",
            )
        )
    )
    benchmark_summary_path = benchmark_root / "benchmark_summary.csv"
    model_family_summary_path = benchmark_root / "model_family_summary.csv"
    if not benchmark_summary_path.exists():
        raise FileNotFoundError(f"EA-MIST benchmark summary not found: {benchmark_summary_path}")

    benchmark_summary = pd.read_csv(benchmark_summary_path)
    model_family_summary = pd.read_csv(model_family_summary_path) if model_family_summary_path.exists() else pd.DataFrame()
    build_result = build_lesion_bags_from_config(cfg)

    tables_root = _ensure_dir(reports_root / "tables" / "eamist")
    figures_root = _ensure_dir(reports_root / "figures" / "eamist")
    benchmarks_root = _ensure_dir(reports_root / "benchmarks")

    dataset_table = build_result.summary.copy()
    dataset_table.to_csv(tables_root / "table1_dataset_composition.csv", index=False)
    benchmark_summary.to_csv(tables_root / "table2_benchmark_results.csv", index=False)
    model_family_summary.to_csv(tables_root / "table3_model_family_summary.csv", index=False)
    build_result.label_table.to_csv(tables_root / "table6_class_balance_and_labels.csv", index=False)

    per_donor_frames: list[pd.DataFrame] = []
    prototype_frames: list[pd.DataFrame] = []
    for per_donor_path in benchmark_root.glob("*/*/fold_*/seed_*/per_donor_metrics.csv"):
        frame = pd.read_csv(per_donor_path).copy()
        frame["artifact_dir"] = str(per_donor_path.parent)
        per_donor_frames.append(frame)
    for prototype_path in benchmark_root.glob("*/*/fold_*/seed_*/prototype_composition.parquet"):
        frame = pd.read_parquet(prototype_path).copy()
        frame["artifact_dir"] = str(prototype_path.parent)
        prototype_frames.append(frame)

    per_donor_table = pd.concat(per_donor_frames, ignore_index=True) if per_donor_frames else pd.DataFrame()
    prototype_table = pd.concat(prototype_frames, ignore_index=True) if prototype_frames else pd.DataFrame()
    if not per_donor_table.empty:
        per_donor_table.to_csv(tables_root / "table5_per_donor_results.csv", index=False)
    if not prototype_table.empty:
        prototype_enrichment = (
            prototype_table.groupby(["stage", "prototype"], as_index=False)["occupancy"]
            .mean()
            .sort_values(["stage", "occupancy"], ascending=[True, False])
        )
        prototype_enrichment.to_csv(tables_root / "table4_prototype_enrichment.csv", index=False)

    save_method_overview_figure(figures_root / "figure1_method_overview.png")

    embedding_candidates = sorted((benchmark_root.parent / "eamist_pretrain").glob("neighborhood_embeddings.parquet"))
    if embedding_candidates:
        embeddings = pd.read_parquet(embedding_candidates[0])
        save_embedding_diagnostics_figure(embeddings, figures_root / "figure2_embedding_diagnostics.png", color_column="stage")

    save_benchmark_comparison_figure(benchmark_summary, figures_root / "figure3_benchmark_comparison.png")

    if not prototype_table.empty:
        save_prototype_interpretation_figure(prototype_table, figures_root / "figure4_prototypes_attention.png")

    best_model = (
        benchmark_summary.groupby("model_family", as_index=False)["auroc"]
        .mean()
        .sort_values("auroc", ascending=False)
    )
    baseline_auroc = float(best_model.loc[best_model["model_family"] == "pooled", "auroc"].iloc[0]) if (best_model["model_family"] == "pooled").any() else float("nan")
    eamist_auroc = float(best_model.loc[best_model["model_family"] == "eamist", "auroc"].iloc[0]) if (best_model["model_family"] == "eamist").any() else float("nan")
    ablation_frame = pd.DataFrame(
        {
            "ablation": ["EA-MIST - pooled"],
            "delta_auroc": [eamist_auroc - baseline_auroc],
        }
    )
    save_ablation_figure(ablation_frame, figures_root / "figure5_ablations.png")

    readme = (
        "# EA-MIST Benchmarks\n\n"
        "Active tables and figures for the lesion-level Prototype-MIL Set Transformer benchmark.\n\n"
        f"- Benchmark summary: `{benchmark_summary_path}`\n"
        f"- Tables: `{tables_root}`\n"
        f"- Figures: `{figures_root}`\n"
    )
    (benchmarks_root / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "benchmark_root": str(benchmark_root),
        "tables_root": str(tables_root),
        "figures_root": str(figures_root),
        "dataset_rows": int(dataset_table.shape[0]),
        "benchmark_rows": int(benchmark_summary.shape[0]),
    }
    (reports_root / "eamist_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "pipeline": "run_eamist_reporting",
        "status": "complete",
        "reports_root": str(reports_root),
        "tables_root": str(tables_root),
        "figures_root": str(figures_root),
    }


__all__ = ["run_eamist_reporting"]
