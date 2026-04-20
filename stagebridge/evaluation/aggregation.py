"""Aggregation utilities for CV results and baseline comparisons.

Handles:
- 5-fold CV x multi-seed aggregation with bootstrap CIs
- Baseline ladder comparison
- Publication-ready metrics formatting
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.evaluation.statistics import bootstrap_ci


def load_cv_results(
    input_dir: Path,
    n_folds: int = 5,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Load all results.json files into a DataFrame.

    Args:
        input_dir: Directory containing fold/seed subdirectories
        n_folds: Number of CV folds
        seeds: List of seeds used

    Returns:
        DataFrame with all results
    """
    if seeds is None:
        seeds = [42, 123, 456]

    rows = []

    for fold in range(n_folds):
        for seed in seeds:
            results_path = input_dir / f"fold{fold}_seed{seed}" / "results.json"

            if not results_path.exists():
                continue

            with open(results_path) as f:
                results = json.load(f)

            row = {"fold": fold, "seed": seed}

            if "metrics" in results:
                for key, value in results["metrics"].items():
                    if isinstance(value, (int, float)):
                        row[key] = value

            for key, value in results.items():
                if isinstance(value, (int, float)) and key not in row:
                    row[key] = value

            rows.append(row)

    if not rows:
        raise ValueError(f"No results found in {input_dir}")

    return pd.DataFrame(rows)


def aggregate_metrics(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics across folds and seeds with bootstrap CIs.

    Args:
        df: DataFrame with metric columns
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        Dictionary of metric -> aggregated stats
    """
    metric_cols = [c for c in df.columns if c not in ["fold", "seed"]]

    aggregated = {}

    for metric in metric_cols:
        values = df[metric].dropna().values

        if len(values) == 0:
            continue

        mean, ci_lower, ci_upper = bootstrap_ci(
            values, n_bootstrap=n_bootstrap, seed=seed
        )
        std = float(np.std(values))

        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_runs": len(values),
            "formatted": f"{mean:.3f} +/- {std:.3f}",
            "formatted_ci": f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]",
        }

    return aggregated


def per_fold_summary(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-fold statistics (mean across seeds within each fold).

    Args:
        df: DataFrame with fold, seed, and metric columns

    Returns:
        Dictionary of fold -> metric stats
    """
    metric_cols = [c for c in df.columns if c not in ["fold", "seed"]]

    fold_stats = {}
    for fold in df["fold"].unique():
        fold_df = df[df["fold"] == fold]
        fold_stats[f"fold_{fold}"] = {
            metric: {
                "mean": float(fold_df[metric].mean()),
                "std": float(fold_df[metric].std()),
                "n_seeds": len(fold_df),
            }
            for metric in metric_cols
            if metric in fold_df.columns
        }

    return fold_stats


def load_baseline_results(
    baseline_dir: Path,
    baselines: list[str],
    n_folds: int = 5,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Load all baseline results.

    Args:
        baseline_dir: Directory with baseline results
        baselines: List of baseline names
        n_folds: Number of folds
        seeds: List of seeds

    Returns:
        DataFrame with all baseline results
    """
    if seeds is None:
        seeds = [42, 123, 456]

    rows = []

    for baseline in baselines:
        for fold in range(n_folds):
            for seed in seeds:
                results_path = baseline_dir / f"{baseline}_fold{fold}_seed{seed}" / "results.json"

                if not results_path.exists():
                    continue

                with open(results_path) as f:
                    results = json.load(f)

                row = {"baseline": baseline, "fold": fold, "seed": seed}

                if "metrics" in results:
                    for key, value in results["metrics"].items():
                        if isinstance(value, (int, float)):
                            row[key] = value

                for key, value in results.items():
                    if isinstance(value, (int, float)) and key not in row:
                        row[key] = value

                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_baseline(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, dict]]:
    """Aggregate metrics by baseline with bootstrap CIs.

    Args:
        df: DataFrame with baseline, fold, seed, and metric columns
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary of baseline -> metric -> aggregated stats
    """
    metric_cols = [c for c in df.columns if c not in ["baseline", "fold", "seed"]]

    aggregated = {}

    for baseline in df["baseline"].unique():
        baseline_df = df[df["baseline"] == baseline]
        baseline_agg = {}

        for metric in metric_cols:
            values = baseline_df[metric].dropna().values
            if len(values) == 0:
                continue

            mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap)
            std = float(np.std(values))

            baseline_agg[metric] = {
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_runs": len(values),
            }

        aggregated[baseline] = baseline_agg

    return aggregated


def compute_improvements(
    main_model_agg: dict[str, dict],
    baseline_agg: dict[str, dict[str, dict]],
    baselines: list[str],
    metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute improvement over best baseline.

    Args:
        main_model_agg: Aggregated main model metrics
        baseline_agg: Aggregated baseline metrics
        baselines: List of baseline names
        metrics: Metrics to compare (defaults to accuracy metrics)

    Returns:
        Dictionary of metric -> improvement stats
    """
    if metrics is None:
        metrics = ["stage_accuracy", "stage_f1_macro", "auroc"]

    improvements = {}

    for metric in metrics:
        if metric not in main_model_agg:
            continue

        main_val = main_model_agg[metric]["mean"]
        best_baseline_val = max(
            baseline_agg[b].get(metric, {}).get("mean", 0)
            for b in baselines
            if b in baseline_agg
        )

        if best_baseline_val > 0:
            rel_improvement = (main_val - best_baseline_val) / best_baseline_val * 100
            improvements[metric] = {
                "main_model": main_val,
                "best_baseline": best_baseline_val,
                "absolute_improvement": main_val - best_baseline_val,
                "relative_improvement_pct": rel_improvement,
            }

    return improvements


def generate_latex_results_table(
    aggregated: dict[str, dict],
    output_path: Path,
    key_metrics: list[tuple[str, str]] | None = None,
) -> None:
    """Generate LaTeX table for publication main results.

    Args:
        aggregated: Aggregated metrics dictionary
        output_path: Output file path
        key_metrics: List of (metric_key, display_name) tuples
    """
    if key_metrics is None:
        key_metrics = [
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("accuracy", "Accuracy"),
            ("f1", "F1 Score"),
            ("val_loss", "Validation Loss"),
            ("flow_recovery_cosine", "Flow Recovery (cosine)"),
            ("niche_influence_corr", "Niche Influence (r)"),
        ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{StageBridge V1 Performance (5-fold CV $\times$ 3 seeds)}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Mean $\pm$ Std & 95\% CI \\",
        r"\midrule",
    ]

    for metric_key, metric_name in key_metrics:
        if metric_key in aggregated:
            m = aggregated[metric_key]
            lines.append(
                f"{metric_name} & ${m['mean']:.3f} \\pm {m['std']:.3f}$ & "
                f"$[{m['ci_lower']:.3f}, {m['ci_upper']:.3f}]$ \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines))


def generate_latex_baseline_table(
    baseline_agg: dict[str, dict[str, dict]],
    main_model_agg: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate LaTeX table comparing baselines to StageBridge.

    Args:
        baseline_agg: Aggregated baseline metrics
        main_model_agg: Aggregated main model metrics
        output_path: Output file path
    """
    baseline_order = [
        ("pooling_mlp", "Pooling + MLP"),
        ("deep_sets", "DeepSets"),
        ("set_transformer", "Set Transformer"),
        ("graph_sage", "GraphSAGE"),
    ]

    metrics = [
        ("stage_accuracy", "Stage Acc."),
        ("stage_f1_macro", "Stage F1"),
        ("transition_cosine", "Flow Cos."),
        ("auroc", "AUROC"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Baseline Ladder: Impact of Receiver-Centered Niche Conditioning}",
        r"\label{tab:baseline_ladder}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        r"Model & " + " & ".join([name for _, name in metrics]) + r" \\",
        r"\midrule",
    ]

    for baseline_key, baseline_name in baseline_order:
        if baseline_key not in baseline_agg:
            continue

        row_parts = [baseline_name]
        for metric_key, _ in metrics:
            if metric_key in baseline_agg[baseline_key]:
                m = baseline_agg[baseline_key][metric_key]
                row_parts.append(f"${m['mean']:.3f} \\pm {m['std']:.3f}$")
            else:
                row_parts.append("--")

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")

    row_parts = [r"\textbf{StageBridge (Ours)}"]
    for metric_key, _ in metrics:
        if metric_key in main_model_agg:
            m = main_model_agg[metric_key]
            row_parts.append(f"$\\mathbf{{{m['mean']:.3f} \\pm {m['std']:.3f}}}$")
        else:
            row_parts.append("--")

    lines.append(" & ".join(row_parts) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"\footnotesize{Results are mean $\pm$ std over 5-fold CV $\times$ 3 seeds. "
        r"StageBridge adds receiver-centered niche context and dual-reference anchoring.}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines))
