#!/usr/bin/env python3
"""Aggregate baseline ladder results and compare to StageBridge.

This script produces the "baseline ladder" table for Nature Methods,
demonstrating that receiver-centered niche conditioning improves performance.

Outputs:
- baseline_comparison.json: Full comparison data
- table_baseline_ladder.tex: LaTeX table for paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]), float(values[0]))

    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                  for _ in range(n_bootstrap)]
    boot_means = np.array(boot_means)

    alpha = 1 - ci
    return (
        float(np.mean(values)),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def load_baseline_results(
    baseline_dir: Path,
    baselines: list[str],
    n_folds: int,
    seeds: list[int],
) -> pd.DataFrame:
    """Load all baseline results."""
    rows = []

    for baseline in baselines:
        for fold in range(n_folds):
            for seed in seeds:
                results_path = baseline_dir / baseline / f"fold{fold}_seed{seed}" / "results.json"

                if not results_path.exists():
                    print(f"  WARNING: Missing {results_path}")
                    continue

                with open(results_path) as f:
                    results = json.load(f)

                row = {"baseline": baseline, "fold": fold, "seed": seed}

                # Flatten metrics
                if "metrics" in results:
                    for key, value in results["metrics"].items():
                        if isinstance(value, (int, float)):
                            row[key] = value

                for key, value in results.items():
                    if isinstance(value, (int, float)) and key not in row:
                        row[key] = value

                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_baseline(df: pd.DataFrame, n_bootstrap: int = 1000) -> dict:
    """Aggregate metrics by baseline with bootstrap CIs."""
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


def generate_latex_table(
    baseline_agg: dict,
    main_model_agg: dict,
    output_path: Path,
) -> None:
    """Generate LaTeX table comparing baselines to StageBridge."""

    # Baseline order (increasing complexity)
    baseline_order = [
        ("pooling_mlp", "Pooling + MLP"),
        ("deep_sets", "DeepSets"),
        ("set_transformer", "Set Transformer"),
        ("graph_sage", "GraphSAGE"),
    ]

    # Key metrics
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

    # Add baseline rows
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

    # Add separator before StageBridge
    lines.append(r"\midrule")

    # Add StageBridge row (from main model results)
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
    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline results")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--main_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baselines", type=str, nargs="+", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating baseline results from {baseline_dir}")
    print(f"  Baselines: {args.baselines}")
    print(f"  Folds: {args.n_folds}, Seeds: {args.seeds}")

    # Load baseline results
    baseline_df = load_baseline_results(
        baseline_dir, args.baselines, args.n_folds, args.seeds
    )
    print(f"\nLoaded {len(baseline_df)} baseline runs")

    # Aggregate baselines
    baseline_agg = aggregate_by_baseline(baseline_df)

    # Load main model results
    with open(args.main_results) as f:
        main_results = json.load(f)

    main_model_agg = main_results.get("aggregated", {})

    # Compute improvement over best baseline
    improvements = {}
    for metric in ["stage_accuracy", "stage_f1_macro", "auroc"]:
        if metric not in main_model_agg:
            continue

        main_val = main_model_agg[metric]["mean"]
        best_baseline_val = max(
            baseline_agg[b].get(metric, {}).get("mean", 0)
            for b in args.baselines
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

    # Save comparison
    comparison = {
        "baselines": baseline_agg,
        "main_model": main_model_agg,
        "improvements": improvements,
        "raw_baseline_results": baseline_df.to_dict(orient="records"),
    }

    comparison_path = output_dir / "baseline_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved to: {comparison_path}")

    # Generate LaTeX table
    latex_path = output_dir / "table_baseline_ladder.tex"
    generate_latex_table(baseline_agg, main_model_agg, latex_path)

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE LADDER COMPARISON")
    print("=" * 60)

    for baseline in args.baselines:
        if baseline not in baseline_agg:
            continue
        print(f"\n{baseline}:")
        for metric in ["stage_accuracy", "auroc"]:
            if metric in baseline_agg[baseline]:
                m = baseline_agg[baseline][metric]
                print(f"  {metric}: {m['mean']:.3f} ± {m['std']:.3f}")

    print("\nStageBridge (main model):")
    for metric in ["stage_accuracy", "auroc"]:
        if metric in main_model_agg:
            m = main_model_agg[metric]
            print(f"  {metric}: {m['mean']:.3f} ± {m['std']:.3f}")

    if improvements:
        print("\nImprovements over best baseline:")
        for metric, imp in improvements.items():
            print(f"  {metric}: +{imp['relative_improvement_pct']:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
