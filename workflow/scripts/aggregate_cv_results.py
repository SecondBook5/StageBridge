#!/usr/bin/env python3
"""Aggregate 5-fold CV × multi-seed results with bootstrap confidence intervals.

Produces final metrics with proper uncertainty quantification.

Outputs:
- cv_results.json: Full results from all fold/seed combinations
- publication_metrics.json: Mean ± std with 95% CIs for each metric
- table_main_results.tex: LaTeX table for paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    values = values[~np.isnan(values)]  # Remove NaNs

    if len(values) == 0:
        return (np.nan, np.nan, np.nan)

    if len(values) == 1:
        return (float(values[0]), float(values[0]), float(values[0]))

    # Bootstrap resampling
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = 1 - ci
    ci_lower = float(np.quantile(boot_means, alpha / 2))
    ci_upper = float(np.quantile(boot_means, 1 - alpha / 2))
    mean = float(np.mean(values))

    return (mean, ci_lower, ci_upper)


def load_results(input_dir: Path, n_folds: int, seeds: list[int]) -> pd.DataFrame:
    """Load all results.json files into a DataFrame."""
    rows = []

    for fold in range(n_folds):
        for seed in seeds:
            results_path = input_dir / f"fold{fold}_seed{seed}" / "results.json"

            if not results_path.exists():
                print(f"  WARNING: Missing {results_path}")
                continue

            with open(results_path) as f:
                results = json.load(f)

            row = {
                "fold": fold,
                "seed": seed,
            }

            # Flatten metrics from results
            if "metrics" in results:
                for key, value in results["metrics"].items():
                    if isinstance(value, (int, float)):
                        row[key] = value

            # Also check top-level numeric values
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
) -> dict[str, Any]:
    """Aggregate metrics across folds and seeds with bootstrap CIs."""

    # Metrics to aggregate (exclude fold, seed)
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
            # Formatted strings for paper
            "formatted": f"{mean:.3f} ± {std:.3f}",
            "formatted_ci": f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]",
        }

    return aggregated


def per_fold_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-fold statistics (mean across seeds within each fold)."""
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


def generate_latex_table(aggregated: dict[str, Any], output_path: Path) -> None:
    """Generate LaTeX table for Nature Methods main results."""

    # Define metrics of interest for the paper (customize as needed)
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
    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate CV results with bootstrap CIs")
    parser.add_argument("--input_dir", type=str, required=True, help="Training output directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for aggregated results")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Seeds used")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results from {input_dir}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Bootstrap samples: {args.n_bootstrap}")

    # Load all results
    df = load_results(input_dir, args.n_folds, args.seeds)
    print(f"\nLoaded {len(df)} training runs")
    print(f"  Folds: {sorted(df['fold'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

    # Compute aggregated metrics with CIs
    aggregated = aggregate_metrics(df, n_bootstrap=args.n_bootstrap)

    # Compute per-fold summary
    fold_summary = per_fold_summary(df)

    # Full CV results
    cv_results = {
        "n_folds": args.n_folds,
        "seeds": args.seeds,
        "n_runs": len(df),
        "n_bootstrap": args.n_bootstrap,
        "raw_results": df.to_dict(orient="records"),
        "per_fold": fold_summary,
        "aggregated": aggregated,
    }

    # Save CV results
    cv_results_path = output_dir / "cv_results.json"
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)
    print(f"\nCV results saved to: {cv_results_path}")

    # Publication metrics (cleaner format)
    publication_metrics = {
        "summary": f"5-fold CV × {len(args.seeds)} seeds = {len(df)} runs",
        "metrics": {
            k: {
                "mean": v["mean"],
                "std": v["std"],
                "ci_95": [v["ci_lower"], v["ci_upper"]],
                "formatted": v["formatted_ci"],
            }
            for k, v in aggregated.items()
        },
    }

    pub_metrics_path = output_dir / "publication_metrics.json"
    with open(pub_metrics_path, "w") as f:
        json.dump(publication_metrics, f, indent=2)
    print(f"Publication metrics saved to: {pub_metrics_path}")

    # Generate LaTeX table
    latex_path = output_dir / "table_main_results.tex"
    generate_latex_table(aggregated, latex_path)

    # Print summary
    print("\n" + "=" * 60)
    print("PUBLICATION METRICS (5-fold CV × 3 seeds)")
    print("=" * 60)
    for metric, values in sorted(aggregated.items()):
        print(f"  {metric}: {values['formatted_ci']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
