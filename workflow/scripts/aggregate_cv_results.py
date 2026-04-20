#!/usr/bin/env python3
"""Aggregate CV results - Snakemake wrapper.

Core logic in stagebridge.evaluation.aggregation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stagebridge.evaluation.aggregation import (
    load_cv_results,
    aggregate_metrics,
    per_fold_summary,
    generate_latex_results_table,
)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CV results with bootstrap CIs")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results from {input_dir}")
    print(f"  Folds: {args.n_folds}, Seeds: {args.seeds}")

    df = load_cv_results(input_dir, n_folds=args.n_folds, seeds=args.seeds)
    print(f"Loaded {len(df)} training runs")

    aggregated = aggregate_metrics(df, n_bootstrap=args.n_bootstrap)
    fold_summary = per_fold_summary(df)

    cv_results = {
        "n_folds": args.n_folds,
        "seeds": args.seeds,
        "n_runs": len(df),
        "n_bootstrap": args.n_bootstrap,
        "raw_results": df.to_dict(orient="records"),
        "per_fold": fold_summary,
        "aggregated": aggregated,
    }

    cv_results_path = output_dir / "cv_results.json"
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=2, default=str)

    publication_metrics = {
        "summary": f"5-fold CV x {len(args.seeds)} seeds = {len(df)} runs",
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

    latex_path = output_dir / "table_main_results.tex"
    generate_latex_results_table(aggregated, latex_path)

    print(f"CV results saved to: {cv_results_path}")
    print(f"Publication metrics saved to: {pub_metrics_path}")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
