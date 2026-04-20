#!/usr/bin/env python3
"""Aggregate baseline results - Snakemake wrapper.

Core logic in stagebridge.evaluation.aggregation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stagebridge.evaluation.aggregation import (
    load_cv_results,
    load_baseline_results,
    aggregate_metrics,
    aggregate_by_baseline,
    compute_improvements,
    generate_latex_baseline_table,
)


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

    baseline_df = load_baseline_results(
        baseline_dir, args.baselines, args.n_folds, args.seeds
    )
    print(f"Loaded {len(baseline_df)} baseline runs")

    baseline_agg = aggregate_by_baseline(baseline_df)

    with open(args.main_results) as f:
        main_results = json.load(f)
    main_model_agg = main_results.get("aggregated", {})

    improvements = compute_improvements(main_model_agg, baseline_agg, args.baselines)

    comparison = {
        "baselines": baseline_agg,
        "main_model": main_model_agg,
        "improvements": improvements,
        "raw_baseline_results": baseline_df.to_dict(orient="records"),
    }

    comparison_path = output_dir / "baseline_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    latex_path = output_dir / "table_baseline_ladder.tex"
    generate_latex_baseline_table(baseline_agg, main_model_agg, latex_path)

    print(f"Comparison saved to: {comparison_path}")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
