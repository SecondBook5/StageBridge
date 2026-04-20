#!/usr/bin/env python3
"""Statistical comparison - Snakemake wrapper.

Core logic in stagebridge.evaluation.statistics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stagebridge.evaluation.statistics import (
    paired_comparison,
    apply_multiple_comparison_correction,
    generate_latex_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(description="Statistical model comparison")
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["wasserstein", "mmd", "stage_accuracy", "auroc"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = []
    model_results = []

    for results_file in Path(args.baseline_dir).glob("**/heldout_evaluation.json"):
        with open(results_file) as f:
            baseline_results.append(json.load(f))

    for results_file in Path(args.model_dir).glob("**/heldout_evaluation.json"):
        with open(results_file) as f:
            model_results.append(json.load(f))

    print(f"Found {len(baseline_results)} baseline results, {len(model_results)} model results")

    comparisons = []
    pvalues = []

    for metric in args.metrics:
        baseline_vals = []
        model_vals = []

        for r in baseline_results:
            if "transition_metrics" in r and metric in r["transition_metrics"]:
                baseline_vals.append(r["transition_metrics"][metric])

        for r in model_results:
            if "transition_metrics" in r and metric in r["transition_metrics"]:
                model_vals.append(r["transition_metrics"][metric])

        if baseline_vals and model_vals:
            min_len = min(len(baseline_vals), len(model_vals))
            comp = paired_comparison(
                np.array(baseline_vals[:min_len]),
                np.array(model_vals[:min_len]),
                metric,
            )
            comparisons.append(comp)
            if "paired_ttest_pvalue" in comp:
                pvalues.append(comp["paired_ttest_pvalue"])

    if pvalues:
        corrected_p, reject = apply_multiple_comparison_correction(pvalues, "fdr_bh")
        for i, comp in enumerate(comparisons):
            if "error" not in comp and i < len(corrected_p):
                comp["fdr_corrected_pvalue"] = corrected_p[i]
                comp["significant_after_fdr"] = reject[i]

    results = {
        "n_baseline_runs": len(baseline_results),
        "n_model_runs": len(model_results),
        "metrics_compared": args.metrics,
        "comparisons": comparisons,
        "correction_method": "fdr_bh",
    }

    results_path = output_dir / "statistical_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    latex_path = output_dir / "table_statistical_comparison.tex"
    generate_latex_comparison_table(comparisons, latex_path)

    print(f"Results saved to: {results_path}")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
