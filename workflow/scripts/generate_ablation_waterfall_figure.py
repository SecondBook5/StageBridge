#!/usr/bin/env python3
"""Generate ablation waterfall figure.

Thin wrapper around stagebridge.viz.ablation_waterfall for Snakemake integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stagebridge.viz.ablation_waterfall import (
    compute_degradation,
    plot_ablation_waterfall,
    plot_multi_metric_waterfall,
    plot_component_importance,
    plot_ablation_summary,
)


def load_ablation_results(ablation_dir: Path) -> dict:
    """Load all ablation evaluation results."""
    results = {}

    for ablation_path in ablation_dir.iterdir():
        if not ablation_path.is_dir():
            continue

        ablation_name = ablation_path.name

        eval_path = ablation_path / "evaluation" / "heldout_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
                results[ablation_name] = data.get("transition_metrics", {})
            continue

        results_path = ablation_path / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
                metrics = data.get("metrics", data.get("final_metrics", {}))
                results[ablation_name] = metrics

    return results


def load_full_model_results(training_dir: Path) -> dict:
    """Load full model evaluation results."""
    eval_path = training_dir / "evaluation" / "heldout_summary.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            return {k: v.get("mean", 0) for k, v in metrics.items() if isinstance(v, dict)}

    cv_path = training_dir / "aggregated" / "cv_results.json"
    if cv_path.exists():
        with open(cv_path) as f:
            data = json.load(f)
            return data.get("aggregated", {})

    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate ablation waterfall figures")
    parser.add_argument("--ablation_dir", type=str, required=True)
    parser.add_argument("--training_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["wasserstein", "stage_accuracy", "auroc"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ablation results...")
    ablation_results = load_ablation_results(Path(args.ablation_dir))
    full_model_results = load_full_model_results(Path(args.training_dir))

    print(f"Found {len(ablation_results)} ablations")
    print(f"Full model metrics: {list(full_model_results.keys())}")

    for metric in args.metrics:
        print(f"\nGenerating waterfall for {metric}...")
        deg_df = compute_degradation(ablation_results, full_model_results, metric)

        metric_labels = {
            "wasserstein": "Wasserstein Distance",
            "stage_accuracy": "Stage Accuracy",
            "auroc": "AUROC"
        }
        plot_ablation_waterfall(
            deg_df,
            output_dir / f"fig_waterfall_{metric}.png",
            metric_name=metric_labels.get(metric, metric),
        )

    print("\nGenerating multi-metric waterfall...")
    plot_multi_metric_waterfall(
        ablation_results, full_model_results,
        output_dir / "fig_waterfall_multi_metric.png",
        metrics=args.metrics,
    )

    print("Generating component importance figure...")
    plot_component_importance(
        ablation_results, full_model_results,
        output_dir / "fig_component_importance.png",
    )

    print("Generating ablation summary figure...")
    plot_ablation_summary(
        ablation_results, full_model_results,
        output_dir / "fig_ablation_summary.png",
    )

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
