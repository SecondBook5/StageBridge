#!/usr/bin/env python3
"""Generate ablation comparison figure for publication.

Creates a publication-quality figure showing:
- Performance degradation for each ablation
- Effect sizes with confidence intervals
- Ranked comparison bar chart

This is Figure 4 or similar in the paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ablation_results(ablation_dir: Path, ablations: list[str]) -> dict:
    """Load evaluation results for all ablations."""
    results = {}
    for ablation in ablations:
        eval_path = ablation_dir / ablation / "evaluation" / "heldout_evaluation.json"
        if eval_path.exists():
            with open(eval_path) as f:
                data = json.load(f)
                results[ablation] = data.get("transition_metrics", {})
    return results


def load_full_model_results(training_dir: Path) -> dict:
    """Load aggregated full model results."""
    summary_path = training_dir / "evaluation" / "heldout_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
            metrics = data.get("metrics", {})
            return {k: v.get("mean", 0) for k, v in metrics.items()}
    return {}


def create_ablation_figure(
    ablation_results: dict,
    full_model_results: dict,
    output_path: Path,
    metrics: list[str] = None,
):
    """Create publication-quality ablation comparison figure."""
    if metrics is None:
        metrics = ["wasserstein", "stage_accuracy"]

    ablation_names = {
        "no_niche": "No Niche",
        "no_wes": "No WES",
        "pooled_niche": "Pooled Niche",
        "flat_hierarchy": "Flat Hierarchy",
        "hlca_only": "HLCA Only",
        "luca_only": "LuCA Only",
        "deterministic": "Deterministic",
        "with_prototypes": "With Prototypes",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ablations = []
        values = []
        degradations = []

        full_val = full_model_results.get(metric, 0)

        for ablation, results in ablation_results.items():
            if metric in results:
                ablations.append(ablation_names.get(ablation, ablation))
                abl_val = results[metric]
                values.append(abl_val)

                if metric in ["wasserstein", "mmd", "mse"]:
                    deg = (abl_val - full_val) / (abs(full_val) + 1e-10) * 100
                else:
                    deg = (full_val - abl_val) / (abs(full_val) + 1e-10) * 100
                degradations.append(deg)

        sorted_idx = np.argsort(degradations)[::-1]
        ablations = [ablations[i] for i in sorted_idx]
        degradations = [degradations[i] for i in sorted_idx]

        colors = ["#d62728" if d > 5 else "#2ca02c" if d < -5 else "#7f7f7f" for d in degradations]

        bars = ax.barh(ablations, degradations, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1, linestyle="-")

        ax.set_xlabel("Performance Degradation (%)", fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")

        for bar, deg in zip(bars, degradations):
            x_pos = bar.get_width()
            ha = "left" if x_pos >= 0 else "right"
            offset = 0.5 if x_pos >= 0 else -0.5
            ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                   f"{deg:+.1f}%", va="center", ha=ha, fontsize=9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

    print(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation comparison figure")
    parser.add_argument("--ablation_dir", type=str, required=True)
    parser.add_argument("--training_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["wasserstein", "stage_accuracy"])
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir)
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        "no_niche", "no_wes", "pooled_niche", "flat_hierarchy",
        "hlca_only", "luca_only", "deterministic", "with_prototypes"
    ]

    print("Loading results...")
    ablation_results = load_ablation_results(ablation_dir, ablations)
    full_model_results = load_full_model_results(training_dir)

    print(f"Found {len(ablation_results)} ablation results")
    print(f"Full model metrics: {list(full_model_results.keys())}")

    output_path = output_dir / "fig_ablation_comparison.png"
    create_ablation_figure(ablation_results, full_model_results, output_path, args.metrics)


if __name__ == "__main__":
    main()
