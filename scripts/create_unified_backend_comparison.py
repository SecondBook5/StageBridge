#!/usr/bin/env python3
"""
Create unified backend comparison figure with ALL backends on the same plot.

Reproduces the 2x2 backend_comparison.png layout but with all backends overlaid.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Color palette for backends
BACKEND_COLORS = {
    "tangram": "#1f77b4",      # blue
    "destvi": "#ff7f0e",       # orange
    "tacco": "#2ca02c",        # green
    "cell2location": "#d62728", # red
    "marker_scoring": "#9467bd", # purple
    "card": "#8c564b",         # brown
    "rctd": "#e377c2",         # pink
    "spotlight": "#7f7f7f",    # gray
}


def load_all_backend_metrics(
    results_dir: Path,
    sample_id: str,
    label_source: str = "hlca",
) -> pd.DataFrame:
    """Load metrics from all backends for a sample."""
    base = results_dir / label_source
    rows = []

    for backend_dir in base.iterdir():
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name

        # Load comparison JSON
        json_path = backend_dir / "samples" / sample_id / "backend_comparison.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)

        # Extract rankings (has the computed scores)
        if "rankings" in data and len(data["rankings"]) > 0:
            ranking = data["rankings"][0]  # First (only) entry
            rows.append({
                "backend": backend,
                "composite_score": ranking.get("composite_score", 0),
                "entropy_score": ranking.get("entropy_score", 0),
                "coverage_score": ranking.get("coverage_score", 0),
                "sparsity_score": ranking.get("sparsity_score", 0),
                "runtime_score": ranking.get("runtime_score", 0),
                "confidence_score": ranking.get("confidence_score", 0),
                "mean_entropy": ranking.get("mean_entropy", 0),
                "coverage": ranking.get("coverage", 0),
                "runtime": ranking.get("runtime", 0),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("composite_score", ascending=False)


def create_unified_comparison_figure(
    df: pd.DataFrame,
    sample_id: str,
    output_path: Path,
):
    """Create the unified 2x2 comparison figure with all backends."""
    if df.empty:
        print(f"  No data for {sample_id}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    backends = df["backend"].tolist()
    colors = [BACKEND_COLORS.get(b, "#333333") for b in backends]

    # 1. Overall Performance (horizontal bar chart)
    ax = axes[0, 0]
    y_pos = np.arange(len(backends))
    ax.barh(y_pos, df["composite_score"], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([b.upper() for b in backends])
    ax.set_xlabel("Composite Score")
    ax.set_title("Overall Performance", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # 2. Radar chart of individual metrics
    ax = axes[0, 1]
    ax.axis('off')  # Turn off regular axes

    # Create polar subplot in same position
    rect = ax.get_position()
    ax_polar = fig.add_axes(rect, projection='polar')

    metrics = ["entropy_score", "coverage_score", "sparsity_score", "runtime_score", "confidence_score"]
    metric_labels = ["entropy", "coverage", "sparsity", "runtime", "confidence"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for idx, row in df.iterrows():
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        color = BACKEND_COLORS.get(row["backend"], "#333333")
        ax_polar.plot(angles, values, "o-", linewidth=2, label=row["backend"].upper(), color=color)
        ax_polar.fill(angles, values, alpha=0.15, color=color)

    ax_polar.set_xticks(angles[:-1])
    ax_polar.set_xticklabels(metric_labels)
    ax_polar.set_ylim(0, 1)
    ax_polar.set_title("Metric Breakdown", fontsize=12, fontweight='bold', y=1.1)

    # 3. Runtime comparison (horizontal bar chart)
    ax = axes[1, 0]
    ax.barh(y_pos, df["runtime"], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([b.upper() for b in backends])
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Computational Cost", fontsize=12, fontweight='bold')
    ax.invert_yaxis()

    # 4. Entropy vs Coverage scatter
    ax = axes[1, 1]
    for idx, row in df.iterrows():
        color = BACKEND_COLORS.get(row["backend"], "#333333")
        ax.scatter(
            row["mean_entropy"],
            row["coverage"],
            s=200,
            c=[color],
            edgecolors="black",
            linewidths=2,
            zorder=5,
        )
        ax.annotate(
            row["backend"].upper(),
            (row["mean_entropy"], row["coverage"]),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight='bold',
        )

    ax.set_xlabel("Mean Entropy (Diversity)")
    ax.set_ylabel("Coverage (Confidence)")
    ax.set_title("Quality Trade-offs", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_handles = [
        Patch(facecolor=BACKEND_COLORS.get(b, "#333333"), edgecolor='black', label=b.upper())
        for b in backends
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(backends),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    # Title
    fig.suptitle(
        f"Backend Comparison: {sample_id}",
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def get_all_samples(results_dir: Path, label_source: str = "hlca") -> list[str]:
    """Get all sample IDs."""
    base = results_dir / label_source
    for backend_dir in base.iterdir():
        if backend_dir.is_dir():
            samples_dir = backend_dir / "samples"
            if samples_dir.exists():
                return [s.name for s in samples_dir.iterdir() if s.is_dir()]
    return []


def process_sample(
    results_dir: Path,
    sample_id: str,
    output_dir: Path,
    label_source: str = "hlca",
):
    """Process a single sample."""
    print(f"Processing {sample_id}...")

    df = load_all_backend_metrics(results_dir, sample_id, label_source)
    if df.empty:
        print(f"  No metrics found for {sample_id}")
        return

    print(f"  Found {len(df)} backends")

    create_unified_comparison_figure(
        df,
        sample_id,
        output_dir / f"{sample_id}_unified_comparison.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create unified backend comparison figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/spatial_benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spatial_benchmark/figures/unified_comparison"),
    )
    parser.add_argument("--label-source", default="hlca")
    parser.add_argument("--sample", type=str, default=None)

    args = parser.parse_args()

    if args.sample:
        process_sample(args.results_dir, args.sample, args.output_dir, args.label_source)
    else:
        samples = get_all_samples(args.results_dir, args.label_source)
        print(f"Processing {len(samples)} samples...")
        for sample in sorted(samples):
            process_sample(args.results_dir, sample, args.output_dir, args.label_source)
