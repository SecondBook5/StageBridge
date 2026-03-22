#!/usr/bin/env python3
"""Demonstration script for publication plotting infrastructure.

This script demonstrates all key publication plotting capabilities:
1. Style configuration
2. Figure creation
3. Stage color usage
4. Multi-format export
5. Advanced plot types

Run: python scripts/demo_publication_plotting.py
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from stagebridge.viz import (
    setup_publication_plotting,
    create_figure,
    create_subplots,
    save_publication_figure,
    get_stage_color,
    add_clean_legend,
    plot_radar_chart,
    plot_parallel_coordinates,
    plot_ridge_distributions,
    PUBLICATION_PALETTE,
)
from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def demo_basic_figure(output_dir: Path) -> None:
    """Demo 1: Basic publication figure with stage colors."""
    log.info("Demo 1: Basic figure with stage colors")

    fig, ax = create_figure(figsize=(8, 6))

    # Plot data for each stage
    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    x = np.linspace(0, 10, 100)

    for i, stage in enumerate(stages):
        y = np.sin(x + i) + i
        ax.plot(x, y, label=stage, color=get_stage_color(stage), linewidth=2)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Expression Level")
    ax.set_title("Gene Expression Across Cancer Stages")
    ax.grid(True, alpha=0.3)
    add_clean_legend(ax, title="Stage")

    paths = save_publication_figure(fig, output_dir / "demo1_basic")
    log.info(f"Saved: {list(paths.values())}")


def demo_multi_panel(output_dir: Path) -> None:
    """Demo 2: Multi-panel figure."""
    log.info("Demo 2: Multi-panel figure (2x2)")

    fig, axes = create_subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Panel A: Line plot
    x = np.linspace(0, 10, 50)
    for stage in ["Normal", "AAH", "AIS"]:
        axes[0, 0].plot(x, np.sin(x), label=stage, color=get_stage_color(stage))
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    add_clean_legend(axes[0, 0])

    # Panel B: Scatter plot
    rng = np.random.default_rng(42)
    for stage in ["MIA", "LUAD"]:
        axes[0, 1].scatter(
            rng.normal(0, 1, 50),
            rng.normal(0, 1, 50),
            label=stage,
            color=get_stage_color(stage),
            alpha=0.6,
        )
    axes[0, 1].set_title("Embedding Space")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    add_clean_legend(axes[0, 1])

    # Panel C: Bar plot
    stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    values = [0.85, 0.78, 0.72, 0.68, 0.65]
    colors = [get_stage_color(s) for s in stages]
    axes[1, 0].bar(stages, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    axes[1, 0].set_title("Classification Accuracy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Panel D: Heatmap
    data = rng.random((5, 5))
    im = axes[1, 1].imshow(data, cmap="viridis", aspect="auto")
    axes[1, 1].set_title("Confusion Matrix")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Add panel labels (A, B, C, D)
    for i, ax in enumerate(axes.flat):
        ax.text(
            -0.15,
            1.05,
            chr(65 + i),
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    paths = save_publication_figure(fig, output_dir / "demo2_multipanel")
    log.info(f"Saved: {list(paths.values())}")


def demo_radar_chart(output_dir: Path) -> None:
    """Demo 3: Radar chart for model comparison."""
    log.info("Demo 3: Radar chart")

    df = pd.DataFrame(
        {
            "label": ["Baseline", "Model A", "Model B"],
            "accuracy": [0.75, 0.85, 0.90],
            "precision": [0.72, 0.82, 0.88],
            "recall": [0.78, 0.87, 0.91],
            "f1_score": [0.74, 0.84, 0.89],
            "auc": [0.76, 0.86, 0.92],
        }
    )

    fig = plot_radar_chart(
        df,
        metrics=["accuracy", "precision", "recall", "f1_score", "auc"],
        labels_col="label",
        output_path=output_dir / "demo3_radar.png",
        title="Model Performance Comparison",
        normalize=True,
    )
    log.info(f"Saved: {output_dir / 'demo3_radar.png'} and .pdf")


def demo_parallel_coordinates(output_dir: Path) -> None:
    """Demo 4: Parallel coordinates plot."""
    log.info("Demo 4: Parallel coordinates")

    df = pd.DataFrame(
        {
            "label": ["Baseline", "Model A", "Model B", "Model C"],
            "metric1": [0.65, 0.75, 0.85, 0.80],
            "metric2": [0.70, 0.80, 0.88, 0.82],
            "metric3": [0.68, 0.78, 0.86, 0.84],
            "metric4": [0.72, 0.82, 0.90, 0.85],
        }
    )

    fig = plot_parallel_coordinates(
        df,
        metrics=["metric1", "metric2", "metric3", "metric4"],
        labels_col="label",
        output_path=output_dir / "demo4_parallel.png",
        title="High-Dimensional Model Comparison",
        normalize=True,
    )
    log.info(f"Saved: {output_dir / 'demo4_parallel.png'} and .pdf")


def demo_ridge_plot(output_dir: Path) -> None:
    """Demo 5: Ridge plot for distribution comparison."""
    log.info("Demo 5: Ridge plot")

    rng = np.random.default_rng(42)
    data_dict = {
        "Normal": rng.normal(0, 1, 500),
        "AAH": rng.normal(1, 1.2, 500),
        "AIS": rng.normal(2, 1.5, 500),
        "MIA": rng.normal(3, 1.8, 500),
        "LUAD": rng.normal(4, 2.0, 500),
    }

    colors = [get_stage_color(stage) for stage in data_dict.keys()]

    fig = plot_ridge_distributions(
        data_dict,
        output_path=output_dir / "demo5_ridge.png",
        title="Expression Distribution Across Stages",
        colors=colors,
    )
    log.info(f"Saved: {output_dir / 'demo5_ridge.png'} and .pdf")


def main() -> None:
    """Run all demos."""
    # Setup publication style
    setup_publication_plotting()

    # Create temp directory for outputs (replace with actual path if needed)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        log.info(f"Output directory: {output_dir}")

        # Run all demos
        demo_basic_figure(output_dir)
        demo_multi_panel(output_dir)
        demo_radar_chart(output_dir)
        demo_parallel_coordinates(output_dir)
        demo_ridge_plot(output_dir)

        # List all generated files
        log.info("\n=== Generated Files ===")
        for path in sorted(output_dir.glob("*")):
            log.info(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    log.info("\n=== Demo Complete ===")
    log.info("All publication plotting utilities working correctly!")
    log.info("Color palette (colorblind-safe):")
    for stage, color in PUBLICATION_PALETTE.items():
        if stage not in ["ink", "grid", "background"]:
            log.info(f"  {stage:10s}: {color}")


if __name__ == "__main__":
    main()
