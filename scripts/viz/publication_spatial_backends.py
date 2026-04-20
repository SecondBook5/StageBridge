#!/usr/bin/env python3
"""Publication-quality spatial backend comparison showcase figure.

Collects per-sample backend comparison grids and creates a showcase with
representative samples from each stage, showing how different deconvolution
methods perform on the same tissue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np


STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

STAGE_COLORS = {
    'Normal': '#2ecc71',
    'AAH': '#f39c12',
    'AIS': '#e74c3c',
    'MIA': '#9b59b6',
    'LUAD': '#1a1a2e',
}


def find_sample_figures(spatial_dir: Path, label_source: str = "hlca") -> dict[str, list[Path]]:
    """Find backend comparison figures organized by stage."""
    figures_by_stage = {stage: [] for stage in STAGE_ORDER}

    backend_comparison_dir = spatial_dir / label_source / "figures" / "backend_comparison"

    if not backend_comparison_dir.exists():
        print(f"  Warning: {backend_comparison_dir} does not exist")
        return figures_by_stage

    for fig_path in backend_comparison_dir.glob("*_backend_comparison.png"):
        sample_name = fig_path.stem.replace("_backend_comparison", "")
        parts = sample_name.split('_')

        stage = None
        for part in parts:
            for s in STAGE_ORDER:
                if s in part or part.startswith(s):
                    stage = s
                    break
            if stage:
                break

        if stage:
            figures_by_stage[stage].append(fig_path)

    return figures_by_stage


def select_representative_samples(figures_by_stage: dict[str, list[Path]], n_per_stage: int = 1) -> list[Path]:
    """Select representative samples from each stage."""
    selected = []

    for stage in STAGE_ORDER:
        figs = figures_by_stage.get(stage, [])
        if figs:
            figs_sorted = sorted(figs, key=lambda p: p.stat().st_size, reverse=True)
            selected.extend(figs_sorted[:n_per_stage])

    return selected


def create_showcase_figure(
    sample_figures: list[Path],
    output_path: Path,
    title: str = "Spatial Deconvolution: Backend Comparison Across Disease Stages",
):
    """Create a showcase figure combining multiple sample comparisons."""
    n_samples = len(sample_figures)

    if n_samples == 0:
        print("  No sample figures to showcase")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No backend comparison figures found.\nRun spatial_unified_figures rule first.",
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    ncols = min(n_samples, 3)
    nrows = (n_samples + ncols - 1) // ncols

    fig = plt.figure(figsize=(8 * ncols, 8 * nrows + 1))

    gs = GridSpec(nrows + 1, ncols, figure=fig, height_ratios=[0.05] + [1] * nrows, hspace=0.1, wspace=0.05)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.5, title, fontsize=16, fontweight='bold', ha='center', va='center')
    title_ax.set_xticks([])
    title_ax.set_yticks([])
    for spine in title_ax.spines.values():
        spine.set_visible(False)

    for idx, fig_path in enumerate(sample_figures):
        row = idx // ncols + 1
        col = idx % ncols

        ax = fig.add_subplot(gs[row, col])

        try:
            img = mpimg.imread(fig_path)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{fig_path.name}", ha='center', va='center', fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        sample_name = fig_path.stem.replace("_backend_comparison", "")
        parts = sample_name.split('_')
        stage = "Unknown"
        for part in parts:
            for s in STAGE_ORDER:
                if s in part:
                    stage = s
                    break

        stage_color = STAGE_COLORS.get(stage, '#333333')
        ax.set_title(f"{sample_name}", fontsize=10, color=stage_color, fontweight='bold', pad=5)

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  Saved showcase: {output_path}")

    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  Saved PDF: {pdf_path}")

    plt.close(fig)


def copy_individual_figures(
    figures_by_stage: dict[str, list[Path]],
    output_dir: Path,
):
    """Copy individual figures to output directory organized by stage."""
    for stage, figs in figures_by_stage.items():
        if not figs:
            continue

        stage_dir = output_dir / "by_stage" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        for fig_path in figs:
            dest = stage_dir / fig_path.name
            shutil.copy2(fig_path, dest)

    print(f"  Copied individual figures to {output_dir / 'by_stage'}")


def main():
    parser = argparse.ArgumentParser(description="Create spatial backend comparison showcase")
    parser.add_argument("--spatial_dir", type=Path, required=True, help="Spatial benchmark directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for figures")
    parser.add_argument("--label_source", type=str, default="hlca", help="Label source (hlca or luca)")
    parser.add_argument("--n_per_stage", type=int, default=1, help="Number of samples per stage in showcase")
    args = parser.parse_args()

    print("=" * 60)
    print("Spatial Backend Comparison Showcase")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nFinding sample figures in {args.spatial_dir}...")
    figures_by_stage = find_sample_figures(args.spatial_dir, args.label_source)

    for stage, figs in figures_by_stage.items():
        print(f"  {stage}: {len(figs)} figures")

    print(f"\nSelecting representative samples...")
    selected = select_representative_samples(figures_by_stage, n_per_stage=args.n_per_stage)
    print(f"  Selected {len(selected)} samples for showcase")

    print(f"\nCreating showcase figure...")
    showcase_path = args.output_dir / "fig_backend_grid_showcase.png"
    create_showcase_figure(selected, showcase_path)

    print(f"\nCopying individual figures...")
    copy_individual_figures(figures_by_stage, args.output_dir)

    manifest = {
        "showcase": str(showcase_path),
        "showcase_pdf": str(showcase_path.with_suffix('.pdf')),
        "n_samples_total": sum(len(figs) for figs in figures_by_stage.values()),
        "n_samples_per_stage": {stage: len(figs) for stage, figs in figures_by_stage.items()},
        "selected_samples": [str(p) for p in selected],
        "label_source": args.label_source,
    }

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {manifest_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
