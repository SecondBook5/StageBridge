#!/usr/bin/env python3
"""
Combine spatial deconvolution backend comparison figures per sample.

Creates a grid showing all backends side-by-side for easy comparison.
"""

import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def get_all_samples(results_dir: Path, label_source: str = "hlca") -> list[str]:
    """Get all sample IDs from the results directory."""
    base = results_dir / label_source
    if not base.exists():
        return []

    # Get samples from any backend directory
    for backend_dir in base.iterdir():
        if backend_dir.is_dir():
            samples_dir = backend_dir / "samples"
            if samples_dir.exists():
                return [s.name for s in samples_dir.iterdir() if s.is_dir()]
    return []


def get_backends(results_dir: Path, label_source: str = "hlca") -> list[str]:
    """Get all available backends."""
    base = results_dir / label_source
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir() and (d / "samples").exists()]


def combine_figures_for_sample(
    results_dir: Path,
    sample_id: str,
    output_dir: Path,
    label_source: str = "hlca",
    backends: list[str] | None = None,
    figsize_per_backend: tuple[int, int] = (6, 5),
):
    """Combine backend comparison figures for a single sample."""
    base = results_dir / label_source

    if backends is None:
        backends = get_backends(results_dir, label_source)

    # Collect available images
    images = {}
    for backend in backends:
        img_path = base / backend / "samples" / sample_id / "backend_comparison.png"
        if img_path.exists():
            images[backend] = img_path

    if not images:
        print(f"  No images found for {sample_id}")
        return None

    n_backends = len(images)

    # Determine grid layout
    if n_backends <= 3:
        ncols = n_backends
        nrows = 1
    elif n_backends <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 4
        nrows = (n_backends + 3) // 4

    # Create figure
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_backend[0] * ncols, figsize_per_backend[1] * nrows),
        squeeze=False,
    )

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Plot each backend
    for idx, (backend, img_path) in enumerate(sorted(images.items())):
        ax = axes_flat[idx]
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(backend.upper(), fontsize=14, fontweight='bold')
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(images), len(axes_flat)):
        axes_flat[idx].axis('off')

    # Add overall title
    fig.suptitle(
        f"Spatial Deconvolution Comparison: {sample_id}",
        fontsize=16,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sample_id}_all_backends.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return output_path


def combine_all_samples(
    results_dir: Path,
    output_dir: Path,
    label_source: str = "hlca",
    samples: list[str] | None = None,
):
    """Combine figures for all samples."""
    if samples is None:
        samples = get_all_samples(results_dir, label_source)

    backends = get_backends(results_dir, label_source)
    print(f"Found {len(backends)} backends: {backends}")
    print(f"Processing {len(samples)} samples...")

    for sample_id in sorted(samples):
        print(f"Processing {sample_id}...")
        combine_figures_for_sample(
            results_dir, sample_id, output_dir, label_source, backends
        )


def create_summary_grid(
    results_dir: Path,
    output_dir: Path,
    label_source: str = "hlca",
    samples_per_page: int = 6,
):
    """Create a summary grid showing one cell type across all samples and backends."""
    # This would be more complex - showing spatial plots for a specific cell type
    # across all backends and samples. For now, just do per-sample combination.
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine backend comparison figures")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/spatial_benchmark"),
        help="Path to spatial benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spatial_benchmark/figures/combined"),
        help="Output directory for combined figures",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default="hlca",
        choices=["hlca", "luca"],
        help="Label source to use",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Process single sample (default: all samples)",
    )

    args = parser.parse_args()

    if args.sample:
        combine_figures_for_sample(
            args.results_dir,
            args.sample,
            args.output_dir,
            args.label_source,
        )
    else:
        combine_all_samples(
            args.results_dir,
            args.output_dir,
            args.label_source,
        )
