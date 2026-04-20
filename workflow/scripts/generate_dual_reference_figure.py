#!/usr/bin/env python3
"""Generate dual-reference trajectory visualization.

Thin wrapper around stagebridge.viz.dual_reference for Snakemake integration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stagebridge.viz.dual_reference import (
    compute_reduced_embedding,
    plot_3d_trajectory,
    plot_reference_contribution,
    plot_flow_field,
    plot_dual_reference_combined,
)


def main():
    parser = argparse.ArgumentParser(description="Generate dual-reference trajectory figures")
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--fused_embedding", type=str, help="Path to fused embedding parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    cells_df = pd.read_parquet(args.cells)

    if args.fused_embedding:
        fused_path = Path(args.fused_embedding)
        if fused_path.exists():
            fused_df = pd.read_parquet(fused_path)
            cells_df = cells_df.merge(fused_df, on="cell_id", how="left")

    print(f"Loaded {len(cells_df)} cells")

    print("\nComputing UMAP embedding...")
    X_reduced, sample_df = compute_reduced_embedding(
        cells_df, method="umap", n_samples=args.n_samples
    )

    print("\nGenerating 3D trajectory plot...")
    plot_3d_trajectory(X_reduced, sample_df, output_dir / "fig_3d_trajectory.png")

    print("Generating reference contribution plot...")
    plot_reference_contribution(cells_df, output_dir / "fig_reference_contribution.png")

    print("Generating flow field plot...")
    plot_flow_field(X_reduced, sample_df, output_dir / "fig_flow_field.png")

    print("Generating combined embedding figure...")
    plot_dual_reference_combined(cells_df, output_dir / "fig_embedding_combined.png")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
