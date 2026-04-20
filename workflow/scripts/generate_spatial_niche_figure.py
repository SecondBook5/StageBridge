#!/usr/bin/env python3
"""Generate spatial niche visualization figure.

Thin wrapper around stagebridge.viz.spatial_niche for Snakemake integration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stagebridge.viz.spatial_niche import (
    compute_niche_scores,
    get_spatial_coords,
    plot_spatial_risk_map,
    plot_niche_composition_map,
    plot_spatial_niche_combined,
)


def main():
    parser = argparse.ArgumentParser(description="Generate spatial niche figures")
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--neighborhoods", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_dir", type=str, help="Directory with model inference")
    parser.add_argument("--sample_id", type=str, help="Specific sample to visualize")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cells_df = pd.read_parquet(args.cells)
    neighborhoods_df = pd.read_parquet(args.neighborhoods)

    if args.inference_dir:
        inf_path = Path(args.inference_dir) / "model_inference.parquet"
        if inf_path.exists():
            inf_df = pd.read_parquet(inf_path)
            cells_df = cells_df.merge(inf_df, on="cell_id", how="left")
            print(f"Loaded inference for {cells_df['transition_prob'].notna().sum()} cells")

    print(f"Loaded {len(cells_df)} cells")

    print("\nComputing niche scores...")
    niche_scores = compute_niche_scores(cells_df, neighborhoods_df)

    print("\nGenerating spatial risk map...")
    plot_spatial_risk_map(
        cells_df,
        output_dir / "fig_spatial_risk.png",
        sample_id=args.sample_id,
    )

    print("Generating niche composition map...")
    plot_niche_composition_map(
        cells_df,
        niche_scores,
        output_dir / "fig_spatial_niche_composition.png",
    )

    print("Generating combined spatial figure...")
    plot_spatial_niche_combined(
        cells_df,
        niche_scores,
        output_dir / "fig_spatial_combined.png",
    )

    niche_scores.to_parquet(output_dir / "niche_scores.parquet", index=False)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
