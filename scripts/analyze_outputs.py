#!/usr/bin/env python3
"""Generate analysis figures from inference outputs.

Usage:
    python scripts/analyze_outputs.py \
        --inference-dir /path/to/inference/output \
        --output-dir /path/to/figures \
        --umap-file /path/to/umap_coords.npy  # optional
        --spatial-file /path/to/spatial_coords.npy  # optional
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.analysis import StageBridgeVisualizer, FlowFieldAnalyzer
from stagebridge.analysis.visualize import PlotConfig


def main():
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--inference-dir", type=Path, required=True,
                        help="Directory containing inference outputs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to save figures")
    parser.add_argument("--umap-file", type=Path, default=None,
                        help="Path to UMAP coordinates (npy or parquet)")
    parser.add_argument("--spatial-file", type=Path, default=None,
                        help="Path to spatial coordinates (npy or parquet)")
    parser.add_argument("--cells-file", type=Path, default=None,
                        help="Path to cells.parquet with cell_id, cell_type, x, y")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Data directory to load stage/spatial info")
    parser.add_argument("--fold-idx", type=int, default=0,
                        help="Fold index for loading test set metadata")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Loading inference outputs from {args.inference_dir}")

    config = PlotConfig(dpi=args.dpi, point_size=args.point_size)
    viz = StageBridgeVisualizer(args.inference_dir, plot_config=config)

    # Load UMAP coordinates if provided
    umap_coords = None
    if args.umap_file:
        if args.umap_file.suffix == ".npy":
            umap_coords = np.load(args.umap_file)
        elif args.umap_file.suffix == ".parquet":
            umap_coords = pd.read_parquet(args.umap_file).values
        print(f"Loaded UMAP coords: {umap_coords.shape}")

    # Load spatial coordinates if provided
    spatial_coords = None
    if args.spatial_file:
        if args.spatial_file.suffix == ".npy":
            spatial_coords = np.load(args.spatial_file)
        elif args.spatial_file.suffix == ".parquet":
            spatial_coords = pd.read_parquet(args.spatial_file).values
        print(f"Loaded spatial coords: {spatial_coords.shape}")

    # Load cell metadata from cells.parquet if provided
    cell_types = None
    if args.cells_file and args.cells_file.exists():
        print(f"Loading cell metadata from {args.cells_file}")
        cells_df = pd.read_parquet(args.cells_file, columns=["cell_id", "cell_type", "x", "y"])

        # Merge with predictions on cell_id
        pred_cell_ids = viz.predictions["cell_id"].values
        cells_df = cells_df.set_index("cell_id")

        # Get cell types for prediction cells
        cell_types = []
        spatial_x = []
        spatial_y = []
        for cid in pred_cell_ids:
            if cid in cells_df.index:
                row = cells_df.loc[cid]
                cell_types.append(row["cell_type"])
                spatial_x.append(row["x"])
                spatial_y.append(row["y"])
            else:
                cell_types.append("")
                spatial_x.append(np.nan)
                spatial_y.append(np.nan)

        cell_types = np.array(cell_types)
        print(f"Matched {(cell_types != '').sum()} / {len(cell_types)} cells with cell types")
        print(f"Cell types: {pd.Series(cell_types).value_counts().head(10).to_dict()}")

        # Use spatial coords from cells.parquet if not provided separately
        if spatial_coords is None:
            spatial_x = np.array(spatial_x)
            spatial_y = np.array(spatial_y)
            valid_spatial = ~np.isnan(spatial_x)
            if valid_spatial.sum() > 0:
                spatial_coords = np.column_stack([spatial_x, spatial_y])
                print(f"Loaded spatial coords: {valid_spatial.sum()} / {len(spatial_x)} cells have coordinates")

    # Try to load from data directory if provided
    if args.data_dir and spatial_coords is None:
        # Check for spatial coordinates in test split
        test_meta = args.data_dir / f"test_fold{args.fold_idx}_metadata.parquet"
        if test_meta.exists():
            meta_df = pd.read_parquet(test_meta)
            if "x" in meta_df.columns and "y" in meta_df.columns:
                spatial_coords = meta_df[["x", "y"]].values
                print(f"Loaded spatial coords from metadata: {spatial_coords.shape}")

    # Get stages from predictions
    stages = None
    if "stage_idx" in viz.predictions.columns:
        stages = viz.predictions["stage_idx"].values
        print(f"Loaded stages: {len(stages)} cells, unique: {sorted(set(stages))}")

    # Generate all figures
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = viz.generate_all_figures(
        umap_coords=umap_coords,
        spatial_coords=spatial_coords,
        stages=stages,
        save_dir=args.output_dir,
    )

    print(f"\nSaved {len(figures)} figures to {args.output_dir}")

    # Generate attention figures if we have cell types
    if cell_types is not None and viz.attention_weights is not None:
        attn_dir = args.output_dir / "attention"
        attn_dir.mkdir(parents=True, exist_ok=True)

        # Attention by cell type bar chart
        # Note: This shows mean attention per cell, not per neighbor cell type
        # (we'd need neighbor cell types for that, which requires neighborhoods data)
        figures["attention_umap"] = viz.plot_attention_on_umap(
            receiver_celltypes=cell_types,
            umap_coords=umap_coords,
            save_dir=attn_dir,
        )

        if spatial_coords is not None:
            figures["attention_spatial"] = viz.plot_attention_on_spatial(
                spatial_coords=spatial_coords,
                receiver_celltypes=cell_types,
                save_dir=attn_dir,
            )

        print(f"Saved attention figures to {attn_dir}")

    # Summary statistics
    print("\n=== Summary Statistics ===")

    if viz.displacements is not None:
        flow = viz.compute_flow_metrics()
        print(f"Drift magnitude: mean={flow.drift_magnitude.mean():.4f}, std={flow.drift_magnitude.std():.4f}")
        print(f"Divergence: mean={flow.divergence.mean():.4f}, std={flow.divergence.std():.4f}")
        print(f"Curl magnitude: mean={flow.curl_magnitude.mean():.4f}, std={flow.curl_magnitude.std():.4f}")
        print(f"Irreversibility: mean={flow.irreversibility.mean():.4f}, std={flow.irreversibility.std():.4f}")

    if viz.pathway_scores is not None:
        print("\nPathway scores (mean +/- std):")
        for col in viz.pathway_scores.columns[:6]:
            vals = viz.pathway_scores[col].values
            print(f"  {col}: {vals.mean():.3f} +/- {vals.std():.3f}")

    if viz.proliferation_scores is not None:
        prolif = viz.proliferation_scores["proliferation_score"].values
        print(f"\nProliferation: mean={prolif.mean():.3f}, std={prolif.std():.3f}")


if __name__ == "__main__":
    main()
