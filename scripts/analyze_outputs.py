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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    cell_metadata = None
    if args.cells_file and args.cells_file.exists():
        print(f"Loading cell metadata from {args.cells_file}")
        # Load key columns for visualization
        cols_to_load = [
            "cell_id", "cell_type", "x", "y", "stage", "stage_idx",
            # Clonality
            "clone_size", "clone_fraction", "clonal_entropy", "clonal_diversity",
            "n_clones", "is_major_clone", "clonal_pattern_idx",
            # Mutations
            "tmb", "kras_mut", "egfr_mut", "tp53_mut", "stk11_mut",
            # Scores
            "emt_score", "senescence_score", "sasp_score", "cytotrace",
            "S_score", "G2M_score", "proliferation_label",
            "caf_fraction", "immune_fraction", "diversity",
            # Pathways (ground truth from decoupleR)
            "pathway_EGFR", "pathway_Hypoxia", "pathway_TGFb", "pathway_JAK-STAT",
            "pathway_NFkB", "pathway_TNFa", "pathway_PI3K", "pathway_MAPK",
        ]
        cells_df = pd.read_parquet(args.cells_file, columns=cols_to_load)

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

        # Build full metadata dataframe for matched cells
        # cells_df is already indexed by cell_id from the loop above
        matched_ids = [cid for cid in pred_cell_ids if cid in cells_df.index]
        cell_metadata = cells_df.loc[matched_ids].reset_index()
        print(f"Cell metadata columns: {cell_metadata.columns.tolist()}")

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

    # Generate clonality and ground-truth comparison figures
    if cell_metadata is not None and len(cell_metadata) > 0:
        extra_dir = args.output_dir / "cell_properties"
        extra_dir.mkdir(parents=True, exist_ok=True)

        coords = umap_coords if umap_coords is not None else (
            viz.embeddings[:, :2] if viz.embeddings is not None else None
        )

        # Clonality plots
        clonal_cols = ["clonal_entropy", "clonal_diversity", "clone_fraction", "n_clones"]
        for col in clonal_cols:
            if col in cell_metadata.columns:
                values = cell_metadata[col].values
                if not np.isnan(values).all() and coords is not None:
                    fig = viz.plot_umap_colored(
                        values, col.replace("_", " ").title(), "viridis",
                        umap_coords=coords,
                        save_path=extra_dir / f"{col}.png",
                    )
                    if fig:
                        plt.close(fig)

        # Score plots (EMT, senescence, cytotrace, etc.)
        score_cols = ["emt_score", "senescence_score", "sasp_score", "cytotrace", "tmb"]
        for col in score_cols:
            if col in cell_metadata.columns:
                values = cell_metadata[col].values
                if not np.isnan(values).all() and coords is not None:
                    fig = viz.plot_umap_colored(
                        values, col.replace("_", " ").title(), "magma",
                        umap_coords=coords,
                        save_path=extra_dir / f"{col}.png",
                    )
                    if fig:
                        plt.close(fig)

        # Ground-truth pathway comparison (predicted vs decoupleR)
        if viz.pathway_scores is not None:
            gt_dir = args.output_dir / "pathway_comparison"
            gt_dir.mkdir(parents=True, exist_ok=True)

            # Debug: check what we have
            print(f"\nDEBUG pathway comparison:")
            print(f"  viz.pathway_scores columns: {viz.pathway_scores.columns.tolist()}")
            print(f"  cell_metadata pathway cols: {[c for c in cell_metadata.columns if 'pathway' in c.lower()]}")
            print(f"  viz.predictions cell_ids sample: {viz.predictions['cell_id'].values[:3]}")
            print(f"  cell_metadata cell_ids sample: {cell_metadata['cell_id'].values[:3]}")

            for pathway in ["EGFR", "Hypoxia", "TGFb", "JAK-STAT", "NFkB", "TNFa"]:
                gt_col = f"pathway_{pathway}"
                if pathway in viz.pathway_scores.columns and gt_col in cell_metadata.columns:
                    # Align predictions with ground truth by cell_id
                    # pathway_scores has same order as predictions
                    pred_cell_ids = viz.predictions["cell_id"].values
                    gt_lookup = cell_metadata.set_index("cell_id")[gt_col]

                    pred = []
                    gt = []
                    for i, cid in enumerate(pred_cell_ids):
                        if cid in gt_lookup.index:
                            pred.append(viz.pathway_scores[pathway].iloc[i])
                            gt.append(gt_lookup.loc[cid])
                    pred = np.array(pred)
                    gt = np.array(gt)
                    print(f"  {pathway}: matched {len(pred)} cells, pred range [{pred.min():.2f}, {pred.max():.2f}], gt range [{gt.min():.2f}, {gt.max():.2f}]")

                    # Scatter plot: predicted vs ground truth
                    valid = ~np.isnan(gt) & ~np.isnan(pred)
                    if valid.sum() > 100:
                        from scipy.stats import pearsonr, spearmanr
                        r_pearson, _ = pearsonr(pred[valid], gt[valid])
                        r_spearman, _ = spearmanr(pred[valid], gt[valid])

                        fig, ax = plt.subplots(figsize=(6, 6), dpi=config.dpi)
                        ax.scatter(gt[valid], pred[valid], s=1, alpha=0.3, rasterized=True)
                        ax.set_xlabel(f"Ground Truth ({gt_col})")
                        ax.set_ylabel(f"Predicted ({pathway})")
                        ax.set_title(f"{pathway}: r={r_pearson:.3f}, rho={r_spearman:.3f}")

                        # Add diagonal
                        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                                max(ax.get_xlim()[1], ax.get_ylim()[1])]
                        ax.plot(lims, lims, 'k--', alpha=0.5)

                        plt.savefig(gt_dir / f"{pathway}_comparison.png", dpi=config.dpi, bbox_inches="tight")
                        plt.close(fig)
                        print(f"  {pathway}: Pearson={r_pearson:.3f}, Spearman={r_spearman:.3f}")

        print(f"Saved cell property figures to {extra_dir}")

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
