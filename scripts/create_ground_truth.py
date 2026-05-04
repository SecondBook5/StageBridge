#!/usr/bin/env python
"""Create ground truth interaction labels for real LUAD data.

Applies AMICI-style ground truth labeling to:
1. snRNA-seq h5ad (with expression for ligand/receptor filtering)
2. Spatial h5ad (with coordinates)
3. cells.parquet (raw cell data)
4. neighborhoods.parquet (pre-computed neighborhoods)

The ground truth enables rigorous evaluation of StageBridge:
- Does attention correlate with sender proximity?
- Are interacting cells reconstructed differently?

Usage:
    python scripts/create_ground_truth.py --snrna /path/to/snrna.h5ad --output /path/to/output
    python scripts/create_ground_truth.py --spatial /path/to/spatial.h5ad --output /path/to/output
    python scripts/create_ground_truth.py --cells /path/to/cells.parquet --output /path/to/output
    python scripts/create_ground_truth.py --neighborhoods /path/to/neighborhoods.parquet --output /path/to/output
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.benchmarks.ground_truth_labeler import (
    GroundTruthLabeler,
    InteractionRule,
    DEFAULT_LUAD_RULES,
)


def label_snrna(
    input_path: Path,
    output_dir: Path,
    cell_type_key: str = "cell_type",
):
    """Label snRNA-seq data with ground truth (uses expression)."""
    import scanpy as sc

    print(f"\n{'='*60}")
    print(f"Processing snRNA-seq: {input_path}")
    print(f"{'='*60}")

    adata = sc.read_h5ad(input_path)
    print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Check for spatial coordinates
    spatial_key = None
    for key in ["spatial", "X_spatial", "X_umap"]:
        if key in adata.obsm:
            spatial_key = key
            break

    if spatial_key is None:
        print("WARNING: No spatial coordinates found, using UMAP as proxy")
        if "X_umap" not in adata.obsm:
            import scanpy as sc
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        spatial_key = "X_umap"

    labeler = GroundTruthLabeler(use_default_rules=True)
    labels = labeler.label_anndata(
        adata,
        cell_type_key=cell_type_key,
        spatial_key=spatial_key,
        ligand_threshold=0.5,
        receptor_threshold=0.5,
    )

    out_path = output_dir / "snrna"
    labels.save(out_path)

    # Also save cell type distribution
    ct_counts = pd.Series(adata.obs[cell_type_key]).value_counts()
    ct_counts.to_csv(out_path / "cell_type_counts.csv")

    return labels


def label_spatial(
    input_path: Path,
    output_dir: Path,
    cell_type_key: str = "cell_type",
    spatial_key: str = "spatial",
):
    """Label spatial data with ground truth."""
    import scanpy as sc

    print(f"\n{'='*60}")
    print(f"Processing spatial: {input_path}")
    print(f"{'='*60}")

    adata = sc.read_h5ad(input_path)
    print(f"Loaded: {adata.n_obs} cells/spots, {adata.n_vars} genes")

    if spatial_key not in adata.obsm:
        # Try to find spatial coordinates
        for key in ["spatial", "X_spatial"]:
            if key in adata.obsm:
                spatial_key = key
                break
        else:
            raise ValueError(f"No spatial coordinates found in obsm: {list(adata.obsm.keys())}")

    labeler = GroundTruthLabeler(use_default_rules=True)
    labels = labeler.label_anndata(
        adata,
        cell_type_key=cell_type_key,
        spatial_key=spatial_key,
    )

    out_path = output_dir / "spatial"
    labels.save(out_path)

    return labels


def label_cells_parquet(
    input_path: Path,
    output_dir: Path,
    cell_type_key: str = "cell_type",
    x_col: str = "x",
    y_col: str = "y",
):
    """Label cells.parquet with ground truth."""
    print(f"\n{'='*60}")
    print(f"Processing cells.parquet: {input_path}")
    print(f"{'='*60}")

    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df)} cells")
    print(f"Columns: {list(df.columns)[:20]}...")

    # Check required columns
    if x_col not in df.columns or y_col not in df.columns:
        # Try to find coordinate columns
        coord_candidates = ["x", "y", "X", "Y", "x_centroid", "y_centroid",
                           "centroid_x", "centroid_y", "spatial_x", "spatial_y"]
        found_x = found_y = None
        for col in df.columns:
            if col.lower() in ["x", "x_centroid", "centroid_x", "spatial_x"]:
                found_x = col
            if col.lower() in ["y", "y_centroid", "centroid_y", "spatial_y"]:
                found_y = col
        if found_x and found_y:
            x_col, y_col = found_x, found_y
            print(f"Using coordinates: {x_col}, {y_col}")
        else:
            raise ValueError(f"Cannot find coordinate columns. Available: {list(df.columns)}")

    # Ensure we have cell_id
    if "cell_id" not in df.columns:
        if df.index.name:
            df["cell_id"] = df.index.astype(str)
        else:
            df["cell_id"] = [f"cell_{i}" for i in range(len(df))]

    # Ensure we have cell types
    if cell_type_key not in df.columns:
        # Try common alternatives
        for alt in ["celltype", "cell_type_fine", "cell_type_coarse", "annotation", "cluster"]:
            if alt in df.columns:
                cell_type_key = alt
                break
        else:
            print(f"WARNING: No cell type column found, using 'unknown'")
            df[cell_type_key] = "unknown"

    # Rename columns for labeler
    df_for_labeler = df.copy()
    df_for_labeler["x"] = df[x_col]
    df_for_labeler["y"] = df[y_col]

    labeler = GroundTruthLabeler(use_default_rules=True)
    labels = labeler.label_neighborhoods(df_for_labeler, cell_type_key=cell_type_key)

    out_path = output_dir / "cells"
    labels.save(out_path)

    # Save cell type distribution
    ct_counts = df[cell_type_key].value_counts()
    ct_counts.to_csv(out_path / "cell_type_counts.csv")

    return labels


def label_neighborhoods_parquet(
    input_path: Path,
    output_dir: Path,
    cell_type_key: str = "cell_type",
):
    """Label neighborhoods.parquet with ground truth."""
    print(f"\n{'='*60}")
    print(f"Processing neighborhoods.parquet: {input_path}")
    print(f"{'='*60}")

    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df)} neighborhoods")

    # neighborhoods.parquet should have x, y, cell_id, cell_type
    required = ["x", "y"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if cell_type_key not in df.columns:
        for alt in ["celltype", "cell_type_fine", "annotation"]:
            if alt in df.columns:
                cell_type_key = alt
                break
        else:
            print(f"WARNING: No cell type column found")
            df[cell_type_key] = "unknown"

    labeler = GroundTruthLabeler(use_default_rules=True)
    labels = labeler.label_neighborhoods(df, cell_type_key=cell_type_key)

    out_path = output_dir / "neighborhoods"
    labels.save(out_path)

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Create ground truth interaction labels for LUAD data"
    )
    parser.add_argument("--snrna", type=str, help="snRNA-seq h5ad file")
    parser.add_argument("--spatial", type=str, help="Spatial h5ad file")
    parser.add_argument("--cells", type=str, help="cells.parquet file")
    parser.add_argument("--neighborhoods", type=str, help="neighborhoods.parquet file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--cell-type-key", type=str, default="cell_type",
                       help="Column name for cell types")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print rules we're using
    print("\nUsing interaction rules:")
    print("="*60)
    for rule in DEFAULT_LUAD_RULES:
        print(f"  {rule.interaction_name}")
        print(f"    Sender: {rule.sender_type} ({rule.ligand_gene}+)")
        print(f"    Receiver: {rule.receiver_type} ({rule.receptor_gene}+)")
        print(f"    Max distance: {rule.max_distance} microns")
    print("="*60)

    results = {}

    if args.snrna:
        labels = label_snrna(Path(args.snrna), output_dir, args.cell_type_key)
        results["snrna"] = {
            "n_cells": len(labels.cell_ids),
            "n_interacting": int(labels.is_interacting.sum()),
        }

    if args.spatial:
        labels = label_spatial(Path(args.spatial), output_dir, args.cell_type_key)
        results["spatial"] = {
            "n_cells": len(labels.cell_ids),
            "n_interacting": int(labels.is_interacting.sum()),
        }

    if args.cells:
        labels = label_cells_parquet(Path(args.cells), output_dir, args.cell_type_key)
        results["cells"] = {
            "n_cells": len(labels.cell_ids),
            "n_interacting": int(labels.is_interacting.sum()),
        }

    if args.neighborhoods:
        labels = label_neighborhoods_parquet(Path(args.neighborhoods), output_dir, args.cell_type_key)
        results["neighborhoods"] = {
            "n_cells": len(labels.cell_ids),
            "n_interacting": int(labels.is_interacting.sum()),
        }

    # Save combined summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, info in results.items():
        pct = 100 * info["n_interacting"] / info["n_cells"]
        print(f"{name}: {info['n_interacting']}/{info['n_cells']} interacting ({pct:.1f}%)")
    print("="*60)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
