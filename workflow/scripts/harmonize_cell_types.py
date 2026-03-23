#!/usr/bin/env python3
"""
Add dual-reference cell type labels to snRNA data.

This script adds both HLCA and LuCA cell type labels to the snRNA h5ad file,
allowing the spatial benchmark to use either source via --label-source.

The primary cell_type column uses HLCA labels (comprehensive, well-validated).
LuCA labels are stored in luca_cell_type for ablation experiments.

Usage:
    python harmonize_cell_types.py \
        --snrna /path/to/snrna.h5ad \
        --labels /path/to/cell_types.parquet \
        --output /path/to/snrna_with_labels.h5ad
"""

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def add_dual_labels(
    snrna_path: Path,
    labels_path: Path,
    output_path: Path,
) -> None:
    """
    Add both HLCA and LuCA labels to snRNA h5ad.

    Args:
        snrna_path: Path to snRNA h5ad
        labels_path: Path to cell_types.parquet with cell_type and luca_cell_type
        output_path: Output path for h5ad with both label columns
    """
    print(f"Loading snRNA data from {snrna_path}...")
    adata = ad.read_h5ad(snrna_path)
    print(f"  Shape: {adata.shape}")

    print(f"\nLoading labels from {labels_path}...")
    labels_df = pd.read_parquet(labels_path)
    print(f"  Shape: {labels_df.shape}")
    print(f"  Columns: {labels_df.columns.tolist()}")

    # Create mapping from cell_id to labels
    if "cell_id" not in labels_df.columns:
        raise ValueError("Labels parquet must have 'cell_id' column")

    # Add HLCA labels (primary cell_type column)
    if "cell_type" in labels_df.columns:
        hlca_map = dict(zip(labels_df["cell_id"], labels_df["cell_type"]))
        adata.obs["cell_type"] = adata.obs.index.map(hlca_map)
        unmapped = adata.obs["cell_type"].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} cells without HLCA label")
            adata.obs["cell_type"] = adata.obs["cell_type"].fillna("Unknown")
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        print(f"  Added cell_type (HLCA): {adata.obs['cell_type'].nunique()} types")
    else:
        print("  WARNING: No 'cell_type' column in labels, skipping HLCA")

    # Add LuCA labels (for ablation)
    if "luca_cell_type" in labels_df.columns:
        luca_map = dict(zip(labels_df["cell_id"], labels_df["luca_cell_type"]))
        adata.obs["luca_cell_type"] = adata.obs.index.map(luca_map)
        unmapped = adata.obs["luca_cell_type"].isna().sum()
        if unmapped > 0:
            print(f"  WARNING: {unmapped} cells without LuCA label")
            adata.obs["luca_cell_type"] = adata.obs["luca_cell_type"].fillna("Unknown")
        adata.obs["luca_cell_type"] = adata.obs["luca_cell_type"].astype("category")
        print(f"  Added luca_cell_type: {adata.obs['luca_cell_type'].nunique()} types")
    else:
        print("  WARNING: No 'luca_cell_type' column in labels, skipping LuCA")

    # Summary
    print("\n=== Label Summary ===")
    if "cell_type" in adata.obs.columns:
        print(f"HLCA (cell_type): {adata.obs['cell_type'].nunique()} types")
        print("Top 10:")
        for label, count in adata.obs["cell_type"].value_counts().head(10).items():
            pct = 100 * count / len(adata)
            print(f"  {label}: {count:,} ({pct:.1f}%)")

    if "luca_cell_type" in adata.obs.columns:
        print(f"\nLuCA (luca_cell_type): {adata.obs['luca_cell_type'].nunique()} types")
        print("Top 10:")
        for label, count in adata.obs["luca_cell_type"].value_counts().head(10).items():
            pct = 100 * count / len(adata)
            print(f"  {label}: {count:,} ({pct:.1f}%)")

    # Save
    print(f"\nSaving to {output_path}...")
    adata.write_h5ad(output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Add dual-reference cell type labels to snRNA")
    parser.add_argument("--snrna", required=True, help="Path to snRNA h5ad")
    parser.add_argument("--labels", required=True, help="Path to cell_types.parquet")
    parser.add_argument("--output", required=True, help="Output h5ad path")
    args = parser.parse_args()

    add_dual_labels(
        snrna_path=Path(args.snrna),
        labels_path=Path(args.labels),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
