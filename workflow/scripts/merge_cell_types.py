#!/usr/bin/env python3
"""Merge HLCA cell type labels into snRNA h5ad file.

This creates a clean dependency for Snakemake - the output file explicitly
contains cell types, rather than modifying the input file in place.
"""

import argparse
from pathlib import Path

import anndata
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge HLCA cell types into snRNA")
    parser.add_argument("--snrna", type=str, required=True, help="Input snRNA h5ad")
    parser.add_argument("--labels", type=str, required=True, help="HLCA labels parquet")
    parser.add_argument("--output", type=str, required=True, help="Output snRNA h5ad with cell types")
    args = parser.parse_args()

    print(f"Loading snRNA: {args.snrna}")
    adata = anndata.read_h5ad(args.snrna)
    print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    print(f"Loading HLCA labels: {args.labels}")
    labels_df = pd.read_parquet(args.labels)
    print(f"  Labels: {len(labels_df):,} cells")

    # Merge cell types
    if "hlca_label" in labels_df.columns:
        label_col = "hlca_label"
    elif "cell_type" in labels_df.columns:
        label_col = "cell_type"
    else:
        raise ValueError(f"No cell type column found. Columns: {labels_df.columns.tolist()}")

    # Create mapping from cell_id to cell_type
    labels_df.index = labels_df.index.astype(str)
    cell_type_map = labels_df[label_col].to_dict()

    # Map to adata
    adata.obs["cell_type"] = adata.obs.index.map(cell_type_map)
    n_mapped = adata.obs["cell_type"].notna().sum()
    n_unique = adata.obs["cell_type"].nunique()

    print(f"  Mapped: {n_mapped:,}/{adata.n_obs:,} cells ({100*n_mapped/adata.n_obs:.1f}%)")
    print(f"  Unique cell types: {n_unique}")

    # Fill any unmapped cells
    unmapped = adata.obs["cell_type"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped:,} cells without cell type, setting to 'Unknown'")
        adata.obs["cell_type"] = adata.obs["cell_type"].fillna("Unknown")

    # Also add probability columns if available
    for col in ["hlca_max_prob", "hlca_entropy", "hlca_uncertain"]:
        if col in labels_df.columns:
            col_map = labels_df[col].to_dict()
            adata.obs[col] = adata.obs.index.map(col_map)

    print(f"Saving: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
