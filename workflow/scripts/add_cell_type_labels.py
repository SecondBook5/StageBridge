#!/usr/bin/env python3
"""
Add dual-reference cell type labels to snRNA data with cell cycle scores.

Reads HLCA and LuCA labels from separate parquet files and adds both to the
snRNA h5ad file. Also scores cell cycle on query cells as continuous features.

- cell_type: HLCA labels (primary, for spatial deconvolution)
- luca_cell_type: LuCA labels (for ablation experiments)
- S_score, G2M_score, phase: Cell cycle as continuous features (not cell types)

Cell cycle is scored as continuous features following Sun et al. 2026 (Nature):
- Cycling is a cell STATE, not an identity
- Cell cycle should be scored directly on spatial spots, not deconvolved as types
- Continuous scores allow flexible downstream analysis

Usage:
    python add_cell_type_labels.py \
        --snrna /path/to/snrna.h5ad \
        --hlca-labels /path/to/hlca_labels.parquet \
        --luca-labels /path/to/luca_labels.parquet \
        --output /path/to/snrna_with_labels.h5ad
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from stagebridge.spatial_mapping.lung_markers import S_PHASE_GENES, G2M_PHASE_GENES


def add_dual_labels(
    snrna_path: Path,
    hlca_labels_path: Path,
    luca_labels_path: Path,
    output_path: Path,
) -> None:
    """
    Add both HLCA and LuCA labels to snRNA h5ad.

    Args:
        snrna_path: Path to snRNA h5ad
        hlca_labels_path: Path to hlca_labels.parquet
        luca_labels_path: Path to luca_labels.parquet
        output_path: Output path for h5ad with both label columns
    """
    print(f"Loading snRNA data from {snrna_path}...")
    adata = ad.read_h5ad(snrna_path)
    print(f"  Shape: {adata.shape}")

    # Load HLCA labels
    print(f"\nLoading HLCA labels from {hlca_labels_path}...")
    hlca_df = pd.read_parquet(hlca_labels_path)
    print(f"  Shape: {hlca_df.shape}")
    print(f"  Columns: {hlca_df.columns.tolist()}")

    # Find the label column in HLCA
    hlca_label_col = None
    for col in ["cell_type", "hlca_label", "predicted_label"]:
        if col in hlca_df.columns:
            hlca_label_col = col
            break
    if hlca_label_col is None:
        raise ValueError(f"No label column found in HLCA parquet. Columns: {hlca_df.columns.tolist()}")

    # Find the cell_id column
    hlca_id_col = "cell_id" if "cell_id" in hlca_df.columns else hlca_df.index.name
    if hlca_id_col is None or hlca_id_col not in hlca_df.columns:
        # Use index as cell_id
        hlca_df = hlca_df.reset_index()
        hlca_id_col = hlca_df.columns[0]

    hlca_map = dict(zip(hlca_df[hlca_id_col], hlca_df[hlca_label_col]))
    adata.obs["cell_type"] = adata.obs.index.map(hlca_map)
    unmapped = adata.obs["cell_type"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} cells without HLCA label")
        adata.obs["cell_type"] = adata.obs["cell_type"].fillna("Unknown")
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    print(f"  Added cell_type (HLCA): {adata.obs['cell_type'].nunique()} types")

    # Load LuCA labels
    print(f"\nLoading LuCA labels from {luca_labels_path}...")
    luca_df = pd.read_parquet(luca_labels_path)
    print(f"  Shape: {luca_df.shape}")
    print(f"  Columns: {luca_df.columns.tolist()}")

    # Find the label column in LuCA
    luca_label_col = None
    for col in ["cell_type", "luca_label", "predicted_label"]:
        if col in luca_df.columns:
            luca_label_col = col
            break
    if luca_label_col is None:
        raise ValueError(f"No label column found in LuCA parquet. Columns: {luca_df.columns.tolist()}")

    # Find the cell_id column
    luca_id_col = "cell_id" if "cell_id" in luca_df.columns else luca_df.index.name
    if luca_id_col is None or luca_id_col not in luca_df.columns:
        luca_df = luca_df.reset_index()
        luca_id_col = luca_df.columns[0]

    luca_map = dict(zip(luca_df[luca_id_col], luca_df[luca_label_col]))
    adata.obs["luca_cell_type"] = adata.obs.index.map(luca_map)
    unmapped = adata.obs["luca_cell_type"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} cells without LuCA label")
        adata.obs["luca_cell_type"] = adata.obs["luca_cell_type"].fillna("Unknown")
    adata.obs["luca_cell_type"] = adata.obs["luca_cell_type"].astype("category")
    print(f"  Added luca_cell_type: {adata.obs['luca_cell_type'].nunique()} types")

    # ==========================================================================
    # Score cell cycle as continuous features (Sun et al. 2026 approach)
    # ==========================================================================
    print("\n=== Scoring Cell Cycle (Continuous Features) ===")

    # Filter to genes present in data
    s_genes_present = [g for g in S_PHASE_GENES if g in adata.var_names]
    g2m_genes_present = [g for g in G2M_PHASE_GENES if g in adata.var_names]
    print(f"  S phase genes: {len(s_genes_present)}/{len(S_PHASE_GENES)} present")
    print(f"  G2M phase genes: {len(g2m_genes_present)}/{len(G2M_PHASE_GENES)} present")

    if len(s_genes_present) < 5 or len(g2m_genes_present) < 5:
        print("  WARNING: Too few cell cycle genes - skipping cell cycle scoring")
    else:
        # Score cell cycle (adds S_score, G2M_score, phase columns)
        sc.tl.score_genes_cell_cycle(
            adata,
            s_genes=s_genes_present,
            g2m_genes=g2m_genes_present,
        )

        # Report cell cycle distribution
        print("  Cell cycle phase distribution:")
        for phase, count in adata.obs['phase'].value_counts().items():
            pct = 100 * count / len(adata)
            print(f"    {phase}: {count:,} ({pct:.1f}%)")

        # Report score statistics
        print(f"  S_score: mean={adata.obs['S_score'].mean():.3f}, max={adata.obs['S_score'].max():.3f}")
        print(f"  G2M_score: mean={adata.obs['G2M_score'].mean():.3f}, max={adata.obs['G2M_score'].max():.3f}")

    # NOTE: Cell cycle is NOT added to cell type names.
    # Following Sun et al. 2026: cycling is a state, not an identity.
    # Cell cycle should be scored directly on spatial spots for analysis.

    # Summary
    print("\n=== Label Summary ===")
    print(f"HLCA (cell_type): {adata.obs['cell_type'].nunique()} types")
    print("Top 10:")
    for label, count in adata.obs["cell_type"].value_counts().head(10).items():
        pct = 100 * count / len(adata)
        print(f"  {label}: {count:,} ({pct:.1f}%)")

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
    parser.add_argument("--hlca-labels", required=True, help="Path to hlca_labels.parquet")
    parser.add_argument("--luca-labels", required=True, help="Path to luca_labels.parquet")
    parser.add_argument("--output", required=True, help="Output h5ad path")
    args = parser.parse_args()

    add_dual_labels(
        snrna_path=Path(args.snrna),
        hlca_labels_path=Path(args.hlca_labels),
        luca_labels_path=Path(args.luca_labels),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
