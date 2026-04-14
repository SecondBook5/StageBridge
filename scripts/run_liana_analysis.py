#!/usr/bin/env python3
"""
Run LIANA L-R analysis on snRNA data.

Part of biological validation for H1.2 hypothesis:
  "IL1B-high proinflammatory niches increase transition probability for alveolar progenitors"

This script validates that IL1B-IL1R1 is a top L-R pair in the dataset.

Usage:
  # Pipeline (via Snakemake)
  python scripts/run_liana_analysis.py \
      --snrna /path/to/snrna_with_celltypes.h5ad \
      --output-dir /path/to/output

  # Local testing
  python scripts/run_liana_analysis.py \
      --snrna $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
      --output-dir results/liana
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LIANA L-R analysis for biological validation"
    )
    parser.add_argument(
        "--snrna",
        type=Path,
        required=True,
        help="Path to snRNA h5ad with cell_type column"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LIANA results"
    )
    parser.add_argument(
        "--cell-type-col",
        type=str,
        default="cell_type",
        help="Column name for cell type labels (default: cell_type)"
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=100,
        help="Number of permutations for significance testing (default: 100)"
    )
    parser.add_argument(
        "--expr-prop",
        type=float,
        default=0.1,
        help="Minimum proportion of cells expressing L/R (default: 0.1)"
    )
    parser.add_argument(
        "--resource",
        type=str,
        default="consensus",
        help="L-R resource database (default: consensus)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import liana here to avoid slow import on --help
    import liana as li

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIANA L-R ANALYSIS")
    print("=" * 70)
    print(f"  Input: {args.snrna}")
    print(f"  Output: {args.output_dir}")
    print(f"  Cell type column: {args.cell_type_col}")
    print(f"  Permutations: {args.n_perms}")

    # Load snRNA data
    print("\n[1/4] Loading snRNA expression data...")
    adata = sc.read_h5ad(args.snrna)
    print(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Check cell type column exists
    if args.cell_type_col not in adata.obs.columns:
        print(f"  ERROR: Column '{args.cell_type_col}' not found in adata.obs")
        print(f"  Available columns: {list(adata.obs.columns)}")
        return 1

    # Filter to cells with cell_type
    n_with_ct = adata.obs[args.cell_type_col].notna().sum()
    print(f"  Cells with {args.cell_type_col}: {n_with_ct:,} / {adata.n_obs:,}")

    if n_with_ct < adata.n_obs:
        adata = adata[adata.obs[args.cell_type_col].notna()].copy()
        print(f"  Filtered to {adata.n_obs:,} cells")

    print(f"\n  Cell type distribution:")
    for ct, count in adata.obs[args.cell_type_col].value_counts().items():
        print(f"    {ct}: {count:,}")

    # Preprocessing for LIANA
    print("\n[2/4] Preprocessing for LIANA...")

    # Check if already normalized (log1p values typically < 10)
    max_val = adata.X.max() if hasattr(adata.X, 'max') else adata.X.toarray().max()
    if max_val > 20:
        print("  Normalizing (data appears to be raw counts)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("  Data appears already normalized, skipping normalization")

    print(f"  Expression range: [{adata.X.min():.2f}, {adata.X.max():.2f}]")

    # Run LIANA
    print("\n[3/4] Running LIANA rank_aggregate...")
    print("  Methods: CellPhoneDB, CellChat, NATMI, Connectome, SingleCellSignalR")
    print(f"  This may take 10-30 minutes for {adata.n_obs:,} cells...")

    li.mt.rank_aggregate(
        adata,
        groupby=args.cell_type_col,
        resource_name=args.resource,
        expr_prop=args.expr_prop,
        use_raw=False,
        verbose=True,
        n_perms=args.n_perms,
    )

    # Get results
    lr_results = adata.uns['liana_res'].copy()
    print(f"\n  Found {len(lr_results):,} L-R interactions")

    # Save full results
    print("\n[4/4] Saving results...")
    lr_results.to_parquet(args.output_dir / "lr_interactions_full.parquet")
    print(f"  Saved: lr_interactions_full.parquet")

    # Extract top interactions
    top_interactions = lr_results.nsmallest(100, 'magnitude_rank')
    top_interactions.to_parquet(args.output_dir / "lr_interactions_top100.parquet")
    print(f"  Saved: lr_interactions_top100.parquet")

    # IL1B-IL1R1 specific (KEY for H1.2 validation)
    il1b_mask = (
        lr_results['ligand_complex'].str.contains('IL1B', case=False, na=False) |
        lr_results['receptor_complex'].str.contains('IL1R1', case=False, na=False)
    )
    il1b_results = lr_results[il1b_mask]

    if len(il1b_results) > 0:
        il1b_results.to_parquet(args.output_dir / "lr_il1b_il1r1.parquet")
        print(f"  Saved: lr_il1b_il1r1.parquet")
        print(f"\n  IL1B/IL1R1 interactions: {len(il1b_results)}")

        # Show top IL1B interactions
        il1b_top = il1b_results.nsmallest(10, 'magnitude_rank')
        print("\n  Top IL1B/IL1R1 interactions:")
        for _, row in il1b_top.iterrows():
            print(f"    {row['source']} -> {row['target']}: "
                  f"{row['ligand_complex']} - {row['receptor_complex']} "
                  f"(rank: {row['magnitude_rank']:.1f})")
    else:
        print("\n  WARNING: No IL1B/IL1R1 interactions found")
        print("  This may indicate gene naming issues or low expression")

    # Summary by cell type pair
    print("\n" + "=" * 70)
    print("TOP CELL TYPE PAIRS BY INTERACTION COUNT")
    print("=" * 70)
    pair_counts = lr_results.groupby(['source', 'target']).size().sort_values(ascending=False)
    print(pair_counts.head(15))

    # Check if IL1B is in top interactions overall
    il1b_in_top100 = top_interactions[
        top_interactions['ligand_complex'].str.contains('IL1B', case=False, na=False) |
        top_interactions['receptor_complex'].str.contains('IL1R1', case=False, na=False)
    ]

    # Save summary JSON
    summary = {
        'total_interactions': len(lr_results),
        'cell_types': list(adata.obs[args.cell_type_col].unique()),
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'il1b_interactions': len(il1b_results),
        'il1b_in_top100': len(il1b_in_top100),
        'il1b_validated': len(il1b_results) > 0 and len(il1b_in_top100) > 0,
        'parameters': {
            'n_perms': args.n_perms,
            'expr_prop': args.expr_prop,
            'resource': args.resource,
        }
    }

    with open(args.output_dir / "liana_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: liana_summary.json")

    # Validation status for H1.2
    print("\n" + "=" * 70)
    print("H1.2 VALIDATION STATUS")
    print("=" * 70)
    if summary['il1b_validated']:
        print("  [PASS] IL1B/IL1R1 interactions detected and ranked in top 100")
        print(f"         {len(il1b_results)} total IL1B/IL1R1 interactions")
        print(f"         {len(il1b_in_top100)} in top 100 by magnitude_rank")
    else:
        print("  [WARN] IL1B/IL1R1 validation inconclusive")
        if len(il1b_results) == 0:
            print("         No IL1B/IL1R1 interactions detected")
        else:
            print(f"         {len(il1b_results)} interactions found but none in top 100")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
