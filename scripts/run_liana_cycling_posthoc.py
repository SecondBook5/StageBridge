#!/usr/bin/env python3
"""Post-hoc analysis of LIANA results stratified by cell cycling status.

After LIANA runs on coarse cell types, this script stratifies results
by cell cycle phase to test whether cycling cells have different
L-R interaction profiles.

Key questions:
1. Do cycling cells (S/G2M) have different IL1B-IL1R1 scores?
2. Are niche interactions different for proliferating vs quiescent cells?

Usage:
    python scripts/run_liana_cycling_posthoc.py \
        --liana-results /path/to/liana/lr_interactions_full.parquet \
        --cells /path/to/cells.parquet \
        --output-dir /path/to/output
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="LIANA cycling post-hoc analysis")
    parser.add_argument("--liana-results", type=Path, required=True,
                        help="Path to LIANA lr_interactions_full.parquet")
    parser.add_argument("--cells", type=Path, required=True,
                        help="Path to cells.parquet with phase/proliferation info")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIANA CELL CYCLING POST-HOC ANALYSIS")
    print("=" * 70)

    # Load LIANA results
    print("\n[1/4] Loading LIANA results...")
    lr_df = pd.read_parquet(args.liana_results)
    print(f"  Loaded {len(lr_df):,} L-R interactions")

    # Load cell metadata
    print("\n[2/4] Loading cell metadata...")
    cells_df = pd.read_parquet(args.cells, columns=[
        'cell_id', 'cell_type', 'phase', 'S_score', 'G2M_score',
        'proliferation_label'
    ])
    print(f"  Loaded {len(cells_df):,} cells")

    # Check for cycling columns
    if 'phase' not in cells_df.columns:
        print("  ERROR: 'phase' column not found in cells.parquet")
        return 1

    # Cell cycle distribution
    print("\n  Cell cycle distribution:")
    phase_counts = cells_df['phase'].value_counts()
    for phase, count in phase_counts.items():
        print(f"    {phase}: {count:,} ({100*count/len(cells_df):.1f}%)")

    # Define cycling vs quiescent
    # Note: 'spatial' phase cells don't have cycle info (from spatial data)
    snrna_mask = cells_df['phase'] != 'spatial'
    cycling_mask = cells_df['phase'].isin(['S', 'G2M'])
    quiescent_mask = cells_df['phase'] == 'G1'

    n_cycling = cycling_mask.sum()
    n_quiescent = quiescent_mask.sum()
    print(f"\n  Cycling (S/G2M): {n_cycling:,}")
    print(f"  Quiescent (G1): {n_quiescent:,}")
    print(f"  Spatial (no phase): {(~snrna_mask).sum():,}")

    # Create cell type x cycling stratification
    print("\n[3/4] Stratifying by cell type and cycling status...")

    # Add cycling status to cells
    cells_df['cycling_status'] = 'unknown'
    cells_df.loc[cycling_mask, 'cycling_status'] = 'cycling'
    cells_df.loc[quiescent_mask, 'cycling_status'] = 'quiescent'

    # Get cell type distribution by cycling status
    celltype_cycling = cells_df[snrna_mask].groupby(
        ['cell_type', 'cycling_status']
    ).size().unstack(fill_value=0)
    print("\n  Cell counts by type and cycling status:")
    print(celltype_cycling)

    # Analyze IL1B-IL1R1 interactions
    print("\n[4/4] Analyzing IL1B-IL1R1 by cycling status...")

    # Filter to IL1B/IL1R1 interactions
    il1b_mask = (
        lr_df['ligand_complex'].str.contains('IL1B', case=False, na=False) |
        lr_df['receptor_complex'].str.contains('IL1R1', case=False, na=False)
    )
    il1b_df = lr_df[il1b_mask].copy()
    print(f"  IL1B/IL1R1 interactions: {len(il1b_df):,}")

    # Get cycling fraction per cell type
    cycling_fraction = {}
    for ct in cells_df['cell_type'].unique():
        ct_cells = cells_df[(cells_df['cell_type'] == ct) & snrna_mask]
        if len(ct_cells) > 0:
            cycling_fraction[ct] = (ct_cells['cycling_status'] == 'cycling').mean()

    # Annotate IL1B interactions with cycling info
    # LIANA results have 'source' and 'target' which are cell types
    il1b_df['source_cycling_frac'] = il1b_df['source'].map(cycling_fraction)
    il1b_df['target_cycling_frac'] = il1b_df['target'].map(cycling_fraction)

    # Compute correlation between cycling fraction and interaction strength
    # Higher magnitude_rank = weaker interaction, so we use 1/rank
    il1b_df['interaction_strength'] = 1 / (il1b_df['magnitude_rank'] + 1)

    # Source cell type analysis
    source_cycling_corr = il1b_df[['source_cycling_frac', 'interaction_strength']].dropna()
    if len(source_cycling_corr) > 5:
        corr_source = source_cycling_corr['source_cycling_frac'].corr(
            source_cycling_corr['interaction_strength']
        )
    else:
        corr_source = np.nan

    # Target cell type analysis
    target_cycling_corr = il1b_df[['target_cycling_frac', 'interaction_strength']].dropna()
    if len(target_cycling_corr) > 5:
        corr_target = target_cycling_corr['target_cycling_frac'].corr(
            target_cycling_corr['interaction_strength']
        )
    else:
        corr_target = np.nan

    # Top IL1B interactions by cycling status
    print("\n  Top IL1B interactions (by sender cycling fraction):")
    top_by_cycling = il1b_df.nsmallest(10, 'magnitude_rank')[[
        'source', 'target', 'ligand_complex', 'receptor_complex',
        'magnitude_rank', 'source_cycling_frac'
    ]]
    for _, row in top_by_cycling.iterrows():
        cycling_pct = row['source_cycling_frac'] * 100 if pd.notna(row['source_cycling_frac']) else 0
        print(f"    {row['source']} ({cycling_pct:.0f}% cycling) -> {row['target']}: "
              f"{row['ligand_complex']}-{row['receptor_complex']}")

    # Summary statistics
    summary = {
        'n_cells': len(cells_df),
        'n_cycling': int(n_cycling),
        'n_quiescent': int(n_quiescent),
        'cycling_fraction_overall': float(n_cycling / (n_cycling + n_quiescent)) if (n_cycling + n_quiescent) > 0 else 0,
        'n_il1b_interactions': len(il1b_df),
        'cycling_il1b_correlation_source': float(corr_source) if pd.notna(corr_source) else None,
        'cycling_il1b_correlation_target': float(corr_target) if pd.notna(corr_target) else None,
        'celltype_cycling_fractions': {k: float(v) for k, v in cycling_fraction.items()},
    }

    # Determine if cycling affects IL1B signaling
    cycling_effect = "inconclusive"
    if pd.notna(corr_source) and abs(corr_source) > 0.2:
        cycling_effect = "positive" if corr_source > 0 else "negative"
        summary['cycling_effect_on_il1b'] = cycling_effect
        summary['cycling_effect_strength'] = abs(float(corr_source))
    else:
        summary['cycling_effect_on_il1b'] = "minimal"
        summary['cycling_effect_strength'] = 0.0

    # Save results
    with open(args.output_dir / "cycling_posthoc_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    il1b_df.to_parquet(args.output_dir / "il1b_with_cycling.parquet", index=False)

    celltype_cycling.to_csv(args.output_dir / "celltype_cycling_counts.csv")

    # Print summary
    print("\n" + "=" * 70)
    print("CYCLING POST-HOC SUMMARY")
    print("=" * 70)
    print(f"  Overall cycling fraction: {summary['cycling_fraction_overall']:.1%}")
    print(f"  IL1B x source cycling correlation: {corr_source:.3f}" if pd.notna(corr_source) else "  IL1B x source cycling correlation: N/A")
    print(f"  IL1B x target cycling correlation: {corr_target:.3f}" if pd.notna(corr_target) else "  IL1B x target cycling correlation: N/A")
    print(f"  Cycling effect on IL1B: {summary['cycling_effect_on_il1b']}")

    if abs(summary.get('cycling_effect_strength', 0)) > 0.2:
        print("\n  FINDING: Cell cycling status correlates with IL1B signaling")
        print("  Consider stratified analysis in downstream modeling")
    else:
        print("\n  FINDING: Cell cycling has minimal effect on IL1B-IL1R1 interactions")
        print("  Coarse cell type analysis is sufficient")

    print(f"\nOutput: {args.output_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
