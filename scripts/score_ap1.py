#!/usr/bin/env python3
"""
Score AP-1 transcription factor activity across cells.

AP-1 is a stress-response TF family implicated in cellular plasticity and drug resistance.
This script scores AP-1 activity to test whether inflammatory niches (IL1B-high)
correlate with AP-1 activation, supporting the stress-induced plasticity hypothesis.

Usage:
    python score_ap1.py

Output:
    /data1/chaunzt1/stagebridge/processed/luad_evo/ap1_scores.parquet
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats

# AP-1 family transcription factors
AP1_GENES = [
    'FOS', 'FOSB', 'FOSL1', 'FOSL2',  # FOS family
    'JUN', 'JUNB', 'JUND',             # JUN family
    'ATF3', 'ATF4', 'ATF6',            # ATF family (AP-1 related)
    'BATF', 'BATF2', 'BATF3',          # BATF family
]

# AP-1 target genes (downstream readout of AP-1 activity)
AP1_TARGETS = [
    'MMP1', 'MMP9', 'IL6', 'IL8', 'CXCL8', 'VEGFA',
    'CCND1', 'BCL2', 'FAS', 'TP53', 'CDKN1A',
]

DATA_PATH = '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad'
OUTPUT_PATH = '/data1/chaunzt1/stagebridge/processed/luad_evo/ap1_scores.parquet'


def main():
    print(f"Loading {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

    # Check which AP-1 genes exist
    ap1_found = [g for g in AP1_GENES if g in adata.var_names]
    ap1_targets_found = [g for g in AP1_TARGETS if g in adata.var_names]

    print(f"\nAP-1 TFs found: {ap1_found}")
    print(f"AP-1 targets found: {ap1_targets_found}")

    if not ap1_found:
        print("ERROR: No AP-1 genes found in dataset!")
        return

    # Score AP-1 TF expression
    sc.tl.score_genes(adata, gene_list=ap1_found, score_name='ap1_tf_score')
    print(f"Computed ap1_tf_score")

    # Score AP-1 target expression (functional readout)
    if ap1_targets_found:
        sc.tl.score_genes(adata, gene_list=ap1_targets_found, score_name='ap1_target_score')
        print(f"Computed ap1_target_score")

    # Build results dataframe
    cols_to_keep = ['stage', 'cell_type', 'ap1_tf_score']
    if 'ap1_target_score' in adata.obs.columns:
        cols_to_keep.append('ap1_target_score')

    # Add donor if available
    if 'donor_id' in adata.obs.columns:
        cols_to_keep.insert(0, 'donor_id')

    results = adata.obs[cols_to_keep].copy()

    # Add individual AP-1 TF expression
    for g in ap1_found:
        expr = adata[:, g].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        results[g] = expr.flatten()

    # Add IL1B if available (for correlation analysis)
    if 'IL1B' in adata.var_names:
        expr = adata[:, 'IL1B'].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        results['IL1B'] = expr.flatten()
        print("Added IL1B expression for correlation analysis")

    # Save results
    results.to_parquet(OUTPUT_PATH)
    print(f"\nSaved {len(results)} cells to {OUTPUT_PATH}")

    # Print summary statistics
    print("\n=== AP-1 Score by Stage ===")
    stage_summary = results.groupby('stage')['ap1_tf_score'].agg(['mean', 'std', 'count'])
    print(stage_summary)

    # Correlation with IL1B if available
    if 'IL1B' in results.columns:
        r, p = stats.spearmanr(results['ap1_tf_score'], results['IL1B'])
        print(f"\nAP-1 vs IL1B correlation: r={r:.3f}, p={p:.2e}")

    # Stage correlation
    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4, 'Invasive': 4}
    if results['stage'].isin(stage_map.keys()).all():
        stage_numeric = results['stage'].map(stage_map)
        r, p = stats.spearmanr(results['ap1_tf_score'], stage_numeric)
        print(f"AP-1 vs Stage correlation: r={r:.3f}, p={p:.2e}")


if __name__ == '__main__':
    main()
