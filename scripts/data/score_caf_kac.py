#!/usr/bin/env python3
"""Score CAF subtypes and KAC signatures on snRNA data.

Standalone script - adds to existing signatures without rerunning full prep.
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path


# CAF subtypes from Elyada et al. 2019 / Ohlund et al. 2017
CAF_SIGNATURES = {
    'myCAF': ['ACTA2', 'TAGLN', 'MYL9', 'TPM1', 'TPM2', 'MMP11', 'POSTN', 'TNC', 'COL11A1', 'COMP'],
    'iCAF': ['IL6', 'CXCL1', 'CXCL2', 'CXCL12', 'CCL2', 'PDGFRA', 'HAS1', 'CFD', 'LMNA', 'DPT'],
    'apCAF': ['CD74', 'HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'SLPI'],  # antigen-presenting CAF
}

# KAC (Kadara Alveolar Cell) - reactive pneumocyte-like progenitors
# From Kadara/Peng papers on LUAD precursors
KAC_SIGNATURE = {
    'KAC': ['SOX9', 'KRT8', 'KRT18', 'LGALS3', 'CLDN4', 'CLDN7', 'NAPSA', 'SLC34A2', 'MUC1', 'CEACAM6'],
    'AT2_progenitor': ['SFTPC', 'SFTPA1', 'SFTPA2', 'SFTPB', 'ABCA3', 'LAMP3', 'NKX2-1'],
    'AT1': ['AGER', 'HOPX', 'PDPN', 'CAV1', 'AQP5'],
}

# Additional relevant signatures
EXTRA_SIGNATURES = {
    'IL1_response': ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN', 'CASP1', 'NLRP3', 'PYCARD'],
    'alveolar_damage': ['KRT8', 'CLDN4', 'LGALS3', 'AREG', 'TP63', 'KRT5'],  # ADI/damage-associated
}


def score_signatures(adata, signatures):
    """Score gene signatures using scanpy."""
    scores = {}
    for name, genes in signatures.items():
        # Filter to genes present in data
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 3:
            print(f"  {name}: only {len(present)}/{len(genes)} genes found, skipping")
            continue

        print(f"  {name}: {len(present)}/{len(genes)} genes")
        sc.tl.score_genes(adata, present, score_name=f'{name}_score', use_raw=False)
        scores[name] = adata.obs[f'{name}_score'].values

    return scores


def main():
    parser = argparse.ArgumentParser(description='Score CAF/KAC signatures')
    parser.add_argument('--snrna', type=str,
                        default='/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_merged.h5ad',
                        help='Path to snRNA h5ad')
    parser.add_argument('--output-dir', type=str,
                        default='/data1/chaunzt1/stagebridge/processed/luad_evo/canonical/signatures',
                        help='Output directory')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.snrna}...')
    adata = sc.read_h5ad(args.snrna)
    print(f'  {adata.n_obs} cells, {adata.n_vars} genes')

    # Combine all signatures
    all_sigs = {}
    all_sigs.update(CAF_SIGNATURES)
    all_sigs.update(KAC_SIGNATURE)
    all_sigs.update(EXTRA_SIGNATURES)

    print('Scoring signatures...')
    scores = score_signatures(adata, all_sigs)

    # Build dataframe
    score_df = pd.DataFrame(scores, index=adata.obs_names)
    score_df.columns = [f'{c}_score' if not c.endswith('_score') else c for c in score_df.columns]

    # Add cell metadata
    if 'stage' in adata.obs.columns:
        score_df['stage'] = adata.obs['stage'].values
    if 'cell_type' in adata.obs.columns:
        score_df['cell_type'] = adata.obs['cell_type'].values

    # Save per-cell scores
    score_df.to_parquet(out / 'caf_kac_scores.parquet')
    print(f'Saved {out / "caf_kac_scores.parquet"}')

    # Stage summary
    if 'stage' in score_df.columns:
        score_cols = [c for c in score_df.columns if c.endswith('_score')]
        stage_summary = score_df.groupby('stage')[score_cols].agg(['mean', 'std'])
        stage_summary.columns = ['_'.join(col) for col in stage_summary.columns]
        stage_summary.to_parquet(out / 'caf_kac_by_stage.parquet')
        print(f'Saved {out / "caf_kac_by_stage.parquet"}')
        print('\nStage summary (means):')
        print(score_df.groupby('stage')[score_cols].mean().round(3))

    # Cell type summary (for CAF analysis)
    if 'cell_type' in score_df.columns:
        score_cols = [c for c in score_df.columns if c.endswith('_score')]
        ct_summary = score_df.groupby('cell_type')[score_cols].agg(['mean', 'std'])
        ct_summary.columns = ['_'.join(col) for col in ct_summary.columns]
        ct_summary.to_parquet(out / 'caf_kac_by_celltype.parquet')
        print(f'\nSaved {out / "caf_kac_by_celltype.parquet"}')

    print('\nDone!')


if __name__ == '__main__':
    main()
