#!/usr/bin/env python
"""Patch cell cycle scores into existing cells.parquet and neighborhoods.parquet.

This is a one-time fix script - does NOT rerun prepare_data.

Usage:
    python scripts/patch_cell_cycle.py \
        --h5ad /path/to/snrna_with_celltypes.h5ad \
        --cells /path/to/cells.parquet \
        --neighborhoods /path/to/neighborhoods.parquet

Or with defaults from config:
    python scripts/patch_cell_cycle.py --use-config
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Tirosh et al. 2016 cell cycle genes
S_GENES = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG',
    'GINS2', 'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP',
    'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76',
    'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51',
    'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
    'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
]

G2M_GENES = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A',
    'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF',
    'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB',
    'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP',
    'CDCA3', 'HN1', 'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1',
    'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR',
    'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF',
    'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]


def compute_cell_cycle_scores(h5ad_path: Path) -> pd.DataFrame:
    """Compute cell cycle scores from h5ad."""
    import scanpy as sc

    print(f'Loading {h5ad_path}...')
    adata = sc.read_h5ad(h5ad_path)
    print(f'  {adata.n_obs:,} cells, {adata.n_vars:,} genes')

    # Filter to genes present
    s_genes = [g for g in S_GENES if g in adata.var_names]
    g2m_genes = [g for g in G2M_GENES if g in adata.var_names]

    print(f'  S phase genes: {len(s_genes)}/{len(S_GENES)} present')
    print(f'  G2M phase genes: {len(g2m_genes)}/{len(G2M_GENES)} present')

    if len(s_genes) < 10 or len(g2m_genes) < 10:
        print('  WARNING: Too few cell cycle genes, scores may be unreliable')

    # Score
    print('  Computing cell cycle scores...')
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

    result = pd.DataFrame({
        'cell_id': adata.obs.index.values,
        'S_score': adata.obs['S_score'].values,
        'G2M_score': adata.obs['G2M_score'].values,
        'phase': adata.obs['phase'].values,
    })

    print(f'  Phase distribution: {result["phase"].value_counts().to_dict()}')

    return result


def patch_cells(cells_path: Path, scores_df: pd.DataFrame, backup: bool = True) -> None:
    """Add cell cycle columns to cells.parquet."""
    print(f'\nPatching {cells_path}...')
    cells = pd.read_parquet(cells_path)
    print(f'  {len(cells):,} cells')

    # Check if already has scores
    if 'S_score' in cells.columns and cells['S_score'].notna().mean() > 0.5:
        print('  WARNING: S_score already present and mostly non-null, skipping')
        return

    # Backup
    if backup:
        backup_path = cells_path.with_suffix('.parquet.bak')
        if not backup_path.exists():
            print(f'  Backing up to {backup_path}')
            cells.to_parquet(backup_path)

    # Merge scores
    scores_df = scores_df.set_index('cell_id')

    # Handle cell_id matching (may need prefix adjustment)
    matched = cells['cell_id'].isin(scores_df.index).sum()
    print(f'  Direct match: {matched:,}/{len(cells):,} cells')

    if matched < len(cells) * 0.5:
        # Try stripping prefixes
        cells['_match_id'] = cells['cell_id'].str.replace('^spatial_', '', regex=True)
        matched = cells['_match_id'].isin(scores_df.index).sum()
        print(f'  After stripping spatial_ prefix: {matched:,}/{len(cells):,} cells')

        if matched > len(cells) * 0.5:
            cells['S_score'] = cells['_match_id'].map(scores_df['S_score'])
            cells['G2M_score'] = cells['_match_id'].map(scores_df['G2M_score'])
            cells['phase'] = cells['_match_id'].map(scores_df['phase'])
        cells = cells.drop(columns=['_match_id'])
    else:
        cells['S_score'] = cells['cell_id'].map(scores_df['S_score'])
        cells['G2M_score'] = cells['cell_id'].map(scores_df['G2M_score'])
        cells['phase'] = cells['cell_id'].map(scores_df['phase'])

    # Report
    filled = cells['S_score'].notna().sum()
    print(f'  Filled {filled:,}/{len(cells):,} cells with scores')

    # Save
    cells.to_parquet(cells_path)
    print(f'  Saved {cells_path}')


def patch_neighborhoods(nhood_path: Path, scores_df: pd.DataFrame, backup: bool = True) -> None:
    """Add cell cycle columns and update stats_z in neighborhoods.parquet."""
    print(f'\nPatching {nhood_path}...')
    nhood = pd.read_parquet(nhood_path)
    print(f'  {len(nhood):,} neighborhoods')

    # Backup
    if backup:
        backup_path = nhood_path.with_suffix('.parquet.bak')
        if not backup_path.exists():
            print(f'  Backing up to {backup_path}')
            nhood.to_parquet(backup_path)

    # Merge scores
    scores_df = scores_df.set_index('cell_id')

    # Handle cell_id matching
    matched = nhood['cell_id'].isin(scores_df.index).sum()
    print(f'  Direct match: {matched:,}/{len(nhood):,} neighborhoods')

    match_col = 'cell_id'
    if matched < len(nhood) * 0.5:
        nhood['_match_id'] = nhood['cell_id'].str.replace('^spatial_', '', regex=True)
        matched = nhood['_match_id'].isin(scores_df.index).sum()
        print(f'  After stripping spatial_ prefix: {matched:,}/{len(nhood):,}')
        match_col = '_match_id'

    nhood['S_score'] = nhood[match_col].map(scores_df['S_score'])
    nhood['G2M_score'] = nhood[match_col].map(scores_df['G2M_score'])
    nhood['phase'] = nhood[match_col].map(scores_df['phase'])

    if '_match_id' in nhood.columns:
        nhood = nhood.drop(columns=['_match_id'])

    filled = nhood['S_score'].notna().sum()
    print(f'  Filled {filled:,}/{len(nhood):,} neighborhoods with scores')

    # Update stats_z to include cell cycle
    # stats_z is [caf_fraction, immune_fraction, diversity, S_score, G2M_score]
    print('  Updating stats_z...')
    stats_cols = ['caf_fraction', 'immune_fraction', 'diversity', 'S_score', 'G2M_score']

    new_stats_z = []
    for _, row in nhood.iterrows():
        stats = []
        for col in stats_cols:
            val = row.get(col, 0)
            stats.append(float(val) if pd.notna(val) else 0.0)
        new_stats_z.append(stats)

    nhood['stats_z'] = new_stats_z

    # Verify
    sample_stats = nhood['stats_z'].iloc[0]
    print(f'  Sample stats_z: {sample_stats}')

    # Save
    nhood.to_parquet(nhood_path)
    print(f'  Saved {nhood_path}')


def main():
    parser = argparse.ArgumentParser(description='Patch cell cycle scores into parquet files')
    parser.add_argument('--h5ad', type=Path, help='Path to snrna_with_celltypes.h5ad')
    parser.add_argument('--cells', type=Path, help='Path to cells.parquet')
    parser.add_argument('--neighborhoods', type=Path, help='Path to neighborhoods.parquet')
    parser.add_argument('--use-config', action='store_true', help='Use paths from workflow/config.yaml')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup files')
    args = parser.parse_args()

    if args.use_config:
        import yaml
        config_path = Path(__file__).parent.parent / 'workflow' / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        h5ad_path = Path(config['paths']['snrna_h5ad'])
        cells_path = Path(config['paths']['data_dir']) / 'cells.parquet'
        nhood_path = Path(config['paths']['data_dir']) / 'neighborhoods.parquet'
    else:
        if not all([args.h5ad, args.cells, args.neighborhoods]):
            parser.error('Must provide --h5ad, --cells, --neighborhoods or use --use-config')
        h5ad_path = args.h5ad
        cells_path = args.cells
        nhood_path = args.neighborhoods

    # Compute scores
    scores_df = compute_cell_cycle_scores(h5ad_path)

    # Patch files
    backup = not args.no_backup
    patch_cells(cells_path, scores_df, backup=backup)
    patch_neighborhoods(nhood_path, scores_df, backup=backup)

    print('\nDone! Cell cycle scores added.')
    print('Backups saved as .parquet.bak files (delete manually if satisfied)')


if __name__ == '__main__':
    main()
