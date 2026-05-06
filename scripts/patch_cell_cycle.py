#!/usr/bin/env python
"""Patch cell cycle scores and cell types into existing parquets.

This is a one-time fix script - does NOT rerun prepare_data.
Adds:
- Cell cycle: S_score, G2M_score, phase (from h5ad)
- Cell types: cell_type_hlca, cell_type_luca (from DestVI results)

Usage:
    python scripts/patch_cell_cycle.py \
        --h5ad /path/to/snrna_with_celltypes.h5ad \
        --cells /path/to/cells.parquet \
        --neighborhoods /path/to/neighborhoods.parquet \
        --destvi-hlca /path/to/hlca/cell_type_proportions.parquet \
        --destvi-luca /path/to/luca/cell_type_proportions.parquet

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


def load_cell_types(hlca_path: Path, luca_path: Path) -> pd.DataFrame:
    """Load cell type assignments from DestVI results.

    Returns DataFrame with cell_id as index and cell_type_hlca, cell_type_luca columns.
    """
    result = pd.DataFrame()

    if luca_path and luca_path.exists():
        print(f'Loading LuCA cell types from {luca_path}...')
        luca = pd.read_parquet(luca_path)
        # Get cell type columns (exclude metadata)
        luca_types = [c for c in luca.columns if c not in ['sample', 'cell_id']]
        if luca_types:
            result['cell_type_luca'] = luca[luca_types].idxmax(axis=1)
            result['cell_type_luca_confidence'] = luca[luca_types].max(axis=1)
            print(f'  {len(result):,} cells, {result["cell_type_luca"].nunique()} unique types')
            print(f'  Top types: {result["cell_type_luca"].value_counts().head(5).to_dict()}')

    if hlca_path and hlca_path.exists():
        print(f'Loading HLCA cell types from {hlca_path}...')
        hlca = pd.read_parquet(hlca_path)
        hlca_types = [c for c in hlca.columns if c not in ['sample', 'cell_id']]
        if hlca_types:
            result['cell_type_hlca'] = hlca[hlca_types].idxmax(axis=1)
            result['cell_type_hlca_confidence'] = hlca[hlca_types].max(axis=1)
            print(f'  {result["cell_type_hlca"].nunique()} unique HLCA types')
            print(f'  Top types: {result["cell_type_hlca"].value_counts().head(5).to_dict()}')

    return result


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


def patch_cells(cells_path: Path, scores_df: pd.DataFrame, celltypes_df: pd.DataFrame | None,
                backup: bool = True) -> None:
    """Add cell cycle and cell type columns to cells.parquet."""
    print(f'\nPatching {cells_path}...')
    cells = pd.read_parquet(cells_path)
    print(f'  {len(cells):,} cells')
    print(f'  BEFORE columns: {list(cells.columns)}')
    print(f'  BEFORE head:')
    print(cells.head(3).to_string())

    # Backup
    if backup:
        backup_path = cells_path.with_suffix('.parquet.bak')
        if not backup_path.exists():
            print(f'  Backing up to {backup_path}')
            cells.to_parquet(backup_path)

    # Merge cell cycle scores
    if scores_df is not None:
        scores_idx = scores_df.set_index('cell_id')

        # Handle cell_id matching (may need prefix adjustment)
        matched = cells['cell_id'].isin(scores_idx.index).sum()
        print(f'  Cell cycle direct match: {matched:,}/{len(cells):,} cells')

        if matched < len(cells) * 0.5:
            # Try stripping prefixes
            cells['_match_id'] = cells['cell_id'].str.replace('^spatial_', '', regex=True)
            matched = cells['_match_id'].isin(scores_idx.index).sum()
            print(f'  After stripping spatial_ prefix: {matched:,}/{len(cells):,} cells')

            if matched > len(cells) * 0.5:
                cells['S_score'] = cells['_match_id'].map(scores_idx['S_score'])
                cells['G2M_score'] = cells['_match_id'].map(scores_idx['G2M_score'])
                cells['phase'] = cells['_match_id'].map(scores_idx['phase'])
            cells = cells.drop(columns=['_match_id'])
        else:
            cells['S_score'] = cells['cell_id'].map(scores_idx['S_score'])
            cells['G2M_score'] = cells['cell_id'].map(scores_idx['G2M_score'])
            cells['phase'] = cells['cell_id'].map(scores_idx['phase'])

        filled = cells['S_score'].notna().sum()
        print(f'  Filled {filled:,}/{len(cells):,} cells with cell cycle scores')

    # Merge cell types (DestVI results are indexed by spot/cell barcode)
    if celltypes_df is not None and len(celltypes_df) > 0:
        # DestVI index is typically the barcode without prefix
        # Try matching with stripped cell_id
        cells['_match_id'] = cells['cell_id'].str.replace('^spatial_', '', regex=True)

        for col in ['cell_type_hlca', 'cell_type_hlca_confidence', 'cell_type_luca', 'cell_type_luca_confidence']:
            if col in celltypes_df.columns:
                cells[col] = cells['_match_id'].map(celltypes_df[col])
                filled = cells[col].notna().sum()
                print(f'  Filled {filled:,}/{len(cells):,} cells with {col}')

        cells = cells.drop(columns=['_match_id'])

    # Save
    print(f'  AFTER columns: {list(cells.columns)}')
    print(f'  AFTER head:')
    print(cells.head(3).to_string())
    cells.to_parquet(cells_path)
    print(f'  Saved {cells_path}')


def patch_neighborhoods(nhood_path: Path, scores_df: pd.DataFrame, celltypes_df: pd.DataFrame | None,
                        backup: bool = True) -> None:
    """Add cell cycle, cell type columns and update stats_z in neighborhoods.parquet."""
    print(f'\nPatching {nhood_path}...')
    nhood = pd.read_parquet(nhood_path)
    print(f'  {len(nhood):,} neighborhoods')
    print(f'  BEFORE columns: {list(nhood.columns)}')
    print(f'  BEFORE head (key cols):')
    key_cols = [c for c in ['cell_id', 'stage', 'S_score', 'G2M_score', 'cell_type_luca', 'stats_z'] if c in nhood.columns]
    print(nhood[key_cols].head(3).to_string())

    # Backup
    if backup:
        backup_path = nhood_path.with_suffix('.parquet.bak')
        if not backup_path.exists():
            print(f'  Backing up to {backup_path}')
            nhood.to_parquet(backup_path)

    # Create match column (strip spatial_ prefix for matching)
    nhood['_match_id'] = nhood['cell_id'].str.replace('^spatial_', '', regex=True)

    # Merge cell cycle scores
    if scores_df is not None:
        scores_idx = scores_df.set_index('cell_id')
        matched = nhood['_match_id'].isin(scores_idx.index).sum()
        print(f'  Cell cycle match: {matched:,}/{len(nhood):,} neighborhoods')

        nhood['S_score'] = nhood['_match_id'].map(scores_idx['S_score'])
        nhood['G2M_score'] = nhood['_match_id'].map(scores_idx['G2M_score'])
        nhood['phase'] = nhood['_match_id'].map(scores_idx['phase'])

        filled = nhood['S_score'].notna().sum()
        print(f'  Filled {filled:,}/{len(nhood):,} neighborhoods with cell cycle scores')

    # Merge cell types
    if celltypes_df is not None and len(celltypes_df) > 0:
        for col in ['cell_type_hlca', 'cell_type_hlca_confidence', 'cell_type_luca', 'cell_type_luca_confidence']:
            if col in celltypes_df.columns:
                nhood[col] = nhood['_match_id'].map(celltypes_df[col])
                filled = nhood[col].notna().sum()
                print(f'  Filled {filled:,}/{len(nhood):,} neighborhoods with {col}')

    nhood = nhood.drop(columns=['_match_id'])

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
    print(f'  AFTER columns: {list(nhood.columns)}')
    print(f'  AFTER head (key cols):')
    key_cols = [c for c in ['cell_id', 'stage', 'S_score', 'G2M_score', 'cell_type_luca', 'cell_type_hlca', 'stats_z'] if c in nhood.columns]
    print(nhood[key_cols].head(3).to_string())
    nhood.to_parquet(nhood_path)
    print(f'  Saved {nhood_path}')


def main():
    parser = argparse.ArgumentParser(description='Patch cell cycle scores and cell types into parquet files')
    parser.add_argument('--h5ad', type=Path, help='Path to snrna_with_celltypes.h5ad')
    parser.add_argument('--cells', type=Path, help='Path to cells.parquet')
    parser.add_argument('--neighborhoods', type=Path, help='Path to neighborhoods.parquet')
    parser.add_argument('--destvi-hlca', type=Path, help='Path to HLCA DestVI cell_type_proportions.parquet')
    parser.add_argument('--destvi-luca', type=Path, help='Path to LuCA DestVI cell_type_proportions.parquet')
    parser.add_argument('--use-config', action='store_true', help='Use paths from workflow/config.yaml')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup files')
    parser.add_argument('--skip-cell-cycle', action='store_true', help='Skip cell cycle scoring (only add cell types)')
    args = parser.parse_args()

    if args.use_config:
        import yaml
        config_path = Path(__file__).parent.parent / 'workflow' / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        h5ad_path = Path(config['paths']['snrna_h5ad'])
        cells_path = Path(config['paths']['data_dir']) / 'cells.parquet'
        nhood_path = Path(config['paths']['data_dir']) / 'neighborhoods.parquet'
        destvi_hlca_path = Path(config['paths']['destvi_hlca'])
        destvi_luca_path = Path(config['paths']['destvi_luca'])
    else:
        h5ad_path = args.h5ad
        cells_path = args.cells
        nhood_path = args.neighborhoods
        destvi_hlca_path = args.destvi_hlca
        destvi_luca_path = args.destvi_luca

        if not cells_path or not nhood_path:
            parser.error('Must provide --cells and --neighborhoods or use --use-config')

    # Compute cell cycle scores
    scores_df = None
    if not args.skip_cell_cycle:
        if h5ad_path and h5ad_path.exists():
            scores_df = compute_cell_cycle_scores(h5ad_path)
        else:
            print(f'WARNING: h5ad not found at {h5ad_path}, skipping cell cycle scoring')

    # Load cell types from DestVI
    celltypes_df = None
    if destvi_hlca_path or destvi_luca_path:
        celltypes_df = load_cell_types(destvi_hlca_path, destvi_luca_path)

    if scores_df is None and (celltypes_df is None or len(celltypes_df) == 0):
        print('ERROR: Nothing to patch - no cell cycle scores and no cell types')
        return

    # Patch files
    backup = not args.no_backup
    patch_cells(cells_path, scores_df, celltypes_df, backup=backup)
    patch_neighborhoods(nhood_path, scores_df, celltypes_df, backup=backup)

    print('\nDone!')
    if scores_df is not None:
        print('  - Cell cycle scores added (S_score, G2M_score, phase)')
    if celltypes_df is not None and len(celltypes_df) > 0:
        print('  - Cell types added (cell_type_hlca, cell_type_luca)')
    print('Backups saved as .parquet.bak files (delete manually if satisfied)')


if __name__ == '__main__':
    main()
