#!/usr/bin/env python
"""Check if snRNA and spatial embeddings are aligned after gamma fix.

If the fix worked, snRNA and spatial should be intermixed in UMAP.
If they're still separated, need scArches surgery.

Usage:
    python scripts/check_embedding_alignment.py --cells /path/to/cells.parquet
    python scripts/check_embedding_alignment.py --use-config
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Check embedding alignment between modalities')
    parser.add_argument('--cells', type=Path, help='Path to cells.parquet')
    parser.add_argument('--use-config', action='store_true', help='Use paths from workflow/config.yaml')
    parser.add_argument('--output', type=Path, default=None, help='Output path for figure')
    parser.add_argument('--n-samples', type=int, default=50000, help='Subsample for speed (0=all)')
    args = parser.parse_args()

    if args.use_config:
        import yaml
        config_path = Path(__file__).parent.parent / 'workflow' / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)
        cells_path = Path(config['paths']['data_dir']) / 'cells.parquet'
        output_path = args.output or Path(config['paths']['figures_dir']) / 'embedding_alignment.png'
    else:
        if not args.cells:
            parser.error('Must provide --cells or use --use-config')
        cells_path = args.cells
        output_path = args.output or Path('embedding_alignment.png')

    print(f'Loading {cells_path}...')
    cells = pd.read_parquet(cells_path)
    print(f'  {len(cells):,} cells')

    # Extract embedding columns
    hlca_cols = [c for c in cells.columns if c.startswith('hlca_latent_')]
    luca_cols = [c for c in cells.columns if c.startswith('luca_latent_')]
    emb_cols = hlca_cols + luca_cols

    if not emb_cols:
        print('ERROR: No embedding columns found (hlca_latent_*, luca_latent_*)')
        return

    print(f'  HLCA dims: {len(hlca_cols)}, LuCA dims: {len(luca_cols)}')

    # Determine modality
    cells['modality'] = ['spatial' if str(cid).startswith('spatial_') else 'snRNA'
                         for cid in cells['cell_id']]

    n_snrna = (cells['modality'] == 'snRNA').sum()
    n_spatial = (cells['modality'] == 'spatial').sum()
    print(f'  snRNA: {n_snrna:,}, spatial: {n_spatial:,}')

    # Subsample if needed
    if args.n_samples > 0 and len(cells) > args.n_samples:
        print(f'  Subsampling to {args.n_samples:,} cells (stratified by modality)...')
        snrna_idx = cells[cells['modality'] == 'snRNA'].sample(
            min(args.n_samples // 2, n_snrna), random_state=42
        ).index
        spatial_idx = cells[cells['modality'] == 'spatial'].sample(
            min(args.n_samples // 2, n_spatial), random_state=42
        ).index
        cells = cells.loc[list(snrna_idx) + list(spatial_idx)]
        print(f'  Subsampled to {len(cells):,} cells')

    # Create AnnData
    X = cells[emb_cols].values.astype(np.float32)

    # Check for NaN/Inf
    nan_mask = np.isnan(X).any(axis=1)
    inf_mask = np.isinf(X).any(axis=1)
    bad_mask = nan_mask | inf_mask
    if bad_mask.sum() > 0:
        print(f'  WARNING: {bad_mask.sum():,} cells have NaN/Inf embeddings, removing...')
        cells = cells[~bad_mask]
        X = X[~bad_mask]

    adata = ad.AnnData(X=X)
    adata.obs['modality'] = cells['modality'].values
    adata.obs['cell_id'] = cells['cell_id'].values

    if 'stage' in cells.columns:
        adata.obs['stage'] = cells['stage'].values

    # Compute UMAP
    print('Computing neighbors...')
    sc.pp.neighbors(adata, use_rep='X', n_neighbors=30)

    print('Computing UMAP...')
    sc.tl.umap(adata)

    # Compute mixing metric (optional)
    print('Computing mixing score...')
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(adata.obsm['X_umap'])
    _, indices = nn.kneighbors(adata.obsm['X_umap'])

    modality_binary = (adata.obs['modality'] == 'spatial').values.astype(int)
    mixing_scores = []
    for i, neighbors in enumerate(indices):
        same_modality = (modality_binary[neighbors] == modality_binary[i]).mean()
        mixing_scores.append(1 - same_modality)  # Higher = more mixed

    mean_mixing = np.mean(mixing_scores)
    print(f'  Mean mixing score: {mean_mixing:.3f} (0=separated, 0.5=well-mixed)')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Color by modality
    ax = axes[0]
    for mod, color in [('snRNA', '#1f77b4'), ('spatial', '#ff7f0e')]:
        mask = adata.obs['modality'] == mod
        ax.scatter(
            adata.obsm['X_umap'][mask, 0],
            adata.obsm['X_umap'][mask, 1],
            c=color, label=mod, s=1, alpha=0.3
        )
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(f'Modality (mixing={mean_mixing:.3f})')
    ax.legend(markerscale=5)

    # Plot 2: Color by stage if available
    ax = axes[1]
    if 'stage' in adata.obs.columns:
        stages = adata.obs['stage'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
        for stage, color in zip(sorted(stages), colors):
            mask = adata.obs['stage'] == stage
            ax.scatter(
                adata.obsm['X_umap'][mask, 0],
                adata.obsm['X_umap'][mask, 1],
                c=[color], label=stage, s=1, alpha=0.3
            )
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title('Stage')
        ax.legend(markerscale=5)
    else:
        ax.text(0.5, 0.5, 'No stage column', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Stage (not available)')

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')

    # Interpretation
    print('\n=== Interpretation ===')
    if mean_mixing > 0.35:
        print('GOOD: Modalities appear well-mixed. Gamma fix likely worked.')
    elif mean_mixing > 0.2:
        print('PARTIAL: Some mixing but still separation. May need scArches.')
    else:
        print('POOR: Modalities still separated. Need scArches surgery.')


if __name__ == '__main__':
    main()
