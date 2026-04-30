"""Data preparation pipeline for StageBridge.

Builds neighborhoods with raw cells for learned ring pooling from existing cells.parquet.
Uses LuCA DestVI deconvolution results for CAF/immune fractions.

Usage:
    python -m stagebridge.pipelines.prepare_data \
        --cells /path/to/cells.parquet \
        --output-dir /path/to/output \
        --destvi /path/to/luca/destvi/cell_type_proportions.parquet \
        --figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import entropy


# LuCA cell type groupings for biological fractions
CAF_TYPES = ['fibroblast of lung', 'bronchus fibroblast of lung', 'stromal cell']
IMMUNE_TYPES = [
    'B cell', 'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell',
    'regulatory T cell', 'natural killer cell', 'alveolar macrophage', 'macrophage',
    'classical monocyte', 'non-classical monocyte', 'myeloid cell', 'neutrophil',
    'dendritic cell', 'conventional dendritic cell', 'CD1c-positive myeloid dendritic cell',
    'plasmacytoid dendritic cell', 'mast cell', 'plasma cell'
]


@dataclass
class PrepConfig:
    """Configuration for data preparation."""
    n_neighbors: int = 50
    ring_radii: tuple = (600, 1200, 2000, 3000)  # Micron scale for Visium
    max_cells_per_ring: int = 50


def load_destvi_fractions(luca_path: Path, hlca_path: Path | None = None) -> pd.DataFrame:
    """Load DestVI proportions and compute CAF/immune fractions + dominant cell types.

    Args:
        luca_path: LuCA DestVI proportions (fine-grained, cancer-aware)
        hlca_path: HLCA DestVI proportions (coarse, healthy reference)

    Returns:
        DataFrame with fractions and cell type annotations from both references.
    """
    luca = pd.read_parquet(luca_path)

    # LuCA cell type columns
    luca_types = [c for c in luca.columns if c not in ['sample']]

    # Compute CAF fraction from LuCA (sum of fibroblast types)
    caf_cols = [c for c in luca_types if c in CAF_TYPES]
    luca['caf_fraction'] = luca[caf_cols].sum(axis=1) if caf_cols else 0.0

    # Compute immune fraction from LuCA (sum of immune types)
    immune_cols = [c for c in luca_types if c in IMMUNE_TYPES]
    luca['immune_fraction'] = luca[immune_cols].sum(axis=1) if immune_cols else 0.0

    # Diversity from LuCA (more cell types = better entropy estimate)
    luca_probs = luca[luca_types].values
    luca['diversity'] = entropy(luca_probs + 1e-10, axis=1)

    # LuCA dominant cell type (fine-grained, includes "malignant cell")
    luca['cell_type_luca'] = luca[luca_types].idxmax(axis=1)
    luca['cell_type_luca_confidence'] = luca[luca_types].max(axis=1)

    # Malignant fraction (key for cancer analysis)
    if 'malignant cell' in luca_types:
        luca['malignant_fraction'] = luca['malignant cell']
    else:
        luca['malignant_fraction'] = 0.0

    result = luca[['caf_fraction', 'immune_fraction', 'diversity', 'malignant_fraction',
                   'cell_type_luca', 'cell_type_luca_confidence']].copy()

    # Add HLCA if provided (coarse cell types for broad categorization)
    if hlca_path and hlca_path.exists():
        hlca = pd.read_parquet(hlca_path)
        hlca_types = [c for c in hlca.columns if c not in ['sample']]

        # HLCA dominant cell type (coarse: AT2, Basal, Macrophages, etc.)
        result['cell_type_hlca'] = hlca[hlca_types].idxmax(axis=1)
        result['cell_type_hlca_confidence'] = hlca[hlca_types].max(axis=1)

    return result


def get_embedding_matrix(df: pd.DataFrame, prefix: str, dim: int) -> np.ndarray:
    """Extract embedding matrix from individual columns."""
    cols = [f'{prefix}_{i}' for i in range(dim)]
    if all(c in df.columns for c in cols):
        return df[cols].values
    elif prefix in df.columns:
        # Stored as array column
        return np.stack(df[prefix].values)
    else:
        raise ValueError(f'Cannot find {prefix} embeddings')


def build_neighborhoods(df: pd.DataFrame, config: PrepConfig) -> pd.DataFrame:
    """Build neighborhoods with raw neighbor cell lists for learned pooling."""
    # Only use cells with spatial coordinates
    spatial_mask = df['x_spatial'].notna() & df['y_spatial'].notna()
    spatial_df = df[spatial_mask].reset_index(drop=True)

    coords = spatial_df[['x_spatial', 'y_spatial']].values
    tree = KDTree(coords)

    n_cells = len(spatial_df)

    # Get embedding matrices
    z_fused = get_embedding_matrix(spatial_df, 'z_fused', 40)
    z_hlca = get_embedding_matrix(spatial_df, 'z_hlca', 30)
    z_luca = get_embedding_matrix(spatial_df, 'z_luca', 10)

    print(f'Building neighborhoods for {n_cells:,} spatial cells...')

    neighborhoods = []

    for i in range(n_cells):
        if i % 50000 == 0:
            print(f'  Processing cell {i:,}/{n_cells:,}...')

        cell = spatial_df.iloc[i]

        # Query all neighbors within max radius
        idx = tree.query_ball_point(coords[i], r=config.ring_radii[-1])
        idx = [j for j in idx if j != i]  # Exclude self

        if not idx:
            continue

        neighbor_coords = coords[idx]
        distances = np.linalg.norm(neighbor_coords - coords[i], axis=1)

        # Assign to rings
        ring_cells = [[] for _ in range(4)]

        prev_r = 0
        for ring_idx, r in enumerate(config.ring_radii):
            ring_mask = (distances > prev_r) & (distances <= r)
            ring_neighbor_idx = [idx[j] for j in np.where(ring_mask)[0]]

            # Limit cells per ring
            if len(ring_neighbor_idx) > config.max_cells_per_ring:
                ring_neighbor_idx = ring_neighbor_idx[:config.max_cells_per_ring]

            for j in ring_neighbor_idx:
                ring_cells[ring_idx].append(z_fused[j].tolist())

            prev_r = r

        # Skip if no neighbors in any ring
        if all(len(rc) == 0 for rc in ring_cells):
            continue

        neighborhoods.append({
            'cell_id': cell['cell_id'],
            'donor_id': cell['donor_id'],
            'stage': cell['stage'],
            'stage_idx': cell.get('stage_idx', 0),
            'receiver_z': z_fused[i].tolist(),
            'hlca_z': z_hlca[i].tolist(),
            'luca_z': z_luca[i].tolist(),
            'ring_1_cells': ring_cells[0],
            'ring_2_cells': ring_cells[1],
            'ring_3_cells': ring_cells[2],
            'ring_4_cells': ring_cells[3],
            'x_spatial': coords[i, 0],
            'y_spatial': coords[i, 1],
        })

    return pd.DataFrame(neighborhoods)


def add_conditioning_features(nhood_df: pd.DataFrame, cells_df: pd.DataFrame,
                              destvi_fractions: pd.DataFrame | None) -> pd.DataFrame:
    """Add conditioning features from cells and DestVI to neighborhoods."""
    # Create lookup from cell_id
    cells_indexed = cells_df.set_index('cell_id')

    # Columns to transfer from cells
    transfer_cols = [
        'S_score', 'G2M_score', 'phase',
        'clonal_pattern', 'clonal_pattern_idx',
        'tmb', 'kras_mut', 'egfr_mut', 'tp53_mut',
        'il1b_raw', 'kac_raw', 'proliferation_label',
    ]

    for col in transfer_cols:
        if col in cells_indexed.columns:
            nhood_df[col] = nhood_df['cell_id'].map(cells_indexed[col])

    # Add DestVI fractions if available
    if destvi_fractions is not None:
        # Strip 'spatial_' prefix from cell_id to match DestVI index
        destvi_cell_ids = nhood_df['cell_id'].str.replace('^spatial_', '', regex=True)
        destvi_cols = ['caf_fraction', 'immune_fraction', 'diversity', 'malignant_fraction',
                       'cell_type_luca', 'cell_type_luca_confidence',
                       'cell_type_hlca', 'cell_type_hlca_confidence']
        for col in destvi_cols:
            if col in destvi_fractions.columns:
                nhood_df[col] = destvi_cell_ids.map(destvi_fractions[col])

    # Build stats_z from conditioning features (7 dims)
    stats_cols = ['caf_fraction', 'immune_fraction', 'diversity',
                  'S_score', 'G2M_score', 'il1b_raw', 'kac_raw']

    stats_z = []
    for _, row in nhood_df.iterrows():
        stats = []
        for col in stats_cols:
            val = row.get(col, 0)
            stats.append(float(val) if pd.notna(val) else 0.0)
        stats_z.append(stats)

    nhood_df['stats_z'] = stats_z

    # Pathway features (from cells if available)
    pathway_cols = [f'pathway_raw_{i}' for i in range(14)]
    if all(col in cells_indexed.columns for col in pathway_cols):
        pathway_z = []
        for cell_id in nhood_df['cell_id']:
            if cell_id in cells_indexed.index:
                vals = [cells_indexed.loc[cell_id, col] for col in pathway_cols]
                # Pad to 40 dims
                vals = vals + [0.0] * (40 - len(vals))
                pathway_z.append(vals)
            else:
                pathway_z.append([0.0] * 40)
        nhood_df['pathway_z'] = pathway_z

    return nhood_df


def create_split_manifest(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> dict:
    """Create cross-validation split manifest (donor-held-out)."""
    np.random.seed(seed)

    donors = df['donor_id'].unique().tolist()
    np.random.shuffle(donors)

    folds = []
    fold_size = len(donors) // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else len(donors)

        test_donors = donors[test_start:test_end]
        remaining = [d for d in donors if d not in test_donors]

        val_size = max(1, len(remaining) // 5)
        val_donors = remaining[:val_size]
        train_donors = remaining[val_size:]

        folds.append({
            'fold': i,
            'train_donors': train_donors,
            'val_donors': val_donors,
            'test_donors': test_donors,
        })

    return {'folds': folds, 'seed': seed, 'n_folds': n_folds}


def generate_qc_figures(nhood_df: pd.DataFrame, output_dir: Path):
    """Generate QC figures for neighborhoods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 1. Feature distributions by stage
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    features = ['caf_fraction', 'immune_fraction', 'diversity', 'S_score', 'G2M_score', 'il1b_raw']

    for ax, feat in zip(axes.flat, features):
        if feat in nhood_df.columns:
            for stage in nhood_df['stage'].unique():
                mask = nhood_df['stage'] == stage
                vals = nhood_df.loc[mask, feat].dropna()
                if len(vals) > 0:
                    ax.hist(vals, bins=50, alpha=0.5, label=stage, density=True)
            ax.set_xlabel(feat)
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / 'feature_distributions.png', dpi=150)
    plt.close()

    # 2. Ring cell counts
    ring_counts = []
    for i in range(1, 5):
        col = f'ring_{i}_cells'
        if col in nhood_df.columns:
            counts = nhood_df[col].apply(len)
            ring_counts.append(counts)

    if ring_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(ring_counts, labels=[f'Ring {i}' for i in range(1, 5)])
        ax.set_ylabel('Cells per ring')
        ax.set_title('Neighborhood ring composition')
        plt.savefig(fig_dir / 'ring_counts.png', dpi=150)
        plt.close()

    # 3. Spatial map of sample donor
    sample_donor = nhood_df['donor_id'].value_counts().index[0]
    donor_df = nhood_df[nhood_df['donor_id'] == sample_donor]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    spatial_features = ['caf_fraction', 'immune_fraction', 'diversity', 'il1b_raw']

    for ax, feat in zip(axes.flat, spatial_features):
        if feat in donor_df.columns:
            vals = donor_df[feat].fillna(0)
            sc = ax.scatter(donor_df['x_spatial'], donor_df['y_spatial'],
                           c=vals, s=1, cmap='viridis')
            ax.set_title(f'{feat} ({sample_donor})')
            plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(fig_dir / 'spatial_features.png', dpi=150)
    plt.close()

    print(f'QC figures saved to {fig_dir}')


def prepare_data(
    cells_path: Path,
    output_dir: Path,
    destvi_luca_path: Path | None = None,
    destvi_hlca_path: Path | None = None,
    config: PrepConfig | None = None,
    make_figures: bool = True,
) -> dict:
    """Full data preparation pipeline."""
    if config is None:
        config = PrepConfig()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading cells from {cells_path}...')
    cells_df = pd.read_parquet(cells_path)
    print(f'Loaded {len(cells_df):,} cells')

    # Load DestVI fractions if provided
    destvi_fractions = None
    if destvi_luca_path and destvi_luca_path.exists():
        print(f'Loading DestVI fractions...')
        print(f'  LuCA: {destvi_luca_path}')
        if destvi_hlca_path:
            print(f'  HLCA: {destvi_hlca_path}')
        destvi_fractions = load_destvi_fractions(destvi_luca_path, destvi_hlca_path)
        print(f'  CAF fraction range: {destvi_fractions["caf_fraction"].min():.3f} - {destvi_fractions["caf_fraction"].max():.3f}')
        print(f'  Immune fraction range: {destvi_fractions["immune_fraction"].min():.3f} - {destvi_fractions["immune_fraction"].max():.3f}')
        print(f'  Malignant fraction range: {destvi_fractions["malignant_fraction"].min():.3f} - {destvi_fractions["malignant_fraction"].max():.3f}')

    # Build neighborhoods
    print('Building neighborhoods with raw cells...')
    nhood_df = build_neighborhoods(cells_df, config)
    print(f'Built {len(nhood_df):,} neighborhoods')

    # Add conditioning features
    print('Adding conditioning features...')
    nhood_df = add_conditioning_features(nhood_df, cells_df, destvi_fractions)

    # Create split manifest
    print('Creating split manifest...')
    manifest = create_split_manifest(nhood_df)

    # Save outputs
    nhood_out = output_dir / 'neighborhoods.parquet'
    manifest_out = output_dir / 'split_manifest.json'
    cells_out = output_dir / 'cells.parquet'

    nhood_df.to_parquet(nhood_out)
    with open(manifest_out, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Symlink cells.parquet (required by contracts)
    if not cells_out.exists():
        cells_out.symlink_to(cells_path.resolve())

    print(f'Saved neighborhoods to {nhood_out}')
    print(f'Saved manifest to {manifest_out}')
    print(f'Linked cells to {cells_out}')

    # Generate figures
    if make_figures:
        print('Generating QC figures...')
        generate_qc_figures(nhood_df, output_dir)

    # Summary
    summary = {
        'n_cells_input': len(cells_df),
        'n_neighborhoods': len(nhood_df),
        'n_donors': nhood_df['donor_id'].nunique(),
        'n_folds': len(manifest['folds']),
        'stages': nhood_df['stage'].unique().tolist(),
        'ring_radii': list(config.ring_radii),
        'destvi_used': destvi_luca_path is not None,
    }

    with open(output_dir / 'prep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Prepare data for StageBridge')
    parser.add_argument('--cells', required=True, type=Path, help='Path to cells.parquet')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    parser.add_argument('--destvi-luca', type=Path, help='LuCA DestVI cell_type_proportions.parquet (fine-grained)')
    parser.add_argument('--destvi-hlca', type=Path, help='HLCA DestVI cell_type_proportions.parquet (coarse)')
    parser.add_argument('--figures', action='store_true', help='Generate QC figures')
    parser.add_argument('--max-cells-per-ring', type=int, default=50)
    args = parser.parse_args()

    config = PrepConfig(max_cells_per_ring=args.max_cells_per_ring)

    summary = prepare_data(
        cells_path=args.cells,
        output_dir=args.output_dir,
        destvi_luca_path=args.destvi_luca,
        destvi_hlca_path=args.destvi_hlca,
        config=config,
        make_figures=args.figures,
    )

    print(f'\nData preparation complete: {summary}')


if __name__ == '__main__':
    main()
