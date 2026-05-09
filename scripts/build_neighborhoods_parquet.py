#!/usr/bin/env python3
"""Build neighborhoods.parquet from cells.parquet and spatial coordinates.

Creates receiver-centered neighborhoods for SPATIAL cells in AMICI format:
- neighbor_cells: flat list of neighbor embeddings sorted by distance
- neighbor_distances: corresponding distances

This is the preferred format for continuous distance attention (AMICI-style).

Usage:
    python scripts/build_neighborhoods_parquet.py \
        --cells $DATA/processed/luad_evo/canonical/cells.parquet \
        --spatial-h5ad $DATA/processed/luad_evo/spatial_merged.h5ad \
        --output $DATA/processed/luad_evo/canonical/neighborhoods.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from tqdm import tqdm

from stagebridge.contracts import (
    HLCA_DIM,
    LUCA_DIM,
    LATENT_DIM,
    STAGE_TO_IDX,
    MAX_NEIGHBORS,
    NEIGHBORHOODS_SCHEMA,
)


def build_neighborhoods_amici(
    cells_df: pd.DataFrame,
    coords: np.ndarray,
    max_neighbors: int = 100,
    max_distance: float = 200.0,
) -> pd.DataFrame:
    """Build neighborhood records in AMICI format (flat neighbor list).

    Args:
        cells_df: DataFrame with cell embeddings and metadata
        coords: (N, 2) array of spatial coordinates
        max_neighbors: Maximum neighbors per cell
        max_distance: Maximum distance to consider (microns)

    Returns:
        DataFrame with AMICI-format neighborhood records
    """
    n_cells = len(cells_df)

    # Identify embedding columns
    z_hlca_cols = sorted([c for c in cells_df.columns if c.startswith('z_hlca_')])
    z_luca_cols = sorted([c for c in cells_df.columns if c.startswith('z_luca_')])
    z_fused_cols = sorted([c for c in cells_df.columns if c.startswith('z_fused_')])

    # Fall back to list columns if individual columns don't exist
    use_list_cols = len(z_fused_cols) == 0 and 'z_fused' in cells_df.columns

    hlca_dim = len(z_hlca_cols) if z_hlca_cols else HLCA_DIM
    luca_dim = len(z_luca_cols) if z_luca_cols else LUCA_DIM
    fused_dim = len(z_fused_cols) if z_fused_cols else LATENT_DIM

    print(f"  Max neighbors: {max_neighbors}")
    print(f"  Max distance: {max_distance} microns")
    print(f"  HLCA dim: {hlca_dim}, LuCA dim: {luca_dim}, Fused dim: {fused_dim}")

    # Extract embedding matrices for fast lookup
    if use_list_cols:
        Z_fused = np.array(cells_df['z_fused'].tolist(), dtype=np.float32)
        Z_hlca = np.array(cells_df['z_hlca'].tolist(), dtype=np.float32)
        Z_luca = np.array(cells_df['z_luca'].tolist(), dtype=np.float32)
    else:
        Z_fused = cells_df[z_fused_cols].values.astype(np.float32)
        Z_hlca = cells_df[z_hlca_cols].values.astype(np.float32)
        Z_luca = cells_df[z_luca_cols].values.astype(np.float32)

    # Build KD-tree
    print("  Building KD-tree...")
    tree = cKDTree(coords)

    # Build neighborhoods
    records = []

    print(f"  Processing {n_cells:,} cells...")
    for idx in tqdm(range(n_cells)):
        row = cells_df.iloc[idx]
        center_coord = coords[idx]

        # Query neighbors within max_distance
        neighbor_indices = tree.query_ball_point(center_coord, max_distance)
        neighbor_indices = np.array(neighbor_indices)

        if len(neighbor_indices) == 0:
            neighbor_indices = np.array([idx])  # At least self

        # Compute distances
        neighbor_coords = coords[neighbor_indices]
        distances = np.sqrt(np.sum((neighbor_coords - center_coord) ** 2, axis=1))

        # Remove self (distance ~0)
        mask = distances > 1e-6
        neighbor_indices = neighbor_indices[mask]
        distances = distances[mask]

        # Sort by distance
        sort_idx = np.argsort(distances)
        neighbor_indices = neighbor_indices[sort_idx]
        distances = distances[sort_idx]

        # Truncate to max_neighbors
        if len(neighbor_indices) > max_neighbors:
            neighbor_indices = neighbor_indices[:max_neighbors]
            distances = distances[:max_neighbors]

        # Get neighbor embeddings
        neighbor_cells = [Z_fused[nidx].tolist() for nidx in neighbor_indices]
        neighbor_distances = distances.tolist()

        record = {
            'cell_id': row['cell_id'],
            'donor_id': row.get('donor_id', 'unknown'),
            'stage': row.get('stage', 'unknown'),

            # Receiver embeddings (as lists for parquet)
            'receiver_z': Z_fused[idx].tolist(),
            'hlca_z': Z_hlca[idx].tolist(),
            'luca_z': Z_luca[idx].tolist(),

            # AMICI format: flat neighbor list sorted by distance
            'neighbor_cells': neighbor_cells,
            'neighbor_distances': neighbor_distances,

            # Count for debugging
            'n_neighbors': len(neighbor_cells),
        }

        # Optional metadata
        if 'cell_type' in cells_df.columns:
            record['cell_type'] = row['cell_type']
        if 'cell_type_hlca' in cells_df.columns:
            record['cell_type_hlca'] = row['cell_type_hlca']
        if 'cell_type_luca' in cells_df.columns:
            record['cell_type_luca'] = row['cell_type_luca']

        # Training targets
        if 'proliferation_label' in cells_df.columns:
            record['proliferation_label'] = float(row.get('proliferation_label', 0.0))

        # Cell cycle scores (for stats token)
        if 'S_score' in cells_df.columns:
            record['S_score'] = float(row.get('S_score', 0.0))
        if 'G2M_score' in cells_df.columns:
            record['G2M_score'] = float(row.get('G2M_score', 0.0))

        # PROGENy pathway targets (14 pathways as a list)
        progeny_pathways = ['Androgen', 'EGFR', 'Estrogen', 'Hypoxia', 'JAK-STAT', 'MAPK',
                            'NFkB', 'PI3K', 'TGFb', 'TNFa', 'Trail', 'VEGF', 'WNT', 'p53']
        pathway_values = []
        for pathway in progeny_pathways:
            col = f'pathway_{pathway}'
            if col in cells_df.columns:
                pathway_values.append(float(row.get(col, 0.0)))
            else:
                pathway_values.append(0.0)
        record['pathway_targets'] = pathway_values

        records.append(record)

    return pd.DataFrame(records)


def build_neighborhoods_per_donor(
    cells_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    max_neighbors: int = 100,
    max_distance: float = 200.0,
) -> pd.DataFrame:
    """Build neighborhoods PER DONOR to prevent cross-donor leakage.

    Spatial coordinates can overlap across donors (different tissue sections),
    so we must build k-NN separately for each donor.

    Args:
        cells_df: DataFrame with cell embeddings and metadata (spatial only)
        coords_df: DataFrame with x,y coordinates indexed by original cell id
        max_neighbors: Maximum neighbors per cell
        max_distance: Maximum distance (microns)

    Returns:
        DataFrame with AMICI-format neighborhood records
    """
    donors = cells_df['donor_id'].unique()
    print(f"  Building per-donor neighborhoods for {len(donors)} donors...")

    all_records = []

    for donor_id in tqdm(donors, desc="  Donors"):
        donor_mask = cells_df['donor_id'] == donor_id
        donor_cells = cells_df[donor_mask].copy().reset_index(drop=True)

        if len(donor_cells) < 2:
            print(f"    Warning: Donor {donor_id} has only {len(donor_cells)} cells, skipping")
            continue

        # Get coordinates for this donor's cells
        # cell_id format is "spatial_<original_id>"
        original_ids = donor_cells['cell_id'].str.replace('spatial_', '', regex=False)
        coords = coords_df.loc[original_ids, ['x', 'y']].values

        # Build neighborhoods for this donor
        donor_neighborhoods = build_neighborhoods_amici(
            donor_cells,
            coords,
            max_neighbors=max_neighbors,
            max_distance=max_distance,
        )

        all_records.append(donor_neighborhoods)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Build neighborhoods.parquet in AMICI format")
    parser.add_argument("--cells", type=Path, required=True,
                        help="Path to cells.parquet")
    parser.add_argument("--spatial-h5ad", type=Path, required=True,
                        help="Path to spatial h5ad for coordinates")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for neighborhoods.parquet")
    parser.add_argument("--max-neighbors", type=int, default=100,
                        help="Maximum neighbors per cell (default: 100)")
    parser.add_argument("--max-distance", type=float, default=200.0,
                        help="Maximum neighbor distance in microns (default: 200)")
    parser.add_argument("--no-per-donor", action="store_true",
                        help="Build global k-NN instead of per-donor (NOT RECOMMENDED)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load cells
    print("Loading cells.parquet...")
    cells_df = pd.read_parquet(args.cells)
    print(f"  Total cells: {len(cells_df):,}")

    # Filter to spatial cells only
    spatial_mask = cells_df['data_type'] == 'spatial'
    spatial_cells = cells_df[spatial_mask].copy().reset_index(drop=True)
    print(f"  Spatial cells: {len(spatial_cells):,}")

    if len(spatial_cells) == 0:
        raise ValueError("No spatial cells found in cells.parquet")

    # Load coordinates from h5ad
    print("\nLoading spatial coordinates...")
    adata = ad.read_h5ad(args.spatial_h5ad, backed='r')

    if 'spatial' not in adata.obsm:
        raise ValueError("No 'spatial' key in adata.obsm")

    coords_df = pd.DataFrame(
        adata.obsm['spatial'][:, :2],
        index=adata.obs_names,
        columns=['x', 'y']
    )
    adata.file.close()

    # Verify coordinate coverage
    original_ids = spatial_cells['cell_id'].str.replace('spatial_', '', regex=False)
    missing = ~original_ids.isin(coords_df.index)
    if missing.any():
        n_missing = missing.sum()
        print(f"  WARNING: {n_missing} cells missing coordinates, dropping them")
        spatial_cells = spatial_cells[~missing].reset_index(drop=True)
        original_ids = spatial_cells['cell_id'].str.replace('spatial_', '', regex=False)

    print(f"  Coordinates available for {len(spatial_cells):,} cells")

    # Build neighborhoods
    print("\nBuilding neighborhoods (AMICI format)...")

    if args.no_per_donor:
        print("  WARNING: Building global k-NN (risk of cross-donor leakage)")
        coords = coords_df.loc[original_ids, ['x', 'y']].values
        neighborhoods_df = build_neighborhoods_amici(
            spatial_cells,
            coords,
            max_neighbors=args.max_neighbors,
            max_distance=args.max_distance,
        )
    else:
        neighborhoods_df = build_neighborhoods_per_donor(
            spatial_cells,
            coords_df,
            max_neighbors=args.max_neighbors,
            max_distance=args.max_distance,
        )

    # Summary stats
    print(f"\nNeighborhoods: {len(neighborhoods_df):,}")
    print(f"  Mean neighbors: {neighborhoods_df['n_neighbors'].mean():.1f}")
    print(f"  Min neighbors: {neighborhoods_df['n_neighbors'].min()}")
    print(f"  Max neighbors: {neighborhoods_df['n_neighbors'].max()}")

    if 'stage' in neighborhoods_df.columns:
        print(f"\nStage distribution:")
        print(neighborhoods_df['stage'].value_counts())

    if 'donor_id' in neighborhoods_df.columns:
        print(f"\nDonors: {neighborhoods_df['donor_id'].nunique()}")

    # Drop helper column before saving
    neighborhoods_df = neighborhoods_df.drop(columns=['n_neighbors'], errors='ignore')

    # Save
    neighborhoods_df.to_parquet(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Validate against contract schema
    print("\nValidating against contract schema...")
    errors = NEIGHBORHOODS_SCHEMA.validate(neighborhoods_df)
    if errors:
        print(f"  Contract violations:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  Contract validation passed!")

    # Verify AMICI format columns specifically
    amici_required = ['cell_id', 'donor_id', 'stage', 'receiver_z', 'hlca_z', 'luca_z', 'neighbor_cells', 'neighbor_distances']
    missing_cols = [c for c in amici_required if c not in neighborhoods_df.columns]
    if missing_cols:
        print(f"  WARNING: Missing AMICI format columns: {missing_cols}")
    else:
        print("  All AMICI format columns present!")


if __name__ == "__main__":
    main()
