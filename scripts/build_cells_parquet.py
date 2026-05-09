#!/usr/bin/env python3
"""Build cells.parquet from direct inference embeddings.

Merges spatial and snRNA embeddings with metadata from h5ad files.
This replaces the generate_cells_table() function from complete_data_prep.py
for the new direct inference workflow where BOTH modalities go through
the same frozen HLCA/LuCA models.

Output matches CELLS_SCHEMA from contracts.py.

Usage:
    python scripts/build_cells_parquet.py \
        --spatial-emb $DATA/processed/luad_evo/reference_embeddings/spatial_fused_direct.parquet \
        --snrna-emb $DATA/processed/luad_evo/reference_embeddings/snrna_fused_direct.parquet \
        --spatial-h5ad $DATA/processed/luad_evo/spatial_merged.h5ad \
        --snrna-h5ad $DATA/processed/luad_evo/snrna_with_celltypes.h5ad \
        --output $DATA/processed/luad_evo/canonical/cells.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm

from stagebridge.contracts import (
    HLCA_DIM,
    LUCA_DIM,
    LATENT_DIM,
    STAGES_3,
    STAGES_5,
    STAGE_5_TO_3,
    STAGE_TO_IDX,
    CELLS_SCHEMA,
    WES_COLS,
)


def extract_stage_from_id(cell_id: str) -> str:
    """Extract stage from cell_id pattern (e.g., 'AAH_sample1_cell1' -> 'AAH')."""
    for stage in STAGES_5:
        if stage in cell_id:
            return stage
    return 'unknown'


def map_stage_to_3(stage: str) -> str:
    """Map 5-stage to 3-stage."""
    if stage in STAGE_5_TO_3:
        return STAGE_5_TO_3[stage]
    elif stage in STAGES_3:
        return stage
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Build cells.parquet from direct inference embeddings")
    parser.add_argument("--spatial-emb", type=Path, required=True,
                        help="Path to spatial_fused_direct.parquet")
    parser.add_argument("--snrna-emb", type=Path, required=True,
                        help="Path to snrna_fused_direct.parquet")
    parser.add_argument("--spatial-h5ad", type=Path, required=True,
                        help="Path to spatial h5ad for metadata and coordinates")
    parser.add_argument("--snrna-h5ad", type=Path, required=True,
                        help="Path to snRNA h5ad for metadata")
    parser.add_argument("--wes", type=Path, default=None,
                        help="Path to WES features parquet (optional)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for cells.parquet")
    parser.add_argument("--stage-system", choices=["3", "5"], default="3",
                        help="Stage system to use (default: 3)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    spatial_emb = pd.read_parquet(args.spatial_emb)
    snrna_emb = pd.read_parquet(args.snrna_emb)
    print(f"  Spatial embeddings: {len(spatial_emb):,}")
    print(f"  snRNA embeddings: {len(snrna_emb):,}")

    # Identify embedding columns
    hlca_cols = sorted([c for c in spatial_emb.columns if c.startswith('hlca_latent_')])
    luca_cols = sorted([c for c in spatial_emb.columns if c.startswith('luca_latent_')])

    hlca_dim = len(hlca_cols)
    luca_dim = len(luca_cols)
    fused_dim = hlca_dim + luca_dim

    print(f"  HLCA dim: {hlca_dim}, LuCA dim: {luca_dim}, Fused dim: {fused_dim}")

    # Load WES features if provided
    wes_df = None
    if args.wes and args.wes.exists():
        print(f"\nLoading WES features from {args.wes}...")
        wes_df = pd.read_parquet(args.wes)
        print(f"  WES records: {len(wes_df):,}")

    # =========================================================================
    # Process snRNA cells
    # =========================================================================
    print("\nProcessing snRNA cells...")
    snrna_adata = ad.read_h5ad(args.snrna_h5ad, backed='r')
    snrna_obs = snrna_adata.obs.copy()
    snrna_adata.file.close()

    # Build snRNA dataframe
    snrna_df = pd.DataFrame(index=snrna_emb.index)
    snrna_df['cell_id'] = 'snrna_' + snrna_df.index.astype(str)
    snrna_df['data_type'] = 'snrna'

    # Add embeddings - rename to contract format
    for i, col in enumerate(hlca_cols):
        snrna_df[f'z_hlca_{i}'] = snrna_emb[col].values
    for i, col in enumerate(luca_cols):
        snrna_df[f'z_luca_{i}'] = snrna_emb[col].values
    # Fused = concat of hlca + luca
    for i in range(hlca_dim):
        snrna_df[f'z_fused_{i}'] = snrna_emb[hlca_cols[i]].values
    for i in range(luca_dim):
        snrna_df[f'z_fused_{hlca_dim + i}'] = snrna_emb[luca_cols[i]].values

    # Add cell type predictions from model
    if 'cell_type_hlca' in snrna_emb.columns:
        snrna_df['cell_type_hlca'] = snrna_emb['cell_type_hlca'].values
    if 'cell_type_luca' in snrna_emb.columns:
        snrna_df['cell_type_luca'] = snrna_emb['cell_type_luca'].values

    # Add metadata from h5ad
    # Handle index alignment carefully
    common_idx = snrna_emb.index.intersection(snrna_obs.index)
    if len(common_idx) < len(snrna_emb):
        print(f"  WARNING: {len(snrna_emb) - len(common_idx)} cells not in h5ad obs, using positional")

    # Stage
    if 'stage' in snrna_obs.columns:
        if len(common_idx) == len(snrna_emb):
            snrna_df['stage'] = snrna_obs.loc[snrna_emb.index, 'stage'].values
        else:
            snrna_df['stage'] = snrna_obs['stage'].iloc[:len(snrna_emb)].values
    else:
        # Try to extract from cell_id
        snrna_df['stage'] = snrna_df.index.map(extract_stage_from_id)

    # Map to 3-stage if requested
    if args.stage_system == "3":
        snrna_df['stage'] = snrna_df['stage'].apply(map_stage_to_3)

    # Donor ID
    for col in ['donor_id', 'patient_id', 'sample_id', 'sample']:
        if col in snrna_obs.columns:
            if len(common_idx) == len(snrna_emb):
                snrna_df['donor_id'] = snrna_obs.loc[snrna_emb.index, col].values
            else:
                snrna_df['donor_id'] = snrna_obs[col].iloc[:len(snrna_emb)].values
            break
    if 'donor_id' not in snrna_df.columns:
        snrna_df['donor_id'] = 'unknown'

    # Cell type from h5ad (may be different from model predictions)
    if 'cell_type' in snrna_obs.columns:
        if len(common_idx) == len(snrna_emb):
            snrna_df['cell_type'] = snrna_obs.loc[snrna_emb.index, 'cell_type'].values
        else:
            snrna_df['cell_type'] = snrna_obs['cell_type'].iloc[:len(snrna_emb)].values
    else:
        # Use model prediction
        snrna_df['cell_type'] = snrna_df.get('cell_type_hlca', 'unknown')

    # snRNA has no spatial coordinates
    snrna_df['x'] = np.nan
    snrna_df['y'] = np.nan

    # Cell cycle scores (for stats token)
    for score_col in ['S_score', 'G2M_score']:
        if score_col in snrna_obs.columns:
            if len(common_idx) == len(snrna_emb):
                snrna_df[score_col] = snrna_obs.loc[snrna_emb.index, score_col].values
            else:
                snrna_df[score_col] = snrna_obs[score_col].iloc[:len(snrna_emb)].values
        else:
            snrna_df[score_col] = 0.0

    # Proliferation label (Ki67 or computed from cell cycle)
    if 'Ki67' in snrna_obs.columns:
        if len(common_idx) == len(snrna_emb):
            snrna_df['proliferation_label'] = snrna_obs.loc[snrna_emb.index, 'Ki67'].values
        else:
            snrna_df['proliferation_label'] = snrna_obs['Ki67'].iloc[:len(snrna_emb)].values
    elif 'proliferation_label' in snrna_obs.columns:
        if len(common_idx) == len(snrna_emb):
            snrna_df['proliferation_label'] = snrna_obs.loc[snrna_emb.index, 'proliferation_label'].values
        else:
            snrna_df['proliferation_label'] = snrna_obs['proliferation_label'].iloc[:len(snrna_emb)].values
    else:
        # Compute from cell cycle scores
        snrna_df['proliferation_label'] = snrna_df['S_score'] + snrna_df['G2M_score']

    # PROGENy pathway activities (14 pathways)
    progeny_pathways = ['Androgen', 'EGFR', 'Estrogen', 'Hypoxia', 'JAK-STAT', 'MAPK',
                        'NFkB', 'PI3K', 'TGFb', 'TNFa', 'Trail', 'VEGF', 'WNT', 'p53']
    pathway_cols_found = []
    for pathway in progeny_pathways:
        # Try various naming conventions
        for col_name in [pathway, f'progeny_{pathway}', f'PROGENy_{pathway}', pathway.lower()]:
            if col_name in snrna_obs.columns:
                if len(common_idx) == len(snrna_emb):
                    snrna_df[f'pathway_{pathway}'] = snrna_obs.loc[snrna_emb.index, col_name].values
                else:
                    snrna_df[f'pathway_{pathway}'] = snrna_obs[col_name].iloc[:len(snrna_emb)].values
                pathway_cols_found.append(pathway)
                break
        else:
            snrna_df[f'pathway_{pathway}'] = 0.0

    if pathway_cols_found:
        print(f"  Found PROGENy pathways: {pathway_cols_found}")

    print(f"  Processed {len(snrna_df):,} snRNA cells")

    # =========================================================================
    # Process spatial cells
    # =========================================================================
    print("\nProcessing spatial cells...")
    spatial_adata = ad.read_h5ad(args.spatial_h5ad, backed='r')
    spatial_obs = spatial_adata.obs.copy()

    # Get spatial coordinates
    spatial_coords = None
    if 'spatial' in spatial_adata.obsm:
        spatial_coords = pd.DataFrame(
            spatial_adata.obsm['spatial'][:, :2],
            index=spatial_adata.obs_names,
            columns=['x', 'y']
        )
    spatial_adata.file.close()

    # Build spatial dataframe
    spatial_df = pd.DataFrame(index=spatial_emb.index)
    spatial_df['cell_id'] = 'spatial_' + spatial_df.index.astype(str)
    spatial_df['data_type'] = 'spatial'

    # Add embeddings
    for i, col in enumerate(hlca_cols):
        spatial_df[f'z_hlca_{i}'] = spatial_emb[col].values
    for i, col in enumerate(luca_cols):
        spatial_df[f'z_luca_{i}'] = spatial_emb[col].values
    for i in range(hlca_dim):
        spatial_df[f'z_fused_{i}'] = spatial_emb[hlca_cols[i]].values
    for i in range(luca_dim):
        spatial_df[f'z_fused_{hlca_dim + i}'] = spatial_emb[luca_cols[i]].values

    # Add cell type predictions
    if 'cell_type_hlca' in spatial_emb.columns:
        spatial_df['cell_type_hlca'] = spatial_emb['cell_type_hlca'].values
    if 'cell_type_luca' in spatial_emb.columns:
        spatial_df['cell_type_luca'] = spatial_emb['cell_type_luca'].values

    # Add metadata from h5ad
    common_idx = spatial_emb.index.intersection(spatial_obs.index)

    # Stage
    if 'stage' in spatial_obs.columns:
        if len(common_idx) == len(spatial_emb):
            spatial_df['stage'] = spatial_obs.loc[spatial_emb.index, 'stage'].values
        else:
            spatial_df['stage'] = spatial_obs['stage'].iloc[:len(spatial_emb)].values
    else:
        spatial_df['stage'] = spatial_df.index.map(extract_stage_from_id)

    if args.stage_system == "3":
        spatial_df['stage'] = spatial_df['stage'].apply(map_stage_to_3)

    # Donor ID
    for col in ['donor_id', 'patient_id', 'sample_id', 'sample']:
        if col in spatial_obs.columns:
            if len(common_idx) == len(spatial_emb):
                spatial_df['donor_id'] = spatial_obs.loc[spatial_emb.index, col].values
            else:
                spatial_df['donor_id'] = spatial_obs[col].iloc[:len(spatial_emb)].values
            break
    if 'donor_id' not in spatial_df.columns:
        spatial_df['donor_id'] = 'unknown'

    # Cell type - spatial spots are mixtures, use model prediction
    spatial_df['cell_type'] = spatial_df.get('cell_type_hlca', 'mixed')

    # Spatial coordinates
    if spatial_coords is not None:
        if len(spatial_coords) == len(spatial_emb):
            spatial_df['x'] = spatial_coords.loc[spatial_emb.index, 'x'].values
            spatial_df['y'] = spatial_coords.loc[spatial_emb.index, 'y'].values
        else:
            # Index mismatch - try positional
            spatial_df['x'] = spatial_coords['x'].iloc[:len(spatial_emb)].values
            spatial_df['y'] = spatial_coords['y'].iloc[:len(spatial_emb)].values
    else:
        spatial_df['x'] = np.nan
        spatial_df['y'] = np.nan

    # Cell cycle scores (for stats token) - spatial may not have these
    for score_col in ['S_score', 'G2M_score']:
        if score_col in spatial_obs.columns:
            if len(common_idx) == len(spatial_emb):
                spatial_df[score_col] = spatial_obs.loc[spatial_emb.index, score_col].values
            else:
                spatial_df[score_col] = spatial_obs[score_col].iloc[:len(spatial_emb)].values
        else:
            spatial_df[score_col] = 0.0

    # Proliferation label
    if 'Ki67' in spatial_obs.columns:
        if len(common_idx) == len(spatial_emb):
            spatial_df['proliferation_label'] = spatial_obs.loc[spatial_emb.index, 'Ki67'].values
        else:
            spatial_df['proliferation_label'] = spatial_obs['Ki67'].iloc[:len(spatial_emb)].values
    elif 'proliferation_label' in spatial_obs.columns:
        if len(common_idx) == len(spatial_emb):
            spatial_df['proliferation_label'] = spatial_obs.loc[spatial_emb.index, 'proliferation_label'].values
        else:
            spatial_df['proliferation_label'] = spatial_obs['proliferation_label'].iloc[:len(spatial_emb)].values
    else:
        spatial_df['proliferation_label'] = spatial_df['S_score'] + spatial_df['G2M_score']

    # PROGENy pathway activities (14 pathways)
    progeny_pathways = ['Androgen', 'EGFR', 'Estrogen', 'Hypoxia', 'JAK-STAT', 'MAPK',
                        'NFkB', 'PI3K', 'TGFb', 'TNFa', 'Trail', 'VEGF', 'WNT', 'p53']
    spatial_pathway_cols = []
    for pathway in progeny_pathways:
        for col_name in [pathway, f'progeny_{pathway}', f'PROGENy_{pathway}', pathway.lower()]:
            if col_name in spatial_obs.columns:
                if len(common_idx) == len(spatial_emb):
                    spatial_df[f'pathway_{pathway}'] = spatial_obs.loc[spatial_emb.index, col_name].values
                else:
                    spatial_df[f'pathway_{pathway}'] = spatial_obs[col_name].iloc[:len(spatial_emb)].values
                spatial_pathway_cols.append(pathway)
                break
        else:
            spatial_df[f'pathway_{pathway}'] = 0.0

    if spatial_pathway_cols:
        print(f"  Found PROGENy pathways: {spatial_pathway_cols}")

    print(f"  Processed {len(spatial_df):,} spatial spots")

    # =========================================================================
    # Combine and finalize
    # =========================================================================
    print("\nCombining datasets...")
    cells_df = pd.concat([snrna_df, spatial_df], ignore_index=True)

    # Add stage_idx
    cells_df['stage_idx'] = cells_df['stage'].map(STAGE_TO_IDX).fillna(-1).astype(int)

    # Add WES features (default to 0)
    for wes_col in WES_COLS:
        cells_df[wes_col] = 0.0

    if wes_df is not None:
        # Merge WES by donor_id + stage
        print("  Merging WES features...")
        cells_df['_merge_key'] = cells_df['donor_id'].astype(str) + '_' + cells_df['stage'].astype(str)

        # Find WES ID column
        wes_id_col = None
        for col in ['donor_id', 'patient_id', 'sample_id']:
            if col in wes_df.columns:
                wes_id_col = col
                break

        if wes_id_col and 'stage' in wes_df.columns:
            wes_df['_merge_key'] = wes_df[wes_id_col].astype(str) + '_' + wes_df['stage'].astype(str)
            wes_df_dedup = wes_df.drop_duplicates('_merge_key')

            for wes_col in WES_COLS:
                if wes_col in wes_df_dedup.columns:
                    wes_lookup = wes_df_dedup.set_index('_merge_key')[wes_col]
                    cells_df[wes_col] = cells_df['_merge_key'].map(wes_lookup).fillna(0.0)

        cells_df = cells_df.drop(columns=['_merge_key'])

    # Create list columns for neighborhoods compatibility
    print("  Creating embedding list columns...")
    z_fused_cols = [f'z_fused_{i}' for i in range(fused_dim)]
    z_hlca_cols = [f'z_hlca_{i}' for i in range(hlca_dim)]
    z_luca_cols = [f'z_luca_{i}' for i in range(luca_dim)]

    cells_df['z_fused'] = cells_df[z_fused_cols].values.tolist()
    cells_df['z_hlca'] = cells_df[z_hlca_cols].values.tolist()
    cells_df['z_luca'] = cells_df[z_luca_cols].values.tolist()

    # Summary
    print(f"\nTotal cells: {len(cells_df):,}")
    print(f"  Spatial: {(cells_df['data_type'] == 'spatial').sum():,}")
    print(f"  snRNA: {(cells_df['data_type'] == 'snrna').sum():,}")

    if 'stage' in cells_df.columns:
        print(f"\nStage distribution:")
        print(cells_df['stage'].value_counts())

    if 'donor_id' in cells_df.columns:
        print(f"\nDonors: {cells_df['donor_id'].nunique()}")

    # Save
    cells_df.to_parquet(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Validate against contract
    print("\nValidating against contract...")
    errors = CELLS_SCHEMA.validate(cells_df)
    if errors:
        print(f"  Contract violations:")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  Contract validation passed!")


if __name__ == "__main__":
    main()
