#!/usr/bin/env python3
"""Unified data preparation pipeline for StageBridge training.

This is the ONE script to run before HPO/training. It handles:
1. Reference mapping (scArches) - maps snRNA/spatial to HLCA/LuCA latent space
2. Feature enrichment - cell cycle scores, proliferation labels, PROGENy pathways
3. cells.parquet generation - unified cell table with embeddings + features
4. neighborhoods.parquet generation - AMICI-format spatial neighborhoods

Usage:
    python scripts/prepare_training_data.py \
        --snrna $DATA/processed/snrna/snrna_processed.h5ad \
        --spatial $DATA/processed/spatial/spatial_merged.h5ad \
        --hlca-model $MODELS/HLCA_reference_model \
        --luca-model $MODELS/LuCA_reference_model \
        --output-dir $DATA/processed/luad_evo/canonical

Or with config file:
    python scripts/prepare_training_data.py --config workflow/config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.spatial import cKDTree
from tqdm import tqdm

from stagebridge.contracts import (
    HLCA_DIM, LUCA_DIM, LATENT_DIM, MAX_NEIGHBORS,
    STAGE_TO_IDX, STAGES_3, STAGE_5_TO_3,
    N_PROGENY_PATHWAYS, CELLS_SCHEMA, NEIGHBORHOODS_SCHEMA,
)


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

PROGENY_PATHWAYS = [
    'Androgen', 'EGFR', 'Estrogen', 'Hypoxia', 'JAK-STAT', 'MAPK',
    'NFkB', 'PI3K', 'TGFb', 'TNFa', 'Trail', 'VEGF', 'WNT', 'p53'
]


def step1_reference_mapping(
    snrna_path: Path,
    spatial_path: Path,
    hlca_model_path: Path,
    luca_model_path: Path,
    output_dir: Path,
    skip_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map snRNA and spatial data to HLCA/LuCA reference latent space via scArches."""
    print("\n" + "="*60)
    print("STEP 1: Reference Mapping (scArches)")
    print("="*60)

    snrna_emb_path = output_dir / "snrna_embeddings.parquet"
    spatial_emb_path = output_dir / "spatial_embeddings.parquet"

    if skip_if_exists and snrna_emb_path.exists() and spatial_emb_path.exists():
        print("  Embeddings already exist, loading...")
        snrna_emb = pd.read_parquet(snrna_emb_path)
        spatial_emb = pd.read_parquet(spatial_emb_path)
        print(f"  snRNA: {len(snrna_emb):,} cells")
        print(f"  Spatial: {len(spatial_emb):,} spots")
        return snrna_emb, spatial_emb

    try:
        import scarches as sca
    except ImportError:
        raise ImportError("scArches required for reference mapping. Install: pip install scarches")

    print(f"  Loading snRNA: {snrna_path}")
    snrna = sc.read_h5ad(snrna_path)
    print(f"    {snrna.n_obs:,} cells, {snrna.n_vars:,} genes")

    print(f"  Loading spatial: {spatial_path}")
    spatial = sc.read_h5ad(spatial_path)
    print(f"    {spatial.n_obs:,} spots, {spatial.n_vars:,} genes")

    def map_to_reference(adata, model_path, prefix):
        """Map query data to reference model."""
        print(f"  Mapping to {prefix} reference...")

        model = sca.models.SCANVI.load_query_data(
            adata=adata,
            reference_model=str(model_path),
            freeze_dropout=True,
        )
        model.train(max_epochs=100, plan_kwargs=dict(weight_decay=0.0))

        latent = model.get_latent_representation()
        latent_df = pd.DataFrame(
            latent,
            index=adata.obs_names,
            columns=[f"{prefix}_latent_{i}" for i in range(latent.shape[1])]
        )

        if hasattr(model, 'predict'):
            labels = model.predict()
            latent_df[f'cell_type_{prefix}'] = labels

        return latent_df

    snrna_hlca = map_to_reference(snrna, hlca_model_path, 'hlca')
    snrna_luca = map_to_reference(snrna, luca_model_path, 'luca')
    snrna_emb = snrna_hlca.join(snrna_luca)

    spatial_hlca = map_to_reference(spatial, hlca_model_path, 'hlca')
    spatial_luca = map_to_reference(spatial, luca_model_path, 'luca')
    spatial_emb = spatial_hlca.join(spatial_luca)

    output_dir.mkdir(parents=True, exist_ok=True)
    snrna_emb.to_parquet(snrna_emb_path)
    spatial_emb.to_parquet(spatial_emb_path)

    print(f"  Saved: {snrna_emb_path}")
    print(f"  Saved: {spatial_emb_path}")

    return snrna_emb, spatial_emb


def step2_enrich_features(
    snrna_path: Path,
    spatial_path: Path,
    output_dir: Path,
    skip_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cell cycle, proliferation, and pathway features."""
    print("\n" + "="*60)
    print("STEP 2: Feature Enrichment")
    print("="*60)

    snrna_features_path = output_dir / "snrna_features.parquet"
    spatial_features_path = output_dir / "spatial_features.parquet"

    if skip_if_exists and snrna_features_path.exists() and spatial_features_path.exists():
        print("  Features already exist, loading...")
        snrna_features = pd.read_parquet(snrna_features_path)
        spatial_features = pd.read_parquet(spatial_features_path)
        return snrna_features, spatial_features

    def compute_features(adata_path: Path, data_type: str) -> pd.DataFrame:
        print(f"  Processing {data_type}: {adata_path}")
        adata = sc.read_h5ad(adata_path)
        print(f"    {adata.n_obs:,} cells, {adata.n_vars:,} genes")

        features = pd.DataFrame(index=adata.obs_names)

        s_genes = [g for g in S_GENES if g in adata.var_names]
        g2m_genes = [g for g in G2M_GENES if g in adata.var_names]

        if len(s_genes) >= 5 and len(g2m_genes) >= 5:
            print(f"    Computing cell cycle ({len(s_genes)} S, {len(g2m_genes)} G2M genes)...")
            sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
            features['S_score'] = adata.obs['S_score'].values
            features['G2M_score'] = adata.obs['G2M_score'].values
            features['phase'] = adata.obs['phase'].astype(str).values
        else:
            print(f"    WARNING: Insufficient cell cycle genes, setting to 0")
            features['S_score'] = 0.0
            features['G2M_score'] = 0.0
            features['phase'] = 'unknown'

        if 'MKI67' in adata.var_names:
            mki67 = adata[:, 'MKI67'].X
            if hasattr(mki67, 'toarray'):
                mki67 = mki67.toarray().flatten()
            else:
                mki67 = np.array(mki67).flatten()
            threshold = np.median(mki67[mki67 > 0]) if (mki67 > 0).any() else 0
            features['proliferation_label'] = (mki67 > threshold).astype(float)
            print(f"    Proliferation: {(features['proliferation_label'] > 0).sum():,} cells")
        else:
            features['proliferation_label'] = (features['S_score'] > 0.5).astype(float)

        print(f"    Computing PROGENy pathways...")
        try:
            import decoupler as dc
            progeny = dc.op.progeny(organism='human')
            dc.mt.ulm(data=adata, net=progeny)
            if 'score_ulm' in adata.obsm:
                pathway_acts = dc.pp.get_obsm(adata=adata, key='score_ulm')
                for pathway in PROGENY_PATHWAYS:
                    col = f'pathway_{pathway}'
                    if pathway in pathway_acts.columns:
                        features[col] = pathway_acts[pathway].values
                    else:
                        features[col] = 0.0
            else:
                for pathway in PROGENY_PATHWAYS:
                    features[f'pathway_{pathway}'] = 0.0
        except Exception as e:
            print(f"    WARNING: PROGENy failed ({e}), using zeros")
            for pathway in PROGENY_PATHWAYS:
                features[f'pathway_{pathway}'] = 0.0

        return features

    snrna_features = compute_features(snrna_path, 'snRNA')
    spatial_features = compute_features(spatial_path, 'spatial')

    output_dir.mkdir(parents=True, exist_ok=True)
    snrna_features.to_parquet(snrna_features_path)
    spatial_features.to_parquet(spatial_features_path)

    return snrna_features, spatial_features


def step3_build_cells_parquet(
    snrna_path: Path,
    spatial_path: Path,
    snrna_emb: pd.DataFrame,
    spatial_emb: pd.DataFrame,
    snrna_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
    output_path: Path,
    destvi_fractions: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build unified cells.parquet with embeddings and features."""
    print("\n" + "="*60)
    print("STEP 3: Build cells.parquet")
    print("="*60)

    snrna = ad.read_h5ad(snrna_path, backed='r')
    spatial = ad.read_h5ad(spatial_path, backed='r')

    records = []

    hlca_cols = [c for c in snrna_emb.columns if c.startswith('hlca_latent_')]
    luca_cols = [c for c in snrna_emb.columns if c.startswith('luca_latent_')]
    hlca_dim = len(hlca_cols)
    luca_dim = len(luca_cols)
    fused_dim = hlca_dim + luca_dim

    print(f"  Embedding dims: HLCA={hlca_dim}, LuCA={luca_dim}, Fused={fused_dim}")

    print(f"  Processing snRNA cells...")
    for cell_id in tqdm(snrna.obs_names, desc="  snRNA"):
        obs = snrna.obs.loc[cell_id]

        donor_id = obs.get('donor_id', obs.get('patient_id', 'unknown'))
        stage = extract_stage(cell_id, str(donor_id))

        z_hlca = snrna_emb.loc[cell_id, hlca_cols].values.astype(np.float32) if cell_id in snrna_emb.index else np.zeros(hlca_dim, dtype=np.float32)
        z_luca = snrna_emb.loc[cell_id, luca_cols].values.astype(np.float32) if cell_id in snrna_emb.index else np.zeros(luca_dim, dtype=np.float32)
        z_fused = np.concatenate([z_hlca, z_luca])

        feat = snrna_features.loc[cell_id] if cell_id in snrna_features.index else {}

        record = {
            'cell_id': cell_id,
            'donor_id': str(donor_id),
            'stage': stage,
            'stage_idx': STAGE_TO_IDX.get(stage, -1),
            'data_type': 'snrna',
            'cell_type': str(obs.get('cell_type', 'unknown')),
            'z_fused': z_fused.tolist(),
            'z_hlca': z_hlca.tolist(),
            'z_luca': z_luca.tolist(),
            'S_score': float(feat.get('S_score', 0.0)),
            'G2M_score': float(feat.get('G2M_score', 0.0)),
            'proliferation_label': float(feat.get('proliferation_label', 0.0)),
            # Biological features
            'emt_score': float(feat.get('emt_score', 0.0)) if not pd.isna(feat.get('emt_score', 0.0)) else 0.0,
            'senescence_score': float(feat.get('senescence_score', 0.0)) if not pd.isna(feat.get('senescence_score', 0.0)) else 0.0,
            'sasp_score': float(feat.get('sasp_score', 0.0)) if not pd.isna(feat.get('sasp_score', 0.0)) else 0.0,
            'cytotrace': float(feat.get('cytotrace', 0.0)) if not pd.isna(feat.get('cytotrace', 0.0)) else 0.0,
            # snRNA doesn't have spatial deconvolution
            'caf_fraction': 0.0,
            'immune_fraction': 0.0,
            'diversity': 0.0,
        }

        for pathway in PROGENY_PATHWAYS:
            record[f'pathway_{pathway}'] = float(feat.get(f'pathway_{pathway}', 0.0)) if not pd.isna(feat.get(f'pathway_{pathway}', 0.0)) else 0.0

        records.append(record)

    snrna.file.close()

    print(f"  Processing spatial spots...")
    spatial_coords = spatial.obsm['spatial'][:, :2] if 'spatial' in spatial.obsm else None

    for i, cell_id in enumerate(tqdm(spatial.obs_names, desc="  Spatial")):
        obs = spatial.obs.loc[cell_id]

        donor_id = obs.get('donor_id', obs.get('patient_id', 'unknown'))
        stage = extract_stage(cell_id, str(donor_id))

        z_hlca = spatial_emb.loc[cell_id, hlca_cols].values.astype(np.float32) if cell_id in spatial_emb.index else np.zeros(hlca_dim, dtype=np.float32)
        z_luca = spatial_emb.loc[cell_id, luca_cols].values.astype(np.float32) if cell_id in spatial_emb.index else np.zeros(luca_dim, dtype=np.float32)
        z_fused = np.concatenate([z_hlca, z_luca])

        feat = spatial_features.loc[cell_id] if cell_id in spatial_features.index else {}

        # Get DestVI fractions for this spot
        caf_frac = 0.0
        immune_frac = 0.0
        diversity = 0.0
        cell_type_luca = 'mixed'
        if destvi_fractions is not None and cell_id in destvi_fractions.index:
            caf_frac = float(destvi_fractions.loc[cell_id, 'caf_fraction'])
            immune_frac = float(destvi_fractions.loc[cell_id, 'immune_fraction'])
            diversity = float(destvi_fractions.loc[cell_id, 'diversity'])
            cell_type_luca = str(destvi_fractions.loc[cell_id, 'cell_type_luca'])

        record = {
            'cell_id': f'spatial_{cell_id}',
            'donor_id': str(donor_id),
            'stage': stage,
            'stage_idx': STAGE_TO_IDX.get(stage, -1),
            'data_type': 'spatial',
            'cell_type': cell_type_luca,
            'z_fused': z_fused.tolist(),
            'z_hlca': z_hlca.tolist(),
            'z_luca': z_luca.tolist(),
            'S_score': float(feat.get('S_score', 0.0)),
            'G2M_score': float(feat.get('G2M_score', 0.0)),
            'proliferation_label': float(feat.get('proliferation_label', 0.0)),
            # Biological features (not available for spatial)
            'emt_score': 0.0,
            'senescence_score': 0.0,
            'sasp_score': 0.0,
            'cytotrace': 0.0,
            # DestVI fractions
            'caf_fraction': caf_frac,
            'immune_fraction': immune_frac,
            'diversity': diversity,
        }

        if spatial_coords is not None:
            record['x'] = float(spatial_coords[i, 0])
            record['y'] = float(spatial_coords[i, 1])

        for pathway in PROGENY_PATHWAYS:
            record[f'pathway_{pathway}'] = float(feat.get(f'pathway_{pathway}', 0.0)) if not pd.isna(feat.get(f'pathway_{pathway}', 0.0)) else 0.0

        records.append(record)

    spatial.file.close()

    cells_df = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cells_df.to_parquet(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Total cells: {len(cells_df):,}")
    print(f"  snRNA: {(cells_df['data_type'] == 'snrna').sum():,}")
    print(f"  Spatial: {(cells_df['data_type'] == 'spatial').sum():,}")
    print(f"  Stages: {cells_df['stage'].value_counts().to_dict()}")

    return cells_df


def step4_build_neighborhoods_parquet(
    cells_df: pd.DataFrame,
    spatial_path: Path,
    output_path: Path,
    max_neighbors: int = 100,
    max_distance: float = 200.0,
) -> pd.DataFrame:
    """Build AMICI-format neighborhoods from spatial cells."""
    print("\n" + "="*60)
    print("STEP 4: Build neighborhoods.parquet (AMICI format)")
    print("="*60)

    spatial_cells = cells_df[cells_df['data_type'] == 'spatial'].copy()
    print(f"  Spatial cells: {len(spatial_cells):,}")

    if len(spatial_cells) == 0:
        print("  WARNING: No spatial cells, skipping neighborhoods")
        return pd.DataFrame()

    spatial = ad.read_h5ad(spatial_path, backed='r')
    coords_df = pd.DataFrame(
        spatial.obsm['spatial'][:, :2],
        index=spatial.obs_names,
        columns=['x', 'y']
    )
    spatial.file.close()

    donors = spatial_cells['donor_id'].unique()
    print(f"  Donors: {len(donors)}")
    print(f"  Max neighbors: {max_neighbors}")
    print(f"  Max distance: {max_distance} microns")

    all_records = []

    for donor_id in tqdm(donors, desc="  Building neighborhoods"):
        donor_mask = spatial_cells['donor_id'] == donor_id
        donor_cells = spatial_cells[donor_mask].copy().reset_index(drop=True)

        if len(donor_cells) < 2:
            continue

        original_ids = donor_cells['cell_id'].str.replace('spatial_', '', regex=False)
        coords = coords_df.loc[original_ids, ['x', 'y']].values

        tree = cKDTree(coords)

        for idx in range(len(donor_cells)):
            row = donor_cells.iloc[idx]
            center = coords[idx]

            neighbor_indices = tree.query_ball_point(center, max_distance)
            neighbor_indices = np.array([i for i in neighbor_indices if i != idx])

            if len(neighbor_indices) == 0:
                continue

            neighbor_coords = coords[neighbor_indices]
            distances = np.sqrt(np.sum((neighbor_coords - center) ** 2, axis=1))

            sort_idx = np.argsort(distances)
            neighbor_indices = neighbor_indices[sort_idx][:max_neighbors]
            distances = distances[sort_idx][:max_neighbors]

            neighbor_cells = []
            for nidx in neighbor_indices:
                neighbor_cells.append(donor_cells.iloc[nidx]['z_fused'])

            pathway_values = [float(row.get(f'pathway_{p}', 0.0)) for p in PROGENY_PATHWAYS]

            record = {
                'cell_id': row['cell_id'],
                'donor_id': row['donor_id'],
                'stage': row['stage'],
                'receiver_z': row['z_fused'],
                'hlca_z': row['z_hlca'],
                'luca_z': row['z_luca'],
                'neighbor_cells': neighbor_cells,
                'neighbor_distances': distances.tolist(),
                'S_score': float(row.get('S_score', 0.0)),
                'G2M_score': float(row.get('G2M_score', 0.0)),
                'proliferation_label': float(row.get('proliferation_label', 0.0)),
                'pathway_targets': pathway_values,
            }

            all_records.append(record)

    neighborhoods_df = pd.DataFrame(all_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    neighborhoods_df.to_parquet(output_path, index=False)

    print(f"\n  Saved: {output_path}")
    print(f"  Neighborhoods: {len(neighborhoods_df):,}")
    if len(neighborhoods_df) > 0:
        n_neighbors = [len(nc) for nc in neighborhoods_df['neighbor_cells']]
        print(f"  Neighbors per cell: mean={np.mean(n_neighbors):.1f}, min={np.min(n_neighbors)}, max={np.max(n_neighbors)}")

    return neighborhoods_df


def extract_stage(cell_id: str, donor_id: str) -> str:
    """Extract disease stage from cell_id or donor mapping."""
    import re
    match = re.search(r'_P\d+_([^:]+):', cell_id)
    if match:
        stage_raw = match.group(1).replace('1', '').replace('-', '').strip()
        return STAGE_5_TO_3.get(stage_raw, stage_raw)
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Unified data preparation for StageBridge")

    parser.add_argument("--snrna", type=Path, required=True, help="Path to snRNA h5ad")
    parser.add_argument("--spatial", type=Path, required=True, help="Path to spatial h5ad")
    parser.add_argument("--hlca-model", type=Path, help="Path to HLCA scArches model")
    parser.add_argument("--luca-model", type=Path, help="Path to LuCA scArches model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    parser.add_argument("--snrna-embeddings", type=Path, help="Pre-computed snRNA embeddings (skip step 1)")
    parser.add_argument("--spatial-embeddings", type=Path, help="Pre-computed spatial embeddings (skip step 1)")

    parser.add_argument("--max-neighbors", type=int, default=100, help="Max neighbors per cell")
    parser.add_argument("--max-distance", type=float, default=5000.0, help="Max neighbor distance (coordinate units)")
    parser.add_argument("--progeny-parquet", type=Path, help="Pre-computed PROGENy pathway_activity_progeny.parquet")
    parser.add_argument("--biological-features", type=Path, help="Pre-computed biological_features.parquet (EMT, senescence, SASP)")
    parser.add_argument("--destvi-luca", type=Path, help="DestVI LuCA cell_type_proportions.parquet for CAF/immune fractions")
    parser.add_argument("--progression", type=Path, help="Pre-computed progression_scores.parquet (cytotrace, pseudotime)")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("StageBridge Training Data Preparation")
    print("="*60)
    print(f"  snRNA: {args.snrna}")
    print(f"  Spatial: {args.spatial}")
    print(f"  Output: {args.output_dir}")

    if args.snrna_embeddings and args.spatial_embeddings:
        print("\n  Using pre-computed embeddings")
        snrna_emb = pd.read_parquet(args.snrna_embeddings)
        spatial_emb = pd.read_parquet(args.spatial_embeddings)
    elif args.hlca_model and args.luca_model:
        snrna_emb, spatial_emb = step1_reference_mapping(
            args.snrna, args.spatial,
            args.hlca_model, args.luca_model,
            args.output_dir,
            skip_if_exists=not args.force,
        )
    else:
        raise ValueError("Must provide either --snrna-embeddings/--spatial-embeddings OR --hlca-model/--luca-model")

    snrna_features, spatial_features = step2_enrich_features(
        args.snrna, args.spatial,
        args.output_dir,
        skip_if_exists=not args.force,
    )

    # Merge pre-computed PROGENy if provided
    if args.progeny_parquet and args.progeny_parquet.exists():
        print("\n  Merging pre-computed PROGENy pathways...")
        progeny_df = pd.read_parquet(args.progeny_parquet)
        if progeny_df.index.name != 'cell_id' and 'cell_id' not in progeny_df.columns:
            progeny_df = progeny_df.reset_index()
            progeny_df.columns = ['cell_id'] + list(progeny_df.columns[1:])
        if 'cell_id' in progeny_df.columns:
            progeny_df = progeny_df.set_index('cell_id')

        for pathway in PROGENY_PATHWAYS:
            if pathway in progeny_df.columns:
                snrna_features[f'pathway_{pathway}'] = snrna_features.index.map(progeny_df[pathway])

        n_matched = snrna_features[[f'pathway_{p}' for p in PROGENY_PATHWAYS]].notna().any(axis=1).sum()
        print(f"    Matched PROGENy for {n_matched:,} snRNA cells")

    # Merge biological features (EMT, senescence, SASP)
    if args.biological_features and args.biological_features.exists():
        print("\n  Merging biological features (EMT, senescence, SASP)...")
        bio_df = pd.read_parquet(args.biological_features)
        if 'cell_id' in bio_df.columns:
            bio_df = bio_df.set_index('cell_id')

        for col in ['emt_score', 'senescence_score', 'sasp_score', 'lr_activity_score']:
            if col in bio_df.columns:
                snrna_features[col] = snrna_features.index.map(bio_df[col])

        n_matched = snrna_features['emt_score'].notna().sum() if 'emt_score' in snrna_features.columns else 0
        print(f"    Matched biological features for {n_matched:,} snRNA cells")

    # Merge progression scores (CytoTRACE plasticity)
    if args.progression and args.progression.exists():
        print("\n  Merging progression scores (CytoTRACE)...")
        prog_df = pd.read_parquet(args.progression)
        if 'cell_id' in prog_df.columns:
            prog_df = prog_df.set_index('cell_id')

        if 'cytotrace' in prog_df.columns:
            snrna_features['cytotrace'] = snrna_features.index.map(prog_df['cytotrace'])
            n_matched = snrna_features['cytotrace'].notna().sum()
            print(f"    Matched CytoTRACE for {n_matched:,} snRNA cells")

    # Merge DestVI CAF/immune fractions for spatial cells
    destvi_fractions = None
    if args.destvi_luca and args.destvi_luca.exists():
        print("\n  Loading DestVI LuCA fractions...")
        destvi_df = pd.read_parquet(args.destvi_luca)
        print(f"    DestVI data: {len(destvi_df):,} spots")

        # Define cell type groupings
        CAF_TYPES = ['fibroblast of lung', 'bronchus fibroblast of lung', 'stromal cell']
        IMMUNE_TYPES = [
            'B cell', 'CD4-positive, alpha-beta T cell', 'CD8-positive, alpha-beta T cell',
            'regulatory T cell', 'natural killer cell', 'alveolar macrophage', 'macrophage',
            'classical monocyte', 'non-classical monocyte', 'myeloid cell', 'neutrophil',
            'dendritic cell', 'mast cell', 'plasma cell'
        ]

        # Compute fractions
        type_cols = [c for c in destvi_df.columns if c not in ['sample', 'cell_id', 'spot_id']]
        caf_cols = [c for c in type_cols if c in CAF_TYPES]
        immune_cols = [c for c in type_cols if c in IMMUNE_TYPES]

        destvi_fractions = pd.DataFrame(index=destvi_df.index)
        destvi_fractions['caf_fraction'] = destvi_df[caf_cols].sum(axis=1) if caf_cols else 0.0
        destvi_fractions['immune_fraction'] = destvi_df[immune_cols].sum(axis=1) if immune_cols else 0.0
        destvi_fractions['diversity'] = (destvi_df[type_cols] > 0.01).sum(axis=1)  # Number of cell types > 1%

        # Get dominant cell type
        destvi_fractions['cell_type_luca'] = destvi_df[type_cols].idxmax(axis=1)

        print(f"    CAF fraction: {destvi_fractions['caf_fraction'].mean():.3f} mean")
        print(f"    Immune fraction: {destvi_fractions['immune_fraction'].mean():.3f} mean")

    cells_path = args.output_dir / "cells.parquet"
    cells_df = step3_build_cells_parquet(
        args.snrna, args.spatial,
        snrna_emb, spatial_emb,
        snrna_features, spatial_features,
        cells_path,
        destvi_fractions=destvi_fractions,
    )

    neighborhoods_path = args.output_dir / "neighborhoods.parquet"
    neighborhoods_df = step4_build_neighborhoods_parquet(
        cells_df,
        args.spatial,
        neighborhoods_path,
        max_neighbors=args.max_neighbors,
        max_distance=args.max_distance,
    )

    manifest = {
        "n_cells": len(cells_df),
        "n_neighborhoods": len(neighborhoods_df),
        "n_donors": int(cells_df['donor_id'].nunique()),
        "stages": sorted(cells_df['stage'].unique().tolist()),
        "files": {
            "cells": "cells.parquet",
            "neighborhoods": "neighborhoods.parquet",
        }
    }
    with open(args.output_dir / "data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"  Output: {args.output_dir}")
    print(f"  Ready for: python -m stagebridge.pipelines.run_hpo --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
