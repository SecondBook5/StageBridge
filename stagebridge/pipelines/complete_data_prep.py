#!/usr/bin/env python3
"""
Complete Real Data Pipeline for StageBridge V1

DEPRECATED: Use scripts/prepare_training_data.py instead.

This module is kept for backward compatibility but will be removed in a future version.
The new unified script handles all data preparation in one command:

    python scripts/prepare_training_data.py \\
        --snrna $DATA/snrna.h5ad \\
        --spatial $DATA/spatial.h5ad \\
        --snrna-embeddings $DATA/snrna_emb.parquet \\
        --spatial-embeddings $DATA/spatial_emb.parquet \\
        --output-dir $DATA/canonical

---

Original description:
Completes all missing pieces from run_data_prep.py:
1. Generate canonical artifacts (cells.parquet, neighborhoods.parquet, etc.)
2. Integrate spatial backend results
3. Build 9-token niche structure
4. Generate donor-held-out CV splits
5. Extract WES features properly
6. Integrate clonal evolution patterns for H3 validation
"""
import warnings
warnings.warn(
    "stagebridge.pipelines.complete_data_prep is deprecated. "
    "Use scripts/prepare_training_data.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import json
import yaml
from tqdm import tqdm
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from stagebridge.utils.data_cache import get_data_cache
from stagebridge.biology.pathway_targets import (
    compute_proliferation_targets,
    compute_pathway_raw,
    PROGENY_PATHWAYS,  # Dict with actual gene lists for scoring
)
from stagebridge.contracts import (
    # Stage definitions
    STAGES_3,
    STAGES_5,
    STAGE_5_TO_3,
    STAGE_TO_IDX,
    # Latent dimensions
    HLCA_DIM,
    LUCA_DIM,
    LATENT_DIM,
    # Feature definitions
    WES_COLS,
    N_PROGENY_PATHWAYS,
    STATS_TOKEN_COLUMNS,
    # Data types
    DATA_TYPES,
)

# Alias for backward compatibility
WES_FEATURE_COLS = WES_COLS
STAGE_MAP_3 = STAGE_5_TO_3
CANONICAL_STAGES_3 = list(STAGES_3)

# Clonal pattern encoding for model input
CLONAL_PATTERN_ENCODING = {
    "1a": 0,      # Direct lineage (precursor -> LUAD)
    "1b": 1,      # Branched evolution (shared + stage-specific clones)
    "2": 2,       # Independent origins (no shared clones)
    "stable": 3,  # Chromosomally stable
    "uncategorized": -1,
    "unknown": -1,
}


def generate_canonical_artifacts(
    snrna_path: Path,
    spatial_path: Path,
    wes_features_path: Path,
    output_dir: Path,
    stage_definitions: dict[str, list[str]],
    n_folds: int = 5,
    reference_geometry_dir: Path | None = None,
    clonal_patterns_path: Path | None = None,
    spatial_embeddings_path: Path | None = None,
    hlca_deconv_dir: Path | None = None,
    luca_deconv_dir: Path | None = None,
):
    """
    Generate all canonical artifacts for StageBridge V1.

    Inputs:
        - snrna_merged.h5ad (from run_data_prep.py)
        - spatial_merged.h5ad (from run_data_prep.py)
        - wes_features.parquet (from run_data_prep.py)
        - spatial_embeddings (from scArches mapping - RECOMMENDED)
        - reference_geometry outputs (fused_embedding.parquet, etc.)
        - clonal_patterns.json (from run_clonal_extraction.py) [optional]

    Outputs:
        - cells.parquet (with clonal_pattern column if clonal data provided)
        - neighborhoods.parquet
        - stage_edges.parquet
        - split_manifest.json
        - feature_spec.yaml
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Canonical Artifacts")
    print("=" * 80)

    # Load data (OPTIMIZED: Use cache for parquet files)
    print("\n[1/6] Loading data...")
    cache = get_data_cache()
    snrna = ad.read_h5ad(snrna_path)
    spatial = ad.read_h5ad(spatial_path)
    wes_df = cache.read_parquet(wes_features_path) if wes_features_path.exists() else None

    # NOTE: Gamma and proportions loaded from hlca_deconv_dir/luca_deconv_dir below
    gamma_df = None  # For backward compat (averaged gamma for storage in cells.parquet)

    print(f"  snRNA: {snrna.shape[0]} cells")
    print(f"  Spatial: {spatial.shape[0]} spots")
    print(f"  WES: {len(wes_df) if wes_df is not None else 0} samples")

    # Load clonal patterns for H3 validation (optional)
    clonal_patterns = None
    if clonal_patterns_path is not None and clonal_patterns_path.exists():
        print(f"  Loading clonal patterns from {clonal_patterns_path}")
        with open(clonal_patterns_path) as f:
            clonal_patterns = json.load(f)
        print(f"  Clonal patterns: {len(clonal_patterns)} patients")
        # Log pattern distribution
        from collections import Counter
        pattern_counts = Counter(clonal_patterns.values())
        for pattern, count in sorted(pattern_counts.items()):
            print(f"    Pattern {pattern}: {count} patients")
    elif clonal_patterns_path is not None:
        print(f"  WARNING: Clonal patterns file not found: {clonal_patterns_path}")
        print("  Run: python -m stagebridge.pipelines.run_clonal_extraction --spatial-h5ad <path>")

    # Generate cells.parquet
    print("\n[2/6] Generating cells.parquet...")
    cells_df = generate_cells_table(
        snrna=snrna,
        spatial=spatial,
        wes_df=wes_df,
        stage_definitions=stage_definitions,
        reference_geometry_dir=reference_geometry_dir,
        clonal_patterns=clonal_patterns,
        spatial_embeddings_path=spatial_embeddings_path,
        hlca_deconv_dir=hlca_deconv_dir,
        luca_deconv_dir=luca_deconv_dir,
    )
    cells_df.to_parquet(output_dir / "cells.parquet", index=False)
    print(f"  Saved {len(cells_df)} cells")
    if clonal_patterns is not None:
        n_with_clonal = (cells_df["clonal_pattern"] != "unknown").sum()
        print(f"  Cells with clonal annotation: {n_with_clonal:,} ({100*n_with_clonal/len(cells_df):.1f}%)")

    # Generate neighborhoods.parquet
    print("\n[3/6] Generating neighborhoods.parquet...")
    # Load LuCA deconvolution for neighborhoods (CAF/immune fraction computation)
    spatial_deconv_luca = {}
    if luca_deconv_dir is not None and luca_deconv_dir.exists():
        samples_dir = luca_deconv_dir / "samples"
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if sample_dir.is_dir():
                    prop_path = sample_dir / "cell_type_proportions.parquet"
                    if prop_path.exists():
                        prop_df = cache.read_parquet(prop_path)
                        prop_sum = prop_df.sum(axis=1)
                        prop_norm = prop_df.div(prop_sum, axis=0).fillna(0)
                        for spot_id in prop_norm.index:
                            spatial_deconv_luca[spot_id] = prop_norm.loc[spot_id].to_dict()
    neighborhoods_df = generate_neighborhoods_table(
        cells_df=cells_df,
        spatial=spatial,
        spatial_deconv_luca=spatial_deconv_luca,
    )
    neighborhoods_df.to_parquet(output_dir / "neighborhoods.parquet", index=False)
    print(f"  Saved {len(neighborhoods_df)} neighborhoods")

    # Generate stage_edges.parquet
    print("\n[4/6] Generating stage_edges.parquet...")
    stage_edges_df = generate_stage_edges_table(stage_definitions)
    stage_edges_df.to_parquet(output_dir / "stage_edges.parquet", index=False)
    print(f"  Saved {len(stage_edges_df)} edges")

    # Generate split_manifest.json
    print("\n[5/6] Generating split_manifest.json...")
    split_manifest = generate_cv_splits(cells_df, n_folds=n_folds)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)
    print(f"  Generated {n_folds}-fold CV splits")

    # Generate feature_spec.yaml
    print("\n[6/7] Generating feature_spec.yaml...")
    feature_spec = generate_feature_spec(cells_df, neighborhoods_df)
    with open(output_dir / "feature_spec.yaml", "w") as f:
        yaml.dump(feature_spec, f)
    print("  Saved feature specifications")

    # Generate data_manifest.json
    print("\n[7/7] Generating data_manifest.json...")
    # Count spatial vs snrna by cell_id prefix (spatial cells have "spatial_" prefix)
    n_spatial = int(cells_df["cell_id"].str.startswith("spatial_").sum())
    n_snrna = len(cells_df) - n_spatial
    data_manifest = {
        "n_cells": len(cells_df),
        "n_snrna_cells": n_snrna,
        "n_spatial_spots": n_spatial,
        "n_neighborhoods": len(neighborhoods_df),
        "n_donors": int(cells_df["donor_id"].nunique()),
        "n_stages": int(cells_df["stage"].nunique()),
        "n_cell_types": int(cells_df["cell_type"].nunique()),
        "stages": sorted(cells_df["stage"].unique().tolist()),
        "n_folds": n_folds,
        "snrna_path": str(snrna_path.resolve()),  # For semi-synthetic benchmark
        "files": {
            "cells": "cells.parquet",
            "neighborhoods": "neighborhoods.parquet",
            "stage_edges": "stage_edges.parquet",
            "split_manifest": "split_manifest.json",
            "feature_spec": "feature_spec.yaml",
        },
    }
    with open(output_dir / "data_manifest.json", "w") as f:
        json.dump(data_manifest, f, indent=2)
    print("  Saved data manifest")

    print("\n" + "=" * 80)
    print(" Canonical artifacts complete!")
    print(f"  Output: {output_dir}")
    print("=" * 80)


def generate_cells_table(
    snrna: ad.AnnData,
    spatial: ad.AnnData,
    wes_df: pd.DataFrame,
    stage_definitions: dict[str, list[str]],
    reference_geometry_dir: Path | None = None,
    clonal_patterns: dict[str, str] | None = None,
    spatial_embeddings_path: Path | None = None,
    hlca_deconv_dir: Path | None = None,
    luca_deconv_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Generate cells.parquet with all required fields.

    Required columns:
    - cell_id: Unique cell identifier
    - donor_id: Donor/patient ID
    - stage: Disease stage (extracted from cell_id if not in donor mapping)
    - stage_idx: Stage index (0-4 for Normal/AAH/AIS/MIA/LUAD)
    - cell_type: Cell type annotation
    - z_fused, z_hlca, z_luca: Latent embeddings from reference geometry
    - WES features: tmb, kras_mut, egfr_mut, tp53_mut, stk11_mut, keap1_mut, smad4_mut, braf_mut
    - x, y: Spatial coordinates (for spatial cells)
    - clonal_pattern: Clonal evolution pattern (1a/1b/2/stable/unknown) [if provided]
    - clonal_pattern_idx: Numeric encoding for model input [if provided]
    """
    records = []
    cache = get_data_cache()

    # Stage order for indexing (3-stage consolidation)
    # Raw 5-stage names used for extraction, then mapped to 3-stage
    stage_order_raw = list(STAGES_5)  # From contracts.py
    stage_order = CANONICAL_STAGES_3  # ["Normal", "Preinvasive", "Invasive"]
    stage_to_idx = STAGE_TO_IDX  # From contracts.py

    # Map donors to stages (fallback, prefer cell_id extraction)
    # Apply 3-stage mapping to stage_definitions too
    donor_to_stage = {}
    for stage, donors in stage_definitions.items():
        # Map the stage definition key to 3-stage
        stage_3 = STAGE_MAP_3.get(stage, stage)
        for donor in donors:
            donor_to_stage[donor] = stage_3

    stages = CANONICAL_STAGES_3  # Always use 3-stage list

    # ==========================================================================
    # Load REAL embeddings from reference geometry (CRITICAL!)
    # Preserve actual dimensions: HLCA=30, LuCA=10, Fused=40
    # ==========================================================================
    fused_emb_df = None
    hlca_emb_df = None
    luca_emb_df = None

    # Dimensions from contracts.py (canonical source of truth)
    hlca_dim = HLCA_DIM  # 30 (HLCA scANVI latent)
    luca_dim = LUCA_DIM  # 10 (LuCA scVI latent)
    fused_dim = LATENT_DIM  # 40 (concatenated)

    if reference_geometry_dir is not None and reference_geometry_dir.exists():
        print("  Loading reference geometry embeddings...")

        fused_path = reference_geometry_dir / "fused_embedding.parquet"
        hlca_path = reference_geometry_dir / "hlca_embedding.parquet"
        luca_path = reference_geometry_dir / "luca_embedding.parquet"

        if fused_path.exists():
            fused_emb_df = cache.read_parquet(fused_path)
            if 'cell_id' in fused_emb_df.columns:
                fused_emb_df = fused_emb_df.set_index('cell_id')
            print(f"    Fused embeddings: {len(fused_emb_df):,} cells")
            # Detect actual fused dim
            fused_cols = [c for c in fused_emb_df.columns if c.startswith('fused_') or c.startswith('z_fused_')]
            if not fused_cols:
                fused_cols = [c for c in fused_emb_df.columns if fused_emb_df[c].dtype in ['float32', 'float64']]
            if fused_cols:
                fused_dim = len(fused_cols)
            print(f"    Fused dim: {fused_dim}")
        else:
            print(f"    WARNING: fused_embedding.parquet not found at {fused_path}")

        if hlca_path.exists():
            hlca_emb_df = cache.read_parquet(hlca_path)
            if 'cell_id' in hlca_emb_df.columns:
                hlca_emb_df = hlca_emb_df.set_index('cell_id')
            # Detect actual HLCA dim
            hlca_cols = [c for c in hlca_emb_df.columns if c.startswith('hlca_') or c.startswith('z_hlca_')]
            if not hlca_cols:
                hlca_cols = [c for c in hlca_emb_df.columns if hlca_emb_df[c].dtype in ['float32', 'float64']]
            if hlca_cols:
                hlca_dim = len(hlca_cols)
            print(f"    HLCA embeddings: {len(hlca_emb_df):,} cells, dim={hlca_dim}")

        if luca_path.exists():
            luca_emb_df = cache.read_parquet(luca_path)
            if 'cell_id' in luca_emb_df.columns:
                luca_emb_df = luca_emb_df.set_index('cell_id')
            # Detect actual LuCA dim
            luca_cols = [c for c in luca_emb_df.columns if c.startswith('luca_') or c.startswith('z_luca_')]
            if not luca_cols:
                luca_cols = [c for c in luca_emb_df.columns if luca_emb_df[c].dtype in ['float32', 'float64']]
            if luca_cols:
                luca_dim = len(luca_cols)
            print(f"    LuCA embeddings: {len(luca_emb_df):,} cells, dim={luca_dim}")
    else:
        print("  WARNING: No reference_geometry_dir provided or doesn't exist!")
        print("           Embeddings will be zeros (placeholder mode)")

    # ==========================================================================
    # Compute mean embeddings per cell type for spatial embedding composition
    # HLCA embeddings use HLCA cell types (9 types)
    # LuCA embeddings use LuCA cell types (33 types)
    # ==========================================================================
    hlca_mean_emb = {}  # {cell_type: mean_hlca_embedding}
    luca_mean_emb = {}  # {cell_type: mean_luca_embedding}
    spatial_deconv_hlca = {}  # {spot_id: {cell_type: proportion}}
    spatial_deconv_luca = {}  # {spot_id: {cell_type: proportion}}

    if reference_geometry_dir is not None and fused_emb_df is not None:
        # Load HLCA cell type labels
        hlca_labels_path = reference_geometry_dir / "cell_types.parquet"
        luca_labels_path = reference_geometry_dir / "luca_mapping" / "luca_labels.parquet"

        if hlca_labels_path.exists():
            hlca_labels_df = cache.read_parquet(hlca_labels_path)
            if 'cell_id' not in hlca_labels_df.columns:
                hlca_labels_df['cell_id'] = hlca_labels_df.index
            hlca_labels_df = hlca_labels_df.set_index('cell_id')

            # Compute mean HLCA embeddings per HLCA cell type
            print("  Computing mean HLCA embeddings per cell type...")
            hlca_cols = [c for c in fused_emb_df.columns if c.startswith('hlca_latent_')]
            for cell_type in hlca_labels_df['cell_type'].unique():
                cell_ids = hlca_labels_df[hlca_labels_df['cell_type'] == cell_type].index
                matching = [cid for cid in cell_ids if cid in fused_emb_df.index]
                if matching:
                    mean_emb = fused_emb_df.loc[matching, hlca_cols].mean().values.astype(np.float32)
                    hlca_mean_emb[cell_type] = mean_emb
            # Update hlca_dim to match actual fused embedding columns (ensures consistency)
            if hlca_cols:
                hlca_dim = len(hlca_cols)
            print(f"    Computed means for {len(hlca_mean_emb)} HLCA cell types (dim={hlca_dim})")

        if luca_labels_path.exists():
            luca_labels_df = cache.read_parquet(luca_labels_path)
            if 'cell_id' not in luca_labels_df.columns:
                # Index might be cell_id
                luca_labels_df = luca_labels_df.reset_index()
                if 'index' in luca_labels_df.columns:
                    luca_labels_df = luca_labels_df.rename(columns={'index': 'cell_id'})

            # Need to align with fused_emb_df
            luca_labels_df = luca_labels_df.set_index('cell_id') if 'cell_id' in luca_labels_df.columns else luca_labels_df

            # Compute mean LuCA embeddings per LuCA cell type
            print("  Computing mean LuCA embeddings per cell type...")
            luca_cols = [c for c in fused_emb_df.columns if c.startswith('luca_latent_')]
            luca_label_col = 'luca_label' if 'luca_label' in luca_labels_df.columns else 'cell_type'
            for cell_type in luca_labels_df[luca_label_col].unique():
                cell_ids = luca_labels_df[luca_labels_df[luca_label_col] == cell_type].index
                matching = [cid for cid in cell_ids if cid in fused_emb_df.index]
                if matching:
                    mean_emb = fused_emb_df.loc[matching, luca_cols].mean().values.astype(np.float32)
                    luca_mean_emb[cell_type] = mean_emb
            # Update luca_dim to match actual fused embedding columns (ensures consistency)
            if luca_cols:
                luca_dim = len(luca_cols)
                fused_dim = hlca_dim + luca_dim  # Update fused dim too
            print(f"    Computed means for {len(luca_mean_emb)} LuCA cell types (dim={luca_dim})")

    # Load spatial deconvolution cell type proportions
    # NOTE: DestVI gamma is in DestVI's latent space (10d), not reference atlas space (30d/10d),
    # so we only use cell type proportions for embedding composition
    if hlca_deconv_dir is not None and hlca_deconv_dir.exists():
        print("  Loading HLCA deconvolution for spatial embedding composition...")
        samples_dir = hlca_deconv_dir / "samples"
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if sample_dir.is_dir():
                    prop_path = sample_dir / "cell_type_proportions.parquet"
                    if prop_path.exists():
                        prop_df = cache.read_parquet(prop_path)
                        # Normalize proportions
                        prop_sum = prop_df.sum(axis=1)
                        prop_norm = prop_df.div(prop_sum, axis=0).fillna(0)
                        for spot_id in prop_norm.index:
                            spatial_deconv_hlca[spot_id] = prop_norm.loc[spot_id].to_dict()
        print(f"    Loaded HLCA deconvolution for {len(spatial_deconv_hlca):,} spots")

    if luca_deconv_dir is not None and luca_deconv_dir.exists():
        print("  Loading LuCA deconvolution for spatial embedding composition...")
        samples_dir = luca_deconv_dir / "samples"
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if sample_dir.is_dir():
                    prop_path = sample_dir / "cell_type_proportions.parquet"
                    if prop_path.exists():
                        prop_df = cache.read_parquet(prop_path)
                        # Normalize proportions
                        prop_sum = prop_df.sum(axis=1)
                        prop_norm = prop_df.div(prop_sum, axis=0).fillna(0)
                        for spot_id in prop_norm.index:
                            spatial_deconv_luca[spot_id] = prop_norm.loc[spot_id].to_dict()
        print(f"    Loaded LuCA deconvolution for {len(spatial_deconv_luca):,} spots")

    def extract_stage_from_cell_id(cell_id: str) -> str:
        """Extract stage from cell_id and map to 3-stage system.

        Cell IDs look like: GSM9237901_P3_Normal:AAACAAGCACCAGCTCACTTTAGG.1
        Extracts raw stage (Normal/AAH/AIS/MIA/LUAD), then maps to 3-stage.
        """
        import re
        match = re.search(r'_P\d+_([^:]+):', cell_id)
        if match:
            stage_raw = match.group(1)
            # Normalize stage names (handle variants like AIS1, AIS-1, etc.)
            stage_clean = stage_raw.replace('1', '').replace('-', '').strip()
            # Check if it matches any 5-stage name
            if stage_clean in stage_order_raw:
                # Map to 3-stage
                return STAGE_MAP_3.get(stage_clean, "unknown")
            # Fuzzy match against raw 5-stage names
            for s in stage_order_raw:
                if s.lower() in stage_clean.lower():
                    return STAGE_MAP_3.get(s, "unknown")
        return "unknown"

    # ==========================================================================
    # Load scArches spatial embeddings if provided (RECOMMENDED)
    # These are embeddings computed by mapping spatial expression directly through
    # the HLCA/LuCA reference models via scArches surgery, putting spatial spots
    # in the SAME latent space as snRNA cells.
    # ==========================================================================
    spatial_emb_df = None
    if spatial_embeddings_path is not None and spatial_embeddings_path.exists():
        print(f"  Loading scArches spatial embeddings from {spatial_embeddings_path}...")
        spatial_emb_df = cache.read_parquet(spatial_embeddings_path)
        if 'cell_id' in spatial_emb_df.columns:
            spatial_emb_df = spatial_emb_df.set_index('cell_id')
        print(f"    Loaded embeddings for {len(spatial_emb_df):,} spatial spots")
        # Detect columns
        spatial_hlca_cols = [c for c in spatial_emb_df.columns if c.startswith('hlca_latent_')]
        spatial_luca_cols = [c for c in spatial_emb_df.columns if c.startswith('luca_latent_')]
        print(f"    HLCA dims: {len(spatial_hlca_cols)}, LuCA dims: {len(spatial_luca_cols)}")
    elif spatial_embeddings_path is not None:
        print(f"  WARNING: spatial_embeddings_path provided but file not found: {spatial_embeddings_path}")
        print("           Falling back to proportions × mean (causes modality separation)")

    def get_embeddings(cell_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get embeddings for a cell from loaded dataframes.

        Returns actual dimensions: HLCA (30d), LuCA (10d), Fused (40d).
        NOT truncated to a single latent_dim.

        For spatial spots (cell_id starting with "spatial_"):
        - If scArches embeddings available: use those directly (RECOMMENDED)
        - Otherwise: compose from proportions × mean (causes modality separation)
        """
        z_fused = np.zeros(fused_dim, dtype=np.float32)
        z_hlca = np.zeros(hlca_dim, dtype=np.float32)
        z_luca = np.zeros(luca_dim, dtype=np.float32)

        # For spatial spots
        if cell_id.startswith("spatial_"):
            spot_id = cell_id[8:]  # Strip "spatial_" prefix

            # Option 1 (PREFERRED): Use scArches embeddings if available
            if spatial_emb_df is not None and spot_id in spatial_emb_df.index:
                row = spatial_emb_df.loc[spot_id]
                # Extract HLCA embedding
                hlca_cols = [c for c in spatial_emb_df.columns if c.startswith('hlca_latent_')]
                if hlca_cols:
                    z_hlca = row[hlca_cols].values.astype(np.float32)
                # Extract LuCA embedding
                luca_cols = [c for c in spatial_emb_df.columns if c.startswith('luca_latent_')]
                if luca_cols:
                    z_luca = row[luca_cols].values.astype(np.float32)
                # Fused = concatenation
                z_fused[:len(z_hlca)] = z_hlca
                z_fused[len(z_hlca):len(z_hlca) + len(z_luca)] = z_luca
                return z_fused, z_hlca, z_luca

            # Option 2 (FALLBACK): Compose from proportions × mean
            # NOTE: This causes modality separation (spatial at centroids, snRNA spread around)
            if spot_id in spatial_deconv_hlca and hlca_mean_emb:
                props = spatial_deconv_hlca[spot_id]
                for cell_type, proportion in props.items():
                    if cell_type in hlca_mean_emb and proportion > 0:
                        z_hlca += proportion * hlca_mean_emb[cell_type]

            if spot_id in spatial_deconv_luca and luca_mean_emb:
                props = spatial_deconv_luca[spot_id]
                for cell_type, proportion in props.items():
                    if cell_type in luca_mean_emb and proportion > 0:
                        z_luca += proportion * luca_mean_emb[cell_type]

            # Fused = concatenation of HLCA and LuCA
            z_fused[:hlca_dim] = z_hlca
            z_fused[hlca_dim:hlca_dim + luca_dim] = z_luca

            return z_fused, z_hlca, z_luca

        # For snRNA cells, use direct reference embeddings
        if fused_emb_df is not None and cell_id in fused_emb_df.index:
            row = fused_emb_df.loc[cell_id]
            # row is a Series when index is unique - get numeric values directly
            vals = pd.to_numeric(row, errors='coerce').dropna().values.astype(np.float32)
            z_fused[:len(vals)] = vals[:fused_dim]

        if hlca_emb_df is not None and cell_id in hlca_emb_df.index:
            row = hlca_emb_df.loc[cell_id]
            vals = pd.to_numeric(row, errors='coerce').dropna().values.astype(np.float32)
            z_hlca[:len(vals)] = vals[:hlca_dim]

        if luca_emb_df is not None and cell_id in luca_emb_df.index:
            row = luca_emb_df.loc[cell_id]
            vals = pd.to_numeric(row, errors='coerce').dropna().values.astype(np.float32)
            z_luca[:len(vals)] = vals[:luca_dim]

        return z_fused, z_hlca, z_luca

    # Determine WES ID column (patient_id vs donor_id)
    wes_id_col = "patient_id" if wes_df is not None and "patient_id" in wes_df.columns else "donor_id"

    # Pre-compute pathway/proliferation targets for all snRNA cells (batch)
    # IMPORTANT: Store RAW means (not z-scored) to prevent train/val leakage.
    # Z-scoring will be done at training time using train-only statistics.
    print("  Computing pathway/proliferation targets (RAW, no z-score)...")
    gene_names = list(snrna.var_names)
    snrna_expr = torch.tensor(
        snrna.X.toarray() if hasattr(snrna.X, "toarray") else snrna.X,
        dtype=torch.float32,
    )
    # Store RAW pathway means - will be z-scored at training time
    pathway_raw = compute_pathway_raw(snrna_expr, gene_names)
    # Proliferation is binary (threshold-based), not z-scored - OK to compute here
    prolif_targets = compute_proliferation_targets(snrna_expr, gene_names, torch.device("cpu"))
    n_pathways = N_PROGENY_PATHWAYS  # From contracts.py
    print(f"    Pathway RAW: {pathway_raw.shape if pathway_raw is not None else 'None'}")
    print(f"    Proliferation targets: {prolif_targets.shape if prolif_targets is not None else 'None'}")

    # VECTORIZED snRNA processing (much faster than per-cell loop)
    print("  Building snRNA records (vectorized)...")

    # Build base DataFrame from snrna.obs
    snrna_df = snrna.obs.copy().reset_index(drop=True)
    snrna_df['cell_id'] = snrna.obs_names.tolist()
    snrna_df['data_type'] = 'snrna'
    snrna_df['x'] = np.nan
    snrna_df['y'] = np.nan

    # Convert categorical columns to string (avoid dtype issues later)
    for col in snrna_df.columns:
        if snrna_df[col].dtype.name == 'category':
            snrna_df[col] = snrna_df[col].astype(str)

    # Extract donor_id
    if 'donor_id' not in snrna_df.columns:
        snrna_df['donor_id'] = snrna_df.get('patient_id', 'unknown')

    # Vectorized stage extraction from cell_id
    def extract_stage_vectorized(cell_ids):
        stages = []
        for cid in cell_ids:
            stages.append(extract_stage_from_cell_id(cid))
        return stages

    snrna_df['stage'] = extract_stage_vectorized(snrna_df['cell_id'].values)
    # Fill unknowns from donor mapping
    unknown_mask = snrna_df['stage'] == 'unknown'
    snrna_df.loc[unknown_mask, 'stage'] = snrna_df.loc[unknown_mask, 'donor_id'].map(donor_to_stage).fillna('unknown')
    snrna_df['stage_idx'] = snrna_df['stage'].map(stage_to_idx).fillna(-1).astype(int)

    # Cell type columns
    if 'cell_type' not in snrna_df.columns:
        snrna_df['cell_type'] = 'unknown'
    if 'cell_type_hlca' not in snrna_df.columns:
        snrna_df['cell_type_hlca'] = None
    if 'cell_type_luca' not in snrna_df.columns:
        snrna_df['cell_type_luca'] = None

    # Cell cycle columns
    for col in ['S_score', 'G2M_score']:
        if col not in snrna_df.columns:
            snrna_df[col] = 0.0
        else:
            snrna_df[col] = snrna_df[col].fillna(0.0)
    if 'phase' not in snrna_df.columns:
        snrna_df['phase'] = 'unknown'
    else:
        # Convert categorical to string first, then fillna
        snrna_df['phase'] = snrna_df['phase'].astype(str).replace('nan', 'unknown').fillna('unknown')

    # Add embeddings from reference geometry (vectorized lookup via merge)
    print("    Adding embeddings...")

    def reset_index_to_cell_id(df):
        """Reset index and ensure column is named 'cell_id'."""
        df = df.reset_index()
        # Index column could be named 'cell_id', 'index', or something else
        if 'cell_id' not in df.columns:
            # Find the index column (first column after reset)
            idx_col = df.columns[0]
            df = df.rename(columns={idx_col: 'cell_id'})
        return df

    if fused_emb_df is not None:
        # Get columns for each embedding type
        fused_cols = [c for c in fused_emb_df.columns if c.startswith('fused_') or c.startswith('z_fused_')]
        if not fused_cols:
            fused_cols = [c for c in fused_emb_df.columns if fused_emb_df[c].dtype in ['float32', 'float64']][:fused_dim]

        # Merge embeddings (emb_df has cell_id as index)
        emb_subset = fused_emb_df[fused_cols].copy()
        emb_subset.columns = [f'z_fused_{i}' for i in range(len(fused_cols))]
        emb_subset = reset_index_to_cell_id(emb_subset)
        snrna_df = snrna_df.merge(emb_subset, on='cell_id', how='left')

    if hlca_emb_df is not None:
        hlca_cols = [c for c in hlca_emb_df.columns if c.startswith('hlca_') or c.startswith('z_hlca_')]
        if not hlca_cols:
            hlca_cols = [c for c in hlca_emb_df.columns if hlca_emb_df[c].dtype in ['float32', 'float64']][:hlca_dim]
        emb_subset = hlca_emb_df[hlca_cols].copy()
        emb_subset.columns = [f'z_hlca_{i}' for i in range(len(hlca_cols))]
        emb_subset = reset_index_to_cell_id(emb_subset)
        snrna_df = snrna_df.merge(emb_subset, on='cell_id', how='left')

    if luca_emb_df is not None:
        luca_cols = [c for c in luca_emb_df.columns if c.startswith('luca_') or c.startswith('z_luca_')]
        if not luca_cols:
            luca_cols = [c for c in luca_emb_df.columns if luca_emb_df[c].dtype in ['float32', 'float64']][:luca_dim]
        emb_subset = luca_emb_df[luca_cols].copy()
        emb_subset.columns = [f'z_luca_{i}' for i in range(len(luca_cols))]
        emb_subset = reset_index_to_cell_id(emb_subset)
        snrna_df = snrna_df.merge(emb_subset, on='cell_id', how='left')

    # Fill missing embeddings with zeros
    for i in range(fused_dim):
        col = f'z_fused_{i}'
        if col not in snrna_df.columns:
            snrna_df[col] = 0.0
        else:
            snrna_df[col] = snrna_df[col].fillna(0.0)
    for i in range(hlca_dim):
        col = f'z_hlca_{i}'
        if col not in snrna_df.columns:
            snrna_df[col] = 0.0
        else:
            snrna_df[col] = snrna_df[col].fillna(0.0)
    for i in range(luca_dim):
        col = f'z_luca_{i}'
        if col not in snrna_df.columns:
            snrna_df[col] = 0.0
        else:
            snrna_df[col] = snrna_df[col].fillna(0.0)

    # Add pathway scores
    print("    Adding pathway scores...")
    if pathway_raw is not None:
        pathway_np = pathway_raw.numpy()
        for p_idx in range(n_pathways):
            snrna_df[f'pathway_raw_{p_idx}'] = pathway_np[:, p_idx]
    else:
        for p_idx in range(n_pathways):
            snrna_df[f'pathway_raw_{p_idx}'] = 0.0

    # Add proliferation
    if prolif_targets is not None:
        snrna_df['proliferation_label'] = prolif_targets.numpy()[:, 0]
    else:
        snrna_df['proliferation_label'] = 0.0

    # Add WES features (vectorized via merge)
    print("    Adding WES features...")
    for wes_col in WES_FEATURE_COLS:
        snrna_df[wes_col] = 0.0

    if wes_df is not None:
        # Create donor-stage key for merging
        snrna_df['_merge_key'] = snrna_df['donor_id'] + '_' + snrna_df['stage']
        wes_df_copy = wes_df.copy()
        wes_df_copy['_merge_key'] = wes_df_copy[wes_id_col].astype(str) + '_' + wes_df_copy['stage'].astype(str)
        wes_df_copy = wes_df_copy.drop_duplicates('_merge_key')

        # Merge WES features
        wes_cols_to_merge = [c for c in WES_FEATURE_COLS if c in wes_df_copy.columns]
        if wes_cols_to_merge:
            wes_subset = wes_df_copy[['_merge_key'] + wes_cols_to_merge].set_index('_merge_key')
            for col in wes_cols_to_merge:
                snrna_df[col] = snrna_df['_merge_key'].map(wes_subset[col]).fillna(0.0)
        snrna_df = snrna_df.drop(columns=['_merge_key'])

    # Create z_fused, z_hlca, z_luca list columns for neighborhoods compatibility
    print("    Creating embedding list columns...")
    z_fused_cols = [f'z_fused_{i}' for i in range(fused_dim)]
    z_hlca_cols = [f'z_hlca_{i}' for i in range(hlca_dim)]
    z_luca_cols = [f'z_luca_{i}' for i in range(luca_dim)]

    snrna_df['z_fused'] = snrna_df[z_fused_cols].values.tolist()
    snrna_df['z_hlca'] = snrna_df[z_hlca_cols].values.tolist()
    snrna_df['z_luca'] = snrna_df[z_luca_cols].values.tolist()

    # Convert to records
    records = snrna_df.to_dict('records')
    print(f"    Processed {len(records):,} snRNA cells")

    # Pre-compute RAW pathway means for spatial spots (z-scored at training time)
    print("  Computing spatial pathway/proliferation targets (RAW, no z-score)...")
    spatial_gene_names = list(spatial.var_names)
    spatial_expr = torch.tensor(
        spatial.X.toarray() if hasattr(spatial.X, "toarray") else spatial.X,
        dtype=torch.float32,
    )
    spatial_pathway_raw = compute_pathway_raw(spatial_expr, spatial_gene_names)
    spatial_prolif_targets = compute_proliferation_targets(spatial_expr, spatial_gene_names, torch.device("cpu"))
    print(f"    Spatial pathway RAW: {spatial_pathway_raw.shape if spatial_pathway_raw is not None else 'None'}")
    print(f"    Spatial proliferation targets: {spatial_prolif_targets.shape if spatial_prolif_targets is not None else 'None'}")

    # VECTORIZED spatial processing
    print("  Building spatial records (vectorized)...")

    spatial_df = spatial.obs.copy()
    spatial_df['cell_id'] = 'spatial_' + spatial.obs_names.astype(str)
    spatial_df['spot_id'] = spatial.obs_names.astype(str)  # Keep original for lookups
    spatial_df['data_type'] = 'spatial'

    # Convert categorical columns to string (avoid dtype issues later)
    for col in spatial_df.columns:
        if spatial_df[col].dtype.name == 'category':
            spatial_df[col] = spatial_df[col].astype(str)

    # Extract donor_id
    if 'donor_id' not in spatial_df.columns:
        spatial_df['donor_id'] = spatial_df.get('patient_id', 'unknown')

    # Vectorized stage extraction
    spatial_df['stage'] = extract_stage_vectorized(spatial_df['spot_id'].values)
    unknown_mask = spatial_df['stage'] == 'unknown'
    spatial_df.loc[unknown_mask, 'stage'] = spatial_df.loc[unknown_mask, 'donor_id'].map(donor_to_stage).fillna('unknown')
    spatial_df['stage_idx'] = spatial_df['stage'].map(stage_to_idx).fillna(-1).astype(int)

    # Cell type columns (spatial spots are mixtures)
    if 'cell_type' not in spatial_df.columns:
        spatial_df['cell_type'] = 'mixed'
    spatial_df['cell_type_hlca'] = None
    spatial_df['cell_type_luca'] = None

    # Spatial-specific: NaN for cell cycle (no single-cell resolution)
    spatial_df['S_score'] = np.nan
    spatial_df['G2M_score'] = np.nan
    spatial_df['phase'] = 'spatial'

    # Spatial coordinates
    spatial_df['x'] = spatial.obsm['spatial'][:, 0]
    spatial_df['y'] = spatial.obsm['spatial'][:, 1]

    # Compute embeddings for spatial spots
    # Two modes:
    # 1. scArches (PREFERRED): Direct lookup from pre-computed embeddings
    # 2. Proportions × mean (FALLBACK): Causes modality separation
    n_spatial = len(spatial_df)

    # Pre-allocate arrays
    z_fused_arr = np.zeros((n_spatial, fused_dim), dtype=np.float32)
    z_hlca_arr = np.zeros((n_spatial, hlca_dim), dtype=np.float32)
    z_luca_arr = np.zeros((n_spatial, luca_dim), dtype=np.float32)

    if spatial_emb_df is not None:
        # ==========================================================================
        # scArches mode: Use pre-computed embeddings (RECOMMENDED)
        # These embeddings are in the SAME latent space as snRNA cells
        # ==========================================================================
        print("    Using scArches spatial embeddings (RECOMMENDED)...")
        spot_ids = spatial_df['spot_id'].tolist()

        # Get column names
        spatial_hlca_cols = [c for c in spatial_emb_df.columns if c.startswith('hlca_latent_')]
        spatial_luca_cols = [c for c in spatial_emb_df.columns if c.startswith('luca_latent_')]

        n_matched = 0
        for idx, spot_id in enumerate(tqdm(spot_ids, desc="    Loading spatial embeddings")):
            if spot_id in spatial_emb_df.index:
                row = spatial_emb_df.loc[spot_id]
                if spatial_hlca_cols:
                    z_hlca_arr[idx] = row[spatial_hlca_cols].values.astype(np.float32)
                if spatial_luca_cols:
                    z_luca_arr[idx] = row[spatial_luca_cols].values.astype(np.float32)
                # Fused = concatenation
                z_fused_arr[idx, :len(z_hlca_arr[idx])] = z_hlca_arr[idx]
                z_fused_arr[idx, len(z_hlca_arr[idx]):len(z_hlca_arr[idx]) + len(z_luca_arr[idx])] = z_luca_arr[idx]
                n_matched += 1

        print(f"    Matched {n_matched:,} / {n_spatial:,} spots to scArches embeddings")
        if n_matched < n_spatial:
            print(f"    WARNING: {n_spatial - n_matched:,} spots have no scArches embedding (using zeros)")
    else:
        # ==========================================================================
        # Fallback mode: Proportions × mean (causes modality separation)
        # ==========================================================================
        print("    Computing spatial embeddings via proportions × mean (FALLBACK)...")
        print("    WARNING: This causes modality separation! Use --spatial_embeddings for scArches.")

        # Determine number of workers
        import os
        from multiprocessing.pool import ThreadPool
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))
        n_workers = min(n_workers, 32)
        print(f"    Using {n_workers} threads for parallel processing")

        cell_ids = spatial_df['cell_id'].tolist()

        def compute_spot_embedding(idx):
            cell_id = cell_ids[idx]
            spot_id = cell_id[8:] if cell_id.startswith("spatial_") else cell_id

            z_fused = np.zeros(fused_dim, dtype=np.float32)
            z_hlca = np.zeros(hlca_dim, dtype=np.float32)
            z_luca = np.zeros(luca_dim, dtype=np.float32)

            # Compose HLCA embedding from HLCA cell type proportions
            if spot_id in spatial_deconv_hlca and hlca_mean_emb:
                props = spatial_deconv_hlca[spot_id]
                for cell_type, proportion in props.items():
                    if cell_type in hlca_mean_emb and proportion > 0:
                        z_hlca += proportion * hlca_mean_emb[cell_type]

            # Compose LuCA embedding from LuCA cell type proportions
            if spot_id in spatial_deconv_luca and luca_mean_emb:
                props = spatial_deconv_luca[spot_id]
                for cell_type, proportion in props.items():
                    if cell_type in luca_mean_emb and proportion > 0:
                        z_luca += proportion * luca_mean_emb[cell_type]

            # Fused = concatenation
            z_fused[:hlca_dim] = z_hlca
            z_fused[hlca_dim:hlca_dim + luca_dim] = z_luca

            return idx, z_fused, z_hlca, z_luca

        # Process in parallel using ThreadPool
        if n_workers > 1:
            with ThreadPool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(compute_spot_embedding, range(n_spatial), chunksize=1000),
                    total=n_spatial,
                    desc="    Spatial embeddings"
                ))
            for idx, z_fused, z_hlca, z_luca in results:
                z_fused_arr[idx] = z_fused
                z_hlca_arr[idx] = z_hlca
                z_luca_arr[idx] = z_luca
        else:
            for idx in tqdm(range(n_spatial), desc="    Spatial embeddings"):
                _, z_fused, z_hlca, z_luca = compute_spot_embedding(idx)
                z_fused_arr[idx] = z_fused
                z_hlca_arr[idx] = z_hlca
                z_luca_arr[idx] = z_luca

    # Add embedding columns
    for i in range(fused_dim):
        spatial_df[f'z_fused_{i}'] = z_fused_arr[:, i]
    for i in range(hlca_dim):
        spatial_df[f'z_hlca_{i}'] = z_hlca_arr[:, i]
    for i in range(luca_dim):
        spatial_df[f'z_luca_{i}'] = z_luca_arr[:, i]

    # Add pathway scores
    print("    Adding pathway scores...")
    if spatial_pathway_raw is not None:
        pathway_np = spatial_pathway_raw.numpy()
        for p_idx in range(n_pathways):
            spatial_df[f'pathway_raw_{p_idx}'] = pathway_np[:, p_idx]
    else:
        for p_idx in range(n_pathways):
            spatial_df[f'pathway_raw_{p_idx}'] = 0.0

    # Add proliferation
    if spatial_prolif_targets is not None:
        spatial_df['proliferation_label'] = spatial_prolif_targets.numpy()[:, 0]
    else:
        spatial_df['proliferation_label'] = 0.0

    # Add WES features (vectorized via merge)
    print("    Adding WES features...")
    for wes_col in WES_FEATURE_COLS:
        spatial_df[wes_col] = 0.0

    if wes_df is not None:
        spatial_df['_merge_key'] = spatial_df['donor_id'] + '_' + spatial_df['stage']
        wes_df_copy = wes_df.copy()
        wes_df_copy['_merge_key'] = wes_df_copy[wes_id_col].astype(str) + '_' + wes_df_copy['stage'].astype(str)
        wes_df_copy = wes_df_copy.drop_duplicates('_merge_key')

        wes_cols_to_merge = [c for c in WES_FEATURE_COLS if c in wes_df_copy.columns]
        if wes_cols_to_merge:
            wes_subset = wes_df_copy[['_merge_key'] + wes_cols_to_merge].set_index('_merge_key')
            for col in wes_cols_to_merge:
                spatial_df[col] = spatial_df['_merge_key'].map(wes_subset[col]).fillna(0.0)
        spatial_df = spatial_df.drop(columns=['_merge_key'])

    # Create embedding list columns
    print("    Creating embedding list columns...")
    z_fused_cols = [f'z_fused_{i}' for i in range(fused_dim)]
    z_hlca_cols = [f'z_hlca_{i}' for i in range(hlca_dim)]
    z_luca_cols = [f'z_luca_{i}' for i in range(luca_dim)]

    spatial_df['z_fused'] = spatial_df[z_fused_cols].values.tolist()
    spatial_df['z_hlca'] = spatial_df[z_hlca_cols].values.tolist()
    spatial_df['z_luca'] = spatial_df[z_luca_cols].values.tolist()

    # Drop spot_id column (not needed in final output)
    spatial_df = spatial_df.drop(columns=['spot_id'])

    # Combine snRNA and spatial records
    print(f"    Processed {len(spatial_df):,} spatial spots")
    spatial_records = spatial_df.to_dict('records')
    records.extend(spatial_records)

    # Convert to DataFrame
    cells_df = pd.DataFrame(records)

    # Add clonal evolution patterns for H3 validation
    # Patterns are per-donor/patient, mapped to all cells from that donor
    if clonal_patterns is not None:
        print(f"  Adding clonal patterns from {len(clonal_patterns)} patients...")
        # Map donor_id -> pattern string
        cells_df["clonal_pattern"] = cells_df["donor_id"].map(clonal_patterns).fillna("unknown")
        # Numeric encoding for model input (see CLONAL_PATTERN_ENCODING at top of file)
        cells_df["clonal_pattern_idx"] = cells_df["clonal_pattern"].map(CLONAL_PATTERN_ENCODING).fillna(-1).astype(int)
        # Log coverage
        n_with_pattern = (cells_df["clonal_pattern"] != "unknown").sum()
        print(f"    Cells with clonal annotation: {n_with_pattern:,} / {len(cells_df):,}")
    else:
        # No clonal data - add placeholder columns for schema consistency
        cells_df["clonal_pattern"] = "unknown"
        cells_df["clonal_pattern_idx"] = -1

    return cells_df


def generate_neighborhoods_table(
    cells_df: pd.DataFrame,
    spatial: ad.AnnData,
    spatial_deconv_luca: dict,
    k_neighbors: int = 100,
    use_amici_format: bool = True,
) -> pd.DataFrame:
    """
    Generate neighborhoods.parquet for AMICI attention.

    AMICI format (default):
    - neighbor_cells: List of k neighbor embeddings (z_fused), sorted by distance
    - neighbor_distances: List of k distances in coordinate units

    Legacy tokenized format (use_amici_format=False):
    - 9 tokens with pooled ring embeddings
    """
    # Build spatial graph
    print("  Building spatial neighborhood graph...")
    spatial_cells = cells_df[~cells_df["x"].isna()].copy()

    if len(spatial_cells) == 0:
        print("  Warning: No spatial cells found, skipping neighborhoods")
        return pd.DataFrame()

    # CRITICAL: Build k-NN PER DONOR to prevent cross-donor leakage
    # Spatial coordinates can overlap across donors (different tissue sections),
    # so global k-NN would create neighbors across donors = data leakage
    from sklearn.neighbors import NearestNeighbors

    records = []
    donors = spatial_cells["donor_id"].unique()
    print(f"  Building per-donor k-NN for {len(donors)} donors (k={k_neighbors})...")

    for donor_id in tqdm(donors, desc="  Donors"):
        donor_cells = spatial_cells[spatial_cells["donor_id"] == donor_id].copy()
        donor_cells = donor_cells.reset_index(drop=True)

        if len(donor_cells) < k_neighbors + 1:
            print(f"    Warning: Donor {donor_id} has only {len(donor_cells)} cells, skipping")
            continue

        # Build k-NN for this donor only
        coords = donor_cells[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Process cells for this donor
        for pos_idx, row in enumerate(donor_cells.itertuples()):
            cell_id = row.cell_id
            stage = row.stage

            # Get neighbors (exclude self) - use positional index within donor
            neighbor_indices = indices[pos_idx][1:]
            neighbor_distances = distances[pos_idx][1:]

            # Sanity check: all neighbors should be from same donor
            neighbor_donors = donor_cells.iloc[neighbor_indices]["donor_id"].unique()
            if len(neighbor_donors) != 1 or neighbor_donors[0] != donor_id:
                raise ValueError(
                    f"CRITICAL: Cross-donor neighbors detected for cell {cell_id} (donor {donor_id}). "
                    f"Neighbor donors: {neighbor_donors}. This indicates a bug in per-donor k-NN."
                )

            if use_amici_format:
                # AMICI format: flat list of neighbor embeddings + distances
                neighbor_cells = [
                    donor_cells.iloc[ni]["z_fused"]
                    for ni in neighbor_indices
                ]
                neighbor_cell_types = [
                    donor_cells.iloc[ni]["cell_type"]
                    for ni in neighbor_indices
                ]

                # Get additional features for the record
                hlca_z = row.z_hlca if hasattr(row, 'z_hlca') else None
                luca_z = row.z_luca if hasattr(row, 'z_luca') else None

                # Pathway targets from cells_df
                pathway_cols = [c for c in cells_df.columns if c.startswith('pathway_')]
                pathway_targets = None
                if pathway_cols:
                    cell_row = cells_df[cells_df['cell_id'] == cell_id]
                    if len(cell_row) > 0:
                        pathway_targets = cell_row[pathway_cols].values[0].tolist()

                # Stats features
                stats_z = None
                if hasattr(row, 'S_score') and hasattr(row, 'G2M_score'):
                    stats_z = [
                        row.S_score if not pd.isna(row.S_score) else 0.0,
                        row.G2M_score if not pd.isna(row.G2M_score) else 0.0,
                        getattr(row, 'caf_fraction', 0.0) or 0.0,
                        getattr(row, 'immune_fraction', 0.0) or 0.0,
                        getattr(row, 'diversity', 0.0) or 0.0,
                    ]

                # Evolution features
                evolution_features = None
                evolution_cols = [c for c in cells_df.columns if c in EVOLUTION_COLS or c.startswith('evolution_')]
                if evolution_cols:
                    cell_row = cells_df[cells_df['cell_id'] == cell_id]
                    if len(cell_row) > 0:
                        evolution_features = cell_row[evolution_cols].values[0].tolist()

                records.append({
                    "cell_id": cell_id,
                    "donor_id": donor_id,
                    "stage": stage,
                    "receiver_z": row.z_fused,
                    "hlca_z": hlca_z,
                    "luca_z": luca_z,
                    "neighbor_cells": neighbor_cells,
                    "neighbor_distances": neighbor_distances.tolist(),
                    "neighbor_cell_types": neighbor_cell_types,
                    "proliferation_label": getattr(row, 'proliferation_label', 0.0),
                    "pathway_targets": pathway_targets,
                    "stats_z": stats_z,
                    "evolution_features": evolution_features,
                })
                continue

            # Legacy tokenized format below
            # Build 9-token structure
            tokens = []

            # Token 0: Receiver
            tokens.append(
                {
                    "token_idx": 0,
                    "token_type": "receiver",
                    "cell_id": cell_id,
                    "cell_type": row.cell_type,
                    "z_fused": row.z_fused,
                }
            )

            # Tokens 1-4: Distance-based concentric rings
            # Ring boundaries defined by distance quantiles for biologically meaningful shells
            # Ring 1: 0-25th percentile (immediate neighbors)
            # Ring 2: 25-50th percentile
            # Ring 3: 50-75th percentile
            # Ring 4: 75-100th percentile (outermost ring)
            max_dist = neighbor_distances.max()
            if max_dist > 0:
                # Normalize distances to [0, 1] for consistent ring boundaries
                normalized_dists = neighbor_distances / max_dist
                ring_boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
            else:
                # All distances are 0 (degenerate case)
                normalized_dists = np.zeros_like(neighbor_distances)
                ring_boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]

            for ring in range(4):
                lower = ring_boundaries[ring]
                upper = ring_boundaries[ring + 1]

                # Include upper bound for last ring
                if ring == 3:
                    ring_mask = (normalized_dists >= lower) & (normalized_dists <= upper)
                else:
                    ring_mask = (normalized_dists >= lower) & (normalized_dists < upper)

                ring_neighbor_indices = neighbor_indices[ring_mask]
                ring_distances = neighbor_distances[ring_mask]

                if len(ring_neighbor_indices) == 0:
                    # Empty ring (gap in spatial distribution)
                    tokens.append(
                        {
                            "token_idx": ring + 1,
                            "token_type": f"ring_{ring + 1}",
                            "n_cells": 0,
                            "mean_distance": float((lower + upper) / 2 * max_dist) if max_dist > 0 else 0.0,
                            "normalized_distance": float((lower + upper) / 2),
                        }
                    )
                    continue

                ring_neighbors = donor_cells.iloc[ring_neighbor_indices]

                # Pool cell types in ring
                celltype_counts = ring_neighbors["cell_type"].value_counts().to_dict()

                # Pool embeddings
                z_pooled = np.mean([z for z in ring_neighbors["z_fused"]], axis=0)

                tokens.append(
                    {
                        "token_idx": ring + 1,
                        "token_type": f"ring_{ring + 1}",
                        "n_cells": len(ring_neighbors),
                        "z_pooled": z_pooled.tolist(),
                        "celltype_composition": celltype_counts,
                        "mean_distance": float(ring_distances.mean()),
                        "normalized_distance": float(normalized_dists[ring_mask].mean()),
                    }
                )

            # Token 5: HLCA context
            tokens.append(
                {
                    "token_idx": 5,
                    "token_type": "hlca",
                    "z_hlca": row.z_hlca,
                }
            )

            # Token 6: LuCA context
            tokens.append(
                {
                    "token_idx": 6,
                    "token_type": "luca",
                    "z_luca": row.z_luca,
                }
            )

            # Token 7: Pathway activity (from LuCA deconvolution - cancer-aware cell types)
            # Strip spatial_ prefix to match deconv keys
            spot_id = cell_id[8:] if cell_id.startswith("spatial_") else cell_id
            spot_proportions = spatial_deconv_luca.get(spot_id, None)

            if spot_proportions is not None:
                # Compute pathway scores from cell type composition (LuCA cell type names)
                caf_fraction = (
                    spot_proportions.get("fibroblast of lung", 0.0) +
                    spot_proportions.get("bronchus fibroblast of lung", 0.0) +
                    spot_proportions.get("stromal cell", 0.0)
                )
                immune_fraction = (
                    spot_proportions.get("alveolar macrophage", 0.0) +
                    spot_proportions.get("macrophage", 0.0) +
                    spot_proportions.get("CD4-positive, alpha-beta T cell", 0.0) +
                    spot_proportions.get("CD8-positive, alpha-beta T cell", 0.0) +
                    spot_proportions.get("regulatory T cell", 0.0) +
                    spot_proportions.get("natural killer cell", 0.0)
                )
                emt_score = 0.6 * caf_fraction + 0.4 * immune_fraction
            else:
                caf_fraction = 0.0
                immune_fraction = 0.0
                emt_score = 0.0

            tokens.append(
                {
                    "token_idx": 7,
                    "token_type": "pathway",
                    "emt_score": float(emt_score),
                    "caf_fraction": float(caf_fraction),
                    "immune_fraction": float(immune_fraction),
                }
            )

            # Token 8: Summary stats
            tokens.append(
                {
                    "token_idx": 8,
                    "token_type": "stats",
                    "n_neighbors": k_neighbors,
                    "mean_distance": float(neighbor_distances.mean()),
                    "diversity": len(donor_cells.iloc[neighbor_indices]["cell_type"].unique()),
                }
            )

            records.append(
                {
                    "cell_id": cell_id,
                    "donor_id": donor_id,
                    "stage": stage,
                    "tokens": tokens,
                }
            )

    return pd.DataFrame(records)


def generate_stage_edges_table(stage_definitions: dict[str, list[str]]) -> pd.DataFrame:
    """
    Generate stage_edges.parquet with valid transitions.

    For LUAD with 3-stage: Normal → Preinvasive → Invasive
    """
    # Always use 3-stage system regardless of stage_definitions
    stages = CANONICAL_STAGES_3
    edges = []

    for i in range(len(stages) - 1):
        source = stages[i]
        target = stages[i + 1]

        edges.append(
            {
                "edge_id": f"{source}_{target}",
                "source_stage": source,
                "target_stage": target,
                "source_idx": i,
                "target_idx": i + 1,
                "is_forward": True,
                "pseudotime_delta": 1.0,
            }
        )

    return pd.DataFrame(edges)


def generate_cv_splits(cells_df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Generate donor-held-out cross-validation splits.

    Each fold holds out different donors for test, uses some for val, rest for train.
    """
    donors = sorted(cells_df["donor_id"].unique())
    n_donors = len(donors)

    splits = {"folds": []}

    for fold_idx in range(n_folds):
        # Round-robin assignment
        test_start = fold_idx * (n_donors // n_folds)
        test_end = (fold_idx + 1) * (n_donors // n_folds)

        if fold_idx == n_folds - 1:
            test_end = n_donors  # Last fold gets remainder

        test_donors = donors[test_start:test_end]
        remaining = [d for d in donors if d not in test_donors]

        # 80-20 split of remaining for train/val
        n_val = max(1, len(remaining) // 5)
        val_donors = remaining[:n_val]
        train_donors = remaining[n_val:]

        splits["folds"].append(
            {
                "fold": fold_idx,
                "train_donors": train_donors,
                "val_donors": val_donors,
                "test_donors": list(test_donors),
            }
        )

    return splits


def generate_feature_spec(cells_df: pd.DataFrame, neighborhoods_df: pd.DataFrame) -> dict:
    """Generate feature specifications for documentation."""
    # Detect actual embedding dimensions from column names
    fused_cols = [c for c in cells_df.columns if c.startswith("z_fused_")]
    hlca_cols = [c for c in cells_df.columns if c.startswith("z_hlca_")]
    luca_cols = [c for c in cells_df.columns if c.startswith("z_luca_")]

    # Count clonal pattern coverage
    has_clonal = "clonal_pattern" in cells_df.columns
    clonal_coverage = {}
    if has_clonal:
        pattern_counts = cells_df["clonal_pattern"].value_counts().to_dict()
        n_with_clonal = len(cells_df) - pattern_counts.get("unknown", 0)
        clonal_coverage = {
            "n_cells_with_clonal": n_with_clonal,
            "coverage_pct": round(100 * n_with_clonal / len(cells_df), 1),
            "pattern_distribution": pattern_counts,
        }

    return {
        "cells": {
            "n_cells": len(cells_df),
            "n_donors": cells_df["donor_id"].nunique(),
            "n_stages": cells_df["stage"].nunique(),
            "stages": sorted(cells_df["stage"].unique().tolist()),
            "embedding_dims": {
                "fused": len(fused_cols),  # Expected: 40 (30 HLCA + 10 LuCA)
                "hlca": len(hlca_cols),    # Expected: 30 (scANVI latent)
                "luca": len(luca_cols),    # Expected: 10 (scVI latent)
            },
            "wes_features": WES_FEATURE_COLS,  # 8 features: tmb + 7 driver mutations
            "cell_cycle_features": {
                "S_score": "S phase score from scanpy (snRNA only, NaN for spatial)",
                "G2M_score": "G2M phase score from scanpy (snRNA only, NaN for spatial)",
                "phase": "Cell cycle phase: G1, S, G2M, or 'spatial' for spots",
            },
            "clonal_patterns": {
                "has_clonal_data": has_clonal,
                "encoding": CLONAL_PATTERN_ENCODING,
                **clonal_coverage,
            },
        },
        "neighborhoods": {
            "n_neighborhoods": len(neighborhoods_df),
            "n_tokens": 9,
            "token_types": [
                "receiver",
                "ring_1",
                "ring_2",
                "ring_3",
                "ring_4",
                "hlca",
                "luca",
                "pathway",
                "stats",
            ],
        },
        "version": "1.2",  # Added cell cycle features (S_score, G2M_score, phase)
    }


def main():
    parser = argparse.ArgumentParser(description="Complete Data Preparation Pipeline")

    # Inputs
    parser.add_argument("--snrna", type=str, required=True, help="Path to snrna_merged.h5ad")
    parser.add_argument("--spatial", type=str, required=True, help="Path to spatial_merged.h5ad")
    parser.add_argument("--wes", type=str, required=True, help="Path to wes_features.parquet")
    parser.add_argument(
        "--reference_geometry", type=str, required=True,
        help="Path to reference_geometry directory (contains fused_embedding.parquet, etc.)"
    )
    parser.add_argument(
        "--clonal_patterns", type=str, default=None,
        help="Path to clonal_patterns.json from run_clonal_extraction.py (for H3 validation)"
    )
    parser.add_argument(
        "--spatial_embeddings", type=str, default=None,
        help="Path to spatial_fused_embedding.parquet from scArches mapping (RECOMMENDED)"
    )
    # Legacy deconvolution args (deprecated, use --spatial_embeddings instead)
    parser.add_argument(
        "--hlca_deconv_dir", type=str, default=None,
        help="[DEPRECATED] Path to HLCA deconvolution results"
    )
    parser.add_argument(
        "--luca_deconv_dir", type=str, default=None,
        help="[DEPRECATED] Path to LuCA deconvolution results"
    )

    # Stage definitions
    parser.add_argument("--stage_config", type=str, help="YAML file with stage definitions")

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    # Load stage definitions
    if args.stage_config and Path(args.stage_config).exists():
        with open(args.stage_config) as f:
            stage_definitions = yaml.safe_load(f)
    else:
        # Default LUAD stages
        stage_definitions = {
            "Normal": ["P001", "P002", "P003"],
            "Preneoplastic": ["P004", "P005", "P006"],
            "Invasive": ["P007", "P008", "P009"],
            "Advanced": ["P010", "P011", "P012"],
        }

    generate_canonical_artifacts(
        snrna_path=Path(args.snrna),
        spatial_path=Path(args.spatial),
        wes_features_path=Path(args.wes),
        output_dir=Path(args.output_dir),
        stage_definitions=stage_definitions,
        n_folds=args.n_folds,
        reference_geometry_dir=Path(args.reference_geometry),
        clonal_patterns_path=Path(args.clonal_patterns) if args.clonal_patterns else None,
        spatial_embeddings_path=Path(args.spatial_embeddings) if args.spatial_embeddings else None,
        hlca_deconv_dir=Path(args.hlca_deconv_dir) if args.hlca_deconv_dir else None,
        luca_deconv_dir=Path(args.luca_deconv_dir) if args.luca_deconv_dir else None,
    )


if __name__ == "__main__":
    main()
