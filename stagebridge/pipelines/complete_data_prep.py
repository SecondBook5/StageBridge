#!/usr/bin/env python3
"""
Complete Real Data Pipeline for StageBridge V1

This script completes all missing pieces from run_data_prep.py:
1. Generate canonical artifacts (cells.parquet, neighborhoods.parquet, etc.)
2. Integrate spatial backend results
3. Build 9-token niche structure
4. Generate donor-held-out CV splits
5. Extract WES features properly
6. Integrate clonal evolution patterns for H3 validation

This is the PRODUCTION-READY version that handles real LUAD data.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import anndata as ad
import json
import yaml
from tqdm import tqdm
import torch
from stagebridge.utils.data_cache import get_data_cache
from stagebridge.biology.pathway_targets import (
    compute_pathway_targets,
    compute_proliferation_targets,
    PROGENY_PATHWAYS,
)
from stagebridge.data.luad_evo.wes import WES_FEATURE_COLS

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
    spatial_backend_dir: Path,
    output_dir: Path,
    stage_definitions: dict[str, list[str]],
    n_folds: int = 5,
    reference_geometry_dir: Path | None = None,
    clonal_patterns_path: Path | None = None,
    hlca_deconv_dir: Path | None = None,
    luca_deconv_dir: Path | None = None,
):
    """
    Generate all canonical artifacts for StageBridge V1.

    Inputs:
        - snrna_merged.h5ad (from run_data_prep.py)
        - spatial_merged.h5ad (from run_data_prep.py)
        - wes_features.parquet (from run_data_prep.py)
        - spatial_backend results (cell_type_proportions.parquet)
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

    # Load spatial backend results (use canonical backend from benchmark)
    backend_results = cache.read_parquet(spatial_backend_dir / "cell_type_proportions.parquet")

    # Load DestVI gamma values if available (intra-cell-type variation)
    # Gamma files are per-sample in samples/*/destvi_gamma_*.csv
    gamma_df = None
    samples_dir = spatial_backend_dir / "samples"
    if samples_dir.exists():
        sample_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
        print(f"  Searching for gamma files in {len(sample_dirs)} sample directories...")

        all_gamma_dfs = []
        for sample_dir in sorted(sample_dirs):
            gamma_files = list(sample_dir.glob("destvi_gamma_*.csv"))
            if gamma_files:
                # Average gamma across cell types for this sample
                gamma_stack = []
                for gf in sorted(gamma_files):
                    gf_df = pd.read_csv(gf, index_col=0)
                    gamma_stack.append(gf_df.values)
                mean_gamma = np.stack(gamma_stack).mean(axis=0)
                n_gamma = mean_gamma.shape[1]
                gamma_cols = [f"gamma_{i}" for i in range(n_gamma)]
                sample_gamma_df = pd.DataFrame(
                    mean_gamma,
                    index=pd.read_csv(gamma_files[0], index_col=0).index,
                    columns=gamma_cols
                )
                all_gamma_dfs.append(sample_gamma_df)

        if all_gamma_dfs:
            gamma_df = pd.concat(all_gamma_dfs, axis=0)
            print(f"  Loaded gamma from {len(all_gamma_dfs)} samples: {gamma_df.shape} (spots x gamma dims)")
        else:
            print("  No DestVI gamma files found in sample directories")
    else:
        print("  No samples directory found (using proportions only)")

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
        gamma_df=gamma_df,
        reference_geometry_dir=reference_geometry_dir,
        clonal_patterns=clonal_patterns,
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
    neighborhoods_df = generate_neighborhoods_table(
        cells_df=cells_df,
        spatial=spatial,
        backend_results=backend_results,
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
    gamma_df: pd.DataFrame | None = None,
    reference_geometry_dir: Path | None = None,
    clonal_patterns: dict[str, str] | None = None,
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
    - x_spatial, y_spatial: Spatial coordinates (for spatial cells)
    - clonal_pattern: Clonal evolution pattern (1a/1b/2/stable/unknown) [if provided]
    - clonal_pattern_idx: Numeric encoding for model input [if provided]
    """
    records = []
    cache = get_data_cache()

    # Stage order for indexing
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    stage_to_idx = {s: i for i, s in enumerate(stage_order)}

    # Map donors to stages (fallback, prefer cell_id extraction)
    donor_to_stage = {}
    for stage, donors in stage_definitions.items():
        for donor in donors:
            donor_to_stage[donor] = stage

    stages = list(stage_definitions.keys())

    # ==========================================================================
    # Load REAL embeddings from reference geometry (CRITICAL!)
    # Preserve actual dimensions: HLCA=30, LuCA=10, Fused=40
    # ==========================================================================
    fused_emb_df = None
    hlca_emb_df = None
    luca_emb_df = None

    # Default dimensions from reference atlases (scArches/scVI models)
    hlca_dim = 30  # HLCA scANVI latent
    luca_dim = 10  # LuCA scVI latent
    fused_dim = hlca_dim + luca_dim  # Concatenated

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
            for cell_type in hlca_labels_df['cell_type'].unique():
                cell_ids = hlca_labels_df[hlca_labels_df['cell_type'] == cell_type].index
                matching = [cid for cid in cell_ids if cid in fused_emb_df.index]
                if matching:
                    hlca_cols = [c for c in fused_emb_df.columns if c.startswith('hlca_latent_')]
                    mean_emb = fused_emb_df.loc[matching, hlca_cols].mean().values.astype(np.float32)
                    hlca_mean_emb[cell_type] = mean_emb
            print(f"    Computed means for {len(hlca_mean_emb)} HLCA cell types")

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
            luca_label_col = 'luca_label' if 'luca_label' in luca_labels_df.columns else 'cell_type'
            for cell_type in luca_labels_df[luca_label_col].unique():
                cell_ids = luca_labels_df[luca_labels_df[luca_label_col] == cell_type].index
                matching = [cid for cid in cell_ids if cid in fused_emb_df.index]
                if matching:
                    luca_cols = [c for c in fused_emb_df.columns if c.startswith('luca_latent_')]
                    mean_emb = fused_emb_df.loc[matching, luca_cols].mean().values.astype(np.float32)
                    luca_mean_emb[cell_type] = mean_emb
            print(f"    Computed means for {len(luca_mean_emb)} LuCA cell types")

    # Load spatial deconvolution results
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
        """Extract stage from cell_id like GSM9237901_P3_Normal:AAACAAGCACCAGCTCACTTTAGG.1"""
        import re
        match = re.search(r'_P\d+_([^:]+):', cell_id)
        if match:
            stage_raw = match.group(1)
            # Normalize stage names (handle variants like AIS1, AIS-1, etc.)
            stage_clean = stage_raw.replace('1', '').replace('-', '').strip()
            if stage_clean in stage_order:
                return stage_clean
            # Fuzzy match
            for s in stage_order:
                if s.lower() in stage_clean.lower():
                    return s
        return "unknown"

    def get_embeddings(cell_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get embeddings for a cell from loaded dataframes.

        Returns actual dimensions: HLCA (30d), LuCA (10d), Fused (40d).
        NOT truncated to a single latent_dim.

        For spatial spots (cell_id starting with "spatial_"), composes embeddings
        from deconvolution proportions:
        - HLCA embedding = sum(HLCA_proportion[type] * mean_HLCA_embedding[type])
        - LuCA embedding = sum(LuCA_proportion[type] * mean_LuCA_embedding[type])
        """
        z_fused = np.zeros(fused_dim, dtype=np.float32)
        z_hlca = np.zeros(hlca_dim, dtype=np.float32)
        z_luca = np.zeros(luca_dim, dtype=np.float32)

        # For spatial spots, compose embeddings from deconvolution
        if cell_id.startswith("spatial_"):
            spot_id = cell_id[8:]  # Strip "spatial_" prefix

            # Compose HLCA embedding from HLCA deconvolution
            if spot_id in spatial_deconv_hlca and hlca_mean_emb:
                props = spatial_deconv_hlca[spot_id]
                for cell_type, proportion in props.items():
                    if cell_type in hlca_mean_emb and proportion > 0:
                        z_hlca += proportion * hlca_mean_emb[cell_type]

            # Compose LuCA embedding from LuCA deconvolution
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
    print("  Computing pathway/proliferation targets...")
    gene_names = list(snrna.var_names)
    snrna_expr = torch.tensor(
        snrna.X.toarray() if hasattr(snrna.X, "toarray") else snrna.X,
        dtype=torch.float32,
    )
    pathway_targets = compute_pathway_targets(snrna_expr, gene_names, torch.device("cpu"))
    prolif_targets = compute_proliferation_targets(snrna_expr, gene_names, torch.device("cpu"))
    n_pathways = len(PROGENY_PATHWAYS)
    print(f"    Pathway targets: {pathway_targets.shape if pathway_targets is not None else 'None'}")
    print(f"    Proliferation targets: {prolif_targets.shape if prolif_targets is not None else 'None'}")

    # Process snRNA cells
    for idx, cell_id in enumerate(tqdm(snrna.obs_names, desc="Processing snRNA")):
        obs = snrna.obs.iloc[idx]

        donor_id = obs.get("donor_id", obs.get("patient_id", "unknown"))

        # Extract stage from cell_id (more reliable than donor mapping)
        stage = extract_stage_from_cell_id(cell_id)
        if stage == "unknown":
            # Fallback to donor mapping
            stage = donor_to_stage.get(donor_id, "unknown")
        stage_idx = stage_to_idx.get(stage, -1)

        # Get REAL embeddings from reference geometry
        z_fused, z_hlca, z_luca = get_embeddings(cell_id)

        # Get WES features if available - lookup by (patient_id, stage) for proper evolutionary tracking
        wes_row = None
        if wes_df is not None:
            # WES data is per-(patient, stage) lesion, not just per-patient
            mask = (wes_df[wes_id_col] == donor_id) & (wes_df["stage"] == stage)
            matching_rows = wes_df[mask]
            if len(matching_rows) > 0:
                wes_row = matching_rows.iloc[0]
            else:
                # Fallback: try patient-level (any stage) if lesion-specific not found
                mask_patient = wes_df[wes_id_col] == donor_id
                if mask_patient.any():
                    wes_row = wes_df[mask_patient].iloc[0]

        record = {
            "cell_id": cell_id,
            "donor_id": donor_id,
            "stage": stage,
            "stage_idx": stage_idx,
            "cell_type": obs.get("cell_type", "unknown"),
            # Cell cycle scores (for identifying cycling/rare cell states)
            "S_score": float(obs.get("S_score", 0.0)) if pd.notna(obs.get("S_score")) else 0.0,
            "G2M_score": float(obs.get("G2M_score", 0.0)) if pd.notna(obs.get("G2M_score")) else 0.0,
            "phase": str(obs.get("phase", "unknown")) if pd.notna(obs.get("phase")) else "unknown",
            "z_fused": z_fused.tolist(),
            "z_hlca": z_hlca.tolist(),
            "z_luca": z_luca.tolist(),
            "x_spatial": np.nan,  # snRNA doesn't have spatial coords
            "y_spatial": np.nan,
        }

        # Add WES features (8 columns for evolutionary regularization)
        for wes_col in WES_FEATURE_COLS:
            record[wes_col] = float(wes_row[wes_col]) if wes_row is not None and wes_col in wes_row.index else 0.0

        # Add latent dimension columns (preserve actual dimensions: HLCA=30, LuCA=10, Fused=40)
        for dim in range(fused_dim):
            record[f"z_fused_{dim}"] = z_fused[dim]
        for dim in range(hlca_dim):
            record[f"z_hlca_{dim}"] = z_hlca[dim]
        for dim in range(luca_dim):
            record[f"z_luca_{dim}"] = z_luca[dim]

        # Add pathway/proliferation targets (pre-computed from real expression)
        if pathway_targets is not None:
            for p_idx in range(n_pathways):
                record[f"pathway_{p_idx}"] = float(pathway_targets[idx, p_idx].item())
        else:
            for p_idx in range(n_pathways):
                record[f"pathway_{p_idx}"] = 0.0
        record["proliferation_label"] = (
            float(prolif_targets[idx, 0].item()) if prolif_targets is not None else 0.0
        )

        records.append(record)

    # Pre-compute pathway/proliferation targets for spatial spots
    print("  Computing spatial pathway/proliferation targets...")
    spatial_gene_names = list(spatial.var_names)
    spatial_expr = torch.tensor(
        spatial.X.toarray() if hasattr(spatial.X, "toarray") else spatial.X,
        dtype=torch.float32,
    )
    spatial_pathway_targets = compute_pathway_targets(spatial_expr, spatial_gene_names, torch.device("cpu"))
    spatial_prolif_targets = compute_proliferation_targets(spatial_expr, spatial_gene_names, torch.device("cpu"))
    print(f"    Spatial pathway targets: {spatial_pathway_targets.shape if spatial_pathway_targets is not None else 'None'}")
    print(f"    Spatial proliferation targets: {spatial_prolif_targets.shape if spatial_prolif_targets is not None else 'None'}")

    # Process spatial spots
    for idx, spot_id in enumerate(tqdm(spatial.obs_names, desc="Processing spatial")):
        obs = spatial.obs.iloc[idx]

        donor_id = obs.get("donor_id", obs.get("patient_id", "unknown"))

        # Extract stage from spot_id or use donor mapping
        cell_id_for_lookup = f"spatial_{spot_id}"
        stage = extract_stage_from_cell_id(spot_id)
        if stage == "unknown":
            stage = donor_to_stage.get(donor_id, "unknown")
        stage_idx = stage_to_idx.get(stage, -1)

        # Spatial coordinates
        spatial_coords = spatial.obsm["spatial"][idx]

        # Get REAL embeddings (spatial spots may not have embeddings, use zeros as fallback)
        z_fused, z_hlca, z_luca = get_embeddings(cell_id_for_lookup)

        # Get WES features - lookup by (patient_id, stage) for proper evolutionary tracking
        wes_row = None
        if wes_df is not None:
            mask = (wes_df[wes_id_col] == donor_id) & (wes_df["stage"] == stage)
            matching_rows = wes_df[mask]
            if len(matching_rows) > 0:
                wes_row = matching_rows.iloc[0]
            else:
                mask_patient = wes_df[wes_id_col] == donor_id
                if mask_patient.any():
                    wes_row = wes_df[mask_patient].iloc[0]

        record = {
            "cell_id": cell_id_for_lookup,
            "donor_id": donor_id,
            "stage": stage,
            "stage_idx": stage_idx,
            "cell_type": obs.get("cell_type", "mixed"),  # Spatial spots are mixtures
            # Cell cycle scores (NaN for spatial - no single-cell resolution)
            "S_score": np.nan,
            "G2M_score": np.nan,
            "phase": "spatial",  # Mark as spatial spot
            "z_fused": z_fused.tolist(),
            "z_hlca": z_hlca.tolist(),
            "z_luca": z_luca.tolist(),
            "x_spatial": spatial_coords[0],
            "y_spatial": spatial_coords[1],
        }

        # Add WES features (8 columns for evolutionary regularization)
        for wes_col in WES_FEATURE_COLS:
            record[wes_col] = float(wes_row[wes_col]) if wes_row is not None and wes_col in wes_row.index else 0.0

        # Add latent dimension columns (preserve actual dimensions: HLCA=30, LuCA=10, Fused=40)
        for dim in range(fused_dim):
            record[f"z_fused_{dim}"] = z_fused[dim]
        for dim in range(hlca_dim):
            record[f"z_hlca_{dim}"] = z_hlca[dim]
        for dim in range(luca_dim):
            record[f"z_luca_{dim}"] = z_luca[dim]

        # Add pathway/proliferation targets (pre-computed from real expression)
        if spatial_pathway_targets is not None:
            for p_idx in range(n_pathways):
                record[f"pathway_{p_idx}"] = float(spatial_pathway_targets[idx, p_idx].item())
        else:
            for p_idx in range(n_pathways):
                record[f"pathway_{p_idx}"] = 0.0
        record["proliferation_label"] = (
            float(spatial_prolif_targets[idx, 0].item()) if spatial_prolif_targets is not None else 0.0
        )

        # Add DestVI gamma values (intra-cell-type variation) for spatial spots
        if gamma_df is not None and spot_id in gamma_df.index:
            gamma_row = gamma_df.loc[spot_id]
            for g_idx, g_col in enumerate(gamma_df.columns):
                record[g_col] = float(gamma_row[g_col])
        elif gamma_df is not None:
            # Gamma available but spot not found - fill with zeros
            for g_col in gamma_df.columns:
                record[g_col] = 0.0

        records.append(record)

    # Add gamma columns to snRNA records (zeros - gamma is spatial only)
    if gamma_df is not None:
        n_gamma = len(gamma_df.columns)
        print(f"  Adding {n_gamma} gamma columns to snRNA cells (zeros - spatial only)...")
        snrna_records = [r for r in records if not r["cell_id"].startswith("spatial_")]
        for r in snrna_records:
            for g_col in gamma_df.columns:
                r[g_col] = 0.0

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
    backend_results: pd.DataFrame,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """
    Generate neighborhoods.parquet with 9-token structure.

    9 tokens:
    0. Receiver cell
    1-4. Ring 1-4 (spatial neighbors)
    5. HLCA context
    6. LuCA context
    7. Pathway activity
    8. Summary stats
    """
    # Build spatial graph
    print("  Building spatial neighborhood graph...")
    spatial_cells = cells_df[~cells_df["x_spatial"].isna()].copy()

    if len(spatial_cells) == 0:
        print("  Warning: No spatial cells found, skipping neighborhoods")
        return pd.DataFrame()

    # Compute k-NN graph
    from sklearn.neighbors import NearestNeighbors

    coords = spatial_cells[["x_spatial", "y_spatial"]].values
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    records = []

    # OPTIMIZED: Use enumerate + itertuples instead of iterrows (10× faster)
    for pos_idx, row in enumerate(
        tqdm(spatial_cells.itertuples(), total=len(spatial_cells), desc="  Building niches")
    ):
        cell_id = row.cell_id
        donor_id = row.donor_id
        stage = row.stage

        # Get neighbors (exclude self) - use positional index
        neighbor_indices = indices[pos_idx][1:]
        neighbor_distances = distances[pos_idx][1:]

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

        # Tokens 1-4: Rings (5 cells per ring)
        cells_per_ring = 5
        for ring in range(4):
            start = ring * cells_per_ring
            end = min((ring + 1) * cells_per_ring, len(neighbor_indices))
            ring_neighbor_indices = neighbor_indices[start:end]

            if len(ring_neighbor_indices) == 0:
                # Empty ring
                tokens.append(
                    {
                        "token_idx": ring + 1,
                        "token_type": f"ring_{ring + 1}",
                        "n_cells": 0,
                    }
                )
                continue

            ring_neighbors = spatial_cells.iloc[ring_neighbor_indices]

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
                    "mean_distance": float(neighbor_distances[start:end].mean()),
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

        # Token 7: Pathway activity (from spatial backend cell type proportions)
        spot_proportions = (
            backend_results.loc[cell_id] if cell_id in backend_results.index else None
        )

        if spot_proportions is not None:
            # Compute pathway scores from cell type composition
            caf_fraction = spot_proportions.get("Fibroblast", 0.0) + spot_proportions.get(
                "CAF", 0.0
            )
            immune_fraction = spot_proportions.get("Macrophage", 0.0) + spot_proportions.get(
                "T_cell", 0.0
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
                "diversity": len(spatial_cells.iloc[neighbor_indices]["cell_type"].unique()),
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

    For LUAD: Normal → Preneoplastic → Invasive → Advanced
    """
    stages = list(stage_definitions.keys())
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
        "--spatial_backend_dir", type=str, required=True, help="Spatial backend results directory"
    )
    parser.add_argument(
        "--reference_geometry", type=str, required=True,
        help="Path to reference_geometry directory (contains fused_embedding.parquet, etc.)"
    )
    parser.add_argument(
        "--clonal_patterns", type=str, default=None,
        help="Path to clonal_patterns.json from run_clonal_extraction.py (for H3 validation)"
    )
    parser.add_argument(
        "--hlca_deconv_dir", type=str, default=None,
        help="Path to HLCA deconvolution results (for spatial embedding composition)"
    )
    parser.add_argument(
        "--luca_deconv_dir", type=str, default=None,
        help="Path to LuCA deconvolution results (for spatial embedding composition)"
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
        spatial_backend_dir=Path(args.spatial_backend_dir),
        output_dir=Path(args.output_dir),
        stage_definitions=stage_definitions,
        n_folds=args.n_folds,
        reference_geometry_dir=Path(args.reference_geometry),
        clonal_patterns_path=Path(args.clonal_patterns) if args.clonal_patterns else None,
        hlca_deconv_dir=Path(args.hlca_deconv_dir) if args.hlca_deconv_dir else None,
        luca_deconv_dir=Path(args.luca_deconv_dir) if args.luca_deconv_dir else None,
    )


if __name__ == "__main__":
    main()
