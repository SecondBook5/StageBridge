#!/usr/bin/env python3
"""Audit data sources for StageBridge data preparation.

Reports what exists and what's missing vs. contract requirements.
"""

from pathlib import Path

import pandas as pd

DATA = Path("/data1/chaunzt1/stagebridge/processed/luad_evo")
CANONICAL = DATA / "canonical"
REF_GEOM = DATA / "reference_geometry"
SPATIAL_BENCH = Path("/data1/chaunzt1/stagebridge/runs/spatial_benchmark")

# Contract requirements
CELLS_REQUIRED = ["cell_id", "donor_id", "stage", "data_type"]
CELLS_EMBEDDINGS = ["z_fused", "z_hlca", "z_luca"]
CELLS_CYCLE = ["S_score", "G2M_score"]
CELLS_WES = ["tmb", "kras_mut", "egfr_mut", "tp53_mut", "stk11_mut", "keap1_mut", "smad4_mut", "braf_mut"]
CELLS_SPATIAL = ["x_spatial", "y_spatial"]

NHOOD_REQUIRED = ["cell_id", "donor_id"]
NHOOD_TOKENS = ["tokens"]  # Format A
NHOOD_RINGS = ["ring_1_cells", "ring_2_cells", "ring_3_cells", "ring_4_cells", "receiver_z", "hlca_z", "luca_z"]  # Format B


def check_file(path: Path, description: str) -> bool:
    exists = path.exists()
    size = f"{path.stat().st_size / 1e9:.2f} GB" if exists else "N/A"
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {description}: {path.name} ({size})")
    return exists


def check_columns(df: pd.DataFrame, required: list, name: str, check_values: bool = True) -> list:
    missing = [c for c in required if c not in df.columns]
    present = [c for c in required if c in df.columns]
    if missing:
        print(f"    MISSING {name}: {missing}")
    if present:
        print(f"    OK {name}: {present}")

    # Check for NaN and value quality
    if check_values and present:
        for col in present:
            n_total = len(df)
            n_nan = df[col].isna().sum()
            n_valid = n_total - n_nan
            pct_nan = 100 * n_nan / n_total if n_total > 0 else 0

            if pct_nan > 50:
                status = "BAD"
            elif pct_nan > 10:
                status = "WARN"
            elif pct_nan > 0:
                status = "ok"
            else:
                status = "OK"

            # Get value stats for numeric columns
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                valid_vals = df[col].dropna()
                if len(valid_vals) > 0:
                    vmin, vmax = valid_vals.min(), valid_vals.max()
                    vmean = valid_vals.mean()
                    print(f"      [{status}] {col}: {n_valid:,}/{n_total:,} valid ({pct_nan:.1f}% NaN), range=[{vmin:.3g}, {vmax:.3g}], mean={vmean:.3g}")
                else:
                    print(f"      [BAD] {col}: ALL NaN")
            elif df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                n_unique = df[col].nunique()
                top_vals = df[col].value_counts().head(3).to_dict()
                print(f"      [{status}] {col}: {n_valid:,}/{n_total:,} valid ({pct_nan:.1f}% NaN), {n_unique} unique, top: {top_vals}")
            else:
                print(f"      [{status}] {col}: {n_valid:,}/{n_total:,} valid ({pct_nan:.1f}% NaN)")

    return missing


def check_embedding_quality(df: pd.DataFrame, col_or_prefix: str, expected_dim: int):
    """Check embedding column(s) for quality."""
    # Check if stored as single array column or multiple columns
    if col_or_prefix in df.columns:
        # Single column with arrays
        sample = df[col_or_prefix].iloc[0]
        if hasattr(sample, "__len__"):
            dim = len(sample)
            if dim != expected_dim:
                print(f"      [WARN] {col_or_prefix}: dimension {dim}, expected {expected_dim}")
            else:
                print(f"      [OK] {col_or_prefix}: dimension {dim}")

            # Check for NaN in embeddings
            import numpy as np
            n_nan = sum(1 for x in df[col_or_prefix] if x is None or (hasattr(x, '__iter__') and any(np.isnan(v) for v in x)))
            if n_nan > 0:
                print(f"      [WARN] {col_or_prefix}: {n_nan} rows with NaN values in embedding")

            # Check value range
            all_vals = []
            for x in df[col_or_prefix].head(1000):
                if x is not None and hasattr(x, '__iter__'):
                    all_vals.extend(x)
            if all_vals:
                import numpy as np
                all_vals = np.array(all_vals)
                print(f"      [INFO] {col_or_prefix}: value range [{all_vals.min():.3g}, {all_vals.max():.3g}], mean={all_vals.mean():.3g}")
        else:
            print(f"      [BAD] {col_or_prefix}: not an array (scalar value)")
    else:
        # Check for individual columns (z_fused_0, z_fused_1, etc.)
        cols = [f"{col_or_prefix}_{i}" for i in range(expected_dim)]
        present = [c for c in cols if c in df.columns]
        if len(present) == expected_dim:
            print(f"      [OK] {col_or_prefix}_*: all {expected_dim} columns present")
            # Check NaN
            n_nan = df[present].isna().any(axis=1).sum()
            if n_nan > 0:
                pct = 100 * n_nan / len(df)
                print(f"      [WARN] {col_or_prefix}_*: {n_nan} rows ({pct:.1f}%) with NaN")
            # Check range
            vals = df[present].values.flatten()
            import numpy as np
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                print(f"      [INFO] {col_or_prefix}_*: range [{vals.min():.3g}, {vals.max():.3g}], mean={vals.mean():.3g}")
        elif len(present) > 0:
            print(f"      [BAD] {col_or_prefix}_*: only {len(present)}/{expected_dim} columns")
        else:
            print(f"      [MISSING] {col_or_prefix}: neither array column nor individual columns found")


def check_h5ad(path: Path, description: str):
    if not path.exists():
        print(f"  [MISSING] {description}: {path.name}")
        return

    import anndata as ad
    adata = ad.read_h5ad(path, backed="r")
    print(f"  [OK] {description}: {path.name}")
    print(f"    Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"    obs columns ({len(adata.obs.columns)}): {sorted(adata.obs.columns.tolist())[:20]}...")
    print(f"    obsm keys: {list(adata.obsm.keys())}")

    # Check for key columns
    has_cycle = "S_score" in adata.obs.columns and "G2M_score" in adata.obs.columns
    has_stage = "stage" in adata.obs.columns
    has_donor = "donor_id" in adata.obs.columns
    print(f"    Cell cycle scores: {'YES' if has_cycle else 'NO'}")
    print(f"    Stage column: {'YES' if has_stage else 'NO'}")
    print(f"    Donor column: {'YES' if has_donor else 'NO'}")

    # Check obsm for embeddings
    for key in ["X_scANVI", "X_scanvi_emb", "X_pca"]:
        if key in adata.obsm:
            print(f"    {key}: {adata.obsm[key].shape}")

    adata.file.close()


def main():
    print("=" * 70)
    print("StageBridge Data Source Audit")
    print("=" * 70)

    # H5AD files
    print("\n[1] H5AD SOURCE FILES")
    check_h5ad(DATA / "snrna_with_celltypes.h5ad", "snRNA with cell types")
    check_h5ad(DATA / "spatial_merged.h5ad", "Spatial merged")

    # Reference geometry
    print("\n[2] REFERENCE GEOMETRY")
    if REF_GEOM.exists():
        for f in sorted(REF_GEOM.glob("*")):
            if f.is_file():
                size = f"{f.stat().st_size / 1e6:.1f} MB"
                print(f"  [OK] {f.name} ({size})")

        # Check fused embedding
        fused = REF_GEOM / "fused_embedding.parquet"
        if fused.exists():
            df = pd.read_parquet(fused)
            print(f"    fused_embedding: {df.shape[0]:,} cells, columns: {list(df.columns)[:10]}...")
    else:
        print(f"  [MISSING] {REF_GEOM}")

    # WES features
    print("\n[3] WES FEATURES")
    wes_path = DATA / "wes_features.parquet"
    if check_file(wes_path, "WES features"):
        df = pd.read_parquet(wes_path)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")

    # DestVI results
    print("\n[4] DESTVI DECONVOLUTION")
    for ref in ["hlca", "luca"]:
        destvi = SPATIAL_BENCH / ref / "destvi" / "cell_type_proportions.parquet"
        if check_file(destvi, f"DestVI {ref.upper()}"):
            df = pd.read_parquet(destvi)
            print(f"    Shape: {df.shape[0]:,} spots x {df.shape[1]} cell types")

    # Current canonical artifacts
    print("\n[5] CURRENT CANONICAL ARTIFACTS")
    cells_path = CANONICAL / "cells.parquet"
    nhood_path = CANONICAL / "neighborhoods.parquet"
    splits_path = CANONICAL / "split_manifest.json"

    if check_file(cells_path, "cells.parquet"):
        df = pd.read_parquet(cells_path)
        print(f"    Shape: {df.shape[0]:,} cells x {df.shape[1]} columns")
        check_columns(df, CELLS_REQUIRED, "required", check_values=True)
        check_columns(df, CELLS_CYCLE, "cell cycle", check_values=True)
        check_columns(df, CELLS_WES, "WES", check_values=True)
        check_columns(df, CELLS_SPATIAL, "spatial coords", check_values=True)

        # Check embeddings with quality checks
        print("    Embedding quality:")
        check_embedding_quality(df, "z_fused", 40)
        check_embedding_quality(df, "z_hlca", 30)
        check_embedding_quality(df, "z_luca", 10)

    if check_file(nhood_path, "neighborhoods.parquet"):
        df = pd.read_parquet(nhood_path)
        print(f"    Shape: {df.shape[0]:,} neighborhoods x {df.shape[1]} columns")
        check_columns(df, NHOOD_REQUIRED, "required", check_values=True)

        has_tokens = "tokens" in df.columns
        has_rings = all(c in df.columns for c in NHOOD_RINGS)
        print(f"    Format A (tokens column): {'YES' if has_tokens else 'NO'}")
        print(f"    Format B (ring columns): {'YES' if has_rings else 'NO'}")

        # Check ring quality
        if has_rings:
            print("    Ring quality:")
            for ring_col in ["ring_1_cells", "ring_2_cells", "ring_3_cells", "ring_4_cells"]:
                if ring_col in df.columns:
                    # Count empty rings
                    n_empty = sum(1 for x in df[ring_col] if x is None or (hasattr(x, '__len__') and len(x) == 0))
                    pct_empty = 100 * n_empty / len(df)
                    # Get average ring size
                    sizes = [len(x) for x in df[ring_col] if x is not None and hasattr(x, '__len__')]
                    avg_size = sum(sizes) / len(sizes) if sizes else 0
                    status = "OK" if pct_empty < 20 else "WARN" if pct_empty < 50 else "BAD"
                    print(f"      [{status}] {ring_col}: {pct_empty:.1f}% empty, avg size={avg_size:.1f}")

            # Check receiver/reference embeddings
            print("    Neighborhood embedding quality:")
            check_embedding_quality(df, "receiver_z", 40)
            check_embedding_quality(df, "hlca_z", 30)
            check_embedding_quality(df, "luca_z", 10)

    check_file(splits_path, "split_manifest.json")

    # Derived data products (from run_full_data_prep.sh)
    print("\n[6] DERIVED DATA PRODUCTS")

    derived = {
        "LIANA L-R interactions": CANONICAL / "liana_interactions.parquet",
        "Biological features": CANONICAL / "biological_features.parquet",
        "Progression scores": CANONICAL / "progression" / "progression_scores.parquet",
    }
    for name, path in derived.items():
        if check_file(path, name):
            df = pd.read_parquet(path)
            print(f"    Shape: {df.shape}")

    print("\n[7] SPATIAL STATISTICS (Squidpy)")
    spatial_stats = CANONICAL / "spatial_stats"
    if spatial_stats.exists():
        for f in sorted(spatial_stats.glob("*.parquet")):
            size = f"{f.stat().st_size / 1e6:.1f} MB"
            print(f"  [OK] {f.name} ({size})")
    else:
        print(f"  [MISSING] {spatial_stats}")

    print("\n[8] DIFFERENTIAL EXPRESSION")
    de_dir = CANONICAL / "de_analysis"
    if de_dir.exists():
        de_files = list(de_dir.glob("de_stage_*.parquet"))
        print(f"  [OK] {len(de_files)} DE files")
        for f in de_files[:3]:
            print(f"    {f.name}")
        if len(de_files) > 3:
            print(f"    ... and {len(de_files) - 3} more")
    else:
        print(f"  [MISSING] {de_dir}")

    print("\n[9] GENE SIGNATURES")
    sig_path = CANONICAL / "signatures" / "gene_signatures.parquet"
    if check_file(sig_path, "Gene signatures"):
        df = pd.read_parquet(sig_path)
        sig_cols = [c for c in df.columns if c != "cell_id"]
        print(f"    Signatures: {sig_cols}")

    print("\n[10] DECOUPLER ACTIVITY")
    activity_dir = CANONICAL / "activity"
    activity_files = {
        "TF activity (CollecTRI)": activity_dir / "tf_activity_collectri.parquet",
        "Pathway activity (PROGENy)": activity_dir / "pathway_activity_progeny.parquet",
    }
    for name, path in activity_files.items():
        if check_file(path, name):
            df = pd.read_parquet(path)
            print(f"    Shape: {df.shape}")

    print("\n[11] TRAJECTORIES")
    traj_dir = CANONICAL / "trajectories"
    traj_files = {
        "Diffusion pseudotime": traj_dir / "diffusion_pseudotime.parquet",
        "Diffmap embedding": traj_dir / "diffmap_embedding.parquet",
    }
    for name, path in traj_files.items():
        check_file(path, name)

    print("\n[12] EMBEDDINGS & CLUSTERING")
    emb_dir = CANONICAL / "embeddings"
    emb_files = {
        "UMAP": emb_dir / "umap_embedding.parquet",
        "PHATE": emb_dir / "phate_embedding.parquet",
        "Clustering": emb_dir / "clustering.parquet",
    }
    for name, path in emb_files.items():
        check_file(path, name)

    print("\n[13] COMMUNICATION")
    comm_dir = CANONICAL / "communication"
    comm_files = {
        "Communication matrix": comm_dir / "communication_matrix.parquet",
        "Top interactions": comm_dir / "top_interactions.parquet",
    }
    for name, path in comm_files.items():
        check_file(path, name)

    print("\n[14] NICHE PHENOTYPES")
    niche_dir = CANONICAL / "niche_phenotypes"
    niche_files = {
        "Spot phenotypes": niche_dir / "spot_niche_phenotypes.parquet",
        "Phenotype centers": niche_dir / "phenotype_centers.parquet",
    }
    for name, path in niche_files.items():
        check_file(path, name)

    print("\n[15] SUMMARY STATISTICS")
    stats_dir = CANONICAL / "summary_stats"
    if stats_dir.exists():
        for f in sorted(stats_dir.glob("*.parquet")):
            print(f"  [OK] {f.name}")
    else:
        print(f"  [MISSING] {stats_dir}")

    print("\n[16] SCENIC REGULONS")
    scenic_dir = CANONICAL / "scenic"
    scenic_files = {
        "AUCell scores": scenic_dir / "aucell_scores.parquet",
        "Regulon scores": scenic_dir / "regulon_scores.parquet",
    }
    for name, path in scenic_files.items():
        check_file(path, name)

    print("\n[17] QC METRICS")
    qc_path = CANONICAL / "qc" / "snrna_qc_metrics.parquet"
    check_file(qc_path, "snRNA QC metrics")

    print("\n[18] RARE CELL SIGNATURES")
    rare_path = CANONICAL / "rare_cells" / "rare_cell_signatures.parquet"
    check_file(rare_path, "Rare cell signatures")

    print("\n[19] VISIUM ANALYSIS")
    visium_dir = CANONICAL / "visium"
    visium_files = {
        "Spot deconvolution": visium_dir / "spot_deconvolution_destvi.parquet",
        "Colocalization": visium_dir / "celltype_colocalization_corr.parquet",
    }
    for name, path in visium_files.items():
        check_file(path, name)

    print("\n[20] EXPRESSION")
    expr_path = CANONICAL / "expression" / "key_genes_expression.parquet"
    check_file(expr_path, "Key genes expression")

    print("\n[21] GSEA PATHWAYS")
    gsea_dir = CANONICAL / "pathways"
    if gsea_dir.exists():
        gsea_files = list(gsea_dir.glob("gsea_*.parquet"))
        print(f"  [OK] {len(gsea_files)} GSEA files") if gsea_files else print("  [MISSING] No GSEA files")
    else:
        print(f"  [MISSING] {gsea_dir}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What's needed to build proper canonical artifacts")
    print("=" * 70)
    print("""
CORE ARTIFACTS (must pass contract validation):

1. CELLS.PARQUET needs:
   - cell_id, donor_id, stage, data_type (required)
   - z_fused (40d), z_hlca (30d), z_luca (10d) embeddings
   - S_score, G2M_score (cell cycle)
   - WES: tmb, kras_mut, egfr_mut, tp53_mut, etc.
   - x_spatial, y_spatial (for spatial cells)

2. NEIGHBORHOODS.PARQUET needs:
   - cell_id, donor_id (required)
   - ring_1_cells..ring_4_cells OR tokens column
   - receiver_z, hlca_z, luca_z per cell

3. SPLIT_MANIFEST.JSON needs:
   - Donor-held-out 5-fold CV splits
   - No donor overlap between train/val/test

DERIVED PRODUCTS (run_full_data_prep.sh):
   - LIANA L-R interactions
   - Squidpy spatial stats
   - DE analysis by stage
   - Gene signatures (AP1, hypoxia, etc.)
   - decoupleR TF/pathway activity
   - Trajectories (diffusion pseudotime)
   - Embeddings (UMAP, PHATE) + clustering
   - Niche phenotypes (HMRF)
   - SCENIC regulons (separate env)
""")


if __name__ == "__main__":
    main()
