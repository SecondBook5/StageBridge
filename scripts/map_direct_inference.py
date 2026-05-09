#!/usr/bin/env python
"""Map both spatial and snRNA through LuCA using DIRECT inference (no surgery).

Surgery fine-tunes separately for each modality, creating different spaces.
Direct inference uses the same frozen model for both, keeping them aligned.

Usage:
    python scripts/map_direct_inference.py \
        --spatial $DATA/processed/luad_evo/spatial_merged.h5ad \
        --snrna $DATA/processed/luad_evo/snrna_qc_normalized.h5ad \
        --luca-model $DATA/references/luca/retrained_model_v3/scanvi_model_hlca_format \
        --output-dir $DATA/processed/luad_evo/reference_geometry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def map_to_luca_direct(
    adata: ad.AnnData,
    model,
    model_var_names: list,
    batch_key: str = "dataset",
    batch_value: str = "query",
) -> np.ndarray:
    """Map adata through LuCA model using direct inference (no surgery)."""
    query = adata.copy()

    # Set batch to existing category (use first one from model)
    ref_batch_cats = list(model.adata_manager.registry["field_registries"]["batch"]["state_registry"]["categorical_mapping"])
    query.obs[batch_key] = pd.Categorical(
        [ref_batch_cats[0]] * query.n_obs,  # Use first ref batch, not "query"
        categories=ref_batch_cats
    )

    # Align genes
    query_genes = set(query.var_names)
    model_genes = set(model_var_names)

    # Check if we need symbol -> ensembl conversion
    overlap = query_genes & model_genes
    if len(overlap) < 100:
        # Try conversion
        if hasattr(model, 'adata') and model.adata is not None and "feature_name" in model.adata.var.columns:
            symbol_to_ensembl = dict(zip(model.adata.var["feature_name"], model.adata.var_names))
            new_names = [symbol_to_ensembl.get(g, g) for g in query.var_names]
            query.var_names = new_names
            overlap = set(query.var_names) & model_genes

    print(f"    Gene overlap: {len(overlap)}/{len(model_var_names)}")

    # Subset to model genes
    common = [g for g in model_var_names if g in query.var_names]
    query = query[:, common].copy()

    # Pad missing genes with zeros
    if len(common) < len(model_var_names):
        missing = [g for g in model_var_names if g not in common]
        from scipy import sparse
        zeros = sparse.csr_matrix((query.n_obs, len(missing)))
        missing_adata = ad.AnnData(X=zeros, obs=query.obs.copy(), var=pd.DataFrame(index=missing))
        query = ad.concat([query, missing_adata], axis=1)
        query = query[:, model_var_names].copy()

    # Get latent
    latent = model.get_latent_representation(query, batch_size=1024)
    return latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--snrna", type=Path, required=True)
    parser.add_argument("--luca-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import scvi

    print("=" * 60)
    print("Direct Inference Mapping (No Surgery)")
    print("=" * 60)

    # Load model with bundled adata
    print(f"\nLoading LuCA model from {args.luca_model}...")
    model_adata = ad.read_h5ad(args.luca_model / "adata.h5ad")
    model = scvi.model.SCANVI.load(str(args.luca_model), adata=model_adata)
    model_var_names = list(model_adata.var_names)
    print(f"  Model loaded: {model.module.n_latent}d latent, {len(model_var_names)} genes")

    # Load spatial
    print(f"\nLoading spatial from {args.spatial}...")
    spatial = ad.read_h5ad(args.spatial)
    print(f"  {spatial.n_obs:,} spots")

    # Load snRNA
    print(f"\nLoading snRNA from {args.snrna}...")
    snrna = ad.read_h5ad(args.snrna)
    print(f"  {snrna.n_obs:,} cells")

    # Map spatial
    print("\nMapping spatial (direct inference)...")
    spatial_latent = map_to_luca_direct(spatial, model, model_var_names)
    print(f"  Spatial latent: {spatial_latent.shape}")

    # Map snRNA
    print("\nMapping snRNA (direct inference)...")
    snrna_latent = map_to_luca_direct(snrna, model, model_var_names)
    print(f"  snRNA latent: {snrna_latent.shape}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spatial_df = pd.DataFrame(
        spatial_latent,
        index=spatial.obs_names,
        columns=[f"luca_latent_{i}" for i in range(spatial_latent.shape[1])]
    )
    spatial_df.to_parquet(args.output_dir / "spatial_luca_direct.parquet")
    print(f"\nSaved spatial: {args.output_dir / 'spatial_luca_direct.parquet'}")

    snrna_df = pd.DataFrame(
        snrna_latent,
        index=snrna.obs_names,
        columns=[f"luca_latent_{i}" for i in range(snrna_latent.shape[1])]
    )
    snrna_df.to_parquet(args.output_dir / "snrna_luca_direct.parquet")
    print(f"Saved snRNA: {args.output_dir / 'snrna_luca_direct.parquet'}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
