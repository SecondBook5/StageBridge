#!/usr/bin/env python3
"""Plot UMAP and PHATE of spatial and snRNA embeddings to verify alignment.

Usage:
    python scripts/plot_alignment_umap.py \
        --embeddings-dir $DATA/processed/luad_evo/reference_embeddings \
        --spatial-h5ad $DATA/processed/spatial/spatial_merged.h5ad \
        --snrna-h5ad $DATA/processed/snrna/snrna_processed.h5ad \
        --output-dir $DATA/processed/luad_evo/reference_embeddings/figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import phate
import scanpy as sc
import anndata as ad


def load_embeddings_with_metadata(embeddings_dir, spatial_h5ad=None, snrna_h5ad=None):
    """Load embeddings and optionally join metadata from h5ad files."""
    spatial = pd.read_parquet(embeddings_dir / "spatial_fused_direct.parquet")
    snrna = pd.read_parquet(embeddings_dir / "snrna_fused_direct.parquet")

    if spatial_h5ad and Path(spatial_h5ad).exists():
        adata = ad.read_h5ad(spatial_h5ad, backed='r')
        obs = adata.obs.copy()
        for col in ['cell_type', 'cell_type_assigned', 'donor', 'sample', 'patient_id', 'stage']:
            if col in obs.columns:
                spatial[col] = obs.loc[spatial.index, col].values
        adata.file.close()

    if snrna_h5ad and Path(snrna_h5ad).exists():
        adata = ad.read_h5ad(snrna_h5ad, backed='r')
        obs = adata.obs.copy()
        for col in ['cell_type', 'cell_type_assigned', 'donor', 'sample', 'patient_id', 'stage']:
            if col in obs.columns:
                snrna[col] = obs.loc[snrna.index, col].values
        adata.file.close()

    return spatial, snrna


def sample_data(spatial, snrna, emb_cols, n_sample, seed):
    """Sample and combine data from both modalities."""
    n_spatial = min(n_sample, len(spatial))
    n_snrna = min(n_sample, len(snrna))

    spatial_sampled = spatial.sample(n_spatial, random_state=seed)
    snrna_sampled = snrna.sample(n_snrna, random_state=seed)

    X_spatial = spatial_sampled[emb_cols].values
    X_snrna = snrna_sampled[emb_cols].values

    X = np.vstack([X_spatial, X_snrna])
    modality = ["Spatial"] * n_spatial + ["snRNA"] * n_snrna

    meta = pd.DataFrame({
        "modality": modality,
        "cell_type_hlca": pd.concat([
            spatial_sampled.get("cell_type_hlca", pd.Series(["unknown"] * n_spatial, index=spatial_sampled.index)),
            snrna_sampled.get("cell_type_hlca", pd.Series(["unknown"] * n_snrna, index=snrna_sampled.index))
        ]).values,
        "cell_type_luca": pd.concat([
            spatial_sampled.get("cell_type_luca", pd.Series(["unknown"] * n_spatial, index=spatial_sampled.index)),
            snrna_sampled.get("cell_type_luca", pd.Series(["unknown"] * n_snrna, index=snrna_sampled.index))
        ]).values,
        "donor": pd.concat([
            spatial_sampled.get("donor", spatial_sampled.get("patient_id", pd.Series(["unknown"] * n_spatial, index=spatial_sampled.index))),
            snrna_sampled.get("donor", snrna_sampled.get("patient_id", pd.Series(["unknown"] * n_snrna, index=snrna_sampled.index)))
        ]).values,
        "stage": pd.concat([
            spatial_sampled.get("stage", pd.Series(["unknown"] * n_spatial, index=spatial_sampled.index)),
            snrna_sampled.get("stage", pd.Series(["unknown"] * n_snrna, index=snrna_sampled.index))
        ]).values,
    })

    return X, meta, spatial_sampled.index.tolist() + snrna_sampled.index.tolist()


def plot_by_modality(emb, meta, method_name, output_path):
    """Plot colored by modality (spatial vs snRNA)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for modality, color in [("Spatial", "#1f77b4"), ("snRNA", "#ff7f0e")]:
        mask = meta["modality"] == modality
        ax.scatter(emb[mask, 0], emb[mask, 1], s=1, alpha=0.5, label=modality, c=color)
    ax.legend(markerscale=5)
    ax.set_xlabel(f"{method_name}1")
    ax.set_ylabel(f"{method_name}2")
    ax.set_title(f"Direct Inference Alignment - {method_name} (Fused 40d)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_by_category(emb, meta, category, method_name, output_path):
    """Plot colored by a categorical variable."""
    fig, ax = plt.subplots(figsize=(12, 10))

    categories = meta[category].dropna().unique()
    n_cats = len(categories)

    if n_cats <= 20:
        cmap = plt.cm.get_cmap("tab20", n_cats)
    else:
        cmap = plt.cm.get_cmap("gist_ncar", n_cats)

    for i, cat in enumerate(sorted(categories)):
        mask = meta[category] == cat
        ax.scatter(emb[mask, 0], emb[mask, 1], s=1, alpha=0.5, label=cat, c=[cmap(i)])

    if n_cats <= 15:
        ax.legend(markerscale=5, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        ax.legend(markerscale=5, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6, ncol=2)

    ax.set_xlabel(f"{method_name}1")
    ax.set_ylabel(f"{method_name}2")
    ax.set_title(f"{category.replace('_', ' ').title()} - {method_name} (Fused 40d)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot alignment UMAP and PHATE")
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--spatial-h5ad", type=Path, default=None,
                        help="Path to spatial h5ad for metadata (cell_type, donor)")
    parser.add_argument("--snrna-h5ad", type=Path, default=None,
                        help="Path to snRNA h5ad for metadata (cell_type, donor)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-phate", action="store_true", help="Skip PHATE (faster)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.embeddings_dir / "figures"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings_dir}")
    spatial, snrna = load_embeddings_with_metadata(
        args.embeddings_dir, args.spatial_h5ad, args.snrna_h5ad
    )

    print(f"  Spatial: {len(spatial):,}")
    print(f"  snRNA: {len(snrna):,}")

    emb_cols = [c for c in spatial.columns if c.startswith("hlca_latent_") or c.startswith("luca_latent_")]
    print(f"  Embedding dim: {len(emb_cols)}")

    X, meta, cell_ids = sample_data(spatial, snrna, emb_cols, args.n_sample, args.seed)
    print(f"  Sampled: {len(X):,} cells")

    # Leiden clustering on fused embeddings
    print("Running Leiden clustering...")
    adata_tmp = ad.AnnData(X)
    sc.pp.neighbors(adata_tmp, n_neighbors=30, use_rep='X')
    sc.tl.leiden(adata_tmp, resolution=1.0)
    meta["leiden"] = adata_tmp.obs["leiden"].values

    # UMAP
    print("Running UMAP...")
    umap_model = UMAP(n_neighbors=30, min_dist=0.3, random_state=args.seed)
    umap_emb = umap_model.fit_transform(X)

    plot_by_modality(umap_emb, meta, "UMAP", args.output_dir / "alignment_umap_modality.png")

    if "cell_type_hlca" in meta.columns and meta["cell_type_hlca"].nunique() > 1:
        plot_by_category(umap_emb, meta, "cell_type_hlca", "UMAP", args.output_dir / "alignment_umap_celltype_hlca.png")

    if "cell_type_luca" in meta.columns and meta["cell_type_luca"].nunique() > 1:
        plot_by_category(umap_emb, meta, "cell_type_luca", "UMAP", args.output_dir / "alignment_umap_celltype_luca.png")

    if "donor" in meta.columns and meta["donor"].nunique() > 1:
        plot_by_category(umap_emb, meta, "donor", "UMAP", args.output_dir / "alignment_umap_donor.png")

    if "stage" in meta.columns and meta["stage"].nunique() > 1:
        plot_by_category(umap_emb, meta, "stage", "UMAP", args.output_dir / "alignment_umap_stage.png")

    if "leiden" in meta.columns:
        plot_by_category(umap_emb, meta, "leiden", "UMAP", args.output_dir / "alignment_umap_leiden.png")

    # PHATE
    if not args.skip_phate:
        print("Running PHATE...")
        phate_model = phate.PHATE(n_components=2, random_state=args.seed, n_jobs=-1)
        phate_emb = phate_model.fit_transform(X)

        plot_by_modality(phate_emb, meta, "PHATE", args.output_dir / "alignment_phate_modality.png")

        if "cell_type_hlca" in meta.columns and meta["cell_type_hlca"].nunique() > 1:
            plot_by_category(phate_emb, meta, "cell_type_hlca", "PHATE", args.output_dir / "alignment_phate_celltype_hlca.png")

        if "cell_type_luca" in meta.columns and meta["cell_type_luca"].nunique() > 1:
            plot_by_category(phate_emb, meta, "cell_type_luca", "PHATE", args.output_dir / "alignment_phate_celltype_luca.png")

        if "donor" in meta.columns and meta["donor"].nunique() > 1:
            plot_by_category(phate_emb, meta, "donor", "PHATE", args.output_dir / "alignment_phate_donor.png")

        if "stage" in meta.columns and meta["stage"].nunique() > 1:
            plot_by_category(phate_emb, meta, "stage", "PHATE", args.output_dir / "alignment_phate_stage.png")

        if "leiden" in meta.columns:
            plot_by_category(phate_emb, meta, "leiden", "PHATE", args.output_dir / "alignment_phate_leiden.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
