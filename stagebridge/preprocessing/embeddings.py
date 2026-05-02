"""Dimensionality reduction and clustering."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def compute_umap(
    adata: "ad.AnnData",
    n_neighbors: int = 30,
    n_pcs: int = 50,
) -> pd.DataFrame:
    """Compute UMAP embedding.

    Args:
        adata: AnnData object
        n_neighbors: Number of neighbors for UMAP
        n_pcs: Number of PCs to use

    Returns:
        DataFrame with UMAP1, UMAP2 columns
    """
    import scanpy as sc

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_pcs)

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")

    sc.tl.umap(adata)

    return pd.DataFrame(
        adata.obsm["X_umap"],
        index=adata.obs.index,
        columns=["UMAP1", "UMAP2"],
    )


def compute_phate(
    adata: "ad.AnnData",
    n_components: int = 2,
    n_pcs: int = 50,
) -> pd.DataFrame | None:
    """Compute PHATE embedding.

    Args:
        adata: AnnData object
        n_components: PHATE dimensions
        n_pcs: Number of PCs to use

    Returns:
        DataFrame with PHATE1, PHATE2 columns, or None if PHATE not installed
    """
    try:
        import phate
    except ImportError:
        print("PHATE not installed")
        return None

    import scanpy as sc

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_pcs)

    phate_op = phate.PHATE(n_components=n_components, n_jobs=-1, random_state=42)
    embedding = phate_op.fit_transform(adata.obsm["X_pca"][:, :n_pcs])
    adata.obsm["X_phate"] = embedding

    return pd.DataFrame(
        embedding,
        index=adata.obs.index,
        columns=[f"PHATE{i+1}" for i in range(n_components)],
    )


def compute_leiden_clustering(
    adata: "ad.AnnData",
    resolutions: list[float] | None = None,
    use_rep: str | None = None,
    neighbors_key: str | None = None,
) -> pd.DataFrame:
    """Compute Leiden clustering at multiple resolutions.

    Args:
        adata: AnnData object with neighbors computed
        resolutions: List of resolution values
        use_rep: Representation to use for neighbors
        neighbors_key: Key for neighbors in uns

    Returns:
        DataFrame with leiden_{res} columns
    """
    import scanpy as sc

    if resolutions is None:
        resolutions = [0.3, 0.5, 0.8, 1.0, 1.5]

    # Compute neighbors if needed
    if neighbors_key is None and "neighbors" not in adata.uns:
        if use_rep:
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=30)
        else:
            if "X_pca" not in adata.obsm:
                sc.pp.pca(adata, n_comps=50)
            sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")

    results = {"cell_id": adata.obs.index.tolist()}

    for res in resolutions:
        key = f"leiden_{res}" if neighbors_key is None else f"leiden_{neighbors_key}_{res}"
        sc.tl.leiden(
            adata,
            resolution=res,
            key_added=key,
            neighbors_key=neighbors_key,
        )
        results[key] = adata.obs[key].values

    return pd.DataFrame(results)


def run_embeddings(
    h5ad_path: str | Path,
    output_dir: str | Path,
    compute_phate_embedding: bool = True,
    resolutions: list[float] | None = None,
) -> dict[str, Path]:
    """Run full embeddings and clustering pipeline.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        compute_phate_embedding: Whether to compute PHATE
        resolutions: Clustering resolutions

    Returns:
        Dict of output file paths
    """
    import scanpy as sc

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolutions is None:
        resolutions = [0.3, 0.5, 0.8, 1.0, 1.5]

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells")

    outputs = {}
    cell_type_key = "cell_type_luca" if "cell_type_luca" in adata.obs.columns else "cell_type"

    # PCA
    if "X_pca" not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, n_comps=50)

    # UMAP
    print("Computing UMAP...")
    umap_df = compute_umap(adata)
    if "stage" in adata.obs.columns:
        umap_df["stage"] = adata.obs["stage"].values
    if cell_type_key in adata.obs.columns:
        umap_df["cell_type"] = adata.obs[cell_type_key].values

    # PHATE (with checkpoint)
    phate_checkpoint = output_dir / "phate_embedding.parquet"
    if compute_phate_embedding:
        if phate_checkpoint.exists():
            print("Loading PHATE from checkpoint...")
            phate_df = pd.read_parquet(phate_checkpoint)
            adata.obsm["X_phate"] = phate_df[["PHATE1", "PHATE2"]].values
        else:
            print("Computing PHATE...")
            phate_df = compute_phate(adata)
            if phate_df is not None:
                if "stage" in adata.obs.columns:
                    phate_df["stage"] = adata.obs["stage"].values
                if cell_type_key in adata.obs.columns:
                    phate_df["cell_type"] = adata.obs[cell_type_key].values
                phate_df.to_parquet(phate_checkpoint)
                outputs["phate"] = phate_checkpoint
                print(f"  Saved {phate_checkpoint}")

    # Leiden clustering (PCA-based)
    print("Computing Leiden clustering...")
    cluster_df = compute_leiden_clustering(adata, resolutions)

    # PHATE-based clustering
    if "X_phate" in adata.obsm:
        print("Computing PHATE-based clustering...")
        sc.pp.neighbors(adata, use_rep="X_phate", n_neighbors=30, key_added="phate_neighbors")
        for res in [0.5, 1.0]:
            key = f"leiden_phate_{res}"
            sc.tl.leiden(adata, resolution=res, neighbors_key="phate_neighbors", key_added=key)
            cluster_df[key] = adata.obs[key].values

    # Save clustering
    out_path = output_dir / "clustering.parquet"
    cluster_df.to_parquet(out_path)
    outputs["clustering"] = out_path
    print(f"  Saved {out_path}")

    # Add clusters to UMAP df
    for res in [0.5, 1.0]:
        key = f"leiden_{res}"
        if key in cluster_df.columns:
            umap_df[key] = cluster_df[key].values
        phate_key = f"leiden_phate_{res}"
        if phate_key in cluster_df.columns:
            umap_df[phate_key] = cluster_df[phate_key].values

    out_path = output_dir / "umap_embedding.parquet"
    umap_df.to_parquet(out_path)
    outputs["umap"] = out_path
    print(f"  Saved {out_path}")

    print("Embeddings complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings and clustering")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-phate", action="store_true", help="Skip PHATE")
    args = parser.parse_args()

    run_embeddings(args.h5ad, args.output, compute_phate_embedding=not args.no_phate)
