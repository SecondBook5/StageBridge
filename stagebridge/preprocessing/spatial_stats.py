"""Spatial statistics via Squidpy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def compute_spatial_neighbors(
    adata: "ad.AnnData",
    coord_type: str = "generic",
) -> "ad.AnnData":
    """Compute spatial neighbor graph.

    Args:
        adata: AnnData with spatial coordinates
        coord_type: Coordinate type for squidpy

    Returns:
        AnnData with spatial connectivity in obsp
    """
    import squidpy as sq

    sq.gr.spatial_neighbors(adata, coord_type=coord_type)
    return adata


def compute_nhood_enrichment(
    adata: "ad.AnnData",
    cluster_key: str,
    n_perms: int = 100,
) -> pd.DataFrame:
    """Compute neighborhood enrichment z-scores.

    Args:
        adata: AnnData with spatial neighbors computed
        cluster_key: Column in obs for cell type labels
        n_perms: Number of permutations

    Returns:
        DataFrame with z-scores (cell_type x cell_type)
    """
    import squidpy as sq

    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, n_perms=n_perms)
    zscore = adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"]
    categories = adata.obs[cluster_key].cat.categories
    return pd.DataFrame(zscore, index=categories, columns=categories)


def compute_morans_i(
    adata: "ad.AnnData",
    genes: list[str],
) -> pd.DataFrame:
    """Compute Moran's I spatial autocorrelation.

    Args:
        adata: AnnData with spatial neighbors computed
        genes: List of genes to compute Moran's I for

    Returns:
        DataFrame with Moran's I statistics
    """
    import squidpy as sq

    available = [g for g in genes if g in adata.var_names]
    if not available:
        return pd.DataFrame()

    sq.gr.spatial_autocorr(adata, genes=available, mode="moran")
    return adata.uns["moranI"]


def run_spatial_stats(
    spatial_path: str | Path,
    output_dir: str | Path,
    cluster_key: str | None = None,
    key_genes: list[str] | None = None,
) -> dict[str, Path]:
    """Run full spatial statistics pipeline.

    Args:
        spatial_path: Path to spatial h5ad
        output_dir: Output directory
        cluster_key: Column for cell types (auto-detect if None)
        key_genes: Genes for Moran's I (defaults to cancer-relevant)

    Returns:
        Dict of output file paths
    """
    import scanpy as sc

    spatial_path = Path(spatial_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {spatial_path}...")
    adata = sc.read_h5ad(spatial_path)
    print(f"  {adata.n_obs} spots")

    outputs = {}

    # Spatial neighbors
    print("Computing spatial neighbors...")
    compute_spatial_neighbors(adata)

    # Neighborhood enrichment
    if cluster_key is None:
        cluster_key = "cell_type_luca" if "cell_type_luca" in adata.obs.columns else "cell_type"

    if cluster_key in adata.obs.columns:
        print(f"Computing neighborhood enrichment ({cluster_key})...")
        nhood_df = compute_nhood_enrichment(adata, cluster_key)
        out_path = output_dir / "nhood_enrichment.parquet"
        nhood_df.to_parquet(out_path)
        outputs["nhood_enrichment"] = out_path
        print(f"  Saved {out_path}")

    # Moran's I
    if key_genes is None:
        key_genes = [
            "IL1B", "IL1R1", "CXCL12", "CXCR4", "EGFR", "SOX9",
            "KRT17", "VIM", "CDH1", "ACTA2", "COL1A1", "CD274", "PDCD1",
        ]

    print("Computing Moran's I...")
    morans_df = compute_morans_i(adata, key_genes)
    if not morans_df.empty:
        out_path = output_dir / "morans_i.parquet"
        morans_df.to_parquet(out_path)
        outputs["morans_i"] = out_path
        print(f"  Saved {out_path}")

    print("Spatial stats complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute spatial statistics")
    parser.add_argument("--spatial", required=True, help="Path to spatial h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--cluster-key", default=None, help="Cell type column")
    args = parser.parse_args()

    run_spatial_stats(args.spatial, args.output, args.cluster_key)
