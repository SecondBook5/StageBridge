"""Trajectory inference (diffusion pseudotime, PAGA)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import anndata as ad


def compute_diffusion_map(
    adata: "ad.AnnData",
    n_comps: int = 15,
) -> "ad.AnnData":
    """Compute diffusion map embedding.

    Args:
        adata: AnnData with neighbors computed
        n_comps: Number of diffusion components

    Returns:
        AnnData with X_diffmap in obsm
    """
    import scanpy as sc

    sc.tl.diffmap(adata, n_comps=n_comps)
    return adata


def compute_diffusion_pseudotime(
    adata: "ad.AnnData",
    root_stage: str = "Normal",
    stage_col: str = "stage",
) -> pd.Series:
    """Compute diffusion pseudotime.

    Args:
        adata: AnnData with diffusion map
        root_stage: Stage to use as root
        stage_col: Column with stage labels

    Returns:
        Series with pseudotime values
    """
    import scanpy as sc

    # Find root cell (median of root stage in diffusion space)
    root_mask = adata.obs[stage_col] == root_stage
    if root_mask.sum() == 0:
        raise ValueError(f"No cells found with {stage_col}={root_stage}")

    root_cells = adata[root_mask].obsm["X_diffmap"]
    median_idx = np.argmin(np.sum((root_cells - root_cells.mean(0)) ** 2, axis=1))
    root_cell_idx = np.where(root_mask)[0][median_idx]

    adata.uns["iroot"] = root_cell_idx
    sc.tl.dpt(adata)

    return adata.obs["dpt_pseudotime"]


def compute_paga(
    adata: "ad.AnnData",
    groups: str = "stage",
) -> dict:
    """Compute PAGA graph.

    Args:
        adata: AnnData with neighbors computed
        groups: Column for grouping

    Returns:
        Dict with connectivities and confidence
    """
    import scanpy as sc

    sc.tl.paga(adata, groups=groups)

    return {
        "connectivities": adata.uns["paga"]["connectivities"].toarray(),
        "connectivities_tree": adata.uns["paga"]["connectivities_tree"].toarray(),
        "groups": adata.obs[groups].cat.categories.tolist(),
    }


def run_trajectories(
    h5ad_path: str | Path,
    output_dir: str | Path,
    root_stage: str = "Normal",
) -> dict[str, Path]:
    """Run full trajectory analysis pipeline.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        root_stage: Stage to use as trajectory root

    Returns:
        Dict of output file paths
    """
    import scanpy as sc

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells")

    outputs = {}

    # Ensure PCA and neighbors
    if "X_pca" not in adata.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata, n_comps=50)

    if "neighbors" not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=30)

    # Diffusion map
    print("Computing diffusion map...")
    compute_diffusion_map(adata)

    # Diffusion pseudotime
    print("Computing diffusion pseudotime...")
    try:
        dpt = compute_diffusion_pseudotime(adata, root_stage)

        dpt_df = pd.DataFrame({
            "cell_id": adata.obs.index,
            "dpt_pseudotime": dpt.values,
            "stage": adata.obs["stage"].values if "stage" in adata.obs.columns else None,
        })
        out_path = output_dir / "diffusion_pseudotime.parquet"
        dpt_df.to_parquet(out_path)
        outputs["dpt"] = out_path
        print(f"  Saved {out_path}")
    except Exception as e:
        print(f"  DPT failed: {e}")

    # PAGA
    if "stage" in adata.obs.columns:
        print("Computing PAGA...")
        try:
            paga = compute_paga(adata, "stage")

            paga_df = pd.DataFrame(
                paga["connectivities"],
                index=paga["groups"],
                columns=paga["groups"],
            )
            out_path = output_dir / "paga_connectivities.parquet"
            paga_df.to_parquet(out_path)
            outputs["paga"] = out_path
            print(f"  Saved {out_path}")
        except Exception as e:
            print(f"  PAGA failed: {e}")

    print("Trajectory analysis complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory inference")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--root", default="Normal", help="Root stage for DPT")
    args = parser.parse_args()

    run_trajectories(args.h5ad, args.output, args.root)
