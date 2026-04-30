"""Progression scoring: CytoTRACE and diffusion pseudotime.

CytoTRACE: Differentiation potential based on gene count (Gulati et al. 2020)
Pseudotime: Diffusion pseudotime rooted at Normal stage (Haghverdi et al. 2016)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_cytotrace(adata) -> np.ndarray:
    """Compute CytoTRACE-like score from gene counts.

    CytoTRACE principle: more differentiated cells express fewer genes.
    Higher score = more stem-like/less differentiated.

    Args:
        adata: AnnData with expression matrix

    Returns:
        Array of CytoTRACE scores normalized to [0, 1]
    """
    if "n_genes" not in adata.obs.columns:
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        adata.obs["n_genes"] = (X > 0).sum(axis=1)

    gene_counts = adata.obs["n_genes"].values.astype(float)

    gc_min, gc_max = gene_counts.min(), gene_counts.max()
    if gc_max > gc_min:
        return (gene_counts - gc_min) / (gc_max - gc_min)
    return np.zeros_like(gene_counts)


def compute_diffusion_pseudotime(
    adata,
    root_stage: str = "Normal",
    stage_col: str = "stage",
    n_neighbors: int = 30,
    n_pcs: int = 50,
) -> np.ndarray:
    """Compute diffusion pseudotime rooted at a reference stage.

    Args:
        adata: AnnData with expression or embedding
        root_stage: Stage to use as pseudotime origin
        stage_col: Column in adata.obs containing stage labels
        n_neighbors: Number of neighbors for graph construction
        n_pcs: Number of PCs if computing from expression

    Returns:
        Array of pseudotime values (0 = root, higher = more progressed)
    """
    import scanpy as sc

    adata = adata.copy()

    # PCA if needed
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1))

    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")

    # Diffusion map
    sc.tl.diffmap(adata, n_comps=15)

    # Find root cell (closest to centroid of root_stage)
    if stage_col in adata.obs.columns and root_stage in adata.obs[stage_col].values:
        root_mask = adata.obs[stage_col] == root_stage
        root_indices = np.where(root_mask)[0]
        diffmap = adata.obsm["X_diffmap"]
        centroid = diffmap[root_mask].mean(axis=0)
        distances = np.linalg.norm(diffmap[root_mask] - centroid, axis=1)
        root_cell = root_indices[np.argmin(distances)]
    else:
        root_cell = 0

    adata.uns["iroot"] = root_cell

    # DPT
    sc.tl.dpt(adata, n_branchings=0)

    return adata.obs["dpt_pseudotime"].values


def compute_progression_scores(
    adata,
    root_stage: str = "Normal",
    stage_col: str = "stage",
) -> pd.DataFrame:
    """Compute both CytoTRACE and pseudotime.

    Args:
        adata: AnnData with expression matrix
        root_stage: Stage to use as pseudotime origin
        stage_col: Column containing stage labels

    Returns:
        DataFrame with cell_id, cytotrace, pseudotime columns
    """
    cytotrace = compute_cytotrace(adata)

    try:
        pseudotime = compute_diffusion_pseudotime(
            adata, root_stage=root_stage, stage_col=stage_col
        )
    except Exception as e:
        print(f"Pseudotime computation failed: {e}")
        pseudotime = np.full(adata.n_obs, np.nan)

    return pd.DataFrame({
        "cell_id": adata.obs_names,
        "cytotrace": cytotrace,
        "pseudotime": pseudotime,
    })


def load_progression_scores(path: Path) -> pd.DataFrame:
    """Load precomputed progression scores.

    Args:
        path: Path to progression_scores.parquet

    Returns:
        DataFrame with cell_id, cytotrace, pseudotime
    """
    return pd.read_parquet(path)
