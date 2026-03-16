"""Gene-context attribution analysis for StageBridge.

Extracts per-cell context vectors c_s from the trained Set Transformer,
then computes Pearson correlations between gene expression and each context
dimension to identify which genes drive the learned population context signal.

Biological expectation (Peng et al. 2026, Cancer Cell):
  Top context-correlated genes should include:
    - KRT8, CEACAM5  (KAC / alveolar intermediate cell markers)
    - IL1B, CCL2, CSF1  (IL1B+ macrophage / proinflammatory niche signals)
    - SFTPC  (AT2 marker, expected to decrease with stage progression)
    - RELA, NFKB1  (NF-κB pathway, dominant in precursor stages)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def extract_context_vectors(
    model: Any,
    adata: Any,
    latent_key: str = "X_hlca",
    stage_col: str = "stage",
    batch_size: int = 64,
    device: str = "cpu",
    output_key: str = "X_context",
) -> np.ndarray:
    """Run ``forward_set_context`` on all cells and store context vectors.

    For each cell, the source set is constructed as: all other cells from the
    same stage (sampled to ``batch_size`` if the stage has more cells).  The
    PMA seed vector output (c_s) is stored as the context vector for that cell.

    Parameters
    ----------
    model : StageBridgeModel
        Trained model.
    adata : AnnData
        Must have obsm[latent_key] and obs[stage_col].
    latent_key : str
        adata.obsm key for latent coordinates.
    stage_col : str
        obs column for stage labels.
    batch_size : int
        Maximum source set size per cell.
    device : str
        Torch device.
    output_key : str
        Key in adata.obsm where context vectors will be stored.

    Returns
    -------
    ndarray, shape (n_cells, context_dim)
        Per-cell context vectors.  Also stored in ``adata.obsm[output_key]``.
    """
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_key '{latent_key}' not in adata.obsm.")

    from stagebridge.data.luad_evo.stages import (
        normalize_stage_label,
        ordered_transitions,
    )

    X = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    stages = adata.obs[stage_col].astype(str).values

    model = model.to(device)
    model.eval()

    context_dim: int | None = None
    context_vectors = np.zeros((adata.n_obs, 1), dtype=np.float32)  # placeholder shape

    rng = np.random.default_rng(42)

    # Get a representative pair id (use first valid pair in the stage order)
    transitions = ordered_transitions()
    valid_pairs = [
        (s, t)
        for s, t in transitions
        if (stages == normalize_stage_label(s)).any()
        and (stages == normalize_stage_label(t)).any()
    ]
    if not valid_pairs:
        raise ValueError("No valid stage transitions found in adata.")

    with torch.no_grad():
        for cell_idx in range(adata.n_obs):
            cell_stage = normalize_stage_label(stages[cell_idx])

            # Build source set: other cells from same stage
            same_stage_idx = np.where(
                np.array([normalize_stage_label(s) for s in stages]) == cell_stage
            )[0]
            same_stage_idx = same_stage_idx[same_stage_idx != cell_idx]

            if len(same_stage_idx) == 0:
                # Only cell of its stage — use itself
                src_idx = np.array([cell_idx])
            elif len(same_stage_idx) > batch_size:
                src_idx = rng.choice(same_stage_idx, size=batch_size, replace=False)
            else:
                src_idx = same_stage_idx

            x_src = torch.tensor(X[src_idx], dtype=torch.float32, device=device)
            src_set = x_src.unsqueeze(0)  # (1, n_src, d)

            c_s = model.forward_set_context(src_set)
            # c_s shape: (1, num_seed_vectors, context_dim) or (1, context_dim)
            c_s_np = c_s.squeeze().cpu().numpy().astype(np.float32)
            if c_s_np.ndim == 0:
                c_s_np = c_s_np.reshape(1)

            # Initialize output array on first cell
            if context_dim is None:
                context_dim = c_s_np.size
                context_vectors = np.zeros((adata.n_obs, context_dim), dtype=np.float32)

            context_vectors[cell_idx] = c_s_np.ravel()[:context_dim]

            if (cell_idx + 1) % 500 == 0:
                log.info("Extracted context vectors: %d / %d", cell_idx + 1, adata.n_obs)

    log.info("Context vectors extracted: shape %s", context_vectors.shape)
    adata.obsm[output_key] = context_vectors
    return context_vectors


def compute_gene_context_correlation(
    adata: Any,
    context_vectors: np.ndarray | None = None,
    context_key: str = "X_context",
    layer: str | None = "log1p",
    use_raw: bool = False,
) -> pd.DataFrame:
    """Compute Pearson correlation between gene expression and context dimensions.

    Parameters
    ----------
    adata : AnnData
        Gene expression data.  Uses layer[layer] if provided, else adata.X.
    context_vectors : ndarray or None
        Shape (n_cells, context_dim).  If None, reads from adata.obsm[context_key].
    context_key : str
        Key in adata.obsm to read context vectors from (used when context_vectors=None).
    layer : str or None
        AnnData layer to use for expression.  If None, uses adata.X.
    use_raw : bool
        Use adata.raw.X instead.

    Returns
    -------
    DataFrame, shape (n_genes, n_context_dims)
        Pearson r values.  Index = gene names.  Columns = context dim labels.
    """
    import scipy.sparse as sp

    if context_vectors is None:
        if context_key not in adata.obsm:
            raise KeyError(
                f"context_key '{context_key}' not in adata.obsm. "
                "Run extract_context_vectors() first."
            )
        context_vectors = np.asarray(adata.obsm[context_key], dtype=np.float32)

    cv = context_vectors.astype(np.float64)

    # Get expression matrix
    if use_raw and adata.raw is not None:
        X = adata.raw.X
        gene_names = list(adata.raw.var_names)
    elif layer is not None and layer in adata.layers:
        X = adata.layers[layer]
        gene_names = list(adata.var_names)
    else:
        X = adata.X
        gene_names = list(adata.var_names)

    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    n_cells, n_genes = X.shape
    n_dims = cv.shape[1]

    if n_cells != cv.shape[0]:
        raise ValueError(
            f"Expression matrix has {n_cells} cells but context_vectors has {cv.shape[0]}"
        )

    log.info("Computing gene-context correlations: %d genes × %d dims", n_genes, n_dims)

    # Pearson correlation: mean-center both matrices, then dot product / (n * std_X * std_cv)
    X_c = X - X.mean(axis=0, keepdims=True)
    cv_c = cv - cv.mean(axis=0, keepdims=True)

    X_std = X_c.std(axis=0, keepdims=True) + 1e-10
    cv_std = cv_c.std(axis=0, keepdims=True) + 1e-10

    X_n = X_c / X_std  # (n_cells, n_genes)
    cv_n = cv_c / cv_std  # (n_cells, n_dims)

    # r[gene, dim] = (1/n) * X_n[:, gene] · cv_n[:, dim]
    r_matrix = (X_n.T @ cv_n) / n_cells  # (n_genes, n_dims)

    dim_labels = [f"ctx_{i}" for i in range(n_dims)]
    df = pd.DataFrame(r_matrix, index=gene_names, columns=dim_labels)
    log.info("Gene-context correlation matrix: %s", df.shape)
    return df


def top_genes_per_context_dim(
    corr_df: pd.DataFrame,
    n_top: int = 20,
    absolute: bool = True,
) -> dict[str, list[str]]:
    """Return the top-N genes for each context dimension.

    Parameters
    ----------
    corr_df : DataFrame
        Gene-context correlation matrix from ``compute_gene_context_correlation``.
    n_top : int
        Number of top genes to return per dimension.
    absolute : bool
        If True, rank by |r| (finds both positive and negative drivers).
        If False, rank by r descending (finds positively correlated genes only).

    Returns
    -------
    dict mapping context dim label to list of top gene names.
    """
    result: dict[str, list[str]] = {}
    for col in corr_df.columns:
        vals = corr_df[col].abs() if absolute else corr_df[col]
        top = vals.nlargest(n_top).index.tolist()
        result[col] = top
    return result
