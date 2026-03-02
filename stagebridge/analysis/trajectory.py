"""Euler trajectory integration and UMAP projection for StageBridge.

Integrates source cells forward from stage_src to stage_tgt using the trained
flow-matching model, then projects the predicted positions onto an existing UMAP
embedding for visualization as quiver arrows.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import (
    infer_stage_pair_id,
    normalize_stage_label,
    ordered_transitions,
)

log = get_logger(__name__)


def integrate_trajectories(
    model: Any,
    adata: Any,
    stage_src: str,
    stage_tgt: str,
    n_steps: int = 20,
    latent_key: str = "X_hlca",
    stage_col: str = "stage",
    context_key: str | None = "X_context",
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-integrate source cells to the predicted target distribution.

    Uses per-cell context vectors from ``adata.obsm[context_key]`` if available,
    otherwise computes a single population-level context from the full source set.

    Parameters
    ----------
    model : StageBridgeModel
        Trained model with ``forward_set_context``, ``forward_vector_field``,
        and ``integrate_euler`` methods.
    adata : AnnData
        Must have obsm[latent_key] and obs[stage_col].
    stage_src : str
        Source stage name (e.g. "AAH").
    stage_tgt : str
        Target stage name (e.g. "AIS").
    n_steps : int
        Number of Euler integration steps.
    latent_key : str
        adata.obsm key for latent coordinates.
    stage_col : str
        obs column for stage labels.
    context_key : str or None
        adata.obsm key for pre-computed context vectors.  If None or not
        present, computes one context vector from the full source population.
    batch_size : int
        Integration batch size (for memory management).
    device : str
        Torch device.

    Returns
    -------
    (x0, x1_pred) : tuple of ndarray, each shape (n_src_cells, latent_dim)
        x0 = source cell latent positions.
        x1_pred = predicted target cell positions after Euler integration.
    """
    if latent_key not in adata.obsm:
        raise KeyError(f"latent_key '{latent_key}' not in adata.obsm.")

    X = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    stages = np.array([normalize_stage_label(s) for s in adata.obs[stage_col].astype(str).values])

    src_norm = normalize_stage_label(stage_src)
    tgt_norm = normalize_stage_label(stage_tgt)
    src_mask = stages == src_norm

    if src_mask.sum() == 0:
        raise ValueError(f"No cells with stage '{stage_src}' (normalized: '{src_norm}') found.")

    x0 = X[src_mask]
    pair_id = infer_stage_pair_id(stage_src, stage_tgt)

    model = model.to(device)
    model.eval()

    # Get or compute context vectors for source cells
    if context_key and context_key in adata.obsm:
        ctx = np.asarray(adata.obsm[context_key], dtype=np.float32)[src_mask]
        has_per_cell_ctx = True
        log.info("Using pre-computed per-cell context vectors from '%s'.", context_key)
    else:
        # Compute a single population-level context from all source cells
        log.info("No per-cell context found. Computing population-level context.")
        with torch.no_grad():
            x_src_t = torch.tensor(x0[:batch_size], dtype=torch.float32, device=device)
            pair_id_t = torch.tensor([pair_id], dtype=torch.long, device=device)
            c_pop = model.forward_set_context(x_src_t.unsqueeze(0))
            # c_pop: (1, context_dim)
            ctx = c_pop.squeeze(0).cpu().numpy().astype(np.float32)
        # Broadcast to all source cells
        ctx = np.tile(ctx, (len(x0), 1))
        has_per_cell_ctx = False

    # Euler integration in batches
    n_src = len(x0)
    x1_pred = np.zeros_like(x0)

    with torch.no_grad():
        for start in range(0, n_src, batch_size):
            end = min(start + batch_size, n_src)
            x_batch = torch.tensor(x0[start:end], dtype=torch.float32, device=device)
            ctx_batch = torch.tensor(ctx[start:end], dtype=torch.float32, device=device)
            pair_batch = torch.full((end - start,), pair_id, dtype=torch.long, device=device)

            x_pred_batch = model.integrate_euler(
                x_batch, ctx_batch, pair_batch, num_steps=n_steps,
            )
            x1_pred[start:end] = x_pred_batch.cpu().numpy()

    log.info(
        "Integrated %d source cells (%s→%s) using %s context, %d steps.",
        n_src, stage_src, stage_tgt,
        "per-cell" if has_per_cell_ctx else "population",
        n_steps,
    )
    return x0, x1_pred


def project_trajectories_to_umap(
    adata: Any,
    x0: np.ndarray,
    x1_pred: np.ndarray,
    umap_key: str = "X_umap",
    latent_key: str = "X_hlca",
    n_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Project latent trajectory endpoints onto existing UMAP coordinates.

    For each source cell x0_i, finds its approximate UMAP position via nearest-
    neighbor lookup in the latent space.  The predicted position x1_pred_i is
    projected by summing the weighted UMAP offsets of its latent-space neighbors.

    This avoids refitting UMAP and preserves the topological structure of the
    original embedding.

    Parameters
    ----------
    adata : AnnData
        Must have obsm[umap_key] (2-D UMAP) and obsm[latent_key] (full latent).
    x0 : ndarray, shape (n, latent_dim)
        Source cell latent positions.
    x1_pred : ndarray, shape (n, latent_dim)
        Predicted target cell latent positions.
    umap_key : str
        Key in adata.obsm for UMAP coordinates.
    latent_key : str
        Key in adata.obsm for full-dimensional latent (used for NN lookup).
    n_neighbors : int
        Number of nearest neighbors for UMAP projection.

    Returns
    -------
    (uv0, uv1_pred) : tuple of ndarray, each shape (n, 2)
        2-D UMAP positions for source and predicted target cells.
    """
    if umap_key not in adata.obsm:
        raise KeyError(f"'{umap_key}' not in adata.obsm. Run sc.tl.umap(adata) first.")
    if latent_key not in adata.obsm:
        raise KeyError(f"'{latent_key}' not in adata.obsm.")

    from sklearn.neighbors import NearestNeighbors

    X_ref = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    umap_ref = np.asarray(adata.obsm[umap_key], dtype=np.float32)[:, :2]

    # Fit KNN on the reference latent space
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", metric="euclidean")
    nn.fit(X_ref)

    def _latent_to_umap(X_query: np.ndarray) -> np.ndarray:
        distances, indices = nn.kneighbors(X_query.astype(np.float32))
        # Inverse-distance weights
        weights = 1.0 / (distances + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        # Weighted average of UMAP coordinates
        coords = np.einsum("nk,nkd->nd", weights, umap_ref[indices])
        return coords.astype(np.float32)

    uv0 = _latent_to_umap(x0)
    uv1_pred = _latent_to_umap(x1_pred)

    log.info("Projected %d trajectories to UMAP space.", len(x0))
    return uv0, uv1_pred
