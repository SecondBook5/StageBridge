"""
Basic quality-control metrics for single-cell / spatial data.

All functions add columns to ``adata.obs`` in place.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import anndata

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def summarize_mapping_qc(compositions: np.ndarray) -> dict[str, float]:
    """Return compact QC metrics for a spot-composition matrix."""
    arr = np.asarray(compositions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D composition matrix, got shape={arr.shape}.")
    row_sums = arr.sum(axis=1)
    max_vals = arr.max(axis=1)
    return {
        "mean_row_sum": float(row_sums.mean()) if row_sums.size else 0.0,
        "std_row_sum": float(row_sums.std()) if row_sums.size else 0.0,
        "mean_max_assignment": float(max_vals.mean()) if max_vals.size else 0.0,
        "n_spots": float(arr.shape[0]),
        "n_features": float(arr.shape[1]),
    }


def compute_basic_qc(adata: anndata.AnnData, layer: str | None = None) -> None:
    """Compute per-cell QC metrics and store them in ``adata.obs``.

    Metrics added
    -------------
    n_genes_by_counts  : int   — number of genes with > 0 counts
    total_counts       : float — sum of counts per cell
    pct_counts_mt      : float — percentage mitochondrial counts (genes starting with MT-)

    Parameters
    ----------
    adata : AnnData
        Modified in place.
    layer : str or None
        Layer to use for counts (default: ``adata.X``).
    """
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(float)
        n_genes = np.diff(X.tocsr().indptr)  # nnz per row
    else:
        X = np.asarray(X, dtype=float)
        total_counts = X.sum(axis=1)
        n_genes = (X > 0).sum(axis=1)

    adata.obs["n_genes_by_counts"] = n_genes.astype(int)
    adata.obs["total_counts"] = total_counts

    # Mitochondrial fraction
    mt_mask = adata.var_names.str.startswith("MT-")
    n_mt = mt_mask.sum()
    if n_mt > 0:
        X_mt = X[:, mt_mask]
        if sp.issparse(X_mt):
            mt_counts = np.asarray(X_mt.sum(axis=1)).ravel().astype(float)
        else:
            mt_counts = X_mt.sum(axis=1)
        pct_mt = np.where(total_counts > 0, 100 * mt_counts / total_counts, 0.0)
    else:
        pct_mt = np.zeros(adata.n_obs, dtype=float)
        log.warning("No MT- genes found; pct_counts_mt will be 0 for all cells.")

    adata.obs["pct_counts_mt"] = pct_mt
    log.info(
        "QC metrics computed: n_cells=%d  median_total_counts=%.1f  n_mt_genes=%d",
        adata.n_obs,
        float(np.median(total_counts)),
        int(n_mt),
    )


def filter_cells(
    adata: anndata.AnnData,
    min_genes: int = 200,
    max_genes: int | None = None,
    max_pct_mt: float | None = 25.0,
) -> anndata.AnnData:
    """Return a filtered copy of *adata* based on QC thresholds.

    Requires :func:`compute_basic_qc` to have been run first.

    Parameters
    ----------
    adata : AnnData
    min_genes : int
        Minimum number of detected genes per cell.
    max_genes : int or None
        Maximum number of detected genes (None → no upper bound).
    max_pct_mt : float or None
        Maximum mitochondrial percentage (None → no limit).

    Returns
    -------
    AnnData
        Filtered copy.
    """
    for col in ("n_genes_by_counts", "total_counts", "pct_counts_mt"):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not in adata.obs.  Run compute_basic_qc() first.")

    mask = adata.obs["n_genes_by_counts"] >= min_genes
    if max_genes is not None:
        mask &= adata.obs["n_genes_by_counts"] <= max_genes
    if max_pct_mt is not None:
        mask &= adata.obs["pct_counts_mt"] <= max_pct_mt

    n_removed = (~mask).sum()
    log.info(
        "filter_cells: keeping %d / %d cells  (removed %d)",
        mask.sum(),
        len(mask),
        n_removed,
    )
    return adata[mask].copy()
