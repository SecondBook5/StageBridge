"""
Additional normalisation utilities (beyond the log1p in harmonize.py).

These are thin wrappers that may delegate to scanpy when available,
or provide fallback pure-numpy implementations.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import anndata

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def scale_to_unit_variance(
    adata: anndata.AnnData,
    layer_in: str = "log1p",
    layer_out: str = "scaled",
    max_value: float | None = 10.0,
) -> None:
    """Zero-mean, unit-variance scaling per gene (clips at *max_value*).

    Operates on dense arrays; converts sparse to dense for this step.
    Write result to ``adata.layers[layer_out]``.

    Parameters
    ----------
    adata : AnnData
    layer_in : str
    layer_out : str
    max_value : float or None
        Clip absolute values to this threshold after scaling (default 10.0).
        Pass None to skip clipping.
    """
    if layer_in not in adata.layers:
        raise KeyError(
            f"Layer '{layer_in}' not in adata.layers.  Run normalize_log1p() first."
        )

    X = adata.layers[layer_in]
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float32)
    else:
        X = np.array(X, dtype=np.float32)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    X_scaled = (X - mean) / std
    if max_value is not None:
        X_scaled = np.clip(X_scaled, -max_value, max_value)

    adata.layers[layer_out] = X_scaled
    log.info(
        "scale_to_unit_variance → adata.layers['%s']  (max_value=%s)",
        layer_out,
        max_value,
    )
