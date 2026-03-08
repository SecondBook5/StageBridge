"""Latent representation builder for StageBridge training.

Provides a single entry-point ``build_latent`` that wraps PCA-based and
HLCA-aligned strategies.  The HLCA strategy gracefully falls back to PCA when
no reference is available so it never blocks training.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dataclasses import dataclass

from stagebridge.logging_utils import get_logger
from stagebridge.data.common.schema import LatentCohort
from stagebridge.data.common.harmonize import add_hlca_latent, pca_fit_transform_snrna

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class LatentStore:
    """Reusable in-memory store for the active reference latent table."""

    cohort: LatentCohort

    def summary(self) -> dict[str, object]:
        return {
            "source_path": str(self.cohort.source_path),
            "latent_key": self.cohort.latent_key,
            "shape": [int(self.cohort.n_obs), int(self.cohort.n_vars)],
            "feature_names": list(self.cohort.feature_names[: min(5, len(self.cohort.feature_names))]),
        }


def build_latent(
    adata: Any,
    method: str = "pca",
    n_components: int = 64,
    output_key: str = "X_hlca",
    pca_layer: str = "log1p",
    hlca_reference: Any | None = None,
) -> None:
    """Build or attach a latent embedding in ``adata.obsm[output_key]``.

    Parameters
    ----------
    adata : AnnData
        Modified in place.  Input is expected to already have a log1p-normalised
        layer (see :func:`stagebridge.data.common.harmonize.normalize_log1p`).
    method : {"pca", "hlca"}
        ``"pca"`` fits TruncatedSVD on log1p-normalised counts and stores the
        result under *output_key*.
        ``"hlca"`` first computes PCA, then aligns to HLCA reference scale via
        :func:`stagebridge.data.common.harmonize.add_hlca_latent`.  If no
        *hlca_reference* is provided the alignment falls back to the raw PCA
        embedding — training can proceed immediately.
    n_components : int
        Number of latent dimensions (capped at ``min(adata.shape) - 1``).
    output_key : str
        Key in ``adata.obsm`` where the latent representation will be stored.
        Typically ``"X_hlca"`` (training default).
    pca_layer : str
        AnnData layer to use as PCA input (default ``"log1p"``).
    hlca_reference : HLCAReference or None
        Loaded HLCA reference returned by
        :func:`stagebridge.reference.hlca_mapper.load_hlca_reference`.  If ``None`` and
        ``method="hlca"`` the function logs a warning and stores the PCA
        embedding under *output_key* without scale alignment.
    """
    if output_key in adata.obsm:
        log.info("Latent key '%s' already present — skipping build_latent.", output_key)
        return

    if method == "pca":
        _build_pca_latent(
            adata=adata,
            n_components=n_components,
            output_key=output_key,
            pca_layer=pca_layer,
        )
    elif method == "hlca":
        _build_hlca_latent(
            adata=adata,
            n_components=n_components,
            output_key=output_key,
            pca_layer=pca_layer,
            reference=hlca_reference,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'pca', 'hlca'."
        )


def _build_pca_latent(
    adata: Any,
    n_components: int,
    output_key: str,
    pca_layer: str,
) -> None:
    """Fit PCA and copy result to *output_key*."""
    if pca_layer not in adata.layers:
        raise KeyError(
            f"Layer '{pca_layer}' not found. Run normalize_log1p() first.\n"
            f"Available layers: {list(adata.layers.keys())}"
        )

    n_eff = min(n_components, min(adata.shape) - 1)
    pca_fit_transform_snrna(adata, n_components=n_eff, use_layer=pca_layer)

    if output_key != "X_pca":
        adata.obsm[output_key] = adata.obsm["X_pca"].copy()

    log.info("Built PCA latent '%s' with %d components.", output_key, n_eff)


def _build_hlca_latent(
    adata: Any,
    n_components: int,
    output_key: str,
    pca_layer: str,
    reference: Any | None,
) -> None:
    """Build HLCA-aligned latent with graceful PCA fallback."""
    pca_key = "X_pca"
    if pca_key not in adata.obsm:
        _build_pca_latent(
            adata=adata,
            n_components=n_components,
            output_key=pca_key,
            pca_layer=pca_layer,
        )

    if reference is None:
        log.warning(
            "No HLCA reference provided for method='hlca'. "
            "Storing PCA embedding under '%s' without scale alignment. "
            "Training can still proceed.",
            output_key,
        )

    add_hlca_latent(
        adata=adata,
        reference=reference,
        source_key=pca_key,
        output_key=output_key,
        n_components=n_components,
    )
    log.info("Built HLCA-aligned latent '%s' with %d components.", output_key, n_components)


def latent_summary(adata: Any, output_key: str = "X_hlca") -> dict[str, object]:
    """Return a short summary dict for the latent embedding."""
    if output_key not in adata.obsm:
        return {"present": False, "key": output_key}
    X = np.asarray(adata.obsm[output_key], dtype=np.float32)
    return {
        "present": True,
        "key": output_key,
        "shape": X.shape,
        "mean": float(X.mean()),
        "std": float(X.std()),
        "min": float(X.min()),
        "max": float(X.max()),
    }
