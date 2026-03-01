"""HLCA reference loading and latent-space alignment helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from stagebridge import config
from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import normalize_stage_series

log = get_logger(__name__)


HLCA_FULL_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad"
)
HLCA_FULL_FILENAME = "hlca_full_v1.h5ad"


@dataclass(slots=True)
class HLCAReference:
    """Reference container for HLCA artifacts."""

    adata: Any
    source_path: Path
    latent_key: str = "X_hlca"


def hlca_reference_dir() -> Path:
    """Return the canonical HLCA reference directory inside data root."""
    return config.resolve_path("data", "reference", "hlca")


def hlca_reference_h5ad(default_filename: str = HLCA_FULL_FILENAME) -> Path:
    """Return canonical path to the HLCA full h5ad."""
    return hlca_reference_dir() / default_filename


def download_hlca_reference(
    output_path: Path | None = None,
    url: str = HLCA_FULL_URL,
) -> Path:
    """Download the HLCA full reference h5ad to the configured reference dir."""
    dst = output_path or hlca_reference_h5ad()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log.info("HLCA file already exists at %s", dst)
        return dst
    log.info("Downloading HLCA reference from %s", url)
    urlretrieve(url, dst)  # nosec: trusted URL is user-provided in project docs
    log.info("Downloaded HLCA reference to %s", dst)
    return dst


def load_hlca_reference(
    h5ad_path: Path | None = None,
    backed: str | None = None,
    latent_key: str = "X_hlca",
) -> HLCAReference:
    """Load HLCA reference AnnData from disk."""
    try:
        import anndata
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "anndata is required to load HLCA reference. "
            "Install dependencies from environment.yml first."
        ) from exc

    path = Path(h5ad_path) if h5ad_path is not None else hlca_reference_h5ad()
    if not path.exists():
        raise FileNotFoundError(
            f"HLCA reference not found at {path}\n"
            "Expected full atlas file (.h5ad) at this path.\n"
            f"You can use download_hlca_reference() with URL:\n{HLCA_FULL_URL}"
        )

    adata = anndata.read_h5ad(path, backed=backed)
    log.info("Loaded HLCA reference: %s with shape %s", path, adata.shape)
    return HLCAReference(adata=adata, source_path=path, latent_key=latent_key)


def harmonize_hlca_metadata(
    adata: Any,
    stage_col: str = "stage",
    donor_col: str = "patient_id",
    sample_col: str = "sample_id",
    modality_col: str = "modality",
) -> None:
    """Standardize key metadata columns in ``adata.obs`` in place."""
    obs = adata.obs

    if stage_col in obs.columns:
        obs[stage_col] = normalize_stage_series(obs[stage_col])
    else:
        obs[stage_col] = "Unknown"

    if donor_col not in obs.columns:
        obs[donor_col] = "unknown_donor"
    if sample_col not in obs.columns:
        obs[sample_col] = "unknown_sample"
    if modality_col not in obs.columns:
        obs[modality_col] = "unknown_modality"

    # Common HLCA naming alternatives
    if "donor_id" in obs.columns and donor_col not in obs.columns:
        obs[donor_col] = obs["donor_id"].astype(str)
    if "sample" in obs.columns and sample_col not in obs.columns:
        obs[sample_col] = obs["sample"].astype(str)


def map_to_hlca_latent(
    adata: Any,
    reference: HLCAReference | None = None,
    input_key: str = "X_pca",
    output_key: str = "X_hlca",
    n_components: int = 64,
) -> None:
    """Create ``adata.obsm[output_key]`` with HLCA-aligned latent representation.

    Strategy:
    1. Use existing ``obsm[input_key]`` as source representation.
    2. If a reference with ``obsm[reference.latent_key]`` is available, match source
       latent to reference scale (z-score -> reference mean/std) for compatibility.
    3. If no reference latent is available, keep deterministic PCA fallback.
    """
    if input_key not in adata.obsm:
        raise KeyError(
            f"Input latent key '{input_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    X = np.asarray(adata.obsm[input_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D latent array in obsm['{input_key}'], got {X.shape}")

    n_components = min(n_components, X.shape[1])
    X = X[:, :n_components]

    if reference is not None and reference.latent_key in reference.adata.obsm:
        X_ref = np.asarray(reference.adata.obsm[reference.latent_key], dtype=np.float32)
        if X_ref.ndim == 2 and X_ref.shape[1] >= n_components:
            X_ref = X_ref[:, :n_components]
            src_mu = X.mean(axis=0, keepdims=True)
            src_sd = X.std(axis=0, keepdims=True) + 1e-6
            ref_mu = X_ref.mean(axis=0, keepdims=True)
            ref_sd = X_ref.std(axis=0, keepdims=True) + 1e-6
            X = (X - src_mu) / src_sd
            X = X * ref_sd + ref_mu
            log.info(
                "Mapped to HLCA-aligned latent (%s) using reference scaling.",
                reference.latent_key,
            )
        else:
            log.warning(
                "Reference latent key '%s' unavailable or mismatched; using fallback latent.",
                reference.latent_key,
            )
    else:
        log.info("No HLCA reference latent provided; using deterministic fallback latent.")

    adata.obsm[output_key] = X.astype(np.float32)

