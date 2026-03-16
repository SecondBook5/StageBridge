"""
StageBridge configuration: data root resolution and path utilities.

The external data root is controlled by the environment variable
``STAGEBRIDGE_DATA_ROOT``.  This variable must be set before running
any data-dependent pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "STAGEBRIDGE_DATA_ROOT"


def get_data_root() -> Path:
    """Return the external data root directory.

    Reads from the ``STAGEBRIDGE_DATA_ROOT`` environment variable.

    Raises
    ------
    EnvironmentError
        If the environment variable is not set.
    ValueError
        If the resolved path does not exist on disk.
    """
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        raise OSError(
            f"Environment variable {_ENV_VAR} is not set.\n"
            f"Set it to the root of your StageBridge data directory:\n"
            f"    export {_ENV_VAR}=/path/to/StageBridge_data"
        )
    root = Path(raw)

    if not root.exists():
        raise ValueError(
            f"Data root does not exist: {root}\nCheck that {_ENV_VAR} points to a valid directory."
        )
    if not root.is_dir():
        raise ValueError(
            f"Data root exists but is not a directory: {root}\n"
            f"Please check the path and try again."
        )
    return root


def resolve_path(*parts: str | Path) -> Path:
    """Join *parts onto the data root and return the absolute Path."""
    root = get_data_root()
    return root.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it does not already exist.

    Returns the path for chaining convenience.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Canonical sub-paths inside the data root
# ---------------------------------------------------------------------------


def raw_geo_dir() -> Path:
    return resolve_path("data", "raw", "geo")


def snrna_extracted_dir() -> Path:
    return resolve_path("data", "raw", "geo", "GSE308103_snrna", "extracted")


def spatial_extracted_dir() -> Path:
    return resolve_path("data", "raw", "geo", "GSE307534_spatial", "extracted")


def spatial_samples_dir() -> Path:
    return resolve_path("data", "raw", "geo", "GSE307534_spatial", "samples")


def interim_snrna_dir() -> Path:
    return resolve_path("interim", "anndata", "snrna")


def interim_spatial_dir() -> Path:
    return resolve_path("interim", "anndata", "spatial")


def processed_anndata_dir() -> Path:
    return resolve_path("processed", "anndata")


def snrna_manifest_csv() -> Path:
    return resolve_path("interim", "anndata", "snrna", "manifest.csv")


def spatial_manifest_csv() -> Path:
    return resolve_path("interim", "anndata", "spatial", "manifest.csv")


def snrna_merged_h5ad() -> Path:
    return resolve_path("processed", "anndata", "snrna_merged.h5ad")


def spatial_merged_h5ad() -> Path:
    return resolve_path("processed", "anndata", "spatial_merged.h5ad")


def models_dir() -> Path:
    return resolve_path("models")


def scvi_model_dir() -> Path:
    return resolve_path("models", "scvi")


def scanvi_model_dir() -> Path:
    return resolve_path("models", "scanvi")


def tangram_model_dir() -> Path:
    return resolve_path("models", "tangram")


def runs_dir() -> Path:
    """Training run artifacts (checkpoints, history JSON)."""
    return resolve_path("runs")


def metrics_dir() -> Path:
    """Saved evaluation metrics and benchmark tables."""
    return resolve_path("metrics")
