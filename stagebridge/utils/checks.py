"""Defensive checking utilities for StageBridge."""

from __future__ import annotations

from pathlib import Path
import anndata


def assert_layer(adata: anndata.AnnData, layer: str) -> None:
    """Raise KeyError with a helpful message if *layer* is missing."""
    if layer not in adata.layers:
        raise KeyError(
            f"Expected layer '{layer}' in AnnData but it was not found.\n"
            f"Available layers: {list(adata.layers.keys())}"
        )


def assert_obsm_key(adata: anndata.AnnData, key: str) -> None:
    """Raise KeyError with a helpful message if *key* is missing from obsm."""
    if key not in adata.obsm:
        raise KeyError(
            f"Expected obsm key '{key}' in AnnData but it was not found.\n"
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )


def assert_file(path: Path, description: str = "file") -> None:
    """Raise FileNotFoundError with context if *path* does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected {description} not found: {path}")
