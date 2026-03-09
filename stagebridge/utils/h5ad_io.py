"""Shared low-level H5AD reading helpers used by tangram_mapper and notebook_api."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def decode_h5_array(values: np.ndarray) -> np.ndarray:
    """Decode byte-string arrays from HDF5 into Python str arrays."""
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value) for value in values],
            dtype=object,
        )
    return values


def read_h5ad_obs_column(group: h5py.Group, name: str, rows: np.ndarray) -> np.ndarray:
    """Read a single obs column (plain or categorical) for the given row indices."""
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return decode_h5_array(obj[rows])
    if "categories" in obj and "codes" in obj:
        categories = decode_h5_array(obj["categories"][:])
        codes = np.asarray(obj["codes"][rows], dtype=np.int64)
        values = np.empty(rows.shape[0], dtype=object)
        missing = codes < 0
        values[missing] = None
        valid = ~missing
        values[valid] = categories[codes[valid]]
        return values
    raise TypeError(f"Unsupported obs column encoding for '{name}'.")


def read_h5ad_obs_column_or_default(
    group: h5py.Group,
    name: str,
    rows: np.ndarray,
    *,
    default: np.ndarray | None = None,
) -> np.ndarray:
    """Read an obs column, falling back to *default* if the column is missing."""
    if name in group:
        return read_h5ad_obs_column(group, name, rows)
    if default is not None:
        return default
    raise KeyError(f"Missing required obs column '{name}'.")


def _resolve_donor_column(group: h5py.Group, name: str, rows: np.ndarray) -> np.ndarray:
    """Handle donor_id / patient_id aliasing."""
    if name == "donor_id":
        col = "donor_id" if "donor_id" in group else "patient_id"
        return read_h5ad_obs_column(group, col, rows)
    if name == "patient_id":
        col = "patient_id" if "patient_id" in group else "donor_id"
        return read_h5ad_obs_column(group, col, rows)
    return read_h5ad_obs_column(group, name, rows)


def read_h5ad_obs_frame(
    h5ad_path: Path,
    *,
    columns: list[str],
    rows: np.ndarray | None = None,
) -> pd.DataFrame:
    """Read selected obs columns from an h5ad file without loading the full object."""
    with h5py.File(h5ad_path, "r") as handle:
        obs_group = handle["obs"]
        chosen_rows = (
            np.arange(obs_group["_index"].shape[0], dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64)
        )
        obs_index = decode_h5_array(obs_group["_index"][chosen_rows])
        values: dict[str, np.ndarray] = {}
        for column in columns:
            if column in {"donor_id", "patient_id"}:
                values[column] = _resolve_donor_column(obs_group, column, chosen_rows)
            elif column in {"spot_id", "barcode", "sample_id"}:
                values[column] = read_h5ad_obs_column_or_default(
                    obs_group, column, chosen_rows, default=obs_index
                )
            else:
                values[column] = read_h5ad_obs_column(obs_group, column, chosen_rows)
        frame = pd.DataFrame(
            values,
            index=pd.Index(obs_index.astype(str), name=str(obs_group.attrs.get("_index", "_index"))),
        )
    return frame


def read_h5ad_spatial_coords(path: Path, rows: np.ndarray) -> np.ndarray:
    """Read spatial coordinates from obsm/spatial for the given row indices."""
    with h5py.File(path, "r") as handle:
        spatial = handle["obsm"]["spatial"]
        return np.asarray(spatial[rows], dtype=np.float32)


def read_h5ad_n_vars(path: Path) -> int:
    """Return the number of variables (genes) in an h5ad file."""
    with h5py.File(path, "r") as handle:
        X = handle["X"]
        if isinstance(X, h5py.Dataset):
            return int(X.shape[1])
        shape = X.attrs.get("shape")
        return int(shape[1])


def read_h5ad_var_index(h5ad_path: Path) -> pd.Index:
    """Read the var index from an h5ad file."""
    with h5py.File(h5ad_path, "r") as handle:
        var_group = handle["var"]
        index = decode_h5_array(var_group["_index"][:])
        return pd.Index(index, name=str(var_group.attrs.get("_index", "_index")))
