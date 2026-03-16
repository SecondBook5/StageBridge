"""
Helpers for robust H5AD writes/reads in long-running data pipelines.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import anndata


def validate_h5ad(path: Path, require_spatial: bool = False) -> tuple[bool, str | None]:
    """Return ``(ok, error_message)`` for an H5AD path."""
    path = Path(path)
    if not path.exists():
        return False, f"missing file: {path}"

    adata = None
    try:
        adata = anndata.read_h5ad(path, backed="r")
        _ = int(adata.n_obs)
        _ = int(adata.n_vars)
        if require_spatial and "spatial" not in adata.obsm:
            return False, f"missing obsm['spatial'] in {path}"
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if adata is not None and hasattr(adata, "isbacked") and adata.isbacked:
            adata.file.close()


def write_h5ad_atomic(
    adata: anndata.AnnData,
    output_path: Path,
    *,
    require_spatial: bool = False,
    compression: str = "lzf",
) -> Path:
    """Write AnnData atomically and validate before finalizing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)

    adata.write_h5ad(tmp_path, compression=compression)
    ok, err = validate_h5ad(tmp_path, require_spatial=require_spatial)
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid temporary H5AD {tmp_path}: {err}")

    os.replace(tmp_path, output_path)
    return output_path


def copy_file_atomic(src_path: Path, dest_path: Path) -> Path:
    """Copy a file into place atomically at the destination."""
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {src_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest_path.with_name(f".{dest_path.name}.tmp")
    tmp_dest.unlink(missing_ok=True)
    shutil.copy2(src_path, tmp_dest)
    os.replace(tmp_dest, dest_path)
    return dest_path
