"""
Multiprocessing-safe worker helpers for building per-sample H5AD shards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from stagebridge.io.h5ad_atomic import validate_h5ad, write_h5ad_atomic


def build_snrna_shard(
    input_path: str,
    output_path: str,
    max_cells_per_sample: int | None = None,
    compression: str = "lzf",
) -> dict[str, Any]:
    """Create or reuse one snRNA sample shard."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    ok, _err = validate_h5ad(out_path, require_spatial=False)
    if ok:
        return {"sample_path": str(in_path), "shard_path": str(out_path), "status": "reused"}

    if out_path.exists():
        out_path.unlink(missing_ok=True)

    from stagebridge.io.geo_snrna import load_snrna_sample

    adata = load_snrna_sample(in_path, max_cells_per_sample=max_cells_per_sample)
    write_h5ad_atomic(
        adata,
        out_path,
        require_spatial=False,
        compression=compression,
    )
    return {
        "sample_path": str(in_path),
        "shard_path": str(out_path),
        "status": "built",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }


def build_spatial_shard(
    input_path: str,
    output_path: str,
    max_spots_per_sample: int | None = None,
    compression: str = "lzf",
) -> dict[str, Any]:
    """Create or reuse one spatial sample shard."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    ok, _err = validate_h5ad(out_path, require_spatial=True)
    if ok:
        return {"sample_path": str(in_path), "shard_path": str(out_path), "status": "reused"}

    if out_path.exists():
        out_path.unlink(missing_ok=True)

    from stagebridge.io.geo_spatial import load_spatial_sample_from_tarball

    adata = load_spatial_sample_from_tarball(
        in_path,
        max_spots_per_sample=max_spots_per_sample,
    )
    write_h5ad_atomic(
        adata,
        out_path,
        require_spatial=True,
        compression=compression,
    )
    return {
        "sample_path": str(in_path),
        "shard_path": str(out_path),
        "status": "built",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    }
