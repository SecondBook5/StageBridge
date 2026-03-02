"""Interim AnnData build orchestration for snRNA and spatial GEO pipelines."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
from typing import Any

import anndata
import numpy as np
import pandas as pd

from stagebridge.io.h5ad_atomic import copy_file_atomic, validate_h5ad, write_h5ad_atomic
from stagebridge.io.manifests import resolve_git_commit_hash, summarize_anndata, write_json
from stagebridge.io.paths import RunPaths
from stagebridge.io.pipeline_workers import build_snrna_shard, build_spatial_shard
from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class InterimBuildResult:
    """Return object for interim AnnData build runs."""

    run_id: str
    output_path: Path
    summary: dict[str, Any]
    run_manifest_path: Path
    data_audit_path: Path
    extra: dict[str, Any]


def default_max_workers(n_tasks: int, configured: int | None = None) -> int:
    """Resolve a bounded process count for shard building."""
    value = configured
    if value is None or value <= 0:
        cpu = os.cpu_count() or 2
        value = max(1, min(4, cpu // 2))
    return max(1, min(int(value), max(1, n_tasks)))


def default_scratch_root() -> Path:
    """Default scratch workspace for shard concat jobs."""
    return Path.home() / ".cache" / "stagebridge_scratch"


def _modality_values(adata: Any) -> list[str]:
    if "modality" not in adata.obs.columns:
        return []
    return sorted({str(v) for v in adata.obs["modality"].astype(str).tolist()})


def _close_if_backed(adata: anndata.AnnData | Any) -> None:
    if hasattr(adata, "isbacked") and adata.isbacked:
        adata.file.close()


def _build_full_snrna_on_disk(
    selected_df: pd.DataFrame,
    scratch_run_dir: Path,
    max_cells_per_sample: int | None,
    max_workers: int,
    compression: str,
) -> Path:
    from anndata.experimental import concat_on_disk

    shard_dir = scratch_run_dir / "snrna_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for row in selected_df.itertuples(index=False):
            shard_path = shard_dir / f"{row.sample_id}.h5ad"
            fut = pool.submit(
                build_snrna_shard,
                str(row.input_path),
                str(shard_path),
                max_cells_per_sample,
                compression,
            )
            futures[fut] = row.sample_id

        for fut in as_completed(futures):
            sample_id = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed snRNA shard build for {sample_id}: {exc}") from exc

    shard_paths = [shard_dir / f"{row.sample_id}.h5ad" for row in selected_df.itertuples(index=False)]
    for shard_path in shard_paths:
        ok, err = validate_h5ad(shard_path, require_spatial=False)
        if not ok:
            raise RuntimeError(f"Invalid snRNA shard before concat: {shard_path} ({err})")

    concat_out = scratch_run_dir / "snrna_concat_full.h5ad"
    concat_out.unlink(missing_ok=True)
    concat_on_disk(
        in_files=[str(p) for p in shard_paths],
        out_file=str(concat_out),
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    ok, err = validate_h5ad(concat_out, require_spatial=False)
    if not ok:
        raise RuntimeError(f"On-disk snRNA concat produced invalid output: {concat_out} ({err})")
    return concat_out


def _build_full_spatial_on_disk(
    selected_df: pd.DataFrame,
    scratch_run_dir: Path,
    max_spots_per_sample: int | None,
    max_workers: int,
    compression: str,
) -> Path:
    from anndata.experimental import concat_on_disk

    shard_dir = scratch_run_dir / "spatial_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    futures = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for row in selected_df.itertuples(index=False):
            shard_path = shard_dir / f"{row.sample_id}.h5ad"
            fut = pool.submit(
                build_spatial_shard,
                str(row.input_path),
                str(shard_path),
                max_spots_per_sample,
                compression,
            )
            futures[fut] = row.sample_id

        for fut in as_completed(futures):
            sample_id = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                raise RuntimeError(f"Failed spatial shard build for {sample_id}: {exc}") from exc

    shard_paths = [shard_dir / f"{row.sample_id}.h5ad" for row in selected_df.itertuples(index=False)]
    for shard_path in shard_paths:
        ok, err = validate_h5ad(shard_path, require_spatial=True)
        if not ok:
            raise RuntimeError(f"Invalid spatial shard before concat: {shard_path} ({err})")

    concat_out = scratch_run_dir / "spatial_concat_full.h5ad"
    concat_out.unlink(missing_ok=True)
    concat_on_disk(
        in_files=[str(p) for p in shard_paths],
        out_file=str(concat_out),
        axis=0,
        join="outer",
        merge="same",
        index_unique=None,
    )
    ok, err = validate_h5ad(concat_out, require_spatial=True)
    if not ok:
        raise RuntimeError(f"On-disk spatial concat produced invalid output: {concat_out} ({err})")
    return concat_out


def build_snrna_interim_anndata(
    *,
    run_paths: RunPaths,
    raw_dir: Path,
    output_path: Path,
    is_smoke: bool,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
    max_cells_per_sample: int | None = None,
    max_workers: int | None = None,
    scratch_root: Path | None = None,
    keep_scratch: bool = False,
    compression: str = "lzf",
    timestamp: str | None = None,
    repo_root: Path | None = None,
) -> InterimBuildResult:
    """Build smoke/full snRNA AnnData and run-scoped manifest/audit JSON."""
    from stagebridge.io.geo_snrna import (
        apply_snrna_smoke_limits,
        discover_snrna_files,
        load_snrna_dataset,
    )

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    discovered_df = discover_snrna_files(raw_dir)
    selected_df = apply_snrna_smoke_limits(
        discovered_df,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
    )
    if selected_df.empty:
        raise RuntimeError("No snRNA samples selected after applying limits.")

    workers = default_max_workers(len(selected_df), configured=max_workers)
    scratch_root = Path(scratch_root) if scratch_root is not None else default_scratch_root()
    scratch_run_root = scratch_root / run_paths.run_id
    scratch_run_dir = scratch_run_root / "snrna"

    adata: anndata.AnnData | Any = None
    loaded_df = selected_df
    try:
        if is_smoke:
            adata, loaded_df = load_snrna_dataset(
                raw_dir=raw_dir,
                max_donors=max_donors,
                max_samples_per_stage=max_samples_per_stage,
                max_cells_per_sample=max_cells_per_sample,
            )
            write_h5ad_atomic(adata, output_path, require_spatial=False, compression=compression)
        else:
            scratch_run_dir.mkdir(parents=True, exist_ok=True)
            concat_path = _build_full_snrna_on_disk(
                selected_df=selected_df,
                scratch_run_dir=scratch_run_dir,
                max_cells_per_sample=max_cells_per_sample,
                max_workers=workers,
                compression=compression,
            )
            copy_file_atomic(concat_path, output_path)
            adata = anndata.read_h5ad(output_path, backed="r")
    finally:
        if scratch_run_dir.exists() and (not is_smoke) and (not keep_scratch):
            shutil.rmtree(scratch_run_dir, ignore_errors=True)
            try:
                scratch_run_root.rmdir()
            except OSError:
                pass

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output was not created: {output_path}")

    summary = summarize_anndata(adata, donor_col="donor_id", stage_col="stage")
    modality_values = _modality_values(adata)
    git_hash = resolve_git_commit_hash(repo_root or Path.cwd())

    run_manifest = {
        "run_id": run_paths.run_id,
        "timestamp": ts,
        "pipeline": "snrna",
        "inputs": {
            "raw_dir": str(raw_dir),
            "total_file_count": int(len(discovered_df)),
            "selected_file_count": int(len(selected_df)),
            "selected_sample_ids": [str(x) for x in loaded_df["sample_id"].tolist()],
        },
        "outputs": {
            "anndata": str(output_path),
            "run_manifest": str(run_paths.run_manifest_json),
            "data_audit": str(run_paths.data_audit_json),
            "config_resolved": str(run_paths.config_resolved_yaml),
        },
        "anndata": summary,
        "git_commit": git_hash,
        "smoke": {
            "enabled": bool(is_smoke),
            "max_donors": max_donors,
            "max_samples_per_stage": max_samples_per_stage,
            "max_cells_per_sample": max_cells_per_sample,
        },
        "performance": {
            "max_workers": workers,
            "h5ad_compression": compression,
            "scratch_root": str(scratch_root),
            "keep_scratch": bool(keep_scratch),
        },
    }
    data_audit = {
        "run_id": run_paths.run_id,
        "timestamp": ts,
        "pipeline": "snrna",
        "ok": True,
        "checks": {
            "output_exists": output_path.exists(),
            "has_counts_layer": "counts" in adata.layers,
            "modality_values": modality_values,
        },
        "anndata": summary,
    }

    write_json(run_paths.run_manifest_json, run_manifest)
    write_json(run_paths.data_audit_json, data_audit)

    _close_if_backed(adata)
    return InterimBuildResult(
        run_id=run_paths.run_id,
        output_path=output_path,
        summary=summary,
        run_manifest_path=run_paths.run_manifest_json,
        data_audit_path=run_paths.data_audit_json,
        extra={},
    )


def build_spatial_interim_anndata(
    *,
    run_paths: RunPaths,
    extracted_dir: Path,
    output_path: Path,
    is_smoke: bool,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
    max_spots_per_sample: int | None = None,
    max_workers: int | None = None,
    scratch_root: Path | None = None,
    keep_scratch: bool = False,
    compression: str = "lzf",
    timestamp: str | None = None,
    repo_root: Path | None = None,
) -> InterimBuildResult:
    """Build smoke/full spatial AnnData and run-scoped manifest/audit JSON."""
    from stagebridge.io.geo_spatial import (
        apply_spatial_smoke_limits,
        discover_spatial_tarballs,
        load_spatial_dataset,
    )

    extracted_dir = Path(extracted_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    discovered_df = discover_spatial_tarballs(extracted_dir)
    selected_df = apply_spatial_smoke_limits(
        discovered_df,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
    )
    if selected_df.empty:
        raise RuntimeError("No spatial samples selected after applying limits.")

    workers = default_max_workers(len(selected_df), configured=max_workers)
    scratch_root = Path(scratch_root) if scratch_root is not None else default_scratch_root()
    scratch_run_root = scratch_root / run_paths.run_id
    scratch_run_dir = scratch_run_root / "spatial"

    adata: anndata.AnnData | Any = None
    loaded_df = selected_df
    try:
        if is_smoke:
            adata, loaded_df = load_spatial_dataset(
                extracted_dir=extracted_dir,
                max_donors=max_donors,
                max_samples_per_stage=max_samples_per_stage,
                max_spots_per_sample=max_spots_per_sample,
            )
            write_h5ad_atomic(adata, output_path, require_spatial=True, compression=compression)
        else:
            scratch_run_dir.mkdir(parents=True, exist_ok=True)
            concat_path = _build_full_spatial_on_disk(
                selected_df=selected_df,
                scratch_run_dir=scratch_run_dir,
                max_spots_per_sample=max_spots_per_sample,
                max_workers=workers,
                compression=compression,
            )
            copy_file_atomic(concat_path, output_path)
            adata = anndata.read_h5ad(output_path, backed="r")
    finally:
        if scratch_run_dir.exists() and (not is_smoke) and (not keep_scratch):
            shutil.rmtree(scratch_run_dir, ignore_errors=True)
            try:
                scratch_run_root.rmdir()
            except OSError:
                pass

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output was not created: {output_path}")
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial AnnData missing required obsm['spatial'] coordinates.")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    coord_min = coords.min(axis=0).tolist()
    coord_max = coords.max(axis=0).tolist()
    summary = summarize_anndata(adata, donor_col="donor_id", stage_col="stage")
    modality_values = _modality_values(adata)
    git_hash = resolve_git_commit_hash(repo_root or Path.cwd())

    run_manifest = {
        "run_id": run_paths.run_id,
        "timestamp": ts,
        "pipeline": "spatial",
        "inputs": {
            "extracted_dir": str(extracted_dir),
            "total_file_count": int(len(discovered_df)),
            "selected_file_count": int(len(selected_df)),
            "selected_sample_ids": [str(x) for x in loaded_df["sample_id"].tolist()],
        },
        "outputs": {
            "anndata": str(output_path),
            "run_manifest": str(run_paths.run_manifest_json),
            "data_audit": str(run_paths.data_audit_json),
            "config_resolved": str(run_paths.config_resolved_yaml),
        },
        "anndata": {
            **summary,
            "spatial_coord_min": coord_min,
            "spatial_coord_max": coord_max,
        },
        "git_commit": git_hash,
        "smoke": {
            "enabled": bool(is_smoke),
            "max_donors": max_donors,
            "max_samples_per_stage": max_samples_per_stage,
            "max_spots_per_sample": max_spots_per_sample,
        },
        "performance": {
            "max_workers": workers,
            "h5ad_compression": compression,
            "scratch_root": str(scratch_root),
            "keep_scratch": bool(keep_scratch),
        },
    }
    data_audit = {
        "run_id": run_paths.run_id,
        "timestamp": ts,
        "pipeline": "spatial",
        "ok": True,
        "checks": {
            "output_exists": output_path.exists(),
            "has_counts_layer": "counts" in adata.layers,
            "has_spatial_coords": "spatial" in adata.obsm,
            "modality_values": modality_values,
        },
        "anndata": {
            **summary,
            "spatial_coord_min": coord_min,
            "spatial_coord_max": coord_max,
        },
    }

    write_json(run_paths.run_manifest_json, run_manifest)
    write_json(run_paths.data_audit_json, data_audit)

    _close_if_backed(adata)
    return InterimBuildResult(
        run_id=run_paths.run_id,
        output_path=output_path,
        summary=summary,
        run_manifest_path=run_paths.run_manifest_json,
        data_audit_path=run_paths.data_audit_json,
        extra={
            "spatial_coord_min": coord_min,
            "spatial_coord_max": coord_max,
        },
    )
