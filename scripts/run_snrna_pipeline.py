#!/usr/bin/env python
"""Build interim snRNA AnnData from GEO raw files (smoke/full)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anndata
import hydra
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.geo_snrna import (
    discover_snrna_files,
    apply_snrna_smoke_limits,
    load_snrna_dataset,
)
from stagebridge.io.h5ad_atomic import copy_file_atomic, validate_h5ad, write_h5ad_atomic
from stagebridge.io.manifests import (
    resolve_git_commit_hash,
    summarize_anndata,
    write_json,
    write_resolved_config_yaml,
)
from stagebridge.io.paths import resolve_run_paths
from stagebridge.io.pipeline_workers import build_snrna_shard
from stagebridge.logging_utils import configure_root_logger, get_logger

configure_root_logger()
log = get_logger(__name__)


def _cfg_pick(cfg: DictConfig, keys: list[str]) -> object | None:
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            return value
    return None


def _resolve_data_root(cfg: DictConfig) -> Path:
    data_root = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
    return Path(str(data_root))


def _resolve_snrna_raw_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.raw.geo_snrna_dir", "data.snrna_raw_dir"])
    if value is not None:
        base = Path(str(value))
        if base.name == "extracted":
            return base
        extracted = base / "extracted"
        return extracted if extracted.exists() else base

    data_root = _resolve_data_root(cfg)
    candidates = [
        data_root / "raw" / "geo" / "GSE308103_snrna" / "extracted",
        data_root / "data" / "raw" / "geo" / "GSE308103_snrna" / "extracted",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_snrna_interim_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.interim.anndata_snrna_dir", "data.snrna_interim_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "interim" / "anndata" / "snrna"


def _smoke_value(cfg: DictConfig, key: str, default_value: int) -> int | None:
    value = OmegaConf.select(cfg, key)
    if value is None:
        return default_value
    return int(value)


def _int_or_none(value: object | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _bool_or_default(value: object | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _resolve_max_workers(cfg: DictConfig, n_tasks: int) -> int:
    configured = _int_or_none(_cfg_pick(cfg, ["pipeline.max_workers", "io.max_workers"]))
    if configured is None:
        cpu = os.cpu_count() or 2
        configured = max(1, min(4, cpu // 2))
    return max(1, min(configured, max(1, n_tasks)))


def _resolve_scratch_root(cfg: DictConfig) -> Path:
    raw = _cfg_pick(cfg, ["pipeline.scratch_root", "io.scratch_root"])
    if raw is not None:
        return Path(str(raw))
    return Path.home() / ".cache" / "stagebridge_scratch"


def _resolve_write_compression(cfg: DictConfig) -> str:
    raw = _cfg_pick(cfg, ["pipeline.h5ad_compression", "io.h5ad_compression"])
    return str(raw) if raw is not None else "lzf"


def _build_full_snrna_on_disk(
    selected_df,
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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    experiment_name = str(OmegaConf.select(cfg, "experiment.name") or "").lower()
    is_smoke = experiment_name == "smoke"

    max_donors: int | None = _int_or_none(_cfg_pick(cfg, ["pipeline.max_donors", "io.max_donors"]))
    max_samples_per_stage: int | None = _int_or_none(
        _cfg_pick(cfg, ["pipeline.max_samples_per_stage", "io.max_samples_per_stage"])
    )
    max_cells_per_sample: int | None = _int_or_none(
        _cfg_pick(cfg, ["pipeline.max_cells_per_sample", "io.max_cells_per_sample"])
    )
    if is_smoke:
        if max_donors is None:
            max_donors = _smoke_value(cfg, "experiment.max_donors", 2)
        if max_samples_per_stage is None:
            max_samples_per_stage = _smoke_value(cfg, "experiment.max_samples_per_stage", 1)
        if max_cells_per_sample is None:
            max_cells_per_sample = _smoke_value(cfg, "experiment.max_cells_per_sample", 20000)

    raw_dir = _resolve_snrna_raw_dir(cfg)
    interim_dir = _resolve_snrna_interim_dir(cfg)
    interim_dir.mkdir(parents=True, exist_ok=True)

    output_name = "snrna_smoke.h5ad" if is_smoke else "snrna_full.h5ad"
    output_path = interim_dir / output_name

    requested_run_id = OmegaConf.select(cfg, "run_id")
    run_id = str(requested_run_id) if requested_run_id else f"snrna_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    discovered_df = discover_snrna_files(raw_dir)
    selected_df = apply_snrna_smoke_limits(
        discovered_df,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
    )
    if selected_df.empty:
        raise RuntimeError("No snRNA samples selected after applying limits.")

    compression = _resolve_write_compression(cfg)
    max_workers = _resolve_max_workers(cfg, n_tasks=len(selected_df))
    scratch_root = _resolve_scratch_root(cfg)
    keep_scratch = _bool_or_default(_cfg_pick(cfg, ["pipeline.keep_scratch", "io.keep_scratch"]), False)
    scratch_run_root = scratch_root / run_paths.run_id
    scratch_run_dir = scratch_run_root / "snrna"

    adata = None
    loaded_df = selected_df
    try:
        if is_smoke:
            adata, loaded_df = load_snrna_dataset(
                raw_dir=raw_dir,
                max_donors=max_donors,
                max_samples_per_stage=max_samples_per_stage,
                max_cells_per_sample=max_cells_per_sample,
            )
            write_h5ad_atomic(adata, output_path, compression=compression)
        else:
            scratch_run_dir.mkdir(parents=True, exist_ok=True)
            concat_path = _build_full_snrna_on_disk(
                selected_df=selected_df,
                scratch_run_dir=scratch_run_dir,
                max_cells_per_sample=max_cells_per_sample,
                max_workers=max_workers,
                compression=compression,
            )
            copy_file_atomic(concat_path, output_path)
            adata = anndata.read_h5ad(output_path, backed="r")
    finally:
        if (
            scratch_run_dir.exists()
            and not is_smoke
            and not keep_scratch
        ):
            shutil.rmtree(scratch_run_dir, ignore_errors=True)
            try:
                scratch_run_root.rmdir()
            except OSError:
                pass

    if not output_path.exists():
        raise FileNotFoundError(f"Expected output was not created: {output_path}")

    summary = summarize_anndata(adata, donor_col="donor_id", stage_col="stage")
    git_hash = resolve_git_commit_hash(Path(__file__).resolve().parent.parent)
    modality_values = (
        sorted({str(v) for v in adata.obs["modality"].astype(str).tolist()})
        if "modality" in adata.obs.columns
        else []
    )

    run_manifest = {
        "run_id": run_paths.run_id,
        "timestamp": timestamp,
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
            "max_workers": max_workers,
            "h5ad_compression": compression,
            "scratch_root": str(scratch_root),
            "keep_scratch": keep_scratch,
        },
    }

    data_audit = {
        "run_id": run_paths.run_id,
        "timestamp": timestamp,
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
    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    log.info("snRNA output written: %s", output_path)
    log.info("Run manifest written: %s", run_paths.run_manifest_json)

    print(f"snRNA output: {output_path}")
    print(f"n_obs={summary['n_obs']} n_vars={summary['n_vars']}")
    print(f"donors={summary['donors']}")
    print(f"stages={summary['stages']}")
    print(f"run_manifest={run_paths.run_manifest_json}")
    print(f"data_audit={run_paths.data_audit_json}")

    if hasattr(adata, "isbacked") and adata.isbacked:
        adata.file.close()


if __name__ == "__main__":
    main()
