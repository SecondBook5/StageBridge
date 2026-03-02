#!/usr/bin/env python
"""Build interim snRNA AnnData from GEO raw files (smoke/full)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.interim_build import build_snrna_interim_anndata
from stagebridge.io.manifests import write_resolved_config_yaml
from stagebridge.io.paths import resolve_run_paths
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


def _resolve_max_workers(cfg: DictConfig) -> int | None:
    return _int_or_none(_cfg_pick(cfg, ["pipeline.max_workers", "io.max_workers"]))


def _resolve_scratch_root(cfg: DictConfig) -> Path:
    raw = _cfg_pick(cfg, ["pipeline.scratch_root", "io.scratch_root"])
    if raw is not None:
        return Path(str(raw))
    return Path.home() / ".cache" / "stagebridge_scratch"


def _resolve_write_compression(cfg: DictConfig) -> str:
    raw = _cfg_pick(cfg, ["pipeline.h5ad_compression", "io.h5ad_compression"])
    return str(raw) if raw is not None else "lzf"


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
    run_id = (
        str(requested_run_id)
        if requested_run_id
        else f"snrna_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    max_workers = _resolve_max_workers(cfg)
    scratch_root = _resolve_scratch_root(cfg)
    keep_scratch = _bool_or_default(_cfg_pick(cfg, ["pipeline.keep_scratch", "io.keep_scratch"]), False)
    compression = _resolve_write_compression(cfg)

    result = build_snrna_interim_anndata(
        run_paths=run_paths,
        raw_dir=raw_dir,
        output_path=output_path,
        is_smoke=is_smoke,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
        max_cells_per_sample=max_cells_per_sample,
        max_workers=max_workers,
        scratch_root=scratch_root,
        keep_scratch=keep_scratch,
        compression=compression,
        timestamp=timestamp,
        repo_root=Path(__file__).resolve().parent.parent,
    )

    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    summary = result.summary
    print(f"snRNA output: {result.output_path}")
    print(f"n_obs={summary['n_obs']} n_vars={summary['n_vars']}")
    print(f"donors={summary['donors']}")
    print(f"stages={summary['stages']}")
    print(f"run_manifest={result.run_manifest_path}")
    print(f"data_audit={result.data_audit_path}")


if __name__ == "__main__":
    main()
