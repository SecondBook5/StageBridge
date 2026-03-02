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

from stagebridge.io.geo_snrna import (
    discover_snrna_files,
    apply_snrna_smoke_limits,
    load_snrna_dataset,
)
from stagebridge.io.manifests import (
    resolve_git_commit_hash,
    summarize_anndata,
    write_json,
    write_resolved_config_yaml,
)
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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    experiment_name = str(OmegaConf.select(cfg, "experiment.name") or "").lower()
    is_smoke = experiment_name == "smoke"

    max_donors: int | None = None
    max_samples_per_stage: int | None = None
    max_cells_per_sample: int | None = None
    if is_smoke:
        max_donors = _smoke_value(cfg, "experiment.max_donors", 2)
        max_samples_per_stage = _smoke_value(cfg, "experiment.max_samples_per_stage", 1)
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

    adata, loaded_df = load_snrna_dataset(
        raw_dir=raw_dir,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
        max_cells_per_sample=max_cells_per_sample,
    )
    adata.write_h5ad(output_path)

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


if __name__ == "__main__":
    main()
