#!/usr/bin/env python
"""Run full-scale HLCA mapping for snRNA AnnData."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.hlca import map_full_snrna_with_hlca
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


def _resolve_input_snrna_full(cfg: DictConfig) -> Path:
    value = _cfg_pick(
        cfg,
        [
            "data.input_snrna_full_h5ad",
            "data.interim.anndata_snrna_dir",
        ],
    )
    if value is None:
        data_root = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
        return Path(str(data_root)) / "interim" / "anndata" / "snrna" / "snrna_full.h5ad"
    path = Path(str(value))
    if path.suffix == ".h5ad":
        return path
    return path / "snrna_full.h5ad"


def _resolve_processed_anndata_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_anndata_dir", "data.processed.anndata_dir"])
    if value is not None:
        return Path(str(value))
    data_root = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
    return Path(str(data_root)) / "processed" / "anndata"


def _resolve_processed_hlca_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_hlca_dir", "data.processed.hlca_dir"])
    if value is not None:
        return Path(str(value))
    data_root = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
    return Path(str(data_root)) / "processed" / "hlca"


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id_cfg = OmegaConf.select(cfg, "run_id")
    run_id = str(run_id_cfg) if run_id_cfg else f"hlca_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    logs_dir = run_paths.run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    snrna_full_path = _resolve_input_snrna_full(cfg)
    processed_anndata_dir = _resolve_processed_anndata_dir(cfg)
    processed_hlca_dir = _resolve_processed_hlca_dir(cfg)
    processed_anndata_dir.mkdir(parents=True, exist_ok=True)
    processed_hlca_dir.mkdir(parents=True, exist_ok=True)

    latent_h5ad = processed_anndata_dir / "snrna_hlca_latent_full.h5ad"
    labels_parquet = processed_hlca_dir / "snrna_full_hlca_labels.parquet"

    mapping_report = run_paths.tables_dir / "hlca_mapping_report.json"
    gene_report = run_paths.tables_dir / "hlca_gene_id_report.json"
    progress_json = logs_dir / "progress.json"

    hlca_cfg = OmegaConf.to_container(cfg.hlca, resolve=True)
    if not isinstance(hlca_cfg, dict):
        raise ValueError("cfg.hlca must resolve to a dictionary.")

    result = map_full_snrna_with_hlca(
        run_id=run_paths.run_id,
        snrna_h5ad_path=snrna_full_path,
        output_latent_h5ad_path=latent_h5ad,
        output_labels_parquet_path=labels_parquet,
        mapping_report_path=mapping_report,
        gene_report_path=gene_report,
        progress_path=progress_json,
        processed_hlca_dir=processed_hlca_dir,
        hlca_cfg=hlca_cfg,
    )

    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    if not latent_h5ad.exists():
        raise FileNotFoundError(f"Missing expected latent output: {latent_h5ad}")
    if not labels_parquet.exists():
        raise FileNotFoundError(f"Missing expected labels output: {labels_parquet}")
    if not mapping_report.exists():
        raise FileNotFoundError(f"Missing expected mapping report: {mapping_report}")
    if not gene_report.exists():
        raise FileNotFoundError(f"Missing expected gene report: {gene_report}")

    print(f"overlap_percent={result.overlap_percent:.2f}")
    print(f"latent_shape={list(result.latent_shape)}")
    print(f"peak_rss_mb={result.peak_rss_mb:.2f}")
    print(f"total_wall_time_seconds={result.wall_time_seconds:.2f}")
    print(f"top10_hlca_label_counts={json.dumps(result.top10_labels)}")

    summary = {
        "ok": True,
        "run_id": result.run_id,
        "latent_h5ad": str(result.latent_h5ad_path),
        "labels_parquet": str(result.labels_parquet_path),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
