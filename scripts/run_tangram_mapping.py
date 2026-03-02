#!/usr/bin/env python
"""Run Tangram mapping from HLCA-labeled snRNA to spatial spots."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.manifests import write_resolved_config_yaml
from stagebridge.io.paths import resolve_run_paths
from stagebridge.io.tangram import run_tangram_hlca_projection
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
    value = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
    return Path(str(value))


def _resolve_experiment_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "experiment.name") or "full").lower()


def _resolve_input_snrna_h5ad(cfg: DictConfig) -> Path:
    exp = _resolve_experiment_name(cfg)
    value = _cfg_pick(cfg, ["data.input_snrna_tangram_h5ad"])
    if value is not None:
        path = Path(str(value))
        if path.suffix == ".h5ad":
            return path

    if exp != "smoke":
        full_value = _cfg_pick(cfg, ["data.input_snrna_full_h5ad"])
        if full_value is not None:
            full_path = Path(str(full_value))
            if full_path.suffix == ".h5ad":
                return full_path

    name = "snrna_smoke.h5ad" if exp == "smoke" else "snrna_full.h5ad"
    interim_dir = _cfg_pick(cfg, ["data.interim.anndata_snrna_dir", "data.snrna_interim_dir"])
    if interim_dir is not None:
        return Path(str(interim_dir)) / name
    return _resolve_data_root(cfg) / "interim" / "anndata" / "snrna" / name


def _resolve_input_spatial_h5ad(cfg: DictConfig) -> Path:
    exp = _resolve_experiment_name(cfg)
    value = _cfg_pick(cfg, ["data.input_spatial_tangram_h5ad"])
    if value is not None:
        path = Path(str(value))
        if path.suffix == ".h5ad":
            return path

    if exp != "smoke":
        full_value = _cfg_pick(cfg, ["data.input_spatial_full_h5ad"])
        if full_value is not None:
            full_path = Path(str(full_value))
            if full_path.suffix == ".h5ad":
                return full_path

    name = "spatial_smoke.h5ad" if exp == "smoke" else "spatial_full.h5ad"
    interim_dir = _cfg_pick(cfg, ["data.interim.anndata_spatial_dir", "data.spatial_interim_dir"])
    if interim_dir is not None:
        return Path(str(interim_dir)) / name
    return _resolve_data_root(cfg) / "interim" / "anndata" / "spatial" / name


def _resolve_processed_hlca_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_hlca_dir", "data.processed.hlca_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "processed" / "hlca"


def _resolve_processed_tangram_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_tangram_dir", "data.processed.tangram_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "processed" / "tangram"


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id_cfg = OmegaConf.select(cfg, "run_id")
    run_id = (
        str(run_id_cfg)
        if run_id_cfg
        else f"tangram_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    snrna_h5ad = _resolve_input_snrna_h5ad(cfg)
    spatial_h5ad = _resolve_input_spatial_h5ad(cfg)
    hlca_dir = _resolve_processed_hlca_dir(cfg)
    tangram_dir = _resolve_processed_tangram_dir(cfg)
    hlca_labels = hlca_dir / "snrna_full_hlca_labels.parquet"

    tangram_dir.mkdir(parents=True, exist_ok=True)
    mapping_h5ad = tangram_dir / "tangram_map_full.h5ad"
    spatial_out_h5ad = tangram_dir / "spatial_tangram_full.h5ad"
    scores_parquet = tangram_dir / "spatial_tangram_celltype_scores.parquet"
    report_path = run_paths.tables_dir / "tangram_report.json"

    tangram_cfg = OmegaConf.select(cfg, "tangram")
    if tangram_cfg is None:
        tangram_cfg_dict: dict[str, object] = {}
    else:
        tangram_cfg_dict = OmegaConf.to_container(tangram_cfg, resolve=True)  # type: ignore[assignment]
        if not isinstance(tangram_cfg_dict, dict):
            raise ValueError("cfg.tangram must resolve to a dictionary when provided.")

    result = run_tangram_hlca_projection(
        run_id=run_paths.run_id,
        snrna_h5ad_path=snrna_h5ad,
        spatial_h5ad_path=spatial_h5ad,
        labels_parquet_path=hlca_labels,
        output_mapping_h5ad_path=mapping_h5ad,
        output_spatial_h5ad_path=spatial_out_h5ad,
        output_scores_parquet_path=scores_parquet,
        report_path=report_path,
        tangram_cfg=tangram_cfg_dict,
    )
    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    if not mapping_h5ad.exists():
        raise FileNotFoundError(f"Missing mapping output: {mapping_h5ad}")
    if not spatial_out_h5ad.exists():
        raise FileNotFoundError(f"Missing spatial output: {spatial_out_h5ad}")
    if not scores_parquet.exists():
        raise FileNotFoundError(f"Missing scores output: {scores_parquet}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing Tangram report: {report_path}")

    counts = result.report.get("counts", {})
    print(f"n_spots={counts.get('n_spots')}")
    print(f"n_training_genes={counts.get('n_training_genes')}")
    print(f"n_label_profiles_used={counts.get('n_label_profiles_used')}")
    print(f"peak_rss_mb={result.report.get('peak_rss_mb')}")

    print(
        json.dumps(
            {
                "ok": True,
                "run_id": result.run_id,
                "mapping_h5ad": str(result.mapping_h5ad_path),
                "spatial_h5ad": str(result.spatial_h5ad_path),
                "scores_parquet": str(result.scores_parquet_path),
            }
        )
    )


if __name__ == "__main__":
    main()
