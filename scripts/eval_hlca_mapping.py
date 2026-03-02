#!/usr/bin/env python
"""Evaluate HLCA full-mapping outputs against reference neighborhoods."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.hlca import evaluate_hlca_mapping_outputs
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
    run_id = (
        str(run_id_cfg)
        if run_id_cfg
        else f"hlca_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    processed_anndata_dir = _resolve_processed_anndata_dir(cfg)
    processed_hlca_dir = _resolve_processed_hlca_dir(cfg)
    processed_anndata_dir.mkdir(parents=True, exist_ok=True)
    processed_hlca_dir.mkdir(parents=True, exist_ok=True)

    latent_h5ad = processed_anndata_dir / "snrna_hlca_latent_full.h5ad"
    labels_parquet = processed_hlca_dir / "snrna_full_hlca_labels.parquet"
    report_path = run_paths.tables_dir / "hlca_validation_report.json"

    eval_cfg = OmegaConf.select(cfg, "hlca_eval")
    if eval_cfg is None:
        eval_cfg_dict: dict[str, object] = {}
    else:
        eval_cfg_dict = OmegaConf.to_container(eval_cfg, resolve=True)  # type: ignore[assignment]
        if not isinstance(eval_cfg_dict, dict):
            raise ValueError("cfg.hlca_eval must resolve to a dictionary when provided.")

    result = evaluate_hlca_mapping_outputs(
        run_id=run_paths.run_id,
        latent_h5ad_path=latent_h5ad,
        labels_parquet_path=labels_parquet,
        report_path=report_path,
        processed_hlca_dir=processed_hlca_dir,
        eval_cfg=eval_cfg_dict,
    )
    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    report = result.report
    knn = report["knn_validation"]
    counts = report["counts"]
    checks = report["checks"]

    print(f"report={result.report_path}")
    print(f"label_coverage={counts['label_coverage']:.4f}")
    print(f"majority_agreement={knn['majority_agreement']:.4f}")
    print(f"mean_neighbor_agreement={knn['mean_neighbor_agreement']:.4f}")
    print(f"n_query_cells_evaluated={knn['n_query_cells_evaluated']}")
    print(f"n_reference_cells_used={knn['n_reference_cells_used']}")
    print(f"peak_rss_mb={report['peak_rss_mb']:.2f}")
    print(f"checks={json.dumps(checks)}")
    print(json.dumps({"ok": bool(report["ok"]), "run_id": run_paths.run_id, "report": str(result.report_path)}))


if __name__ == "__main__":
    main()
