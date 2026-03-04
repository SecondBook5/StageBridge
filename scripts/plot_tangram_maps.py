#!/usr/bin/env python
"""Plot Tangram per-celltype and winner-label spatial maps."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anndata
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.io.manifests import write_json, write_resolved_config_yaml
from stagebridge.io.paths import resolve_run_paths
from stagebridge.logging_utils import configure_root_logger, get_logger
from stagebridge.viz.spatial_plots import plot_tangram_celltype_maps, plot_tangram_winner_map

configure_root_logger()
log = get_logger(__name__)


def _cfg_pick(cfg: DictConfig, keys: list[str]) -> object | None:
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            return value
    return None


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_data_root(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.data_root"]) or "/mnt/e/StageBridge_data"
    return Path(str(value))


def _resolve_tangram_spatial_h5ad(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["tangram_plot.input_spatial_h5ad"])
    if value is not None:
        path = Path(str(value))
        if path.suffix == ".h5ad":
            return path

    tdir = _cfg_pick(cfg, ["data.output_processed_tangram_dir", "data.processed.tangram_dir"])
    if tdir is not None:
        return Path(str(tdir)) / "spatial_tangram_full.h5ad"

    return _resolve_data_root(cfg) / "processed" / "tangram" / "spatial_tangram_full.h5ad"


def _resolve_output_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["tangram_plot.output_dir"])
    if value is not None:
        out = Path(str(value))
        if out.is_absolute():
            return out
        return _resolve_repo_root() / out
    return _resolve_repo_root() / "outputs" / "figures"


def _resolve_sample_id(cfg: DictConfig) -> str | None:
    value = _cfg_pick(cfg, ["tangram_plot.sample_id"])
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _resolve_float(cfg: DictConfig, key: str, default: float) -> float:
    value = OmegaConf.select(cfg, key)
    return float(value) if value is not None else default


def _resolve_str(cfg: DictConfig, key: str, default: str) -> str:
    value = OmegaConf.select(cfg, key)
    return str(value) if value is not None else default


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id_cfg = OmegaConf.select(cfg, "run_id")
    run_id = (
        str(run_id_cfg)
        if run_id_cfg
        else f"tangram_plot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_paths = resolve_run_paths(cfg, run_id=run_id)

    input_h5ad = _resolve_tangram_spatial_h5ad(cfg)
    output_dir = _resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_sample = _resolve_sample_id(cfg)
    point_size = _resolve_float(cfg, "tangram_plot.point_size", 2.0)
    celltype_cmap = _resolve_str(cfg, "tangram_plot.celltype_cmap", "viridis")
    winner_cmap = _resolve_str(cfg, "tangram_plot.winner_cmap", "tab10")

    if not input_h5ad.exists():
        raise FileNotFoundError(f"Tangram spatial h5ad not found: {input_h5ad}")

    adata_tg = anndata.read_h5ad(input_h5ad, backed="r")
    try:
        sample_series = (
            adata_tg.obs["sample_id"].astype(str)
            if "sample_id" in adata_tg.obs.columns
            else pd.Series("all", index=adata_tg.obs_names)
        )
        sample_used = requested_sample or str(sample_series.iloc[0])

        celltype_map = output_dir / f"tangram_celltype_maps_{sample_used}.png"
        winner_map = output_dir / f"tangram_winner_map_{sample_used}.png"

        sample_used = plot_tangram_celltype_maps(
            adata_tangram=adata_tg,
            output_path=celltype_map,
            sample_id=sample_used,
            point_size=point_size,
            cmap=celltype_cmap,
        )
        sample_used_2, winner_counts = plot_tangram_winner_map(
            adata_tangram=adata_tg,
            output_path=winner_map,
            sample_id=sample_used,
            point_size=point_size,
            cmap=winner_cmap,
        )
        if sample_used_2 != sample_used:
            sample_used = sample_used_2

        if "X_tangram_ct" not in adata_tg.obsm:
            raise KeyError("Tangram spatial h5ad missing obsm['X_tangram_ct']")
        ct_raw = np.asarray(adata_tg.obsm["X_tangram_ct"], dtype=np.float32)
        row_sum = ct_raw.sum(axis=1)

        report = {
            "ok": True,
            "run_id": run_paths.run_id,
            "inputs": {
                "spatial_tangram_h5ad": str(input_h5ad),
                "sample_id_requested": requested_sample,
                "sample_id_used": sample_used,
            },
            "outputs": {
                "celltype_map_png": str(celltype_map),
                "winner_map_png": str(winner_map),
                "run_report": str(run_paths.tables_dir / "tangram_plot_report.json"),
                "config_resolved": str(run_paths.config_resolved_yaml),
            },
            "shape": {
                "n_spots": int(adata_tg.n_obs),
                "n_genes": int(adata_tg.n_vars),
                "n_celltype_cols": int(ct_raw.shape[1]),
            },
            "row_sum_stats": {
                "min": float(np.min(row_sum)),
                "median": float(np.median(row_sum)),
                "mean": float(np.mean(row_sum)),
                "max": float(np.max(row_sum)),
            },
            "winner_label_top10": [
                [str(k), int(v)] for k, v in winner_counts.head(10).items()
            ],
        }
    finally:
        if getattr(adata_tg, "isbacked", False) and getattr(adata_tg, "file", None) is not None:
            adata_tg.file.close()

    report_path = run_paths.tables_dir / "tangram_plot_report.json"
    write_json(report_path, report)
    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)

    print(f"celltype_map={report['outputs']['celltype_map_png']}")
    print(f"winner_map={report['outputs']['winner_map_png']}")
    print(f"sample_id={report['inputs']['sample_id_used']}")
    print(f"run_report={report_path}")
    print(json.dumps({
        "ok": True,
        "run_id": run_paths.run_id,
        "celltype_map_png": report["outputs"]["celltype_map_png"],
        "winner_map_png": report["outputs"]["winner_map_png"],
    }))


if __name__ == "__main__":
    main()
