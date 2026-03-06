"""Notebook-facing pure workflow functions for StageBridge pipeline steps."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from stagebridge.io.hlca import map_full_snrna_with_hlca
from stagebridge.io.interim_build import build_snrna_interim_anndata, build_spatial_interim_anndata
from stagebridge.io.manifests import write_resolved_config_yaml
from stagebridge.io.paths import resolve_run_paths
from stagebridge.io.tangram import run_tangram_hlca_projection
from stagebridge.logging_utils import get_logger
from stagebridge.runs import default_run_id

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


def _resolve_spatial_extracted_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.raw.geo_spatial_dir", "data.spatial_raw_dir"])
    if value is not None:
        base = Path(str(value))
        if base.name == "extracted":
            return base
        extracted = base / "extracted"
        return extracted if extracted.exists() else base

    data_root = _resolve_data_root(cfg)
    candidates = [
        data_root / "raw" / "geo" / "GSE307534_spatial" / "extracted",
        data_root / "raw" / "geo" / "GSE307534_spatial",
        data_root / "data" / "raw" / "geo" / "GSE307534_spatial" / "extracted",
        data_root / "data" / "raw" / "geo" / "GSE307534_spatial",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.name == "extracted":
                return candidate
            extracted = candidate / "extracted"
            if extracted.exists():
                return extracted
            return candidate
    return candidates[0]


def _resolve_spatial_interim_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.interim.anndata_spatial_dir", "data.spatial_interim_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "interim" / "anndata" / "spatial"


def _resolve_input_snrna_full(cfg: DictConfig) -> Path:
    value = _cfg_pick(
        cfg,
        [
            "data.input_snrna_full_h5ad",
            "data.interim.anndata_snrna_dir",
        ],
    )
    if value is None:
        return _resolve_data_root(cfg) / "interim" / "anndata" / "snrna" / "snrna_full.h5ad"
    path = Path(str(value))
    if path.suffix == ".h5ad":
        return path
    return path / "snrna_full.h5ad"


def _resolve_processed_anndata_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_anndata_dir", "data.processed.anndata_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "processed" / "anndata"


def _resolve_processed_hlca_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_hlca_dir", "data.processed.hlca_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "processed" / "hlca"


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


def _resolve_processed_tangram_dir(cfg: DictConfig) -> Path:
    value = _cfg_pick(cfg, ["data.output_processed_tangram_dir", "data.processed.tangram_dir"])
    if value is not None:
        return Path(str(value))
    return _resolve_data_root(cfg) / "processed" / "tangram"


def build_snrna(cfg: DictConfig, *, run_id: str | None = None) -> dict[str, Any]:
    """Build interim snRNA AnnData artifact from raw GEO files."""
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_name = str(OmegaConf.select(cfg, "experiment.name") or "").lower()
    is_smoke = experiment_name == "smoke"

    max_donors = _int_or_none(_cfg_pick(cfg, ["pipeline.max_donors", "io.max_donors"]))
    max_samples_per_stage = _int_or_none(_cfg_pick(cfg, ["pipeline.max_samples_per_stage", "io.max_samples_per_stage"]))
    max_cells_per_sample = _int_or_none(_cfg_pick(cfg, ["pipeline.max_cells_per_sample", "io.max_cells_per_sample"]))

    if is_smoke:
        if max_donors is None:
            max_donors = int(OmegaConf.select(cfg, "experiment.max_donors") or 2)
        if max_samples_per_stage is None:
            max_samples_per_stage = int(OmegaConf.select(cfg, "experiment.max_samples_per_stage") or 1)
        if max_cells_per_sample is None:
            max_cells_per_sample = int(OmegaConf.select(cfg, "experiment.max_cells_per_sample") or 20000)

    raw_dir = _resolve_snrna_raw_dir(cfg)
    interim_dir = _resolve_snrna_interim_dir(cfg)
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_name = "snrna_smoke.h5ad" if is_smoke else "snrna_full.h5ad"
    output_path = interim_dir / output_name

    requested_run_id = run_id or OmegaConf.select(cfg, "run_id")
    resolved_run_id = str(requested_run_id) if requested_run_id else default_run_id("snrna")
    run_paths = resolve_run_paths(cfg, run_id=resolved_run_id)

    result = build_snrna_interim_anndata(
        run_paths=run_paths,
        raw_dir=raw_dir,
        output_path=output_path,
        is_smoke=is_smoke,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
        max_cells_per_sample=max_cells_per_sample,
        max_workers=_resolve_max_workers(cfg),
        scratch_root=_resolve_scratch_root(cfg),
        keep_scratch=_bool_or_default(_cfg_pick(cfg, ["pipeline.keep_scratch", "io.keep_scratch"]), False),
        compression=_resolve_write_compression(cfg),
        timestamp=timestamp,
        repo_root=Path(__file__).resolve().parent.parent.parent,
    )

    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)
    summary = result.summary
    payload = {
        "ok": True,
        "step": "build_snrna",
        "run_id": run_paths.run_id,
        "output_path": str(result.output_path),
        "n_obs": int(summary["n_obs"]),
        "n_vars": int(summary["n_vars"]),
        "donors": summary["donors"],
        "stages": summary["stages"],
        "run_manifest": str(result.run_manifest_path),
        "data_audit": str(result.data_audit_path),
    }
    log.info("%s", json.dumps(payload))
    return payload


def build_spatial(cfg: DictConfig, *, run_id: str | None = None) -> dict[str, Any]:
    """Build interim spatial AnnData artifact from GEO tarballs."""
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_name = str(OmegaConf.select(cfg, "experiment.name") or "").lower()
    is_smoke = experiment_name == "smoke"

    max_donors = _int_or_none(_cfg_pick(cfg, ["pipeline.max_donors", "io.max_donors"]))
    max_samples_per_stage = _int_or_none(_cfg_pick(cfg, ["pipeline.max_samples_per_stage", "io.max_samples_per_stage"]))
    max_spots_per_sample = _int_or_none(_cfg_pick(cfg, ["pipeline.max_spots_per_sample", "io.max_spots_per_sample"]))

    if is_smoke:
        if max_donors is None:
            max_donors = int(OmegaConf.select(cfg, "experiment.max_donors") or 2)
        if max_samples_per_stage is None:
            max_samples_per_stage = int(OmegaConf.select(cfg, "experiment.max_samples_per_stage") or 1)
        if max_spots_per_sample is None:
            max_spots_per_sample = int(OmegaConf.select(cfg, "experiment.max_spots_per_sample") or 20000)

    extracted_dir = _resolve_spatial_extracted_dir(cfg)
    interim_dir = _resolve_spatial_interim_dir(cfg)
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_name = "spatial_smoke.h5ad" if is_smoke else "spatial_full.h5ad"
    output_path = interim_dir / output_name

    requested_run_id = run_id or OmegaConf.select(cfg, "run_id")
    resolved_run_id = str(requested_run_id) if requested_run_id else default_run_id("spatial")
    run_paths = resolve_run_paths(cfg, run_id=resolved_run_id)

    result = build_spatial_interim_anndata(
        run_paths=run_paths,
        extracted_dir=extracted_dir,
        output_path=output_path,
        is_smoke=is_smoke,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
        max_spots_per_sample=max_spots_per_sample,
        max_workers=_resolve_max_workers(cfg),
        scratch_root=_resolve_scratch_root(cfg),
        keep_scratch=_bool_or_default(_cfg_pick(cfg, ["pipeline.keep_scratch", "io.keep_scratch"]), False),
        compression=_resolve_write_compression(cfg),
        timestamp=timestamp,
        repo_root=Path(__file__).resolve().parent.parent.parent,
    )

    write_resolved_config_yaml(run_paths.config_resolved_yaml, cfg)
    summary = result.summary
    payload = {
        "ok": True,
        "step": "build_spatial",
        "run_id": run_paths.run_id,
        "output_path": str(result.output_path),
        "n_obs": int(summary["n_obs"]),
        "n_vars": int(summary["n_vars"]),
        "donors": summary["donors"],
        "stages": summary["stages"],
        "spatial_coord_min": result.extra.get("spatial_coord_min"),
        "spatial_coord_max": result.extra.get("spatial_coord_max"),
        "run_manifest": str(result.run_manifest_path),
        "data_audit": str(result.data_audit_path),
    }
    log.info("%s", json.dumps(payload))
    return payload


def map_hlca(cfg: DictConfig, *, run_id: str | None = None) -> dict[str, Any]:
    """Map full snRNA data to HLCA latent space."""
    requested_run_id = run_id or OmegaConf.select(cfg, "run_id")
    resolved_run_id = str(requested_run_id) if requested_run_id else default_run_id("hlca")
    run_paths = resolve_run_paths(cfg, run_id=resolved_run_id)

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

    payload = {
        "ok": True,
        "step": "map_hlca",
        "run_id": result.run_id,
        "latent_h5ad": str(result.latent_h5ad_path),
        "labels_parquet": str(result.labels_parquet_path),
        "overlap_percent": float(result.overlap_percent),
        "latent_shape": list(result.latent_shape),
        "peak_rss_mb": float(result.peak_rss_mb),
        "total_wall_time_seconds": float(result.wall_time_seconds),
        "top10_hlca_label_counts": result.top10_labels,
    }
    log.info("%s", json.dumps(payload))
    return payload


def run_tangram(cfg: DictConfig, *, run_id: str | None = None) -> dict[str, Any]:
    """Run Tangram projection from HLCA-labeled snRNA to spatial spots."""
    requested_run_id = run_id or OmegaConf.select(cfg, "run_id")
    resolved_run_id = str(requested_run_id) if requested_run_id else default_run_id("tangram")
    run_paths = resolve_run_paths(cfg, run_id=resolved_run_id)

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
        tangram_cfg_dict = OmegaConf.to_container(tangram_cfg, resolve=True)
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

    counts = result.report.get("counts", {})
    payload = {
        "ok": True,
        "step": "run_tangram",
        "run_id": result.run_id,
        "mapping_h5ad": str(result.mapping_h5ad_path),
        "spatial_h5ad": str(result.spatial_h5ad_path),
        "scores_parquet": str(result.scores_parquet_path),
        "n_spots": counts.get("n_spots"),
        "n_training_genes": counts.get("n_training_genes"),
        "n_label_profiles_used": counts.get("n_label_profiles_used"),
        "peak_rss_mb": result.report.get("peak_rss_mb"),
    }
    log.info("%s", json.dumps(payload))
    return payload
