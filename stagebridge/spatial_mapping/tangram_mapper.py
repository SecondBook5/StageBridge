"""Tangram mapping utilities for HLCA-labeled snRNA -> spatial projection."""
from __future__ import annotations

import gc
from dataclasses import dataclass
import hashlib
import h5py
from importlib import metadata
import json
from pathlib import Path
import time
from typing import Any

import anndata
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from tqdm.auto import tqdm

from stagebridge.logging_utils import get_logger
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
from stagebridge.spatial_mapping.base import SpatialMappingResult
from stagebridge.spatial_mapping.qc import summarize_mapping_qc
from stagebridge.utils.h5ad_io import (
    decode_h5_array,
    read_h5ad_obs_column,
    read_h5ad_obs_column_or_default,
    read_h5ad_obs_frame,
    read_h5ad_var_index,
)

log = get_logger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _provider_version(package_name: str, fallback: str = "unknown") -> str:
    try:
        return metadata.version(package_name)
    except Exception:
        return fallback


def _mapping_cache_root(method: str) -> Path:
    root = _REPO_ROOT / "outputs" / "scratch" / "cache" / "spatial_mapping" / method
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stable_hash(payload: dict[str, Any]) -> str:
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): _normalize(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_normalize(item) for item in value]
        try:
            from omegaconf import DictConfig, ListConfig, OmegaConf

            if isinstance(value, (DictConfig, ListConfig)):
                return _normalize(OmegaConf.to_container(value, resolve=True))
        except Exception:
            pass
        return str(value)

    normalized = _normalize(payload)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _normalize_obs_fields(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    if "patient_id" not in out.columns and "donor_id" in out.columns:
        out["patient_id"] = out["donor_id"].astype(str)
    if "donor_id" not in out.columns and "patient_id" in out.columns:
        out["donor_id"] = out["patient_id"].astype(str)
    if "sample_id" not in out.columns:
        out["sample_id"] = out.index.astype(str)
    out["stage"] = out["stage"].astype(str)
    out["donor_id"] = out["donor_id"].astype(str)
    out["patient_id"] = out["patient_id"].astype(str)
    return out


def _select_stage_rows(
    obs: pd.DataFrame,
    *,
    stages: list[str] | None,
    donors: list[str] | None,
    max_rows_per_stage: int | None,
    seed: int,
) -> np.ndarray:
    mask = np.ones(obs.shape[0], dtype=bool)
    if stages:
        mask &= obs["stage"].astype(str).isin([str(stage) for stage in stages]).to_numpy()
    if donors:
        mask &= obs["donor_id"].astype(str).isin([str(donor) for donor in donors]).to_numpy()
    if max_rows_per_stage is not None and max_rows_per_stage > 0:
        rng = np.random.default_rng(int(seed))
        chosen = np.zeros(obs.shape[0], dtype=bool)
        masked_positions = np.flatnonzero(mask)
        masked_stages = obs.iloc[masked_positions]["stage"].to_numpy()
        for stage_name in pd.unique(masked_stages):
            rows = masked_positions[masked_stages == stage_name]
            if rows.shape[0] <= max_rows_per_stage:
                chosen[rows] = True
                continue
            keep = rng.choice(rows, size=int(max_rows_per_stage), replace=False)
            chosen[keep] = True
        mask &= chosen
    return np.flatnonzero(mask)


def _aligned_label_series_from_sources(
    *,
    obs: pd.DataFrame,
    obs_index: pd.Index,
    label_col: str,
    fallback_labels_parquet_path: Path | None = None,
    fallback_latent_h5ad_path: Path | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    if label_col in obs.columns:
        series = obs[label_col].astype(str).copy()
        series.index = obs_index.astype(str)
        return series, {"source": "snrna_obs", "path": None}

    if fallback_labels_parquet_path is not None and Path(fallback_labels_parquet_path).exists():
        labels_df = pd.read_parquet(fallback_labels_parquet_path)
        if label_col in labels_df.columns:
            aligned = pd.Series(index=obs_index.astype(str), dtype=object, name=label_col)
            labels_df.index = labels_df.index.astype(str)
            overlap = aligned.index.intersection(labels_df.index)
            aligned.loc[overlap] = labels_df.loc[overlap, label_col].astype(str).to_numpy()
            if aligned.notna().any():
                return aligned, {"source": "labels_parquet", "path": str(Path(fallback_labels_parquet_path))}

    if fallback_latent_h5ad_path is not None and Path(fallback_latent_h5ad_path).exists():
        latent = anndata.read_h5ad(fallback_latent_h5ad_path, backed="r")
        latent_obs = latent.obs.copy()
        latent_index = latent_obs.index.astype(str)
        if label_col not in latent_obs.columns and "cell_id" in latent_obs.columns:
            latent_index = latent_obs["cell_id"].astype(str)
        if label_col in latent_obs.columns:
            aligned = pd.Series(index=obs_index.astype(str), dtype=object, name=label_col)
            source = pd.Series(latent_obs[label_col].astype(str).to_numpy(), index=latent_index, name=label_col)
            overlap = aligned.index.intersection(source.index)
            aligned.loc[overlap] = source.loc[overlap].to_numpy()
            if aligned.notna().any():
                return aligned, {"source": "latent_h5ad", "path": str(Path(fallback_latent_h5ad_path))}

    raise KeyError(
        f"Missing '{label_col}' in raw snRNA obs and no usable fallback labels were found."
    )




def _read_h5ad_csr_rows(h5ad_path: Path, rows: np.ndarray, *, group_name: str = "X") -> sp.csr_matrix:
    with h5py.File(h5ad_path, "r") as handle:
        group = handle[group_name]
        if isinstance(group, h5py.Dataset):
            return sp.csr_matrix(np.asarray(group[rows, :], dtype=np.float32))

        shape = tuple(int(x) for x in group.attrs["shape"])
        indptr = np.asarray(group["indptr"], dtype=np.int64)
        row_starts = indptr[rows]
        row_ends = indptr[rows + 1]
        nnz = int(np.sum(row_ends - row_starts))
        data = np.empty(nnz, dtype=np.float32)
        indices = np.empty(nnz, dtype=np.int32)
        new_indptr = np.zeros(rows.shape[0] + 1, dtype=np.int64)
        cursor = 0
        data_ds = group["data"]
        indices_ds = group["indices"]
        for i, (start, end) in enumerate(zip(row_starts.tolist(), row_ends.tolist(), strict=False)):
            length = int(end - start)
            if length:
                data[cursor : cursor + length] = np.asarray(data_ds[start:end], dtype=np.float32)
                indices[cursor : cursor + length] = np.asarray(indices_ds[start:end], dtype=np.int32)
            cursor += length
            new_indptr[i + 1] = cursor
    return sp.csr_matrix((data, indices, new_indptr), shape=(rows.shape[0], shape[1]), dtype=np.float32)




def _write_label_parquet_from_snrna(
    *,
    snrna_h5ad_path: Path,
    labels_parquet_path: Path,
    label_col: str,
    stages: list[str] | None,
    max_cells_per_label: int,
    seed: int,
    fallback_labels_parquet_path: Path | None = None,
    fallback_latent_h5ad_path: Path | None = None,
) -> dict[str, Any]:
    obs = _normalize_obs_fields(
        read_h5ad_obs_frame(
            snrna_h5ad_path,
            columns=["donor_id", "patient_id", "sample_id", "stage"],
        )
    )
    rows = _select_stage_rows(
        obs,
        stages=stages,
        donors=None,
        max_rows_per_stage=None,
        seed=seed,
    )
    obs = obs.iloc[rows].copy()
    labels, label_meta = _aligned_label_series_from_sources(
        obs=obs,
        obs_index=obs.index,
        label_col=label_col,
        fallback_labels_parquet_path=fallback_labels_parquet_path,
        fallback_latent_h5ad_path=fallback_latent_h5ad_path,
    )
    obs[label_col] = labels
    obs = obs.loc[obs[label_col].notna()].copy()
    if max_cells_per_label > 0:
        rng = np.random.default_rng(int(seed))
        keep_rows: list[int] = []
        for _, frame in obs.groupby(label_col, sort=True):
            local = np.arange(frame.shape[0], dtype=np.int64)
            if frame.shape[0] <= max_cells_per_label:
                keep_rows.extend(frame.index.tolist())
                continue
            keep_idx = rng.choice(local, size=int(max_cells_per_label), replace=False)
            keep_rows.extend(frame.index.to_numpy()[keep_idx].tolist())
        obs = obs.loc[keep_rows].copy()
    labels_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({label_col: obs[label_col].astype(str)}, index=obs.index.astype(str)).to_parquet(
        labels_parquet_path,
        index=True,
        engine="pyarrow",
    )
    return {
        "label_col": label_col,
        "n_cells": int(obs.shape[0]),
        "n_labels": int(obs[label_col].nunique()),
        "label_source": label_meta,
    }


def _write_snrna_subset_h5ad_from_labels(
    *,
    snrna_h5ad_path: Path,
    labels_parquet_path: Path,
    subset_h5ad_path: Path,
    label_col: str,
) -> dict[str, Any]:
    labels_df = pd.read_parquet(labels_parquet_path)
    labels_df.index = labels_df.index.astype(str)
    all_obs = _normalize_obs_fields(
        read_h5ad_obs_frame(
            snrna_h5ad_path,
            columns=["donor_id", "patient_id", "sample_id", "stage"],
        )
    )
    row_lookup = pd.Series(np.arange(all_obs.shape[0], dtype=np.int64), index=all_obs.index.astype(str))
    matched_rows = row_lookup.reindex(labels_df.index).dropna()
    if matched_rows.empty:
        raise RuntimeError(
            f"No snRNA rows matched labels parquet {labels_parquet_path} for subset creation."
        )
    rows = matched_rows.to_numpy(dtype=np.int64)
    subset_obs = all_obs.loc[matched_rows.index].copy()
    if label_col in labels_df.columns:
        subset_obs[label_col] = labels_df.loc[matched_rows.index, label_col].astype(str).to_numpy()

    subset_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    subset = anndata.AnnData(
        X=_read_h5ad_csr_rows(snrna_h5ad_path, rows, group_name="X"),
        obs=subset_obs,
        var=pd.DataFrame(index=read_h5ad_var_index(snrna_h5ad_path)),
    )
    try:
        subset.layers["counts"] = _read_h5ad_csr_rows(snrna_h5ad_path, rows, group_name="layers/counts")
    except Exception:
        pass
    subset.write_h5ad(subset_h5ad_path, compression="lzf")
    return {
        "n_cells": int(subset.n_obs),
        "n_genes": int(subset.n_vars),
        "label_col": label_col,
    }


def _write_spatial_subset_h5ad(
    *,
    spatial_h5ad_path: Path,
    subset_h5ad_path: Path,
    stages: list[str] | None,
    donors: list[str] | None,
    max_spots_per_stage: int | None,
    seed: int,
) -> dict[str, Any]:
    subset_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(spatial_h5ad_path, "r") as handle:
        obs_group = handle["obs"]
        all_rows = np.arange(obs_group["_index"].shape[0], dtype=np.int64)
        obs_index = decode_h5_array(obs_group["_index"][:])
        donor_values = (
            read_h5ad_obs_column(obs_group, "donor_id", all_rows)
            if "donor_id" in obs_group
            else read_h5ad_obs_column(obs_group, "patient_id", all_rows)
        )
        patient_values = (
            read_h5ad_obs_column(obs_group, "patient_id", all_rows)
            if "patient_id" in obs_group
            else donor_values
        )
        obs = pd.DataFrame(
            {
                "spot_id": read_h5ad_obs_column_or_default(obs_group, "spot_id", all_rows, default=obs_index),
                "barcode": read_h5ad_obs_column_or_default(obs_group, "barcode", all_rows, default=obs_index),
                "donor_id": donor_values,
                "patient_id": patient_values,
                "stage": read_h5ad_obs_column(obs_group, "stage", all_rows),
                "sample_id": read_h5ad_obs_column_or_default(obs_group, "sample_id", all_rows, default=obs_index),
            },
            index=pd.Index(obs_index, name=str(obs_group.attrs.get("_index", "_index"))),
        )
        obs = _normalize_obs_fields(obs)
        rows = _select_stage_rows(
            obs,
            stages=stages,
            donors=donors,
            max_rows_per_stage=max_spots_per_stage,
            seed=seed,
        )
        subset_obs = obs.iloc[rows].copy()
        var_index = pd.Index(decode_h5_array(handle["var"]["_index"][:]), name="gene")
        spatial_coords = np.asarray(handle["obsm"]["spatial"][rows], dtype=np.float32)

    subset = anndata.AnnData(
        X=_read_h5ad_csr_rows(spatial_h5ad_path, rows, group_name="X"),
        obs=subset_obs,
        var=pd.DataFrame(index=var_index),
    )
    try:
        subset.layers["counts"] = _read_h5ad_csr_rows(spatial_h5ad_path, rows, group_name="layers/counts")
    except Exception:
        pass
    subset.obsm["spatial"] = spatial_coords
    subset.write_h5ad(subset_h5ad_path, compression="lzf")
    return {
        "n_spots": int(subset.n_obs),
        "n_genes": int(subset.n_vars),
        "stages": sorted(subset.obs["stage"].astype(str).unique().tolist()),
    }


def _tangram_cache_bundle(
    cfg: Any,
    *,
    stages: list[str] | None,
    donors: list[str] | None,
    max_spots_per_stage: int | None,
    seed: int,
) -> dict[str, Path]:
    paths = resolve_luad_evo_paths(cfg)
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    cache_key = _stable_hash(
        {
            "method": "tangram",
            "snrna_h5ad": str(paths.snrna_h5ad),
            "snrna_latent_h5ad": str(paths.snrna_latent_h5ad),
            "hlca_labels_parquet": str(paths.hlca_labels_parquet),
            "spatial_h5ad": str(paths.spatial_h5ad),
            "stages": stages,
            "donors": donors,
            "max_spots_per_stage": max_spots_per_stage,
            "seed": seed,
            "provider_cfg": provider_cfg,
        }
    )
    cache_dir = _mapping_cache_root("tangram") / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "labels_parquet": cache_dir / "snrna_labels.parquet",
        "snrna_subset_h5ad": cache_dir / "snrna_subset.h5ad",
        "spatial_subset_h5ad": cache_dir / "spatial_subset.h5ad",
        "mapping_h5ad": cache_dir / "mapping.h5ad",
        "spatial_h5ad": cache_dir / "spatial_projected.h5ad",
        "scores_parquet": cache_dir / "scores.parquet",
        "report_json": cache_dir / "report.json",
    }


def load_active_tangram_mapping(
    cfg: Any,
    *,
    mapping_h5ad_path: Path | None = None,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialMappingResult:
    """Load the active LUAD Tangram mapping output as the Mission 3 provider."""
    cohort = load_luad_evo_spatial_mapping(
        cfg,
        mapping_h5ad_path=mapping_h5ad_path,
        composition_key="X_tangram_ct",
        columns_key="tangram_ct_columns",
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    return SpatialMappingResult(
        method="tangram",
        status="complete",
        provider_version=_provider_version("tangram-sc", _provider_version("tangram")),
        execution_mode="load_precomputed",
        compositions=cohort.compositions,
        coords=cohort.coords,
        obs=cohort.obs,
        feature_names=cohort.feature_names,
        source_path=cohort.source_path,
        qc=summarize_mapping_qc(cohort.compositions),
        provenance={"mode": "loaded", "cache_dir": None},
        notes="Loaded precomputed Tangram spot-level composition outputs for luad_evo.",
    )


@dataclass(slots=True)
class TangramMappingResult:
    """Artifacts and summary metrics from Tangram mapping."""

    run_id: str
    mapping_h5ad_path: Path
    spatial_h5ad_path: Path
    scores_parquet_path: Path
    report_path: Path
    report: dict[str, Any]


def run_tangram(
    cfg: Any,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialMappingResult:
    """Run or load Tangram through the active provider contract."""
    provider_cfg = dict(cfg.get("spatial_mapping", {})) if hasattr(cfg, "get") else dict(cfg["spatial_mapping"])
    execution_mode = str(provider_cfg.get("execution_mode", "load_precomputed"))
    provider_version = _provider_version("tangram-sc", _provider_version("tangram"))

    if execution_mode == "load_precomputed":
        return load_active_tangram_mapping(
            cfg,
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )
    if execution_mode not in {"rebuild_cached", "force_rebuild"}:
        raise ValueError(
            f"Unsupported Tangram execution_mode '{execution_mode}'. "
            "Use 'load_precomputed', 'rebuild_cached', or 'force_rebuild'."
        )

    paths = resolve_luad_evo_paths(cfg)
    cache = _tangram_cache_bundle(
        cfg,
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    needs_rebuild = execution_mode == "force_rebuild" or not cache["spatial_h5ad"].exists()
    label_meta: dict[str, Any] | None = None
    snrna_meta: dict[str, Any] | None = None
    spatial_meta: dict[str, Any] | None = None
    report_path = cache["report_json"]
    if needs_rebuild:
        label_meta = _write_label_parquet_from_snrna(
            snrna_h5ad_path=paths.snrna_h5ad,
            labels_parquet_path=cache["labels_parquet"],
            label_col=str(provider_cfg.get("label_col", "hlca_label")),
            stages=stages,
            max_cells_per_label=int(provider_cfg.get("max_reference_cells_per_label", 4000)),
            seed=seed,
            fallback_labels_parquet_path=paths.hlca_labels_parquet,
            fallback_latent_h5ad_path=paths.snrna_latent_h5ad,
        )
        snrna_meta = _write_snrna_subset_h5ad_from_labels(
            snrna_h5ad_path=paths.snrna_h5ad,
            labels_parquet_path=cache["labels_parquet"],
            subset_h5ad_path=cache["snrna_subset_h5ad"],
            label_col=str(provider_cfg.get("label_col", "hlca_label")),
        )
        spatial_meta = _write_spatial_subset_h5ad(
            spatial_h5ad_path=paths.spatial_h5ad,
            subset_h5ad_path=cache["spatial_subset_h5ad"],
            stages=stages,
            donors=donors,
            max_spots_per_stage=max_spots_per_stage,
            seed=seed,
        )
        tangram_result = run_tangram_hlca_projection(
            run_id=cache["cache_dir"].name,
            snrna_h5ad_path=cache["snrna_subset_h5ad"],
            spatial_h5ad_path=cache["spatial_subset_h5ad"],
            labels_parquet_path=cache["labels_parquet"],
            output_mapping_h5ad_path=cache["mapping_h5ad"],
            output_spatial_h5ad_path=cache["spatial_h5ad"],
            output_scores_parquet_path=cache["scores_parquet"],
            report_path=cache["report_json"],
            tangram_cfg=provider_cfg,
        )
        report_path = tangram_result.report_path

    result = load_active_tangram_mapping(
        cfg,
        mapping_h5ad_path=cache["spatial_h5ad"],
        stages=stages,
        donors=donors,
        max_spots_per_stage=max_spots_per_stage,
        seed=seed,
    )
    return SpatialMappingResult(
        method="tangram",
        status=result.status,
        provider_version=provider_version,
        execution_mode=execution_mode,
        compositions=result.compositions,
        coords=result.coords,
        obs=result.obs,
        feature_names=result.feature_names,
        source_path=result.source_path,
        qc=result.qc,
        provenance={
            "mode": "rebuilt" if needs_rebuild else "cached",
            "cache_dir": str(cache["cache_dir"]),
            "report_path": str(report_path),
            "labels_path": str(cache["labels_parquet"]),
            "label_metadata": label_meta,
            "snrna_subset_metadata": snrna_meta,
            "spatial_subset_metadata": spatial_meta,
        },
        notes=(
            "Tangram mapping rebuilt from raw snRNA and spatial assets."
            if needs_rebuild
            else "Tangram mapping loaded from the reusable rebuild cache."
        ),
    )


def _now_rss_mb(process: psutil.Process) -> float:
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _coerce_csr_float32(x: Any) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32, copy=False)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _is_cuda_driver_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda driver error" in msg or ("cuda" in msg and "unknown error" in msg)


def _clear_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _parse_labels(
    obs_names: pd.Index,
    labels_parquet_path: Path,
    label_col: str,
) -> tuple[pd.Series, float]:
    labels_df = pd.read_parquet(labels_parquet_path)
    labels_df.index = labels_df.index.astype(str)
    if label_col not in labels_df.columns:
        raise KeyError(
            f"Missing label column '{label_col}' in {labels_parquet_path}. "
            f"Columns available: {list(labels_df.columns)}"
        )

    labels = pd.Series(index=obs_names.astype(str), dtype=object, name=label_col)
    overlap = labels.index.intersection(labels_df.index)
    labels.loc[overlap] = labels_df.loc[overlap, label_col].astype(str).to_numpy()
    coverage = float(labels.notna().mean())
    return labels, coverage


def _sorted_shared_genes(sc_var: pd.Index, sp_var: pd.Index) -> list[str]:
    shared = sorted(set(sc_var.astype(str)) & set(sp_var.astype(str)))
    return shared


def _subset_training_genes(shared_genes: list[str], max_training_genes: int) -> list[str]:
    if max_training_genes > 0 and len(shared_genes) > max_training_genes:
        return shared_genes[:max_training_genes]
    return shared_genes


def _aggregate_cluster_profiles_backed(
    source_matrix: Any,
    label_codes: np.ndarray,
    n_labels: int,
    gene_idx: np.ndarray,
    chunk_rows: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate backed matrix rows into per-label mean expression vectors."""
    n_obs = int(label_codes.shape[0])
    n_genes = int(gene_idx.shape[0])

    sums = np.zeros((n_labels, n_genes), dtype=np.float64)
    counts = np.zeros((n_labels,), dtype=np.int64)

    starts = range(0, n_obs, chunk_rows)
    if show_progress:
        starts = tqdm(
            starts,
            total=(n_obs + chunk_rows - 1) // chunk_rows,
            desc="Tangram: aggregate HLCA profiles",
            unit="chunk",
        )
    for start in starts:
        end = min(start + chunk_rows, n_obs)
        codes = label_codes[start:end]
        valid_mask = codes >= 0
        if not np.any(valid_mask):
            continue

        chunk = _coerce_csr_float32(source_matrix[start:end, gene_idx])
        valid_codes = codes.copy()
        valid_codes[~valid_mask] = -1
        unique_codes = np.unique(valid_codes[valid_codes >= 0])
        for code in unique_codes.tolist():
            row_idx = np.where(codes == code)[0]
            if row_idx.size == 0:
                continue
            vec = np.asarray(chunk[row_idx].sum(axis=0)).ravel()
            sums[code, :] += vec
            counts[code] += int(row_idx.size)

    denom = np.maximum(counts[:, None], 1)
    means = (sums / denom).astype(np.float32, copy=False)
    return means, counts


def _build_cluster_adata(
    *,
    source_matrix: Any,
    source_var_names: pd.Index,
    labels: pd.Series,
    shared_genes: list[str],
    chunk_rows: int,
    show_progress: bool,
) -> tuple[anndata.AnnData, dict[str, int]]:
    label_values = labels.to_numpy(dtype=object)
    valid_mask = pd.notna(label_values)
    valid_labels = label_values[valid_mask].astype(str, copy=False)
    label_cat = pd.Categorical(valid_labels)
    label_names = label_cat.categories.astype(str)
    label_codes = np.full(label_values.shape[0], -1, dtype=np.int32)
    label_codes[valid_mask] = label_cat.codes.astype(np.int32, copy=False)

    var_to_pos = pd.Series(np.arange(len(source_var_names), dtype=np.int64), index=source_var_names.astype(str))
    gene_idx = var_to_pos.reindex(shared_genes).to_numpy(dtype=np.int64)
    if np.any(gene_idx < 0):
        raise RuntimeError("Internal error while building gene index for shared genes.")

    means, counts = _aggregate_cluster_profiles_backed(
        source_matrix=source_matrix,
        label_codes=label_codes,
        n_labels=len(label_names),
        gene_idx=gene_idx,
        chunk_rows=chunk_rows,
        show_progress=show_progress,
    )
    keep = counts > 0
    means = means[keep, :]
    kept_labels = label_names[keep]
    kept_counts = counts[keep]

    obs = pd.DataFrame(index=pd.Index(kept_labels, name="cluster_id"))
    obs["hlca_label"] = kept_labels
    obs["n_cells_source"] = kept_counts.astype(np.int64)
    var = pd.DataFrame(index=pd.Index(shared_genes, name="gene"))
    adata_cluster = anndata.AnnData(X=means, obs=obs, var=var)

    label_sizes = {str(k): int(v) for k, v in zip(kept_labels.tolist(), kept_counts.tolist())}
    return adata_cluster, label_sizes


def run_tangram_hlca_projection(
    *,
    run_id: str,
    snrna_h5ad_path: Path,
    spatial_h5ad_path: Path,
    labels_parquet_path: Path,
    output_mapping_h5ad_path: Path,
    output_spatial_h5ad_path: Path,
    output_scores_parquet_path: Path,
    report_path: Path,
    tangram_cfg: dict[str, Any] | None = None,
) -> TangramMappingResult:
    """Run Tangram projection using HLCA labels as cell-type annotations."""
    import tangram as tg

    tangram_cfg = dict(tangram_cfg or {})
    wall_t0 = time.perf_counter()
    stage_times: dict[str, float] = {}
    process = psutil.Process()
    peak_rss_mb = _now_rss_mb(process)

    def mark_peak() -> None:
        nonlocal peak_rss_mb
        peak_rss_mb = max(peak_rss_mb, _now_rss_mb(process))

    def stage_start() -> float:
        return time.perf_counter()

    def stage_done(name: str, t0: float) -> None:
        stage_times[name] = time.perf_counter() - t0

    label_col = str(tangram_cfg.get("label_col", "hlca_label"))
    use_counts_layer = bool(tangram_cfg.get("use_counts_layer", True))
    min_label_coverage = float(tangram_cfg.get("min_label_coverage", 0.9))
    min_cells_per_label = int(tangram_cfg.get("min_cells_per_label", 20))
    max_training_genes = int(tangram_cfg.get("max_training_genes", 3000))
    min_shared_genes = int(tangram_cfg.get("min_shared_genes", 300))
    aggregate_profiles = bool(tangram_cfg.get("aggregate_profiles", True))
    chunk_rows = int(tangram_cfg.get("aggregate_chunk_rows", 20000))
    show_progress = bool(tangram_cfg.get("show_progress", True))

    mode_requested = str(tangram_cfg.get("mode", "clusters"))
    mode = "cells" if aggregate_profiles else mode_requested
    if mode not in {"cells", "clusters", "constrained"}:
        raise ValueError(f"Unsupported tangram mode: {mode!r}")
    if mode == "constrained":
        raise ValueError("mode='constrained' is not supported in this StageBridge wrapper.")

    device = _choose_device(str(tangram_cfg.get("device", "auto")))
    learning_rate = float(tangram_cfg.get("learning_rate", 0.1))
    num_epochs = int(tangram_cfg.get("num_epochs", 500))
    density_prior = tangram_cfg.get("density_prior", "rna_count_based")
    random_seed = int(tangram_cfg.get("random_seed", 0))
    scale = bool(tangram_cfg.get("scale", True))
    lambda_d = float(tangram_cfg.get("lambda_d", 0.0))
    lambda_g1 = float(tangram_cfg.get("lambda_g1", 1.0))
    lambda_g2 = float(tangram_cfg.get("lambda_g2", 0.0))
    lambda_r = float(tangram_cfg.get("lambda_r", 0.0))
    verbose = bool(tangram_cfg.get("verbose", True))
    cuda_retry_on_failure = bool(tangram_cfg.get("cuda_retry_on_failure", True))
    cuda_max_retries = int(tangram_cfg.get("cuda_max_retries", 1))
    cuda_retry_sleep_seconds = float(tangram_cfg.get("cuda_retry_sleep_seconds", 2.0))
    cuda_clear_cache_before_map = bool(tangram_cfg.get("cuda_clear_cache_before_map", True))

    output_mapping_h5ad_path = Path(output_mapping_h5ad_path)
    output_spatial_h5ad_path = Path(output_spatial_h5ad_path)
    output_scores_parquet_path = Path(output_scores_parquet_path)
    report_path = Path(report_path)
    for p in (
        output_mapping_h5ad_path,
        output_spatial_h5ad_path,
        output_scores_parquet_path,
        report_path,
    ):
        p.parent.mkdir(parents=True, exist_ok=True)

    t0 = stage_start()
    adata_sc_backed = anndata.read_h5ad(snrna_h5ad_path, backed="r")
    adata_sp_backed = anndata.read_h5ad(spatial_h5ad_path, backed="r")
    adata_sp_var = pd.Index(adata_sp_backed.var_names.astype(str))
    adata_sc_var = pd.Index(adata_sc_backed.var_names.astype(str))

    source_matrix = adata_sc_backed.layers["counts"] if (use_counts_layer and "counts" in adata_sc_backed.layers) else adata_sc_backed.X
    labels, label_coverage = _parse_labels(
        obs_names=adata_sc_backed.obs_names,
        labels_parquet_path=labels_parquet_path,
        label_col=label_col,
    )
    if label_coverage < min_label_coverage:
        raise RuntimeError(
            f"Label coverage {label_coverage:.3f} < required {min_label_coverage:.3f}. "
            f"Check {labels_parquet_path} against {snrna_h5ad_path}."
        )

    label_counts_all = labels.dropna().astype(str).value_counts()
    keep_labels = label_counts_all[label_counts_all >= min_cells_per_label].index.astype(str)
    keep_mask = labels.astype(str).isin(set(keep_labels.tolist())).to_numpy()
    kept_labels = labels.copy()
    kept_labels.loc[~keep_mask] = np.nan

    shared_genes = _sorted_shared_genes(
        sc_var=adata_sc_var,
        sp_var=adata_sp_var,
    )
    if len(shared_genes) < min_shared_genes:
        raise RuntimeError(
            f"Only {len(shared_genes)} shared genes between snRNA and spatial, "
            f"below min_shared_genes={min_shared_genes}."
        )
    training_genes = _subset_training_genes(shared_genes, max_training_genes=max_training_genes)
    stage_done("prepare_inputs", t0)
    mark_peak()

    t0 = stage_start()
    sp_source = (
        adata_sp_backed.layers["counts"]
        if (use_counts_layer and "counts" in adata_sp_backed.layers)
        else adata_sp_backed.X
    )
    sp_var_to_pos = pd.Series(
        np.arange(len(adata_sp_var), dtype=np.int64),
        index=adata_sp_var,
    )
    sp_gene_idx = sp_var_to_pos.reindex(training_genes).to_numpy(dtype=np.int64)
    if np.any(sp_gene_idx < 0):
        raise RuntimeError("Internal error while indexing spatial genes for Tangram.")

    X_sp = _coerce_csr_float32(sp_source[:, sp_gene_idx])
    obs_sp = adata_sp_backed.obs.copy()
    var_sp = pd.DataFrame(index=pd.Index(training_genes, name="gene"))
    adata_sp = anndata.AnnData(X=X_sp, obs=obs_sp, var=var_sp)

    if "spatial" in adata_sp_backed.obsm:
        adata_sp.obsm["spatial"] = np.asarray(adata_sp_backed.obsm["spatial"])
    else:
        raise KeyError(f"Spatial AnnData is missing required obsm['spatial']: {spatial_h5ad_path}")

    if hasattr(adata_sp_backed, "isbacked") and adata_sp_backed.isbacked:
        adata_sp_backed.file.close()
    stage_done("load_spatial", t0)
    mark_peak()

    t0 = stage_start()
    if aggregate_profiles:
        adata_sc_use, label_sizes = _build_cluster_adata(
            source_matrix=source_matrix,
            source_var_names=adata_sc_backed.var_names,
            labels=kept_labels,
            shared_genes=training_genes,
            chunk_rows=chunk_rows,
            show_progress=show_progress,
        )
        adata_sc_use.obs[label_col] = adata_sc_use.obs["hlca_label"].astype(str)
    else:
        adata_sc = anndata.read_h5ad(snrna_h5ad_path)
        if use_counts_layer and "counts" in adata_sc.layers:
            adata_sc.X = adata_sc.layers["counts"]
        adata_sc.var_names_make_unique()
        adata_sc.obs[label_col] = kept_labels.reindex(adata_sc.obs_names).astype(str)
        adata_sc = adata_sc[adata_sc.obs[label_col].notna(), training_genes].copy()
        label_sizes = {
            str(k): int(v)
            for k, v in adata_sc.obs[label_col].astype(str).value_counts().sort_values(ascending=False).items()
        }
        adata_sc_use = adata_sc
    stage_done("build_sc_reference", t0)
    mark_peak()

    if hasattr(adata_sc_backed, "isbacked") and adata_sc_backed.isbacked:
        adata_sc_backed.file.close()

    t0 = stage_start()
    tg.pp_adatas(adata_sc_use, adata_sp, genes=training_genes)
    stage_done("tangram_pp", t0)
    mark_peak()

    t0 = stage_start()
    cluster_label = label_col if mode == "clusters" else None
    map_attempts = 0
    while True:
        map_attempts += 1
        try:
            if device.startswith("cuda") and cuda_clear_cache_before_map:
                _clear_cuda_cache()
                gc.collect()
            adata_map = tg.map_cells_to_space(
                adata_sc=adata_sc_use,
                adata_sp=adata_sp,
                cluster_label=cluster_label,
                mode=mode,
                device=device,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                scale=scale,
                lambda_d=lambda_d,
                lambda_g1=lambda_g1,
                lambda_g2=lambda_g2,
                lambda_r=lambda_r,
                random_state=random_seed,
                verbose=verbose,
                density_prior=density_prior,
            )
            break
        except RuntimeError as exc:
            can_retry = (
                device.startswith("cuda")
                and cuda_retry_on_failure
                and map_attempts <= cuda_max_retries
                and (_is_cuda_driver_error(exc) or _is_cuda_oom(exc))
            )
            if not can_retry:
                raise
            log.warning(
                "Tangram CUDA map failure on attempt %d/%d (%s). "
                "Clearing CUDA cache and retrying in %.1fs.",
                map_attempts,
                cuda_max_retries + 1,
                str(exc).splitlines()[0],
                cuda_retry_sleep_seconds,
            )
            _clear_cuda_cache()
            gc.collect()
            time.sleep(cuda_retry_sleep_seconds)
    stage_done("tangram_map", t0)
    mark_peak()

    t0 = stage_start()
    if label_col not in adata_map.obs.columns:
        if aggregate_profiles and "hlca_label" in adata_map.obs.columns:
            adata_map.obs[label_col] = adata_map.obs["hlca_label"].astype(str)
        else:
            raise KeyError(f"Mapped AnnData is missing annotation column '{label_col}'.")
    tg.project_cell_annotations(adata_map, adata_sp, annotation=label_col)
    ct_pred = adata_sp.obsm["tangram_ct_pred"]
    if not isinstance(ct_pred, pd.DataFrame):
        ct_pred = pd.DataFrame(
            np.asarray(ct_pred),
            index=adata_sp.obs_names.astype(str),
        )
    ct_pred.index = adata_sp.obs_names.astype(str)
    stage_done("project_annotations", t0)
    mark_peak()

    t0 = stage_start()
    adata_map.write_h5ad(output_mapping_h5ad_path, compression="lzf")
    ct_columns = [str(c) for c in ct_pred.columns.tolist()]
    adata_sp.obsm["X_tangram_ct"] = ct_pred.to_numpy(dtype=np.float32, copy=False)
    adata_sp.uns["tangram_ct_columns"] = ct_columns
    adata_sp.uns["tangram_label_col"] = label_col
    adata_sp.write_h5ad(output_spatial_h5ad_path, compression="lzf")
    ct_pred.to_parquet(output_scores_parquet_path, index=True, engine="pyarrow")
    stage_done("write_outputs", t0)

    top_labels = sorted(label_sizes.items(), key=lambda kv: kv[1], reverse=True)[:10]
    report = {
        "ok": True,
        "run_id": run_id,
        "inputs": {
            "snrna_h5ad": str(snrna_h5ad_path),
            "spatial_h5ad": str(spatial_h5ad_path),
            "labels_parquet": str(labels_parquet_path),
            "label_col": label_col,
            "label_coverage": label_coverage,
        },
        "config_effective": {
            "mode_requested": mode_requested,
            "mode_used": mode,
            "aggregate_profiles": aggregate_profiles,
            "device": device,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "density_prior": density_prior,
            "max_training_genes": max_training_genes,
            "min_shared_genes": min_shared_genes,
            "use_counts_layer": use_counts_layer,
            "cuda_retry_on_failure": cuda_retry_on_failure,
            "cuda_max_retries": cuda_max_retries,
            "cuda_retry_sleep_seconds": cuda_retry_sleep_seconds,
            "cuda_clear_cache_before_map": cuda_clear_cache_before_map,
        },
        "counts": {
            "n_spots": int(adata_sp.n_obs),
            "n_shared_genes_total": int(len(shared_genes)),
            "n_training_genes": int(len(training_genes)),
            "n_label_profiles_used": int(len(label_sizes)),
            "map_attempts": int(map_attempts),
            "label_sizes_top10": [(str(k), int(v)) for k, v in top_labels],
        },
        "outputs": {
            "mapping_h5ad": str(output_mapping_h5ad_path),
            "spatial_h5ad": str(output_spatial_h5ad_path),
            "scores_parquet": str(output_scores_parquet_path),
        },
        "timing_seconds": stage_times,
        "total_wall_time_seconds": time.perf_counter() - wall_t0,
        "peak_rss_mb": peak_rss_mb,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return TangramMappingResult(
        run_id=run_id,
        mapping_h5ad_path=output_mapping_h5ad_path,
        spatial_h5ad_path=output_spatial_h5ad_path,
        scores_parquet_path=output_scores_parquet_path,
        report_path=report_path,
        report=report,
    )
