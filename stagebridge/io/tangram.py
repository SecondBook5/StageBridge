"""Tangram mapping utilities for HLCA-labeled snRNA -> spatial projection."""
from __future__ import annotations

import gc
from dataclasses import dataclass
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

log = get_logger(__name__)


@dataclass(slots=True)
class TangramMappingResult:
    """Artifacts and summary metrics from Tangram mapping."""

    run_id: str
    mapping_h5ad_path: Path
    spatial_h5ad_path: Path
    scores_parquet_path: Path
    report_path: Path
    report: dict[str, Any]


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
