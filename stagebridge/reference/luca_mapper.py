"""LuCA reference loading and latent-space alignment helpers.

This module provides canonical LuCA (Lung Cancer Atlas) mapping functionality,
mirroring the structure of hlca_mapper.py for consistency.

LuCA Core Atlas:
- 790K cells, 10 latent dimensions
- scANVI model with cell type labels
- Latent key: X_scVI (or X_scANVI from model)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import anndata
import numpy as np
import pandas as pd
import psutil

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class LuCAMappingResult:
    """Artifacts and summary metrics from full-scale LuCA mapping."""

    run_id: str
    latent_h5ad_path: Path
    labels_parquet_path: Path
    mapping_report_path: Path
    gene_report_path: Path
    overlap_percent: float
    latent_shape: tuple[int, int]
    top10_labels: list[tuple[str, int]]
    peak_rss_mb: float
    wall_time_seconds: float
    report: dict[str, Any]


def _now_rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024 * 1024)


def map_full_snrna_with_luca(
    *,
    run_id: str,
    snrna_h5ad_path: Path,
    luca_ref_h5ad_path: Path,
    output_latent_h5ad_path: Path,
    output_labels_parquet_path: Path,
    mapping_report_path: Path,
    gene_report_path: Path,
    progress_path: Path,
    luca_model_dir: Path,
    luca_cfg: dict[str, Any],
) -> LuCAMappingResult:
    """Map full snRNA AnnData into LuCA latent and label space at scale.

    This mirrors hlca_mapper.map_full_snrna_with_hlca() for consistency.

    Parameters
    ----------
    run_id : str
        Unique identifier for this run
    snrna_h5ad_path : Path
        Path to input snRNA h5ad file
    luca_ref_h5ad_path : Path
        Path to LuCA reference h5ad (required for model loading)
    output_latent_h5ad_path : Path
        Path to save latent h5ad
    output_labels_parquet_path : Path
        Path to save labels parquet
    mapping_report_path : Path
        Path to save mapping report JSON
    gene_report_path : Path
        Path to save gene overlap report JSON
    progress_path : Path
        Path to save progress JSON
    luca_model_dir : Path
        Path to LuCA scANVI model directory
    luca_cfg : dict
        Configuration dict with keys:
        - surgery_epochs: int (default 200)
        - batch_size_infer: int (default 1024)
        - export_probs: bool (default True)
        - show_progress: bool (default True)

    Returns
    -------
    LuCAMappingResult
        Result object with paths and metrics
    """
    from scvi.model import SCANVI
    import torch

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

    # Parse config
    surgery_epochs = int(luca_cfg.get("surgery_epochs", 200))
    batch_size_infer = int(luca_cfg.get("batch_size_infer", 1024))
    export_probs = bool(luca_cfg.get("export_probs", True))
    show_progress = bool(luca_cfg.get("show_progress", True))

    # Ensure output directories exist
    output_latent_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_report_path.parent.mkdir(parents=True, exist_ok=True)

    # Load query data
    t0 = stage_start()
    log.info("Loading query snRNA data: %s", snrna_h5ad_path)
    query_adata = anndata.read_h5ad(snrna_h5ad_path)
    n_obs = query_adata.n_obs
    log.info("  Query: %d cells, %d genes", n_obs, query_adata.n_vars)
    stage_done("load_query", t0)
    mark_peak()

    # Load LuCA reference adata (required for model loading)
    t0 = stage_start()
    log.info("Loading LuCA reference: %s", luca_ref_h5ad_path)
    ref_adata = anndata.read_h5ad(luca_ref_h5ad_path)
    log.info("  Reference: %d cells, %d genes", ref_adata.n_obs, ref_adata.n_vars)
    stage_done("load_reference", t0)
    mark_peak()

    # Load LuCA model
    t0 = stage_start()
    log.info("Loading LuCA scANVI model from: %s", luca_model_dir)

    # Get model's expected var_names
    model_state = torch.load(luca_model_dir / "model.pt", map_location="cpu", weights_only=False)
    model_var_names = list(model_state["var_names"])
    log.info("  Model expects %d genes", len(model_var_names))

    # Check if model uses ENSG IDs
    model_uses_ensg = model_var_names[0].startswith("ENSG") if model_var_names else False

    # Subset reference adata to model's expected genes
    ref_genes = set(ref_adata.var_names.astype(str))
    model_genes_set = set(model_var_names)
    common_genes = ref_genes & model_genes_set
    log.info(
        "  Reference has %d genes, model expects %d, overlap: %d",
        len(ref_genes),
        len(model_genes_set),
        len(common_genes),
    )

    if len(common_genes) < len(model_var_names):
        # Subset reference to model genes (in model's order)
        genes_to_keep = [g for g in model_var_names if g in ref_genes]
        ref_adata_subset = ref_adata[:, genes_to_keep].copy()
        log.info("  Subset reference to %d model genes", ref_adata_subset.n_vars)
    else:
        # Reorder to match model
        ref_adata_subset = ref_adata[:, model_var_names].copy()
        log.info("  Reordered reference to match model gene order")

    # Load reference model WITH subset reference adata (required for scArches surgery)
    try:
        ref_model = SCANVI.load(str(luca_model_dir), adata=ref_adata_subset)
        log.info("  LuCA model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load LuCA scANVI model: {e}") from e

    stage_done("load_model", t0)
    mark_peak()

    # Gene matching
    t0 = stage_start()
    query_genes = set(query_adata.var_names.astype(str))

    # Handle ENSG conversion if needed
    if model_uses_ensg and "ensembl_id" in query_adata.var.columns:
        log.info("  Converting query var_names to ENSG IDs...")
        query_adata.var["gene_symbol"] = query_adata.var_names.tolist()
        new_var_names = []
        for i, symbol in enumerate(query_adata.var_names):
            ensg = query_adata.var["ensembl_id"].iloc[i]
            new_var_names.append(ensg if ensg else symbol)
        query_adata.var_names = new_var_names
        query_genes = set(query_adata.var_names.astype(str))
        n_converted = sum(1 for n in new_var_names if n.startswith("ENSG"))
        log.info("  Converted %d/%d genes to ENSG IDs", n_converted, len(new_var_names))

    model_genes = set(model_var_names)
    overlap = query_genes & model_genes
    overlap_ratio = len(overlap) / len(model_genes) if model_genes else 0
    log.info("  Gene overlap: %d/%d (%.1f%%)", len(overlap), len(model_genes), 100 * overlap_ratio)

    gene_report = {
        "query_genes": len(query_genes),
        "model_genes": len(model_genes),
        "overlap": len(overlap),
        "overlap_percent": 100 * overlap_ratio,
        "model_uses_ensg": model_uses_ensg,
    }
    with open(gene_report_path, "w") as f:
        json.dump(gene_report, f, indent=2)

    stage_done("gene_overlap", t0)
    mark_peak()

    # Prepare query for scANVI
    t0 = stage_start()
    log.info("Preparing query anndata for scANVI...")
    query_copy = query_adata.copy()

    # Set required columns
    query_copy.obs["scanvi_label"] = "unlabeled"
    if "dataset" not in query_copy.obs.columns:
        query_copy.obs["dataset"] = "query_dataset"

    try:
        SCANVI.prepare_query_anndata(query_copy, ref_model)
        log.info("  Query prepared - genes matched to reference")
    except Exception as e:
        raise ValueError(f"Failed to prepare query anndata: {e}") from e

    stage_done("prepare_query", t0)
    mark_peak()

    # Load query into model
    t0 = stage_start()
    log.info("Loading query into LuCA model...")
    try:
        query_model = SCANVI.load_query_data(query_copy, ref_model)
        log.info("  Query model created for surgery")
    except Exception as e:
        raise ValueError(f"Failed to load query data: {e}") from e

    stage_done("load_query_model", t0)
    mark_peak()

    # scArches surgery
    t0 = stage_start()
    log.info("Running scArches surgery (max %d epochs)...", surgery_epochs)
    train_kwargs = {
        "max_epochs": surgery_epochs,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_validation",
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.5,
        "plan_kwargs": {"weight_decay": 0.0, "lr": 2e-4},
        "check_val_every_n_epoch": 1,
        "train_size": 0.9,
        "enable_progress_bar": show_progress,
    }

    try:
        query_model.train(**train_kwargs)
        log.info("  Surgery complete")
    except Exception as e:
        log.warning("  Surgery training had issues: %s", e)
        log.warning("  Proceeding to get embeddings anyway...")

    stage_done("surgery", t0)
    mark_peak()

    # Get latent representation
    t0 = stage_start()
    log.info("Getting latent representation...")
    try:
        latent = query_model.get_latent_representation(query_copy, batch_size=batch_size_infer)
        latent = np.asarray(latent, dtype=np.float32)
        log.info("  Latent shape: %s", latent.shape)
    except Exception as e:
        raise ValueError(f"Failed to get latent representation: {e}") from e

    stage_done("latent_inference", t0)
    mark_peak()

    # Get cell type predictions
    t0 = stage_start()
    log.info("Getting cell type predictions...")
    try:
        pred = query_model.predict(query_copy, batch_size=batch_size_infer)
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0].to_numpy()
        luca_labels = np.asarray(pred, dtype=object).astype(str)
        log.info("  Predicted %d cell type labels", len(luca_labels))

        # Get prediction probabilities
        max_probs = None
        entropy = None
        if export_probs:
            probs = query_model.predict(query_copy, soft=True, batch_size=batch_size_infer)
            if isinstance(probs, tuple):
                probs = probs[0]
            if isinstance(probs, pd.DataFrame):
                probs = probs.to_numpy(dtype=np.float32)
            probs = np.asarray(probs, dtype=np.float32)
            max_probs = probs.max(axis=1)
            entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
            log.info("  Got prediction probabilities: shape %s", probs.shape)
    except Exception as e:
        log.warning("  Cell type prediction failed: %s", e)
        luca_labels = np.full(n_obs, "unknown", dtype=object)
        max_probs = None
        entropy = None

    stage_done("label_inference", t0)
    mark_peak()

    # Build output dataframes
    t0 = stage_start()
    obs_index = query_adata.obs.index.copy()

    # Labels parquet
    labels_df = pd.DataFrame(index=obs_index)
    labels_df.index.name = "cell_id"
    labels_df["luca_label"] = luca_labels
    if max_probs is not None:
        labels_df["luca_max_prob"] = max_probs
        labels_df["luca_entropy"] = entropy
        labels_df["luca_uncertain"] = max_probs < 0.2
    labels_df.to_parquet(output_labels_parquet_path, index=True, engine="pyarrow")
    log.info("  Saved labels: %s", output_labels_parquet_path)

    # Latent h5ad
    latent_obs = pd.DataFrame(index=obs_index)
    latent_obs.index.name = "cell_id"
    if "donor_id" in query_adata.obs.columns:
        latent_obs["donor_id"] = query_adata.obs["donor_id"].values
    if "stage" in query_adata.obs.columns:
        latent_obs["stage"] = query_adata.obs["stage"].values
    if "sample_id" in query_adata.obs.columns:
        latent_obs["sample_id"] = query_adata.obs["sample_id"].values
    latent_obs["luca_label"] = luca_labels
    if max_probs is not None:
        latent_obs["luca_max_prob"] = max_probs
        latent_obs["luca_entropy"] = entropy

    latent_var = pd.DataFrame(
        index=pd.Index([f"latent_{i}" for i in range(latent.shape[1])], name="latent")
    )
    latent_adata = anndata.AnnData(X=latent, obs=latent_obs, var=latent_var)
    latent_adata.write_h5ad(output_latent_h5ad_path, compression="lzf")
    log.info("  Saved latent: %s", output_latent_h5ad_path)

    stage_done("save_outputs", t0)
    mark_peak()

    # Build report
    label_counts = pd.Series(luca_labels).value_counts().head(10)
    top10 = [(str(idx), int(val)) for idx, val in label_counts.items()]

    wall_time = time.perf_counter() - wall_t0

    report = {
        "ok": True,
        "run_id": run_id,
        "inputs": {
            "snrna_h5ad": str(snrna_h5ad_path),
            "n_obs": n_obs,
            "luca_model_dir": str(luca_model_dir),
        },
        "outputs": {
            "latent_h5ad": str(output_latent_h5ad_path),
            "labels_parquet": str(output_labels_parquet_path),
        },
        "luca_mapping": {
            "overlap_percent": 100.0 * overlap_ratio,
            "latent_dim": latent.shape[1],
            "top10_labels": top10,
        },
        "wall_time_seconds": stage_times,
        "total_wall_time_seconds": wall_time,
        "peak_rss_mb": peak_rss_mb,
    }

    with open(mapping_report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("  Saved report: %s", mapping_report_path)

    # Save progress
    progress = {
        "status": "completed",
        "run_id": run_id,
        "finished_at_unix": time.time(),
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    return LuCAMappingResult(
        run_id=run_id,
        latent_h5ad_path=output_latent_h5ad_path,
        labels_parquet_path=output_labels_parquet_path,
        mapping_report_path=mapping_report_path,
        gene_report_path=gene_report_path,
        overlap_percent=100.0 * overlap_ratio,
        latent_shape=(n_obs, latent.shape[1]),
        top10_labels=top10,
        peak_rss_mb=peak_rss_mb,
        wall_time_seconds=wall_time,
        report=report,
    )
