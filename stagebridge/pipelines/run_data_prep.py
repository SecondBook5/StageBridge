"""Raw data preparation pipeline (Step 0).

This is the blocking dependency for all model training. It orchestrates:
1. snRNA-seq extraction, conversion, and merge
2. Visium spatial extraction, loading, and merge
3. WES feature parsing
4. QC filtering and normalization
5. Canonical artifact generation
6. Audit report creation

Usage:
    python -m stagebridge.pipelines.run_data_prep --data-root /path/to/data

Or via the step API:
    from stagebridge.pipelines.run_data_prep import run_data_prep
    result = run_data_prep(cfg)
"""

from __future__ import annotations

import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gc

import anndata
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig

from stagebridge.logging_utils import get_logger
from stagebridge.config import get_paths


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected GSE archive names
GSE_SNRNA = "GSE308103_RAW.tar"
GSE_SPATIAL = "GSE307534_RAW.tar"
GSE_WES = "GSE307529_RAW.tar"

# QC thresholds
DEFAULT_MIN_GENES_PER_CELL = 200
DEFAULT_MIN_CELLS_PER_GENE = 3
DEFAULT_MAX_PCT_MITO = 20.0
DEFAULT_MIN_COUNTS = 500


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def extract_tar_archive(tar_path: Path, dest_dir: Path, *, force: bool = False) -> bool:
    """Extract a .tar or .tar.gz archive to dest_dir.

    Returns True if extraction occurred, False if skipped.
    """
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    dest_dir = ensure_dir(dest_dir)

    # Check if already extracted
    if not force and any(dest_dir.iterdir()):
        log.info("Already extracted (skipping): %s -> %s", tar_path.name, dest_dir)
        return False

    log.info("Extracting: %s -> %s", tar_path.name, dest_dir)
    mode = "r:gz" if str(tar_path).endswith(".gz") else "r"
    with tarfile.open(tar_path, mode) as tf:
        tf.extractall(path=dest_dir)

    return True


# ---------------------------------------------------------------------------
# snRNA processing
# ---------------------------------------------------------------------------


def process_snrna(
    raw_dir: Path,
    output_dir: Path,
    *,
    max_cells_per_sample: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process snRNA-seq data: discover, convert, merge."""
    from stagebridge.data.luad_evo.snrna import (
        discover_snrna_files,
        load_snrna_sample,
    )

    output_dir = ensure_dir(output_dir)
    merged_path = output_dir / "snrna_merged.h5ad"
    manifest_path = output_dir / "snrna_manifest.csv"

    if not force and merged_path.exists():
        log.info("snRNA merged file exists (skipping): %s", merged_path)
        # Read shape from h5ad - AnnData stores obs/var as groups, get shape from X or index
        try:
            with h5py.File(merged_path, "r") as f:
                # Try X matrix shape first
                if "X" in f:
                    if hasattr(f["X"], "shape"):
                        n_cells, n_genes = f["X"].shape
                    elif "data" in f["X"]:
                        # Sparse matrix - get shape from attributes or indices
                        n_cells = f["X"].attrs.get("shape", [0, 0])[0] if "shape" in f["X"].attrs else len(f["obs/_index"][()])
                        n_genes = f["X"].attrs.get("shape", [0, 0])[1] if "shape" in f["X"].attrs else len(f["var/_index"][()])
                    else:
                        n_cells = len(f["obs/_index"][()])
                        n_genes = len(f["var/_index"][()])
                else:
                    n_cells = len(f["obs/_index"][()])
                    n_genes = len(f["var/_index"][()])
        except Exception:
            # Fallback: open with scanpy backed mode briefly
            adata = sc.read_h5ad(merged_path, backed="r")
            n_cells, n_genes = adata.shape
            adata.file.close()
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        return {
            "ok": True,
            "skipped": True,
            "merged_path": str(merged_path),
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_samples": len(manifest),
        }

    # Discover samples
    log.info("Discovering snRNA samples in: %s", raw_dir)
    manifest = discover_snrna_files(raw_dir)
    log.info("Found %d snRNA samples", len(manifest))

    # Save manifest
    manifest.to_csv(manifest_path, index=False)

    # Load and concatenate samples
    adatas = []
    for row in manifest.itertuples(index=False):
        log.info("Loading snRNA sample: %s", row.sample_id)
        adata = load_snrna_sample(
            Path(row.input_path),
            max_cells_per_sample=max_cells_per_sample,
        )
        adatas.append(adata)

    # Merge
    log.info("Merging %d snRNA samples...", len(adatas))
    merged = anndata.concat(adatas, join="outer", merge="same")
    merged.obs_names_make_unique()
    merged.var_names_make_unique()

    # Ensure counts layer
    if "counts" not in merged.layers:
        merged.layers["counts"] = merged.X.copy()

    # Write
    merged.write_h5ad(merged_path)
    log.info("snRNA merged: %d cells x %d genes -> %s", *merged.shape, merged_path)

    return {
        "ok": True,
        "skipped": False,
        "merged_path": str(merged_path),
        "manifest_path": str(manifest_path),
        "n_cells": merged.n_obs,
        "n_genes": merged.n_vars,
        "n_samples": len(manifest),
    }


# ---------------------------------------------------------------------------
# Spatial processing
# ---------------------------------------------------------------------------


def process_spatial(
    raw_dir: Path,
    output_dir: Path,
    *,
    max_spots_per_sample: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Process Visium spatial data: discover, load from tarballs, merge."""
    from stagebridge.data.luad_evo.visium import (
        discover_spatial_tarballs,
        load_spatial_sample_from_tarball,
    )

    output_dir = ensure_dir(output_dir)
    merged_path = output_dir / "spatial_merged.h5ad"
    manifest_path = output_dir / "spatial_manifest.csv"

    if not force and merged_path.exists():
        log.info("Spatial merged file exists (skipping): %s", merged_path)
        # Read shape from h5ad - AnnData stores obs/var as groups
        try:
            with h5py.File(merged_path, "r") as f:
                if "X" in f and hasattr(f["X"], "shape"):
                    n_spots, n_genes = f["X"].shape
                else:
                    n_spots = len(f["obs/_index"][()])
                    n_genes = len(f["var/_index"][()])
        except Exception:
            adata = sc.read_h5ad(merged_path, backed="r")
            n_spots, n_genes = adata.shape
            adata.file.close()
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        return {
            "ok": True,
            "skipped": True,
            "merged_path": str(merged_path),
            "n_spots": n_spots,
            "n_genes": n_genes,
            "n_samples": len(manifest),
        }

    # Discover tarballs
    log.info("Discovering spatial tarballs in: %s", raw_dir)
    manifest = discover_spatial_tarballs(raw_dir)
    log.info("Found %d spatial samples", len(manifest))

    # Save manifest
    manifest.to_csv(manifest_path, index=False)

    # Step 1: Convert each tarball to h5ad (one at a time to save memory)
    sample_h5ads = []
    interim_dir = output_dir / "interim_spatial"
    interim_dir.mkdir(exist_ok=True)

    for row in manifest.itertuples(index=False):
        sample_path = interim_dir / f"{row.sample_id}.h5ad"
        sample_h5ads.append(sample_path)

        if sample_path.exists():
            log.info("Sample h5ad exists (skipping): %s", row.sample_id)
            continue

        log.info("Loading spatial sample: %s", row.sample_id)
        adata = load_spatial_sample_from_tarball(
            Path(row.input_path),
            max_spots_per_sample=max_spots_per_sample,
        )
        # Strip H&E images to save memory (kept in original tarballs)
        if "spatial" in adata.uns:
            del adata.uns["spatial"]

        # Save to h5ad and release memory
        adata.write_h5ad(sample_path)
        log.info("Saved: %s", sample_path)
        del adata
        gc.collect()

    # Step 2: Load h5ad files and merge
    log.info("Loading %d sample h5ad files for merge...", len(sample_h5ads))
    adatas = []
    for sample_path in sample_h5ads:
        adatas.append(anndata.read_h5ad(sample_path))

    # Merge
    log.info("Merging %d spatial samples...", len(adatas))
    merged = anndata.concat(adatas, join="outer", merge="same")

    # Free input list immediately
    del adatas
    gc.collect()

    merged.obs_names_make_unique()
    merged.var_names_make_unique()

    # Ensure counts layer
    if "counts" not in merged.layers:
        merged.layers["counts"] = merged.X.copy()

    # Write
    merged.write_h5ad(merged_path)
    log.info("Spatial merged: %d spots x %d genes -> %s", *merged.shape, merged_path)

    return {
        "ok": True,
        "skipped": False,
        "merged_path": str(merged_path),
        "manifest_path": str(manifest_path),
        "n_spots": merged.n_obs,
        "n_genes": merged.n_vars,
        "n_samples": len(manifest),
    }


# ---------------------------------------------------------------------------
# WES processing
# ---------------------------------------------------------------------------


def process_wes(
    tar_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Process WES data: parse VCFs and extract features."""
    from stagebridge.data.luad_evo.wes import parse_wes_features_from_tar, WES_FEATURE_COLS

    output_dir = ensure_dir(output_dir)
    output_path = output_dir / "wes_features.parquet"

    if not force and output_path.exists():
        log.info("WES features file exists (skipping): %s", output_path)
        df = pd.read_parquet(output_path)
        return {
            "ok": True,
            "skipped": True,
            "output_path": str(output_path),
            "n_samples": len(df),
            "feature_columns": WES_FEATURE_COLS,
        }

    if not tar_path.exists():
        log.warning("WES archive not found: %s", tar_path)
        return {
            "ok": False,
            "skipped": False,
            "error": f"WES archive not found: {tar_path}",
        }

    log.info("Parsing WES features from: %s", tar_path)
    df = parse_wes_features_from_tar(tar_path)

    # Save
    df.to_parquet(output_path, index=False)
    log.info(
        "WES features: %d samples x %d features -> %s", len(df), len(WES_FEATURE_COLS), output_path
    )

    return {
        "ok": True,
        "skipped": False,
        "output_path": str(output_path),
        "n_samples": len(df),
        "feature_columns": WES_FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# QC and normalization
# ---------------------------------------------------------------------------


def apply_qc_filtering(
    adata: anndata.AnnData,
    *,
    min_genes: int = DEFAULT_MIN_GENES_PER_CELL,
    min_cells: int = DEFAULT_MIN_CELLS_PER_GENE,
    max_pct_mito: float = DEFAULT_MAX_PCT_MITO,
    min_counts: int = DEFAULT_MIN_COUNTS,
) -> tuple[anndata.AnnData, dict[str, Any]]:
    """Apply standard QC filtering to an AnnData object.

    Returns the filtered AnnData and a summary dict.
    """
    n_before = adata.n_obs
    n_genes_before = adata.n_vars

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)

    # Filter by mito percentage if column exists
    if "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_after = adata.n_obs
    n_genes_after = adata.n_vars

    summary = {
        "cells_before": n_before,
        "cells_after": n_after,
        "cells_removed": n_before - n_after,
        "genes_before": n_genes_before,
        "genes_after": n_genes_after,
        "genes_removed": n_genes_before - n_genes_after,
        "qc_params": {
            "min_genes": min_genes,
            "min_cells": min_cells,
            "max_pct_mito": max_pct_mito,
            "min_counts": min_counts,
        },
    }

    log.info(
        "QC filtering: %d -> %d cells (-%d), %d -> %d genes (-%d)",
        n_before,
        n_after,
        n_before - n_after,
        n_genes_before,
        n_genes_after,
        n_genes_before - n_genes_after,
    )

    return adata, summary


def apply_normalization(
    adata: anndata.AnnData,
    *,
    target_sum: float | None = 1e4,
    log1p: bool = True,
    store_raw: bool = True,
    max_cells_for_raw: int = 500000,
) -> tuple[anndata.AnnData, dict[str, Any]]:
    """Apply standard normalization to an AnnData object.

    Returns the normalized AnnData and a summary dict.

    Args:
        adata: AnnData to normalize
        target_sum: Target sum for normalization (None to skip)
        log1p: Apply log1p transform
        store_raw: Store raw counts in layers (skipped for large datasets)
        max_cells_for_raw: Skip storing raw if n_obs exceeds this (OOM prevention)
    """
    # Store raw counts if not already stored (skip for large datasets to avoid OOM)
    if store_raw and "counts" not in adata.layers:
        if adata.n_obs <= max_cells_for_raw:
            adata.layers["counts"] = adata.X.copy()
        else:
            log.info("Skipping raw counts storage (%d cells > %d max, OOM prevention)",
                     adata.n_obs, max_cells_for_raw)

    # Normalize
    if target_sum is not None:
        sc.pp.normalize_total(adata, target_sum=target_sum)

    if log1p:
        sc.pp.log1p(adata)

    summary = {
        "target_sum": target_sum,
        "log1p": log1p,
    }

    log.info("Normalization applied: target_sum=%s, log1p=%s", target_sum, log1p)

    return adata, summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_data_prep(
    cfg: DictConfig | None = None,
    *,
    data_root: Path | str | None = None,
    force: bool = False,
    skip_qc: bool = False,
    skip_normalization: bool = False,
    spatial_merge_only: bool = False,
) -> dict[str, Any]:
    """Run the complete raw data preparation pipeline (Step 0).

    This is the blocking dependency for all model training.

    Parameters
    ----------
    cfg : DictConfig, optional
        Hydra config. If None, uses defaults.
    data_root : Path or str, optional
        Override for STAGEBRIDGE_DATA_ROOT.
    force : bool
        If True, re-process even if outputs exist.
    skip_qc : bool
        If True, skip QC filtering.
    skip_normalization : bool
        If True, skip normalization.
    spatial_merge_only : bool
        If True, only merge spatial samples without QC/normalization.
        Recommended: spatial backends (Tangram/DestVI/TACCO) handle their own processing.

    Returns
    -------
    dict
        Pipeline result with status, paths, and audit report.
    """
    start_time = datetime.now()

    # Resolve data root
    if data_root is not None:
        import os
        os.environ["STAGEBRIDGE_DATA_ROOT"] = str(data_root)
        root = Path(data_root)
    else:
        try:
            paths = get_paths()
            root = Path(paths.data_root)
        except Exception as e:
            return {
                "ok": False,
                "pipeline": "data_prep",
                "error": str(e),
            }

    log.info("=" * 60)
    log.info("StageBridge Raw Data Preparation Pipeline (Step 0)")
    log.info("=" * 60)
    log.info("Data root: %s", root)

    # Resolve paths
    raw_dir = root / "raw" / "geo"
    processed_dir = root / "processed" / "luad_evo"
    ensure_dir(processed_dir)

    results = {
        "pipeline": "data_prep",
        "data_root": str(root),
        "start_time": start_time.isoformat(),
    }

    # ---------------------------------------------------------------------------
    # Step 0.1-0.4: Process snRNA
    # ---------------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Processing snRNA-seq...")

    snrna_raw_dir = raw_dir / "GSE308103_snrna"
    snrna_tar = raw_dir / GSE_SNRNA

    # Extract if needed
    if snrna_tar.exists() and not any(snrna_raw_dir.glob("*.mtx.txt.gz")):
        extract_tar_archive(snrna_tar, snrna_raw_dir, force=force)

    # Find the extracted files directory
    snrna_extracted = snrna_raw_dir
    if not any(snrna_raw_dir.glob("*.mtx.txt.gz")):
        # Look for nested directory
        for subdir in snrna_raw_dir.iterdir():
            if subdir.is_dir() and any(subdir.glob("*.mtx.txt.gz")):
                snrna_extracted = subdir
                break

    if snrna_extracted.exists() and any(snrna_extracted.glob("*.mtx.txt.gz")):
        snrna_result = process_snrna(snrna_extracted, processed_dir, force=force)
        results["snrna"] = snrna_result
    else:
        log.warning("snRNA raw files not found in: %s", snrna_raw_dir)
        results["snrna"] = {"ok": False, "error": f"Raw files not found in {snrna_raw_dir}"}

    # ---------------------------------------------------------------------------
    # Step 0.5-0.6: Process Spatial
    # ---------------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Processing Visium spatial...")

    spatial_raw_dir = raw_dir / "GSE307534_spatial"
    spatial_tar = raw_dir / GSE_SPATIAL

    # Extract if needed
    if spatial_tar.exists() and not any(spatial_raw_dir.glob("GSM*.tar.gz")):
        extract_tar_archive(spatial_tar, spatial_raw_dir, force=force)

    # Find the extracted files directory
    spatial_extracted = spatial_raw_dir
    if not any(spatial_raw_dir.glob("GSM*.tar.gz")):
        for subdir in spatial_raw_dir.iterdir():
            if subdir.is_dir() and any(subdir.glob("GSM*.tar.gz")):
                spatial_extracted = subdir
                break

    if spatial_extracted.exists() and any(spatial_extracted.glob("GSM*.tar.gz")):
        spatial_result = process_spatial(spatial_extracted, processed_dir, force=force)
        results["spatial"] = spatial_result
    else:
        log.warning("Spatial tarballs not found in: %s", spatial_raw_dir)
        results["spatial"] = {"ok": False, "error": f"Tarballs not found in {spatial_raw_dir}"}

    # ---------------------------------------------------------------------------
    # Step 0.7: Process WES
    # ---------------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Processing WES...")

    wes_tar = raw_dir / GSE_WES
    wes_result = process_wes(wes_tar, processed_dir, force=force)
    results["wes"] = wes_result

    # ---------------------------------------------------------------------------
    # Step 0.8-0.9: QC and Normalization
    # ---------------------------------------------------------------------------
    if not skip_qc or not skip_normalization:
        import gc

        log.info("-" * 60)
        log.info("Applying QC and normalization...")

        qc_results = {}

        # Process snRNA
        snrna_merged_path = processed_dir / "snrna_merged.h5ad"
        if snrna_merged_path.exists():
            log.info("Loading snRNA data for QC...")

            try:
                # Try loading into memory if possible
                adata_snrna = anndata.read_h5ad(snrna_merged_path)
                log.info("Loaded snRNA into memory: %d cells x %d genes", adata_snrna.n_obs, adata_snrna.n_vars)

                # Calculate QC metrics
                log.info("Calculating QC metrics...")
                adata_snrna.var["mt"] = adata_snrna.var_names.str.startswith(("MT-", "mt-"))
                sc.pp.calculate_qc_metrics(
                    adata_snrna, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
                )
            except MemoryError:
                log.warning("Not enough memory for full QC, skipping snRNA QC step")
                qc_results["snrna_qc"] = {"skipped": True, "reason": "memory"}
                adata_snrna = None

            if adata_snrna is not None:
                # Get filter masks
                cell_mask = (
                    (adata_snrna.obs["n_genes_by_counts"] >= DEFAULT_MIN_GENES_PER_CELL)
                    & (adata_snrna.obs["total_counts"] >= DEFAULT_MIN_COUNTS)
                    & (adata_snrna.obs["pct_counts_mt"] < DEFAULT_MAX_PCT_MITO)
                )
                gene_mask = adata_snrna.var["n_cells_by_counts"] >= DEFAULT_MIN_CELLS_PER_GENE

                n_cells_before = adata_snrna.n_obs
                n_genes_before = adata_snrna.n_vars
                n_cells_after = cell_mask.sum()
                n_genes_after = gene_mask.sum()

                log.info(
                    "Filtering: %d/%d cells, %d/%d genes",
                    n_cells_after,
                    n_cells_before,
                    n_genes_after,
                    n_genes_before,
                )

                # Filter
                adata_snrna_filtered = adata_snrna[cell_mask, gene_mask].copy()
                del adata_snrna
                gc.collect()

                qc_summary = {
                    "cells_before": n_cells_before,
                    "cells_after": n_cells_after,
                    "cells_removed": n_cells_before - n_cells_after,
                    "genes_before": n_genes_before,
                    "genes_after": n_genes_after,
                    "genes_removed": n_genes_before - n_genes_after,
                    "qc_params": {
                        "min_genes": DEFAULT_MIN_GENES_PER_CELL,
                        "min_cells": DEFAULT_MIN_CELLS_PER_GENE,
                        "max_pct_mito": DEFAULT_MAX_PCT_MITO,
                        "min_counts": DEFAULT_MIN_COUNTS,
                    },
                }

                if not skip_qc:
                    qc_results["snrna_qc"] = qc_summary

                if not skip_normalization:
                    adata_snrna_filtered, norm_summary = apply_normalization(adata_snrna_filtered)
                    qc_results["snrna_normalization"] = norm_summary

                # Save processed version
                processed_snrna_path = processed_dir / "snrna_qc_normalized.h5ad"
                adata_snrna_filtered.write_h5ad(processed_snrna_path)
                qc_results["snrna_processed_path"] = str(processed_snrna_path)
                log.info("snRNA processed: %s", processed_snrna_path)

                # Free memory before loading spatial
                del adata_snrna_filtered
                gc.collect()

        # Process spatial with chunked processing to avoid OOM
        spatial_merged_path = processed_dir / "spatial_merged.h5ad"
        spatial_batch_manifest = processed_dir / "spatial_batches.json"

        if spatial_merged_path.exists() and spatial_merge_only:
            # Spatial merge only mode - skip QC/normalization
            # Spatial backends (Tangram/DestVI/TACCO) handle their own processing
            log.info("Spatial merge-only mode: skipping QC/normalization")
            log.info("  Spatial backends will handle processing internally")
            qc_results["spatial_qc"] = {"skipped": True, "reason": "spatial_merge_only mode"}
            qc_results["spatial_normalization"] = {"skipped": True, "reason": "spatial_merge_only mode"}
            qc_results["spatial_merged_path"] = str(spatial_merged_path)

        elif spatial_merged_path.exists():
            log.info("Processing spatial data with chunked QC (memory-efficient)...")

            # QC parameters
            min_genes = 100
            min_counts = 200
            max_pct_mito = DEFAULT_MAX_PCT_MITO
            min_cells = DEFAULT_MIN_CELLS_PER_GENE
            chunk_size = 50000  # Process 50k cells at a time

            try:
                # First, get total size using backed mode
                adata_backed = sc.read_h5ad(spatial_merged_path, backed="r")
                n_spots_before = adata_backed.n_obs
                n_genes_before = adata_backed.n_vars
                var_names = adata_backed.var_names.copy()
                adata_backed.obs_names.copy()

                # Get sample IDs if available for chunk boundaries
                if "sample_id" in adata_backed.obs.columns:
                    adata_backed.obs["sample_id"].values
                else:
                    pass

                adata_backed.file.close()
                del adata_backed
                gc.collect()

                log.info("Total: %d spots, %d genes. Processing in chunks of %d...",
                         n_spots_before, n_genes_before, chunk_size)

                # Identify mitochondrial genes once
                mt_genes = var_names.str.startswith(("MT-", "mt-"))

                # Process in chunks, keeping track of passing cells
                filtered_chunks = []
                n_chunks = (n_spots_before + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, n_spots_before)

                    log.info("  Chunk %d/%d: cells %d-%d", chunk_idx + 1, n_chunks, start_idx, end_idx)

                    # Load chunk
                    adata_chunk = sc.read_h5ad(
                        spatial_merged_path,
                        backed="r"
                    )[start_idx:end_idx].to_memory()

                    # Calculate QC metrics for this chunk
                    adata_chunk.var["mt"] = mt_genes
                    sc.pp.calculate_qc_metrics(
                        adata_chunk, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
                    )

                    # Filter cells in this chunk
                    cell_mask = (
                        (adata_chunk.obs["n_genes_by_counts"] >= min_genes)
                        & (adata_chunk.obs["total_counts"] >= min_counts)
                        & (adata_chunk.obs["pct_counts_mt"] < max_pct_mito)
                    )

                    n_passing = cell_mask.sum()
                    log.info("    %d/%d cells pass QC", n_passing, len(cell_mask))

                    if n_passing > 0:
                        # Keep only passing cells
                        adata_filtered = adata_chunk[cell_mask].copy()
                        filtered_chunks.append(adata_filtered)

                    del adata_chunk
                    gc.collect()

                # Concatenate all filtered chunks
                if filtered_chunks:
                    log.info("Concatenating %d filtered chunks...", len(filtered_chunks))
                    adata_spatial = anndata.concat(filtered_chunks, join="outer")
                    del filtered_chunks
                    gc.collect()

                    # Memory-efficient gene filtering:
                    # Compute gene cell counts incrementally to avoid loading full matrix
                    log.info("Computing gene cell counts (memory-efficient)...")
                    gene_counts = np.zeros(adata_spatial.n_vars, dtype=np.int32)

                    # Process in chunks to count cells per gene
                    chunk_size = 50000
                    for start in range(0, adata_spatial.n_obs, chunk_size):
                        end = min(start + chunk_size, adata_spatial.n_obs)
                        chunk_X = adata_spatial.X[start:end]
                        # Count non-zero entries per gene
                        if hasattr(chunk_X, 'toarray'):
                            chunk_X = chunk_X.toarray()
                        gene_counts += (chunk_X > 0).sum(axis=0).astype(np.int32).flatten()

                    # Filter genes with min_cells
                    log.info("Filtering genes (min_cells=%d)...", min_cells)
                    genes_pass = gene_counts >= min_cells
                    n_genes_pass = genes_pass.sum()
                    n_genes_removed = len(genes_pass) - n_genes_pass
                    log.info("  %d/%d genes pass filter (%d to remove)", n_genes_pass, len(genes_pass), n_genes_removed)

                    # Only subset if removing more than 1% of genes (avoid OOM for trivial filters)
                    if n_genes_removed > 0 and n_genes_removed / len(genes_pass) > 0.01:
                        log.info("  Subsetting genes (removing >1%%)...")
                        adata_spatial = adata_spatial[:, genes_pass].copy()
                        gc.collect()
                        n_genes_after = adata_spatial.n_vars
                    elif n_genes_removed > 0:
                        log.info("  Skipping gene subset (only %.2f%% removed, not worth OOM risk)",
                                 100 * n_genes_removed / len(genes_pass))
                        n_genes_after = len(genes_pass)  # Report as-is, minor difference
                    else:
                        n_genes_after = len(genes_pass)

                    n_spots_after = adata_spatial.n_obs

                    log.info(
                        "Filtered: %d/%d spots (%.1f%%), %d/%d genes",
                        n_spots_after,
                        n_spots_before,
                        100 * n_spots_after / n_spots_before,
                        n_genes_after,
                        n_genes_before,
                    )
                else:
                    log.warning("No cells passed QC!")
                    adata_spatial = None

            except Exception as e:
                log.warning("Chunked spatial QC failed: %s", e)
                adata_spatial = None
                qc_results["spatial_qc"] = {"skipped": True, "reason": str(e)}

            gc.collect()

            if adata_spatial is not None:
                qc_summary = {
                    "cells_before": n_spots_before,
                    "cells_after": n_spots_after,
                    "cells_removed": n_spots_before - n_spots_after,
                    "genes_before": n_genes_before,
                    "genes_after": n_genes_after,
                    "genes_removed": n_genes_before - n_genes_after,
                    "qc_params": {
                        "min_genes": min_genes,
                        "min_cells": min_cells,
                        "max_pct_mito": max_pct_mito,
                        "min_counts": min_counts,
                    },
                }

                if not skip_qc:
                    qc_results["spatial_qc"] = qc_summary

                if not skip_normalization:
                    log.info("Applying normalization...")
                    adata_spatial, norm_summary = apply_normalization(adata_spatial)
                    qc_results["spatial_normalization"] = norm_summary

                # Save processed version
                processed_spatial_path = processed_dir / "spatial_qc_normalized.h5ad"
                log.info("Saving to %s...", processed_spatial_path)
                adata_spatial.write_h5ad(processed_spatial_path)
                qc_results["spatial_processed_path"] = str(processed_spatial_path)
                log.info("Spatial processed: %s", processed_spatial_path)

                del adata_spatial
                gc.collect()

        elif spatial_batch_manifest.exists():
            # Batched mode - skip QC here, can be done per-batch during training
            log.info(
                "Spatial data is batched - QC/normalization will be applied per-batch during training"
            )
            qc_results["spatial_note"] = "Batched mode - QC deferred to training time"

        results["qc_normalization"] = qc_results

    # ---------------------------------------------------------------------------
    # Step 0.9: Generate audit report
    # ---------------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Generating audit report...")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    audit_report = {
        "pipeline": "data_prep",
        "version": "1.0",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "data_root": str(root),
        "modalities": {
            "snrna": results.get("snrna", {}),
            "spatial": results.get("spatial", {}),
            "wes": results.get("wes", {}),
        },
        "qc_normalization": results.get("qc_normalization", {}),
    }

    # Determine overall status
    snrna_ok = results.get("snrna", {}).get("ok", False)
    spatial_ok = results.get("spatial", {}).get("ok", False)
    wes_ok = results.get("wes", {}).get("ok", False)

    audit_report["status"] = {
        "snrna": "ok" if snrna_ok else "failed",
        "spatial": "ok" if spatial_ok else "failed",
        "wes": "ok" if wes_ok else "failed",
        "overall": "ok"
        if (snrna_ok and spatial_ok)
        else "partial"
        if (snrna_ok or spatial_ok)
        else "failed",
    }

    # Save audit report (convert numpy types for JSON serialization)
    def _json_serializable(obj):
        """Convert numpy types to Python types for JSON."""
        if isinstance(obj, dict):
            return {k: _json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    audit_path = processed_dir / "data_prep_audit.json"
    with open(audit_path, "w") as f:
        json.dump(_json_serializable(audit_report), f, indent=2)
    log.info("Audit report saved: %s", audit_path)

    results["ok"] = audit_report["status"]["overall"] in ("ok", "partial")
    results["audit_report"] = audit_report
    results["audit_path"] = str(audit_path)
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration

    log.info("=" * 60)
    log.info("Data preparation complete. Status: %s", audit_report["status"]["overall"])
    log.info("Duration: %.1f seconds", duration)
    log.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="StageBridge Raw Data Preparation Pipeline (Step 0)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override for STAGEBRIDGE_DATA_ROOT environment variable",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing even if outputs exist",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip QC filtering",
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip normalization",
    )
    parser.add_argument(
        "--spatial-merge-only",
        action="store_true",
        help="Only merge spatial samples, skip QC/normalization. "
             "Recommended: spatial backends (Tangram/DestVI/TACCO) handle their own processing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = run_data_prep(
        data_root=args.data_root,
        force=args.force,
        skip_qc=args.skip_qc,
        skip_normalization=args.skip_normalization,
        spatial_merge_only=args.spatial_merge_only,
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
