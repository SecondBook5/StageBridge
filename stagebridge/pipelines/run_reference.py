"""Dual-reference mapping pipeline entrypoint.

Maps query cells to BOTH HLCA (healthy) and LuCA (cancer) reference spaces,
producing separate and fused latent embeddings for each cell.

Dual-reference design:
- HLCA = healthy lung anchor
- LuCA = disease-aware / malignant-progressive anchor
- Fused = comparative coordinate system for progression-relevant cells

Supports three modes:
1. HLCA-only (--hlca-only)
2. LuCA-only (--luca-only)
3. HLCA+LuCA fused (default)

Usage:
    python -m stagebridge.pipelines.run_reference \
        --data-root /path/to/stagebridge/data

Output:
    Creates reference_geometry/ directory with:
    - hlca_embedding.parquet
    - luca_embedding.parquet
    - fused_embedding.parquet
    - reference_confidence.parquet
    - reference_manifest.json
    - feature_overlap_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal
import uuid
import numpy as np


def calibrate_confidence_percentile(
    distances: np.ndarray,
    method: str = "percentile",
) -> tuple[np.ndarray, str]:
    """Calibrate confidence from distances using percentile/rank-based approach.

    This ensures confidence is comparable across references with different densities.
    Lower distance = higher percentile rank = higher confidence.

    Parameters
    ----------
    distances : np.ndarray
        Raw k-NN mean distances for each cell (lower = closer match)
    method : str
        Calibration method: "percentile" (default), "robust_zscore", or "sigmoid"

    Returns
    -------
    confidence : np.ndarray
        Calibrated confidence values in [0, 1]
    method_used : str
        Description of calibration method
    """
    if distances is None or len(distances) == 0:
        return np.array([]), "none"

    distances = np.asarray(distances, dtype=np.float64)

    # Handle NaN/inf
    valid_mask = np.isfinite(distances)
    if not valid_mask.all():
        distances = np.where(valid_mask, distances, np.nanmedian(distances[valid_mask]))

    if method == "percentile":
        # Percentile rank: lower distance = higher confidence
        # Rank from 0 (worst/highest distance) to 1 (best/lowest distance)
        from scipy.stats import rankdata
        ranks = rankdata(distances, method='average')
        # Invert: highest rank (highest distance) -> lowest confidence
        confidence = 1.0 - (ranks - 1) / (len(ranks) - 1 + 1e-10)
        method_used = "percentile_rank"

    elif method == "robust_zscore":
        # Robust z-score using median and MAD
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        mad = max(mad, 1e-10)  # Avoid division by zero

        # Z-score (inverted: lower distance = higher z)
        z = (median - distances) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std

        # Map to [0, 1] using sigmoid
        confidence = 1.0 / (1.0 + np.exp(-z))
        method_used = "robust_zscore_sigmoid"

    elif method == "sigmoid":
        # Center by median, scale by IQR, then sigmoid
        median = np.median(distances)
        q75, q25 = np.percentile(distances, [75, 25])
        iqr = max(q75 - q25, 1e-10)

        # Normalized distance (inverted)
        normalized = (median - distances) / iqr

        # Sigmoid mapping to [0, 1]
        confidence = 1.0 / (1.0 + np.exp(-normalized))
        method_used = "iqr_sigmoid"

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Ensure output is in [0, 1]
    confidence = np.clip(confidence, 0.0, 1.0)

    return confidence.astype(np.float32), method_used


def normalize_latent_space(
    embeddings: np.ndarray,
    method: str = "l2",
) -> np.ndarray:
    """Normalize latent embeddings before fusion.

    Parameters
    ----------
    embeddings : np.ndarray
        Latent embeddings (n_cells, latent_dim)
    method : str
        Normalization method: "l2" (unit norm), "zscore" (per-dimension), "none"

    Returns
    -------
    normalized : np.ndarray
        Normalized embeddings
    """
    if embeddings is None:
        return None

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if method == "l2":
        # L2 normalize each cell's embedding to unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms

    elif method == "zscore":
        # Z-score normalize each dimension
        mean = embeddings.mean(axis=0, keepdims=True)
        std = embeddings.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-10)
        return (embeddings - mean) / std

    elif method == "none":
        return embeddings

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def find_reference_paths(data_root: Path) -> dict[str, Path | None]:
    """Find HLCA and LuCA reference paths."""
    results = {"hlca": None, "luca": None, "hlca_hub_cache": None}

    # HLCA: Check for HubModel cache or h5ad
    hlca_candidates = [
        data_root / "references/hlca/hlca_reference.h5ad",
        data_root / "references/hlca/hlca_core.h5ad",
    ]
    for candidate in hlca_candidates:
        if candidate.exists():
            results["hlca"] = candidate
            break

    # HLCA HubModel cache (for scANVI surgery)
    hub_cache = data_root / "references/hlca/hub_cache"
    if hub_cache.exists():
        results["hlca_hub_cache"] = hub_cache

    # LuCA: Check for h5ad
    luca_candidates = [
        data_root / "references/luca/luca_core_atlas.h5ad",  # Primary: CORE atlas
        data_root / "references/luca/luca_reference.h5ad",
        data_root / "references/luca/luca_luad.h5ad",
    ]
    for candidate in luca_candidates:
        if candidate.exists():
            results["luca"] = candidate
            break

    return results


def extract_hlca_reference_from_hub(hub_cache: Path, output_path: Path) -> Path:
    """Extract HLCA reference h5ad from HubModel cache."""
    from scvi.hub import HubModel

    print("Loading HLCA reference from HubModel cache...")
    hubmodel = HubModel.pull_from_huggingface_hub(
        "scvi-tools/human-lung-cell-atlas-scanvi",
        cache_dir=hub_cache,
    )

    ref_adata = hubmodel.adata
    print(f"  Reference cells: {ref_adata.n_obs:,}")
    print(f"  Reference genes: {ref_adata.n_vars:,}")

    # Ensure latent embedding exists
    if "X_scanvi_emb" not in ref_adata.obsm:
        print("  Computing latent embeddings...")
        ref_latent = hubmodel.model.get_latent_representation(ref_adata)
        ref_adata.obsm["X_scanvi_emb"] = ref_latent

    # Reindex to gene symbols if feature_name column exists
    if "feature_name" in ref_adata.var.columns:
        print("  Reindexing to gene symbols (feature_name)...")
        # Store original ENSG IDs
        ref_adata.var["ensembl_id"] = ref_adata.var_names.copy()
        # Get symbols and drop the column to avoid conflict
        symbols = ref_adata.var["feature_name"].astype(str).tolist()
        del ref_adata.var["feature_name"]
        # Use symbols as var_names
        ref_adata.var_names = symbols
        # Handle duplicates by making unique
        ref_adata.var_names_make_unique()
        print(f"  Gene names: {list(ref_adata.var_names[:5])}")

    print(f"  Saving reference to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref_adata.write_h5ad(output_path)

    return output_path


def reindex_reference_to_symbols(ref_path: Path, max_size_gb: float = 5.0) -> Path:
    """Reindex a reference h5ad to use gene symbols instead of ENSG IDs.

    For large files (> max_size_gb), creates a gene mapping instead of rewriting.
    """
    import anndata

    file_size_gb = ref_path.stat().st_size / (1024**3)
    print(f"Checking gene format in {ref_path.name} ({file_size_gb:.1f} GB)...")

    # For large files, just check and warn - don't rewrite
    if file_size_gb > max_size_gb:
        print("  Large file - checking with backed mode...")
        try:
            adata = anndata.read_h5ad(ref_path, backed='r')
            first_gene = str(adata.var_names[0])
            if first_gene.startswith("ENSG") and "feature_name" in adata.var.columns:
                # Create a mapping file instead of rewriting
                mapping_path = ref_path.parent / f"{ref_path.stem}_gene_mapping.parquet"
                if not mapping_path.exists():
                    import pandas as pd
                    gene_map = pd.DataFrame({
                        "ensembl_id": adata.var_names.astype(str),
                        "gene_symbol": adata.var["feature_name"].astype(str),
                    })
                    gene_map.to_parquet(mapping_path)
                    print(f"  Created gene mapping: {mapping_path}")
                print("  NOTE: Large file uses ENSG IDs. Pipeline will use feature_name for matching.")
            adata.file.close()
        except (ValueError, OSError) as e:
            # h5py version incompatibility - skip check and proceed
            print(f"  Warning: Could not read in backed mode ({e.__class__.__name__}), skipping gene format check")
        return ref_path

    adata = anndata.read_h5ad(ref_path)

    # Check if already using symbols (not ENSG)
    first_gene = str(adata.var_names[0])
    if not first_gene.startswith("ENSG"):
        print(f"  Already using gene symbols: {first_gene}")
        return ref_path

    # Check for feature_name column
    if "feature_name" not in adata.var.columns:
        print("  WARNING: No feature_name column, keeping ENSG IDs")
        return ref_path

    print("  Reindexing from ENSG to gene symbols...")
    adata.var["ensembl_id"] = adata.var_names.copy()
    adata.var_names = adata.var["feature_name"].astype(str)
    adata.var_names_make_unique()

    # Save back
    adata.write_h5ad(ref_path)
    print(f"  Done. Gene names: {list(adata.var_names[:5])}")

    return ref_path


def run_reference(cfg: Any) -> dict[str, Any]:
    """Compatibility wrapper for old API.

    TODO: Implement proper config-based reference mapping.
    For now, this is a stub to fix imports.
    """
    raise NotImplementedError(
        "run_reference() is deprecated. Use run_hpc_reference_mapping() or run_dual_reference_mapping() directly."
    )


def run_hpc_reference_mapping(
    query_path: Path,
    hlca_path: Path | None,
    luca_path: Path | None,
    output_dir: Path,
    *,
    data_root: Path | None = None,
    mode: Literal["both", "hlca_only", "luca_only"] = "both",
    k_neighbors: int = 50,
    hlca_latent_key: str = "X_scanvi_emb",
    luca_latent_key: str = "X_scVI",
    chunk_size: int = 50000,
    smoke_mode: bool = False,
    run_id: str | None = None,
    use_model_based: bool = True,
) -> int:
    """HPC mode: Model-based reference projection (primary) or chunked k-NN (fallback)."""
    import anndata
    import pandas as pd
    import json
    import time

    run_id = run_id or f"ref_geo_hpc_{uuid.uuid4().hex[:8]}"

    # Determine which references to use
    use_hlca = mode in ("both", "hlca_only") and hlca_path is not None
    use_luca = mode in ("both", "luca_only") and luca_path is not None

    if not use_hlca and not use_luca:
        print("ERROR: No references available for the selected mode.")
        return 1

    print()
    print("=" * 60)
    print("HPC Dual-Reference Mapping (Chunked/Streaming)")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Mode: {mode}")
    print(f"  Query: {query_path}")
    print(f"  HLCA: {hlca_path if use_hlca else 'disabled'}")
    print(f"  LuCA: {luca_path if use_luca else 'disabled'}")
    print(f"  Output: {output_dir}")
    print(f"  k-neighbors: {k_neighbors}")
    print(f"  Chunk size: {chunk_size:,}")
    if smoke_mode:
        print("  SMOKE MODE: Using 1000 cells only")
    print()

    t0 = time.perf_counter()

    # Load query data in backed mode to avoid memory explosion
    # 787K cells x 15K genes in dense = 47GB - keep sparse/on-disk
    print("Loading query data (backed mode)...")
    query_adata = anndata.read_h5ad(query_path, backed='r')
    print(f"  Query: {query_adata.n_obs:,} cells, {query_adata.n_vars:,} genes")

    if smoke_mode:
        n_smoke = min(1000, query_adata.n_obs)
        idx = np.random.choice(query_adata.n_obs, n_smoke, replace=False)
        # Need to copy for smoke mode since we're subsetting
        query_adata_full = anndata.read_h5ad(query_path)
        query_adata = query_adata_full[idx].copy()
        del query_adata_full
        print(f"  Smoke mode: subsampled to {query_adata.n_obs} cells")

    # Determine model paths (for model-based mapping)
    hlca_model_path = None
    luca_model_path = None

    if use_model_based and use_hlca:
        # Check for HLCA scANVI model (HubModel cache or saved model)
        hlca_dir = hlca_path.parent if hlca_path else None
        if hlca_dir:
            candidates = [
                # HubModel cache (downloaded from scvi-hub) - note: model is -scanvi suffix
                hlca_dir / "hub_cache" / "models--scvi-tools--human-lung-cell-atlas-scanvi" / "snapshots",
                # Legacy path without hub_cache
                hlca_dir / "models--scvi-tools--human-lung-cell-atlas-scanvi" / "snapshots",
                # Direct model directory
                hlca_dir / "hlca_scanvi_model",
                hlca_dir / "model",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue

                # For HubModel cache, find the snapshot subdirectory
                if "snapshots" in str(candidate):
                    try:
                        snapshot_dirs = list(candidate.iterdir())
                        if snapshot_dirs:
                            model_dir = snapshot_dirs[0]  # Latest snapshot
                            if (model_dir / "model.pt").exists():
                                hlca_model_path = model_dir
                                print(f"  Found HLCA scANVI HubModel: {hlca_model_path.name}")
                                break
                    except Exception:
                        continue
                elif (candidate / "model.pt").exists():
                    hlca_model_path = candidate
                    print(f"  Found HLCA scANVI model: {hlca_model_path}")
                    break

    if use_model_based and use_luca:
        # Check for LuCA scANVI model (extracted from core_atlas_scanvi_model.tar.gz)
        # Use passed data_root, or try to infer from luca_path
        luca_data_root = data_root or (luca_path.parent.parent.parent if luca_path else None)
        if luca_data_root:
            candidates = [
                # Extracted core atlas model (nested path from tar.gz)
                luca_data_root / "processed/LuCA/data/20_build_atlas/annotate_datasets/35_final_atlas/full_atlas_hvg_integrated_scvi_scanvi_model",
                # Alternative: if moved to references/
                luca_data_root / "references/luca/luca_scanvi_model",
                luca_data_root / "references/luca/model",
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "model.pt").exists():
                    luca_model_path = candidate
                    print(f"  Found LuCA scANVI model: {luca_model_path}")
                    break

    # Run model-based mapping (primary) or chunked k-NN (fallback)
    if use_model_based and (hlca_model_path or luca_model_path):
        print("  Using MODEL-BASED projection (principled)")
        from stagebridge.reference.map_query_model import map_to_dual_reference_model_based
        results = map_to_dual_reference_model_based(
            query_adata,
            hlca_model_path if use_hlca else None,
            luca_model_path if use_luca else None,
            fallback_hlca_h5ad=hlca_path if use_hlca else None,
            fallback_luca_h5ad=luca_path if use_luca else None,
            batch_size=10000,
        )
    else:
        print("  Using k-NN projection (models not available)")
        from stagebridge.reference.map_query_chunked import map_to_dual_reference_chunked
        results = map_to_dual_reference_chunked(
            query_adata,
            hlca_path if use_hlca else None,
            luca_path if use_luca else None,
            hlca_latent_key=hlca_latent_key,
            luca_latent_key=luca_latent_key,
            k_neighbors=k_neighbors,
            use_faiss=True,
        )

    wall_time = time.perf_counter() - t0

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get metadata from query
    cell_ids = results["cell_ids"]
    donor_ids = query_adata.obs.get("donor_id", pd.Series(["unknown"] * len(cell_ids))).astype(str).values
    sample_ids = query_adata.obs.get("sample_id", pd.Series(["unknown"] * len(cell_ids))).astype(str).values
    stage_ids = query_adata.obs.get("stage", pd.Series(["unknown"] * len(cell_ids))).astype(str).values

    # Normalize latent spaces BEFORE fusion (per-reference L2 normalization)
    hlca_emb_normalized = normalize_latent_space(results["hlca_embeddings"], method="l2")
    luca_emb_normalized = normalize_latent_space(results["luca_embeddings"], method="l2")

    # Build fused embedding from normalized latents
    if hlca_emb_normalized is not None and luca_emb_normalized is not None:
        fused_normalized = np.concatenate([hlca_emb_normalized, luca_emb_normalized], axis=1)
    elif hlca_emb_normalized is not None:
        fused_normalized = hlca_emb_normalized
    elif luca_emb_normalized is not None:
        fused_normalized = luca_emb_normalized
    else:
        fused_normalized = None

    # Calibrate confidence using percentile rank (not naive distance conversion)
    # This ensures HLCA and LuCA confidences are comparable despite density differences
    hlca_conf, hlca_conf_method = calibrate_confidence_percentile(
        results["hlca_distances"], method="percentile"
    ) if results["hlca_distances"] is not None else (np.zeros(len(cell_ids), dtype=np.float32), "none")

    luca_conf, luca_conf_method = calibrate_confidence_percentile(
        results["luca_distances"], method="percentile"
    ) if results["luca_distances"] is not None else (np.zeros(len(cell_ids), dtype=np.float32), "none")

    print(f"  Confidence calibration: HLCA={hlca_conf_method}, LuCA={luca_conf_method}")

    # HLCA embedding (normalized)
    if hlca_emb_normalized is not None:
        hlca_df = pd.DataFrame({
            "cell_id": cell_ids,
            "donor_id": donor_ids,
            "sample_id": sample_ids,
            "stage_id": stage_ids,
        })
        for i in range(hlca_emb_normalized.shape[1]):
            hlca_df[f"hlca_latent_{i}"] = hlca_emb_normalized[:, i]
        hlca_df.to_parquet(output_dir / "hlca_embedding.parquet", index=False)
        print(f"  Saved hlca_embedding.parquet: {hlca_emb_normalized.shape}")

    # LuCA embedding (normalized)
    if luca_emb_normalized is not None:
        luca_df = pd.DataFrame({
            "cell_id": cell_ids,
            "donor_id": donor_ids,
            "sample_id": sample_ids,
            "stage_id": stage_ids,
        })
        for i in range(luca_emb_normalized.shape[1]):
            luca_df[f"luca_latent_{i}"] = luca_emb_normalized[:, i]
        luca_df.to_parquet(output_dir / "luca_embedding.parquet", index=False)
        print(f"  Saved luca_embedding.parquet: {luca_emb_normalized.shape}")

    # Fused embedding (from normalized latents)
    if fused_normalized is not None:
        fused_df = pd.DataFrame({
            "cell_id": cell_ids,
            "donor_id": donor_ids,
            "sample_id": sample_ids,
            "stage_id": stage_ids,
            "reference_mode_used": mode,
        })
        for i in range(fused_normalized.shape[1]):
            fused_df[f"fused_latent_{i}"] = fused_normalized[:, i]
        fused_df.to_parquet(output_dir / "fused_embedding.parquet", index=False)
        print(f"  Saved fused_embedding.parquet: {fused_normalized.shape}")

    # Confidence with calibration (includes raw distances for reference)
    conf_df = pd.DataFrame({
        "cell_id": cell_ids,
        "donor_id": donor_ids,
        "sample_id": sample_ids,
        "stage_id": stage_ids,
        "reference_mode_used": mode,
        # Calibrated confidence (comparable across references)
        "hlca_confidence": hlca_conf if len(hlca_conf) > 0 else 0.0,
        "luca_confidence": luca_conf if len(luca_conf) > 0 else 0.0,
        # Raw distances (for debugging/analysis)
        "hlca_raw_distance": results["hlca_distances"] if results["hlca_distances"] is not None else 0.0,
        "luca_raw_distance": results["luca_distances"] if results["luca_distances"] is not None else 0.0,
        # Calibration method used
        "hlca_confidence_method": hlca_conf_method,
        "luca_confidence_method": luca_conf_method,
    })
    conf_df.to_parquet(output_dir / "reference_confidence.parquet", index=False)
    print("  Saved reference_confidence.parquet with calibrated confidence")

    # Feature overlap report
    feature_overlap = {
        "hlca": results["metadata"].get("hlca", {}),
        "luca": results["metadata"].get("luca", {}),
    }
    with open(output_dir / "feature_overlap_report.json", "w") as f:
        json.dump(feature_overlap, f, indent=2, default=str)

    # Manifest
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "reference_mode_used": mode,
        "processing_mode": "hpc_chunked",
        "n_cells": len(cell_ids),
        "hlca_dim": hlca_emb_normalized.shape[1] if hlca_emb_normalized is not None else 0,
        "luca_dim": luca_emb_normalized.shape[1] if luca_emb_normalized is not None else 0,
        "fused_dim": fused_normalized.shape[1] if fused_normalized is not None else 0,
        "k_neighbors": k_neighbors,
        "chunk_size": chunk_size,
        "wall_time_seconds": wall_time,
        "latent_normalization": "l2",
        "confidence_calibration": {
            "hlca_method": hlca_conf_method,
            "luca_method": luca_conf_method,
            "description": "Percentile rank calibration ensures comparable confidence across references with different densities",
        },
        "metadata": results["metadata"],
    }
    with open(output_dir / "reference_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Generate figures
    print("\nGenerating figures...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        _generate_reference_figures(
            results=results,
            hlca_conf=hlca_conf,
            luca_conf=luca_conf,
            fused_embeddings=fused_normalized,
            stage_ids=stage_ids,
            output_dir=plots_dir,
        )
        print(f"  Figures saved to: {plots_dir}")
    except Exception as e:
        print(f"  Warning: Figure generation failed: {e}")

    # Plot training curves if available (from model-based mapping)
    try:
        hlca_history = results["metadata"].get("hlca", {}).get("training_history")
        luca_history = results["metadata"].get("luca", {}).get("training_history")
        if hlca_history or luca_history:
            _plot_training_curves(hlca_history, luca_history, plots_dir)
            print(f"  Training curves saved to: {plots_dir / 'scarches_training_curves.png'}")
    except Exception as e:
        print(f"  Warning: Training curve plotting failed: {e}")

    print()
    print("=" * 60)
    print("HPC Reference Mapping Complete")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Mode: {mode}")
    print(f"  Cells: {len(cell_ids):,}")
    print(f"  HLCA dim: {manifest['hlca_dim']}")
    print(f"  LuCA dim: {manifest['luca_dim']}")
    print(f"  Fused dim: {manifest['fused_dim']}")
    print("  Latent normalization: L2 (per-reference)")
    print("  Confidence calibration: percentile rank (comparable across refs)")
    print(f"  Wall time: {wall_time:.1f}s")
    # Run diagnostics
    print("\nRunning diagnostics...")
    diagnostics = _run_reference_diagnostics(
        hlca_distances=results["hlca_distances"],
        luca_distances=results["luca_distances"],
        hlca_conf=hlca_conf,
        luca_conf=luca_conf,
        hlca_emb=hlca_emb_normalized,
        luca_emb=luca_emb_normalized,
        fused_emb=fused_normalized,
        output_dir=output_dir,
    )

    # Save diagnostics
    with open(output_dir / "diagnostics_report.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    print(f"  Diagnostics saved to: {output_dir / 'diagnostics_report.json'}")

    # Cleanup backed file handle
    try:
        if hasattr(query_adata, 'file') and query_adata.file is not None:
            query_adata.file.close()
    except Exception:
        pass  # Ignore cleanup errors

    print()
    print(f"Outputs saved to: {output_dir}")
    print("  - hlca_embedding.parquet (L2-normalized latents)")
    print("  - luca_embedding.parquet (L2-normalized latents)")
    print("  - fused_embedding.parquet (concatenated normalized latents)")
    print("  - reference_confidence.parquet (calibrated confidence + raw distances)")
    print("  - reference_manifest.json")
    print("  - feature_overlap_report.json")
    print("  - diagnostics_report.json")
    print()
    print("Next step: run_spatial_benchmark.py")

    return 0


def _run_reference_diagnostics(
    hlca_distances: np.ndarray | None,
    luca_distances: np.ndarray | None,
    hlca_conf: np.ndarray,
    luca_conf: np.ndarray,
    hlca_emb: np.ndarray | None,
    luca_emb: np.ndarray | None,
    fused_emb: np.ndarray | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run diagnostics on dual-reference mapping outputs.

    Checks:
    1. Raw distance distributions (HLCA vs LuCA)
    2. Calibrated confidence distributions
    3. Whether one branch dominates after calibration
    4. Data integrity (NaN, shape)
    5. Readiness summary for each mode
    """
    diagnostics = {
        "status": "complete",
        "issues": [],
        "warnings": [],
    }

    # 1. Raw distance statistics
    dist_stats = {"hlca": {}, "luca": {}}

    if hlca_distances is not None and len(hlca_distances) > 0:
        dist_stats["hlca"] = {
            "mean": float(np.nanmean(hlca_distances)),
            "std": float(np.nanstd(hlca_distances)),
            "median": float(np.nanmedian(hlca_distances)),
            "min": float(np.nanmin(hlca_distances)),
            "max": float(np.nanmax(hlca_distances)),
            "q25": float(np.nanpercentile(hlca_distances, 25)),
            "q75": float(np.nanpercentile(hlca_distances, 75)),
            "nan_count": int(np.isnan(hlca_distances).sum()),
        }
    else:
        dist_stats["hlca"] = {"status": "not_available"}

    if luca_distances is not None and len(luca_distances) > 0:
        dist_stats["luca"] = {
            "mean": float(np.nanmean(luca_distances)),
            "std": float(np.nanstd(luca_distances)),
            "median": float(np.nanmedian(luca_distances)),
            "min": float(np.nanmin(luca_distances)),
            "max": float(np.nanmax(luca_distances)),
            "q25": float(np.nanpercentile(luca_distances, 25)),
            "q75": float(np.nanpercentile(luca_distances, 75)),
            "nan_count": int(np.isnan(luca_distances).sum()),
        }
    else:
        dist_stats["luca"] = {"status": "not_available"}

    diagnostics["raw_distance_stats"] = dist_stats

    # Check for systematic distance bias (before calibration)
    if "mean" in dist_stats["hlca"] and "mean" in dist_stats["luca"]:
        hlca_mean = dist_stats["hlca"]["mean"]
        luca_mean = dist_stats["luca"]["mean"]
        ratio = hlca_mean / (luca_mean + 1e-10)
        diagnostics["raw_distance_ratio_hlca_over_luca"] = float(ratio)
        if ratio > 2.0:
            diagnostics["warnings"].append(
                f"HLCA raw distances {ratio:.1f}x larger than LuCA - density difference detected, calibration applied"
            )
        elif ratio < 0.5:
            diagnostics["warnings"].append(
                f"LuCA raw distances {1/ratio:.1f}x larger than HLCA - unusual, check references"
            )

    # 2. Calibrated confidence statistics
    conf_stats = {"hlca": {}, "luca": {}}

    if len(hlca_conf) > 0 and hlca_conf.max() > 0:
        conf_stats["hlca"] = {
            "mean": float(np.nanmean(hlca_conf)),
            "std": float(np.nanstd(hlca_conf)),
            "median": float(np.nanmedian(hlca_conf)),
            "min": float(np.nanmin(hlca_conf)),
            "max": float(np.nanmax(hlca_conf)),
            "nan_count": int(np.isnan(hlca_conf).sum()),
        }
    else:
        conf_stats["hlca"] = {"status": "not_available"}

    if len(luca_conf) > 0 and luca_conf.max() > 0:
        conf_stats["luca"] = {
            "mean": float(np.nanmean(luca_conf)),
            "std": float(np.nanstd(luca_conf)),
            "median": float(np.nanmedian(luca_conf)),
            "min": float(np.nanmin(luca_conf)),
            "max": float(np.nanmax(luca_conf)),
            "nan_count": int(np.isnan(luca_conf).sum()),
        }
    else:
        conf_stats["luca"] = {"status": "not_available"}

    diagnostics["calibrated_confidence_stats"] = conf_stats

    # 3. Check if one branch dominates after calibration
    if "mean" in conf_stats["hlca"] and "mean" in conf_stats["luca"]:
        hlca_mean_conf = conf_stats["hlca"]["mean"]
        luca_mean_conf = conf_stats["luca"]["mean"]
        conf_ratio = hlca_mean_conf / (luca_mean_conf + 1e-10)
        diagnostics["calibrated_confidence_ratio_hlca_over_luca"] = float(conf_ratio)

        # After percentile calibration, means should be ~0.5 for both
        if abs(hlca_mean_conf - 0.5) > 0.1 or abs(luca_mean_conf - 0.5) > 0.1:
            diagnostics["warnings"].append(
                f"Calibrated confidence means deviate from expected 0.5: HLCA={hlca_mean_conf:.3f}, LuCA={luca_mean_conf:.3f}"
            )

        if abs(conf_ratio - 1.0) < 0.2:
            diagnostics["calibration_balance"] = "GOOD - references are balanced after calibration"
        else:
            diagnostics["calibration_balance"] = f"CHECK - ratio={conf_ratio:.2f}, may indicate reference quality difference"

    # 4. Embedding integrity checks
    emb_checks = {}

    if hlca_emb is not None:
        emb_checks["hlca"] = {
            "shape": list(hlca_emb.shape),
            "nan_count": int(np.isnan(hlca_emb).sum()),
            "inf_count": int(np.isinf(hlca_emb).sum()),
            "mean_norm": float(np.linalg.norm(hlca_emb, axis=1).mean()),
            "ready": bool(np.isnan(hlca_emb).sum() == 0 and np.isinf(hlca_emb).sum() == 0),
        }
        if not emb_checks["hlca"]["ready"]:
            diagnostics["issues"].append("HLCA embeddings contain NaN or Inf")
    else:
        emb_checks["hlca"] = {"status": "not_available", "ready": False}

    if luca_emb is not None:
        emb_checks["luca"] = {
            "shape": list(luca_emb.shape),
            "nan_count": int(np.isnan(luca_emb).sum()),
            "inf_count": int(np.isinf(luca_emb).sum()),
            "mean_norm": float(np.linalg.norm(luca_emb, axis=1).mean()),
            "ready": bool(np.isnan(luca_emb).sum() == 0 and np.isinf(luca_emb).sum() == 0),
        }
        if not emb_checks["luca"]["ready"]:
            diagnostics["issues"].append("LuCA embeddings contain NaN or Inf")
    else:
        emb_checks["luca"] = {"status": "not_available", "ready": False}

    if fused_emb is not None:
        emb_checks["fused"] = {
            "shape": list(fused_emb.shape),
            "nan_count": int(np.isnan(fused_emb).sum()),
            "inf_count": int(np.isinf(fused_emb).sum()),
            "ready": bool(np.isnan(fused_emb).sum() == 0 and np.isinf(fused_emb).sum() == 0),
        }
        if not emb_checks["fused"]["ready"]:
            diagnostics["issues"].append("Fused embeddings contain NaN or Inf")
    else:
        emb_checks["fused"] = {"status": "not_available", "ready": False}

    diagnostics["embedding_integrity"] = emb_checks

    # 5. Readiness summary for each mode
    readiness = {
        "hlca_only": emb_checks.get("hlca", {}).get("ready", False),
        "luca_only": emb_checks.get("luca", {}).get("ready", False),
        "fused": emb_checks.get("fused", {}).get("ready", False),
    }
    diagnostics["mode_readiness"] = readiness

    # Overall status
    if diagnostics["issues"]:
        diagnostics["status"] = "issues_detected"
    elif diagnostics["warnings"]:
        diagnostics["status"] = "complete_with_warnings"
    else:
        diagnostics["status"] = "complete"

    # Print summary
    print("\n  === Diagnostics Summary ===")
    print(f"  Status: {diagnostics['status']}")
    if "raw_distance_ratio_hlca_over_luca" in diagnostics:
        print(f"  Raw distance ratio (HLCA/LuCA): {diagnostics['raw_distance_ratio_hlca_over_luca']:.2f}")
    if "calibrated_confidence_ratio_hlca_over_luca" in diagnostics:
        print(f"  Calibrated confidence ratio (HLCA/LuCA): {diagnostics['calibrated_confidence_ratio_hlca_over_luca']:.2f}")
    if "calibration_balance" in diagnostics:
        print(f"  Calibration balance: {diagnostics['calibration_balance']}")
    print(f"  Mode readiness: HLCA-only={readiness['hlca_only']}, LuCA-only={readiness['luca_only']}, Fused={readiness['fused']}")
    if diagnostics["warnings"]:
        print(f"  Warnings: {len(diagnostics['warnings'])}")
        for w in diagnostics["warnings"]:
            print(f"    - {w}")
    if diagnostics["issues"]:
        print(f"  Issues: {len(diagnostics['issues'])}")
        for i in diagnostics["issues"]:
            print(f"    - {i}")

    return diagnostics


def _generate_reference_figures(
    results: dict[str, Any],
    hlca_conf: np.ndarray,
    luca_conf: np.ndarray,
    fused_embeddings: np.ndarray | None,
    stage_ids: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate visualization figures for reference mapping results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Stage colors
    STAGE_COLORS = {
        "Normal": "#00BA38",
        "AAH": "#F8766D",
        "AIS": "#619CFF",
        "MIA": "#E58700",
        "LUAD": "#A3A500",
        "Unknown": "#999999",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dual-Reference Mapping Results (Calibrated Confidence)", fontsize=14, fontweight='bold')

    # 1. Fused embedding UMAP/PCA colored by stage
    ax = axes[0, 0]
    if fused_embeddings is not None and not np.isnan(fused_embeddings).any():
        emb = fused_embeddings
        # Use PCA for speed (UMAP would be better but slower)
        if emb.shape[1] > 2:
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb)
            var_explained = pca.explained_variance_ratio_.sum() * 100
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        else:
            emb_2d = emb
            var_explained = 100
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")

        for stage in STAGE_COLORS:
            mask = stage_ids == stage
            if mask.sum() > 0:
                ax.scatter(
                    emb_2d[mask, 0], emb_2d[mask, 1],
                    c=STAGE_COLORS[stage], label=stage,
                    s=10, alpha=0.5, edgecolors='none'
                )
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f"Fused Embedding (PCA, {var_explained:.0f}% var)")
    else:
        ax.text(0.5, 0.5, "No valid fused embedding", ha='center', va='center')
        ax.set_title("Fused Embedding")

    # 2. HLCA vs LuCA calibrated confidence scatter
    ax = axes[0, 1]
    if len(hlca_conf) > 0 and len(luca_conf) > 0 and hlca_conf.max() > 0 and luca_conf.max() > 0:
        for stage in STAGE_COLORS:
            mask = stage_ids == stage
            if mask.sum() > 0:
                ax.scatter(
                    hlca_conf[mask], luca_conf[mask],
                    c=STAGE_COLORS[stage], label=stage,
                    s=10, alpha=0.5, edgecolors='none'
                )

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal confidence')
        ax.set_xlabel("HLCA Confidence (healthy, calibrated)")
        ax.set_ylabel("LuCA Confidence (cancer, calibrated)")
        ax.set_title("Calibrated Reference Confidence")
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "Need both references", ha='center', va='center')
        ax.set_title("Reference Confidence")

    # 3. Distance distributions by stage
    ax = axes[1, 0]
    if results["hlca_distances"] is not None:
        stages_present = [s for s in STAGE_COLORS if (stage_ids == s).sum() > 0]
        data = [results["hlca_distances"][stage_ids == s] for s in stages_present]
        colors = [STAGE_COLORS[s] for s in stages_present]

        bp = ax.boxplot(data, labels=stages_present, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("HLCA Distance")
        ax.set_title("HLCA Mapping Distance by Stage")
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, "No HLCA mapping", ha='center', va='center')
        ax.set_title("HLCA Distance by Stage")

    # 4. Stage composition / summary stats
    ax = axes[1, 1]
    stage_counts = {}
    for stage in STAGE_COLORS:
        count = (stage_ids == stage).sum()
        if count > 0:
            stage_counts[stage] = count

    if stage_counts:
        stages = list(stage_counts.keys())
        counts = list(stage_counts.values())
        colors = [STAGE_COLORS[s] for s in stages]

        bars = ax.bar(stages, counts, color=colors, alpha=0.8)
        ax.set_ylabel("Cell Count")
        ax.set_title("Stage Distribution")
        ax.tick_params(axis='x', rotation=45)

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No stage data", ha='center', va='center')
        ax.set_title("Stage Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "reference_mapping_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Additional: Latent dimension heatmap
    if results["fused_embeddings"] is not None:
        _plot_latent_heatmap(results, stage_ids, output_dir)

    # Additional: Calibration diagnostics
    _plot_calibration_diagnostics(
        results["hlca_distances"],
        results["luca_distances"],
        hlca_conf,
        luca_conf,
        output_dir,
    )


def _plot_latent_heatmap(
    results: dict[str, Any],
    stage_ids: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot average latent values per stage as heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, emb) in enumerate([
        ("HLCA", results.get("hlca_embeddings")),
        ("LuCA", results.get("luca_embeddings")),
    ]):
        ax = axes[idx]
        if emb is None:
            ax.text(0.5, 0.5, f"No {name} embedding", ha='center', va='center')
            ax.set_title(f"{name} Latent by Stage")
            continue

        # Compute mean latent per stage
        stages_present = [s for s in STAGE_ORDER if (stage_ids == s).sum() > 0]
        mean_latent = np.zeros((len(stages_present), emb.shape[1]))

        for i, stage in enumerate(stages_present):
            mask = stage_ids == stage
            mean_latent[i] = emb[mask].mean(axis=0)

        # Plot heatmap
        im = ax.imshow(mean_latent, aspect='auto', cmap='RdBu_r')
        ax.set_yticks(range(len(stages_present)))
        ax.set_yticklabels(stages_present)
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Stage")
        ax.set_title(f"{name} Mean Latent by Stage")
        plt.colorbar(im, ax=ax, label="Value")

    plt.tight_layout()
    plt.savefig(output_dir / "latent_heatmap_by_stage.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_calibration_diagnostics(
    hlca_distances: np.ndarray | None,
    luca_distances: np.ndarray | None,
    hlca_conf: np.ndarray,
    luca_conf: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot diagnostic figures comparing raw distances vs calibrated confidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Confidence Calibration Diagnostics", fontsize=14, fontweight='bold')

    # 1. Raw distance distributions (showing density bias)
    ax = axes[0, 0]
    if hlca_distances is not None:
        ax.hist(hlca_distances, bins=50, alpha=0.6, label=f"HLCA (mean={np.nanmean(hlca_distances):.3f})", color='blue', density=True)
    if luca_distances is not None:
        ax.hist(luca_distances, bins=50, alpha=0.6, label=f"LuCA (mean={np.nanmean(luca_distances):.3f})", color='red', density=True)
    ax.set_xlabel("Raw k-NN Distance")
    ax.set_ylabel("Density")
    ax.set_title("Raw Distance Distributions (BEFORE calibration)")
    ax.legend()
    ax.axvline(np.nanmean(hlca_distances) if hlca_distances is not None else 0, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(np.nanmean(luca_distances) if luca_distances is not None else 0, color='red', linestyle='--', alpha=0.5)

    # 2. Calibrated confidence distributions (should be similar)
    ax = axes[0, 1]
    if len(hlca_conf) > 0 and hlca_conf.max() > 0:
        ax.hist(hlca_conf, bins=50, alpha=0.6, label=f"HLCA (mean={np.nanmean(hlca_conf):.3f})", color='blue', density=True)
    if len(luca_conf) > 0 and luca_conf.max() > 0:
        ax.hist(luca_conf, bins=50, alpha=0.6, label=f"LuCA (mean={np.nanmean(luca_conf):.3f})", color='red', density=True)
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Calibrated Confidence Distributions (AFTER calibration)")
    ax.legend()
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Expected mean')

    # 3. Raw distance scatter (HLCA vs LuCA per cell)
    ax = axes[1, 0]
    if hlca_distances is not None and luca_distances is not None:
        ax.scatter(hlca_distances, luca_distances, s=5, alpha=0.3, c='gray')
        ax.plot([0, max(hlca_distances.max(), luca_distances.max())],
                [0, max(hlca_distances.max(), luca_distances.max())],
                'k--', alpha=0.3, label='Equal distance')
        ax.set_xlabel("HLCA Raw Distance")
        ax.set_ylabel("LuCA Raw Distance")
        ax.set_title("Raw Distance: HLCA vs LuCA per cell")
        # Add correlation
        corr = np.corrcoef(hlca_distances, luca_distances)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va='top')
    else:
        ax.text(0.5, 0.5, "Need both references", ha='center', va='center')

    # 4. Calibrated confidence scatter (should be more balanced)
    ax = axes[1, 1]
    if len(hlca_conf) > 0 and len(luca_conf) > 0 and hlca_conf.max() > 0 and luca_conf.max() > 0:
        ax.scatter(hlca_conf, luca_conf, s=5, alpha=0.3, c='gray')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal confidence')
        ax.set_xlabel("HLCA Calibrated Confidence")
        ax.set_ylabel("LuCA Calibrated Confidence")
        ax.set_title("Calibrated Confidence: HLCA vs LuCA per cell")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Add correlation
        corr = np.corrcoef(hlca_conf, luca_conf)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va='top')
    else:
        ax.text(0.5, 0.5, "Need both references", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_training_curves(
    hlca_history: dict | None,
    luca_history: dict | None,
    output_dir: Path,
) -> None:
    """Plot scArches surgery training curves (ELBO loss over epochs).

    Parameters
    ----------
    hlca_history : dict | None
        Training history from HLCA scArches surgery
    luca_history : dict | None
        Training history from LuCA scArches surgery
    output_dir : Path
        Directory to save plots
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json

    if hlca_history is None and luca_history is None:
        return

    n_plots = sum(1 for h in [hlca_history, luca_history] if h is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot HLCA training curve
    if hlca_history is not None:
        ax = axes[plot_idx]
        plot_idx += 1

        if 'elbo_train' in hlca_history:
            epochs = list(range(1, len(hlca_history['elbo_train']) + 1))
            ax.plot(epochs, hlca_history['elbo_train'], 'b-', label='Training ELBO', linewidth=2)
        if 'elbo_validation' in hlca_history:
            epochs = list(range(1, len(hlca_history['elbo_validation']) + 1))
            ax.plot(epochs, hlca_history['elbo_validation'], 'r--', label='Validation ELBO', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ELBO Loss', fontsize=12)
        ax.set_title('HLCA scArches Surgery Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add final loss annotation
        if 'elbo_train' in hlca_history:
            final_loss = hlca_history['elbo_train'][-1]
            ax.annotate(f'Final: {final_loss:.2f}',
                       xy=(len(hlca_history['elbo_train']), final_loss),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, ha='left')

    # Plot LuCA training curve
    if luca_history is not None:
        ax = axes[plot_idx]

        if 'elbo_train' in luca_history:
            epochs = list(range(1, len(luca_history['elbo_train']) + 1))
            ax.plot(epochs, luca_history['elbo_train'], 'b-', label='Training ELBO', linewidth=2)
        if 'elbo_validation' in luca_history:
            epochs = list(range(1, len(luca_history['elbo_validation']) + 1))
            ax.plot(epochs, luca_history['elbo_validation'], 'r--', label='Validation ELBO', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('ELBO Loss', fontsize=12)
        ax.set_title('LuCA scArches Surgery Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if 'elbo_train' in luca_history:
            final_loss = luca_history['elbo_train'][-1]
            ax.annotate(f'Final: {final_loss:.2f}',
                       xy=(len(luca_history['elbo_train']), final_loss),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig(output_dir / "scarches_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save training history as JSON for future analysis
    history_data = {}
    if hlca_history is not None:
        history_data['hlca'] = hlca_history
    if luca_history is not None:
        history_data['luca'] = luca_history

    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history_data, f, indent=2)


def run_dual_reference_mapping(
    query_path: Path,
    hlca_path: Path | None,
    luca_path: Path | None,
    output_dir: Path,
    *,
    mode: Literal["both", "hlca_only", "luca_only"] = "both",
    mapping_method: str = "knn_projection",
    fusion_method: str = "concat",
    k_neighbors: int = 50,
    hlca_latent_key: str = "X_scanvi_emb",
    luca_latent_key: str = "X_scVI",
    smoke_mode: bool = False,
    run_id: str | None = None,
) -> int:
    """Run dual-reference mapping pipeline."""
    from stagebridge.reference.pipeline import (
        ReferenceGeometryConfig,
        run_reference_pipeline,
    )

    run_id = run_id or f"ref_geo_{uuid.uuid4().hex[:8]}"

    # Determine which references to use
    use_hlca = mode in ("both", "hlca_only") and hlca_path is not None
    use_luca = mode in ("both", "luca_only") and luca_path is not None

    if not use_hlca and not use_luca:
        print("ERROR: No references available for the selected mode.")
        return 1

    print()
    print("=" * 60)
    print("Dual-Reference Mapping Pipeline")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print(f"  Mode: {mode}")
    print(f"  Query: {query_path}")
    print(f"  HLCA: {hlca_path if use_hlca else 'disabled'}")
    print(f"  LuCA: {luca_path if use_luca else 'disabled'}")
    print(f"  Output: {output_dir}")
    print(f"  Mapping method: {mapping_method}")
    print(f"  Fusion method: {fusion_method}")
    print(f"  k-neighbors: {k_neighbors}")
    if smoke_mode:
        print("  SMOKE MODE: Using 1000 cells only")
    print()

    config = ReferenceGeometryConfig(
        hlca_reference_path=str(hlca_path) if use_hlca else None,
        luca_reference_path=str(luca_path) if use_luca else None,
        query_data_path=str(query_path),
        mapping_method=mapping_method,
        k_neighbors=k_neighbors,
        hlca_latent_key=hlca_latent_key,
        luca_latent_key=luca_latent_key,
        fusion_method=fusion_method,
        normalize_fused=True,
        smoke_mode=smoke_mode,
        smoke_n_cells=1000,
    )

    def progress_callback(step: str, pct: float) -> None:
        print(f"  [{pct*100:5.1f}%] {step}")

    result = run_reference_pipeline(
        config,
        run_dir=output_dir,
        run_id=run_id,
        progress_callback=progress_callback,
    )

    print()
    print("=" * 60)
    if result.success:
        print("Dual-Reference Mapping Complete")
    else:
        print("Dual-Reference Mapping FAILED")
    print("=" * 60)
    print(f"  Run ID: {result.run_id}")
    print(f"  Cells: {result.n_cells:,}")
    print(f"  HLCA dim: {result.hlca_dim}")
    print(f"  LuCA dim: {result.luca_dim}")
    print(f"  Fused dim: {result.fused_dim}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")
    print(f"  Validation: {result.validation_status}")

    if result.errors:
        print("\nErrors:")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print("\nWarnings:")
        for warn in result.warnings:
            print(f"  - {warn}")

    print(f"\nOutputs saved to: {result.output_dir}")
    print("  - hlca_embedding.parquet")
    print("  - luca_embedding.parquet")
    print("  - fused_embedding.parquet")
    print("  - reference_confidence.parquet")
    print("  - reference_manifest.json")

    if result.success:
        print("\nNext step: run_spatial_benchmark.py")
        return 0
    else:
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Map query cells to HLCA and LuCA reference spaces"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing processed/ and references/",
    )
    parser.add_argument(
        "--snrna",
        type=str,
        default=None,
        help="Path to snRNA h5ad (default: {data-root}/processed/luad_evo/snrna_qc_normalized.h5ad)",
    )
    parser.add_argument(
        "--hlca",
        type=str,
        default=None,
        help="Path to HLCA reference h5ad (auto-detected if not specified)",
    )
    parser.add_argument(
        "--luca",
        type=str,
        default=None,
        help="Path to LuCA reference h5ad (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {data-root}/processed/luad_evo/reference_geometry/)",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--hlca-only",
        action="store_true",
        help="Map to HLCA only (no LuCA)",
    )
    mode_group.add_argument(
        "--luca-only",
        action="store_true",
        help="Map to LuCA only (no HLCA)",
    )

    # Latent keys (in case references use different names)
    parser.add_argument(
        "--hlca-latent-key",
        type=str,
        default="X_scanvi_emb",
        help="Key in HLCA obsm containing latent embeddings",
    )
    parser.add_argument(
        "--luca-latent-key",
        type=str,
        default="X_scVI",
        help="Key in LuCA obsm containing latent embeddings",
    )

    # Mapping parameters
    parser.add_argument(
        "--mapping-method",
        type=str,
        choices=["knn_projection", "pca_projection"],
        default="knn_projection",
        help="Method for mapping query to references",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        choices=["concat", "average", "weighted"],
        default="concat",
        help="Method for fusing HLCA and LuCA embeddings",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for k-NN projection",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (1000 cells only)",
    )
    parser.add_argument(
        "--hpc",
        action="store_true",
        help="HPC mode: chunked processing, FAISS GPU, memory-efficient",
    )
    parser.add_argument(
        "--use-model-based",
        action="store_true",
        default=True,
        help="Use pretrained scVI/scANVI models for projection (default: True)",
    )
    parser.add_argument(
        "--no-model-based",
        dest="use_model_based",
        action="store_false",
        help="Force k-NN projection instead of model-based",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Reference chunk size for streaming (HPC mode with k-NN fallback)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for tracking (default: auto-generated)",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    snrna_path = Path(args.snrna) if args.snrna else data_root / "processed/luad_evo/snrna_qc_normalized.h5ad"
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "processed/luad_evo/reference_geometry"

    # Determine mode
    if args.hlca_only:
        mode = "hlca_only"
    elif args.luca_only:
        mode = "luca_only"
    else:
        mode = "both"

    # Find or use specified reference paths
    if args.hlca:
        hlca_path = Path(args.hlca)
    elif args.luca_only:
        hlca_path = None
    else:
        ref_paths = find_reference_paths(data_root)
        hlca_path = ref_paths["hlca"]

        # If no h5ad but HubModel cache exists, extract reference
        if hlca_path is None and ref_paths["hlca_hub_cache"] is not None:
            print("HLCA h5ad not found, extracting from HubModel cache...")
            hlca_path = data_root / "references/hlca/hlca_reference.h5ad"
            try:
                extract_hlca_reference_from_hub(ref_paths["hlca_hub_cache"], hlca_path)
            except Exception as e:
                print(f"ERROR: Failed to extract HLCA reference: {e}")
                if mode != "luca_only":
                    return 1

    if args.luca:
        luca_path = Path(args.luca)
    elif args.hlca_only:
        luca_path = None
    else:
        ref_paths = find_reference_paths(data_root)
        luca_path = ref_paths["luca"]

    # Validate inputs
    if not snrna_path.exists():
        print(f"ERROR: snRNA file not found: {snrna_path}")
        print("Run run_data_prep.py first.")
        return 1

    if mode == "both":
        if hlca_path is None:
            print("WARNING: HLCA reference not found, falling back to HLCA-only mode")
            print("  Download with: python -m stagebridge.pipelines.download_references --download_hlca")
        if luca_path is None:
            print("WARNING: LuCA reference not found")
            print("  If LuCA is not available, use --hlca-only mode")
            if hlca_path is not None:
                print("  Proceeding with HLCA-only...")
                mode = "hlca_only"
            else:
                return 1

    if hlca_path is not None and not hlca_path.exists():
        print(f"ERROR: HLCA reference not found: {hlca_path}")
        return 1

    if luca_path is not None and not luca_path.exists():
        print(f"ERROR: LuCA reference not found: {luca_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure references use gene symbols (not ENSG IDs) to match query
    if hlca_path is not None:
        reindex_reference_to_symbols(hlca_path)
    if luca_path is not None:
        reindex_reference_to_symbols(luca_path)

    # Use HPC mode (chunked, memory-efficient) or standard mode
    if args.hpc:
        return run_hpc_reference_mapping(
            query_path=snrna_path,
            hlca_path=hlca_path,
            luca_path=luca_path,
            output_dir=output_dir,
            data_root=data_root,
            mode=mode,
            k_neighbors=args.k_neighbors,
            hlca_latent_key=args.hlca_latent_key,
            luca_latent_key=args.luca_latent_key,
            chunk_size=args.chunk_size,
            smoke_mode=args.smoke,
            run_id=args.run_id,
            use_model_based=args.use_model_based,
        )
    else:
        return run_dual_reference_mapping(
            query_path=snrna_path,
            hlca_path=hlca_path,
            luca_path=luca_path,
            output_dir=output_dir,
            mode=mode,
            mapping_method=args.mapping_method,
            fusion_method=args.fusion_method,
            k_neighbors=args.k_neighbors,
            hlca_latent_key=args.hlca_latent_key,
            luca_latent_key=args.luca_latent_key,
            smoke_mode=args.smoke,
            run_id=args.run_id,
        )


if __name__ == "__main__":
    raise SystemExit(main())
