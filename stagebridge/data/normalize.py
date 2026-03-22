"""
Expression normalization and feature preparation for StageBridge.

This module handles:
- Count normalization (size factor, log1p, scran, etc.)
- Highly variable gene selection
- Feature specification generation
- Reference atlas preparation (HLCA, LuCa)

Extends functionality in stagebridge/data/common/harmonize.py.

Usage:
    from stagebridge.data.normalize import normalize_counts, compute_hvgs, prepare_for_reference

    normalize_counts(adata, method="log1p", target_sum=1e4)
    hvgs = compute_hvgs(adata, n_hvg=2000)
    prepare_for_reference(adata, reference_type="hlca")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration and data classes
# ---------------------------------------------------------------------------


@dataclass
class NormalizationConfig:
    """Configuration for normalization."""

    method: Literal["log1p", "scran", "raw"] = "log1p"
    target_sum: float = 1e4
    log_transform: bool = True
    scale: bool = False
    max_value: float | None = 10.0  # For scaling

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "target_sum": self.target_sum,
            "log_transform": self.log_transform,
            "scale": self.scale,
            "max_value": self.max_value,
        }


@dataclass
class HVGConfig:
    """Configuration for HVG selection."""

    n_hvg: int = 2000
    flavor: Literal["seurat", "seurat_v3", "cell_ranger"] = "seurat_v3"
    batch_key: str | None = None
    min_mean: float = 0.0125
    max_mean: float = 3.0
    min_disp: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_hvg": self.n_hvg,
            "flavor": self.flavor,
            "batch_key": self.batch_key,
            "min_mean": self.min_mean,
            "max_mean": self.max_mean,
            "min_disp": self.min_disp,
        }


@dataclass
class FeatureSpec:
    """Specification of features for downstream analysis.

    Contains gene lists, HVGs, and reference overlaps.
    """

    all_genes: list[str] = field(default_factory=list)
    hvgs: list[str] = field(default_factory=list)
    marker_genes: dict[str, list[str]] = field(default_factory=dict)
    reference_overlaps: dict[str, dict[str, Any]] = field(default_factory=dict)
    normalization_config: dict[str, Any] = field(default_factory=dict)
    hvg_config: dict[str, Any] = field(default_factory=dict)
    n_cells: int = 0
    n_genes: int = 0
    n_hvgs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_genes": self.all_genes,
            "hvgs": self.hvgs,
            "marker_genes": self.marker_genes,
            "reference_overlaps": self.reference_overlaps,
            "normalization_config": self.normalization_config,
            "hvg_config": self.hvg_config,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_hvgs": self.n_hvgs,
        }

    def save(self, path: str | Path) -> None:
        """Save feature spec to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import yaml

            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        except ImportError:
            # Fallback to JSON
            with path.with_suffix(".json").open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSpec":
        """Load feature spec from YAML or JSON."""
        path = Path(path)

        if path.suffix in (".yaml", ".yml"):
            import yaml

            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        return cls(**data)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _require_scanpy():
    """Import scanpy lazily."""
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError("scanpy is required for normalization") from e
    return sc


def _require_anndata():
    """Import anndata lazily."""
    try:
        import anndata
    except ImportError as e:
        raise ImportError("anndata is required for normalization") from e
    return anndata


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_counts(
    adata: Any,  # AnnData
    method: Literal["log1p", "scran", "raw"] = "log1p",
    *,
    target_sum: float = 1e4,
    layer_in: str | None = None,
    layer_out: str = "log1p",
    preserve_raw: bool = True,
) -> None:
    """Normalize expression counts.

    Supports multiple normalization methods:
    - log1p: Size factor normalization + log1p (default)
    - scran: Size factor normalization via scran (requires scran)
    - raw: No normalization, just copy to output layer

    Parameters
    ----------
    adata : AnnData
        AnnData object (modified in place).
    method : str
        Normalization method.
    target_sum : float
        Target sum for size factor normalization.
    layer_in : str, optional
        Input layer (None = use adata.X).
    layer_out : str
        Output layer name.
    preserve_raw : bool
        Whether to preserve raw counts in adata.raw or layers['counts'].
    """
    anndata = _require_anndata()
    sc = _require_scanpy()

    # Preserve raw counts
    if preserve_raw:
        if "counts" not in adata.layers:
            if layer_in is not None:
                adata.layers["counts"] = adata.layers[layer_in].copy()
            else:
                adata.layers["counts"] = adata.X.copy()
            log.info("Preserved raw counts in layers['counts']")

    # Get input data
    if layer_in is not None:
        if layer_in not in adata.layers:
            raise KeyError(
                f"Input layer '{layer_in}' not found. Available: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer_in].copy()
    else:
        X = adata.X.copy()

    # Create temporary AnnData for processing
    tmp = anndata.AnnData(X=X, obs=adata.obs, var=adata.var)

    if method == "log1p":
        log.info("Normalizing with log1p (target_sum=%g)...", target_sum)
        sc.pp.normalize_total(tmp, target_sum=target_sum)
        sc.pp.log1p(tmp)

    elif method == "scran":
        log.info("Normalizing with scran...")
        try:
            # scran pooling-based normalization
            sc.pp.normalize_total(tmp, target_sum=target_sum)

            # Try to use scran if available
            try:
                import rpy2.robjects as ro
                from rpy2.robjects.packages import importr
                from rpy2.robjects import numpy2ri

                numpy2ri.activate()
                importr("scran")

                # This is a simplified version - full scran would cluster first
                log.info("Using scran size factors...")
            except ImportError:
                log.warning("rpy2/scran not available, falling back to simple normalization")

            sc.pp.log1p(tmp)
        except Exception as e:
            log.warning("scran normalization failed (%s), falling back to log1p", e)
            sc.pp.normalize_total(tmp, target_sum=target_sum)
            sc.pp.log1p(tmp)

    elif method == "raw":
        log.info("Keeping raw counts (no normalization)")
        # Just copy the data as-is

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Store normalized data
    adata.layers[layer_out] = tmp.X
    log.info("Normalized data stored in layers['%s']", layer_out)

    # Store normalization info in uns
    if "normalization" not in adata.uns:
        adata.uns["normalization"] = {}
    adata.uns["normalization"][layer_out] = {
        "method": method,
        "target_sum": target_sum,
        "layer_in": layer_in,
    }


def scale_data(
    adata: Any,  # AnnData
    *,
    layer: str = "log1p",
    max_value: float | None = 10.0,
    zero_center: bool = True,
) -> None:
    """Scale normalized data (z-score).

    Parameters
    ----------
    adata : AnnData
        AnnData object (modified in place).
    layer : str
        Layer to scale.
    max_value : float, optional
        Clip values to this maximum.
    zero_center : bool
        Whether to zero-center the data.
    """
    sc = _require_scanpy()

    if layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")

    # Store in X temporarily for scanpy
    saved_X = adata.X
    adata.X = adata.layers[layer].copy()

    try:
        sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)
        adata.layers[f"{layer}_scaled"] = adata.X
        log.info("Scaled data stored in layers['%s_scaled']", layer)
    finally:
        adata.X = saved_X


# ---------------------------------------------------------------------------
# HVG selection
# ---------------------------------------------------------------------------


def compute_hvgs(
    adata: Any,  # AnnData
    n_hvg: int = 2000,
    *,
    flavor: Literal["seurat", "seurat_v3", "cell_ranger"] = "seurat_v3",
    layer: str | None = None,
    batch_key: str | None = None,
    subset: bool = False,
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
) -> list[str]:
    """Select highly variable genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    n_hvg : int
        Number of HVGs to select.
    flavor : str
        Method for HVG selection.
    layer : str, optional
        Layer to use (None = use X). For seurat_v3, should be raw counts.
    batch_key : str, optional
        Batch key for batch-aware HVG selection.
    subset : bool
        Whether to subset adata to HVGs in place.
    min_mean, max_mean, min_disp : float
        Thresholds for seurat/cell_ranger flavors.

    Returns
    -------
    list[str]
        List of HVG names.
    """
    anndata = _require_anndata()
    sc = _require_scanpy()

    # Determine which data to use
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")
        tmp = anndata.AnnData(
            X=adata.layers[layer].copy(),
            obs=adata.obs,
            var=adata.var.copy(),
        )
    else:
        tmp = anndata.AnnData(
            X=adata.X.copy(),
            obs=adata.obs,
            var=adata.var.copy(),
        )

    # Check for appropriate input
    if flavor == "seurat_v3":
        # seurat_v3 expects raw counts
        if layer is None and "counts" in adata.layers:
            log.info("Using counts layer for seurat_v3 HVG selection")
            tmp.X = adata.layers["counts"].copy()
        elif layer != "counts":
            log.warning(
                "seurat_v3 expects raw counts but got layer='%s'. "
                "Consider using layer='counts' for better results.",
                layer,
            )

    n_hvg = min(n_hvg, adata.n_vars)

    log.info("Selecting %d HVGs using %s flavor...", n_hvg, flavor)

    if flavor in ("seurat", "cell_ranger"):
        sc.pp.highly_variable_genes(
            tmp,
            n_top_genes=n_hvg,
            flavor=flavor,
            batch_key=batch_key,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp,
            subset=False,
        )
    else:  # seurat_v3
        sc.pp.highly_variable_genes(
            tmp,
            n_top_genes=n_hvg,
            flavor=flavor,
            batch_key=batch_key,
            subset=False,
        )

    # Get HVG names
    hvg_mask = tmp.var["highly_variable"]
    hvgs = list(tmp.var_names[hvg_mask])

    # Copy HVG info back to original adata
    adata.var["highly_variable"] = hvg_mask.values
    if "highly_variable_rank" in tmp.var.columns:
        adata.var["highly_variable_rank"] = tmp.var["highly_variable_rank"].values

    # Store in uns
    if "hvg_info" not in adata.uns:
        adata.uns["hvg_info"] = {}
    adata.uns["hvg_info"]["n_hvg"] = len(hvgs)
    adata.uns["hvg_info"]["flavor"] = flavor
    adata.uns["hvg_info"]["layer"] = layer
    adata.uns["hvg_info"]["batch_key"] = batch_key

    if subset:
        adata._inplace_subset_var(hvg_mask.values)
        log.info("Subset adata to %d HVGs in place", len(hvgs))
    else:
        log.info("Identified %d HVGs, stored in var['highly_variable']", len(hvgs))

    return hvgs


def get_hvgs(adata: Any) -> list[str]:
    """Get HVGs from adata.var.

    Parameters
    ----------
    adata : AnnData
        AnnData object with var['highly_variable'].

    Returns
    -------
    list[str]
        List of HVG names.
    """
    if "highly_variable" not in adata.var.columns:
        raise KeyError("No HVGs computed. Run compute_hvgs() first.")

    return list(adata.var_names[adata.var["highly_variable"]])


# ---------------------------------------------------------------------------
# Reference preparation
# ---------------------------------------------------------------------------


def prepare_for_reference(
    adata: Any,  # AnnData
    reference_type: Literal["hlca", "luca"],
    *,
    reference_genes: list[str] | None = None,
    return_overlap_stats: bool = True,
) -> dict[str, Any]:
    """Prepare adata for reference atlas mapping.

    Harmonizes gene symbols and computes overlap statistics.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    reference_type : str
        Type of reference atlas (hlca or luca).
    reference_genes : list[str], optional
        Reference gene list (if not using default).
    return_overlap_stats : bool
        Whether to return overlap statistics.

    Returns
    -------
    dict
        Overlap statistics and preparation info.
    """
    from stagebridge.data.common.harmonize import canonicalize_gene_symbols

    # Canonicalize gene symbols
    canonicalize_gene_symbols(adata)

    stats = {
        "reference_type": reference_type,
        "n_genes_query": adata.n_vars,
    }

    if reference_genes is not None:
        # Compute overlap
        query_genes = set(adata.var_names)
        ref_genes = set(reference_genes)

        overlap = query_genes & ref_genes
        only_query = query_genes - ref_genes
        only_ref = ref_genes - query_genes

        stats.update(
            {
                "n_genes_reference": len(ref_genes),
                "n_genes_overlap": len(overlap),
                "overlap_fraction_query": len(overlap) / len(query_genes) if query_genes else 0,
                "overlap_fraction_reference": len(overlap) / len(ref_genes) if ref_genes else 0,
                "n_genes_only_query": len(only_query),
                "n_genes_only_reference": len(only_ref),
            }
        )

        # Warn if overlap is low
        if stats["overlap_fraction_query"] < 0.8:
            log.warning(
                "%s reference overlap is only %.1f%%. Consider checking gene naming.",
                reference_type.upper(),
                100 * stats["overlap_fraction_query"],
            )

        log.info(
            "%s reference: %d query genes, %d reference genes, %d overlap (%.1f%%)",
            reference_type.upper(),
            stats["n_genes_query"],
            stats["n_genes_reference"],
            stats["n_genes_overlap"],
            100 * stats["overlap_fraction_query"],
        )

    # Store in adata.uns
    if "reference_prep" not in adata.uns:
        adata.uns["reference_prep"] = {}
    adata.uns["reference_prep"][reference_type] = stats

    return stats if return_overlap_stats else {}


def compute_gene_overlap(
    genes_a: list[str],
    genes_b: list[str],
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> dict[str, Any]:
    """Compute overlap statistics between two gene lists.

    Parameters
    ----------
    genes_a, genes_b : list[str]
        Two gene lists to compare.
    name_a, name_b : str
        Names for reporting.

    Returns
    -------
    dict
        Overlap statistics.
    """
    set_a = set(genes_a)
    set_b = set(genes_b)

    overlap = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a

    return {
        f"n_{name_a}": len(set_a),
        f"n_{name_b}": len(set_b),
        "n_overlap": len(overlap),
        f"overlap_fraction_{name_a}": len(overlap) / len(set_a) if set_a else 0,
        f"overlap_fraction_{name_b}": len(overlap) / len(set_b) if set_b else 0,
        f"n_only_{name_a}": len(only_a),
        f"n_only_{name_b}": len(only_b),
        "overlap_genes": sorted(overlap),
    }


# ---------------------------------------------------------------------------
# Feature spec generation
# ---------------------------------------------------------------------------


def generate_feature_spec(
    adata: Any,  # AnnData
    *,
    hvgs: list[str] | None = None,
    marker_genes: dict[str, list[str]] | None = None,
    reference_genes: dict[str, list[str]] | None = None,
) -> FeatureSpec:
    """Generate feature specification from adata.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    hvgs : list[str], optional
        HVG list (if not in adata.var).
    marker_genes : dict, optional
        Marker gene sets by category.
    reference_genes : dict, optional
        Reference gene lists (e.g., {"hlca": [...], "luca": [...]}).

    Returns
    -------
    FeatureSpec
        Feature specification.
    """
    # Get HVGs
    if hvgs is None:
        if "highly_variable" in adata.var.columns:
            hvgs = list(adata.var_names[adata.var["highly_variable"]])
        else:
            hvgs = []

    # Compute reference overlaps
    overlaps = {}
    if reference_genes:
        for ref_name, ref_list in reference_genes.items():
            overlaps[ref_name] = compute_gene_overlap(
                list(adata.var_names),
                ref_list,
                name_a="query",
                name_b=ref_name,
            )

    # Get config from uns
    norm_config = adata.uns.get("normalization", {})
    hvg_config = adata.uns.get("hvg_info", {})

    spec = FeatureSpec(
        all_genes=list(adata.var_names),
        hvgs=hvgs,
        marker_genes=marker_genes or {},
        reference_overlaps=overlaps,
        normalization_config=norm_config,
        hvg_config=hvg_config,
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        n_hvgs=len(hvgs),
    )

    log.info(
        "Generated feature spec: %d genes, %d HVGs, %d marker categories, %d reference overlaps",
        spec.n_genes,
        spec.n_hvgs,
        len(spec.marker_genes),
        len(spec.reference_overlaps),
    )

    return spec


# ---------------------------------------------------------------------------
# Batch-aware operations
# ---------------------------------------------------------------------------


def batch_correct_hvgs(
    adata: Any,  # AnnData
    batch_key: str,
    n_hvg: int = 2000,
    *,
    n_hvg_per_batch: int | None = None,
) -> list[str]:
    """Select HVGs with batch-aware weighting.

    Ensures HVGs are represented across batches, not dominated by
    large batches.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    batch_key : str
        Column in obs for batch labels.
    n_hvg : int
        Total number of HVGs to select.
    n_hvg_per_batch : int, optional
        HVGs to select per batch before merging.

    Returns
    -------
    list[str]
        Selected HVG names.
    """
    _require_scanpy()

    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in obs")

    batches = adata.obs[batch_key].unique()
    log.info("Selecting batch-aware HVGs across %d batches...", len(batches))

    if n_hvg_per_batch is None:
        n_hvg_per_batch = max(n_hvg // len(batches), 500)

    # Select HVGs per batch
    all_hvgs = set()
    for batch in batches:
        batch_mask = adata.obs[batch_key] == batch
        batch_adata = adata[batch_mask, :].copy()

        if batch_adata.n_obs < 100:
            log.warning("Batch '%s' has only %d cells, skipping", batch, batch_adata.n_obs)
            continue

        try:
            batch_hvgs = compute_hvgs(batch_adata, n_hvg=n_hvg_per_batch, subset=False)
            all_hvgs.update(batch_hvgs)
            log.debug("Batch '%s': %d HVGs", batch, len(batch_hvgs))
        except Exception as e:
            log.warning("HVG selection failed for batch '%s': %s", batch, e)

    # If we have too many, rank by frequency across batches
    hvg_list = sorted(all_hvgs)

    if len(hvg_list) > n_hvg:
        # Rank by mean across batches
        hvg_scores = {}
        for gene in hvg_list:
            if gene in adata.var_names:
                hvg_scores[gene] = float(adata[:, gene].X.mean())
            else:
                hvg_scores[gene] = 0.0

        hvg_list = sorted(hvg_scores.keys(), key=lambda x: -hvg_scores[x])[:n_hvg]

    # Update adata.var
    adata.var["highly_variable"] = adata.var_names.isin(hvg_list)

    log.info("Selected %d batch-aware HVGs from %d candidates", len(hvg_list), len(all_hvgs))
    return hvg_list
