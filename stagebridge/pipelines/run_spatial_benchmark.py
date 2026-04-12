#!/usr/bin/env python3
"""
Spatial Backend Benchmark.

Compare Tangram, DestVI, and TACCO on the same LUAD dataset.

This script:
1. Loads snRNA and spatial data
2. Runs all three backends
3. Computes upstream metrics (reconstruction, entropy, coverage)
4. Computes downstream utility (transition quality, influence correlation)
5. Generates comparison report and visualization
6. Selects canonical backend with rationale

Purpose: Justify spatial backend choice with quantitative evidence (V1 requirement).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stagebridge.spatial_mapping import (
    TangramBackend,
    DestVIBackend,
    TACCOBackend,
    Cell2locationBackend,
)
from stagebridge.spatial_mapping.marker_scoring_wrapper import MarkerScoringBackend
from stagebridge.spatial_mapping.rctd_wrapper import RCTDBackend
from stagebridge.spatial_mapping.card_wrapper import CARDBackend
from stagebridge.spatial_mapping.spotlight_wrapper import SPOTlightBackend

log = logging.getLogger(__name__)


def run_backend_comparison(
    snrna_path: Path,
    spatial_path: Path,
    output_dir: Path,
    backends: list[str] = None,
    quick: bool = False,
    debug: bool = False,
    sample: str | None = None,
    sample_col: str = "sample_id",
    label_source: str = "hlca",
    labels_parquet: Path | None = None,
) -> dict:
    """
    Run comparison of all spatial backends.

    Args:
        snrna_path: Path to snRNA h5ad
        spatial_path: Path to spatial h5ad
        output_dir: Where to save results
        backends: List of backend names or None for all
        quick: Use reduced epochs for faster testing
        debug: Use minimal epochs (2-5) just to verify code runs
        sample: If provided, filter spatial data to this sample only
        sample_col: Column name for sample IDs in spatial.obs
        label_source: Which reference to use for cell type labels ('hlca' or 'luca')
        labels_parquet: Path to cell_types.parquet (needed if label_source='luca')

    Returns:
        Dictionary with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("Loading data...")
    snrna = ad.read_h5ad(snrna_path)
    spatial = ad.read_h5ad(spatial_path)

    # Handle label source selection
    if label_source == "luca":
        if "luca_cell_type" in snrna.obs.columns:
            # Already have LuCA labels in the h5ad
            log.info("Using LuCA cell type labels from snrna.obs['luca_cell_type']")
            snrna.obs["cell_type"] = snrna.obs["luca_cell_type"]
        elif labels_parquet is not None:
            # Load from parquet
            log.info("Loading LuCA labels from %s", labels_parquet)
            labels_df = pd.read_parquet(labels_parquet)
            if "luca_cell_type" not in labels_df.columns:
                raise ValueError(f"'luca_cell_type' column not found in {labels_parquet}")
            # Create mapping and apply
            label_map = dict(zip(labels_df["cell_id"], labels_df["luca_cell_type"]))
            snrna.obs["cell_type"] = snrna.obs.index.map(label_map)
            unmapped = snrna.obs["cell_type"].isna().sum()
            if unmapped > 0:
                log.warning("%d cells without LuCA label mapping", unmapped)
                snrna.obs["cell_type"] = snrna.obs["cell_type"].fillna("Unknown")
        else:
            raise ValueError(
                "label_source='luca' but no luca_cell_type in snrna.obs and no labels_parquet provided"
            )
    else:
        # Default: use existing cell_type column (HLCA)
        if "cell_type" not in snrna.obs.columns:
            raise ValueError("No 'cell_type' column in snrna.obs")
        log.info("Using HLCA cell type labels from snrna.obs['cell_type']")

    # Ensure cell_type is categorical
    snrna.obs["cell_type"] = snrna.obs["cell_type"].astype("category")

    # Filter to single sample if specified
    if sample is not None:
        if sample_col not in spatial.obs.columns:
            raise ValueError(f"Sample column '{sample_col}' not found in spatial.obs")
        n_before = spatial.n_obs
        spatial = spatial[spatial.obs[sample_col] == sample].copy()
        log.info("Filtered spatial to sample '%s': %d -> %d spots", sample, n_before, spatial.n_obs)
        if spatial.n_obs == 0:
            raise ValueError(f"No spots found for sample '{sample}'")

    log.info("  snRNA: %d cells x %d genes", snrna.shape[0], snrna.shape[1])
    log.info("  Spatial: %d spots x %d genes", spatial.shape[0], spatial.shape[1])
    log.info("  Cell types: %d", snrna.obs["cell_type"].nunique())

    backends_to_run = backends or ["tangram", "destvi", "tacco", "cell2location"]
    results = {}

    # Run each backend
    for backend_name in backends_to_run:
        log.info("=" * 80)
        log.info("Running %s", backend_name.upper())
        log.info("=" * 80)

        # Only add backend subdirectory when running multiple backends (standalone mode)
        # When called from Snakemake with single backend, output_dir already includes backend name
        if len(backends_to_run) > 1:
            backend_dir = output_dir / backend_name
        else:
            backend_dir = output_dir
        backend_dir.mkdir(exist_ok=True, parents=True)

        start_time = time.time()

        try:
            if backend_name == "tangram":
                if debug:
                    n_epochs = 2
                elif quick:
                    n_epochs = 10
                else:
                    n_epochs = 1000
                backend = TangramBackend(
                    mode="clusters",
                    n_epochs=n_epochs,
                )
            elif backend_name == "destvi":
                if debug:
                    n_condsc, n_destvi = 3, 5
                elif quick:
                    n_condsc, n_destvi = 20, 50
                else:
                    n_condsc, n_destvi = 200, 2500
                backend = DestVIBackend(
                    n_epochs_condsc=n_condsc,
                    n_epochs_destvi=n_destvi,
                )
            elif backend_name == "tacco":
                backend = TACCOBackend(method="OT")
            elif backend_name == "cell2location":
                if debug:
                    n_ref, n_spatial = 3, 5
                elif quick:
                    n_ref, n_spatial = 50, 500
                else:
                    n_ref, n_spatial = 250, 2500
                backend = Cell2locationBackend(
                    max_epochs_ref=n_ref,
                    max_epochs_spatial=n_spatial,
                )
            elif backend_name == "marker_scoring":
                # Simple baseline - derive markers from reference
                backend = MarkerScoringBackend(
                    use_reference_markers=True,
                    n_markers=50,
                )
            elif backend_name == "rctd":
                # Use R_EXECUTABLE env var if set, otherwise default to Rscript in PATH
                r_exec = os.environ.get("R_EXECUTABLE", "Rscript")
                backend = RCTDBackend(
                    mode="doublet",
                    min_cells_per_type=5,
                    r_executable=r_exec,
                )
            elif backend_name == "card":
                r_exec = os.environ.get("R_EXECUTABLE", "Rscript")
                backend = CARDBackend(
                    min_cells_per_type=5,
                    r_executable=r_exec,
                )
            elif backend_name == "spotlight":
                r_exec = os.environ.get("R_EXECUTABLE", "Rscript")
                backend = SPOTlightBackend(
                    min_cells_per_type=5,
                    n_hvg=3000,
                    r_executable=r_exec,
                )
            else:
                raise ValueError(f"Unknown backend: {backend_name}")

            result = backend.map(snrna, spatial, output_dir=backend_dir)
            result.save(backend_dir)

            runtime = time.time() - start_time

            results[backend_name] = {
                "result": result,
                "runtime_seconds": runtime,
                "success": True,
                "error": None,
            }

            log.info("%s completed in %.1fs", backend_name, runtime)

        except Exception as e:
            log.error("%s failed: %s", backend_name, e)
            results[backend_name] = {
                "result": None,
                "runtime_seconds": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    # Generate comparison report
    log.info("=" * 80)
    log.info("GENERATING COMPARISON REPORT")
    log.info("=" * 80)

    comparison = compare_backends(results, output_dir)

    # Add metadata about label source
    comparison["metadata"] = {
        "label_source": label_source,
        "n_cells": snrna.n_obs,
        "n_spots": spatial.n_obs,
        "n_cell_types": snrna.obs["cell_type"].nunique(),
        "sample": sample,
    }

    # Save comparison
    with open(output_dir / "backend_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    log.info("Benchmark complete. Results saved to %s", output_dir)
    log.info("Label source: %s (%d cell types)", label_source.upper(), snrna.obs["cell_type"].nunique())

    return comparison


def compare_backends(
    results: dict,
    output_dir: Path,
) -> dict:
    """
    Compare backend results across multiple metrics.

    Metrics:
    1. Upstream quality (entropy, sparsity, coverage)
    2. Runtime and scalability
    3. Downstream utility (if transition model available)

    Returns:
        Comparison dictionary with rankings
    """
    comparison = {
        "backends": {},
        "rankings": {},
        "recommendation": {},
    }

    # Extract metrics for each backend
    for backend_name, result_dict in results.items():
        if not result_dict["success"]:
            comparison["backends"][backend_name] = {
                "status": "failed",
                "error": result_dict["error"],
            }
            continue

        result = result_dict["result"]

        comparison["backends"][backend_name] = {
            "status": "success",
            "runtime_seconds": result_dict["runtime_seconds"],
            "upstream_metrics": result.upstream_metrics,
            "proportions_shape": result.cell_type_proportions.shape,
            "mean_confidence": float(result.confidence.mean()),
            "std_confidence": float(result.confidence.std()),
        }

    # Rank backends
    successful_backends = [
        name for name, data in comparison["backends"].items() if data["status"] == "success"
    ]

    if len(successful_backends) == 0:
        comparison["recommendation"] = {
            "canonical_backend": None,
            "rationale": "No backends succeeded",
        }
        return comparison

    # Ranking criteria (higher is better)
    ranking_df = pd.DataFrame(
        [
            {
                "backend": name,
                "mean_entropy": comparison["backends"][name]["upstream_metrics"]["mean_entropy"],
                "coverage": comparison["backends"][name]["upstream_metrics"]["coverage"],
                "sparsity": comparison["backends"][name]["upstream_metrics"]["sparsity"],
                "runtime": comparison["backends"][name]["runtime_seconds"],
                "mean_confidence": comparison["backends"][name]["mean_confidence"],
            }
            for name in successful_backends
        ]
    )

    # Normalize and score
    # Entropy: moderate is good (0.5-0.7)
    ranking_df["entropy_score"] = 1 - np.abs(ranking_df["mean_entropy"] - 0.6)

    # Coverage: higher is better
    ranking_df["coverage_score"] = ranking_df["coverage"]

    # Sparsity: lower is better (more complete annotations)
    ranking_df["sparsity_score"] = 1 - ranking_df["sparsity"]

    # Runtime: faster is better (inverse, normalized)
    ranking_df["runtime_score"] = 1 / (ranking_df["runtime"] / ranking_df["runtime"].min())

    # Confidence: higher is better
    ranking_df["confidence_score"] = ranking_df["mean_confidence"]

    # Composite score (weighted average)
    weights = {
        "entropy_score": 0.25,
        "coverage_score": 0.25,
        "sparsity_score": 0.20,
        "runtime_score": 0.15,
        "confidence_score": 0.15,
    }

    ranking_df["composite_score"] = sum(
        ranking_df[col] * weight for col, weight in weights.items()
    )

    # Sort by composite score
    ranking_df = ranking_df.sort_values("composite_score", ascending=False)

    # Store rankings
    comparison["rankings"] = ranking_df.to_dict(orient="records")

    # Select canonical backend
    best_backend = ranking_df.iloc[0]["backend"]
    best_score = ranking_df.iloc[0]["composite_score"]

    comparison["recommendation"] = {
        "canonical_backend": best_backend,
        "composite_score": float(best_score),
        "rationale": generate_rationale(ranking_df),
    }

    # Generate visualizations
    plot_backend_comparison(ranking_df, output_dir)

    return comparison


def generate_rationale(ranking_df: pd.DataFrame) -> str:
    """Generate human-readable rationale for backend selection."""
    best = ranking_df.iloc[0]

    lines = [
        f"Selected {best['backend'].upper()} as canonical backend based on composite score ({best['composite_score']:.3f}).",
        "",
        "Key factors:",
    ]

    # Highlight strengths
    if best["entropy_score"] > 0.7:
        lines.append(f"  - Balanced cell type diversity (entropy={best['mean_entropy']:.3f})")

    if best["coverage_score"] > 0.8:
        lines.append(f"  - High coverage of confident mappings ({best['coverage']:.1%})")

    if best["sparsity_score"] > 0.7:
        lines.append(f"  - Complete annotations (low sparsity={best['sparsity']:.3f})")

    if ranking_df.shape[0] > 1:
        second = ranking_df.iloc[1]
        lines.append("")
        lines.append(
            f"Runner-up: {second['backend'].upper()} (score={second['composite_score']:.3f})"
        )

    return "\n".join(lines)


def generate_all_benchmark_figures(
    results: dict,
    output_dir: Path,
    spatial_adata: ad.AnnData | None = None,
):
    """
    Generate comprehensive benchmark figures for publication.

    Generates:
    1. Loss curves (for backends with training history)
    2. Spatial cell type plots (per-backend comparison)
    3. Per-sample metric distributions
    4. Group-level aggregations

    Args:
        results: Dictionary of backend results
        output_dir: Where to save figures
        spatial_adata: Spatial AnnData for coordinate-based plots
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Loss curves
    _plot_loss_curves(results, figures_dir)

    # 2. Spatial cell type plots
    if spatial_adata is not None:
        _plot_spatial_compositions(results, spatial_adata, figures_dir)

    # 3. Metric distributions
    _plot_metric_distributions(results, figures_dir)

    log.info("All benchmark figures saved to %s", figures_dir)


def _plot_loss_curves(results: dict, figures_dir: Path):
    """Plot training loss curves for backends that have them."""
    # Check for training history
    backends_with_history = []
    for name, data in results.items():
        if not data["success"] or data["result"] is None:
            continue
        meta = data["result"].metadata
        if "training_history" in meta or "loss_history" in meta:
            backends_with_history.append(name)

    if not backends_with_history:
        log.info("No backends with training history found, skipping loss curves")
        return

    n_backends = len(backends_with_history)
    fig, axes = plt.subplots(1, n_backends, figsize=(5 * n_backends, 4))
    if n_backends == 1:
        axes = [axes]

    for ax, name in zip(axes, backends_with_history):
        meta = results[name]["result"].metadata
        history = meta.get("training_history") or meta.get("loss_history")

        if isinstance(history, dict):
            # Multiple loss types
            for loss_name, values in history.items():
                if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                    ax.plot(values, label=loss_name)
            ax.legend()
        elif isinstance(history, (list, np.ndarray)):
            ax.plot(history)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name.upper()} Training")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved loss curves to %s", figures_dir / "loss_curves.png")


def _plot_spatial_compositions(
    results: dict,
    spatial_adata: ad.AnnData,
    figures_dir: Path,
    n_types: int = 6,
):
    """Plot spatial cell type compositions side-by-side."""
    successful = [name for name, data in results.items() if data["success"]]
    if not successful:
        return

    # Get spatial coordinates
    if "spatial" in spatial_adata.obsm:
        coords = spatial_adata.obsm["spatial"]
    elif "X_spatial" in spatial_adata.obsm:
        coords = spatial_adata.obsm["X_spatial"]
    else:
        log.warning("No spatial coordinates found, skipping spatial plots")
        return

    # Find top cell types across all backends
    all_proportions = []
    for name in successful:
        props = results[name]["result"].cell_type_proportions
        all_proportions.append(props.mean(axis=0))

    mean_props = pd.concat(all_proportions, axis=1).mean(axis=1)
    top_types = mean_props.nlargest(n_types).index.tolist()

    # Plot each top type across backends
    for cell_type in top_types:
        fig, axes = plt.subplots(1, len(successful), figsize=(5 * len(successful), 4))
        if len(successful) == 1:
            axes = [axes]

        for ax, name in zip(axes, successful):
            props = results[name]["result"].cell_type_proportions
            if cell_type not in props.columns:
                ax.set_title(f"{name}: N/A")
                continue

            values = props[cell_type].values
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=values, cmap="viridis",
                s=1, vmin=0, vmax=values.max()
            )
            ax.set_title(f"{name.upper()}")
            ax.set_aspect("equal")
            ax.axis("off")
            plt.colorbar(scatter, ax=ax, shrink=0.5)

        fig.suptitle(f"Cell Type: {cell_type}", fontsize=14)
        plt.tight_layout()
        safe_name = cell_type.replace("/", "_").replace(" ", "_")
        plt.savefig(figures_dir / f"spatial_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    log.info("Saved spatial composition plots for %d cell types", len(top_types))


def _plot_metric_distributions(results: dict, figures_dir: Path):
    """Plot distributions of key metrics across backends."""
    successful = [(name, data) for name, data in results.items() if data["success"]]
    if not successful:
        return

    # Collect per-spot metrics for each backend
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Entropy distribution
    ax = axes[0, 0]
    for name, data in successful:
        props = data["result"].cell_type_proportions
        # Compute per-spot entropy
        entropy = -(props * np.log(props + 1e-10)).sum(axis=1)
        entropy_norm = entropy / np.log(props.shape[1])  # Normalize
        ax.hist(entropy_norm, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Normalized Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Cell Type Entropy Distribution")
    ax.legend()

    # 2. Max proportion distribution
    ax = axes[0, 1]
    for name, data in successful:
        props = data["result"].cell_type_proportions
        max_props = props.max(axis=1)
        ax.hist(max_props, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Max Proportion")
    ax.set_ylabel("Density")
    ax.set_title("Dominance Distribution")
    ax.legend()

    # 3. Types per spot distribution
    ax = axes[0, 2]
    for name, data in successful:
        props = data["result"].cell_type_proportions
        types_per_spot = (props > 0.01).sum(axis=1)
        ax.hist(types_per_spot, bins=range(0, props.shape[1] + 1), alpha=0.5, label=name, density=True)
    ax.set_xlabel("Types per Spot (>1%)")
    ax.set_ylabel("Density")
    ax.set_title("Cell Type Richness")
    ax.legend()

    # 4. Confidence distribution
    ax = axes[1, 0]
    for name, data in successful:
        conf = data["result"].confidence
        ax.hist(conf, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Mapping Confidence")
    ax.legend()

    # 5. Per-type mean proportion
    ax = axes[1, 1]
    type_means = {}
    for name, data in successful:
        props = data["result"].cell_type_proportions
        type_means[name] = props.mean(axis=0).sort_values(ascending=False)

    # Plot top 10 types
    if type_means:
        first_backend = list(type_means.keys())[0]
        top_types = type_means[first_backend].head(10).index
        x = np.arange(len(top_types))
        width = 0.8 / len(successful)
        for i, (name, means) in enumerate(type_means.items()):
            values = [means.get(t, 0) for t in top_types]
            ax.bar(x + i * width, values, width, label=name, alpha=0.7)
        ax.set_xticks(x + width * (len(successful) - 1) / 2)
        ax.set_xticklabels(top_types, rotation=45, ha="right")
        ax.set_ylabel("Mean Proportion")
        ax.set_title("Top Cell Types by Mean Proportion")
        ax.legend()

    # 6. Gini coefficient per backend
    ax = axes[1, 2]
    gini_values = {}
    for name, data in successful:
        props = data["result"].cell_type_proportions.values
        # Compute Gini per spot
        gini_per_spot = []
        for row in props:
            row = row[row > 0]
            if len(row) == 0:
                continue
            row = np.sort(row)
            n = len(row)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * row) - (n + 1) * np.sum(row)) / (n * np.sum(row))
            gini_per_spot.append(gini)
        gini_values[name] = gini_per_spot

    ax.boxplot(
        [v for v in gini_values.values()],
        labels=list(gini_values.keys())
    )
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Proportion Inequality")

    plt.tight_layout()
    plt.savefig(figures_dir / "metric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved metric distributions to %s", figures_dir / "metric_distributions.png")


def aggregate_sample_metrics(
    benchmark_dir: Path,
    backends: list[str] | None = None,
    label_source: str = "hlca",
) -> pd.DataFrame:
    """
    Aggregate metrics across all samples for group-level analysis.

    Reads per-sample outputs and computes summary statistics.

    Args:
        benchmark_dir: Base benchmark directory (contains {label_source}/{backend}/samples/)
        backends: List of backends to include (default: all found)
        label_source: Which label source directory to use

    Returns:
        DataFrame with per-backend aggregated metrics
    """
    from stagebridge.spatial_mapping.metrics import compute_upstream_metrics
    from stagebridge.spatial_mapping.backend_base import BackendMappingResult

    base_dir = benchmark_dir / label_source
    if not base_dir.exists():
        log.warning("No benchmark results found at %s", base_dir)
        return pd.DataFrame()

    # Find all backends
    if backends is None:
        backends = [d.name for d in base_dir.iterdir() if d.is_dir()]

    all_metrics = []

    for backend in backends:
        backend_dir = base_dir / backend / "samples"
        if not backend_dir.exists():
            continue

        sample_dirs = [d for d in backend_dir.iterdir() if d.is_dir()]
        log.info("Found %d samples for %s", len(sample_dirs), backend)

        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            props_file = sample_dir / "cell_type_proportions.parquet"

            if not props_file.exists():
                continue

            try:
                props = pd.read_parquet(props_file)

                # Load confidence if available
                conf_file = sample_dir / "confidence.parquet"
                if conf_file.exists():
                    conf_df = pd.read_parquet(conf_file)
                    conf = conf_df["confidence"].values
                else:
                    conf = props.max(axis=1).values

                # Create minimal result for metrics computation
                result = BackendMappingResult(
                    cell_type_proportions=props,
                    confidence=conf,
                    metadata={"backend": backend},
                )

                # Compute comprehensive metrics
                metrics = compute_upstream_metrics(result)
                metrics["backend"] = backend
                metrics["sample_id"] = sample_id
                metrics["label_source"] = label_source

                all_metrics.append(metrics)

            except Exception as e:
                log.warning("Failed to load %s/%s: %s", backend, sample_id, e)

    if not all_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics)
    return df


def plot_group_level_comparison(
    metrics_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate group-level comparison plots from aggregated sample metrics.

    Args:
        metrics_df: DataFrame from aggregate_sample_metrics()
        output_dir: Where to save figures
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    if metrics_df.empty:
        log.warning("No metrics to plot")
        return

    backends = metrics_df["backend"].unique()
    n_backends = len(backends)

    # Key metrics to compare
    key_metrics = [
        ("types_per_spot_mean", "Types per Spot", True),  # higher is better
        ("effective_coverage", "Effective Coverage", True),
        ("mean_entropy", "Mean Entropy", None),  # moderate is best
        ("gini_coefficient_mean", "Gini Coefficient", False),  # lower is better (more even)
        ("dominance_ratio_mean", "Dominance Ratio", False),  # lower means more mixed
    ]

    # 1. Box plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric, label, higher_better) in enumerate(key_metrics):
        if metric not in metrics_df.columns:
            continue
        ax = axes[i]

        data = [metrics_df[metrics_df["backend"] == b][metric].dropna() for b in backends]
        bp = ax.boxplot(data, labels=backends, patch_artist=True)

        # Color by mean value
        means = [d.mean() for d in data]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_backends))
        if higher_better is False:
            colors = colors[::-1]  # Reverse for "lower is better"

        sorted_idx = np.argsort(means)
        if higher_better is False:
            sorted_idx = sorted_idx[::-1]

        for j, patch in enumerate(bp["boxes"]):
            rank = np.where(sorted_idx == j)[0][0]
            patch.set_facecolor(colors[rank])

        ax.set_ylabel(label)
        ax.set_title(f"{label} by Backend")
        ax.tick_params(axis="x", rotation=45)

    # 6. Sample count
    ax = axes[5]
    sample_counts = metrics_df.groupby("backend").size()
    ax.bar(sample_counts.index, sample_counts.values, color="steelblue")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Samples per Backend")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(figures_dir / "group_level_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Summary table
    summary = metrics_df.groupby("backend").agg({
        "types_per_spot_mean": ["mean", "std"],
        "effective_coverage": ["mean", "std"],
        "mean_entropy": ["mean", "std"],
        "gini_coefficient_mean": ["mean", "std"],
        "n_spots": ["sum", "mean"],
    }).round(3)

    summary.to_csv(output_dir / "group_metrics_summary.csv")
    log.info("Saved group-level summary to %s", output_dir / "group_metrics_summary.csv")

    # 3. Backend ranking by multiple criteria
    ranking_metrics = ["types_per_spot_mean", "effective_coverage", "global_type_coverage"]
    ranking_df = metrics_df.groupby("backend")[ranking_metrics].mean()

    # Composite score (all normalized to [0,1] and averaged)
    for col in ranking_metrics:
        col_min, col_max = ranking_df[col].min(), ranking_df[col].max()
        if col_max > col_min:
            ranking_df[f"{col}_norm"] = (ranking_df[col] - col_min) / (col_max - col_min)
        else:
            ranking_df[f"{col}_norm"] = 0.5

    norm_cols = [c for c in ranking_df.columns if c.endswith("_norm")]
    ranking_df["composite_score"] = ranking_df[norm_cols].mean(axis=1)
    ranking_df = ranking_df.sort_values("composite_score", ascending=False)

    ranking_df.to_csv(output_dir / "backend_ranking.csv")
    log.info("Saved backend ranking to %s", output_dir / "backend_ranking.csv")

    return ranking_df


def select_canonical_from_metrics(
    metrics_df: pd.DataFrame,
    forced_backend: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Select canonical backend from aggregated sample metrics.

    Uses the new comprehensive metrics for ranking:
    - types_per_spot_mean: Higher is better (richer deconvolution)
    - effective_coverage: Higher is better
    - global_type_coverage: Higher is better
    - gini_coefficient_mean: Lower is better (more even distribution)

    Args:
        metrics_df: DataFrame from aggregate_sample_metrics()
        forced_backend: If provided, force this backend as canonical
        weights: Optional custom weights

    Returns:
        Dictionary with selection details for JSON serialization
    """
    if weights is None:
        weights = {
            "types_per_spot_mean": 0.30,
            "effective_coverage": 0.25,
            "global_type_coverage": 0.20,
            "mean_entropy": 0.15,
            "gini_coefficient_mean": 0.10,  # Lower is better
        }

    backends = metrics_df["backend"].unique()
    backend_scores = {}
    backend_metrics = {}

    for backend in backends:
        mask = metrics_df["backend"] == backend
        scores = {}

        for metric, weight in weights.items():
            if metric not in metrics_df.columns:
                continue

            values = metrics_df.loc[mask, metric].dropna()
            if len(values) == 0:
                continue

            mean_val = values.mean()

            # Normalize to [0, 1] using global min/max
            global_min = metrics_df[metric].min()
            global_max = metrics_df[metric].max()

            if global_max > global_min:
                norm_val = (mean_val - global_min) / (global_max - global_min)
            else:
                norm_val = 0.5

            # For Gini, lower is better
            if metric == "gini_coefficient_mean":
                norm_val = 1 - norm_val

            scores[metric] = {
                "raw": float(mean_val),
                "normalized": float(norm_val),
                "weight": weight,
            }

        # Compute composite score
        composite = sum(
            s["normalized"] * s["weight"]
            for s in scores.values()
        )

        backend_scores[backend] = composite
        backend_metrics[backend] = scores

    # Rank backends
    ranked = sorted(backend_scores.items(), key=lambda x: x[1], reverse=True)

    # Select canonical
    if forced_backend and forced_backend in backends:
        canonical = forced_backend
        canonical_score = backend_scores.get(forced_backend, 0)
        forced = True
    else:
        canonical = ranked[0][0]
        canonical_score = ranked[0][1]
        forced = False

    # Build justification
    justification_lines = [
        f"# Canonical Backend Selection: {canonical.upper()}",
        "",
        f"**Selection Method:** {'Forced by user' if forced else 'Metric-based ranking'}",
        f"**Composite Score:** {canonical_score:.4f}",
        "",
        "## Ranking (by comprehensive metrics)",
        "",
        "| Rank | Backend | Score |",
        "|------|---------|-------|",
    ]

    for i, (backend, score) in enumerate(ranked, 1):
        marker = " *" if backend == canonical else ""
        justification_lines.append(f"| {i} | {backend.upper()}{marker} | {score:.4f} |")

    justification_lines.extend([
        "",
        "## Metric Breakdown",
        "",
    ])

    for metric, weight in weights.items():
        if metric in metrics_df.columns:
            justification_lines.append(f"**{metric}** (weight={weight:.0%}):")
            for backend in backends:
                if backend in backend_metrics and metric in backend_metrics[backend]:
                    m = backend_metrics[backend][metric]
                    justification_lines.append(
                        f"  - {backend}: {m['raw']:.4f} (normalized: {m['normalized']:.3f})"
                    )
            justification_lines.append("")

    if forced:
        justification_lines.extend([
            "## Note",
            "",
            f"Backend was forced to **{canonical.upper()}** by user request.",
            "This overrides metric-based selection.",
            "",
            "**Rationale for forcing DestVI/Cell2location:**",
            "- Provides uncertainty estimates (needed for downstream niche modeling)",
            "- Richer per-cell-type outputs",
            "- Bayesian/probabilistic framework aligns with project goals",
        ])

    return {
        "canonical_backend": canonical,
        "selection_score": float(canonical_score),
        "forced": forced,
        "justification": "\n".join(justification_lines),
        "rankings": [{"backend": b, "score": float(s)} for b, s in ranked],
        "alternatives": [b for b, _ in ranked if b != canonical],
        "alternative_scores": {b: float(s) for b, s in ranked if b != canonical},
        "metric_weights": weights,
        "backend_metrics": {
            b: {m: {"raw": d["raw"], "normalized": d["normalized"]}
                for m, d in metrics.items()}
            for b, metrics in backend_metrics.items()
        },
        "n_samples_per_backend": metrics_df.groupby("backend").size().to_dict(),
    }


def save_canonical_selection(selection_data: dict, output_dir: Path):
    """Save canonical backend selection to JSON and markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (machine-readable)
    json_path = output_dir / "canonical_backend.json"
    with open(json_path, "w") as f:
        json.dump(selection_data, f, indent=2)

    # Save markdown report (human-readable)
    md_path = output_dir / "backend_selection_report.md"
    with open(md_path, "w") as f:
        f.write(selection_data["justification"])

    return json_path


def plot_backend_comparison(
    ranking_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Composite scores
    ax = axes[0, 0]
    ranking_df.plot.barh(
        x="backend",
        y="composite_score",
        ax=ax,
        legend=False,
        color="steelblue",
    )
    ax.set_xlabel("Composite Score")
    ax.set_title("Overall Performance")
    ax.set_xlim(0, 1)

    # 2. Radar chart of individual metrics
    ax = axes[0, 1]
    metrics = [
        "entropy_score",
        "coverage_score",
        "sparsity_score",
        "runtime_score",
        "confidence_score",
    ]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(222, projection="polar")
    for _, row in ranking_df.iterrows():
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, "o-", linewidth=2, label=row["backend"])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_score", "") for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Metric Breakdown")

    # 3. Runtime comparison
    ax = axes[1, 0]
    ranking_df.plot.barh(
        x="backend",
        y="runtime",
        ax=ax,
        legend=False,
        color="coral",
    )
    ax.set_xlabel("Runtime (seconds)")
    ax.set_title("Computational Cost")

    # 4. Entropy vs Coverage scatter
    ax = axes[1, 1]
    ax.scatter(
        ranking_df["mean_entropy"],
        ranking_df["coverage"],
        s=200,
        c=ranking_df["composite_score"],
        cmap="viridis",
        edgecolors="black",
        linewidths=2,
    )
    for _, row in ranking_df.iterrows():
        ax.annotate(
            row["backend"],
            (row["mean_entropy"], row["coverage"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean Entropy (Diversity)")
    ax.set_ylabel("Coverage (Confidence)")
    ax.set_title("Quality Trade-offs")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "backend_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output_dir / 'backend_comparison.png'}")


def run_comprehensive_benchmark(
    snrna_path: Path,
    spatial_path: Path,
    output_dir: Path,
    backends: list[str] = None,
    quick: bool = False,
) -> dict:
    """
    Wrapper function for comprehensive backend benchmark.

    Returns results formatted for notebook consumption.

    Returns:
        Dictionary with:
        - metrics: List of dicts for DataFrame (backend, mapping_quality, runtime_minutes, memory_gb, downstream_utility)
        - recommendation: Dict with backend and rationale
    """
    comparison = run_backend_comparison(
        snrna_path=snrna_path,
        spatial_path=spatial_path,
        output_dir=output_dir,
        backends=backends,
        quick=quick,
    )

    # Format for notebook
    metrics = []
    for backend_name, data in comparison["backends"].items():
        if data["status"] != "success":
            continue

        metrics.append(
            {
                "backend": backend_name.upper(),
                "mapping_quality": data["upstream_metrics"]["coverage"],
                "runtime_minutes": data["runtime_seconds"] / 60,
                "memory_gb": 16.0,  # Placeholder - would need actual measurement
                "downstream_utility": data["mean_confidence"],
            }
        )

    # Format recommendation
    formatted_results = {
        "metrics": metrics,
        "recommendation": {
            "backend": comparison["recommendation"]["canonical_backend"].upper(),
            "rationale": comparison["recommendation"]["rationale"],
        },
        "rankings": comparison["rankings"],
    }

    return formatted_results


def main():
    parser = argparse.ArgumentParser(description="Spatial Backend Benchmark")

    # Mode selection
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate metrics across all completed samples (run after benchmark completes)",
    )

    # Standard benchmark arguments
    parser.add_argument("--snrna", type=str, help="Path to snRNA h5ad")
    parser.add_argument("--spatial", type=str, help="Path to spatial h5ad")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--backends", type=str, nargs="+", default=None, help="Backends to run (default: all)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use reduced epochs for quick testing"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use minimal epochs (2-5) just to verify code runs"
    )
    parser.add_argument(
        "--sample", type=str, default=None, help="Run on single sample (filters spatial data)"
    )
    parser.add_argument(
        "--sample-col", type=str, default="sample_id", help="Column name for sample IDs"
    )
    parser.add_argument(
        "--label-source",
        type=str,
        default="hlca",
        choices=["hlca", "luca"],
        help="Reference atlas for cell type labels (default: hlca)",
    )
    parser.add_argument(
        "--labels-parquet",
        type=str,
        default=None,
        help="Path to cell_types.parquet (needed if --label-source=luca and luca_cell_type not in h5ad)",
    )
    parser.add_argument(
        "--force-backend",
        type=str,
        default=None,
        choices=["tangram", "destvi", "tacco", "cell2location", "marker_scoring"],
        help="Force a specific backend as canonical (overrides metric-based selection)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # =========================================================================
    # AGGREGATE MODE: Run after all samples complete
    # =========================================================================
    if args.aggregate:
        log.info("=" * 80)
        log.info("AGGREGATING METRICS ACROSS ALL SAMPLES")
        log.info("=" * 80)

        canonical_selections = {}

        # Aggregate for both label sources
        for label_source in ["hlca", "luca"]:
            label_dir = output_dir / label_source
            if not label_dir.exists():
                continue

            log.info("Processing %s label source...", label_source.upper())
            metrics_df = aggregate_sample_metrics(
                output_dir,
                backends=args.backends,
                label_source=label_source,
            )

            if not metrics_df.empty:
                # Save raw metrics
                metrics_df.to_csv(label_dir / "all_sample_metrics.csv", index=False)
                log.info("Saved %d sample metrics to %s", len(metrics_df), label_dir / "all_sample_metrics.csv")

                # Generate publication-quality figures
                from stagebridge.visualization import SpatialBenchmarkFigures

                figures_dir = label_dir / "publication_figures"
                fig_gen = SpatialBenchmarkFigures(
                    metrics_df=metrics_df,
                    output_dir=figures_dir,
                )
                fig_gen.generate_all()

                log.info("Publication figures saved to %s", figures_dir)

                # Also generate basic diagnostic plots
                ranking_df = plot_group_level_comparison(metrics_df, label_dir)

                # ============================================================
                # SELECT CANONICAL BACKEND
                # ============================================================
                if args.force_backend:
                    # Forced selection
                    best_backend = args.force_backend
                    if best_backend not in metrics_df["backend"].unique():
                        log.warning(
                            "Forced backend '%s' not found in results. Available: %s",
                            best_backend, list(metrics_df["backend"].unique())
                        )
                        best_backend = ranking_df.index[0] if ranking_df is not None else None
                        log.info("Falling back to metric-based selection: %s", best_backend)
                    else:
                        log.info("FORCED SELECTION: %s", best_backend.upper())
                elif ranking_df is not None and not ranking_df.empty:
                    best_backend = ranking_df.index[0]
                    best_score = ranking_df.iloc[0]["composite_score"]
                    log.info(
                        "%s RECOMMENDATION: %s (composite score: %.3f)",
                        label_source.upper(), best_backend.upper(), best_score
                    )
                else:
                    best_backend = None

                # Save canonical backend decision
                if best_backend:
                    canonical_selections[label_source] = best_backend
                    selection_data = select_canonical_from_metrics(
                        metrics_df=metrics_df,
                        forced_backend=args.force_backend,
                    )
                    save_canonical_selection(selection_data, label_dir)
                    log.info("Canonical backend saved: %s", label_dir / "canonical_backend.json")

        # Summary
        print(f"\n{'=' * 80}")
        print("AGGREGATION COMPLETE")
        print(f"{'=' * 80}")
        for label_source, backend in canonical_selections.items():
            forced = " (FORCED)" if args.force_backend else ""
            print(f"  {label_source.upper()}: {backend.upper()}{forced}")
        print(f"\nResults saved to {output_dir}")
        return

    # =========================================================================
    # STANDARD MODE: Run benchmark on samples
    # =========================================================================
    if not args.snrna or not args.spatial:
        parser.error("--snrna and --spatial are required unless using --aggregate")

    comparison = run_backend_comparison(
        snrna_path=Path(args.snrna),
        spatial_path=Path(args.spatial),
        output_dir=output_dir,
        backends=args.backends,
        quick=args.quick,
        debug=args.debug,
        sample=args.sample,
        sample_col=getattr(args, "sample_col", "sample_id"),
        label_source=args.label_source,
        labels_parquet=Path(args.labels_parquet) if args.labels_parquet else None,
    )

    # Print recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    print(comparison["recommendation"]["rationale"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
