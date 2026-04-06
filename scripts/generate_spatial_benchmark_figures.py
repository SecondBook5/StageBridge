#!/usr/bin/env python3
"""
Generate publication figures for spatial backend benchmark.

This script creates all visualizations comparing spatial deconvolution backends
(Tangram, DestVI, TACCO, Cell2location) including:

1. Spatial maps showing predicted cell type proportions per backend
2. Backend comparison across abundant/medium/rare cell types
3. Per-cell-type accuracy analysis (which backend is best for which cell type)
4. Upstream metrics comparison (entropy, sparsity, coverage)
5. Cross-backend consistency analysis

Usage:
    python scripts/generate_spatial_benchmark_figures.py \
        --benchmark-dir /scratch/chaunzt1/stagebridge/runs/spatial_benchmark \
        --spatial-h5ad /scratch/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad \
        --output-dir figures/spatial_benchmark \
        --label-source hlca \
        --sample GSM9226178_P5_AIS  # Optional: specific sample, or 'all' for aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# StageBridge visualization imports
from stagebridge.spatial_mapping.viz_utils import (
    plot_backend_comparison_spatial,
    plot_entropy_vs_sparsity,
    plot_proportion_heatmap,
    plot_dominant_cell_type_map,
    plot_multi_backend_radar,
    plot_cell_type_colocalization,
)
from stagebridge.spatial_mapping.abundance_viz import (
    plot_backend_comparison_boxplot,
    plot_backend_comparison_heatmap,
)
from stagebridge.viz import configure_lungpca_style

# Configure publication style
try:
    configure_lungpca_style()
except:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
    })

BACKENDS = ["tangram", "destvi", "tacco", "cell2location", "marker_scoring"]
BACKEND_COLORS = {
    "tangram": "#1f77b4",
    "destvi": "#ff7f0e",
    "tacco": "#2ca02c",
    "cell2location": "#d62728",
    "marker_scoring": "#9467bd",
}


def load_backend_results(
    benchmark_dir: Path,
    label_source: str,
    sample: str,
) -> dict[str, dict]:
    """Load results from all backends for a specific sample."""
    results = {}

    for backend in BACKENDS:
        sample_dir = benchmark_dir / label_source / backend / "samples" / sample

        if not sample_dir.exists():
            print(f"  Skipping {backend} - no results for {sample}")
            continue

        props_path = sample_dir / "cell_type_proportions.parquet"
        metrics_path = sample_dir / "upstream_metrics.json"

        if props_path.exists():
            props = pd.read_parquet(props_path)
            results[backend] = {"proportions": props}

            if metrics_path.exists():
                with open(metrics_path) as f:
                    results[backend]["metrics"] = json.load(f)

            print(f"  Loaded {backend}: {props.shape[0]} spots, {props.shape[1]} cell types")
        else:
            print(f"  Skipping {backend} - no proportions file")

    return results


def get_cell_types_by_abundance(
    results: dict[str, dict],
    n_per_category: int = 5,
) -> dict[str, list[str]]:
    """Categorize cell types by abundance (abundant/medium/rare)."""
    # Aggregate proportions across backends
    all_props = []
    for backend_data in results.values():
        all_props.append(backend_data["proportions"])

    combined = pd.concat(all_props, axis=0)
    mean_abundance = combined.mean(axis=0).sort_values(ascending=False)

    n_types = len(mean_abundance)
    abundant_cutoff = n_types // 3
    rare_cutoff = 2 * n_types // 3

    return {
        "abundant": mean_abundance.iloc[:abundant_cutoff].index[:n_per_category].tolist(),
        "medium": mean_abundance.iloc[abundant_cutoff:rare_cutoff].index[:n_per_category].tolist(),
        "rare": mean_abundance.iloc[rare_cutoff:].index[:n_per_category].tolist(),
    }


def compute_backend_agreement(results: dict[str, dict]) -> pd.DataFrame:
    """
    Compute pairwise agreement between backends for each cell type.

    Returns DataFrame with correlation between each backend pair per cell type.
    """
    backends = list(results.keys())
    props_dict = {b: results[b]["proportions"] for b in backends}

    # Find common cell types
    common_types = set(props_dict[backends[0]].columns)
    for b in backends[1:]:
        common_types &= set(props_dict[b].columns)
    common_types = sorted(common_types)

    # Compute correlations
    records = []
    for cell_type in common_types:
        for i, b1 in enumerate(backends):
            for b2 in backends[i+1:]:
                v1 = props_dict[b1][cell_type].values
                v2 = props_dict[b2][cell_type].values

                # Handle edge cases
                if np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
                    corr = np.nan
                else:
                    corr, _ = stats.spearmanr(v1, v2)

                records.append({
                    "cell_type": cell_type,
                    "backend_1": b1,
                    "backend_2": b2,
                    "correlation": corr,
                })

    return pd.DataFrame(records)


def find_best_backend_per_celltype(
    results: dict[str, dict],
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine which backend is most consistent with others for each cell type.

    Without ground truth, we use cross-backend consensus as a proxy for accuracy.
    The backend with highest average correlation to all others is considered "best".
    """
    backends = list(results.keys())
    records = []

    for cell_type in agreement_df["cell_type"].unique():
        ct_df = agreement_df[agreement_df["cell_type"] == cell_type]

        # Calculate average correlation for each backend
        backend_scores = {}
        for backend in backends:
            # Get all correlations involving this backend
            mask = (ct_df["backend_1"] == backend) | (ct_df["backend_2"] == backend)
            correlations = ct_df.loc[mask, "correlation"].dropna()
            backend_scores[backend] = correlations.mean() if len(correlations) > 0 else np.nan

        # Find best backend (highest consensus)
        best_backend = max(backend_scores, key=lambda x: backend_scores.get(x, -1))

        # Calculate variance across backends (lower = more agreement)
        props_values = []
        for b in backends:
            if cell_type in results[b]["proportions"].columns:
                props_values.append(results[b]["proportions"][cell_type].mean())
        variance = np.var(props_values) if len(props_values) > 1 else np.nan

        records.append({
            "cell_type": cell_type,
            "best_backend": best_backend,
            "consensus_score": backend_scores[best_backend],
            "cross_backend_variance": variance,
            **{f"{b}_score": backend_scores.get(b, np.nan) for b in backends},
        })

    return pd.DataFrame(records).sort_values("consensus_score", ascending=False)


def plot_best_backend_summary(
    best_df: pd.DataFrame,
    output_path: Path,
):
    """Create summary plot showing which backend is best for each cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    # Left: Count of "wins" per backend
    ax = axes[0]
    win_counts = best_df["best_backend"].value_counts()
    colors = [BACKEND_COLORS.get(b, "#888888") for b in win_counts.index]
    win_counts.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Number of cell types", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Best Backend by Cell Type (Consensus-Based)", fontsize=12, fontweight="bold")
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(win_counts.values):
        ax.text(i, v + 0.3, str(v), ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: Heatmap of backend scores per cell type (top 15)
    ax = axes[1]
    score_cols = [c for c in best_df.columns if c.endswith("_score")]
    top_types = best_df.head(15)

    heatmap_data = top_types.set_index("cell_type")[score_cols]
    heatmap_data.columns = [c.replace("_score", "") for c in heatmap_data.columns]

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="RdYlGn",
        center=0.5,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Consensus Score"},
    )
    ax.set_title("Backend Consensus Scores (Top 15 Cell Types)", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def plot_celltype_variance_analysis(
    results: dict[str, dict],
    output_path: Path,
):
    """
    Plot variance analysis showing which cell types have highest/lowest
    cross-backend agreement.
    """
    backends = list(results.keys())

    # Get common cell types
    common_types = set(results[backends[0]]["proportions"].columns)
    for b in backends[1:]:
        common_types &= set(results[b]["proportions"].columns)

    # Calculate cross-backend variance for each cell type
    variance_data = []
    for ct in common_types:
        means = [results[b]["proportions"][ct].mean() for b in backends]
        stds = [results[b]["proportions"][ct].std() for b in backends]

        variance_data.append({
            "cell_type": ct,
            "mean_proportion": np.mean(means),
            "cross_backend_cv": np.std(means) / (np.mean(means) + 1e-10),  # CV across backends
            "avg_within_backend_std": np.mean(stds),
        })

    var_df = pd.DataFrame(variance_data).sort_values("cross_backend_cv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    # Left: Most agreed upon cell types (lowest variance)
    ax = axes[0]
    top_agreed = var_df.head(10)
    ax.barh(top_agreed["cell_type"], top_agreed["cross_backend_cv"], color="#2ca02c", edgecolor="black")
    ax.set_xlabel("Cross-Backend CV (lower = more agreement)", fontsize=11)
    ax.set_title("Most Consistent Cell Types Across Backends", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: Most disagreed cell types (highest variance)
    ax = axes[1]
    top_disagreed = var_df.tail(10).iloc[::-1]
    ax.barh(top_disagreed["cell_type"], top_disagreed["cross_backend_cv"], color="#d62728", edgecolor="black")
    ax.set_xlabel("Cross-Backend CV (higher = more disagreement)", fontsize=11)
    ax.set_title("Most Variable Cell Types Across Backends", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")

    return var_df


def generate_all_figures(
    benchmark_dir: Path,
    spatial_h5ad: Path,
    output_dir: Path,
    label_source: str = "hlca",
    sample: Optional[str] = None,
):
    """Generate all spatial benchmark figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPATIAL BACKEND BENCHMARK FIGURE GENERATION")
    print("=" * 70)

    # Find available samples if not specified
    if sample is None or sample == "all":
        # Get first available sample
        sample_dirs = list((benchmark_dir / label_source / "tangram" / "samples").glob("*"))
        if not sample_dirs:
            raise ValueError(f"No samples found in {benchmark_dir / label_source}")
        sample = sample_dirs[0].name
        print(f"Using sample: {sample}")

    # Load results
    print(f"\nLoading results from {benchmark_dir}...")
    results = load_backend_results(benchmark_dir, label_source, sample)

    if len(results) < 2:
        raise ValueError(f"Need at least 2 backends for comparison, got {len(results)}")

    # Load spatial data for coordinates
    print(f"\nLoading spatial data from {spatial_h5ad}...")
    spatial = ad.read_h5ad(spatial_h5ad, backed='r')

    # Filter to sample
    if "sample_id" in spatial.obs.columns:
        sample_mask = spatial.obs["sample_id"] == sample
        spatial_subset = spatial[sample_mask].to_memory()
    else:
        spatial_subset = spatial.to_memory()

    print(f"  {spatial_subset.n_obs} spots for sample {sample}")

    # Get cell type categories
    cell_types_by_abundance = get_cell_types_by_abundance(results)
    print(f"\nCell types by abundance:")
    for cat, types in cell_types_by_abundance.items():
        print(f"  {cat}: {types}")

    # =========================================================================
    # Figure 1: Spatial maps for key cell types across backends
    # =========================================================================
    print("\n" + "=" * 70)
    print("Figure 1: Spatial Cell Type Maps")
    print("=" * 70)

    props_dict = {b: results[b]["proportions"] for b in results}

    # Plot abundant cell types
    for cell_type in cell_types_by_abundance["abundant"][:3]:
        print(f"  Plotting {cell_type}...")
        plot_backend_comparison_spatial(
            spatial_subset,
            props_dict,
            cell_type=cell_type,
            title_prefix=f"{sample} |",
            save_path=output_dir / f"spatial_comparison_{cell_type.replace(' ', '_').replace('/', '_')}.png",
        )

    # Plot rare cell types (often more interesting biologically)
    for cell_type in cell_types_by_abundance["rare"][:3]:
        print(f"  Plotting {cell_type}...")
        plot_backend_comparison_spatial(
            spatial_subset,
            props_dict,
            cell_type=cell_type,
            title_prefix=f"{sample} |",
            save_path=output_dir / f"spatial_comparison_{cell_type.replace(' ', '_').replace('/', '_')}.png",
        )

    # =========================================================================
    # Figure 2: Backend agreement analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Figure 2: Backend Agreement Analysis")
    print("=" * 70)

    agreement_df = compute_backend_agreement(results)
    best_backend_df = find_best_backend_per_celltype(results, agreement_df)

    # Save analysis results
    best_backend_df.to_csv(output_dir / "best_backend_per_celltype.csv", index=False)
    agreement_df.to_csv(output_dir / "pairwise_backend_agreement.csv", index=False)
    print(f"  Saved analysis CSVs")

    # Plot best backend summary
    plot_best_backend_summary(best_backend_df, output_dir / "best_backend_summary.png")

    # =========================================================================
    # Figure 3: Variance analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("Figure 3: Cross-Backend Variance Analysis")
    print("=" * 70)

    variance_df = plot_celltype_variance_analysis(results, output_dir / "variance_analysis.png")
    variance_df.to_csv(output_dir / "celltype_variance.csv", index=False)

    # =========================================================================
    # Figure 4: Upstream metrics comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("Figure 4: Upstream Metrics Comparison")
    print("=" * 70)

    metrics_data = []
    for backend, data in results.items():
        if "metrics" in data:
            metrics_data.append({
                "backend": backend,
                **data["metrics"]
            })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")

        metric_cols = ["mean_entropy", "mean_sparsity", "coverage"]
        titles = ["Mean Entropy (higher = more diverse)",
                  "Mean Sparsity (lower = better)",
                  "Coverage (higher = more cell types)"]

        for ax, col, title in zip(axes, metric_cols, titles):
            if col in metrics_df.columns:
                colors = [BACKEND_COLORS.get(b, "#888888") for b in metrics_df["backend"]]
                metrics_df.plot.bar(x="backend", y=col, ax=ax, color=colors,
                                   edgecolor="black", legend=False)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel("")
                ax.tick_params(axis='x', rotation=45)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / "upstream_metrics.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(output_dir / "upstream_metrics.pdf", bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {output_dir / 'upstream_metrics.png'}")

    # =========================================================================
    # Figure 5: Dominant cell type maps per backend
    # =========================================================================
    print("\n" + "=" * 70)
    print("Figure 5: Dominant Cell Type Maps")
    print("=" * 70)

    for backend, data in results.items():
        try:
            plot_dominant_cell_type_map(
                data["proportions"],
                spatial_subset.obsm["spatial"],
                title=f"{backend.upper()} - Dominant Cell Type",
                save_path=output_dir / f"dominant_celltype_{backend}.png",
            )
        except Exception as e:
            print(f"  Warning: Could not plot dominant map for {backend}: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nAll figures saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Spatial comparison maps: spatial_comparison_*.png")
    print(f"  - Best backend analysis: best_backend_summary.png")
    print(f"  - Variance analysis: variance_analysis.png")
    print(f"  - Upstream metrics: upstream_metrics.png")
    print(f"  - Dominant cell type maps: dominant_celltype_*.png")
    print(f"  - CSV data: *.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial backend benchmark figures"
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Path to spatial benchmark results directory",
    )
    parser.add_argument(
        "--spatial-h5ad",
        type=Path,
        required=True,
        help="Path to spatial AnnData file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/spatial_benchmark"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--label-source",
        choices=["hlca", "luca"],
        default="hlca",
        help="Which label source to use",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Specific sample ID (default: first available)",
    )

    args = parser.parse_args()

    generate_all_figures(
        benchmark_dir=args.benchmark_dir,
        spatial_h5ad=args.spatial_h5ad,
        output_dir=args.output_dir,
        label_source=args.label_source,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
