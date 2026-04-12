#!/usr/bin/env python3
"""Regenerate ALL spatial benchmark figures with Peng Kadara color palette.

This script recreates all figures in results/spatial_benchmark/figures/
using the official LungPCA publication colors from stagebridge/viz/lungpca_style.py
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stagebridge.viz.lungpca_style import (
    STAGE_COLORS,
    STAGE_ORDER,
    MAJOR_CELLTYPE_COLORS,
    EPITHELIAL_COLORS,
    STROMAL_COLORS,
    configure_lungpca_style,
    save_lungpca_figure,
    plot_boxplot_jitter,
    plot_heatmap,
    get_stage_color,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Paths - data is in local results/ directory
BENCHMARK_DIR = Path("results/spatial_benchmark")
OUTPUT_DIR = Path("results/spatial_benchmark/figures")


def load_sample_metadata() -> pd.DataFrame:
    """Load sample metadata with stage info - parses stage from sample_id."""
    # We parse stage directly from sample IDs like GSM9226179_P5_LUAD
    return pd.DataFrame()  # Stage is parsed from sample_id in get_sample_stage


def collect_backend_results() -> pd.DataFrame:
    """Collect results from all backends and samples."""
    results = defaultdict(list)
    backends = ["tangram", "destvi", "tacco", "cell2location", "marker_scoring"]
    label_sources = ["hlca", "luca"]

    for label_source in label_sources:
        for backend in backends:
            samples_dir = BENCHMARK_DIR / label_source / backend / "samples"
            if not samples_dir.exists():
                continue

            for sample_dir in samples_dir.iterdir():
                if not sample_dir.is_dir():
                    continue

                sample_id = sample_dir.name
                # Try multiple metric file names
                metrics_file = sample_dir / "upstream_metrics.json"
                if not metrics_file.exists():
                    metrics_file = sample_dir / "metrics.json"

                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    results["backend"].append(backend)
                    results["label_source"].append(label_source)
                    results["sample_id"].append(sample_id)
                    # Handle both naming conventions
                    results["entropy"].append(metrics.get("mean_entropy", metrics.get("entropy", np.nan)))
                    results["coverage"].append(metrics.get("coverage", np.nan))
                    results["sparsity"].append(metrics.get("sparsity", np.nan))
                    results["confidence"].append(metrics.get("mean_confidence", np.nan))
                    results["n_cell_types"].append(metrics.get("n_cell_types", 0))

    return pd.DataFrame(results)


def get_sample_stage(sample_id: str, metadata: pd.DataFrame = None) -> str:
    """Get stage for a sample from sample_id.

    Sample IDs are like: GSM9226179_P5_LUAD, GSM9226170_P2_AAH, GSM9226218_P23_AIS-1
    Stage is the last part after the final underscore (may have -N suffix).
    """
    # Parse from sample_id - stage is after final underscore
    parts = sample_id.split("_")
    if len(parts) >= 3:
        # Last part is stage (may have -1, -2 suffix for duplicates)
        stage_part = parts[-1].split("-")[0].upper()

        stage_map = {
            "NORMAL": "Normal",
            "AAH": "AAH",
            "AIS": "AIS",
            "MIA": "MIA",
            "LUAD": "LUAD",
            "ADC": "LUAD",
        }
        return stage_map.get(stage_part, "Unknown")
    return "Unknown"


# =============================================================================
# FIGURE 1: Training Curves (DestVI loss)
# =============================================================================

def fig_training_curves():
    """Training curves for DestVI (the only backend with loss curves)."""
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Look for DestVI training history in both HLCA and LuCA
    samples_plotted = []
    for label_source in ["luca", "hlca"]:
        destvi_dir = BENCHMARK_DIR / label_source / "destvi" / "samples"
        if not destvi_dir.exists():
            continue

        for sample_dir in sorted(destvi_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if len(samples_plotted) >= 10:  # Limit to 10 samples
                break

            history_file = sample_dir / "destvi_training_history.csv"
            if history_file.exists():
                try:
                    history = pd.read_csv(history_file)
                    if "train_loss" in history.columns:
                        losses = history["train_loss"].values
                        stage = get_sample_stage(sample_dir.name)
                        color = get_stage_color(stage)
                        ax.plot(losses, color=color, alpha=0.6, linewidth=1.2)
                        samples_plotted.append(sample_dir.name)
                except Exception as e:
                    log.warning(f"Could not read {history_file}: {e}")

    if not samples_plotted:
        log.warning("No DestVI training history found")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Training Loss (ELBO)", fontsize=9)
    ax.set_title("DestVI Training Convergence by Disease Stage", fontsize=11)

    # Legend with stage colors
    stage_patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER if s in STAGE_COLORS]
    ax.legend(handles=stage_patches, loc="upper right", fontsize=7, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_training_curves.png")
    plt.close(fig)
    log.info(f"Saved fig_training_curves.png/pdf ({len(samples_plotted)} samples)")


# =============================================================================
# FIGURE 2: Upstream Metrics Comparison
# =============================================================================

def fig_upstream_metrics():
    """Compare upstream metrics (entropy, coverage, sparsity) across backends."""
    configure_lungpca_style()
    results = collect_backend_results()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    metrics = ["entropy", "coverage", "sparsity"]
    titles = ["Entropy (higher = more diverse)", "Coverage (higher = better)", "Sparsity (lower = sparser)"]

    backends = ["tangram", "destvi", "tacco", "cell2location"]
    backend_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Distinct colors

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]

        data_by_backend = []
        for backend in backends:
            values = results[results["backend"] == backend][metric].dropna().values
            if len(values) > 0:
                data_by_backend.append(values)
            else:
                data_by_backend.append(np.array([np.nan]))

        positions = np.arange(len(backends))
        bp = ax.boxplot(data_by_backend, positions=positions, widths=0.6, patch_artist=True)

        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(backend_colors[j])
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels([b.capitalize() for b in backends], rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(metric.capitalize(), fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_upstream_metrics.png")
    plt.close(fig)
    log.info("Saved fig_upstream_metrics.png/pdf")


# =============================================================================
# FIGURE 3: Stage Comparison (Key figure with Peng Kadara colors)
# =============================================================================

def fig_stage_comparison():
    """Compare metrics across disease stages with Peng Kadara palette."""
    configure_lungpca_style()
    results = collect_backend_results()
    metadata = load_sample_metadata()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    # Add stage to results
    results["stage"] = results["sample_id"].apply(lambda x: get_sample_stage(x, metadata))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Entropy by stage
    ax = axes[0]
    stages_present = [s for s in STAGE_ORDER if s in results["stage"].unique()]

    data_by_stage = []
    colors = []
    for stage in stages_present:
        values = results[results["stage"] == stage]["entropy"].dropna().values
        if len(values) > 0:
            data_by_stage.append(values)
            colors.append(STAGE_COLORS.get(stage, "#999999"))
        else:
            data_by_stage.append(np.array([np.nan]))
            colors.append("#999999")

    plot_boxplot_jitter(
        ax, data_by_stage,
        positions=list(range(len(stages_present))),
        colors=colors,
        labels=stages_present,
        jitter_size=10,
    )
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("Cell Type Entropy by Disease Stage", fontsize=10)

    # Panel B: Coverage by stage
    ax = axes[1]
    data_by_stage = []
    for stage in stages_present:
        values = results[results["stage"] == stage]["coverage"].dropna().values
        if len(values) > 0:
            data_by_stage.append(values)
        else:
            data_by_stage.append(np.array([np.nan]))

    plot_boxplot_jitter(
        ax, data_by_stage,
        positions=list(range(len(stages_present))),
        colors=colors,
        labels=stages_present,
        jitter_size=10,
    )
    ax.set_ylabel("Coverage", fontsize=8)
    ax.set_title("Cell Type Coverage by Disease Stage", fontsize=10)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_stage_comparison.png")
    plt.close(fig)
    log.info("Saved fig_stage_comparison.png/pdf")


# =============================================================================
# FIGURE 4: Cross-Method Correlation
# =============================================================================

def fig_cross_method_correlation():
    """Correlation matrix between backends."""
    configure_lungpca_style()
    results = collect_backend_results()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Pivot to get backend x sample matrix for entropy
    pivot = results.pivot_table(
        index="sample_id",
        columns="backend",
        values="entropy",
        aggfunc="mean"
    ).dropna()

    if len(pivot) < 3:
        log.warning("Not enough samples for correlation")
        plt.close(fig)
        return

    corr = pivot.corr()

    im = plot_heatmap(
        ax, corr.values,
        row_labels=list(corr.index),
        col_labels=list(corr.columns),
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        show_values=True,
    )

    ax.set_title("Backend Correlation (Entropy)", fontsize=10)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=7)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_cross_method_correlation.png")
    plt.close(fig)
    log.info("Saved fig_cross_method_correlation.png/pdf")


# =============================================================================
# FIGURE 5: Atlas Comparison (HLCA vs LuCA label sources)
# =============================================================================

def fig_atlas_comparison():
    """Compare HLCA vs LuCA label sources."""
    configure_lungpca_style()
    results = collect_backend_results()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Panel A: Entropy comparison
    ax = axes[0]
    hlca_entropy = results[results["label_source"] == "hlca"]["entropy"].dropna().values
    luca_entropy = results[results["label_source"] == "luca"]["entropy"].dropna().values

    data = [hlca_entropy if len(hlca_entropy) > 0 else np.array([np.nan]),
            luca_entropy if len(luca_entropy) > 0 else np.array([np.nan])]

    # Use muted versions of stage colors for HLCA (healthy) and LUAD color for LuCA (cancer)
    colors = [STAGE_COLORS["Normal"], STAGE_COLORS["LUAD"]]  # Green for HLCA, Orange for LuCA

    plot_boxplot_jitter(
        ax, data,
        positions=[0, 1],
        colors=colors,
        labels=["HLCA (healthy)", "LuCA (cancer)"],
        jitter_size=8,
    )
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("Cell Type Diversity by Reference Atlas", fontsize=10)

    # Panel B: Coverage comparison
    ax = axes[1]
    hlca_cov = results[results["label_source"] == "hlca"]["coverage"].dropna().values
    luca_cov = results[results["label_source"] == "luca"]["coverage"].dropna().values

    data = [hlca_cov if len(hlca_cov) > 0 else np.array([np.nan]),
            luca_cov if len(luca_cov) > 0 else np.array([np.nan])]

    plot_boxplot_jitter(
        ax, data,
        positions=[0, 1],
        colors=colors,
        labels=["HLCA (healthy)", "LuCA (cancer)"],
        jitter_size=8,
    )
    ax.set_ylabel("Coverage", fontsize=8)
    ax.set_title("Cell Type Coverage by Reference Atlas", fontsize=10)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_atlas_comparison.png")
    plt.close(fig)
    log.info("Saved fig_atlas_comparison.png/pdf")


# =============================================================================
# FIGURE 6: Stage Cell Type Composition
# =============================================================================

def fig_stage_spatial_celltypes():
    """Cell type proportions by disease stage with Peng Kadara colors."""
    configure_lungpca_style()
    results = collect_backend_results()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    # Collect cell type proportions from deconvolution outputs (parquet files)
    celltype_data = defaultdict(lambda: defaultdict(list))

    for _, row in results.iterrows():
        sample_dir = BENCHMARK_DIR / row["label_source"] / row["backend"] / "samples" / row["sample_id"]
        props_file = sample_dir / "cell_type_proportions.parquet"

        if props_file.exists():
            try:
                props_df = pd.read_parquet(props_file)
                # Get mean proportion per cell type
                stage = get_sample_stage(row["sample_id"])
                for col in props_df.columns:
                    if col not in ["spot_id", "barcode", "x", "y"]:
                        mean_prop = props_df[col].mean()
                        celltype_data[stage][col].append(mean_prop)
            except Exception as e:
                log.warning(f"Could not read {props_file}: {e}")
                continue

    if not celltype_data:
        log.warning("No cell type proportion data found")
        return

    # Get top cell types across all stages
    all_celltypes = set()
    for stage_data in celltype_data.values():
        all_celltypes.update(stage_data.keys())

    # Calculate mean proportions
    mean_props = {}
    for ct in all_celltypes:
        total = sum(np.mean(celltype_data[s].get(ct, [0])) for s in STAGE_ORDER if s in celltype_data)
        mean_props[ct] = total

    # Top 10 cell types
    top_celltypes = sorted(mean_props.keys(), key=lambda x: mean_props[x], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(12, 6))

    stages_present = [s for s in STAGE_ORDER if s in celltype_data]
    x = np.arange(len(stages_present))
    width = 0.08
    n_types = len(top_celltypes)

    # Get colors for cell types
    ct_colors = []
    for ct in top_celltypes:
        color = (EPITHELIAL_COLORS.get(ct) or
                 MAJOR_CELLTYPE_COLORS.get(ct) or
                 STROMAL_COLORS.get(ct) or
                 "#999999")
        ct_colors.append(color)

    for i, ct in enumerate(top_celltypes):
        means = [np.mean(celltype_data[s].get(ct, [0])) for s in stages_present]
        stds = [np.std(celltype_data[s].get(ct, [0])) for s in stages_present]

        offset = (i - n_types/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=ct, color=ct_colors[i], alpha=0.8)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none", color="black", capsize=2, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(stages_present, fontsize=9)

    # Color the x-tick labels by stage
    for tick, stage in zip(ax.get_xticklabels(), stages_present):
        tick.set_color(STAGE_COLORS.get(stage, "black"))
        tick.set_fontweight("bold")

    ax.set_ylabel("Mean Proportion", fontsize=9)
    ax.set_title("Cell Type Composition Across Disease Stages", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_stage_spatial_celltypes.png")
    plt.close(fig)
    log.info("Saved fig_stage_spatial_celltypes.png/pdf")


# =============================================================================
# FIGURE 7: Stage Spatial Metrics
# =============================================================================

def fig_stage_spatial_metrics():
    """Spatial metrics (entropy, coverage) by stage with Peng Kadara colors."""
    configure_lungpca_style()
    results = collect_backend_results()
    metadata = load_sample_metadata()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    results["stage"] = results["sample_id"].apply(lambda x: get_sample_stage(x, metadata))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    stages_present = [s for s in STAGE_ORDER if s in results["stage"].unique()]
    colors = [STAGE_COLORS.get(s, "#999999") for s in stages_present]

    # Panel A: Entropy by stage (all backends combined)
    ax = axes[0, 0]
    data = [results[results["stage"] == s]["entropy"].dropna().values for s in stages_present]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(len(stages_present))), colors, stages_present, jitter_size=6)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("Cell Type Entropy", fontsize=10)

    # Panel B: Coverage by stage
    ax = axes[0, 1]
    data = [results[results["stage"] == s]["coverage"].dropna().values for s in stages_present]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(len(stages_present))), colors, stages_present, jitter_size=6)
    ax.set_ylabel("Coverage", fontsize=8)
    ax.set_title("Cell Type Coverage", fontsize=10)

    # Panel C: Sparsity by stage
    ax = axes[1, 0]
    data = [results[results["stage"] == s]["sparsity"].dropna().values for s in stages_present]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(len(stages_present))), colors, stages_present, jitter_size=6)
    ax.set_ylabel("Sparsity", fontsize=8)
    ax.set_title("Deconvolution Sparsity", fontsize=10)

    # Panel D: Number of cell types detected
    ax = axes[1, 1]
    data = [results[results["stage"] == s]["n_cell_types"].dropna().values for s in stages_present]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(len(stages_present))), colors, stages_present, jitter_size=6)
    ax.set_ylabel("N Cell Types", fontsize=8)
    ax.set_title("Cell Types Detected", fontsize=10)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_stage_spatial_metrics.png")
    plt.close(fig)
    log.info("Saved fig_stage_spatial_metrics.png/pdf")


# =============================================================================
# FIGURE 8: Quantitative Trends
# =============================================================================

def fig_stage_quantitative_trends():
    """Quantitative trends across progression with trend lines."""
    configure_lungpca_style()
    results = collect_backend_results()
    metadata = load_sample_metadata()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    results["stage"] = results["sample_id"].apply(lambda x: get_sample_stage(x, metadata))

    # Assign numeric stage order for trend
    stage_to_num = {s: i for i, s in enumerate(STAGE_ORDER)}
    results["stage_num"] = results["stage"].map(stage_to_num)
    results = results.dropna(subset=["stage_num"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    metrics = ["entropy", "coverage", "n_cell_types"]
    titles = ["Entropy vs Progression", "Coverage vs Progression", "Cell Type Count vs Progression"]
    ylabels = ["Entropy", "Coverage", "N Cell Types"]

    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]

        # Plot points with stage colors
        for stage in STAGE_ORDER:
            mask = results["stage"] == stage
            if mask.any():
                ax.scatter(
                    results.loc[mask, "stage_num"] + np.random.uniform(-0.15, 0.15, mask.sum()),
                    results.loc[mask, metric],
                    c=STAGE_COLORS.get(stage, "#999999"),
                    s=20,
                    alpha=0.7,
                    label=stage,
                    edgecolors="none"
                )

        # Add trend line
        valid = results[[metric, "stage_num"]].dropna()
        if len(valid) > 5:
            slope, intercept, r, p, se = stats.linregress(valid["stage_num"], valid[metric])
            x_line = np.array([0, len(STAGE_ORDER) - 1])
            ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1, alpha=0.5)
            ax.text(0.98, 0.02, f"r={r:.2f}, p={p:.2e}", transform=ax.transAxes,
                   fontsize=6, ha="right", va="bottom")

        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, fontsize=7)

        # Color x-tick labels
        for tick, stage in zip(ax.get_xticklabels(), STAGE_ORDER):
            tick.set_color(STAGE_COLORS.get(stage, "black"))

        ax.set_xlabel("Disease Stage", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Single legend
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_stage_quantitative_trends.png")
    plt.close(fig)
    log.info("Saved fig_stage_quantitative_trends.png/pdf")


# =============================================================================
# FIGURE 9: Comprehensive Spatial Analysis
# =============================================================================

def fig_spatial_analysis_comprehensive():
    """Comprehensive 4-panel spatial analysis figure."""
    configure_lungpca_style()
    results = collect_backend_results()
    metadata = load_sample_metadata()

    if len(results) == 0:
        log.warning("No backend results found")
        return

    results["stage"] = results["sample_id"].apply(lambda x: get_sample_stage(x, metadata))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    backends = ["tangram", "destvi", "tacco", "cell2location"]
    backend_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Panel A: Backend comparison (entropy)
    ax = axes[0, 0]
    data = [results[results["backend"] == b]["entropy"].dropna().values for b in backends]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(4)), backend_colors,
                       [b.capitalize() for b in backends], jitter_size=6)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("A. Backend Comparison", fontsize=10)

    # Panel B: Stage progression (using best backend - tangram)
    ax = axes[0, 1]
    tangram_results = results[results["backend"] == "tangram"]
    stages_present = [s for s in STAGE_ORDER if s in tangram_results["stage"].unique()]
    stage_colors = [STAGE_COLORS.get(s, "#999999") for s in stages_present]

    data = [tangram_results[tangram_results["stage"] == s]["entropy"].dropna().values for s in stages_present]
    data = [d if len(d) > 0 else np.array([np.nan]) for d in data]
    plot_boxplot_jitter(ax, data, list(range(len(stages_present))), stage_colors,
                       stages_present, jitter_size=6)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("B. Stage Progression (Tangram)", fontsize=10)

    # Panel C: HLCA vs LuCA
    ax = axes[1, 0]
    hlca = results[results["label_source"] == "hlca"]["entropy"].dropna().values
    luca = results[results["label_source"] == "luca"]["entropy"].dropna().values
    data = [hlca if len(hlca) > 0 else np.array([np.nan]),
            luca if len(luca) > 0 else np.array([np.nan])]
    colors = [STAGE_COLORS["Normal"], STAGE_COLORS["LUAD"]]
    plot_boxplot_jitter(ax, data, [0, 1], colors, ["HLCA", "LuCA"], jitter_size=6)
    ax.set_ylabel("Entropy", fontsize=8)
    ax.set_title("C. Reference Atlas Comparison", fontsize=10)

    # Panel D: Sample counts by stage
    ax = axes[1, 1]
    stage_counts = results.groupby("stage")["sample_id"].nunique()
    stages_present = [s for s in STAGE_ORDER if s in stage_counts.index]
    counts = [stage_counts.get(s, 0) for s in stages_present]
    colors = [STAGE_COLORS.get(s, "#999999") for s in stages_present]

    bars = ax.bar(range(len(stages_present)), counts, color=colors, alpha=0.8)
    ax.set_xticks(range(len(stages_present)))
    ax.set_xticklabels(stages_present, fontsize=8)
    ax.set_ylabel("N Samples", fontsize=8)
    ax.set_title("D. Samples per Stage", fontsize=10)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha="center", fontsize=7)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_spatial_analysis_comprehensive.png")
    plt.close(fig)
    log.info("Saved fig_spatial_analysis_comprehensive.png/pdf")


# =============================================================================
# FIGURE 10: Spatial Overlay (example spatial plot)
# =============================================================================

def fig_spatial_overlay():
    """Example spatial overlay with cell type predictions."""
    configure_lungpca_style()

    # Try to import scanpy
    try:
        import scanpy as sc
    except ImportError:
        log.warning("scanpy not available, skipping spatial overlay")
        return

    # Find a TACCO sample with h5ad (has spatial coordinates)
    sample_file = None
    sample_name = None
    for label_source in ["luca", "hlca"]:
        tacco_dir = BENCHMARK_DIR / label_source / "tacco" / "samples"
        if tacco_dir.exists():
            for sd in tacco_dir.iterdir():
                h5ad = sd / "tacco_annotated_spatial.h5ad"
                if h5ad.exists():
                    sample_file = h5ad
                    sample_name = sd.name
                    break
        if sample_file:
            break

    if sample_file is None:
        log.warning("No TACCO h5ad with spatial coordinates found")
        return

    try:
        adata = sc.read_h5ad(sample_file)
    except Exception as e:
        log.warning(f"Could not read {sample_file}: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Get coordinates from obsm
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        x, y = coords[:, 0], coords[:, 1]
    else:
        log.warning("No spatial coordinates in h5ad")
        plt.close(fig)
        return

    # Get cell type predictions
    celltype_col = None
    for col in ["predicted_cell_type", "cell_type", "tacco_cell_type"]:
        if col in adata.obs.columns:
            celltype_col = col
            break

    if celltype_col is None:
        # Try tacco_celltype in obsm (DataFrame with proportions per cell type)
        if "tacco_celltype" in adata.obsm:
            tacco_df = adata.obsm["tacco_celltype"]
            celltypes = tacco_df.idxmax(axis=1).values
        else:
            # Try to find proportion columns and use argmax
            prop_cols = [c for c in adata.obs.columns if "proportion" in c.lower() or "q05" in c.lower()]
            if prop_cols:
                celltypes = adata.obs[prop_cols].idxmax(axis=1).values
                # Clean up names
                celltypes = [c.split("_")[-1] if "_" in c else c for c in celltypes]
            else:
                log.warning("No cell type column found")
                plt.close(fig)
                return
    else:
        celltypes = adata.obs[celltype_col].values

    # Simplify cell type names
    def simplify_celltype(ct):
        ct = str(ct)
        # Remove long prefixes
        if "q05cell_abundance" in ct:
            ct = ct.split("_mu_fg_")[-1]
        return ct

    celltypes = [simplify_celltype(ct) for ct in celltypes]

    # Build color array
    unique_types = list(set(celltypes))
    color_list = []
    for ct in celltypes:
        color = (EPITHELIAL_COLORS.get(ct) or
                 MAJOR_CELLTYPE_COLORS.get(ct) or
                 STROMAL_COLORS.get(ct) or
                 "#999999")
        color_list.append(color)

    ax.scatter(x, y, c=color_list, s=3, alpha=0.8, edgecolors="none")
    ax.set_aspect("equal")
    ax.axis("off")

    stage = get_sample_stage(sample_name)
    ax.set_title(f"Spatial Cell Types: {sample_name} ({stage})", fontsize=10,
                color=STAGE_COLORS.get(stage, "black"))

    # Legend - top 10 types by count
    type_counts = pd.Series(celltypes).value_counts()
    top_types = type_counts.head(10).index.tolist()
    patches = []
    for ct in top_types:
        color = (EPITHELIAL_COLORS.get(ct) or MAJOR_CELLTYPE_COLORS.get(ct) or
                 STROMAL_COLORS.get(ct) or "#999999")
        patches.append(mpatches.Patch(color=color, label=ct[:25]))  # Truncate long names

    ax.legend(handles=patches, loc="upper right", fontsize=5, frameon=False)

    plt.tight_layout()
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_spatial_overlay.png")
    plt.close(fig)
    log.info(f"Saved fig_spatial_overlay.png/pdf ({sample_name})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Regenerate all figures with Peng Kadara palette."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Regenerating all spatial benchmark figures with Peng Kadara palette...")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info("")

    # Run all figure generators
    fig_training_curves()
    fig_upstream_metrics()
    fig_stage_comparison()
    fig_cross_method_correlation()
    fig_atlas_comparison()
    fig_stage_spatial_celltypes()
    fig_stage_spatial_metrics()
    fig_stage_quantitative_trends()
    fig_spatial_analysis_comprehensive()
    fig_spatial_overlay()

    log.info("")
    log.info("Done! All figures regenerated with Peng Kadara color palette.")


if __name__ == "__main__":
    main()
