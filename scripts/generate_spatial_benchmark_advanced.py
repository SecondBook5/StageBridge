#!/usr/bin/env python3
"""Generate ADVANCED publication-quality figures for spatial benchmark.

Creates multi-panel figures similar to generate_advanced_figures.py style:
- 4x4 spatial overview grid showing samples across stages
- Backend comparison panels
- Stage-resolved cell type heatmaps
- Spatial metrics progression analysis
- UMAP/t-SNE of deconvolution results

Uses EXACT Peng Kadara colors from stagebridge/viz/lungpca_style.py
"""

import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from stagebridge.viz.lungpca_style import (
    STAGE_COLORS,
    STAGE_ORDER,
    MAJOR_CELLTYPE_COLORS,
    EPITHELIAL_COLORS,
    STROMAL_COLORS,
    configure_lungpca_style,
    save_lungpca_figure,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Paths
BENCHMARK_DIR = Path("results/spatial_benchmark")
OUTPUT_DIR = Path("results/spatial_benchmark/figures")

# Backend config
BACKENDS = ["tangram", "destvi", "tacco", "cell2location"]
BACKEND_COLORS = {"tangram": "#1f77b4", "destvi": "#ff7f0e", "tacco": "#2ca02c", "cell2location": "#d62728", "marker_scoring": "#9467bd"}


def get_sample_stage(sample_id: str) -> str:
    """Parse stage from sample ID like GSM9226179_P5_LUAD."""
    parts = sample_id.split("_")
    if len(parts) >= 3:
        stage_part = parts[-1].split("-")[0].upper()
        stage_map = {"NORMAL": "Normal", "AAH": "AAH", "AIS": "AIS", "MIA": "MIA", "LUAD": "LUAD", "ADC": "LUAD"}
        return stage_map.get(stage_part, "Unknown")
    return "Unknown"


def collect_all_data() -> tuple[pd.DataFrame, dict, dict]:
    """Collect metrics, cell type proportions, and training histories."""
    metrics_rows = []
    celltype_data = defaultdict(lambda: defaultdict(list))  # stage -> celltype -> [proportions]
    training_histories = defaultdict(list)  # backend -> [history_dfs]

    for label_source in ["hlca", "luca"]:
        for backend in BACKENDS + ["marker_scoring"]:
            samples_dir = BENCHMARK_DIR / label_source / backend / "samples"
            if not samples_dir.exists():
                continue

            for sample_dir in samples_dir.iterdir():
                if not sample_dir.is_dir():
                    continue

                sample_id = sample_dir.name
                stage = get_sample_stage(sample_id)

                # Metrics
                metrics_file = sample_dir / "upstream_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        m = json.load(f)
                    metrics_rows.append({
                        "sample_id": sample_id,
                        "stage": stage,
                        "backend": backend,
                        "label_source": label_source,
                        "entropy": m.get("mean_entropy", np.nan),
                        "coverage": m.get("coverage", np.nan),
                        "sparsity": m.get("sparsity", np.nan),
                        "confidence": m.get("mean_confidence", np.nan),
                    })

                # Cell type proportions
                props_file = sample_dir / "cell_type_proportions.parquet"
                if props_file.exists():
                    try:
                        props_df = pd.read_parquet(props_file)
                        for col in props_df.columns:
                            if col not in ["spot_id", "barcode", "x", "y"]:
                                # Simplify column name
                                ct = col.split("_mu_fg_")[-1] if "_mu_fg_" in col else col
                                celltype_data[stage][ct].append(props_df[col].mean())
                    except Exception:
                        pass

                # Training history (DestVI)
                history_file = sample_dir / "destvi_training_history.csv"
                if history_file.exists():
                    try:
                        hist = pd.read_csv(history_file)
                        hist["sample_id"] = sample_id
                        hist["stage"] = stage
                        training_histories[backend].append(hist)
                    except Exception:
                        pass

    return pd.DataFrame(metrics_rows), dict(celltype_data), dict(training_histories)


def load_spatial_data():
    """Load h5ad files with spatial coordinates for visualization."""
    try:
        import scanpy as sc
    except ImportError:
        return {}

    spatial_samples = {}
    for label_source in ["luca", "hlca"]:
        tacco_dir = BENCHMARK_DIR / label_source / "tacco" / "samples"
        if not tacco_dir.exists():
            continue

        for sample_dir in sorted(tacco_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            h5ad = sample_dir / "tacco_annotated_spatial.h5ad"
            if h5ad.exists() and len(spatial_samples) < 20:  # Limit to 20
                try:
                    adata = sc.read_h5ad(h5ad)
                    stage = get_sample_stage(sample_dir.name)
                    spatial_samples[sample_dir.name] = {
                        "adata": adata,
                        "stage": stage,
                        "label_source": label_source,
                    }
                except Exception:
                    pass

    return spatial_samples


# =============================================================================
# FIGURE 1: 4x4 Spatial Grid (16 panels)
# =============================================================================

def fig_spatial_grid_16panel(spatial_samples: dict):
    """16-panel spatial grid showing samples across all stages."""
    configure_lungpca_style()

    # Group by stage
    by_stage = defaultdict(list)
    for sample_id, data in spatial_samples.items():
        by_stage[data["stage"]].append((sample_id, data))

    # Select up to 3-4 samples per stage for 16 panels
    selected = []
    for stage in STAGE_ORDER:
        samples = by_stage.get(stage, [])[:4]
        selected.extend(samples)

    if len(selected) < 4:
        log.warning("Not enough spatial samples for 16-panel figure")
        return

    # Pad to 16 if needed
    while len(selected) < 16:
        selected.append((None, None))

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.15, wspace=0.1)

    for idx, (sample_id, data) in enumerate(selected[:16]):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])

        if sample_id is None or data is None:
            ax.axis("off")
            continue

        adata = data["adata"]
        stage = data["stage"]

        # Get spatial coords
        if "spatial" not in adata.obsm:
            ax.axis("off")
            continue

        coords = adata.obsm["spatial"]
        x, y = coords[:, 0], coords[:, 1]

        # Get cell type predictions from tacco_celltype
        if "tacco_celltype" in adata.obsm:
            tacco_df = adata.obsm["tacco_celltype"]
            dominant_ct = tacco_df.idxmax(axis=1).values

            # Color by cell type
            ct_counts = pd.Series(dominant_ct).value_counts()
            top_ct = ct_counts.head(8).index.tolist()

            colors = []
            for ct in dominant_ct:
                if ct in EPITHELIAL_COLORS:
                    colors.append(EPITHELIAL_COLORS[ct])
                elif ct in MAJOR_CELLTYPE_COLORS:
                    colors.append(MAJOR_CELLTYPE_COLORS[ct])
                elif ct in STROMAL_COLORS:
                    colors.append(STROMAL_COLORS[ct])
                else:
                    colors.append("#cccccc")

            ax.scatter(x, y, c=colors, s=1, alpha=0.7, rasterized=True)
        else:
            # Just plot coordinates
            ax.scatter(x, y, c=STAGE_COLORS.get(stage, "#999"), s=1, alpha=0.5, rasterized=True)

        ax.set_aspect("equal")
        ax.axis("off")

        # Title with stage color
        ax.set_title(f"{sample_id[-15:]}\n{stage}", fontsize=8,
                    color=STAGE_COLORS.get(stage, "black"), fontweight="bold")

    # Add legend
    stage_patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
    fig.legend(handles=stage_patches, loc="lower center", ncol=5, fontsize=10, frameon=False)

    fig.suptitle("Spatial Cell Type Maps Across Disease Progression", fontsize=16, fontweight="bold", y=0.98)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_lungpca_figure(fig, OUTPUT_DIR / "fig_spatial_grid_16panel.png")
    plt.close(fig)
    log.info("Saved fig_spatial_grid_16panel.png/pdf")


# =============================================================================
# FIGURE 2: Comprehensive Backend + Stage Analysis (12 panels)
# =============================================================================

def fig_comprehensive_analysis(metrics_df: pd.DataFrame, celltype_data: dict, histories: dict):
    """12-panel comprehensive analysis figure."""
    configure_lungpca_style()

    if len(metrics_df) == 0:
        log.warning("No metrics data for comprehensive analysis")
        return

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Backend comparison (entropy)
    ax = fig.add_subplot(gs[0, 0])
    data_by_backend = []
    colors = []
    labels = []
    for backend in BACKENDS:
        vals = metrics_df[metrics_df["backend"] == backend]["entropy"].dropna().values
        if len(vals) > 0:
            data_by_backend.append(vals)
            colors.append(BACKEND_COLORS.get(backend, "#999"))
            labels.append(backend.capitalize())

    if data_by_backend:
        bp = ax.boxplot(data_by_backend, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Entropy", fontsize=9)
    ax.set_title("A. Backend Comparison", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel B: Backend comparison (coverage)
    ax = fig.add_subplot(gs[0, 1])
    data_by_backend = []
    for backend in BACKENDS:
        vals = metrics_df[metrics_df["backend"] == backend]["coverage"].dropna().values
        if len(vals) > 0:
            data_by_backend.append(vals)

    if data_by_backend:
        bp = ax.boxplot(data_by_backend, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Coverage", fontsize=9)
    ax.set_title("B. Cell Type Coverage", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel C: Stage progression (entropy)
    ax = fig.add_subplot(gs[0, 2])
    stages_present = [s for s in STAGE_ORDER if s in metrics_df["stage"].unique()]
    stage_colors = [STAGE_COLORS.get(s, "#999") for s in stages_present]

    data_by_stage = []
    for stage in stages_present:
        vals = metrics_df[metrics_df["stage"] == stage]["entropy"].dropna().values
        data_by_stage.append(vals if len(vals) > 0 else np.array([np.nan]))

    if data_by_stage:
        bp = ax.boxplot(data_by_stage, patch_artist=True)
        for patch, color in zip(bp["boxes"], stage_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticklabels(stages_present, fontsize=9)
    ax.set_ylabel("Entropy", fontsize=9)
    ax.set_title("C. Stage Progression", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel D: HLCA vs LuCA comparison
    ax = fig.add_subplot(gs[0, 3])
    hlca_ent = metrics_df[metrics_df["label_source"] == "hlca"]["entropy"].dropna().values
    luca_ent = metrics_df[metrics_df["label_source"] == "luca"]["entropy"].dropna().values
    data = [hlca_ent if len(hlca_ent) > 0 else np.array([np.nan]),
            luca_ent if len(luca_ent) > 0 else np.array([np.nan])]
    bp = ax.boxplot(data, patch_artist=True)
    bp["boxes"][0].set_facecolor(STAGE_COLORS["Normal"])
    bp["boxes"][1].set_facecolor(STAGE_COLORS["LUAD"])
    ax.set_xticklabels(["HLCA (healthy)", "LuCA (cancer)"], fontsize=9)
    ax.set_ylabel("Entropy", fontsize=9)
    ax.set_title("D. Reference Atlas Effect", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel E: Sample counts by stage
    ax = fig.add_subplot(gs[1, 0])
    stage_counts = metrics_df.groupby("stage")["sample_id"].nunique()
    stages_present = [s for s in STAGE_ORDER if s in stage_counts.index]
    counts = [stage_counts.get(s, 0) for s in stages_present]
    colors = [STAGE_COLORS.get(s, "#999") for s in stages_present]
    bars = ax.bar(range(len(stages_present)), counts, color=colors, alpha=0.8)
    ax.set_xticks(range(len(stages_present)))
    ax.set_xticklabels(stages_present, fontsize=9)
    ax.set_ylabel("N Samples", fontsize=9)
    ax.set_title("E. Sample Distribution", fontsize=11, fontweight="bold")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha="center", fontsize=8, fontweight="bold")

    # Panel F: Entropy vs Coverage scatter
    ax = fig.add_subplot(gs[1, 1])
    for stage in STAGE_ORDER:
        mask = metrics_df["stage"] == stage
        if mask.any():
            ax.scatter(metrics_df.loc[mask, "entropy"], metrics_df.loc[mask, "coverage"],
                      c=STAGE_COLORS.get(stage, "#999"), label=stage, s=40, alpha=0.7, edgecolors="white")
    ax.set_xlabel("Entropy", fontsize=9)
    ax.set_ylabel("Coverage", fontsize=9)
    ax.set_title("F. Entropy vs Coverage", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # Panel G: Confidence by stage
    ax = fig.add_subplot(gs[1, 2])
    data_by_stage = []
    for stage in stages_present:
        vals = metrics_df[metrics_df["stage"] == stage]["confidence"].dropna().values
        data_by_stage.append(vals if len(vals) > 0 else np.array([np.nan]))

    if data_by_stage and any(len(d) > 0 for d in data_by_stage):
        bp = ax.boxplot(data_by_stage, patch_artist=True)
        for patch, color in zip(bp["boxes"], stage_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticklabels(stages_present, fontsize=9)
    ax.set_ylabel("Mapping Confidence", fontsize=9)
    ax.set_title("G. Deconvolution Confidence", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel H: Backend correlation heatmap
    ax = fig.add_subplot(gs[1, 3])
    pivot = metrics_df.pivot_table(index="sample_id", columns="backend", values="entropy", aggfunc="mean")
    pivot = pivot.dropna(thresh=2)  # Need at least 2 backends
    if len(pivot) >= 3 and len(pivot.columns) >= 2:
        corr = pivot.corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels([c[:8] for c in corr.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c[:8] for c in corr.index], fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("H. Backend Correlation", fontsize=11, fontweight="bold")

    # Panel I: Training curves (if available)
    ax = fig.add_subplot(gs[2, 0])
    if "destvi" in histories and histories["destvi"]:
        all_hist = pd.concat(histories["destvi"][:10])  # First 10
        for sample_id in all_hist["sample_id"].unique()[:8]:
            sub = all_hist[all_hist["sample_id"] == sample_id]
            stage = sub["stage"].iloc[0]
            color = STAGE_COLORS.get(stage, "#999")
            ax.plot(sub["train_loss"].values, color=color, alpha=0.6, linewidth=1)
        stage_patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
        ax.legend(handles=stage_patches, fontsize=6, loc="upper right")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Training Loss", fontsize=9)
    ax.set_title("I. DestVI Training Curves", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel J: Cell type composition heatmap
    ax = fig.add_subplot(gs[2, 1])
    if celltype_data:
        # Get top 10 cell types
        all_cts = set()
        for stage_data in celltype_data.values():
            all_cts.update(stage_data.keys())

        ct_means = {}
        for ct in all_cts:
            total = sum(np.mean(celltype_data[s].get(ct, [0])) for s in STAGE_ORDER if s in celltype_data)
            ct_means[ct] = total
        top_cts = sorted(ct_means.keys(), key=lambda x: ct_means[x], reverse=True)[:10]

        # Build matrix
        stages_present = [s for s in STAGE_ORDER if s in celltype_data]
        matrix = np.zeros((len(stages_present), len(top_cts)))
        for i, stage in enumerate(stages_present):
            for j, ct in enumerate(top_cts):
                matrix[i, j] = np.mean(celltype_data[stage].get(ct, [0]))

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_yticks(range(len(stages_present)))
        ax.set_yticklabels(stages_present, fontsize=9)
        ax.set_xticks(range(len(top_cts)))
        ax.set_xticklabels([ct[:15] for ct in top_cts], rotation=45, ha="right", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")
    ax.set_title("J. Cell Type x Stage", fontsize=11, fontweight="bold")

    # Panel K: Trend with regression
    ax = fig.add_subplot(gs[2, 2])
    stage_to_num = {s: i for i, s in enumerate(STAGE_ORDER)}
    metrics_df["stage_num"] = metrics_df["stage"].map(stage_to_num)
    valid = metrics_df[["entropy", "stage_num"]].dropna()

    if len(valid) > 5:
        for stage in STAGE_ORDER:
            mask = metrics_df["stage"] == stage
            if mask.any():
                jitter = np.random.uniform(-0.15, 0.15, mask.sum())
                ax.scatter(metrics_df.loc[mask, "stage_num"] + jitter,
                          metrics_df.loc[mask, "entropy"],
                          c=STAGE_COLORS.get(stage, "#999"), s=30, alpha=0.7, edgecolors="white")

        slope, intercept, r, p, se = stats.linregress(valid["stage_num"], valid["entropy"])
        x_line = np.array([0, len(STAGE_ORDER) - 1])
        ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, alpha=0.7)
        ax.text(0.98, 0.02, f"r={r:.2f}, p={p:.1e}", transform=ax.transAxes,
               fontsize=8, ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=9)
    ax.set_xlabel("Disease Stage", fontsize=9)
    ax.set_ylabel("Entropy", fontsize=9)
    ax.set_title("K. Progression Trend", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel L: Summary statistics table
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")

    summary_data = [
        ["Metric", "Value"],
        ["Total Samples", str(metrics_df["sample_id"].nunique())],
        ["Backends", str(len(metrics_df["backend"].unique()))],
        ["Stages", str(len(stages_present))],
        ["Mean Entropy", f"{metrics_df['entropy'].mean():.3f}"],
        ["Mean Coverage", f"{metrics_df['coverage'].mean():.3f}"],
        ["Mean Confidence", f"{metrics_df['confidence'].mean():.3f}"],
    ]

    table = ax.table(cellText=summary_data, cellLoc="left", loc="center",
                     bbox=[0.1, 0.2, 0.8, 0.6], edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(2):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("L. Summary Statistics", fontsize=11, fontweight="bold")

    fig.suptitle("Spatial Deconvolution Benchmark: Comprehensive Analysis",
                fontsize=18, fontweight="bold", y=0.98)

    save_lungpca_figure(fig, OUTPUT_DIR / "fig_comprehensive_12panel.png")
    plt.close(fig)
    log.info("Saved fig_comprehensive_12panel.png/pdf")


# =============================================================================
# FIGURE 3: Stage-Specific Cell Type Spatial Maps (5x2 panels)
# =============================================================================

def fig_stage_spatial_maps(spatial_samples: dict):
    """5x2 panel showing 2 samples per stage with cell type coloring."""
    configure_lungpca_style()

    # Group by stage
    by_stage = defaultdict(list)
    for sample_id, data in spatial_samples.items():
        by_stage[data["stage"]].append((sample_id, data))

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.2, wspace=0.1)

    for row_idx, stage in enumerate(STAGE_ORDER):
        samples = by_stage.get(stage, [])[:2]

        for col_idx in range(2):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx >= len(samples):
                ax.axis("off")
                ax.text(0.5, 0.5, f"No sample\n({stage})", ha="center", va="center",
                       fontsize=10, color="#999")
                continue

            sample_id, data = samples[col_idx]
            adata = data["adata"]

            if "spatial" not in adata.obsm:
                ax.axis("off")
                continue

            coords = adata.obsm["spatial"]
            x, y = coords[:, 0], coords[:, 1]

            # Get cell type predictions
            if "tacco_celltype" in adata.obsm:
                tacco_df = adata.obsm["tacco_celltype"]
                dominant_ct = tacco_df.idxmax(axis=1).values

                # Color by cell type
                colors = []
                for ct in dominant_ct:
                    if ct in EPITHELIAL_COLORS:
                        colors.append(EPITHELIAL_COLORS[ct])
                    elif ct in MAJOR_CELLTYPE_COLORS:
                        colors.append(MAJOR_CELLTYPE_COLORS[ct])
                    elif ct in STROMAL_COLORS:
                        colors.append(STROMAL_COLORS[ct])
                    else:
                        colors.append("#cccccc")

                ax.scatter(x, y, c=colors, s=2, alpha=0.8, rasterized=True)
            else:
                ax.scatter(x, y, c=STAGE_COLORS.get(stage, "#999"), s=2, alpha=0.5, rasterized=True)

            ax.set_aspect("equal")
            ax.axis("off")

            # Border color by stage
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(STAGE_COLORS.get(stage, "#999"))
                spine.set_linewidth(3)

            ax.set_title(f"{sample_id[-20:]}", fontsize=9, color=STAGE_COLORS.get(stage, "black"))

        # Row label
        fig.text(0.02, 0.9 - row_idx * 0.18, stage, fontsize=14, fontweight="bold",
                color=STAGE_COLORS.get(stage, "black"), rotation=90, va="center")

    fig.suptitle("Spatial Cell Type Distribution by Disease Stage",
                fontsize=16, fontweight="bold", y=0.99)

    save_lungpca_figure(fig, OUTPUT_DIR / "fig_stage_spatial_maps.png")
    plt.close(fig)
    log.info("Saved fig_stage_spatial_maps.png/pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all advanced spatial benchmark figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ADVANCED Spatial Benchmark Figures")
    log.info("Using EXACT Peng Kadara colors from lungpca_style.py")
    log.info("=" * 60)
    log.info("")
    log.info("Stage colors:")
    for stage, color in STAGE_COLORS.items():
        log.info(f"  {stage}: {color}")
    log.info("")

    # Collect data
    log.info("Collecting metrics data...")
    metrics_df, celltype_data, histories = collect_all_data()
    log.info(f"  Found {len(metrics_df)} metric records")
    log.info(f"  Found {len(celltype_data)} stages with cell type data")

    log.info("Loading spatial samples...")
    spatial_samples = load_spatial_data()
    log.info(f"  Found {len(spatial_samples)} spatial samples")

    # Generate figures
    log.info("")
    log.info("Generating figures...")

    if spatial_samples:
        fig_spatial_grid_16panel(spatial_samples)
        fig_stage_spatial_maps(spatial_samples)

    if len(metrics_df) > 0:
        fig_comprehensive_analysis(metrics_df, celltype_data, histories)

    log.info("")
    log.info("=" * 60)
    log.info("Done! Advanced figures saved to:")
    log.info(f"  {OUTPUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
