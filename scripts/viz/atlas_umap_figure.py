#!/usr/bin/env python
"""
Atlas UMAP Figure — Multi-panel visualization of HLCA and LuCA atlas features
at the neighborhood level, colored by stage and atlas similarity scores.

Produces a publication-quality figure showing:
  Panel A: UMAP of combined atlas features, colored by histological stage
  Panel B: Same UMAP, colored by HLCA normal-likeness score
  Panel C: Same UMAP, colored by LuCA tumor-adoption score
  Panel D: Same UMAP, colored by LuCA invasive-like score
  Panel E: Violin plots of key atlas scores stratified by stage
  Panel F: HLCA-only UMAP vs LuCA-only UMAP side-by-side

Usage:
    python scripts/viz/atlas_umap_figure.py [--n-sample 20000] [--output figures/atlas_umap.png]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import LinearSegmentedColormap

# ── Feature indices ──────────────────────────────────────────────────────────
# HLCA (13-dim)
HLCA_NORMAL_LIKENESS = 5       # cosine sim to Normal-stage baseline distribution
HLCA_DEVIATION = 6             # 1 - normal_likeness
HLCA_LINEAGE_FIDELITY = 7     # max_state_sim / dominant_lineage
HLCA_TOPK_ENTROPY = 9         # entropy over top-k similarities
HLCA_EPITHELIAL = 10          # epithelial-lineage similarity
HLCA_IMMUNE = 11              # immune-lineage similarity
HLCA_STROMAL = 12             # stromal/endothelial similarity

# LuCA (15-dim; index 14 is state_count=51, constant — drop it)
LUCA_TUMOR_ADOPTION = 5       # mean sim to malignant LuCA states
LUCA_INVASIVE_LIKE = 6        # mean sim to invasive-like LuCA states
LUCA_ECOSYSTEM = 7            # malignant × max(immune, stromal) interplay
LUCA_TOPK_ENTROPY = 9         # entropy over top-k
LUCA_MALIGNANT = 10           # mean sim to all malignant states
LUCA_IMMUNE = 11              # mean sim to immune states
LUCA_STROMAL = 12             # mean sim to stromal states
LUCA_EPITHELIAL = 13          # mean sim to epithelial states
LUCA_STATE_COUNT = 14         # constant=51, drop

# ── Stage configuration ─────────────────────────────────────────────────────
STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
STAGE_COLORS = {
    "Normal": "#2ca02c",   # green
    "AAH":    "#98df8a",   # light green
    "AIS":    "#ff7f0e",   # orange
    "MIA":    "#d62728",   # red
    "LUAD":   "#9467bd",   # purple
}

# Ordinal grouping for violin panels
ORDINAL_GROUPS = {
    "Normal": "Pre-invasive",
    "AAH":    "Pre-invasive",
    "AIS":    "In-situ",
    "MIA":    "In-situ",
    "LUAD":   "Invasive",
}
GROUP_ORDER = ["Pre-invasive", "In-situ", "Invasive"]
GROUP_COLORS = {
    "Pre-invasive": "#2ca02c",
    "In-situ":      "#ff7f0e",
    "Invasive":     "#9467bd",
}


def load_and_unpack(parquet_path: str):
    """Load parquet and unpack nested arrays into flat neighborhood-level arrays."""
    df = pd.read_parquet(parquet_path)
    all_hlca, all_luca, all_stages, all_lesion_ids = [], [], [], []

    for _, row in df.iterrows():
        hlca_arr = np.array(list(row["hlca_features"]))  # (N, 13)
        luca_arr = np.array(list(row["luca_features"]))  # (N, 15)
        n = len(hlca_arr)
        all_hlca.append(hlca_arr)
        all_luca.append(luca_arr)
        all_stages.extend([row["stage_label"]] * n)
        all_lesion_ids.extend([row["lesion_id"]] * n)

    hlca = np.vstack(all_hlca)   # (639816, 13)
    luca = np.vstack(all_luca)   # (639816, 15)
    stages = np.array(all_stages)
    lesion_ids = np.array(all_lesion_ids)

    return hlca, luca, stages, lesion_ids


def subsample_stratified(hlca, luca, stages, lesion_ids, n_sample=20000, seed=42):
    """Stratified subsample by stage to preserve stage proportions."""
    rng = np.random.default_rng(seed)
    n_total = len(stages)
    if n_total <= n_sample:
        return hlca, luca, stages, lesion_ids

    # Proportional sampling per stage
    unique_stages, counts = np.unique(stages, return_counts=True)
    fracs = counts / n_total
    indices = []
    for st, frac in zip(unique_stages, fracs):
        stage_mask = stages == st
        stage_idx = np.where(stage_mask)[0]
        n_pick = max(1, int(n_sample * frac))
        chosen = rng.choice(stage_idx, size=min(n_pick, len(stage_idx)), replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return hlca[indices], luca[indices], stages[indices], lesion_ids[indices]


def compute_umap(features, n_neighbors=30, min_dist=0.3, seed=42):
    """Compute 2D UMAP embedding."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=seed,
        n_jobs=-1,
    )
    return reducer.fit_transform(features)


def plot_scatter_by_stage(ax, coords, stages, title, point_size=1.0, alpha=0.4):
    """Scatter plot colored by categorical stage."""
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=STAGE_COLORS[stage], label=stage,
            s=point_size, alpha=alpha, edgecolors="none", rasterized=True,
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)


def plot_scatter_by_score(ax, coords, scores, title, cmap="viridis",
                          point_size=1.0, alpha=0.4, vmin=None, vmax=None):
    """Scatter plot colored by continuous score."""
    if vmin is None:
        vmin = np.percentile(scores, 2)
    if vmax is None:
        vmax = np.percentile(scores, 98)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=scores, cmap=cmap,
        s=point_size, alpha=alpha, edgecolors="none",
        vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    return sc


def plot_violins(ax, hlca, luca, stages, feature_name, feature_idx, source="hlca"):
    """Violin plot of a feature stratified by ordinal group."""
    if source == "hlca":
        values = hlca[:, feature_idx]
    else:
        values = luca[:, feature_idx]

    groups = np.array([ORDINAL_GROUPS[s] for s in stages])
    data_by_group = [values[groups == g] for g in GROUP_ORDER]

    parts = ax.violinplot(data_by_group, positions=range(len(GROUP_ORDER)),
                          showmeans=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(GROUP_COLORS[GROUP_ORDER[i]])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")

    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER, fontsize=8)
    ax.set_ylabel(feature_name, fontsize=9)
    ax.set_title(feature_name, fontsize=10, fontweight="bold")


def make_figure(hlca, luca, stages, lesion_ids, n_sample=20000, output_path=None):
    """Create the full multi-panel atlas UMAP figure."""
    print(f"Subsampling to {n_sample} neighborhoods (stratified by stage)...")
    hlca_s, luca_s, stages_s, _ = subsample_stratified(
        hlca, luca, stages, lesion_ids, n_sample=n_sample
    )
    print(f"  Got {len(stages_s)} neighborhoods after subsampling")

    # Drop the constant state_count column from LuCA
    luca_s_trim = luca_s[:, :LUCA_STATE_COUNT]  # (N, 14)

    # Combined features for main UMAP
    combined = np.hstack([hlca_s, luca_s_trim])  # (N, 27)
    print(f"Computing combined UMAP ({combined.shape[1]}d → 2d)...")
    umap_combined = compute_umap(combined)

    # HLCA-only UMAP
    print(f"Computing HLCA-only UMAP ({hlca_s.shape[1]}d → 2d)...")
    umap_hlca = compute_umap(hlca_s)

    # LuCA-only UMAP
    print(f"Computing LuCA-only UMAP ({luca_s_trim.shape[1]}d → 2d)...")
    umap_luca = compute_umap(luca_s_trim)

    # ── Build figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), dpi=150, facecolor="white")
    gs = gridspec.GridSpec(3, 4, hspace=0.30, wspace=0.30,
                           left=0.05, right=0.95, top=0.94, bottom=0.05)

    # Custom colormaps
    cmap_healthy = LinearSegmentedColormap.from_list(
        "healthy", ["#d62728", "#ffffbf", "#2ca02c"])  # red→yellow→green
    cmap_cancer = LinearSegmentedColormap.from_list(
        "cancer", ["#2ca02c", "#ffffbf", "#9467bd"])    # green→yellow→purple
    cmap_invasive = LinearSegmentedColormap.from_list(
        "invasive", ["#2ca02c", "#ffffbf", "#d62728"])  # green→yellow→red

    ps = 0.8   # point size
    al = 0.35  # alpha

    # Row 1: Combined UMAP — 4 colorings
    ax_a = fig.add_subplot(gs[0, 0])
    plot_scatter_by_stage(ax_a, umap_combined, stages_s,
                          "A. Combined Atlas UMAP\n(colored by stage)", ps, al)
    leg = ax_a.legend(loc="lower right", fontsize=7, markerscale=5,
                      framealpha=0.9, edgecolor="gray")

    ax_b = fig.add_subplot(gs[0, 1])
    sc_b = plot_scatter_by_score(
        ax_b, umap_combined, hlca_s[:, HLCA_NORMAL_LIKENESS],
        "B. HLCA Normal-Likeness\n(healthy reference similarity)",
        cmap=cmap_healthy, point_size=ps, alpha=al)
    plt.colorbar(sc_b, ax=ax_b, fraction=0.046, pad=0.04)

    ax_c = fig.add_subplot(gs[0, 2])
    sc_c = plot_scatter_by_score(
        ax_c, umap_combined, luca_s[:, LUCA_TUMOR_ADOPTION],
        "C. LuCA Tumor-Adoption\n(cancer reference similarity)",
        cmap=cmap_cancer, point_size=ps, alpha=al)
    plt.colorbar(sc_c, ax=ax_c, fraction=0.046, pad=0.04)

    ax_d = fig.add_subplot(gs[0, 3])
    sc_d = plot_scatter_by_score(
        ax_d, umap_combined, luca_s[:, LUCA_INVASIVE_LIKE],
        "D. LuCA Invasive-Like Score\n(invasive cancer similarity)",
        cmap=cmap_invasive, point_size=ps, alpha=al)
    plt.colorbar(sc_d, ax=ax_d, fraction=0.046, pad=0.04)

    # Row 2: HLCA-only vs LuCA-only UMAPs + atlas-specific score overlays
    ax_e = fig.add_subplot(gs[1, 0])
    plot_scatter_by_stage(ax_e, umap_hlca, stages_s,
                          "E. HLCA-Only UMAP\n(13 healthy reference features)", ps, al)

    ax_f = fig.add_subplot(gs[1, 1])
    sc_f = plot_scatter_by_score(
        ax_f, umap_hlca, hlca_s[:, HLCA_DEVIATION],
        "F. HLCA Deviation from Normal\n(1 − normal-likeness)",
        cmap=cmap_invasive, point_size=ps, alpha=al)
    plt.colorbar(sc_f, ax=ax_f, fraction=0.046, pad=0.04)

    ax_g = fig.add_subplot(gs[1, 2])
    plot_scatter_by_stage(ax_g, umap_luca, stages_s,
                          "G. LuCA-Only UMAP\n(14 cancer reference features)", ps, al)

    ax_h = fig.add_subplot(gs[1, 3])
    sc_h = plot_scatter_by_score(
        ax_h, umap_luca, luca_s[:, LUCA_ECOSYSTEM],
        "H. LuCA Tumor-Immune-Stromal\n(ecosystem interplay score)",
        cmap=cmap_cancer, point_size=ps, alpha=al)
    plt.colorbar(sc_h, ax=ax_h, fraction=0.046, pad=0.04)

    # Row 3: Violin plots of key atlas scores by ordinal group
    violin_specs = [
        ("HLCA Normal-Likeness", HLCA_NORMAL_LIKENESS, "hlca"),
        ("HLCA Deviation from Normal", HLCA_DEVIATION, "hlca"),
        ("LuCA Tumor-Adoption", LUCA_TUMOR_ADOPTION, "luca"),
        ("LuCA Invasive-Like", LUCA_INVASIVE_LIKE, "luca"),
    ]
    for col_idx, (name, feat_idx, source) in enumerate(violin_specs):
        ax_v = fig.add_subplot(gs[2, col_idx])
        plot_violins(ax_v, hlca_s, luca_s, stages_s, name, feat_idx, source)

    fig.suptitle(
        "Atlas-Guided Neighborhood Characterization: HLCA (Healthy) vs LuCA (Cancer) Reference Features",
        fontsize=14, fontweight="bold", y=0.98,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {output_path}")

        # Also save a high-res PDF
        pdf_path = output_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved PDF to {pdf_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Atlas UMAP figure generation")
    parser.add_argument("--parquet", default="/mnt/e/StageBridge_data/processed/features/eamist_bags.parquet")
    parser.add_argument("--n-sample", type=int, default=20000,
                        help="Number of neighborhoods to subsample for UMAP")
    parser.add_argument("--output", default="reports/figures/atlas_umap_figure.png")
    args = parser.parse_args()

    print(f"Loading data from {args.parquet}...")
    hlca, luca, stages, lesion_ids = load_and_unpack(args.parquet)
    print(f"Loaded {len(stages)} neighborhoods from {len(np.unique(lesion_ids))} lesions")

    make_figure(hlca, luca, stages, lesion_ids,
                n_sample=args.n_sample, output_path=args.output)


if __name__ == "__main__":
    main()
