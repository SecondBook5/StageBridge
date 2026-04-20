#!/usr/bin/env python3
"""Publication showcase figures - dramatic visualizations for posters and papers.

Creates visually striking figures using:
- Dual-reference embeddings
- Stage progression
- Niche composition
- Cell type distributions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Circle, ConnectionPatch
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False


# =============================================================================
# STYLE
# =============================================================================

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

STAGE_COLORS = {
    'Normal': '#3B82F6',   # Blue
    'AAH': '#22D3EE',      # Cyan
    'AIS': '#10B981',      # Emerald
    'MIA': '#F59E0B',      # Amber
    'LUAD': '#EF4444',     # Red
}

STAGE_CMAP = LinearSegmentedColormap.from_list(
    'stages', [STAGE_COLORS[s] for s in STAGE_ORDER]
)

DARK_BG = '#0a0a0f'
LIGHT_BG = '#fafafa'


def load_data(data_dir: Path):
    """Load cells and neighborhoods."""
    cells = pd.read_parquet(data_dir / "cells.parquet")

    neighborhoods = None
    npath = data_dir / "neighborhoods.parquet"
    if npath.exists():
        neighborhoods = pd.read_parquet(npath)

    return cells, neighborhoods


def get_embeddings(df, prefix="z_fused_"):
    """Extract embedding columns."""
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not cols:
        return None
    X = df[cols].values.astype(np.float32)
    # Filter NaN
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], df[valid].copy()


def sample_balanced(df, n_per_stage=5000, seed=42):
    """Sample balanced across stages."""
    np.random.seed(seed)
    samples = []
    for stage in STAGE_ORDER:
        stage_df = df[df['stage'] == stage]
        n = min(len(stage_df), n_per_stage)
        if n > 0:
            samples.append(stage_df.sample(n, random_state=seed))
    return pd.concat(samples, ignore_index=True)


# =============================================================================
# FIGURE 1: Dramatic 3D Stage Progression
# =============================================================================

def figure_3d_progression(cells, output_dir):
    """3D UMAP with dramatic lighting and stage gradient."""
    print("\n[Fig 1] 3D Stage Progression...")

    cells_s = sample_balanced(cells, n_per_stage=8000)
    X, cells_s = get_embeddings(cells_s)

    if X is None:
        print("  ERROR: No embeddings found")
        return

    # 3D reduction
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(X)
    else:
        coords = PCA(n_components=3, random_state=42).fit_transform(X)

    stages = cells_s['stage'].values
    stage_idx = np.array([STAGE_ORDER.index(s) if s in STAGE_ORDER else 0 for s in stages])

    # Create figure
    fig = plt.figure(figsize=(14, 12), facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=DARK_BG)

    # Plot points with stage colors
    for i, stage in enumerate(STAGE_ORDER):
        mask = stages == stage
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=STAGE_COLORS[stage], s=3, alpha=0.6, label=stage,
            rasterized=True
        )

    # Stage centroids with glow
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            centroid = coords[mask].mean(axis=0)
            ax.scatter(*centroid, s=500, c=STAGE_COLORS[stage], alpha=0.3, edgecolors='none')
            ax.scatter(*centroid, s=200, c=STAGE_COLORS[stage], edgecolors='white', linewidth=2)

    # Connect centroids with arrows
    centroids = {stage: coords[stages == stage].mean(axis=0) for stage in STAGE_ORDER if (stages == stage).sum() > 0}
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        if s1 in centroids and s2 in centroids:
            c1, c2 = centroids[s1], centroids[s2]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                   'w-', linewidth=2, alpha=0.7)

    ax.set_xlabel('UMAP 1', color='white', fontsize=12)
    ax.set_ylabel('UMAP 2', color='white', fontsize=12)
    ax.set_zlabel('UMAP 3', color='white', fontsize=12)
    ax.tick_params(colors='white')

    # Legend
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
    legend.get_frame().set_facecolor('#1a1a2e')
    for text in legend.get_texts():
        text.set_color('white')

    ax.set_title('LUAD Progression in Dual-Reference Embedding Space',
                fontsize=16, fontweight='bold', color='white', pad=20)

    # Style panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    plt.tight_layout()
    fig.savefig(output_dir / "fig_3d_progression.png", dpi=300, facecolor=DARK_BG, bbox_inches='tight')
    fig.savefig(output_dir / "fig_3d_progression.pdf", facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_3d_progression.png/pdf")


# =============================================================================
# FIGURE 2: Density Contour Progression
# =============================================================================

def figure_density_contours(cells, output_dir):
    """Overlaid density contours showing stage distributions."""
    print("\n[Fig 2] Density Contour Progression...")

    cells_s = sample_balanced(cells, n_per_stage=10000)
    X, cells_s = get_embeddings(cells_s)

    if X is None:
        print("  ERROR: No embeddings found")
        return

    # 2D reduction
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(X)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)

    stages = cells_s['stage'].values

    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    # Compute and plot density contours for each stage
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() < 100:
            continue

        x, y = coords[mask, 0], coords[mask, 1]

        try:
            kde = gaussian_kde(np.vstack([x, y]))

            # Create grid
            xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
            yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

            # Plot contours
            levels = np.percentile(Zi[Zi > 0], [50, 75, 90])
            ax.contour(Xi, Yi, Zi, levels=levels, colors=[STAGE_COLORS[stage]],
                      linewidths=[1, 1.5, 2], alpha=0.8)
            ax.contourf(Xi, Yi, Zi, levels=[levels[-1], Zi.max()],
                       colors=[STAGE_COLORS[stage]], alpha=0.15)
        except Exception as e:
            print(f"    Warning: KDE failed for {stage}: {e}")

    # Add legend
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=s, alpha=0.7) for s in STAGE_ORDER]
    ax.legend(handles=handles, loc='upper right', fontsize=11)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Stage-Specific Cell State Distributions', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(output_dir / "fig_density_contours.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "fig_density_contours.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_density_contours.png/pdf")


# =============================================================================
# FIGURE 3: PAGA-style Stage Graph
# =============================================================================

def figure_paga_style(cells, output_dir):
    """PAGA-style graph showing stage connectivity."""
    print("\n[Fig 3] PAGA-style Stage Graph...")

    cells_s = sample_balanced(cells, n_per_stage=5000)
    X, cells_s = get_embeddings(cells_s)

    if X is None:
        print("  ERROR: No embeddings found")
        return

    stages = cells_s['stage'].values

    # Compute stage transition matrix based on embedding similarity
    from scipy.spatial.distance import cdist

    centroids = {}
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            centroids[stage] = X[mask].mean(axis=0)

    # Compute pairwise distances between stage centroids
    stage_list = [s for s in STAGE_ORDER if s in centroids]
    n_stages = len(stage_list)

    dist_matrix = np.zeros((n_stages, n_stages))
    for i, s1 in enumerate(stage_list):
        for j, s2 in enumerate(stage_list):
            dist_matrix[i, j] = np.linalg.norm(centroids[s1] - centroids[s2])

    # Convert to connectivity (inverse distance)
    connectivity = 1 / (dist_matrix + 0.1)
    np.fill_diagonal(connectivity, 0)
    connectivity = connectivity / connectivity.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')

    # Position nodes in a circle
    angles = np.linspace(0, 2*np.pi, n_stages, endpoint=False) - np.pi/2
    positions = {stage: (np.cos(angles[i]) * 3, np.sin(angles[i]) * 3)
                for i, stage in enumerate(stage_list)}

    # Draw edges with width proportional to connectivity
    for i, s1 in enumerate(stage_list):
        for j, s2 in enumerate(stage_list):
            if i < j and connectivity[i, j] > 0.1:
                pos1, pos2 = positions[s1], positions[s2]
                width = connectivity[i, j] * 8
                alpha = min(connectivity[i, j] * 1.5, 0.8)
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                       color='#666666', linewidth=width, alpha=alpha, zorder=1)

    # Draw nodes
    for stage in stage_list:
        pos = positions[stage]
        mask = stages == stage
        size = mask.sum() / len(stages) * 5000 + 500

        # Glow
        ax.scatter(*pos, s=size*1.5, c=STAGE_COLORS[stage], alpha=0.2, zorder=2)
        # Main node
        circle = plt.Circle(pos, 0.6, facecolor=STAGE_COLORS[stage],
                           edgecolor='white', linewidth=3, zorder=3)
        ax.add_patch(circle)

        # Label
        ax.text(pos[0], pos[1], stage, fontsize=12, fontweight='bold',
               ha='center', va='center', color='white', zorder=4)

    # Add arrows for progression direction
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        if s1 in positions and s2 in positions:
            pos1, pos2 = np.array(positions[s1]), np.array(positions[s2])
            direction = pos2 - pos1
            direction = direction / np.linalg.norm(direction)
            start = pos1 + direction * 0.7
            end = pos2 - direction * 0.7

            arrow = FancyArrowPatch(start, end, arrowstyle='-|>',
                                   mutation_scale=20, linewidth=2,
                                   color='#333333', zorder=5)
            ax.add_patch(arrow)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Stage Connectivity Graph\n(node size = population, edge width = similarity)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_paga_style.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "fig_paga_style.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_paga_style.png/pdf")


# =============================================================================
# FIGURE 4: Pseudotime Ridge Plot
# =============================================================================

def figure_pseudotime_ridge(cells, output_dir):
    """Ridge plot showing pseudotime distributions per stage."""
    print("\n[Fig 4] Pseudotime Ridge Plot...")

    cells_s = sample_balanced(cells, n_per_stage=10000)
    X, cells_s = get_embeddings(cells_s)

    if X is None:
        print("  ERROR: No embeddings found")
        return

    stages = cells_s['stage'].values

    # Compute diffusion pseudotime proxy using first PC projected onto stage axis
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)

    # Project onto progression axis (Normal -> LUAD direction)
    normal_mask = stages == 'Normal'
    luad_mask = stages == 'LUAD'

    if normal_mask.sum() > 0 and luad_mask.sum() > 0:
        normal_center = X_pca[normal_mask].mean(axis=0)
        luad_center = X_pca[luad_mask].mean(axis=0)
        progression_axis = luad_center - normal_center
        progression_axis = progression_axis / np.linalg.norm(progression_axis)

        pseudotime = X_pca @ progression_axis
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())
    else:
        pseudotime = X_pca[:, 0]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

    # Create ridge plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    y_offset = 0
    y_scale = 1.5

    for i, stage in enumerate(STAGE_ORDER):
        mask = stages == stage
        if mask.sum() < 50:
            continue

        pt = pseudotime[mask]

        # KDE
        try:
            kde = gaussian_kde(pt, bw_method=0.1)
            x = np.linspace(0, 1, 200)
            density = kde(x)
            density = density / density.max() * 0.8  # Normalize

            # Fill
            ax.fill_between(x, y_offset, y_offset + density,
                           color=STAGE_COLORS[stage], alpha=0.7)
            ax.plot(x, y_offset + density, color=STAGE_COLORS[stage],
                   linewidth=2, alpha=0.9)

            # Label
            ax.text(-0.05, y_offset + 0.3, stage, fontsize=12, fontweight='bold',
                   ha='right', va='center', color=STAGE_COLORS[stage])

            y_offset += y_scale
        except Exception as e:
            print(f"    Warning: Ridge plot failed for {stage}: {e}")

    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.2, y_offset + 0.5)
    ax.set_xlabel('Pseudotime (progression score)', fontsize=12)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Cell Distribution Along Progression Axis', fontsize=14, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "fig_pseudotime_ridge.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "fig_pseudotime_ridge.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_pseudotime_ridge.png/pdf")


# =============================================================================
# FIGURE 5: Cell Type Composition Alluvial
# =============================================================================

def figure_celltype_alluvial(cells, output_dir):
    """Alluvial diagram showing cell type composition changes across stages."""
    print("\n[Fig 5] Cell Type Composition Alluvial...")

    if 'cell_type' not in cells.columns:
        print("  Skipping: no cell_type column")
        return

    # Get composition per stage
    compositions = {}
    for stage in STAGE_ORDER:
        stage_cells = cells[cells['stage'] == stage]
        if len(stage_cells) > 0:
            comp = stage_cells['cell_type'].value_counts(normalize=True)
            compositions[stage] = comp

    if not compositions:
        print("  Skipping: no composition data")
        return

    # Get top cell types
    all_types = set()
    for comp in compositions.values():
        all_types.update(comp.head(8).index)
    cell_types = sorted(list(all_types))[:10]

    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

    x = np.arange(len(STAGE_ORDER))
    bottom = np.zeros(len(STAGE_ORDER))

    cmap = plt.cm.tab20

    for i, ct in enumerate(cell_types):
        values = [compositions.get(stage, pd.Series()).get(ct, 0) for stage in STAGE_ORDER]
        color = cmap(i / len(cell_types))
        ax.fill_between(x, bottom, bottom + values, label=ct, color=color, alpha=0.8)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER, fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Cell Type Composition Across Disease Stages', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_xlim(-0.5, len(STAGE_ORDER) - 0.5)
    ax.set_ylim(0, 1)

    # Color x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])

    plt.tight_layout()
    fig.savefig(output_dir / "fig_celltype_alluvial.png", dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / "fig_celltype_alluvial.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_celltype_alluvial.png/pdf")


# =============================================================================
# FIGURE 6: Dramatic Hero Summary
# =============================================================================

def figure_hero_summary(cells, output_dir):
    """Multi-panel hero figure summarizing the progression story."""
    print("\n[Fig 6] Hero Summary Figure...")

    cells_s = sample_balanced(cells, n_per_stage=5000)
    X, cells_s = get_embeddings(cells_s)

    if X is None:
        print("  ERROR: No embeddings found")
        return

    # 2D reduction
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
        coords = reducer.fit_transform(X)
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)

    stages = cells_s['stage'].values

    # Create multi-panel figure
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)

    # Main embedding panel (large, left)
    ax1 = fig.add_axes([0.05, 0.35, 0.55, 0.6])
    ax1.set_facecolor(DARK_BG)

    for stage in STAGE_ORDER:
        mask = stages == stage
        ax1.scatter(coords[mask, 0], coords[mask, 1], c=STAGE_COLORS[stage],
                   s=5, alpha=0.5, label=stage, rasterized=True)

    # Centroids with connections
    centroids = {}
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            centroids[stage] = coords[mask].mean(axis=0)
            c = centroids[stage]
            ax1.scatter(*c, s=300, c=STAGE_COLORS[stage], edgecolors='white',
                       linewidth=2, zorder=10)

    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        if s1 in centroids and s2 in centroids:
            c1, c2 = centroids[s1], centroids[s2]
            ax1.annotate('', xy=c2, xytext=c1,
                        arrowprops=dict(arrowstyle='-|>', color='white', lw=2))

    legend = ax1.legend(loc='upper left', fontsize=11, framealpha=0.8)
    legend.get_frame().set_facecolor('#1a1a2e')
    for text in legend.get_texts():
        text.set_color('white')

    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_title('LUAD Progression in Dual-Reference Space', fontsize=14,
                 fontweight='bold', color='white', pad=15)

    # Stage count bars (right top)
    ax2 = fig.add_axes([0.65, 0.55, 0.3, 0.35])
    ax2.set_facecolor(DARK_BG)

    counts = [len(cells[cells['stage'] == s]) for s in STAGE_ORDER]
    bars = ax2.barh(STAGE_ORDER, counts, color=[STAGE_COLORS[s] for s in STAGE_ORDER])
    ax2.set_xlabel('Number of Cells', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.set_title('Dataset Composition', fontsize=12, fontweight='bold', color='white')
    for spine in ax2.spines.values():
        spine.set_color('white')

    # Progression arrow (bottom)
    ax3 = fig.add_axes([0.05, 0.08, 0.9, 0.15])
    ax3.set_facecolor(DARK_BG)

    x_positions = np.linspace(0.1, 0.9, len(STAGE_ORDER))
    for i, stage in enumerate(STAGE_ORDER):
        ax3.scatter(x_positions[i], 0.5, s=800, c=STAGE_COLORS[stage],
                   edgecolors='white', linewidth=2, zorder=10)
        ax3.text(x_positions[i], 0.1, stage, ha='center', va='top',
                fontsize=11, fontweight='bold', color='white')

        if i < len(STAGE_ORDER) - 1:
            ax3.annotate('', xy=(x_positions[i+1] - 0.05, 0.5),
                        xytext=(x_positions[i] + 0.05, 0.5),
                        arrowprops=dict(arrowstyle='-|>', color='white', lw=2))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.2, 1)
    ax3.axis('off')
    ax3.text(0.5, 0.9, 'Disease Progression', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='white')

    fig.savefig(output_dir / "fig_hero_summary.png", dpi=300, facecolor=DARK_BG, bbox_inches='tight')
    fig.savefig(output_dir / "fig_hero_summary.pdf", facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: fig_hero_summary.png/pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication showcase figures")
    parser.add_argument("--data_dir", type=Path, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Publication Showcase Figures")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from {args.data_dir}...")
    cells, neighborhoods = load_data(args.data_dir)
    print(f"  Loaded {len(cells):,} cells")
    if neighborhoods is not None:
        print(f"  Loaded {len(neighborhoods):,} neighborhoods")

    # Generate all figures
    figure_3d_progression(cells, args.output_dir)
    figure_density_contours(cells, args.output_dir)
    figure_paga_style(cells, args.output_dir)
    figure_pseudotime_ridge(cells, args.output_dir)
    figure_celltype_alluvial(cells, args.output_dir)
    figure_hero_summary(cells, args.output_dir)

    # Manifest
    manifest = {
        "figures": [
            "fig_3d_progression.png",
            "fig_density_contours.png",
            "fig_paga_style.png",
            "fig_pseudotime_ridge.png",
            "fig_celltype_alluvial.png",
            "fig_hero_summary.png",
        ],
        "description": "Publication showcase figures for LUAD progression analysis"
    }

    with open(args.output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("Done! Generated 6 showcase figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
