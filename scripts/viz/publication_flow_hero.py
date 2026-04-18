#!/usr/bin/env python3
"""Publication-quality flow dynamics hero figure.

Nature Methods style: clean, dramatic, interpretable.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA

# =============================================================================
# STYLE - Nature Methods aesthetic
# =============================================================================

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Carefully chosen palette - distinct but harmonious
STAGE_COLORS = {
    'Normal': '#3B82F6',   # Blue
    'AAH': '#22D3EE',      # Cyan
    'AIS': '#10B981',      # Emerald
    'MIA': '#F59E0B',      # Amber
    'LUAD': '#EF4444',     # Red
}

# For trajectories - a fire gradient
TRAJECTORY_CMAP = LinearSegmentedColormap.from_list(
    'trajectory', ['#1e3a5f', '#3b82f6', '#22d3ee', '#10b981', '#f59e0b', '#ef4444']
)

# Dark theme for hero figure
DARK_BG = '#0a0a0f'
GRID_COLOR = '#1a1a2e'


def load_data(data_dir: Path):
    """Load cells data."""
    cells = pd.read_parquet(data_dir / "cells.parquet")
    return cells


def get_embeddings(df, prefix):
    """Extract embedding columns."""
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    return df[cols].values.astype(np.float32) if cols else None


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


def compute_ot_plan(source_coords, target_coords, n_samples=1500, reg=0.05):
    """Compute optimal transport plan between distributions."""
    if not HAS_OT:
        return None, None, None, None

    n_src = min(n_samples, len(source_coords))
    n_tgt = min(n_samples, len(target_coords))

    src_idx = np.random.choice(len(source_coords), n_src, replace=False)
    tgt_idx = np.random.choice(len(target_coords), n_tgt, replace=False)

    src = source_coords[src_idx]
    tgt = target_coords[tgt_idx]

    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt

    M = cdist(src, tgt, metric='sqeuclidean')
    M = M / M.max()

    try:
        T = ot.sinkhorn(a, b, M, reg=reg, numItermax=1000)
    except:
        return None, None, None, None

    # Barycentric projection
    target_bary = T @ tgt / (T.sum(axis=1, keepdims=True) + 1e-10)
    velocities = target_bary - src

    W = np.sum(T * M)

    return src, tgt, velocities, W


def sample_trajectories(coords_2d, stages, n_trajectories=200, seed=42):
    """Sample individual cell trajectories through stage progression."""
    np.random.seed(seed)

    trajectories = []

    for _ in range(n_trajectories):
        traj = []

        # Start from a random Normal cell
        normal_mask = stages == 'Normal'
        if normal_mask.sum() == 0:
            continue
        start_idx = np.random.choice(np.where(normal_mask)[0])
        traj.append(coords_2d[start_idx])
        current_pos = coords_2d[start_idx]

        # Progress through stages using nearest neighbor in next stage
        for next_stage in STAGE_ORDER[1:]:
            stage_mask = stages == next_stage
            if stage_mask.sum() == 0:
                break

            stage_coords = coords_2d[stage_mask]

            # Find nearest cell in next stage (with some noise for variety)
            dists = np.linalg.norm(stage_coords - current_pos, axis=1)

            # Weighted random selection favoring closer cells
            weights = np.exp(-dists / np.percentile(dists, 30))
            weights /= weights.sum()

            chosen_idx = np.random.choice(len(stage_coords), p=weights)
            current_pos = stage_coords[chosen_idx]
            traj.append(current_pos)

        if len(traj) >= 3:  # At least 3 stages
            trajectories.append(np.array(traj))

    return trajectories


def draw_trajectory(ax, traj, cmap, alpha=0.6, linewidth=1.5):
    """Draw a single trajectory with gradient coloring."""
    if len(traj) < 2:
        return

    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by progress (0 = start, 1 = end)
    colors = np.linspace(0, 1, len(segments))

    lc = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=linewidth)
    lc.set_array(colors)
    ax.add_collection(lc)

    # Add arrow at end
    if len(traj) >= 2:
        dx = traj[-1, 0] - traj[-2, 0]
        dy = traj[-1, 1] - traj[-2, 1]
        ax.annotate('', xy=traj[-1], xytext=traj[-2],
                   arrowprops=dict(arrowstyle='->', color=cmap(0.95),
                                  lw=linewidth, mutation_scale=8),
                   zorder=5)


def figure_flow_hero(cells, output_dir):
    """Generate the hero flow dynamics figure."""
    print("\nGenerating Flow Dynamics Hero Figure...")

    if not HAS_OT:
        print("  ERROR: POT library required. pip install POT")
        return

    # Sample and compute embeddings
    cells_s = sample_balanced(cells, n_per_stage=6000)
    fused = get_embeddings(cells_s, "z_fused_")

    if fused is None:
        print("  ERROR: No z_fused_ embeddings found")
        return

    print("  Computing UMAP...")
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        coords_2d = reducer.fit_transform(fused)
    else:
        coords_2d = PCA(n_components=2, random_state=42).fit_transform(fused)

    stages = cells_s['stage'].values

    # Compute OT between consecutive stages
    print("  Computing optimal transport...")
    ot_results = {}
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        mask1 = stages == s1
        mask2 = stages == s2

        if mask1.sum() < 50 or mask2.sum() < 50:
            continue

        src, tgt, vel, W = compute_ot_plan(coords_2d[mask1], coords_2d[mask2])
        if src is not None:
            ot_results[f'{s1}->{s2}'] = {'src': src, 'vel': vel, 'W': W}
            print(f"    {s1} -> {s2}: W = {W:.4f}")

    # Sample trajectories
    print("  Sampling trajectories...")
    trajectories = sample_trajectories(coords_2d, stages, n_trajectories=300)
    print(f"    Generated {len(trajectories)} trajectories")

    # =========================================================================
    # CREATE FIGURE - 2 panel layout
    # =========================================================================

    fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)

    # Panel A: Main flow visualization (larger)
    ax1 = fig.add_axes([0.02, 0.08, 0.58, 0.84])
    ax1.set_facecolor(DARK_BG)

    # Panel B: Stage progression with Wasserstein
    ax2 = fig.add_axes([0.64, 0.08, 0.34, 0.84])
    ax2.set_facecolor(DARK_BG)

    # =========================================================================
    # Panel A: Trajectories on dark background
    # =========================================================================

    # Subtle point cloud background
    for stage in STAGE_ORDER:
        mask = stages == stage
        ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=STAGE_COLORS[stage], s=2, alpha=0.08, rasterized=True)

    # Draw trajectories
    for traj in trajectories:
        draw_trajectory(ax1, traj, TRAJECTORY_CMAP, alpha=0.4, linewidth=0.8)

    # Stage centroids with glow effect - smart label placement to avoid overlap
    centroids = {}
    for stage in STAGE_ORDER:
        mask = stages == stage
        centroids[stage] = coords_2d[mask].mean(axis=0)

    # Compute label offsets based on local density to avoid overlaps
    def get_label_offset(centroid, all_centroids, stage_idx):
        """Smart label placement - offset away from crowded areas."""
        # Default: below the point
        offset_y = -2.0
        offset_x = 0.0

        # Check if another centroid is directly below
        for other_stage, other_centroid in all_centroids.items():
            if other_stage == STAGE_ORDER[stage_idx]:
                continue
            dx = centroid[0] - other_centroid[0]
            dy = centroid[1] - other_centroid[1]

            # If another centroid is close and below, move label to side
            if abs(dx) < 3 and -4 < dy < 0:
                offset_x = 2.5 if centroid[0] < np.mean(coords_2d[:, 0]) else -2.5
                offset_y = 0.5

        return offset_x, offset_y

    for i, stage in enumerate(STAGE_ORDER):
        centroid = centroids[stage]

        # Outer glow
        ax1.scatter(*centroid, s=800, c=STAGE_COLORS[stage], alpha=0.15, zorder=8)
        ax1.scatter(*centroid, s=400, c=STAGE_COLORS[stage], alpha=0.3, zorder=9)

        # Main point
        ax1.scatter(*centroid, s=200, c=STAGE_COLORS[stage],
                   edgecolor='white', linewidth=2, zorder=10)

        # Label with smart offset
        off_x, off_y = get_label_offset(centroid, centroids, i)
        txt = ax1.text(centroid[0] + off_x, centroid[1] + off_y, stage,
                      fontsize=11, fontweight='bold', color='white',
                      ha='center', va='top' if off_y < 0 else 'center', zorder=11)
        txt.set_path_effects([
            pe.withStroke(linewidth=4, foreground=DARK_BG)
        ])

    # Arrows between centroids showing main flow direction
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        c1, c2 = centroids[s1], centroids[s2]

        # Offset to not overlap with centroids
        direction = c2 - c1
        direction = direction / np.linalg.norm(direction)
        start = c1 + direction * 2.5
        end = c2 - direction * 2.5

        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=2.5,
            color='white',
            alpha=0.7,
            zorder=7,
            connectionstyle='arc3,rad=0.1'
        )
        ax1.add_patch(arrow)

    ax1.set_xlim(coords_2d[:, 0].min() - 2, coords_2d[:, 0].max() + 2)
    ax1.set_ylim(coords_2d[:, 1].min() - 2, coords_2d[:, 1].max() + 2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')

    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax1.set_title('Cell State Trajectories via Optimal Transport',
                 fontsize=14, fontweight='bold', color='white', pad=15)

    # =========================================================================
    # Panel B: Stage ladder with Wasserstein distances
    # =========================================================================

    # Vertical stage progression
    y_positions = np.linspace(0.9, 0.1, len(STAGE_ORDER))
    x_center = 0.3

    # Draw stages as nodes
    for i, stage in enumerate(STAGE_ORDER):
        y = y_positions[i]

        # Node
        circle = plt.Circle((x_center, y), 0.06,
                           facecolor=STAGE_COLORS[stage],
                           edgecolor='white', linewidth=2,
                           transform=ax2.transAxes, zorder=10)
        ax2.add_patch(circle)

        # Label
        ax2.text(x_center + 0.12, y, stage,
                fontsize=12, fontweight='bold', color='white',
                ha='left', va='center', transform=ax2.transAxes)

    # Draw edges with Wasserstein distances
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        key = f'{s1}->{s2}'

        y1, y2 = y_positions[i], y_positions[i+1]

        # Edge line
        ax2.plot([x_center, x_center], [y1 - 0.06, y2 + 0.06],
                color='white', linewidth=2, alpha=0.5,
                transform=ax2.transAxes, zorder=5)

        # Wasserstein distance annotation
        if key in ot_results:
            W = ot_results[key]['W']

            # Bar showing relative distance
            bar_width = W * 0.8  # Scale for visibility
            bar_y = (y1 + y2) / 2

            ax2.barh(bar_y, bar_width, height=0.04,
                    left=x_center + 0.15, color=STAGE_COLORS[s1],
                    alpha=0.8, transform=ax2.transAxes, zorder=6)

            ax2.text(x_center + 0.18 + bar_width, bar_y, f'W={W:.3f}',
                    fontsize=9, color='white', alpha=0.9,
                    ha='left', va='center', transform=ax2.transAxes)

    # Title for panel B
    ax2.text(0.5, 0.98, 'Progression Cost',
            fontsize=13, fontweight='bold', color='white',
            ha='center', va='top', transform=ax2.transAxes)

    ax2.text(0.5, 0.02, 'Wasserstein distance\nmeasures transcriptional\nreorganization between stages',
            fontsize=9, color='#888888',
            ha='center', va='bottom', transform=ax2.transAxes,
            style='italic')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')

    for spine in ax2.spines.values():
        spine.set_visible(False)

    # =========================================================================
    # Save
    # =========================================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    # High-res PNG
    fig.savefig(output_dir / "fig_flow_hero.png", dpi=300,
               facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')

    # PDF for publication
    fig.savefig(output_dir / "fig_flow_hero.pdf",
               facecolor=DARK_BG, edgecolor='none', bbox_inches='tight')

    # Also save a light version for journals that prefer white backgrounds
    fig.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')
    ax2.set_facecolor('#f8f9fa')

    # Update text colors for light background
    for txt in ax1.texts:
        txt.set_color('#1a1a2e')
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
    for txt in ax2.texts:
        txt.set_color('#1a1a2e')
    ax1.set_title('Cell State Trajectories via Optimal Transport',
                 fontsize=14, fontweight='bold', color='#1a1a2e', pad=15)

    fig.savefig(output_dir / "fig_flow_hero_light.png", dpi=300,
               facecolor='white', edgecolor='none', bbox_inches='tight')

    plt.close(fig)

    print(f"  Saved: fig_flow_hero.png, fig_flow_hero.pdf, fig_flow_hero_light.png")

    # Save metrics
    metrics = {
        'n_trajectories': len(trajectories),
        'wasserstein_distances': {k: float(v['W']) for k, v in ot_results.items()},
        'total_wasserstein': sum(v['W'] for v in ot_results.values()),
    }

    import json
    with open(output_dir / "flow_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures/dynamics"))
    args = parser.parse_args()

    print("=" * 60)
    print("Flow Dynamics Hero Figure")
    print("=" * 60)

    # Try canonical location first, fall back to data_dir
    canonical = args.data_dir / "canonical"
    if (canonical / "cells.parquet").exists():
        data_path = canonical
    elif (args.data_dir / "cells.parquet").exists():
        data_path = args.data_dir
    else:
        print(f"ERROR: No cells.parquet found in {args.data_dir} or {canonical}")
        return

    cells = load_data(data_path)
    print(f"Loaded {len(cells):,} cells")

    metrics = figure_flow_hero(cells, args.output_dir)

    if metrics:
        print(f"\nMetrics:")
        print(f"  Trajectories: {metrics['n_trajectories']}")
        print(f"  Total Wasserstein: {metrics['total_wasserstein']:.4f}")
        for trans, W in metrics['wasserstein_distances'].items():
            print(f"    {trans}: {W:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
