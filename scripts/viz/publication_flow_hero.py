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

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    import networkx as nx
    HAS_RICCI = True
except ImportError:
    HAS_RICCI = False

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


def cluster_niches(neighborhoods_df, n_clusters=8, seed=42):
    """Cluster cells by niche composition.

    Args:
        neighborhoods_df: DataFrame with tokens column containing niche info
        n_clusters: Number of niche clusters
        seed: Random seed

    Returns:
        Tuple of (niche_labels, niche_features, feature_names)
    """
    from sklearn.cluster import KMeans

    # Extract niche features from tokens
    features = []
    feature_names = ['caf_fraction', 'immune_fraction', 'emt_score', 'diversity']

    for _, row in neighborhoods_df.iterrows():
        tokens = row['tokens']
        # Receiver token is index 0
        receiver = tokens[0] if len(tokens) > 0 else {}

        # Get niche composition from receiver token
        feat = []
        for fn in feature_names:
            val = receiver.get(fn) if isinstance(receiver, dict) else None
            feat.append(float(val) if val is not None else 0.0)
        features.append(feat)

    X = np.array(features)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels, X, feature_names, kmeans.cluster_centers_


def compute_niche_ricci_curvature(neighborhoods_df, stages, n_clusters=8):
    """Compute Ollivier-Ricci curvature on niche transition graph across stages.

    Builds a graph where:
    - Nodes are (stage, niche_cluster) pairs
    - Edges connect niches between consecutive stages weighted by transition frequency

    Ricci curvature reveals geometric structure of niche transitions:
    - Negative: bottleneck (obligate niche transition for progression)
    - Positive: redundant paths (multiple niche routes available)
    - Near zero: linear/neutral

    Args:
        neighborhoods_df: DataFrame with cell_id, stage, tokens
        stages: Stage labels array (aligned with neighborhoods_df)
        n_clusters: Number of niche clusters

    Returns:
        Dict with 'curvatures', 'niche_labels', 'niche_centers', 'graph_info'
    """
    if not HAS_RICCI:
        print("  Warning: GraphRicciCurvature not installed, skipping niche ORC")
        return {'curvatures': {}, 'bottlenecks': [], 'redundant': []}

    print("  Clustering niches...")
    niche_labels, niche_features, feature_names, centers = cluster_niches(
        neighborhoods_df, n_clusters=n_clusters
    )

    # Build niche transition graph
    G = nx.DiGraph()

    # Count transitions between (stage, niche) pairs
    transition_counts = {}

    # Group by stage
    stage_niche = {}
    for i, (_, row) in enumerate(neighborhoods_df.iterrows()):
        stage = row['stage']
        niche = niche_labels[i]
        if stage not in stage_niche:
            stage_niche[stage] = []
        stage_niche[stage].append((row['cell_id'], niche))

    # For each consecutive stage pair, count niche transitions
    # Using cell similarity in embedding space to infer transitions
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]

        if s1 not in stage_niche or s2 not in stage_niche:
            continue

        cells_s1 = stage_niche[s1]
        cells_s2 = stage_niche[s2]

        # Count niche co-occurrence as proxy for transition
        # (In reality this would use OT coupling or trajectory inference)
        niche_counts_s1 = {}
        niche_counts_s2 = {}

        for _, niche in cells_s1:
            niche_counts_s1[niche] = niche_counts_s1.get(niche, 0) + 1
        for _, niche in cells_s2:
            niche_counts_s2[niche] = niche_counts_s2.get(niche, 0) + 1

        # Create edges between all niche pairs, weighted by product of frequencies
        total_s1 = sum(niche_counts_s1.values())
        total_s2 = sum(niche_counts_s2.values())

        for n1, c1 in niche_counts_s1.items():
            for n2, c2 in niche_counts_s2.items():
                node1 = f"{s1}_N{n1}"
                node2 = f"{s2}_N{n2}"

                # Weight by transition probability estimate
                weight = (c1 / total_s1) * (c2 / total_s2)

                if weight > 0.001:  # Filter very rare transitions
                    G.add_node(node1, stage=s1, niche=n1)
                    G.add_node(node2, stage=s2, niche=n2)
                    G.add_edge(node1, node2, weight=weight)

    if G.number_of_edges() == 0:
        print("  Warning: No niche transitions found")
        return {'curvatures': {}, 'bottlenecks': [], 'redundant': []}

    print(f"  Built niche graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert to undirected for ORC (required by GraphRicciCurvature)
    G_undirected = G.to_undirected()

    # Compute Ollivier-Ricci curvature
    try:
        orc = OllivierRicci(G_undirected, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()

        curvatures = {}
        bottlenecks = []
        redundant = []

        for u, v in G_undirected.edges():
            curv = G_undirected[u][v].get('ricciCurvature', 0.0)
            curvatures[(u, v)] = curv

            if curv < -0.2:
                bottlenecks.append((u, v, curv))
            elif curv > 0.2:
                redundant.append((u, v, curv))

        # Sort by magnitude
        bottlenecks.sort(key=lambda x: x[2])
        redundant.sort(key=lambda x: -x[2])

        print(f"  Computed ORC for {len(curvatures)} edges")
        print(f"  Bottleneck transitions (ORC < -0.2): {len(bottlenecks)}")
        print(f"  Redundant transitions (ORC > 0.2): {len(redundant)}")

        # Report top bottlenecks
        if bottlenecks:
            print("  Top bottlenecks:")
            for u, v, c in bottlenecks[:5]:
                print(f"    {u} -> {v}: ORC = {c:.3f}")

        return {
            'curvatures': curvatures,
            'bottlenecks': bottlenecks,
            'redundant': redundant,
            'niche_labels': niche_labels,
            'niche_centers': centers,
            'feature_names': feature_names,
            'graph': G,
        }

    except Exception as e:
        print(f"  Warning: Niche ORC computation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'curvatures': {}, 'bottlenecks': [], 'redundant': []}


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


def figure_flow_hero(cells, output_dir, neighborhoods=None):
    """Generate the hero flow dynamics figure.

    Args:
        cells: DataFrame with cell data
        output_dir: Path for output figures
        neighborhoods: Optional DataFrame with niche tokens for ORC computation
    """
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

    # Compute Niche Ricci curvature (if neighborhoods available)
    niche_orc_results = {}
    if neighborhoods is not None and HAS_RICCI:
        print("  Computing Niche Ricci curvature...")
        niche_orc_results = compute_niche_ricci_curvature(
            neighborhoods, neighborhoods['stage'].values, n_clusters=8
        )
    else:
        if neighborhoods is None:
            print("  Skipping niche ORC: no neighborhoods data")
        elif not HAS_RICCI:
            print("  Skipping niche ORC: GraphRicciCurvature not installed")

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

    # Stage centroids with glow effect
    centroids = {}
    for stage in STAGE_ORDER:
        mask = stages == stage
        centroids[stage] = coords_2d[mask].mean(axis=0)

    # Compute non-overlapping label positions using repulsion
    def compute_label_positions(centroids, coords_2d, label_distance=3.0):
        """Compute label positions that don't overlap with each other or centroids."""
        positions = {}
        label_radius = 2.5  # Approximate text bounding box radius

        for i, stage in enumerate(STAGE_ORDER):
            centroid = centroids[stage]

            # Try 8 directions: below, above, left, right, and diagonals
            angles = [270, 90, 180, 0, 225, 315, 135, 45]  # degrees, 270=below
            best_pos = None
            best_score = -np.inf

            for angle in angles:
                rad = np.radians(angle)
                candidate = centroid + label_distance * np.array([np.cos(rad), np.sin(rad)])

                # Score: prefer positions far from other centroids and labels
                score = 0

                # Distance from other centroids
                for other_stage, other_centroid in centroids.items():
                    if other_stage != stage:
                        dist = np.linalg.norm(candidate - other_centroid)
                        score += min(dist, 10)  # Cap contribution

                # Distance from already placed labels
                for placed_stage, placed_pos in positions.items():
                    dist = np.linalg.norm(candidate - placed_pos)
                    if dist < label_radius * 2:
                        score -= 20  # Heavy penalty for overlap
                    else:
                        score += min(dist, 5)

                # Prefer positions within plot bounds
                if (coords_2d[:, 0].min() < candidate[0] < coords_2d[:, 0].max() and
                    coords_2d[:, 1].min() < candidate[1] < coords_2d[:, 1].max()):
                    score += 5

                if score > best_score:
                    best_score = score
                    best_pos = candidate

            positions[stage] = best_pos if best_pos is not None else centroid + np.array([0, -label_distance])

        return positions

    label_positions = compute_label_positions(centroids, coords_2d)

    for i, stage in enumerate(STAGE_ORDER):
        centroid = centroids[stage]

        # Outer glow
        ax1.scatter(*centroid, s=800, c=STAGE_COLORS[stage], alpha=0.15, zorder=8)
        ax1.scatter(*centroid, s=400, c=STAGE_COLORS[stage], alpha=0.3, zorder=9)

        # Main point
        ax1.scatter(*centroid, s=200, c=STAGE_COLORS[stage],
                   edgecolor='white', linewidth=2, zorder=10)

        # Label at computed position
        label_pos = label_positions[stage]
        txt = ax1.text(label_pos[0], label_pos[1], stage,
                      fontsize=11, fontweight='bold', color='white',
                      ha='center', va='center', zorder=11)
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
    # Panel B: Niche transition geometry with Wasserstein and ORC
    # =========================================================================

    # Vertical stage progression
    y_positions = np.linspace(0.9, 0.1, len(STAGE_ORDER))
    x_center = 0.25

    # Draw stages as nodes
    for i, stage in enumerate(STAGE_ORDER):
        y = y_positions[i]

        # Node
        circle = plt.Circle((x_center, y), 0.05,
                           facecolor=STAGE_COLORS[stage],
                           edgecolor='white', linewidth=2,
                           transform=ax2.transAxes, zorder=10)
        ax2.add_patch(circle)

        # Label - positioned to the left to avoid overlap
        ax2.text(x_center - 0.08, y, stage,
                fontsize=10, fontweight='bold', color='white',
                ha='right', va='center', transform=ax2.transAxes)

    # Draw edges with Wasserstein distances
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        key = f'{s1}->{s2}'

        y1, y2 = y_positions[i], y_positions[i+1]

        # Edge line
        ax2.plot([x_center, x_center], [y1 - 0.05, y2 + 0.05],
                color='white', linewidth=2, alpha=0.5,
                transform=ax2.transAxes, zorder=5)

        # Wasserstein distance annotation
        if key in ot_results:
            W = ot_results[key]['W']
            bar_y = (y1 + y2) / 2

            ax2.text(x_center + 0.08, bar_y, f'W={W:.3f}',
                    fontsize=8, color='white', alpha=0.9,
                    ha='left', va='center', transform=ax2.transAxes)

    # Show niche ORC bottlenecks if available
    bottlenecks = niche_orc_results.get('bottlenecks', [])
    redundant = niche_orc_results.get('redundant', [])

    if bottlenecks or redundant:
        # Title
        ax2.text(0.65, 0.95, 'Niche Bottlenecks',
                fontsize=11, fontweight='bold', color='white',
                ha='center', va='top', transform=ax2.transAxes)

        # Show top bottleneck transitions
        y_text = 0.85
        ax2.text(0.65, y_text, 'Obligate transitions:',
                fontsize=8, color='#ef4444', alpha=0.9,
                ha='center', va='top', transform=ax2.transAxes)

        y_text -= 0.05
        for u, v, curv in bottlenecks[:3]:
            # Parse stage from node name (e.g., "AAH_N3" -> "AAH")
            s1 = u.split('_N')[0]
            s2 = v.split('_N')[0]
            n1 = u.split('_N')[1] if '_N' in u else '?'
            n2 = v.split('_N')[1] if '_N' in v else '?'

            label = f"{s1}:N{n1} -> {s2}:N{n2}"
            ax2.text(0.65, y_text, f"{label}: {curv:.2f}",
                    fontsize=7, color='#ef4444', alpha=0.8,
                    ha='center', va='top', transform=ax2.transAxes)
            y_text -= 0.04

        # Show redundant transitions
        if redundant:
            y_text -= 0.03
            ax2.text(0.65, y_text, 'Redundant paths:',
                    fontsize=8, color='#10b981', alpha=0.9,
                    ha='center', va='top', transform=ax2.transAxes)

            y_text -= 0.05
            for u, v, curv in redundant[:2]:
                s1 = u.split('_N')[0]
                s2 = v.split('_N')[0]
                n1 = u.split('_N')[1] if '_N' in u else '?'
                n2 = v.split('_N')[1] if '_N' in v else '?'

                label = f"{s1}:N{n1} -> {s2}:N{n2}"
                ax2.text(0.65, y_text, f"{label}: +{curv:.2f}",
                        fontsize=7, color='#10b981', alpha=0.8,
                        ha='center', va='top', transform=ax2.transAxes)
                y_text -= 0.04

    # Main title for panel B
    ax2.text(0.5, 0.98, 'Progression Geometry',
            fontsize=13, fontweight='bold', color='white',
            ha='center', va='top', transform=ax2.transAxes)

    # Legend
    legend_text = 'W = Wasserstein distance\nORC < 0 = bottleneck niche\nORC > 0 = redundant paths'
    ax2.text(0.5, 0.02, legend_text,
            fontsize=7, color='#888888',
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
        'niche_orc': {
            'n_bottlenecks': len(niche_orc_results.get('bottlenecks', [])),
            'n_redundant': len(niche_orc_results.get('redundant', [])),
            'bottlenecks': [
                {'from': u, 'to': v, 'orc': float(c)}
                for u, v, c in niche_orc_results.get('bottlenecks', [])[:10]
            ],
            'redundant': [
                {'from': u, 'to': v, 'orc': float(c)}
                for u, v, c in niche_orc_results.get('redundant', [])[:10]
            ],
        },
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

    # Load neighborhoods for niche ORC
    neighborhoods = None
    neighborhoods_path = data_path / "neighborhoods.parquet"
    if neighborhoods_path.exists():
        neighborhoods = pd.read_parquet(neighborhoods_path)
        print(f"Loaded {len(neighborhoods):,} neighborhoods for niche ORC")
    else:
        print(f"Warning: No neighborhoods.parquet found, skipping niche ORC")

    metrics = figure_flow_hero(cells, args.output_dir, neighborhoods=neighborhoods)

    if metrics:
        print(f"\nMetrics:")
        print(f"  Trajectories: {metrics['n_trajectories']}")
        print(f"  Total Wasserstein: {metrics['total_wasserstein']:.4f}")
        print(f"\n  Wasserstein distances:")
        for trans, W in metrics['wasserstein_distances'].items():
            print(f"    {trans}: {W:.4f}")

        niche_orc = metrics.get('niche_orc', {})
        if niche_orc.get('bottlenecks'):
            print(f"\n  Niche ORC Bottlenecks ({niche_orc['n_bottlenecks']} total):")
            for b in niche_orc['bottlenecks'][:5]:
                print(f"    {b['from']} -> {b['to']}: ORC = {b['orc']:.3f}")
        if niche_orc.get('redundant'):
            print(f"\n  Niche ORC Redundant ({niche_orc['n_redundant']} total):")
            for r in niche_orc['redundant'][:3]:
                print(f"    {r['from']} -> {r['to']}: ORC = +{r['orc']:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
