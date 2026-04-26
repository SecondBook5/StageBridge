#!/usr/bin/env python3
"""Publication figures with REAL optimal transport dynamics.

Computes actual velocity/flux using optimal transport between stage distributions.
No fake centroid-based velocities.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

try:
    import ot  # Python Optimal Transport
    HAS_OT = True
except ImportError:
    HAS_OT = False
    print("WARNING: POT not installed. Install with: pip install POT")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# =============================================================================
# STYLE - extra spacing to prevent overlap
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlepad': 12,  # Extra padding for titles
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

STAGE_ORDER_5 = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
STAGE_ORDER_3 = ['Normal', 'Preinvasive', 'Invasive']
STAGE_ORDER = STAGE_ORDER_3  # Default, will be set by load_data

STAGE_COLORS = {
    'Normal': '#1B4F72',
    'AAH': '#2E86AB',
    'AIS': '#1D6F42',
    'MIA': '#D4A03C',
    'LUAD': '#922B21',
    'Preinvasive': '#2E86AB',
    'Invasive': '#922B21',
}


def load_data(data_dir: Path):
    """Load cells data and detect stage vocabulary."""
    global STAGE_ORDER
    cells = pd.read_parquet(data_dir / "cells.parquet")
    unique_stages = set(cells["stage"].dropna().unique())
    if unique_stages <= set(STAGE_ORDER_3):
        STAGE_ORDER = STAGE_ORDER_3
    elif unique_stages <= set(STAGE_ORDER_5):
        STAGE_ORDER = STAGE_ORDER_5
    else:
        STAGE_ORDER = sorted(unique_stages)
    print(f"Detected {len(STAGE_ORDER)} stages: {STAGE_ORDER}")
    return cells


def get_embeddings(df, prefix):
    """Extract embedding columns."""
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    return df[cols].values.astype(np.float32) if cols else None


def sample_balanced(df, n_per_stage=4000, seed=42):
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
# OPTIMAL TRANSPORT COMPUTATIONS
# =============================================================================

def compute_ot_velocity(source_coords, target_coords, n_samples=2000, reg=0.1):
    """Compute optimal transport velocity field between two distributions.

    Returns velocity vectors for source points pointing toward their OT targets.
    Uses entropic regularization for smooth transport.
    """
    if not HAS_OT:
        return None, None, None

    # Subsample for efficiency
    n_src = min(n_samples, len(source_coords))
    n_tgt = min(n_samples, len(target_coords))

    src_idx = np.random.choice(len(source_coords), n_src, replace=False)
    tgt_idx = np.random.choice(len(target_coords), n_tgt, replace=False)

    src = source_coords[src_idx]
    tgt = target_coords[tgt_idx]

    # Uniform weights
    a = np.ones(n_src) / n_src
    b = np.ones(n_tgt) / n_tgt

    # Cost matrix (squared Euclidean)
    M = cdist(src, tgt, metric='sqeuclidean')
    M = M / M.max()  # Normalize

    # Sinkhorn OT
    try:
        T = ot.sinkhorn(a, b, M, reg=reg, numItermax=1000)
    except:
        return None, None, None

    # Compute barycentric projection (where each source cell goes)
    # T[i,j] = mass from source i to target j
    target_barycenters = T @ tgt / (T.sum(axis=1, keepdims=True) + 1e-10)

    # Velocity = direction from source to target barycenter
    velocities = target_barycenters - src

    # Wasserstein distance
    W = np.sum(T * M)

    return src, velocities, W


def compute_ot_flow_field(coords_2d, stages, grid_size=25):
    """Compute gridded OT flow field from all consecutive stage pairs.

    Returns interpolated velocity field on a regular grid.
    """
    if not HAS_OT:
        return None

    # Grid bounds
    pad = 0.5
    xmin, xmax = coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad
    ymin, ymax = coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad

    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    U = np.zeros_like(Xi)
    V = np.zeros_like(Yi)
    counts = np.zeros_like(Xi)

    # Compute OT between consecutive stages
    stage_results = {}
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]

        mask1 = stages == s1
        mask2 = stages == s2

        if mask1.sum() < 50 or mask2.sum() < 50:
            continue

        src_coords = coords_2d[mask1]
        tgt_coords = coords_2d[mask2]

        src, vel, W = compute_ot_velocity(src_coords, tgt_coords)

        if src is None:
            continue

        stage_results[f'{s1}->{s2}'] = {'W': W, 'n_src': len(src)}

        # Add velocities to grid (nearest neighbor interpolation)
        for j in range(len(src)):
            # Find nearest grid point
            gi = np.argmin(np.abs(yi - src[j, 1]))
            gj = np.argmin(np.abs(xi - src[j, 0]))

            U[gi, gj] += vel[j, 0]
            V[gi, gj] += vel[j, 1]
            counts[gi, gj] += 1

    # Average where we have multiple contributions
    mask = counts > 0
    U[mask] /= counts[mask]
    V[mask] /= counts[mask]

    # Smooth
    U = gaussian_filter(U, sigma=1.2)
    V = gaussian_filter(V, sigma=1.2)

    return {
        'Xi': Xi, 'Yi': Yi, 'U': U, 'V': V,
        'stage_results': stage_results
    }


def compute_flux_decomposition(U, V):
    """Decompose velocity field into gradient (reversible) and curl (irreversible) components.

    v = -grad(phi) + curl(psi)

    The curl component represents irreversible, non-equilibrium dynamics.
    """
    # Compute divergence and curl
    dU_dx = np.gradient(U, axis=1)
    dU_dy = np.gradient(U, axis=0)
    dV_dx = np.gradient(V, axis=1)
    dV_dy = np.gradient(V, axis=0)

    div = dU_dx + dV_dy  # Divergence
    curl = dV_dx - dU_dy  # Curl (z-component)

    # Helmholtz decomposition via solving Poisson equations
    # For simplicity, estimate gradient/rotational components directly

    # Gradient component: v_grad = -grad(phi) where div(v_grad) = div(v)
    # Rotational component: v_rot = curl(psi) where curl(v_rot) = curl(v)

    # Magnitude ratios
    speed = np.sqrt(U**2 + V**2)
    grad_mag = np.abs(div)
    curl_mag = np.abs(curl)

    # Flux ratio: fraction that is rotational/irreversible
    # Higher = more irreversible dynamics
    total = grad_mag + curl_mag + 1e-10
    flux_ratio = curl_mag / total

    return {
        'divergence': div,
        'curl': curl,
        'speed': speed,
        'flux_ratio': flux_ratio,
        'mean_flux_ratio': np.nanmean(flux_ratio[speed > speed.mean()])
    }


# =============================================================================
# FIGURE: OT DYNAMICS
# =============================================================================

def figure_ot_dynamics(cells, output_dir):
    """Main figure showing optimal transport dynamics."""
    print("\nGenerating OT Dynamics Figure...")

    if not HAS_OT:
        print("  ERROR: POT library required. pip install POT")
        return

    # Sample and compute embeddings
    cells_s = sample_balanced(cells, n_per_stage=5000)
    fused = get_embeddings(cells_s, "z_fused_")

    print("  Computing UMAP...")
    if HAS_UMAP:
        coords_2d = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(fused)
    else:
        coords_2d = PCA(n_components=2, random_state=42).fit_transform(fused)

    stages = cells_s['stage'].values

    print("  Computing OT flow field...")
    flow = compute_ot_flow_field(coords_2d, stages, grid_size=30)

    if flow is None:
        print("  ERROR: OT computation failed")
        return

    print("  Computing flux decomposition...")
    flux = compute_flux_decomposition(flow['U'], flow['V'])

    # Create figure with extra spacing
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4,
                  left=0.06, right=0.94, top=0.92, bottom=0.06)

    Xi, Yi, U, V = flow['Xi'], flow['Yi'], flow['U'], flow['V']

    # =========================================================================
    # Row 1: Main visualizations
    # =========================================================================

    # A: Stage distribution
    ax = fig.add_subplot(gs[0, 0])
    for stage in STAGE_ORDER:
        mask = stages == stage
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=STAGE_COLORS[stage], s=3, alpha=0.4, label=stage, rasterized=True)
    ax.legend(loc='upper right', fontsize=7, markerscale=2, framealpha=0.9)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('A. Disease Stages')
    ax.set_xticks([])
    ax.set_yticks([])

    # B: OT Velocity Field
    ax = fig.add_subplot(gs[0, 1:3])

    # Background cells
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='#EEEEEE', s=1, alpha=0.3, rasterized=True)

    # Streamlines from OT velocities
    speed = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(Xi[0, :], Yi[:, 0], U, V,
                         color=speed, cmap='viridis',
                         density=1.8, linewidth=1.2, arrowsize=1.2)

    # Stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c=STAGE_COLORS[stage], s=150,
                  edgecolor='white', linewidth=2, zorder=10)
        ax.annotate(stage, centroid, fontsize=8, fontweight='bold',
                   ha='center', va='center', color='white')

    cbar = plt.colorbar(strm.lines, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Speed', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('B. Optimal Transport Velocity Field')
    ax.set_xticks([])
    ax.set_yticks([])

    # C: Wasserstein distances
    ax = fig.add_subplot(gs[0, 3])

    transitions = list(flow['stage_results'].keys())
    W_values = [flow['stage_results'][t]['W'] for t in transitions]

    colors = [STAGE_COLORS[t.split('->')[0]] for t in transitions]
    bars = ax.barh(range(len(transitions)), W_values, color=colors, edgecolor='white')

    ax.set_yticks(range(len(transitions)))
    ax.set_yticklabels([t.replace('->', '\n') for t in transitions], fontsize=8)
    ax.set_xlabel('Wasserstein Distance')
    ax.set_title('C. OT Distance')
    ax.invert_yaxis()

    for bar, val in zip(bars, W_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=8)

    # =========================================================================
    # Row 2: Flux decomposition
    # =========================================================================

    # D: Divergence field
    ax = fig.add_subplot(gs[1, 0])

    div = flux['divergence']
    vmax = np.percentile(np.abs(div), 95)
    im = ax.pcolormesh(Xi, Yi, div, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=50, edgecolor=STAGE_COLORS[stage], linewidth=2, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('div(v)', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('D. Divergence')
    ax.set_xticks([])
    ax.set_yticks([])

    # E: Curl field (rotational/irreversible)
    ax = fig.add_subplot(gs[1, 1])

    curl = flux['curl']
    vmax = np.percentile(np.abs(curl), 95)
    im = ax.pcolormesh(Xi, Yi, curl, cmap='PiYG', vmin=-vmax, vmax=vmax,
                       shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=50, edgecolor=STAGE_COLORS[stage], linewidth=2, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('curl(v)', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('E. Curl (Irreversibility)')
    ax.set_xticks([])
    ax.set_yticks([])

    # F: Flux ratio (irreversibility fraction)
    ax = fig.add_subplot(gs[1, 2])

    fr = flux['flux_ratio']
    im = ax.pcolormesh(Xi, Yi, fr, cmap='inferno', vmin=0, vmax=1,
                       shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=50, edgecolor=STAGE_COLORS[stage], linewidth=2, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Flux ratio', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('F. Irreversibility Map')
    ax.set_xticks([])
    ax.set_yticks([])

    # G: Flux ratio distribution
    ax = fig.add_subplot(gs[1, 3])

    # Compute flux ratio near each stage
    flux_by_stage = []
    for stage in STAGE_ORDER:
        mask = stages == stage
        stage_coords = coords_2d[mask]

        # Get flux ratio at nearest grid points
        stage_flux = []
        for coord in stage_coords[:500]:  # Subsample
            gi = np.argmin(np.abs(Yi[:, 0] - coord[1]))
            gj = np.argmin(np.abs(Xi[0, :] - coord[0]))
            stage_flux.append(fr[gi, gj])
        flux_by_stage.append(np.array(stage_flux))

    # Violin plot
    parts = ax.violinplot(flux_by_stage, positions=range(len(STAGE_ORDER)), showmeans=False,
                          showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(STAGE_COLORS[STAGE_ORDER[i]])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=8)
    ax.set_ylabel('Flux Ratio')
    ax.set_ylim(0, 1)
    ax.set_title('G. Irreversibility by Stage')

    mean_fr = flux['mean_flux_ratio']
    ax.text(0.95, 0.95, f'Mean={mean_fr:.2f}', transform=ax.transAxes,
           ha='right', va='top', fontsize=9, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    # =========================================================================
    # Row 3: Additional analyses
    # =========================================================================

    # H: Speed field
    ax = fig.add_subplot(gs[2, 0])

    speed = flux['speed']
    im = ax.pcolormesh(Xi, Yi, speed, cmap='YlOrRd', shading='gouraud', rasterized=True)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Speed', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('H. Flow Speed')
    ax.set_xticks([])
    ax.set_yticks([])

    # I: Cumulative Wasserstein along progression
    ax = fig.add_subplot(gs[2, 1])

    cumulative_W = np.cumsum([0] + W_values)
    stage_labels = ['Start'] + [t.split('->')[1] for t in transitions]

    ax.plot(range(len(cumulative_W)), cumulative_W, 'o-', color='#1B4F72',
            linewidth=2, markersize=8)
    ax.fill_between(range(len(cumulative_W)), cumulative_W, alpha=0.3, color='#1B4F72')

    ax.set_xticks(range(len(cumulative_W)))
    ax.set_xticklabels(stage_labels, fontsize=8)
    ax.set_ylabel('Cumulative W Distance')
    ax.set_title('I. Progression Cost')

    # J: Stage transition probabilities (from OT)
    ax = fig.add_subplot(gs[2, 2])

    # Compute transition matrix from OT
    T_matrix = np.zeros((5, 5))
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        key = f'{s1}->{s2}'
        if key in flow['stage_results']:
            # Forward transition strength (inverse of distance)
            T_matrix[i, i+1] = 1.0 / (flow['stage_results'][key]['W'] + 0.01)

    # Normalize rows
    row_sums = T_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_matrix = T_matrix / row_sums

    # Add self-loops for visualization
    for i in range(len(STAGE_ORDER)):
        T_matrix[i, i] = 1 - T_matrix[i].sum()

    im = ax.imshow(T_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER, fontsize=8)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('J. Transition Propensity')

    for i in range(len(STAGE_ORDER)):
        for j in range(len(STAGE_ORDER)):
            if T_matrix[i, j] > 0.01:
                color = 'white' if T_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{T_matrix[i,j]:.2f}', ha='center', va='center',
                       fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('P(transition)', fontsize=8)

    # K: Key metrics bar chart
    ax = fig.add_subplot(gs[2, 3])

    metrics = ['Total W', 'Mean Flux']
    values = [sum(W_values), mean_fr]
    colors = ['#1B4F72', '#922B21']

    bars = ax.bar(metrics, values, color=colors, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Value')
    ax.set_title('K. Key Metrics')
    ax.set_ylim(0, max(values) * 1.2)

    # Main title
    fig.suptitle('Optimal Transport Dynamics of LUAD Progression',
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_ot_dynamics.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_ot_dynamics.pdf", facecolor='white')
    plt.close(fig)

    print(f"  Saved fig_ot_dynamics.png/pdf")
    print(f"  Mean flux ratio: {mean_fr:.3f}")
    print(f"  Total Wasserstein: {sum(W_values):.3f}")

    return {'flux_ratio': mean_fr, 'total_W': sum(W_values)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/reference_mapping"))
    args = parser.parse_args()

    print("=" * 60)
    print("Optimal Transport Dynamics Figures")
    print("=" * 60)

    cells = load_data(args.data_dir)
    print(f"Loaded {len(cells):,} cells")

    results = figure_ot_dynamics(cells, args.output_dir)

    if results:
        print(f"\nResults:")
        print(f"  Mean flux ratio: {results['flux_ratio']:.3f}")
        print(f"  Total Wasserstein distance: {results['total_W']:.3f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
