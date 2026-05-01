#!/usr/bin/env python3
"""Generate individual publication-quality figures (one panel per file).

Based on StageBridge_V1 visualization scripts. Each panel saved separately
for manual assembly into multi-panel figures.

Supports multiple embedding spaces for OT dynamics:
- UMAP (default): Good for visualization, may distort distances
- PHATE: Better for trajectories, preserves diffusion geometry
- Spatial: Actual tissue coordinates (x_spatial, y_spatial)
- Latent: Raw 40d fused embedding (OT computed here, projected to 2D for viz)

Usage:
    python scripts/generate_individual_figures.py --data-dir /path/to/canonical --output-dir figures/panels
    python scripts/generate_individual_figures.py --figures ot_velocity,curl,flux_ratio  # Specific panels
    python scripts/generate_individual_figures.py --embedding phate  # Use PHATE instead of UMAP
    python scripts/generate_individual_figures.py --embedding spatial  # OT in tissue space
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import argparse

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    print("WARNING: POT not installed. pip install POT")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap not installed. Using PCA fallback.")

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False
    print("WARNING: phate not installed. pip install phate")

# =============================================================================
# STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

STAGE_ORDER_5 = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
STAGE_ORDER_3 = ['Normal', 'Preinvasive', 'Invasive']
STAGE_ORDER = STAGE_ORDER_3

STAGE_COLORS = {
    'Normal': '#1B4F72',
    'AAH': '#2E86AB',
    'AIS': '#1D6F42',
    'MIA': '#D4A03C',
    'LUAD': '#922B21',
    'Preinvasive': '#2E86AB',
    'Invasive': '#922B21',
}

CELL_TYPE_COLORS = {
    'T cell lineage': '#E64B35',
    'Myeloid lineage': '#4DBBD5',
    'Fibroblast lineage': '#00A087',
    'Epithelial': '#3C5488',
    'B cell lineage': '#F39B7F',
    'Endothelial': '#8491B4',
    'Mast cells': '#91D1C2',
    'DC': '#DC7EC0',
    'Capillary': '#7E6148',
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


def get_embeddings(df, prefix="z_fused_"):
    """Extract embedding columns or array column."""
    # Check for array column first
    if "z_fused" in df.columns and hasattr(df["z_fused"].iloc[0], "__len__"):
        return np.array([np.array(x) for x in df["z_fused"].values])
    # Fall back to prefix columns
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


def compute_umap(fused, n_neighbors=30, min_dist=0.3, seed=42):
    """Compute UMAP or PCA fallback."""
    if HAS_UMAP:
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(fused)
    return PCA(n_components=2, random_state=seed).fit_transform(fused)


def compute_phate(fused, n_components=2, knn=15, seed=42):
    """Compute PHATE embedding - better for trajectories than UMAP."""
    if HAS_PHATE:
        phate_op = phate.PHATE(n_components=n_components, knn=knn, n_jobs=-1, random_state=seed)
        return phate_op.fit_transform(fused)
    print("  PHATE not available, falling back to UMAP")
    return compute_umap(fused)


def get_spatial_coords(cells_df):
    """Extract spatial coordinates for cells that have them."""
    if 'x_spatial' in cells_df.columns and 'y_spatial' in cells_df.columns:
        valid = cells_df['x_spatial'].notna() & cells_df['y_spatial'].notna()
        return cells_df[valid], cells_df.loc[valid, ['x_spatial', 'y_spatial']].values
    return None, None


def compute_embedding(cells_s, fused, embedding_type='umap', seed=42):
    """Compute 2D embedding based on type.

    Args:
        cells_s: DataFrame with cell data
        fused: High-dimensional embeddings
        embedding_type: 'umap', 'phate', 'spatial', or 'latent'
        seed: Random seed

    Returns:
        coords_2d: 2D coordinates
        cells_subset: Potentially filtered cells (for spatial)
        label: String label for axis/title
    """
    if embedding_type == 'umap':
        coords_2d = compute_umap(fused, seed=seed)
        return coords_2d, cells_s, 'UMAP'

    elif embedding_type == 'phate':
        coords_2d = compute_phate(fused, seed=seed)
        return coords_2d, cells_s, 'PHATE'

    elif embedding_type == 'spatial':
        cells_spatial, coords = get_spatial_coords(cells_s)
        if coords is None:
            print("  No spatial coordinates available, falling back to UMAP")
            coords_2d = compute_umap(fused, seed=seed)
            return coords_2d, cells_s, 'UMAP'
        # Need to also subset the embeddings for later use
        return coords, cells_spatial, 'Spatial (μm)'

    elif embedding_type == 'latent':
        # Compute OT in full latent space, project to 2D for visualization
        # Use PCA to get 2D projection of the 40d space
        coords_2d = PCA(n_components=2, random_state=seed).fit_transform(fused)
        return coords_2d, cells_s, 'Latent PC'

    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# =============================================================================
# OPTIMAL TRANSPORT
# =============================================================================

def compute_ot_velocity(source_coords, target_coords, n_samples=2000, reg=0.1):
    """Compute OT velocity field between two distributions."""
    if not HAS_OT:
        return None, None, None

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
        return None, None, None

    target_barycenters = T @ tgt / (T.sum(axis=1, keepdims=True) + 1e-10)
    velocities = target_barycenters - src
    W = np.sum(T * M)

    return src, velocities, W


def compute_ot_flow_field(coords_2d, stages, grid_size=25):
    """Compute gridded OT flow field from consecutive stage pairs."""
    if not HAS_OT:
        return None

    pad = 0.5
    xmin, xmax = coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad
    ymin, ymax = coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad

    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    U = np.zeros_like(Xi)
    V = np.zeros_like(Yi)
    counts = np.zeros_like(Xi)

    stage_results = {}
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        mask1 = stages == s1
        mask2 = stages == s2

        if mask1.sum() < 50 or mask2.sum() < 50:
            continue

        src, vel, W = compute_ot_velocity(coords_2d[mask1], coords_2d[mask2])
        if src is None:
            continue

        stage_results[f'{s1}->{s2}'] = {'W': W, 'n_src': len(src)}

        for j in range(len(src)):
            gi = np.argmin(np.abs(yi - src[j, 1]))
            gj = np.argmin(np.abs(xi - src[j, 0]))
            U[gi, gj] += vel[j, 0]
            V[gi, gj] += vel[j, 1]
            counts[gi, gj] += 1

    mask = counts > 0
    U[mask] /= counts[mask]
    V[mask] /= counts[mask]

    U = gaussian_filter(U, sigma=1.2)
    V = gaussian_filter(V, sigma=1.2)

    return {'Xi': Xi, 'Yi': Yi, 'U': U, 'V': V, 'stage_results': stage_results}


def compute_flux_decomposition(U, V):
    """Helmholtz decomposition: gradient (reversible) + curl (irreversible)."""
    dU_dx = np.gradient(U, axis=1)
    dU_dy = np.gradient(U, axis=0)
    dV_dx = np.gradient(V, axis=1)
    dV_dy = np.gradient(V, axis=0)

    div = dU_dx + dV_dy
    curl = dV_dx - dU_dy

    speed = np.sqrt(U**2 + V**2)
    grad_mag = np.abs(div)
    curl_mag = np.abs(curl)

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
# INDIVIDUAL FIGURE FUNCTIONS
# =============================================================================

def save_fig(fig, output_dir, name):
    """Save figure as PNG and PDF."""
    fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(output_dir / f"{name}.pdf", facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.png/pdf")


def fig_stage_umap(coords_2d, stages, output_dir, embed_label='UMAP'):
    """Embedding colored by disease stage."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for stage in STAGE_ORDER:
        mask = stages == stage
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)
    ax.legend(loc='upper right', markerscale=3, framealpha=0.9)
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    ax.set_title('Disease Stages in Embedding Space')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, output_dir, f'stage_{embed_label.lower().replace(" ", "_").replace("(μm)", "")}')


def fig_stage_density_contours(coords_2d, stages, output_dir, embed_label='UMAP'):
    """Stage density contours."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='#EEEEEE', s=1, alpha=0.2, rasterized=True)
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 100:
            sns.kdeplot(x=coords_2d[mask, 0], y=coords_2d[mask, 1],
                       color=STAGE_COLORS[stage], levels=5, linewidths=1.5,
                       alpha=0.8, ax=ax, label=stage)
    ax.legend(loc='upper right')
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    ax.set_title('Stage Density Contours')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, output_dir, f'stage_density_contours_{embed_label.lower().split()[0]}')


def fig_ot_velocity_field(coords_2d, stages, flow, output_dir, embed_label='UMAP'):
    """Optimal transport velocity field with streamlines."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c='#EEEEEE', s=1, alpha=0.3, rasterized=True)

    Xi, Yi, U, V = flow['Xi'], flow['Yi'], flow['U'], flow['V']
    speed = np.sqrt(U**2 + V**2)

    strm = ax.streamplot(Xi[0, :], Yi[:, 0], U, V,
                        color=speed, cmap='viridis',
                        density=2.0, linewidth=1.5, arrowsize=1.5)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c=STAGE_COLORS[stage], s=200,
                  edgecolor='white', linewidth=2.5, zorder=10)
        ax.annotate(stage, centroid, fontsize=10, fontweight='bold',
                   ha='center', va='center', color='white')

    cbar = plt.colorbar(strm.lines, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Flow Speed', fontsize=11)
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    title_suffix = ' (Tissue Space)' if 'Spatial' in embed_label else ''
    ax.set_title(f'Optimal Transport Velocity Field{title_suffix}')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, output_dir, f'ot_velocity_field_{embed_label.lower().split()[0]}')


def fig_wasserstein_distances(flow, output_dir):
    """Wasserstein distances between consecutive stages."""
    fig, ax = plt.subplots(figsize=(6, 5))

    transitions = list(flow['stage_results'].keys())
    W_values = [flow['stage_results'][t]['W'] for t in transitions]
    colors = [STAGE_COLORS[t.split('->')[0]] for t in transitions]

    bars = ax.barh(range(len(transitions)), W_values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(transitions)))
    ax.set_yticklabels([t.replace('->', ' → ') for t in transitions])
    ax.set_xlabel('Wasserstein Distance')
    ax.set_title('Optimal Transport Cost per Transition')
    ax.invert_yaxis()

    for bar, val in zip(bars, W_values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    save_fig(fig, output_dir, 'wasserstein_distances')


def fig_divergence(Xi, Yi, flux, coords_2d, stages, output_dir, embed_label='UMAP'):
    """Divergence field (sources/sinks)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    div = flux['divergence']
    vmax = np.percentile(np.abs(div), 95)
    im = ax.pcolormesh(Xi, Yi, div, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                      shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=80, edgecolor=STAGE_COLORS[stage], linewidth=2.5, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Divergence ∇·v', fontsize=11)
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    ax.set_title('Divergence Field (Sources & Sinks)')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, output_dir, f'divergence_{embed_label.lower().split()[0]}')


def fig_curl(Xi, Yi, flux, coords_2d, stages, output_dir, embed_label='UMAP'):
    """Curl field (rotational/irreversible component)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    curl = flux['curl']
    vmax = np.percentile(np.abs(curl), 95)
    im = ax.pcolormesh(Xi, Yi, curl, cmap='PiYG', vmin=-vmax, vmax=vmax,
                      shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=80, edgecolor=STAGE_COLORS[stage], linewidth=2.5, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Curl ∇×v', fontsize=11)
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    ax.set_title('Curl Field (Irreversible Dynamics)')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])
    save_fig(fig, output_dir, f'curl_irreversibility_{embed_label.lower().split()[0]}')


def fig_flux_ratio_map(Xi, Yi, flux, coords_2d, stages, output_dir, embed_label='UMAP'):
    """Flux ratio map (fraction of irreversible dynamics)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    fr = flux['flux_ratio']
    im = ax.pcolormesh(Xi, Yi, fr, cmap='inferno', vmin=0, vmax=1,
                      shading='gouraud', rasterized=True)

    for stage in STAGE_ORDER:
        mask = stages == stage
        centroid = coords_2d[mask].mean(axis=0)
        ax.scatter(*centroid, c='white', s=80, edgecolor=STAGE_COLORS[stage], linewidth=2.5, zorder=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Flux Ratio (Irreversibility)', fontsize=11)
    ax.set_xlabel(f'{embed_label} 1')
    ax.set_ylabel(f'{embed_label} 2')
    ax.set_title('Irreversibility Map')
    if 'Spatial' not in embed_label:
        ax.set_xticks([])
        ax.set_yticks([])

    mean_fr = flux['mean_flux_ratio']
    ax.text(0.02, 0.98, f'Mean = {mean_fr:.2f}', transform=ax.transAxes,
           ha='left', va='top', fontsize=12, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

    save_fig(fig, output_dir, f'flux_ratio_map_{embed_label.lower().split()[0]}')


def fig_flux_ratio_by_stage(Xi, Yi, flux, coords_2d, stages, output_dir, embed_label='UMAP'):
    """Violin plot of flux ratio by stage."""
    fig, ax = plt.subplots(figsize=(7, 6))

    fr = flux['flux_ratio']
    flux_by_stage = []
    for stage in STAGE_ORDER:
        mask = stages == stage
        stage_coords = coords_2d[mask]
        stage_flux = []
        for coord in stage_coords[:500]:
            gi = np.argmin(np.abs(Yi[:, 0] - coord[1]))
            gj = np.argmin(np.abs(Xi[0, :] - coord[0]))
            stage_flux.append(fr[gi, gj])
        flux_by_stage.append(np.array(stage_flux))

    parts = ax.violinplot(flux_by_stage, positions=range(len(STAGE_ORDER)),
                          showmeans=False, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(STAGE_COLORS[STAGE_ORDER[i]])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Equilibrium threshold')
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Flux Ratio')
    ax.set_ylim(0, 1)
    space_name = 'Tissue' if 'Spatial' in embed_label else embed_label.split()[0]
    ax.set_title(f'Irreversibility by Disease Stage ({space_name} space)')
    ax.legend(loc='upper right')

    mean_fr = flux['mean_flux_ratio']
    ax.text(0.98, 0.02, f'Mean = {mean_fr:.2f}', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

    save_fig(fig, output_dir, f'flux_ratio_by_stage_{embed_label.lower().split()[0]}')


def fig_flow_speed(Xi, Yi, flux, output_dir):
    """Flow speed heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    speed = flux['speed']
    im = ax.pcolormesh(Xi, Yi, speed, cmap='YlOrRd', shading='gouraud', rasterized=True)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Flow Speed', fontsize=11)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Flow Speed')
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, output_dir, 'flow_speed')


def fig_cumulative_wasserstein(flow, output_dir):
    """Cumulative Wasserstein distance along progression."""
    fig, ax = plt.subplots(figsize=(7, 5))

    transitions = list(flow['stage_results'].keys())
    W_values = [flow['stage_results'][t]['W'] for t in transitions]
    cumulative_W = np.cumsum([0] + W_values)
    stage_labels = ['Start'] + [t.split('->')[1] for t in transitions]

    ax.plot(range(len(cumulative_W)), cumulative_W, 'o-', color='#1B4F72',
            linewidth=2.5, markersize=10)
    ax.fill_between(range(len(cumulative_W)), cumulative_W, alpha=0.3, color='#1B4F72')

    ax.set_xticks(range(len(cumulative_W)))
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel('Cumulative Wasserstein Distance')
    ax.set_title('Progression Cost')
    save_fig(fig, output_dir, 'cumulative_wasserstein')


def fig_transition_matrix(flow, output_dir):
    """Transition propensity matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))

    n = len(STAGE_ORDER)
    T_matrix = np.zeros((n, n))
    for i in range(n - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        key = f'{s1}->{s2}'
        if key in flow['stage_results']:
            T_matrix[i, i+1] = 1.0 / (flow['stage_results'][key]['W'] + 0.01)

    row_sums = T_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_matrix = T_matrix / row_sums

    for i in range(n):
        T_matrix[i, i] = 1 - T_matrix[i].sum()

    im = ax.imshow(T_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(STAGE_ORDER, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(STAGE_ORDER)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('Transition Propensity')

    for i in range(n):
        for j in range(n):
            if T_matrix[i, j] > 0.01:
                color = 'white' if T_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{T_matrix[i,j]:.2f}', ha='center', va='center',
                       fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('P(transition)', fontsize=10)
    save_fig(fig, output_dir, 'transition_matrix')


def fig_stage_distribution_bar(stages, output_dir):
    """Stage distribution bar chart."""
    fig, ax = plt.subplots(figsize=(7, 5))

    counts = pd.Series(stages).value_counts()
    counts = counts.reindex(STAGE_ORDER).fillna(0)

    bars = ax.bar(range(len(STAGE_ORDER)), counts.values,
                  color=[STAGE_COLORS[s] for s in STAGE_ORDER], edgecolor='white')

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Cell Count')
    ax.set_title('Stage Distribution')

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'{int(val):,}', ha='center', va='bottom', fontsize=9)

    save_fig(fig, output_dir, 'stage_distribution')


def fig_cell_cycle_umap(coords_2d, cells_s, output_dir):
    """UMAP colored by cell cycle phase."""
    if 'phase' not in cells_s.columns:
        print("  Skipping cell_cycle_umap: no phase column")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    phase_colors = {'G1': '#3C5488', 'S': '#E64B35', 'G2M': '#00A087'}
    for phase in ['G1', 'S', 'G2M']:
        mask = cells_s['phase'] == phase
        if mask.sum() > 0:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                      c=phase_colors.get(phase, 'gray'), s=5, alpha=0.5,
                      label=f'{phase} (n={mask.sum():,})', rasterized=True)

    ax.legend(loc='upper right', markerscale=3)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Cell Cycle Phase')
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, output_dir, 'cell_cycle_umap')


def fig_proliferation_umap(coords_2d, cells_s, output_dir):
    """UMAP colored by proliferation score."""
    # Try different proliferation columns
    for col in ['proliferation', 'S_score', 'G2M_score', 'MKI67']:
        if col in cells_s.columns and cells_s[col].notna().sum() > 100:
            break
    else:
        print("  Skipping proliferation_umap: no proliferation column")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    values = cells_s[col].values
    valid = ~np.isnan(values)
    sc = ax.scatter(coords_2d[valid, 0], coords_2d[valid, 1],
                   c=values[valid], cmap='viridis', s=5, alpha=0.5, rasterized=True)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(col, fontsize=11)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'Proliferation ({col})')
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(fig, output_dir, 'proliferation_umap')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate individual publication figures")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to canonical data")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/panels"), help="Output directory")
    parser.add_argument("--figures", type=str, default="all", help="Comma-separated figure names or 'all'")
    parser.add_argument("--n-per-stage", type=int, default=5000, help="Cells per stage for sampling")
    parser.add_argument("--embedding", type=str, default="umap",
                       choices=["umap", "phate", "spatial", "latent"],
                       help="Embedding space for OT dynamics: umap, phate, spatial, or latent")
    parser.add_argument("--spatial-only", action="store_true",
                       help="Only use spatial cells (filter out snRNA)")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Individual Publication Figures")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    cells = load_data(args.data_dir)
    print(f"  {len(cells):,} cells")

    # Filter to spatial only if requested
    if args.spatial_only or args.embedding == 'spatial':
        if 'data_type' in cells.columns:
            cells = cells[cells['data_type'] == 'spatial']
        elif 'x_spatial' in cells.columns:
            cells = cells[cells['x_spatial'].notna()]
        print(f"  Filtered to {len(cells):,} spatial cells")

    # Sample and compute embeddings
    print("\nSampling and computing embeddings...")
    cells_s = sample_balanced(cells, n_per_stage=args.n_per_stage)
    fused = get_embeddings(cells_s)
    if fused is None:
        print("ERROR: Could not extract embeddings")
        return

    print(f"  Sampled {len(cells_s):,} cells")
    print(f"  Embedding dim: {fused.shape[1]}")

    print(f"\nComputing {args.embedding.upper()} embedding...")
    coords_2d, cells_s, embed_label = compute_embedding(cells_s, fused, args.embedding)

    # If spatial embedding, need to re-extract embeddings for the filtered subset
    if args.embedding == 'spatial' and len(cells_s) < len(fused):
        fused = get_embeddings(cells_s)

    stages = cells_s['stage'].values
    print(f"  Using {embed_label} coordinates")

    # OT computations
    print("\nComputing optimal transport flow field...")
    flow = compute_ot_flow_field(coords_2d, stages, grid_size=30)

    if flow is not None:
        print("Computing Helmholtz decomposition...")
        flux = compute_flux_decomposition(flow['U'], flow['V'])
        Xi, Yi = flow['Xi'], flow['Yi']
    else:
        print("WARNING: OT computation failed")
        flux = None

    # Generate figures
    print("\nGenerating figures...")

    all_figures = [
        ('stage_embedding', lambda: fig_stage_umap(coords_2d, stages, args.output_dir, embed_label)),
        ('stage_density_contours', lambda: fig_stage_density_contours(coords_2d, stages, args.output_dir, embed_label)),
        ('stage_distribution', lambda: fig_stage_distribution_bar(stages, args.output_dir)),
        ('cell_cycle', lambda: fig_cell_cycle_umap(coords_2d, cells_s, args.output_dir)),
        ('proliferation', lambda: fig_proliferation_umap(coords_2d, cells_s, args.output_dir)),
    ]

    if flow is not None and flux is not None:
        all_figures.extend([
            ('ot_velocity_field', lambda: fig_ot_velocity_field(coords_2d, stages, flow, args.output_dir, embed_label)),
            ('wasserstein_distances', lambda: fig_wasserstein_distances(flow, args.output_dir)),
            ('divergence', lambda: fig_divergence(Xi, Yi, flux, coords_2d, stages, args.output_dir, embed_label)),
            ('curl_irreversibility', lambda: fig_curl(Xi, Yi, flux, coords_2d, stages, args.output_dir, embed_label)),
            ('flux_ratio_map', lambda: fig_flux_ratio_map(Xi, Yi, flux, coords_2d, stages, args.output_dir, embed_label)),
            ('flux_ratio_by_stage', lambda: fig_flux_ratio_by_stage(Xi, Yi, flux, coords_2d, stages, args.output_dir, embed_label)),
            ('flow_speed', lambda: fig_flow_speed(Xi, Yi, flux, args.output_dir)),
            ('cumulative_wasserstein', lambda: fig_cumulative_wasserstein(flow, args.output_dir)),
            ('transition_matrix', lambda: fig_transition_matrix(flow, args.output_dir)),
        ])

    # Filter if specific figures requested
    if args.figures != "all":
        requested = set(args.figures.split(','))
        all_figures = [(name, fn) for name, fn in all_figures if name in requested]

    for name, fn in all_figures:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print("\n" + "=" * 60)
    print(f"Figures saved to: {args.output_dir}")
    if flux is not None:
        print(f"Mean flux ratio (irreversibility): {flux['mean_flux_ratio']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
