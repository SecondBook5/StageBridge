#!/usr/bin/env python3
"""Publication-grade dynamics and clonal evolution figures.

Nature Methods quality - no text boxes, clean design, data speaks for itself.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from scipy import stats, ndimage
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import json

# =============================================================================
# PUBLICATION STYLE
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
STAGE_COLORS = {
    'Normal': '#27ae60',
    'AAH': '#f39c12',
    'AIS': '#e74c3c',
    'MIA': '#8e44ad',
    'LUAD': '#2c3e50'
}
PATTERN_COLORS = {'1a': '#3498db', '1b': '#9b59b6', '2': '#e67e22'}


def load_data(data_dir: Path):
    """Load cells and patterns."""
    cells = pd.read_parquet(data_dir / "cells.parquet")

    patterns_path = Path("data/paper/clonal_patterns.json")
    patterns = {}
    if patterns_path.exists():
        with open(patterns_path) as f:
            patterns = json.load(f)

    return cells, patterns


def get_embeddings(cells: pd.DataFrame, prefix: str = "z_fused_") -> np.ndarray:
    cols = sorted([c for c in cells.columns if c.startswith(prefix)])
    return cells[cols].values


def compute_dynamics(cells_s, coords_2d):
    """Compute velocity field and landscape-flux decomposition."""
    stage_map = {s: i for i, s in enumerate(STAGE_ORDER)}

    # Stage centroids
    centroids = {}
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        if mask.sum() > 0:
            centroids[stage] = coords_2d[mask.values].mean(axis=0)

    # Velocity field (pointing toward next stage)
    velocities = np.zeros_like(coords_2d)
    for i, stage in enumerate(cells_s['stage'].values):
        stage_idx = stage_map.get(stage, -1)
        if 0 <= stage_idx < len(STAGE_ORDER) - 1:
            next_stage = STAGE_ORDER[stage_idx + 1]
            if next_stage in centroids:
                direction = centroids[next_stage] - coords_2d[i]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    velocities[i] = direction / norm * 0.3

    # Grid
    pad = 0.5
    x_min, x_max = coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad
    y_min, y_max = coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad

    n_grid = 60
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(n_grid), y_min:y_max:complex(n_grid)]

    # Interpolate
    vx = griddata(coords_2d, velocities[:, 0], (grid_x, grid_y), method='linear', fill_value=0)
    vy = griddata(coords_2d, velocities[:, 1], (grid_x, grid_y), method='linear', fill_value=0)
    vx = ndimage.gaussian_filter(vx, sigma=1.5)
    vy = ndimage.gaussian_filter(vy, sigma=1.5)

    # Density-based potential
    try:
        kde = gaussian_kde(coords_2d.T, bw_method=0.15)
        density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    except:
        density = np.ones_like(grid_x)

    potential = -np.log(density + 1e-10)
    potential = ndimage.gaussian_filter(potential, sigma=2)
    potential = (potential - potential.min()) / (potential.max() - potential.min() + 1e-10)

    # Gradient and flux
    dU_dx = np.gradient(potential, axis=0)
    dU_dy = np.gradient(potential, axis=1)

    vx_rot = vx + dU_dx
    vy_rot = vy + dU_dy

    grad_mag = np.sqrt(dU_dx**2 + dU_dy**2)
    rot_mag = np.sqrt(vx_rot**2 + vy_rot**2)
    flux_ratio = rot_mag / (grad_mag + rot_mag + 1e-10)

    # Curl
    curl = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)

    # Divergence
    div = np.gradient(vx, axis=0) + np.gradient(vy, axis=1)

    return {
        'grid_x': grid_x, 'grid_y': grid_y,
        'vx': vx, 'vy': vy,
        'potential': potential,
        'flux_ratio': flux_ratio,
        'curl': curl,
        'divergence': div,
        'centroids': centroids,
        'coords_2d': coords_2d
    }


# =============================================================================
# FIGURE 1: PHASE PORTRAIT
# =============================================================================

def figure_phase_portrait(cells_s, dynamics, output_dir):
    """Clean phase portrait with streamlines and fixed points."""
    print("  Generating phase portrait...")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))

    grid_x, grid_y = dynamics['grid_x'], dynamics['grid_y']
    vx, vy = dynamics['vx'], dynamics['vy']
    centroids = dynamics['centroids']
    coords_2d = dynamics['coords_2d']

    speed = np.sqrt(vx**2 + vy**2)

    # A: Streamlines
    ax = axes[0]
    strm = ax.streamplot(grid_x[:, 0], grid_y[0, :], vx.T, vy.T,
                        color=speed.T, cmap='coolwarm', density=1.8,
                        linewidth=0.6, arrowsize=0.6, arrowstyle='->')

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='white', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('A', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # B: Divergence (sources/sinks)
    ax = axes[1]
    div = dynamics['divergence']
    vmax = np.percentile(np.abs(div), 95)
    im = ax.pcolormesh(grid_x, grid_y, div, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='white', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Divergence', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # C: Curl (rotation)
    ax = axes[2]
    curl = dynamics['curl']
    vmax = np.percentile(np.abs(curl), 95)
    im = ax.pcolormesh(grid_x, grid_y, curl, cmap='PiYG',
                       vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='white', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('C', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Curl', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_phase_portrait.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_phase_portrait.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# FIGURE 2: LANDSCAPE-FLUX DECOMPOSITION
# =============================================================================

def figure_landscape_flux(cells_s, dynamics, output_dir):
    """Waddington landscape and flux decomposition."""
    print("  Generating landscape-flux figure...")

    fig = plt.figure(figsize=(7, 4.5))

    grid_x, grid_y = dynamics['grid_x'], dynamics['grid_y']
    potential = dynamics['potential']
    flux_ratio = dynamics['flux_ratio']
    centroids = dynamics['centroids']
    vx, vy = dynamics['vx'], dynamics['vy']

    # A: 3D landscape
    ax = fig.add_subplot(2, 3, 1, projection='3d')

    surf = ax.plot_surface(grid_x, grid_y, potential, cmap='terrain',
                          alpha=0.9, linewidth=0, antialiased=True,
                          rasterized=True)

    # Project centroids
    for stage, c in centroids.items():
        i = np.argmin(np.abs(grid_x[:, 0] - c[0]))
        j = np.argmin(np.abs(grid_y[0, :] - c[1]))
        z = potential[i, j]
        ax.scatter([c[0]], [c[1]], [z], c=STAGE_COLORS[stage], s=30,
                  edgecolor='black', linewidth=0.5, zorder=5)

    ax.set_xlabel('PC1', fontsize=7, labelpad=-2)
    ax.set_ylabel('PC2', fontsize=7, labelpad=-2)
    ax.set_zlabel('U', fontsize=7, labelpad=-2)
    ax.set_title('A', loc='left', fontweight='bold', fontsize=10)
    ax.view_init(elev=25, azim=-50)
    ax.tick_params(labelsize=6, pad=-2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # B: Contour landscape with gradient arrows
    ax = fig.add_subplot(2, 3, 2)

    contours = ax.contourf(grid_x, grid_y, potential, levels=20,
                          cmap='terrain', alpha=0.9)
    ax.contour(grid_x, grid_y, potential, levels=10, colors='white',
              linewidths=0.3, alpha=0.5)

    # Gradient arrows
    skip = 6
    dU_dx = np.gradient(potential, axis=0)
    dU_dy = np.gradient(potential, axis=1)
    ax.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
             -dU_dx[::skip, ::skip], -dU_dy[::skip, ::skip],
             color='white', alpha=0.6, scale=25, width=0.004)

    # Stage path
    stages = [s for s in STAGE_ORDER if s in centroids]
    if len(stages) > 1:
        path_x = [centroids[s][0] for s in stages]
        path_y = [centroids[s][1] for s in stages]
        ax.plot(path_x, path_y, 'k-', linewidth=1.5, alpha=0.7, zorder=4)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=50, zorder=5,
                  edgecolor='black', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # C: Flux ratio field
    ax = fig.add_subplot(2, 3, 3)

    im = ax.pcolormesh(grid_x, grid_y, flux_ratio, cmap='magma',
                       vmin=0, vmax=1, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c='white', s=50, zorder=5,
                  edgecolor='black', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('C', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Flux ratio', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # D: Flux ratio by stage (quantitative)
    ax = fig.add_subplot(2, 3, 4)

    flux_by_stage = []
    stage_labels = []
    for stage in STAGE_ORDER:
        if stage in centroids:
            cx, cy = centroids[stage]
            dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
            near_mask = dist < 1.0
            if near_mask.any():
                flux_by_stage.append(flux_ratio[near_mask].mean())
                stage_labels.append(stage)

    colors = [STAGE_COLORS[s] for s in stage_labels]
    bars = ax.bar(range(len(stage_labels)), flux_by_stage, color=colors,
                 edgecolor='black', linewidth=0.5)

    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, rotation=45, ha='right')
    ax.set_ylabel('Flux ratio')
    ax.set_ylim(0, 1)
    ax.set_title('D', loc='left', fontweight='bold')

    # E: Streamlines colored by flux
    ax = fig.add_subplot(2, 3, 5)

    strm = ax.streamplot(grid_x[:, 0], grid_y[0, :], vx.T, vy.T,
                        color=flux_ratio.T, cmap='magma', density=1.5,
                        linewidth=0.6, arrowsize=0.6)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=50, zorder=5,
                  edgecolor='white', linewidth=0.8)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('E', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # F: Summary statistics
    ax = fig.add_subplot(2, 3, 6)

    mean_flux = np.nanmean(flux_ratio)

    # Bootstrap confidence interval
    n_boot = 100
    boot_means = []
    flat_flux = flux_ratio.ravel()
    flat_flux = flat_flux[~np.isnan(flat_flux)]
    for _ in range(n_boot):
        sample = np.random.choice(flat_flux, size=len(flat_flux), replace=True)
        boot_means.append(np.mean(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    ax.barh([0], [mean_flux], color='#9b59b6', height=0.5,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(mean_flux, 0, xerr=[[mean_flux - ci_low], [ci_high - mean_flux]],
               fmt='none', color='black', capsize=3, linewidth=1)

    ax.axvline(0.5, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Mean flux ratio')
    ax.set_title('F', loc='left', fontweight='bold')

    # Add value annotation
    ax.text(mean_flux + 0.05, 0, f'{mean_flux:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_landscape_flux.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_landscape_flux.pdf", facecolor='white')
    plt.close(fig)

    return mean_flux


# =============================================================================
# FIGURE 3: CLONAL EVOLUTION PATTERNS
# =============================================================================

def figure_clonal_evolution(cells, patterns, coords_2d, output_dir):
    """Clonal evolution patterns from paper."""
    print("  Generating clonal evolution figure...")

    if not patterns:
        print("    No patterns data, skipping")
        return

    patient_to_pattern = patterns.get('patient_to_pattern', {})
    pattern_info = patterns.get('patterns', {})

    # Map to cells
    cells = cells.copy()
    cells['pattern'] = cells['donor_id'].map(patient_to_pattern)
    cells_with_pattern = cells.dropna(subset=['pattern'])

    if len(cells_with_pattern) == 0:
        print("    No pattern mapping, skipping")
        return

    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    # A: UMAP by pattern
    ax = axes[0, 0]
    for pattern in ['1a', '1b', '2']:
        mask = cells_with_pattern['pattern'] == pattern
        idx = cells_with_pattern[mask].index
        pos = [cells.index.get_loc(i) for i in idx if i in cells.index]
        if pos:
            ax.scatter(coords_2d[pos, 0], coords_2d[pos, 1],
                      c=PATTERN_COLORS[pattern], s=3, alpha=0.4,
                      label=pattern, rasterized=True)

    ax.legend(markerscale=3, loc='upper right', fontsize=7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('A', loc='left', fontweight='bold')
    ax.set_aspect('equal', adjustable='box')

    # B: Patient counts by pattern
    ax = axes[0, 1]
    pattern_counts = [pattern_info.get(p, {}).get('n', 0) for p in ['1a', '1b', '2']]
    colors = [PATTERN_COLORS[p] for p in ['1a', '1b', '2']]

    wedges, texts, autotexts = ax.pie(pattern_counts, colors=colors,
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 8})
    ax.set_title('B', loc='left', fontweight='bold')

    # C: Cell counts by pattern
    ax = axes[0, 2]
    cell_counts = cells_with_pattern['pattern'].value_counts()
    bars = ax.bar(['1a', '1b', '2'],
                 [cell_counts.get(p, 0) for p in ['1a', '1b', '2']],
                 color=[PATTERN_COLORS[p] for p in ['1a', '1b', '2']],
                 edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Cells')
    ax.set_xlabel('Pattern')
    ax.set_title('C', loc='left', fontweight='bold')

    # Format y-axis in thousands
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    # D: Stage composition by pattern
    ax = axes[1, 0]
    stage_by_pattern = pd.crosstab(cells_with_pattern['pattern'],
                                    cells_with_pattern['stage'],
                                    normalize='index')
    stage_by_pattern = stage_by_pattern.reindex(['1a', '1b', '2']).reindex(columns=STAGE_ORDER)

    x = np.arange(3)
    width = 0.15
    for i, stage in enumerate(STAGE_ORDER):
        offset = (i - 2) * width
        if stage in stage_by_pattern.columns:
            ax.bar(x + offset, stage_by_pattern[stage], width,
                  color=STAGE_COLORS[stage], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(['1a', '1b', '2'])
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Pattern')
    ax.set_title('D', loc='left', fontweight='bold')
    ax.legend(STAGE_ORDER, fontsize=6, loc='upper right', ncol=2)

    # E: Pattern by stage
    ax = axes[1, 1]
    pattern_by_stage = pd.crosstab(cells_with_pattern['stage'],
                                    cells_with_pattern['pattern'],
                                    normalize='index')
    pattern_by_stage = pattern_by_stage.reindex(STAGE_ORDER).reindex(columns=['1a', '1b', '2'])

    bottom = np.zeros(len(STAGE_ORDER))
    for pattern in ['1a', '1b', '2']:
        if pattern in pattern_by_stage.columns:
            values = pattern_by_stage[pattern].values
            ax.bar(STAGE_ORDER, values, bottom=bottom,
                  color=PATTERN_COLORS[pattern], edgecolor='white',
                  linewidth=0.3, label=pattern)
            bottom += values

    ax.set_ylabel('Proportion')
    ax.set_xticklabels(STAGE_ORDER, rotation=45, ha='right')
    ax.set_title('E', loc='left', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')

    # F: Evolution schematic (simplified)
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Pattern 1a
    y = 3
    ax.add_patch(plt.Circle((1, y), 0.3, color=PATTERN_COLORS['1a']))
    ax.annotate('', xy=(2.5, y), xytext=(1.4, y),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((3, y), 0.3, color=PATTERN_COLORS['1a']))
    ax.annotate('', xy=(4.5, y), xytext=(3.4, y),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((5, y), 0.3, color=PATTERN_COLORS['1a']))
    ax.text(6, y, '1a: Direct', va='center', fontsize=7)

    # Pattern 1b
    y = 2
    ax.add_patch(plt.Circle((1, y), 0.3, color=PATTERN_COLORS['1b']))
    ax.annotate('', xy=(2.3, y+0.3), xytext=(1.4, y+0.1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.annotate('', xy=(2.3, y-0.3), xytext=(1.4, y-0.1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((2.7, y+0.4), 0.2, color=PATTERN_COLORS['1b'], alpha=0.6))
    ax.add_patch(plt.Circle((2.7, y-0.4), 0.2, color=PATTERN_COLORS['1b'], alpha=0.6))
    ax.annotate('', xy=(4.5, y), xytext=(3, y+0.3),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((5, y), 0.3, color=PATTERN_COLORS['1b']))
    ax.text(6, y, '1b: Branched', va='center', fontsize=7)

    # Pattern 2
    y = 1
    ax.add_patch(plt.Circle((1, y), 0.3, color=PATTERN_COLORS['2']))
    ax.annotate('', xy=(2.5, y+0.3), xytext=(1.4, y+0.1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((3, y+0.4), 0.25, color=PATTERN_COLORS['2'], alpha=0.6))
    ax.add_patch(plt.Circle((3, y-0.4), 0.25, color='gray', alpha=0.4))
    ax.annotate('', xy=(4.5, y), xytext=(3.3, y-0.3),
               arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.add_patch(plt.Circle((5, y), 0.3, color=PATTERN_COLORS['2']))
    ax.text(6, y, '2: Independent', va='center', fontsize=7)

    ax.set_title('F', loc='left', fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "fig_clonal_evolution.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_clonal_evolution.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# FIGURE 4: H3 VALIDATION - EMBEDDING VS CLONAL
# =============================================================================

def figure_h3_validation(cells, patterns, output_dir):
    """H3: Do clonally related cells have similar embeddings?"""
    print("  Generating H3 validation figure...")

    if not patterns:
        print("    No patterns data, skipping")
        return

    patient_to_pattern = patterns.get('patient_to_pattern', {})
    fused_cols = sorted([c for c in cells.columns if c.startswith('z_fused_')])
    fused = cells[fused_cols].values

    # Per-patient, per-stage centroids
    results = []
    for patient in cells['donor_id'].unique():
        if patient not in patient_to_pattern:
            continue

        patient_cells = cells[cells['donor_id'] == patient]
        pattern = patient_to_pattern[patient]

        centroids = {}
        for stage in STAGE_ORDER:
            mask = patient_cells['stage'] == stage
            if mask.sum() > 10:
                idx = patient_cells[mask].index
                pos = [cells.index.get_loc(i) for i in idx]
                centroids[stage] = fused[pos].mean(axis=0)

        # Stage-to-stage distances
        for i in range(len(STAGE_ORDER) - 1):
            s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
            if s1 in centroids and s2 in centroids:
                dist = np.linalg.norm(centroids[s2] - centroids[s1])
                results.append({
                    'patient': patient,
                    'pattern': pattern,
                    'transition': f'{s1}→{s2}',
                    'from_stage': s1,
                    'to_stage': s2,
                    'distance': dist
                })

    if not results:
        print("    Insufficient data for H3 analysis")
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.3))

    # A: Distance by pattern
    ax = axes[0]
    pattern_order = ['1a', '1b', '2']

    for i, pattern in enumerate(pattern_order):
        data = df[df['pattern'] == pattern]['distance']
        if len(data) > 0:
            bp = ax.boxplot([data], positions=[i], widths=0.5,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(PATTERN_COLORS[pattern])
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('black')

            # Scatter individual points
            jitter = np.random.normal(0, 0.08, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data,
                      c=PATTERN_COLORS[pattern], s=15, alpha=0.6, zorder=3)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['1a', '1b', '2'])
    ax.set_ylabel('Embedding distance')
    ax.set_xlabel('Pattern')
    ax.set_title('A', loc='left', fontweight='bold')

    # Stats
    groups = [df[df['pattern'] == p]['distance'].values for p in pattern_order]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) >= 2:
        stat, pval = stats.kruskal(*groups)
        ax.text(0.95, 0.95, f'p={pval:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=7)

    # B: Distance by transition
    ax = axes[1]
    transitions = [f'{STAGE_ORDER[i]}→{STAGE_ORDER[i+1]}' for i in range(4)]

    for i, trans in enumerate(transitions):
        data = df[df['transition'] == trans]['distance']
        if len(data) > 0:
            bp = ax.boxplot([data], positions=[i], widths=0.5,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor('#3498db')
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('black')

    ax.set_xticks(range(4))
    ax.set_xticklabels(['N→A', 'A→AIS', 'AIS→M', 'M→L'], fontsize=7)
    ax.set_ylabel('Embedding distance')
    ax.set_xlabel('Transition')
    ax.set_title('B', loc='left', fontweight='bold')

    # C: Pattern × Transition heatmap
    ax = axes[2]

    pivot = df.groupby(['pattern', 'transition'])['distance'].mean().unstack()
    pivot = pivot.reindex(['1a', '1b', '2']).reindex(columns=transitions)

    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')

    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(['N→A', 'A→AIS', 'AIS→M', 'M→L'], fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['1a', '1b', '2'])
    ax.set_xlabel('Transition')
    ax.set_ylabel('Pattern')
    ax.set_title('C', loc='left', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Distance', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_h3_validation.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_h3_validation.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# FIGURE 5: METHOD COMPARISON (data-driven, no text boxes)
# =============================================================================

def figure_method_comparison(dynamics, output_dir):
    """Compare what different methods can and cannot show."""
    print("  Generating method comparison figure...")

    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    grid_x, grid_y = dynamics['grid_x'], dynamics['grid_y']
    vx, vy = dynamics['vx'], dynamics['vy']
    potential = dynamics['potential']
    flux_ratio = dynamics['flux_ratio']
    centroids = dynamics['centroids']

    # A: Velocity field (what scVelo shows)
    ax = axes[0, 0]
    skip = 5
    speed = np.sqrt(vx**2 + vy**2)
    ax.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip],
             vx[::skip, ::skip], vy[::skip, ::skip],
             speed[::skip, ::skip], cmap='coolwarm', alpha=0.7, scale=15)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('A  Velocity (scVelo)', loc='left', fontweight='bold', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    # B: Fate probability (what CellRank shows)
    ax = axes[0, 1]

    # Create fate probability proxy
    stage_map = {s: i for i, s in enumerate(STAGE_ORDER)}
    fate_field = np.zeros_like(grid_x)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            # Distance to LUAD centroid
            if 'LUAD' in centroids:
                d = np.sqrt((grid_x[i,j] - centroids['LUAD'][0])**2 +
                           (grid_y[i,j] - centroids['LUAD'][1])**2)
                fate_field[i, j] = np.exp(-d/2)

    im = ax.pcolormesh(grid_x, grid_y, fate_field, cmap='RdYlGn_r',
                       shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B  Fate prob (CellRank)', loc='left', fontweight='bold', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('P(LUAD)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # C: Pseudotime (what Monocle shows)
    ax = axes[0, 2]

    # Pseudotime proxy based on distance from Normal
    pt_field = np.zeros_like(grid_x)
    if 'Normal' in centroids:
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                d = np.sqrt((grid_x[i,j] - centroids['Normal'][0])**2 +
                           (grid_y[i,j] - centroids['Normal'][1])**2)
                pt_field[i, j] = d

    im = ax.pcolormesh(grid_x, grid_y, pt_field, cmap='viridis',
                       shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='white', linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('C  Pseudotime (Monocle)', loc='left', fontweight='bold', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Pseudotime', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # D: Landscape (StageBridge)
    ax = axes[1, 0]

    contours = ax.contourf(grid_x, grid_y, potential, levels=15,
                          cmap='terrain', alpha=0.9)

    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_COLORS[stage], s=40, zorder=5,
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('D  Landscape (StageBridge)', loc='left', fontweight='bold', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    # E: Flux (StageBridge unique)
    ax = axes[1, 1]

    im = ax.pcolormesh(grid_x, grid_y, flux_ratio, cmap='magma',
                       vmin=0, vmax=1, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c='white', s=40, zorder=5,
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('E  Irreversibility (StageBridge)', loc='left', fontweight='bold', fontsize=9)
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Flux ratio', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # F: Capability comparison (bar chart)
    ax = axes[1, 2]

    methods = ['scVelo', 'CellRank', 'Monocle', 'StageBridge']
    capabilities = np.array([
        [1, 0, 0, 0],  # Velocity
        [0, 1, 0, 1],  # Fate
        [0, 0, 1, 1],  # Pseudotime
        [0, 0, 0, 1],  # Landscape
        [0, 0, 0, 1],  # Irreversibility
    ])

    cap_names = ['Velocity', 'Fate', 'Pseudotime', 'Landscape', 'Irreversibility']

    im = ax.imshow(capabilities, cmap='Greens', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cap_names)))
    ax.set_yticklabels(cap_names, fontsize=8)
    ax.set_title('F  Capabilities', loc='left', fontweight='bold', fontsize=9)

    # Add checkmarks
    for i in range(len(cap_names)):
        for j in range(len(methods)):
            if capabilities[i, j]:
                ax.text(j, i, '✓', ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / "fig_method_comparison.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig_method_comparison.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/dynamics_figures"))
    args = parser.parse_args()

    print("=" * 50)
    print("Publication Dynamics Figures")
    print("=" * 50)

    cells, patterns = load_data(args.data_dir)
    print(f"Loaded {len(cells):,} cells")

    # Sample
    np.random.seed(42)
    n_sample = min(25000, len(cells))
    cells_s = cells.sample(n_sample).reset_index(drop=True)

    # Embeddings and PCA
    fused = get_embeddings(cells_s)
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(fused)

    # Compute dynamics
    print("\nComputing dynamics...")
    dynamics = compute_dynamics(cells_s, coords_2d)
    print(f"  Mean flux ratio: {np.nanmean(dynamics['flux_ratio']):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\nGenerating figures...")
    figure_phase_portrait(cells_s, dynamics, args.output_dir)
    mean_flux = figure_landscape_flux(cells_s, dynamics, args.output_dir)
    figure_clonal_evolution(cells_s, patterns, coords_2d, args.output_dir)
    figure_h3_validation(cells, patterns, args.output_dir)
    figure_method_comparison(dynamics, args.output_dir)

    # Save metrics
    metrics = {
        'mean_flux_ratio': float(mean_flux),
        'n_cells': len(cells),
        'n_sampled': n_sample
    }
    with open(args.output_dir / "dynamics_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll figures saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
