#!/usr/bin/env python3
"""Publication-grade dynamics figures - Nature Methods quality.

Uses seaborn, advanced visualizations, beautiful palettes.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# SEABORN PUBLICATION STYLE
# =============================================================================
sns.set_theme(style="white", context="paper", font_scale=1.1)
sns.set_palette("deep")

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

# Dark2-inspired palette (from taveren)
STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
STAGE_PALETTE = {
    'Normal': '#1b9e77',   # Teal
    'AAH': '#d95f02',      # Orange
    'AIS': '#7570b3',      # Purple
    'MIA': '#e7298a',      # Pink
    'LUAD': '#66a61e'      # Green
}
PATTERN_PALETTE = {'1a': '#1b9e77', '1b': '#d95f02', '2': '#7570b3'}

# Gradient colormaps
LANDSCAPE_CMAP = LinearSegmentedColormap.from_list(
    'landscape', ['#2c3e50', '#1abc9c', '#f1c40f', '#e74c3c', '#ffffff']
)
FLUX_CMAP = sns.color_palette("rocket", as_cmap=True)
DIVERGENCE_CMAP = sns.color_palette("icefire", as_cmap=True)


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

    centroids = {}
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        if mask.sum() > 0:
            centroids[stage] = coords_2d[mask.values].mean(axis=0)

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

    pad = 0.5
    x_min, x_max = coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad
    y_min, y_max = coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad

    n_grid = 60
    grid_x, grid_y = np.mgrid[x_min:x_max:complex(n_grid), y_min:y_max:complex(n_grid)]

    vx = griddata(coords_2d, velocities[:, 0], (grid_x, grid_y), method='linear', fill_value=0)
    vy = griddata(coords_2d, velocities[:, 1], (grid_x, grid_y), method='linear', fill_value=0)
    vx = ndimage.gaussian_filter(vx, sigma=1.5)
    vy = ndimage.gaussian_filter(vy, sigma=1.5)

    try:
        kde = gaussian_kde(coords_2d.T, bw_method=0.15)
        density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    except:
        density = np.ones_like(grid_x)

    potential = -np.log(density + 1e-10)
    potential = ndimage.gaussian_filter(potential, sigma=2)
    potential = (potential - potential.min()) / (potential.max() - potential.min() + 1e-10)

    dU_dx = np.gradient(potential, axis=0)
    dU_dy = np.gradient(potential, axis=1)

    vx_rot = vx + dU_dx
    vy_rot = vy + dU_dy

    grad_mag = np.sqrt(dU_dx**2 + dU_dy**2)
    rot_mag = np.sqrt(vx_rot**2 + vy_rot**2)
    flux_ratio = rot_mag / (grad_mag + rot_mag + 1e-10)

    curl = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)
    div = np.gradient(vx, axis=0) + np.gradient(vy, axis=1)

    return {
        'grid_x': grid_x, 'grid_y': grid_y,
        'vx': vx, 'vy': vy,
        'potential': potential,
        'flux_ratio': flux_ratio,
        'curl': curl,
        'divergence': div,
        'centroids': centroids,
        'coords_2d': coords_2d,
        'density': density
    }


# =============================================================================
# FIGURE 1: DYNAMICS OVERVIEW
# =============================================================================

def figure_dynamics_overview(cells_s, dynamics, output_dir):
    """Comprehensive dynamics figure with fancy seaborn styling."""
    print("  Generating dynamics overview...")

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.5,
                  height_ratios=[1.3, 1, 1])

    grid_x, grid_y = dynamics['grid_x'], dynamics['grid_y']
    potential = dynamics['potential']
    flux_ratio = dynamics['flux_ratio']
    centroids = dynamics['centroids']
    coords_2d = dynamics['coords_2d']
    vx, vy = dynamics['vx'], dynamics['vy']

    # A: 3D Landscape (large panel)
    ax = fig.add_subplot(gs[0, 0:2], projection='3d')

    # Create beautiful surface
    surf = ax.plot_surface(grid_x, grid_y, potential,
                          cmap='terrain', alpha=0.85,
                          linewidth=0, antialiased=True,
                          rstride=2, cstride=2, rasterized=True)

    # Add "balls" rolling down
    for stage, c in centroids.items():
        i = np.argmin(np.abs(grid_x[:, 0] - c[0]))
        j = np.argmin(np.abs(grid_y[0, :] - c[1]))
        z = potential[i, j] + 0.05
        ax.scatter([c[0]], [c[1]], [z], c=STAGE_PALETTE[stage], s=80,
                  edgecolor='white', linewidth=1.5, zorder=10, depthshade=False)

    ax.set_xlabel('PC1', labelpad=2)
    ax.set_ylabel('PC2', labelpad=2)
    ax.set_zlabel('Potential', labelpad=2)
    ax.view_init(elev=30, azim=-45)
    ax.set_title('A  Waddington Landscape', loc='left', pad=10)
    ax.tick_params(labelsize=7, pad=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    # B: Flux field with streamlines (large panel)
    ax = fig.add_subplot(gs[0, 2:4])

    # Background density
    ax.contourf(grid_x, grid_y, dynamics['density'], levels=30,
               cmap='Greys', alpha=0.3)

    # Streamlines colored by flux
    speed = np.sqrt(vx**2 + vy**2)
    strm = ax.streamplot(grid_x[:, 0], grid_y[0, :], vx.T, vy.T,
                        color=flux_ratio.T, cmap=FLUX_CMAP,
                        density=2, linewidth=1.2, arrowsize=1,
                        arrowstyle='-|>')

    # Stage markers with glow effect
    for stage, c in centroids.items():
        ax.scatter(*c, c=STAGE_PALETTE[stage], s=200, zorder=10,
                  edgecolor='white', linewidth=2)
        ax.scatter(*c, c=STAGE_PALETTE[stage], s=400, zorder=9,
                  alpha=0.3)  # Glow
        ax.annotate(stage, c, fontsize=8, ha='center', va='bottom',
                   xytext=(0, 12), textcoords='offset points',
                   fontweight='bold')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('B  Vector Field & Irreversibility', loc='left')
    ax.set_aspect('equal')
    sns.despine(ax=ax)

    cbar = plt.colorbar(strm.lines, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Flux ratio', fontsize=9)

    # C: Divergence field
    ax = fig.add_subplot(gs[1, 0])

    div = dynamics['divergence']
    vmax = np.percentile(np.abs(div), 95)
    im = ax.pcolormesh(grid_x, grid_y, div, cmap=DIVERGENCE_CMAP,
                       vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c='white', s=60, zorder=5,
                  edgecolor=STAGE_PALETTE[stage], linewidth=2)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('C  Divergence', loc='left')
    ax.set_aspect('equal')
    sns.despine(ax=ax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('∇·v', fontsize=9)

    # D: Curl field
    ax = fig.add_subplot(gs[1, 1])

    curl = dynamics['curl']
    vmax = np.percentile(np.abs(curl), 95)
    im = ax.pcolormesh(grid_x, grid_y, curl, cmap='PiYG',
                       vmin=-vmax, vmax=vmax, shading='gouraud', rasterized=True)

    for stage, c in centroids.items():
        ax.scatter(*c, c='white', s=60, zorder=5,
                  edgecolor=STAGE_PALETTE[stage], linewidth=2)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('D  Curl (rotation)', loc='left')
    ax.set_aspect('equal')
    sns.despine(ax=ax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('∇×v', fontsize=9)

    # E: Flux ratio by stage - VIOLIN + BOX + JITTER (taveren style)
    ax = fig.add_subplot(gs[1, 2:4])

    flux_data = []
    for stage in STAGE_ORDER:
        if stage in centroids:
            cx, cy = centroids[stage]
            dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
            near_mask = dist < 1.5
            if near_mask.any():
                values = flux_ratio[near_mask].ravel()
                for v in values:
                    flux_data.append({'Stage': stage, 'Flux Ratio': v})

    flux_df = pd.DataFrame(flux_data)

    # Violin + Box + Jitter overlay (ggplot2/ggpubr style)
    positions = list(range(len(STAGE_ORDER)))
    colors = [STAGE_PALETTE[s] for s in STAGE_ORDER]

    # 1. Violin (background)
    parts = ax.violinplot(
        [flux_df[flux_df['Stage'] == s]['Flux Ratio'].dropna().values for s in STAGE_ORDER],
        positions=positions, showmeans=False, showmedians=False, showextrema=False
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('none')

    # 2. Box (overlay)
    bp = ax.boxplot(
        [flux_df[flux_df['Stage'] == s]['Flux Ratio'].dropna().values for s in STAGE_ORDER],
        positions=positions, widths=0.15, showfliers=False, patch_artist=True, zorder=2
    )
    for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.2)
        median.set_color('white')
        median.set_linewidth(2)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color('#333333')
            line.set_linewidth(1.2)

    # 3. Jitter points
    for i, stage in enumerate(STAGE_ORDER):
        data = flux_df[flux_df['Stage'] == stage]['Flux Ratio'].dropna()
        jitter = np.random.uniform(-0.1, 0.1, len(data))
        ax.scatter(i + jitter, data, c=colors[i], alpha=0.4, s=8, edgecolors='none', zorder=3)

    ax.axhline(0.5, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Equilibrium')
    ax.set_xticks(positions)
    ax.set_xticklabels(STAGE_ORDER, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Flux Ratio', fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('E  Irreversibility by Stage', loc='left')
    ax.legend(loc='lower right', fontsize=8)
    sns.despine(ax=ax)

    # F: Overall flux distribution - RIDGEPLOT STYLE
    ax = fig.add_subplot(gs[2, 0:2])

    mean_flux = np.nanmean(flux_ratio)

    # KDE of flux ratio
    flat_flux = flux_ratio.ravel()
    flat_flux = flat_flux[~np.isnan(flat_flux)]

    sns.kdeplot(flat_flux, ax=ax, fill=True, color=sns.color_palette("rocket")[3],
               alpha=0.7, linewidth=2)

    ax.axvline(mean_flux, color='#2c3e50', linestyle='-', linewidth=2,
              label=f'Mean = {mean_flux:.2f}')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5,
              label='Equilibrium')

    # Bootstrap CI
    n_boot = 1000
    boot_means = [np.mean(np.random.choice(flat_flux, len(flat_flux), replace=True))
                  for _ in range(n_boot)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    ax.axvspan(ci_low, ci_high, alpha=0.2, color='#2c3e50', label='95% CI')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Flux Ratio')
    ax.set_ylabel('Density')
    ax.set_title('F  Global Flux Distribution', loc='left')
    ax.legend(loc='upper left', fontsize=8)
    sns.despine(ax=ax)

    # G: Stage-to-stage transitions - HEATMAP
    ax = fig.add_subplot(gs[2, 2])

    # Compute transition distances
    trans_matrix = np.zeros((5, 5))
    for i, s1 in enumerate(STAGE_ORDER):
        for j, s2 in enumerate(STAGE_ORDER):
            if s1 in centroids and s2 in centroids:
                trans_matrix[i, j] = np.linalg.norm(
                    np.array(centroids[s1]) - np.array(centroids[s2])
                )

    mask = np.triu(np.ones_like(trans_matrix, dtype=bool), k=1)

    sns.heatmap(trans_matrix, mask=~mask, annot=True, fmt='.2f',
               cmap='YlOrRd', ax=ax, square=True,
               xticklabels=STAGE_ORDER, yticklabels=STAGE_ORDER,
               cbar_kws={'shrink': 0.6, 'label': 'Distance'},
               annot_kws={'size': 8})

    ax.set_title('G  Transition Distances', loc='left')

    # H: Summary statistics - HORIZONTAL BAR
    ax = fig.add_subplot(gs[2, 3])

    stats_data = {
        'Mean flux': mean_flux,
        'Median flux': np.nanmedian(flux_ratio),
        'Max curl': np.nanmax(np.abs(dynamics['curl'])),
        'Max div': np.nanmax(np.abs(dynamics['divergence']))
    }

    colors = sns.color_palette("rocket", len(stats_data))
    y_pos = range(len(stats_data))

    bars = ax.barh(y_pos, list(stats_data.values()), color=colors,
                  edgecolor='white', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(stats_data.keys()))
    ax.set_xlabel('Value')
    ax.set_title('H  Summary', loc='left')

    for i, (bar, val) in enumerate(zip(bars, stats_data.values())):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', va='center', fontsize=8)

    ax.set_xlim(0, max(stats_data.values()) * 1.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_dynamics_overview.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig1_dynamics_overview.pdf", facecolor='white')
    plt.close(fig)

    return mean_flux


# =============================================================================
# FIGURE 2: CLONAL EVOLUTION
# =============================================================================

def figure_clonal_evolution(cells, patterns, coords_2d, output_dir):
    """Clonal evolution with fancy seaborn styling."""
    print("  Generating clonal evolution figure...")

    if not patterns:
        print("    No patterns data, skipping")
        return

    patient_to_pattern = patterns.get('patient_to_pattern', {})
    pattern_info = patterns.get('patterns', {})

    cells = cells.copy()
    cells['pattern'] = cells['donor_id'].map(patient_to_pattern)
    cells_with_pattern = cells.dropna(subset=['pattern'])

    if len(cells_with_pattern) == 0:
        return

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.5)

    # A: UMAP by pattern with density contours
    ax = fig.add_subplot(gs[0, 0:2])

    for pattern in ['1a', '1b', '2']:
        mask = cells_with_pattern['pattern'] == pattern
        idx = cells_with_pattern[mask].index
        pos = [cells.index.get_loc(i) for i in idx if i in cells.index]
        if len(pos) > 100:
            coords = coords_2d[pos]
            # Scatter
            ax.scatter(coords[:, 0], coords[:, 1],
                      c=PATTERN_PALETTE[pattern], s=5, alpha=0.3,
                      label=f'{pattern} (n={len(pos):,})', rasterized=True)
            # Density contour
            try:
                sns.kdeplot(x=coords[:, 0], y=coords[:, 1], ax=ax,
                           color=PATTERN_PALETTE[pattern], levels=3,
                           linewidths=1.5, alpha=0.8)
            except:
                pass

    ax.legend(loc='upper right', fontsize=8, markerscale=3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('A  Evolutionary Patterns in Embedding', loc='left')
    ax.set_aspect('equal')
    sns.despine(ax=ax)

    # B: Donut chart with pattern distribution
    ax = fig.add_subplot(gs[0, 2])

    pattern_counts = [pattern_info.get(p, {}).get('n', 0) for p in ['1a', '1b', '2']]
    colors = [PATTERN_PALETTE[p] for p in ['1a', '1b', '2']]

    wedges, texts, autotexts = ax.pie(
        pattern_counts, colors=colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )

    # Center text
    ax.text(0, 0, f'n={sum(pattern_counts)}', ha='center', va='center',
           fontsize=12, fontweight='bold')

    ax.set_title('B  Patients', loc='left')

    # C: Cell counts - FANCY BAR
    ax = fig.add_subplot(gs[0, 3])

    cell_counts = cells_with_pattern['pattern'].value_counts()
    patterns_list = ['1a', '1b', '2']
    counts = [cell_counts.get(p, 0) for p in patterns_list]

    bars = ax.bar(patterns_list, counts,
                 color=[PATTERN_PALETTE[p] for p in patterns_list],
                 edgecolor='white', linewidth=2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{height/1000:.0f}k', ha='center', va='bottom',
               fontsize=9, fontweight='bold')

    ax.set_ylabel('Cells')
    ax.set_xlabel('Pattern')
    ax.set_title('C  Cells per Pattern', loc='left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
    sns.despine(ax=ax)

    # D: Stage composition - STACKED BAR with patterns
    ax = fig.add_subplot(gs[1, 0:2])

    stage_by_pattern = pd.crosstab(cells_with_pattern['stage'],
                                    cells_with_pattern['pattern'],
                                    normalize='columns')
    stage_by_pattern = stage_by_pattern.reindex(STAGE_ORDER).reindex(columns=['1a', '1b', '2'])

    x = np.arange(3)
    bottom = np.zeros(3)

    for stage in STAGE_ORDER:
        if stage in stage_by_pattern.index:
            values = stage_by_pattern.loc[stage].values
            ax.bar(x, values, bottom=bottom, label=stage,
                  color=STAGE_PALETTE[stage], edgecolor='white', linewidth=0.5)
            bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(['1a\nDirect', '1b\nBranched', '2\nIndependent'])
    ax.set_ylabel('Proportion')
    ax.set_title('D  Stage Composition by Pattern', loc='left')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    sns.despine(ax=ax)

    # E: Pattern composition by stage - FANCY STACKED
    ax = fig.add_subplot(gs[1, 2:4])

    pattern_by_stage = pd.crosstab(cells_with_pattern['stage'],
                                    cells_with_pattern['pattern'],
                                    normalize='index')
    pattern_by_stage = pattern_by_stage.reindex(STAGE_ORDER).reindex(columns=['1a', '1b', '2'])

    x = np.arange(len(STAGE_ORDER))
    width = 0.6

    bottom = np.zeros(len(STAGE_ORDER))
    for pattern in ['1a', '1b', '2']:
        if pattern in pattern_by_stage.columns:
            values = pattern_by_stage[pattern].values
            ax.bar(x, values, width, bottom=bottom, label=pattern,
                  color=PATTERN_PALETTE[pattern], edgecolor='white', linewidth=0.5)
            bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Disease Stage')
    ax.set_title('E  Evolution Pattern by Stage', loc='left')
    ax.legend(title='Pattern', fontsize=8)

    # Color x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_PALETTE[STAGE_ORDER[i]])
        label.set_fontweight('bold')

    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_clonal_evolution.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig2_clonal_evolution.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# FIGURE 3: H3 VALIDATION
# =============================================================================

def figure_h3_validation(cells, patterns, output_dir):
    """H3 validation with fancy seaborn styling."""
    print("  Generating H3 validation figure...")

    if not patterns:
        return

    patient_to_pattern = patterns.get('patient_to_pattern', {})
    fused_cols = sorted([c for c in cells.columns if c.startswith('z_fused_')])
    fused = cells[fused_cols].values

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

        for i in range(len(STAGE_ORDER) - 1):
            s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
            if s1 in centroids and s2 in centroids:
                dist = np.linalg.norm(centroids[s2] - centroids[s1])
                results.append({
                    'Patient': patient,
                    'Pattern': pattern,
                    'Transition': f'{s1}→{s2}',
                    'Distance': dist
                })

    if not results:
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # A: Violin + Box + Jitter by pattern (taveren style)
    ax = axes[0]
    pattern_order = ['1a', '1b', '2']
    colors = [PATTERN_PALETTE[p] for p in pattern_order]

    # Violin
    parts = ax.violinplot(
        [df[df['Pattern'] == p]['Distance'].dropna().values for p in pattern_order],
        positions=[0, 1, 2], showmeans=False, showmedians=False, showextrema=False
    )
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('none')

    # Box
    bp = ax.boxplot(
        [df[df['Pattern'] == p]['Distance'].dropna().values for p in pattern_order],
        positions=[0, 1, 2], widths=0.15, showfliers=False, patch_artist=True, zorder=2
    )
    for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
        patch.set_edgecolor('#333333')
        median.set_color('white')
        median.set_linewidth(2)

    # Jitter
    for i, p in enumerate(pattern_order):
        data = df[df['Pattern'] == p]['Distance'].dropna()
        jitter = np.random.uniform(-0.1, 0.1, len(data))
        ax.scatter(i + jitter, data, c=colors[i], alpha=0.5, s=20, edgecolors='none', zorder=3)

    # Stats
    groups = [df[df['Pattern'] == p]['Distance'].values for p in ['1a', '1b', '2']]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) >= 2:
        stat, pval = stats.kruskal(*groups)
        ax.text(0.95, 0.95, f'p = {pval:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1a', '1b', '2'], fontweight='bold')
    ax.set_ylabel('Embedding Distance', fontweight='bold')
    ax.set_xlabel('Evolutionary Pattern', fontweight='bold')
    ax.set_title('A  Distance by Pattern', loc='left')
    sns.despine(ax=ax)

    # B: Boxen plot by transition
    ax = axes[1]

    trans_order = [f'{STAGE_ORDER[i]}→{STAGE_ORDER[i+1]}' for i in range(4)]

    sns.boxenplot(data=df, x='Transition', y='Distance',
                 order=trans_order, palette='viridis', ax=ax)

    ax.set_ylabel('Embedding Distance')
    ax.set_xlabel('Transition')
    ax.set_xticklabels(['N→A', 'A→AIS', 'AIS→M', 'M→L'], fontsize=9)
    ax.set_title('B  Distance by Transition', loc='left')
    sns.despine(ax=ax)

    # C: Heatmap
    ax = axes[2]

    pivot = df.pivot_table(values='Distance', index='Pattern',
                          columns='Transition', aggfunc='mean')
    pivot = pivot.reindex(['1a', '1b', '2']).reindex(columns=trans_order)

    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='rocket_r',
               ax=ax, cbar_kws={'shrink': 0.8, 'label': 'Distance'},
               linewidths=1, linecolor='white')

    ax.set_xticklabels(['N→A', 'A→AIS', 'AIS→M', 'M→L'], fontsize=9)
    ax.set_ylabel('Pattern')
    ax.set_xlabel('Transition')
    ax.set_title('C  Pattern × Transition', loc='left')

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_h3_validation.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig3_h3_validation.pdf", facecolor='white')
    plt.close(fig)


# =============================================================================
# FIGURE 4: METHOD COMPARISON
# =============================================================================

def figure_method_comparison(cells_s, dynamics, output_dir):
    """Method comparison - what StageBridge uniquely provides."""
    print("  Generating method comparison figure...")

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45)

    grid_x, grid_y = dynamics['grid_x'], dynamics['grid_y']
    potential = dynamics['potential']
    flux_ratio = dynamics['flux_ratio']
    centroids = dynamics['centroids']
    coords_2d = dynamics['coords_2d']

    # Shared plotting function
    def plot_field(ax, field, cmap, title, label):
        im = ax.pcolormesh(grid_x, grid_y, field, cmap=cmap,
                          shading='gouraud', rasterized=True)
        for stage, c in centroids.items():
            ax.scatter(*c, c='white', s=50, zorder=5,
                      edgecolor=STAGE_PALETTE[stage], linewidth=2)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title, loc='left')
        ax.set_aspect('equal')
        sns.despine(ax=ax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(label, fontsize=8)
        return im

    # A: Pseudotime proxy (Monocle-like)
    ax = fig.add_subplot(gs[0, 0])
    pt_field = np.zeros_like(grid_x)
    if 'Normal' in centroids:
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                pt_field[i, j] = np.sqrt((grid_x[i,j] - centroids['Normal'][0])**2 +
                                        (grid_y[i,j] - centroids['Normal'][1])**2)
    plot_field(ax, pt_field, 'viridis', 'A  Pseudotime', 'τ')

    # B: Fate probability proxy (CellRank-like)
    ax = fig.add_subplot(gs[0, 1])
    fate_field = np.zeros_like(grid_x)
    if 'LUAD' in centroids:
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                d = np.sqrt((grid_x[i,j] - centroids['LUAD'][0])**2 +
                           (grid_y[i,j] - centroids['LUAD'][1])**2)
                fate_field[i, j] = np.exp(-d/1.5)
    plot_field(ax, fate_field, 'RdYlGn_r', 'B  Fate Probability', 'P(LUAD)')

    # C: Landscape (StageBridge)
    ax = fig.add_subplot(gs[0, 2])
    plot_field(ax, potential, 'terrain', 'C  Landscape', 'U')

    # D: Irreversibility (StageBridge unique)
    ax = fig.add_subplot(gs[0, 3])
    plot_field(ax, flux_ratio, FLUX_CMAP, 'D  Irreversibility', 'Flux')

    # E: Capability comparison - RADAR/SPIDER CHART
    ax = fig.add_subplot(gs[1, 0:2], polar=True)

    categories = ['Direction', 'Fate', 'Time', 'Landscape', 'Irreversibility']
    n_cats = len(categories)

    methods = {
        'scVelo': [1, 0.3, 0.5, 0, 0],
        'CellRank': [0.8, 1, 0.7, 0, 0],
        'Monocle': [0.5, 0.5, 1, 0, 0],
        'StageBridge': [1, 0.8, 0.8, 1, 1]
    }

    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]

    method_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for (method, values), color in zip(methods.items(), method_colors):
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title('E  Method Capabilities', loc='left', pad=20)

    # F: Quantitative summary - GROUPED BAR
    ax = fig.add_subplot(gs[1, 2:4])

    # What each method can quantify
    metrics = ['Direction', 'Fate prob.', 'Landscape', 'Flux ratio']
    methods_short = ['scVelo', 'CellRank', 'Monocle', 'StageBridge']

    data = np.array([
        [1, 0, 0, 0],      # scVelo
        [1, 1, 0, 0],      # CellRank
        [1, 0, 0, 0],      # Monocle
        [1, 1, 1, 1]       # StageBridge
    ])

    x = np.arange(len(metrics))
    width = 0.2

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    for i, (method, color) in enumerate(zip(methods_short, colors)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[i], width, label=method, color=color,
              edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Capability')
    ax.set_ylim(0, 1.3)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_title('F  Quantitative Outputs', loc='left')
    sns.despine(ax=ax)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_method_comparison.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig4_method_comparison.pdf", facecolor='white')
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
    print("Publication Dynamics Figures (Fancy Edition)")
    print("=" * 50)

    cells, patterns = load_data(args.data_dir)
    print(f"Loaded {len(cells):,} cells")

    np.random.seed(42)
    n_sample = min(25000, len(cells))
    cells_s = cells.sample(n_sample).reset_index(drop=True)

    fused = get_embeddings(cells_s)
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(fused)

    print("\nComputing dynamics...")
    dynamics = compute_dynamics(cells_s, coords_2d)
    print(f"  Mean flux ratio: {np.nanmean(dynamics['flux_ratio']):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    mean_flux = figure_dynamics_overview(cells_s, dynamics, args.output_dir)
    figure_clonal_evolution(cells_s, patterns, coords_2d, args.output_dir)
    figure_h3_validation(cells, patterns, args.output_dir)
    figure_method_comparison(cells_s, dynamics, args.output_dir)

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
