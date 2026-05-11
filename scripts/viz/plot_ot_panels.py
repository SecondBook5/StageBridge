#!/usr/bin/env python3
"""Generate individual OT dynamics panels for poster/publication.

Creates separate figure files for each panel:
A. Disease stages UMAP
B. OT velocity field with streamlines
C. OT distance bar chart
D. Divergence map
E. Curl (irreversibility) map
F. Irreversibility flux ratio map
G. Irreversibility by stage violin
H. Flow speed map
I. Progression cost curve
J. Transition propensity heatmap
K. Key metrics bar chart
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Color scheme matching the reference
STAGE_COLORS = {
    'Normal': '#1f77b4',  # blue
    'AAH': '#17becf',     # cyan
    'AIS': '#2ca02c',     # green
    'MIA': '#ff7f0e',     # orange
    'LUAD': '#d62728',    # red
}

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']


def load_data(data_dir: Path):
    """Load embeddings and OT results."""
    data = {}

    # Try loading embeddings
    emb_path = data_dir / 'embeddings'
    if emb_path.exists():
        for f in emb_path.glob('*.parquet'):
            data[f.stem] = pd.read_parquet(f)

    # Try loading cells
    cells_path = data_dir / 'cells.parquet'
    if cells_path.exists():
        data['cells'] = pd.read_parquet(cells_path)

    # Try loading GW alignment results
    gw_path = data_dir / 'gw_alignment'
    if gw_path.exists():
        for f in gw_path.glob('*.parquet'):
            data[f.stem] = pd.read_parquet(f)
        for f in gw_path.glob('*.npy'):
            data[f.stem] = np.load(f)

    return data


def panel_a_stages(umap: np.ndarray, stages: np.ndarray, output_dir: Path):
    """A. Disease Stages UMAP."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            ax.scatter(umap[mask, 0], umap[mask, 1],
                      c=STAGE_COLORS[stage], s=1, alpha=0.5,
                      label=stage, rasterized=True)

    ax.legend(markerscale=5, frameon=True, loc='upper right')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('A. Disease Stages')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_a_stages.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_a_stages.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_a_stages")


def panel_b_velocity_field(umap: np.ndarray, velocity: np.ndarray,
                           stages: np.ndarray, output_dir: Path):
    """B. OT Velocity Field with streamlines."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create grid for streamlines
    x_min, x_max = umap[:, 0].min(), umap[:, 0].max()
    y_min, y_max = umap[:, 1].min(), umap[:, 1].max()

    # Interpolate velocity to grid
    from scipy.interpolate import griddata

    n_grid = 30
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    Ui = griddata(umap, velocity[:, 0], (Xi, Yi), method='linear', fill_value=0)
    Vi = griddata(umap, velocity[:, 1], (Xi, Yi), method='linear', fill_value=0)

    # Speed for coloring
    speed = np.sqrt(Ui**2 + Vi**2)

    # Plot streamlines
    strm = ax.streamplot(xi, yi, Ui, Vi, color=speed, cmap='viridis',
                         density=1.5, linewidth=1, arrowsize=1)
    plt.colorbar(strm.lines, ax=ax, label='Speed')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
            ax.scatter(cx, cy, c=STAGE_COLORS[stage], s=100,
                      edgecolors='white', linewidths=2, zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('B. Optimal Transport Velocity Field')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_b_velocity.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_b_velocity.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_b_velocity")


def panel_c_ot_distance(w_distances: dict, output_dir: Path):
    """C. OT Distance bar chart."""
    fig, ax = plt.subplots(figsize=(4, 4))

    transitions = ['Normal\nAAH', 'AAH\nAIS', 'AIS\nMIA', 'MIA\nLUAD']
    colors = [STAGE_COLORS['AAH'], STAGE_COLORS['AIS'],
              STAGE_COLORS['MIA'], STAGE_COLORS['LUAD']]

    # Get distances
    distances = []
    for i, (s1, s2) in enumerate([('Normal', 'AAH'), ('AAH', 'AIS'),
                                   ('AIS', 'MIA'), ('MIA', 'LUAD')]):
        key = f'{s1}_{s2}'
        if key in w_distances:
            distances.append(w_distances[key])
        else:
            distances.append(0.1 * (i + 1))  # placeholder

    bars = ax.barh(transitions, distances, color=colors)

    # Add value labels
    for bar, val in zip(bars, distances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    ax.set_xlabel('Wasserstein Distance')
    ax.set_title('C. OT Distance')
    ax.invert_yaxis()

    fig.savefig(output_dir / 'panel_c_ot_distance.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_c_ot_distance.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_c_ot_distance")


def panel_d_divergence(umap: np.ndarray, divergence: np.ndarray,
                       stages: np.ndarray, output_dir: Path):
    """D. Divergence map."""
    fig, ax = plt.subplots(figsize=(6, 5))

    vmax = np.percentile(np.abs(divergence), 98)
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=divergence,
                        cmap='RdBu_r', s=1, alpha=0.7,
                        vmin=-vmax, vmax=vmax, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='div(v)')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
            ax.scatter(cx, cy, c=STAGE_COLORS[stage], s=80,
                      edgecolors='white', linewidths=1.5, zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('D. Divergence')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_d_divergence.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_d_divergence.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_d_divergence")


def panel_e_curl(umap: np.ndarray, curl: np.ndarray,
                 stages: np.ndarray, output_dir: Path):
    """E. Curl (Irreversibility) map."""
    fig, ax = plt.subplots(figsize=(6, 5))

    vmax = np.percentile(np.abs(curl), 98)
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=curl,
                        cmap='RdBu_r', s=1, alpha=0.7,
                        vmin=-vmax, vmax=vmax, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='curl(v)')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
            ax.scatter(cx, cy, c=STAGE_COLORS[stage], s=80,
                      edgecolors='white', linewidths=1.5, zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('E. Curl (Irreversibility)')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_e_curl.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_e_curl.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_e_curl")


def panel_f_irreversibility_map(umap: np.ndarray, flux_ratio: np.ndarray,
                                 stages: np.ndarray, output_dir: Path):
    """F. Irreversibility flux ratio map."""
    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=flux_ratio,
                        cmap='hot', s=1, alpha=0.7,
                        vmin=0, vmax=1, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Flux ratio')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
            ax.scatter(cx, cy, c='white', s=80,
                      edgecolors=STAGE_COLORS[stage], linewidths=2, zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('F. Irreversibility Map')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_f_irreversibility.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_f_irreversibility.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_f_irreversibility")


def panel_g_irreversibility_violin(flux_ratio: np.ndarray, stages: np.ndarray,
                                    output_dir: Path):
    """G. Irreversibility by Stage violin plot."""
    fig, ax = plt.subplots(figsize=(5, 4))

    data = []
    positions = []
    colors = []

    for i, stage in enumerate(STAGE_ORDER):
        mask = stages == stage
        if mask.sum() > 0:
            data.append(flux_ratio[mask])
            positions.append(i)
            colors.append(STAGE_COLORS[stage])

    parts = ax.violinplot(data, positions=positions, showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Add boxplots
    bp = ax.boxplot(data, positions=positions, widths=0.15,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=1)

    # Mean line
    mean_flux = np.mean(flux_ratio)
    ax.axhline(mean_flux, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_flux:.2f}')
    ax.legend(loc='upper right')

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Flux Ratio')
    ax.set_title('G. Irreversibility by Stage')
    ax.set_ylim(0, 1)

    fig.savefig(output_dir / 'panel_g_irreversibility_violin.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_g_irreversibility_violin.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_g_irreversibility_violin")


def panel_h_flow_speed(umap: np.ndarray, speed: np.ndarray,
                       stages: np.ndarray, output_dir: Path):
    """H. Flow Speed map."""
    fig, ax = plt.subplots(figsize=(6, 5))

    vmax = np.percentile(speed, 98)
    scatter = ax.scatter(umap[:, 0], umap[:, 1], c=speed,
                        cmap='hot', s=1, alpha=0.7,
                        vmin=0, vmax=vmax, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Speed')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            cx, cy = umap[mask, 0].mean(), umap[mask, 1].mean()
            ax.scatter(cx, cy, c='white', s=80,
                      edgecolors=STAGE_COLORS[stage], linewidths=2, zorder=10)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('H. Flow Speed')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_h_flow_speed.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_h_flow_speed.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_h_flow_speed")


def panel_i_progression_cost(w_distances: dict, output_dir: Path):
    """I. Progression Cost cumulative curve."""
    fig, ax = plt.subplots(figsize=(5, 4))

    stages = ['Start', 'AAH', 'AIS', 'MIA', 'LUAD']

    # Get cumulative distances
    cumulative = [0]
    pairs = [('Normal', 'AAH'), ('AAH', 'AIS'), ('AIS', 'MIA'), ('MIA', 'LUAD')]

    for s1, s2 in pairs:
        key = f'{s1}_{s2}'
        if key in w_distances:
            cumulative.append(cumulative[-1] + w_distances[key])
        else:
            cumulative.append(cumulative[-1] + 0.1)  # placeholder

    ax.fill_between(range(len(stages)), cumulative, alpha=0.3, color='steelblue')
    ax.plot(range(len(stages)), cumulative, 'o-', color='steelblue',
            markersize=8, linewidth=2)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.set_ylabel('Cumulative W Distance')
    ax.set_title('I. Progression Cost')

    fig.savefig(output_dir / 'panel_i_progression_cost.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_i_progression_cost.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_i_progression_cost")


def panel_j_transition_matrix(transition_probs: np.ndarray, output_dir: Path):
    """J. Transition Propensity heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(STAGE_ORDER)):
        for j in range(len(STAGE_ORDER)):
            val = transition_probs[i, j]
            if val > 0.01:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, rotation=45, ha='right')
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER)
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title('J. Transition Propensity')

    plt.colorbar(im, ax=ax, label='P(transition)')

    fig.savefig(output_dir / 'panel_j_transition.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_j_transition.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_j_transition")


def panel_k_key_metrics(metrics: dict, output_dir: Path):
    """K. Key Metrics bar chart."""
    fig, ax = plt.subplots(figsize=(4, 4))

    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['steelblue', '#b5651d']  # blue, brown

    bars = ax.bar(names, values, color=colors[:len(names)])

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Value')
    ax.set_title('K. Key Metrics')
    ax.set_ylim(0, max(values) * 1.15)

    fig.savefig(output_dir / 'panel_k_metrics.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_k_metrics.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel_k_metrics")


def main():
    parser = argparse.ArgumentParser(description='Generate OT dynamics panels')
    parser.add_argument('--data_dir', type=Path, required=True, help='Data directory with embeddings/gw_alignment')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--panels', type=str, default='all', help='Panels to generate (comma-separated: a,b,c,... or "all")')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args.data_dir)

    # Get UMAP and stages
    if 'cells' in data:
        cells = data['cells']
        if 'cell_id' in cells.columns:
            cells = cells.set_index('cell_id')
        stages = cells['stage'].values if 'stage' in cells.columns else None
    else:
        stages = None

    # Get UMAP coordinates
    umap = None
    if 'umap_coords' in data:
        umap = data['umap_coords']
    elif 'cells' in data and 'umap_1' in data['cells'].columns:
        umap = data['cells'][['umap_1', 'umap_2']].values

    # Determine which panels to generate
    if args.panels == 'all':
        panels = list('abcdefghijk')
    else:
        panels = [p.strip().lower() for p in args.panels.split(',')]

    print(f"Generating panels: {panels}")

    # Generate each panel
    if 'a' in panels and umap is not None and stages is not None:
        panel_a_stages(umap, stages, args.output_dir)

    if 'b' in panels and umap is not None and 'velocity' in data:
        panel_b_velocity_field(umap, data['velocity'], stages, args.output_dir)

    if 'c' in panels and 'w_distances' in data:
        panel_c_ot_distance(data['w_distances'], args.output_dir)

    if 'd' in panels and umap is not None and 'divergence' in data:
        panel_d_divergence(umap, data['divergence'], stages, args.output_dir)

    if 'e' in panels and umap is not None and 'curl' in data:
        panel_e_curl(umap, data['curl'], stages, args.output_dir)

    if 'f' in panels and umap is not None and 'flux_ratio' in data:
        panel_f_irreversibility_map(umap, data['flux_ratio'], stages, args.output_dir)

    if 'g' in panels and 'flux_ratio' in data and stages is not None:
        panel_g_irreversibility_violin(data['flux_ratio'], stages, args.output_dir)

    if 'h' in panels and umap is not None and 'speed' in data:
        panel_h_flow_speed(umap, data['speed'], stages, args.output_dir)

    if 'i' in panels and 'w_distances' in data:
        panel_i_progression_cost(data['w_distances'], args.output_dir)

    if 'j' in panels and 'transition_probs' in data:
        panel_j_transition_matrix(data['transition_probs'], args.output_dir)

    if 'k' in panels and 'metrics' in data:
        panel_k_key_metrics(data['metrics'], args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
