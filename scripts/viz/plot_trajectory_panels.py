#!/usr/bin/env python3
"""Generate advanced trajectory and pseudotime panels (Fig 5).

Creates publication-quality panels showing:
A. Progression velocity field
B. Diffusion pseudotime UMAP
C. Diffusion components (DC1 vs pseudotime)
D. Pseudotime by stage (ridgeline)
E. Transition heterogeneity
F. Pathway dynamics along progression
G. Cell density landscape
H. Stage mixing (entropy)
I. Diffusion spectrum
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import gaussian_filter
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Publication color scheme
STAGE_COLORS = {
    'Normal': '#228B22',
    'AAH': '#4682B4',
    'AIS': '#4169E1',
    'MIA': '#8B008B',
    'LUAD': '#CB4154',
}
STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Pathway colors
PATHWAY_COLORS = {
    'EMT': '#E41A1C',
    'Proliferation': '#377EB8',
    'Hypoxia': '#FF7F00',
    'Inflammation': '#984EA3',
    'Apoptosis': '#4DAF4A',
    'Angiogenesis': '#F781BF',
}


def load_data(data_dir: Path):
    """Load trajectory analysis data."""
    data = {}

    # Load cells/metadata
    for pattern in ['cells.parquet', 'cell_metadata.parquet']:
        path = data_dir / pattern
        if path.exists():
            data['cells'] = pd.read_parquet(path)
            print(f"Loaded {path}: {len(data['cells'])} cells")
            break

    # Load UMAP
    for pattern in ['umap_coords.npy', 'umap.npy']:
        path = data_dir / pattern
        if path.exists():
            data['umap'] = np.load(path)
            print(f"Loaded UMAP: {data['umap'].shape}")
            break

    # Load diffusion pseudotime
    for pattern in ['diffusion_pseudotime.parquet', 'dpt.parquet']:
        path = data_dir / pattern
        if path.exists():
            dpt = pd.read_parquet(path)
            data['pseudotime'] = dpt['dpt_pseudotime'].values if 'dpt_pseudotime' in dpt.columns else dpt.iloc[:, 0].values
            print(f"Loaded pseudotime")
            break

    # Load diffusion components
    for pattern in ['diffmap.npy', 'X_diffmap.npy']:
        path = data_dir / pattern
        if path.exists():
            data['diffmap'] = np.load(path)
            print(f"Loaded diffmap: {data['diffmap'].shape}")
            break

    # Load velocity if available
    for pattern in ['velocity.npy', 'velocity_field.npy']:
        path = data_dir / pattern
        if path.exists():
            data['velocity'] = np.load(path)
            print(f"Loaded velocity: {data['velocity'].shape}")
            break

    # Load scores for pathway dynamics
    for pattern in ['caf_kac_scores.parquet', 'signatures/caf_kac_scores.parquet']:
        path = data_dir / pattern
        if path.exists():
            scores = pd.read_parquet(path)
            data['scores'] = scores
            print(f"Loaded scores: {len(scores.columns)} columns")
            break

    return data


def compute_velocity_field(umap: np.ndarray, stages: np.ndarray, n_neighbors: int = 30):
    """Compute velocity field based on stage progression."""
    from sklearn.neighbors import NearestNeighbors

    n_cells = len(umap)
    velocity = np.zeros((n_cells, 2))

    stage_order = [s for s in STAGE_ORDER if s in stages]

    for i, stage in enumerate(stage_order[:-1]):
        next_stage = stage_order[i + 1]
        current_mask = stages == stage
        next_mask = stages == next_stage

        if not np.any(current_mask) or not np.any(next_mask):
            continue

        current_cells = umap[current_mask]
        next_cells = umap[next_mask]

        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(next_cells)))
        nn.fit(next_cells)
        _, indices = nn.kneighbors(current_cells)

        for j, cell_idx in enumerate(np.where(current_mask)[0]):
            neighbor_positions = next_cells[indices[j]]
            direction = neighbor_positions.mean(axis=0) - umap[cell_idx]
            norm = np.linalg.norm(direction)
            if norm > 0:
                velocity[cell_idx] = direction / norm

    return velocity


def panel_a_velocity_field(data: dict, output_dir: Path):
    """A. Progression velocity field."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel A: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    if 'stage' not in cells.columns:
        print("  Skipping panel A: no stage column")
        return

    stages = cells['stage'].values

    # Compute or use existing velocity
    if 'velocity' in data:
        velocity = data['velocity']
    else:
        print("  Computing velocity field...")
        velocity = compute_velocity_field(umap, stages)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot cells
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            ax.scatter(umap[mask, 0], umap[mask, 1],
                      c=STAGE_COLORS.get(stage, 'gray'),
                      s=3, alpha=0.4, label=stage, rasterized=True)

    # Plot velocity field (subsample)
    has_vel = np.linalg.norm(velocity, axis=1) > 0
    valid_idx = np.where(has_vel)[0]
    if len(valid_idx) > 500:
        selected = np.random.choice(valid_idx, 500, replace=False)
    else:
        selected = valid_idx

    ax.quiver(umap[selected, 0], umap[selected, 1],
              velocity[selected, 0], velocity[selected, 1],
              angles='xy', scale_units='xy', scale=8,
              color='black', alpha=0.6, width=0.003)

    # Stage centroids with arrows
    centroids = []
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            centroids.append(umap[mask].mean(axis=0))

    centroids = np.array(centroids)
    for i in range(len(centroids) - 1):
        ax.annotate('', xy=centroids[i+1], xytext=centroids[i],
                   arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax.legend(markerscale=3, frameon=False, loc='upper right')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('A. Progression Velocity Field')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_A_velocity_field.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_A_velocity_field.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel A")


def panel_b_diffusion_pseudotime(data: dict, output_dir: Path):
    """B. Diffusion pseudotime UMAP."""
    if 'umap' not in data:
        print("  Skipping panel B: missing UMAP")
        return

    umap = data['umap']

    # Get or compute pseudotime
    if 'pseudotime' in data:
        pseudotime = data['pseudotime']
    elif 'cells' in data and 'dpt_pseudotime' in data['cells'].columns:
        pseudotime = data['cells']['dpt_pseudotime'].values
    else:
        print("  Skipping panel B: no pseudotime data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    valid = ~np.isnan(pseudotime)
    scatter = ax.scatter(umap[valid, 0], umap[valid, 1],
                        c=pseudotime[valid], cmap='viridis',
                        s=3, alpha=0.6, rasterized=True)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Diffusion Pseudotime')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('B. Diffusion Pseudotime')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_B_diffusion_pseudotime.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel B")


def panel_c_diffusion_components(data: dict, output_dir: Path):
    """C. Diffusion components (DC1 vs pseudotime)."""
    if 'diffmap' not in data:
        print("  Skipping panel C: no diffusion map")
        return

    diffmap = data['diffmap']
    cells = data.get('cells', pd.DataFrame())

    if 'pseudotime' in data:
        pseudotime = data['pseudotime']
    elif 'dpt_pseudotime' in cells.columns:
        pseudotime = cells['dpt_pseudotime'].values
    else:
        pseudotime = None

    fig, ax = plt.subplots(figsize=(6, 5))

    if 'stage' in cells.columns:
        stages = cells['stage'].values
        for stage in STAGE_ORDER:
            mask = stages == stage
            if mask.sum() > 0:
                ax.scatter(diffmap[mask, 0],
                          pseudotime[mask] if pseudotime is not None else diffmap[mask, 1],
                          c=STAGE_COLORS.get(stage, 'gray'),
                          s=3, alpha=0.5, label=stage, rasterized=True)
        ax.legend(markerscale=3, frameon=False)
    else:
        ax.scatter(diffmap[:, 0],
                  pseudotime if pseudotime is not None else diffmap[:, 1],
                  c='steelblue', s=3, alpha=0.5, rasterized=True)

    ax.set_xlabel('DC1')
    ax.set_ylabel('Diffusion Pseudotime' if pseudotime is not None else 'DC2')
    ax.set_title('C. Diffusion Components')

    fig.savefig(output_dir / 'panel_C_diffusion_components.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel C")


def panel_d_pseudotime_ridgeline(data: dict, output_dir: Path):
    """D. Pseudotime by stage (ridgeline plot)."""
    cells = data.get('cells', pd.DataFrame())

    if 'stage' not in cells.columns:
        print("  Skipping panel D: no stage column")
        return

    if 'pseudotime' in data:
        pseudotime = data['pseudotime']
    elif 'dpt_pseudotime' in cells.columns:
        pseudotime = cells['dpt_pseudotime'].values
    else:
        print("  Skipping panel D: no pseudotime")
        return

    stages = cells['stage'].values
    stage_order = [s for s in STAGE_ORDER if s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create ridgeline plot
    for i, stage in enumerate(reversed(stage_order)):
        mask = (stages == stage) & (~np.isnan(pseudotime))
        vals = pseudotime[mask]
        if len(vals) < 10:
            continue

        # KDE
        kde = stats.gaussian_kde(vals)
        x = np.linspace(0, 1, 200)
        y = kde(x)

        # Scale and offset
        y_scaled = y / y.max() * 0.8
        offset = i

        ax.fill_between(x, offset, offset + y_scaled,
                       color=STAGE_COLORS.get(stage, 'gray'), alpha=0.7)
        ax.plot(x, offset + y_scaled, color='black', lw=1)

    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(list(reversed(stage_order)))
    ax.set_xlabel('Diffusion Pseudotime')
    ax.set_xlim(0, 1)
    ax.set_title('D. Pseudotime by Stage (Ridgeline)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'panel_D_pseudotime_ridgeline.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_D_pseudotime_ridgeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel D")


def panel_e_transition_heterogeneity(data: dict, output_dir: Path):
    """E. Transition heterogeneity (local variance in pseudotime)."""
    if 'umap' not in data:
        print("  Skipping panel E: missing UMAP")
        return

    umap = data['umap']

    if 'pseudotime' in data:
        pseudotime = data['pseudotime']
    elif 'cells' in data and 'dpt_pseudotime' in data['cells'].columns:
        pseudotime = data['cells']['dpt_pseudotime'].values
    else:
        print("  Skipping panel E: no pseudotime")
        return

    # Compute local variance using KNN
    from sklearn.neighbors import NearestNeighbors

    valid = ~np.isnan(pseudotime)
    umap_valid = umap[valid]
    pt_valid = pseudotime[valid]

    nn = NearestNeighbors(n_neighbors=30)
    nn.fit(umap_valid)
    _, indices = nn.kneighbors(umap_valid)

    local_var = np.zeros(len(pt_valid))
    for i in range(len(pt_valid)):
        neighbor_pt = pt_valid[indices[i]]
        local_var[i] = np.var(neighbor_pt)

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(umap_valid[:, 0], umap_valid[:, 1],
                        c=local_var, cmap='hot', s=3, alpha=0.7,
                        vmin=0, vmax=np.percentile(local_var, 95),
                        rasterized=True)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Transition Heterogeneity')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('E. Transition Heterogeneity')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_E_transition_heterogeneity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel E")


def panel_f_pathway_dynamics(data: dict, output_dir: Path):
    """F. Pathway dynamics along progression."""
    cells = data.get('cells', pd.DataFrame())
    scores = data.get('scores', pd.DataFrame())

    if 'pseudotime' in data:
        pseudotime = data['pseudotime']
    elif 'dpt_pseudotime' in cells.columns:
        pseudotime = cells['dpt_pseudotime'].values
    else:
        print("  Skipping panel F: no pseudotime")
        return

    # Find pathway score columns
    pathway_mapping = {
        'EMT': ['EMT_score', 'emt_score'],
        'Proliferation': ['entropic_score', 'proliferation_score'],
        'Hypoxia': ['hypoxia_score'],
        'Inflammation': ['IL1_axis_score', 'NFkB_score', 'inflammation_score'],
        'Apoptosis': ['p53_pathway_score', 'apoptosis_score'],
        'Angiogenesis': ['angiogenesis_score', 'VEGF_score'],
    }

    # Merge scores into cells if separate
    if len(scores) > 0:
        for col in scores.columns:
            if col.endswith('_score') and col not in cells.columns:
                cells[col] = scores[col].values[:len(cells)]

    fig, ax = plt.subplots(figsize=(8, 5))

    valid = ~np.isnan(pseudotime)
    pt_valid = pseudotime[valid]

    # Bin pseudotime
    n_bins = 50
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(pt_valid, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for pathway, candidates in pathway_mapping.items():
        col = None
        for c in candidates:
            if c in cells.columns:
                col = c
                break

        if col is None:
            continue

        values = cells[col].values[valid]

        # Compute mean per bin
        bin_means = np.zeros(n_bins)
        for i in range(n_bins):
            mask = bin_idx == i
            if mask.sum() > 0:
                bin_means[i] = np.nanmean(values[mask])

        # Smooth
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(bin_means, sigma=2)

        # Z-score normalize for comparison
        smoothed_z = (smoothed - smoothed.mean()) / (smoothed.std() + 1e-10)

        ax.plot(bin_centers, smoothed_z,
               color=PATHWAY_COLORS.get(pathway, 'gray'),
               lw=2, label=pathway)

    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel('Diffusion Pseudotime')
    ax.set_ylabel('Pathway Score (z-normalized)')
    ax.set_title('F. Pathway Dynamics Along Progression')
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'panel_F_pathway_dynamics.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_F_pathway_dynamics.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel F")


def panel_g_density_landscape(data: dict, output_dir: Path):
    """G. Cell density landscape."""
    if 'umap' not in data:
        print("  Skipping panel G: missing UMAP")
        return

    umap = data['umap']

    fig, ax = plt.subplots(figsize=(6, 5))

    # 2D histogram / density
    h, xedges, yedges = np.histogram2d(umap[:, 0], umap[:, 1], bins=100)
    h = gaussian_filter(h.T, sigma=2)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(np.log1p(h), origin='lower', extent=extent,
                  cmap='magma', aspect='auto')

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Log Density')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('G. Cell Density Landscape')

    fig.savefig(output_dir / 'panel_G_density_landscape.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel G")


def panel_h_stage_entropy(data: dict, output_dir: Path):
    """H. Stage mixing (entropy) UMAP."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel H: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    if 'stage' not in cells.columns:
        print("  Skipping panel H: no stage column")
        return

    stages = cells['stage'].values

    # Compute local stage entropy using KNN
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(umap)
    _, indices = nn.kneighbors(umap)

    stage_to_idx = {s: i for i, s in enumerate(STAGE_ORDER)}
    n_stages = len(STAGE_ORDER)

    entropy = np.zeros(len(umap))
    for i in range(len(umap)):
        neighbor_stages = stages[indices[i]]
        counts = np.zeros(n_stages)
        for s in neighbor_stages:
            if s in stage_to_idx:
                counts[stage_to_idx[s]] += 1

        p = counts / counts.sum()
        p = p[p > 0]
        entropy[i] = -np.sum(p * np.log2(p))

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(umap[:, 0], umap[:, 1],
                        c=entropy, cmap='coolwarm', s=3, alpha=0.7,
                        vmin=0, vmax=np.log2(n_stages),
                        rasterized=True)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Stage Entropy')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('H. Stage Mixing (Entropy)')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_dir / 'panel_H_stage_entropy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel H")


def panel_i_diffusion_spectrum(data: dict, output_dir: Path):
    """I. Diffusion spectrum (eigenvalue decay)."""
    if 'diffmap' not in data:
        # Try to compute from cells
        print("  Skipping panel I: no diffusion map available")
        return

    diffmap = data['diffmap']

    # Compute variance explained by each component
    variance = np.var(diffmap, axis=0)
    variance_ratio = variance / variance.sum()

    fig, ax = plt.subplots(figsize=(6, 4))

    n_comps = min(10, len(variance_ratio))
    x = range(1, n_comps + 1)

    ax.bar(x, variance_ratio[:n_comps], color='steelblue', alpha=0.7)
    ax.plot(x, variance_ratio[:n_comps], 'ko-', ms=6)

    # Highlight gap if present
    if n_comps > 2:
        ratios = variance_ratio[:n_comps-1] / variance_ratio[1:n_comps]
        max_gap_idx = np.argmax(ratios)
        ax.axvline(max_gap_idx + 1.5, color='red', ls='--', lw=2, alpha=0.7)

    ax.set_xlabel('Diffusion Component')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('I. Diffusion Spectrum')
    ax.set_xticks(x)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'panel_I_diffusion_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel I")


def main():
    parser = argparse.ArgumentParser(description='Generate trajectory panels (Fig 5)')
    parser.add_argument('--data_dir', type=Path, required=True, help='Directory with data')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args.data_dir)

    print("\nGenerating panels...")
    panel_a_velocity_field(data, args.output_dir)
    panel_b_diffusion_pseudotime(data, args.output_dir)
    panel_c_diffusion_components(data, args.output_dir)
    panel_d_pseudotime_ridgeline(data, args.output_dir)
    panel_e_transition_heterogeneity(data, args.output_dir)
    panel_f_pathway_dynamics(data, args.output_dir)
    panel_g_density_landscape(data, args.output_dir)
    panel_h_stage_entropy(data, args.output_dir)
    panel_i_diffusion_spectrum(data, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
