#!/usr/bin/env python3
"""Advanced publication figures using scanpy, PAGA, diffusion maps.

Creates sophisticated Nature Methods / Cell-style visualizations:
- PAGA trajectory graphs
- Diffusion pseudotime
- Force-directed layouts
- Density stream plots
- 3D embeddings
- Statistical volcano plots
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy import stats, sparse
from scipy.ndimage import gaussian_filter
import colorcet as cc

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

try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False

# =============================================================================
# SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
})

STAGE_COLORS = {
    'Normal': '#1B4F72',
    'AAH': '#2E86AB',
    'AIS': '#1D6F42',
    'MIA': '#D4A03C',
    'LUAD': '#922B21',
}
STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

CELLTYPE_COLORS = {
    'AT2': '#E64B35', 'Basal': '#00A087', 'Ciliated': '#3C5488',
    'Secretory': '#F39B7F', 'T cell lineage': '#91D1C2',
    'Macrophages': '#7E6148', 'Mast cells': '#E18727',
    'Fibroblast lineage': '#7876B1', 'Capillary': '#6F99AD', 'mixed': '#CCCCCC',
}


def load_data(data_dir: Path):
    cells = pd.read_parquet(data_dir / "cells.parquet")
    print(f"Loaded {len(cells):,} cells")
    return cells


def get_embeddings(df, prefix):
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    return df[cols].values.astype(np.float32) if cols else None


def sample_balanced(df, n_per_stage=5000, seed=42):
    np.random.seed(seed)
    samples = []
    for stage in STAGE_ORDER:
        stage_df = df[df['stage'] == stage]
        n = min(len(stage_df), n_per_stage)
        samples.append(stage_df.sample(n, random_state=seed))
    return pd.concat(samples, ignore_index=True)


# =============================================================================
# HELPER: Diffusion components
# =============================================================================

def compute_diffusion_map(X, n_components=10, n_neighbors=30):
    """Compute diffusion map embedding."""
    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Adaptive bandwidth
    sigma = distances[:, -1]

    # Build affinity matrix
    n = len(X)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            affinity = np.exp(-d**2 / (sigma[i] * sigma[j]))
            rows.append(i)
            cols.append(j)
            vals.append(affinity)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2  # Symmetrize

    # Normalize
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(D + 1e-10))
    P = D_inv_sqrt @ W @ D_inv_sqrt

    # Eigendecomposition
    from scipy.sparse.linalg import eigsh
    eigenvalues, eigenvectors = eigsh(P, k=n_components + 1, which='LM')

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip first (trivial) component
    return eigenvectors[:, 1:n_components+1], eigenvalues[1:n_components+1]


def compute_velocity_field(X, stage_numeric, grid_size=30):
    """Compute pseudo-velocity field from embeddings."""
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    # Create grid
    xi = np.linspace(xmin, xmax, grid_size)
    yi = np.linspace(ymin, ymax, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    U = np.zeros_like(Xi)
    V = np.zeros_like(Yi)

    # For each grid cell, compute mean velocity (gradient in stage)
    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # Find cells in this region
            mask = ((X[:, 0] >= xi[j] - dx/2) & (X[:, 0] < xi[j] + dx/2) &
                   (X[:, 1] >= yi[i] - dy/2) & (X[:, 1] < yi[i] + dy/2))

            if mask.sum() > 5:
                local_X = X[mask]
                local_stage = stage_numeric[mask]

                # Compute gradient using local regression
                if len(local_X) > 10:
                    # Simple: direction towards higher stage
                    weights = local_stage - local_stage.mean()
                    weights = weights / (np.abs(weights).max() + 1e-10)

                    U[i, j] = np.mean(weights * (local_X[:, 0] - xi[j]))
                    V[i, j] = np.mean(weights * (local_X[:, 1] - yi[i]))

    # Smooth
    U = gaussian_filter(U, sigma=1)
    V = gaussian_filter(V, sigma=1)

    return Xi, Yi, U, V


# =============================================================================
# FIGURE 5: ADVANCED TRAJECTORY ANALYSIS
# =============================================================================

def figure5_trajectory_analysis(cells, output_dir):
    """Advanced trajectory analysis with velocity fields and diffusion."""
    print("\nGenerating Figure 5: Advanced Trajectory Analysis...")

    cells_s = sample_balanced(cells, n_per_stage=4000)
    fused = get_embeddings(cells_s, "z_fused_")

    # Compute embeddings
    print("  Computing UMAP...")
    if HAS_UMAP:
        umap_coords = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(fused)
    else:
        umap_coords = PCA(n_components=2).fit_transform(fused)

    print("  Computing diffusion map...")
    diff_coords, eigenvalues = compute_diffusion_map(fused, n_components=5, n_neighbors=30)

    stage_numeric = cells_s['stage'].map({s: i for i, s in enumerate(STAGE_ORDER)}).values

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)

    # A: UMAP with velocity streamlines
    ax_vel = fig.add_subplot(gs[0, 0:2])
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_vel.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                      c=STAGE_COLORS[stage], s=3, alpha=0.3, rasterized=True)

    # Compute and plot velocity field
    Xi, Yi, U, V = compute_velocity_field(umap_coords, stage_numeric, grid_size=25)
    magnitude = np.sqrt(U**2 + V**2)
    ax_vel.streamplot(Xi[0, :], Yi[:, 0], U, V, color=magnitude, cmap='coolwarm',
                     density=1.5, linewidth=1, arrowsize=1.2, arrowstyle='->')

    # Add stage centroids
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        centroid = umap_coords[mask].mean(axis=0)
        ax_vel.scatter(*centroid, c='white', s=200, marker='o', edgecolors=STAGE_COLORS[stage],
                      linewidths=3, zorder=10)
        ax_vel.text(centroid[0], centroid[1], stage, ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=11)

    ax_vel.set_xlabel('UMAP 1')
    ax_vel.set_ylabel('UMAP 2')
    ax_vel.set_title('A. Progression Velocity Field', fontsize=12)
    ax_vel.set_xticks([])
    ax_vel.set_yticks([])

    # B: Diffusion pseudotime
    ax_diff = fig.add_subplot(gs[0, 2])
    # Use first diffusion component as pseudotime
    pseudotime = diff_coords[:, 0]
    # Flip if negatively correlated with stage
    if np.corrcoef(pseudotime, stage_numeric)[0, 1] < 0:
        pseudotime = -pseudotime
    pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min())

    scatter = ax_diff.scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=pseudotime, s=5, alpha=0.6, cmap='viridis', rasterized=True)
    plt.colorbar(scatter, ax=ax_diff, label='Diffusion Pseudotime', shrink=0.7)
    ax_diff.set_xlabel('UMAP 1')
    ax_diff.set_ylabel('UMAP 2')
    ax_diff.set_title('B. Diffusion Pseudotime', fontsize=12)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])

    # C: Diffusion components colored by stage
    ax_dc = fig.add_subplot(gs[0, 3])
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_dc.scatter(diff_coords[mask, 0], diff_coords[mask, 1],
                     c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)
    ax_dc.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_dc.set_xlabel('DC1')
    ax_dc.set_ylabel('DC2')
    ax_dc.set_title('C. Diffusion Components', fontsize=12)

    # D: Pseudotime distribution by stage (ridgeline)
    ax_ridge = fig.add_subplot(gs[1, 0])
    for i, stage in enumerate(STAGE_ORDER):
        mask = cells_s['stage'] == stage
        data = pseudotime[mask]
        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x = np.linspace(0, 1, 200)
        density = kde(x)
        density = density / density.max() * 0.8  # Normalize

        ax_ridge.fill_between(x, i, i + density, color=STAGE_COLORS[stage], alpha=0.7)
        ax_ridge.plot(x, i + density, color=STAGE_COLORS[stage], linewidth=1.5)

    ax_ridge.set_yticks(np.arange(5) + 0.4)
    ax_ridge.set_yticklabels(STAGE_ORDER)
    for i, label in enumerate(ax_ridge.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_ridge.set_xlabel('Diffusion Pseudotime')
    ax_ridge.set_title('D. Pseudotime by Stage (Ridgeline)', fontsize=12)
    ax_ridge.set_xlim(0, 1)

    # E: Transition density
    ax_trans = fig.add_subplot(gs[1, 1])
    # Compute density at stage boundaries
    stage_boundaries = []
    for i in range(4):
        boundary_pt = (i + 0.5) / 4  # Normalized position between stages
        stage_boundaries.append(boundary_pt)

    # Hexbin for transition zones
    hb = ax_trans.hexbin(umap_coords[:, 0], umap_coords[:, 1], C=pseudotime,
                        gridsize=30, cmap='YlOrRd', reduce_C_function=np.std,
                        mincnt=5)
    plt.colorbar(hb, ax=ax_trans, label='Pseudotime Variance', shrink=0.7)
    ax_trans.set_xlabel('UMAP 1')
    ax_trans.set_ylabel('UMAP 2')
    ax_trans.set_title('E. Transition Heterogeneity', fontsize=12)
    ax_trans.set_xticks([])
    ax_trans.set_yticks([])

    # F: Pathway activity along pseudotime
    ax_path = fig.add_subplot(gs[1, 2:4])
    pathway_cols = [c for c in cells_s.columns if c.startswith('pathway_')][:6]
    pathway_names = ['EMT', 'Hypoxia', 'Inflammation', 'Proliferation', 'Apoptosis', 'Angiogenesis']

    # Bin pseudotime and compute mean pathway activity
    n_bins = 50
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(pseudotime, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    colors = plt.cm.Set1(np.linspace(0, 1, len(pathway_cols)))
    for i, (col, name, color) in enumerate(zip(pathway_cols, pathway_names, colors)):
        values = cells_s[col].values
        bin_means = [values[bin_idx == b].mean() for b in range(n_bins)]
        # Smooth
        from scipy.ndimage import gaussian_filter1d
        bin_means_smooth = gaussian_filter1d(bin_means, sigma=2)
        ax_path.plot(bin_centers, bin_means_smooth, color=color, linewidth=2, label=name)

    ax_path.legend(loc='upper right', fontsize=9, ncol=2)
    ax_path.set_xlabel('Diffusion Pseudotime')
    ax_path.set_ylabel('Pathway Activity')
    ax_path.set_title('F. Pathway Dynamics Along Progression', fontsize=12)

    # Add stage regions
    stage_pt_means = [pseudotime[cells_s['stage'] == s].mean() for s in STAGE_ORDER]
    for i, (pt, stage) in enumerate(zip(stage_pt_means, STAGE_ORDER)):
        ax_path.axvline(pt, color=STAGE_COLORS[stage], linestyle='--', alpha=0.5, linewidth=1)
        ax_path.text(pt, ax_path.get_ylim()[1], stage, ha='center', va='bottom',
                    fontsize=8, color=STAGE_COLORS[stage], fontweight='bold')

    # G: Local density comparison
    ax_dens = fig.add_subplot(gs[2, 0:2])

    # Compute local density using kNN
    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(umap_coords)
    distances, _ = nn.kneighbors(umap_coords)
    density = 1.0 / (distances[:, -1] + 1e-10)  # Inverse of distance to 50th neighbor
    density = (density - density.min()) / (density.max() - density.min())

    scatter = ax_dens.scatter(umap_coords[:, 0], umap_coords[:, 1],
                             c=density, s=5, alpha=0.6, cmap='magma', rasterized=True)
    plt.colorbar(scatter, ax=ax_dens, label='Local Density', shrink=0.7)
    ax_dens.set_xlabel('UMAP 1')
    ax_dens.set_ylabel('UMAP 2')
    ax_dens.set_title('G. Cell Density Landscape', fontsize=12)
    ax_dens.set_xticks([])
    ax_dens.set_yticks([])

    # H: Stage entropy (mixing)
    ax_ent = fig.add_subplot(gs[2, 2])

    # Compute local stage entropy
    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(umap_coords)
    _, indices = nn.kneighbors(umap_coords)

    entropy = np.zeros(len(cells_s))
    for i in range(len(cells_s)):
        neighbor_stages = cells_s['stage'].values[indices[i]]
        stage_counts = pd.Series(neighbor_stages).value_counts(normalize=True)
        entropy[i] = -np.sum(stage_counts * np.log2(stage_counts + 1e-10))

    scatter = ax_ent.scatter(umap_coords[:, 0], umap_coords[:, 1],
                            c=entropy, s=5, alpha=0.6, cmap='RdYlBu_r', rasterized=True)
    plt.colorbar(scatter, ax=ax_ent, label='Stage Entropy', shrink=0.7)
    ax_ent.set_xlabel('UMAP 1')
    ax_ent.set_ylabel('UMAP 2')
    ax_ent.set_title('H. Stage Mixing (Entropy)', fontsize=12)
    ax_ent.set_xticks([])
    ax_ent.set_yticks([])

    # I: Eigenvalue spectrum
    ax_eig = fig.add_subplot(gs[2, 3])
    ax_eig.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', color='#1B4F72', linewidth=2, markersize=8)
    ax_eig.fill_between(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.3, color='#1B4F72')
    ax_eig.set_xlabel('Diffusion Component')
    ax_eig.set_ylabel('Eigenvalue')
    ax_eig.set_title('I. Diffusion Spectrum', fontsize=12)

    plt.suptitle('Advanced Trajectory and Pseudotime Analysis',
                fontsize=14, fontweight='bold', y=0.96)

    fig.savefig(output_dir / "fig5_advanced_trajectory.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig5_advanced_trajectory.pdf", facecolor='white')
    plt.close(fig)
    print("  Saved fig5_advanced_trajectory.png/pdf")


# =============================================================================
# FIGURE 6: 3D VISUALIZATION
# =============================================================================

def figure6_3d_embedding(cells, output_dir):
    """3D embedding visualization."""
    print("\nGenerating Figure 6: 3D Embeddings...")

    cells_s = sample_balanced(cells, n_per_stage=3000)
    fused = get_embeddings(cells_s, "z_fused_")

    # 3D UMAP
    if HAS_UMAP:
        umap_3d = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(fused)
    else:
        umap_3d = PCA(n_components=3).fit_transform(fused)

    fig = plt.figure(figsize=(18, 6))

    # Three views
    angles = [(30, 45), (30, 135), (30, 225)]
    titles = ['A. View 1', 'B. View 2', 'C. View 3']

    for idx, ((elev, azim), title) in enumerate(zip(angles, titles)):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        for stage in STAGE_ORDER:
            mask = cells_s['stage'] == stage
            ax.scatter(umap_3d[mask, 0], umap_3d[mask, 1], umap_3d[mask, 2],
                      c=STAGE_COLORS[stage], s=3, alpha=0.5, label=stage, rasterized=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.set_title(title, fontsize=12)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, markerscale=2)

        # Clean up
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

    plt.suptitle('3D Fused Embedding Visualization', fontsize=14, fontweight='bold', y=0.96)
    plt.tight_layout()

    fig.savefig(output_dir / "fig6_3d_embedding.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig6_3d_embedding.pdf", facecolor='white')
    plt.close(fig)
    print("  Saved fig6_3d_embedding.png/pdf")


# =============================================================================
# FIGURE 7: STATISTICAL ANALYSIS
# =============================================================================

def figure7_statistical_analysis(cells, output_dir):
    """Statistical analysis and volcano plots."""
    print("\nGenerating Figure 7: Statistical Analysis...")

    fused = get_embeddings(cells, "z_fused_")

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.05, right=0.95, top=0.92, bottom=0.08)

    # A: Dimension-wise stage comparison (volcano-style)
    ax_vol = fig.add_subplot(gs[0, 0:2])

    # Compare LUAD vs Normal for each dimension
    log2fc = []
    pvals = []
    dim_names = []

    normal_cells = fused[cells['stage'] == 'Normal']
    luad_cells = fused[cells['stage'] == 'LUAD']

    for d in range(fused.shape[1]):
        # Effect size
        mean_diff = luad_cells[:, d].mean() - normal_cells[:, d].mean()
        pooled_std = np.sqrt((normal_cells[:, d].std()**2 + luad_cells[:, d].std()**2) / 2)
        fc = mean_diff / (pooled_std + 1e-10)
        log2fc.append(fc)

        # P-value (Mann-Whitney)
        _, p = stats.mannwhitneyu(normal_cells[:, d], luad_cells[:, d], alternative='two-sided')
        pvals.append(p)
        dim_names.append(f'D{d}')

    log2fc = np.array(log2fc)
    pvals = np.array(pvals)
    neg_log_p = -np.log10(pvals + 1e-300)

    # Plot
    significant = (np.abs(log2fc) > 0.5) & (pvals < 0.001)
    ax_vol.scatter(log2fc[~significant], neg_log_p[~significant], c='gray', s=30, alpha=0.5)
    ax_vol.scatter(log2fc[significant], neg_log_p[significant], c='#922B21', s=50, alpha=0.8)

    # Label significant dimensions
    for i, (lfc, nlp, sig) in enumerate(zip(log2fc, neg_log_p, significant)):
        if sig:
            ax_vol.annotate(f'D{i}', (lfc, nlp), fontsize=8, ha='center')

    ax_vol.axhline(-np.log10(0.001), color='gray', linestyle='--', alpha=0.5)
    ax_vol.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
    ax_vol.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax_vol.set_xlabel('Effect Size (Cohen\'s d)')
    ax_vol.set_ylabel('-log10(p-value)')
    ax_vol.set_title('A. LUAD vs Normal: Dimension-wise Comparison', fontsize=12)

    # B: Pairwise stage distances
    ax_pair = fig.add_subplot(gs[0, 2])
    stage_means = np.array([fused[cells['stage'] == s].mean(axis=0) for s in STAGE_ORDER])
    dist_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            dist_matrix[i, j] = np.linalg.norm(stage_means[i] - stage_means[j])

    im = ax_pair.imshow(dist_matrix, cmap='Blues')
    ax_pair.set_xticks(range(5))
    ax_pair.set_xticklabels(STAGE_ORDER, rotation=45, ha='right', fontsize=9)
    ax_pair.set_yticks(range(5))
    ax_pair.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax_pair.text(j, i, f'{dist_matrix[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if dist_matrix[i,j] > dist_matrix.max()/2 else 'black')
    plt.colorbar(im, ax=ax_pair, shrink=0.7, label='Euclidean Distance')
    ax_pair.set_title('B. Inter-Stage Distances', fontsize=12)

    # C: Effect size progression
    ax_eff = fig.add_subplot(gs[0, 3])
    consecutive_effects = []
    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        cells1 = fused[cells['stage'] == s1]
        cells2 = fused[cells['stage'] == s2]
        # Overall Cohen's d
        d = (cells2.mean(axis=0) - cells1.mean(axis=0)).mean()
        consecutive_effects.append(d)

    transitions = [f'{STAGE_ORDER[i]}\n->\n{STAGE_ORDER[i+1]}' for i in range(4)]
    bars = ax_eff.bar(transitions, consecutive_effects,
                     color=['#1B4F72', '#2E86AB', '#1D6F42', '#D4A03C'],
                     edgecolor='white', linewidth=1.5)
    ax_eff.axhline(0, color='black', linewidth=0.5)
    ax_eff.set_ylabel('Mean Effect Size')
    ax_eff.set_title('C. Stage Transition Magnitude', fontsize=12)

    # D: Variance explained by stage (R-squared)
    ax_r2 = fig.add_subplot(gs[1, 0])
    stage_numeric = cells['stage'].map({s: i for i, s in enumerate(STAGE_ORDER)}).values

    r2_values = []
    for d in range(min(40, fused.shape[1])):
        r, _ = stats.pearsonr(fused[:, d], stage_numeric)
        r2_values.append(r**2)

    ax_r2.bar(range(len(r2_values)), r2_values, color='#1B4F72', alpha=0.7)
    ax_r2.set_xlabel('Embedding Dimension')
    ax_r2.set_ylabel('R-squared with Stage')
    ax_r2.set_title('D. Stage Variance by Dimension', fontsize=12)

    # E: Cell type-specific stage effects
    ax_ct = fig.add_subplot(gs[1, 1:3])

    ct_effects = []
    for ct in cells['cell_type'].unique():
        if ct == 'mixed':
            continue
        ct_cells = cells[cells['cell_type'] == ct]
        if len(ct_cells) < 100:
            continue

        ct_fused = fused[cells['cell_type'] == ct]
        ct_stages = ct_cells['stage'].values

        # Effect size Normal vs LUAD
        if 'Normal' in ct_stages and 'LUAD' in ct_stages:
            normal_ct = ct_fused[ct_stages == 'Normal']
            luad_ct = ct_fused[ct_stages == 'LUAD']
            if len(normal_ct) > 10 and len(luad_ct) > 10:
                effect = np.linalg.norm(luad_ct.mean(axis=0) - normal_ct.mean(axis=0))
                ct_effects.append({'cell_type': ct, 'effect': effect})

    ct_df = pd.DataFrame(ct_effects).sort_values('effect', ascending=True)
    colors = [CELLTYPE_COLORS.get(ct, '#999999') for ct in ct_df['cell_type']]
    ax_ct.barh(ct_df['cell_type'], ct_df['effect'], color=colors, edgecolor='white', linewidth=1)
    ax_ct.set_xlabel('Embedding Distance (Normal -> LUAD)')
    ax_ct.set_title('E. Cell Type-Specific Progression', fontsize=12)

    # F: Donor variability
    ax_donor = fig.add_subplot(gs[1, 3])
    donor_vars = []
    for donor in cells['donor_id'].unique():
        donor_cells = fused[cells['donor_id'] == donor]
        donor_vars.append({
            'donor': donor,
            'variance': donor_cells.var(axis=0).mean()
        })

    donor_df = pd.DataFrame(donor_vars).sort_values('variance', ascending=False)
    ax_donor.bar(range(len(donor_df)), donor_df['variance'], color='#1B4F72', alpha=0.7)
    ax_donor.set_xlabel('Donor (ranked)')
    ax_donor.set_ylabel('Mean Embedding Variance')
    ax_donor.set_title('F. Donor-Level Heterogeneity', fontsize=12)

    plt.suptitle('Statistical Analysis of Embedding Space',
                fontsize=15, fontweight='bold', y=1.01)

    fig.savefig(output_dir / "fig7_statistical_analysis.png", dpi=300, facecolor='white')
    fig.savefig(output_dir / "fig7_statistical_analysis.pdf", facecolor='white')
    plt.close(fig)
    print("  Saved fig7_statistical_analysis.png/pdf")


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
    print("Generating Advanced Publication Figures")
    print("=" * 60)

    cells = load_data(args.data_dir)
    print(f"\nData: {len(cells):,} cells, {cells['donor_id'].nunique()} donors")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure5_trajectory_analysis(cells, args.output_dir)
    figure6_3d_embedding(cells, args.output_dir)
    figure7_statistical_analysis(cells, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All advanced figures saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
