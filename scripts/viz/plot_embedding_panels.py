#!/usr/bin/env python3
"""Generate embedding overview figure panels (Fig 1).

Creates publication-quality panels showing:
A. Fused dual-reference embeddings by disease stage
B. Stage density contours
C. Cell type distribution
D. Cell cycle phase
E. Proliferation score
F. Stage distribution (bar chart)
G. Cell type by stage (heatmap)
H. Mean fused embedding (heatmap)
I. Stage-discriminative dimensions (F-ratio)
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

CELL_TYPE_COLORS = {
    'T cell lineage': '#E41A1C',
    'Macrophage': '#377EB8',
    'Fibroblast lineage': '#4DAF4A',
    'Epithelial': '#984EA3',
    'AT2': '#FF7F00',
    'Mast cells': '#FFFF33',
    'Endothelial': '#A65628',
    'Secretory': '#F781BF',
}

CELL_CYCLE_COLORS = {
    'G1': '#2ecc71',
    'S': '#e74c3c',
    'G2M': '#9b59b6',
}


def load_data(data_dir: Path):
    """Load embedding and metadata."""
    data = {}

    # Try various file patterns
    for pattern in ['cells.parquet', 'cell_metadata.parquet']:
        path = data_dir / pattern
        if path.exists():
            data['cells'] = pd.read_parquet(path)
            print(f"Loaded {path}: {len(data['cells'])} cells")
            break

    # Load embeddings
    for pattern in ['fused_embeddings.npy', 'embeddings.npy']:
        path = data_dir / pattern
        if path.exists():
            data['embeddings'] = np.load(path)
            print(f"Loaded embeddings: {data['embeddings'].shape}")
            break

    # Load UMAP if available
    for pattern in ['umap_coords.npy', 'umap.npy']:
        path = data_dir / pattern
        if path.exists():
            data['umap'] = np.load(path)
            print(f"Loaded UMAP: {data['umap'].shape}")
            break

    return data


def plot_panel_a_stage_umap(data: dict, output_dir: Path):
    """A. Fused embeddings by disease stage."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel A: missing UMAP or cells")
        return

    umap = data['umap']
    cells = data['cells']

    fig, ax = plt.subplots(figsize=(8, 6))

    for stage in STAGE_ORDER:
        if 'stage' in cells.columns:
            mask = cells['stage'] == stage
        else:
            continue
        n = mask.sum()
        if n > 0:
            ax.scatter(
                umap[mask, 0], umap[mask, 1],
                c=STAGE_COLORS.get(stage, 'gray'),
                s=1, alpha=0.5, label=f'{stage} (n={n:,})',
                rasterized=True
            )

    ax.legend(markerscale=5, frameon=False, loc='upper right')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('A. Fused Dual-Reference Embeddings by Disease Stage')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'panel_A_stage_umap.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'panel_A_stage_umap.pdf', bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel A")


def plot_panel_b_density_contours(data: dict, output_dir: Path):
    """B. Stage density contours."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel B: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    fig, ax = plt.subplots(figsize=(6, 5))

    # Background scatter (light gray)
    ax.scatter(umap[:, 0], umap[:, 1], c='lightgray', s=0.5, alpha=0.3, rasterized=True)

    # Contours for each stage
    for stage in STAGE_ORDER:
        if 'stage' not in cells.columns:
            continue
        mask = cells['stage'] == stage
        if mask.sum() < 100:
            continue

        x, y = umap[mask, 0], umap[mask, 1]

        # KDE
        try:
            xmin, xmax = umap[:, 0].min(), umap[:, 0].max()
            ymin, ymax = umap[:, 1].min(), umap[:, 1].max()
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            kernel = stats.gaussian_kde(np.vstack([x, y]))
            z = np.reshape(kernel(positions).T, xx.shape)

            ax.contour(xx, yy, z, levels=3, colors=[STAGE_COLORS.get(stage, 'gray')],
                      alpha=0.8, linewidths=1.5)
        except Exception as e:
            print(f"    Could not compute contour for {stage}: {e}")

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('B. Stage Density Contours')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'panel_B_density_contours.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel B")


def plot_panel_c_cell_types(data: dict, output_dir: Path):
    """C. Cell type distribution."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel C: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    ct_col = None
    for col in ['cell_type', 'celltype', 'cell_type_fine', 'ann_level_2']:
        if col in cells.columns:
            ct_col = col
            break

    if ct_col is None:
        print("  Skipping panel C: no cell type column")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    cell_types = cells[ct_col].unique()
    for ct in cell_types:
        mask = cells[ct_col] == ct
        if mask.sum() > 0:
            color = CELL_TYPE_COLORS.get(ct, plt.cm.tab20(hash(ct) % 20))
            ax.scatter(
                umap[mask, 0], umap[mask, 1],
                c=[color], s=1, alpha=0.5, label=ct, rasterized=True
            )

    ax.legend(markerscale=5, frameon=False, loc='upper right', fontsize=8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('C. Cell Type Distribution')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'panel_C_cell_types.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel C")


def plot_panel_d_cell_cycle(data: dict, output_dir: Path):
    """D. Cell cycle phase."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel D: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    cc_col = None
    for col in ['cell_cycle_phase', 'phase', 'cell_cycle']:
        if col in cells.columns:
            cc_col = col
            break

    if cc_col is None:
        print("  Skipping panel D: no cell cycle column")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for phase in ['G1', 'S', 'G2M']:
        mask = cells[cc_col] == phase
        n = mask.sum()
        if n > 0:
            ax.scatter(
                umap[mask, 0], umap[mask, 1],
                c=CELL_CYCLE_COLORS.get(phase, 'gray'),
                s=1, alpha=0.5, label=f'{phase} (n={n:,})',
                rasterized=True
            )

    ax.legend(markerscale=5, frameon=False)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('D. Cell Cycle Phase')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'panel_D_cell_cycle.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel D")


def plot_panel_e_proliferation(data: dict, output_dir: Path):
    """E. Proliferation score."""
    if 'umap' not in data or 'cells' not in data:
        print("  Skipping panel E: missing data")
        return

    umap = data['umap']
    cells = data['cells']

    prolif_col = None
    for col in ['proliferation_score', 'entropic_score', 'S_score', 'G2M_score']:
        if col in cells.columns:
            prolif_col = col
            break

    if prolif_col is None:
        print("  Skipping panel E: no proliferation column")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    vals = cells[prolif_col].values
    vmin, vmax = np.nanpercentile(vals, [2, 98])

    scatter = ax.scatter(
        umap[:, 0], umap[:, 1],
        c=vals, cmap='viridis', s=1, alpha=0.7,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    plt.colorbar(scatter, ax=ax, label='Proliferation', shrink=0.6)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('E. Proliferation Score')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(output_dir / 'panel_E_proliferation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel E")


def plot_panel_f_stage_distribution(data: dict, output_dir: Path):
    """F. Stage distribution bar chart."""
    if 'cells' not in data:
        print("  Skipping panel F: missing cells")
        return

    cells = data['cells']
    if 'stage' not in cells.columns:
        print("  Skipping panel F: no stage column")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    counts = cells['stage'].value_counts()
    counts = counts.reindex(STAGE_ORDER).fillna(0)

    bars = ax.bar(range(len(STAGE_ORDER)), counts.values,
                  color=[STAGE_COLORS[s] for s in STAGE_ORDER])

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(count):,}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, rotation=45, ha='right')
    ax.set_ylabel('Cell Count')
    ax.set_title('F. Stage Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'panel_F_stage_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel F")


def plot_panel_g_celltype_by_stage(data: dict, output_dir: Path):
    """G. Cell type by stage heatmap."""
    if 'cells' not in data:
        print("  Skipping panel G: missing cells")
        return

    cells = data['cells']

    ct_col = None
    for col in ['cell_type', 'celltype', 'cell_type_fine']:
        if col in cells.columns:
            ct_col = col
            break

    if ct_col is None or 'stage' not in cells.columns:
        print("  Skipping panel G: missing columns")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Cross-tabulation normalized by stage
    ct_stage = pd.crosstab(cells[ct_col], cells['stage'], normalize='columns')
    ct_stage = ct_stage.reindex(columns=STAGE_ORDER, fill_value=0)

    im = ax.imshow(ct_stage.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_yticks(range(len(ct_stage.index)))
    ax.set_yticklabels(ct_stage.index, fontsize=8)
    ax.set_title('G. Cell Type by Stage')

    fig.savefig(output_dir / 'panel_G_celltype_stage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel G")


def plot_panel_h_mean_embedding(data: dict, output_dir: Path):
    """H. Mean fused embedding by stage."""
    if 'embeddings' not in data or 'cells' not in data:
        print("  Skipping panel H: missing data")
        return

    emb = data['embeddings']
    cells = data['cells']

    if 'stage' not in cells.columns:
        print("  Skipping panel H: no stage column")
        return

    fig, ax = plt.subplots(figsize=(8, 3))

    # Compute mean embedding per stage
    mean_emb = []
    for stage in STAGE_ORDER:
        mask = cells['stage'] == stage
        if mask.sum() > 0:
            mean_emb.append(emb[mask].mean(axis=0))
        else:
            mean_emb.append(np.zeros(emb.shape[1]))

    mean_emb = np.array(mean_emb)

    # Only show first 32 dims for clarity
    n_dims = min(32, mean_emb.shape[1])

    im = ax.imshow(mean_emb[:, :n_dims], cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER)
    ax.set_xlabel('Embedding Dimension')
    ax.set_title('H. Mean Fused Embedding')

    fig.savefig(output_dir / 'panel_H_mean_embedding.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel H")


def plot_panel_i_discriminative_dims(data: dict, output_dir: Path):
    """I. Stage-discriminative dimensions (F-ratio)."""
    if 'embeddings' not in data or 'cells' not in data:
        print("  Skipping panel I: missing data")
        return

    emb = data['embeddings']
    cells = data['cells']

    if 'stage' not in cells.columns:
        print("  Skipping panel I: no stage column")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Compute F-ratio (between/within variance) for each dimension
    f_ratios = []
    n_dims = min(32, emb.shape[1])

    for d in range(n_dims):
        groups = [emb[cells['stage'] == s, d] for s in STAGE_ORDER if (cells['stage'] == s).sum() > 0]
        if len(groups) >= 2:
            try:
                f_stat, _ = stats.f_oneway(*groups)
                f_ratios.append(f_stat if np.isfinite(f_stat) else 0)
            except:
                f_ratios.append(0)
        else:
            f_ratios.append(0)

    ax.barh(range(len(f_ratios)), f_ratios, color='steelblue')
    ax.set_yticks(range(len(f_ratios)))
    ax.set_yticklabels([f'D{i+1}' for i in range(len(f_ratios))], fontsize=7)
    ax.set_xlabel('F-ratio (Between/Within)')
    ax.set_title('I. Stage-Discriminative Dims')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    fig.savefig(output_dir / 'panel_I_discriminative_dims.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved panel I")


def main():
    parser = argparse.ArgumentParser(description='Generate embedding overview panels (Fig 1)')
    parser.add_argument('--data_dir', type=Path, required=True, help='Directory with embeddings and metadata')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory for panels')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(args.data_dir)

    print("\nGenerating panels...")
    plot_panel_a_stage_umap(data, args.output_dir)
    plot_panel_b_density_contours(data, args.output_dir)
    plot_panel_c_cell_types(data, args.output_dir)
    plot_panel_d_cell_cycle(data, args.output_dir)
    plot_panel_e_proliferation(data, args.output_dir)
    plot_panel_f_stage_distribution(data, args.output_dir)
    plot_panel_g_celltype_by_stage(data, args.output_dir)
    plot_panel_h_mean_embedding(data, args.output_dir)
    plot_panel_i_discriminative_dims(data, args.output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
