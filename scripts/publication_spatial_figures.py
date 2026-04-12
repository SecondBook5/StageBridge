#!/usr/bin/env python3
"""
Publication-quality spatial transcriptomics figures.

Creates sophisticated visualizations:
- Violin + boxplot + jitter overlay plots
- Spatial plots with cell type coloring and scale bars
- Statistical annotations

Uses a colorblind-friendly progression palette for disease stages.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from pathlib import Path
from scipy import stats
import scanpy as sc
from typing import Optional
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import warnings

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# COLOR PALETTES (Dark, rich colors as requested)
# =============================================================================

# Disease stage progression palette - DARK, RICH colors
STAGE_COLORS = {
    'Normal': '#1B4F72',   # Navy blue - healthy
    'AAH': '#2E86AB',      # Steel blue - early lesion
    'AIS': '#1D6F42',      # Forest green - in situ
    'MIA': '#D4A03C',      # Dark gold - minimally invasive
    'LUAD': '#922B21',     # Brick red - adenocarcinoma
}

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Alternative even darker palette option
STAGE_COLORS_DARK = {
    'Normal': '#0D3B66',   # Deep navy
    'AAH': '#1A5276',      # Royal blue
    'AIS': '#145A32',      # Dark forest
    'MIA': '#B7950B',      # Dark gold
    'LUAD': '#7B241C',     # Dark brick red
}

# Cell type palette (20 distinct colors, colorblind-optimized)
CELLTYPE_COLORS = {
    # Epithelial
    'AT1': '#E64B35',
    'AT2': '#4DBBD5',
    'Basal': '#00A087',
    'Ciliated': '#3C5488',
    'Secretory': '#F39B7F',
    'Club': '#8491B4',

    # Immune
    'T cell lineage': '#91D1C2',
    'B cell lineage': '#DC0000',
    'Macrophages': '#7E6148',
    'Monocytes': '#B09C85',
    'Mast cells': '#E18727',
    'NK cells': '#BC3C29',
    'Dendritic': '#0072B5',
    'Plasma': '#20854E',

    # Stromal
    'Fibroblast lineage': '#7876B1',
    'Endothelial': '#6F99AD',
    'Capillary': '#FFDC91',
    'Smooth muscle': '#EE4C97',
    'Pericytes': '#631879',

    # Other
    'Mesothelial': '#A6D854',
    'Other': '#999999',
}

def get_celltype_color(celltype: str) -> str:
    """Get color for cell type with fuzzy matching."""
    if celltype in CELLTYPE_COLORS:
        return CELLTYPE_COLORS[celltype]
    # Try partial match
    for key, color in CELLTYPE_COLORS.items():
        if key.lower() in celltype.lower() or celltype.lower() in key.lower():
            return color
    return '#999999'  # Default gray


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spatial_data(sample_dir: Path, backend: str = 'tangram') -> Optional[sc.AnnData]:
    """Load spatial data with cell type proportions."""
    # Try different naming conventions
    possible_names = [
        f"{backend}_spatial_annotated.h5ad",  # tangram, destvi
        f"{backend}_annotated_spatial.h5ad",  # tacco uses this
        "spatial_annotated.h5ad",
        "annotated_spatial.h5ad",
    ]

    for name in possible_names:
        h5ad_file = sample_dir / name
        if h5ad_file.exists():
            return sc.read_h5ad(h5ad_file)

    # If no h5ad, try to construct from parquet + original spatial data
    props_file = sample_dir / 'cell_type_proportions.parquet'
    if props_file.exists():
        # This backend only saved parquet, need spatial coords from elsewhere
        # For now, return None - would need original spatial h5ad
        pass

    return None


def extract_stage(sample_name: str) -> str:
    """Extract disease stage from sample name."""
    parts = sample_name.split("_")
    if len(parts) >= 3:
        stage = "_".join(parts[2:])
        stage = stage.replace("-1", "").replace("-2", "")
        return stage
    return "Unknown"


def collect_all_data(base_dir: Path, backend: str = 'tangram',
                     label_source: str = 'hlca') -> pd.DataFrame:
    """Collect cell type proportions from all samples."""
    samples_dir = base_dir / label_source / backend / 'samples'

    all_data = []
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        # Load proportions
        props_file = sample_dir / 'cell_type_proportions.parquet'
        if props_file.exists():
            df = pd.read_parquet(props_file)
            df['sample'] = sample_dir.name
            df['stage'] = extract_stage(sample_dir.name)
            df['patient'] = sample_dir.name.split('_')[1] if '_' in sample_dir.name else 'Unknown'
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# VIOLIN + BOXPLOT + JITTER PLOTS
# =============================================================================

def violin_box_jitter(data: pd.DataFrame, x: str, y: str, ax: plt.Axes,
                      palette: dict = None, order: list = None,
                      show_stats: bool = True, jitter_alpha: float = 0.3,
                      jitter_size: float = 2):
    """
    Create a sophisticated violin + boxplot + jitter overlay plot.

    Similar to Nature Methods / Cell style figures.
    """
    if palette is None:
        palette = STAGE_COLORS
    if order is None:
        order = [o for o in STAGE_ORDER if o in data[x].unique()]

    # Filter to valid stages
    data = data[data[x].isin(order)]

    # Violin plot (half violins)
    violin_parts = ax.violinplot(
        [data[data[x] == stage][y].dropna().values for stage in order],
        positions=range(len(order)),
        showmeans=False, showmedians=False, showextrema=False
    )

    # Color the violins
    for i, (pc, stage) in enumerate(zip(violin_parts['bodies'], order)):
        pc.set_facecolor(palette.get(stage, '#999999'))
        pc.set_edgecolor('none')
        pc.set_alpha(0.3)

    # Boxplot overlay
    bp = ax.boxplot(
        [data[data[x] == stage][y].dropna().values for stage in order],
        positions=range(len(order)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
    )

    # Style boxplot
    for i, (box, stage) in enumerate(zip(bp['boxes'], order)):
        box.set_facecolor('white')
        box.set_edgecolor(palette.get(stage, '#999999'))
        box.set_linewidth(1.5)
    for element in ['whiskers', 'caps']:
        for i, item in enumerate(bp[element]):
            stage = order[i // 2]
            item.set_color(palette.get(stage, '#999999'))
            item.set_linewidth(1.5)
    for i, median in enumerate(bp['medians']):
        median.set_color(palette.get(order[i], '#999999'))
        median.set_linewidth(2)

    # Jitter overlay
    for i, stage in enumerate(order):
        stage_data = data[data[x] == stage][y].dropna().values
        n_points = len(stage_data)
        if n_points > 0:
            # Add jitter
            jitter = np.random.normal(0, 0.08, n_points)
            ax.scatter(
                np.full(n_points, i) + jitter,
                stage_data,
                c=palette.get(stage, '#999999'),
                s=jitter_size,
                alpha=jitter_alpha,
                edgecolors='none',
                rasterized=True,  # For smaller file size
            )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)

    # Add sample counts
    counts = [len(data[data[x] == stage]) for stage in order]
    for i, (stage, count) in enumerate(zip(order, counts)):
        ax.annotate(
            f'n={count}',
            xy=(i, ax.get_ylim()[0]),
            ha='center', va='top',
            fontsize=7, color='gray',
            xytext=(0, -5), textcoords='offset points'
        )

    # Statistical test (Kruskal-Wallis)
    if show_stats and len(order) > 1:
        groups = [data[data[x] == stage][y].dropna().values for stage in order]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            stat, pval = stats.kruskal(*groups)
            sig_text = format_pvalue(pval)
            ax.annotate(
                f'Kruskal-Wallis {sig_text}',
                xy=(0.5, 1.02), xycoords='axes fraction',
                ha='center', fontsize=8, style='italic'
            )


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.0001:
        return 'p < 0.0001'
    elif p < 0.001:
        return f'p = {p:.4f}'
    elif p < 0.01:
        return f'p = {p:.3f}'
    elif p < 0.05:
        return f'p = {p:.2f}'
    else:
        return f'p = {p:.2f} (ns)'


# =============================================================================
# SPATIAL PLOTS WITH SCALE BAR
# =============================================================================

def add_scale_bar(ax: plt.Axes, coords: np.ndarray, bar_length_um: float = 500,
                  pixels_per_um: float = 1.0, loc: str = 'lower right',
                  fontsize: int = 8):
    """
    Add a scale bar to spatial plot.

    Parameters
    ----------
    coords : array
        Spatial coordinates (Nx2)
    bar_length_um : float
        Scale bar length in micrometers
    pixels_per_um : float
        Conversion factor (default assumes coords are in um)
    loc : str
        Scale bar location
    """
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    # Estimate pixel size (assuming Visium ~100um spot spacing)
    # If coords are in pixels, typical Visium is ~0.5 um/pixel
    bar_length = bar_length_um * pixels_per_um

    # Position based on loc
    if 'right' in loc:
        x_start = coords[:, 0].max() - bar_length - x_range * 0.05
    else:
        x_start = coords[:, 0].min() + x_range * 0.05

    if 'lower' in loc:
        y_pos = coords[:, 1].min() + y_range * 0.05
    else:
        y_pos = coords[:, 1].max() - y_range * 0.05

    # Draw scale bar
    ax.plot([x_start, x_start + bar_length], [y_pos, y_pos],
            color='black', linewidth=3, solid_capstyle='butt')
    ax.text(x_start + bar_length/2, y_pos, f'{int(bar_length_um)} um',
            ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


def spatial_celltype_plot(adata: sc.AnnData, ax: plt.Axes,
                          celltype_col_prefix: str = 'tangram_',
                          spot_size: float = 10, alpha: float = 0.8,
                          show_scalebar: bool = True):
    """
    Create a spatial plot colored by dominant cell type.

    Colors each spot by its highest-proportion cell type.
    """
    coords = adata.obsm['spatial']

    # Find cell type columns
    ct_cols = [c for c in adata.obs.columns if c.startswith(celltype_col_prefix)]
    if not ct_cols:
        raise ValueError(f"No cell type columns found with prefix '{celltype_col_prefix}'")

    # Get dominant cell type for each spot
    ct_data = adata.obs[ct_cols].values
    dominant_idx = np.argmax(ct_data, axis=1)
    dominant_ct = [ct_cols[i].replace(celltype_col_prefix, '') for i in dominant_idx]

    # Get colors
    colors = [get_celltype_color(ct) for ct in dominant_ct]

    # Plot
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=spot_size, alpha=alpha,
        edgecolors='none', rasterized=True
    )

    ax.set_aspect('equal')
    ax.axis('off')

    # Scale bar
    if show_scalebar:
        add_scale_bar(ax, coords)

    # Legend
    unique_cts = list(set(dominant_ct))
    legend_elements = [
        mpatches.Patch(facecolor=get_celltype_color(ct), label=ct)
        for ct in sorted(unique_cts)
    ]
    ax.legend(
        handles=legend_elements, loc='upper left',
        fontsize=6, frameon=False, ncol=2,
        bbox_to_anchor=(1.02, 1)
    )

    return dominant_ct


def spatial_proportion_plot(adata: sc.AnnData, ax: plt.Axes,
                           celltype: str, celltype_col_prefix: str = 'tangram_',
                           spot_size: float = 10, cmap: str = 'viridis',
                           show_scalebar: bool = True, vmin: float = 0,
                           vmax: float = None):
    """
    Create spatial plot showing proportion of a specific cell type.
    """
    coords = adata.obsm['spatial']
    col_name = f"{celltype_col_prefix}{celltype}"

    if col_name not in adata.obs.columns:
        # Try without prefix
        if celltype in adata.obs.columns:
            col_name = celltype
        else:
            raise ValueError(f"Cell type column not found: {col_name}")

    values = adata.obs[col_name].values

    if vmax is None:
        vmax = np.percentile(values, 99)

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, s=spot_size, alpha=0.8,
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='none', rasterized=True
    )

    ax.set_aspect('equal')
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{celltype} proportion', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if show_scalebar:
        add_scale_bar(ax, coords)

    return scatter


# =============================================================================
# MAIN FIGURE GENERATION
# =============================================================================

def figure_1_celltype_composition(data: pd.DataFrame, output_dir: Path):
    """
    Figure 1: Cell type composition across disease stages.

    Multi-panel violin + boxplot + jitter for key cell types.
    """
    # Key cell types for LUAD progression
    key_celltypes = ['AT2', 'Macrophages', 'Fibroblast lineage', 'T cell lineage']
    available = [ct for ct in key_celltypes if ct in data.columns]

    if len(available) < 2:
        print(f"Warning: Only {len(available)} key cell types found")
        available = [c for c in data.columns if c not in ['sample', 'stage', 'patient']][:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, ct in zip(axes, available):
        violin_box_jitter(data, x='stage', y=ct, ax=ax)
        ax.set_ylabel(f'{ct}\nproportion')
        ax.set_xlabel('')
        ax.set_title(ct, fontweight='bold')

    # Hide unused axes
    for ax in axes[len(available):]:
        ax.set_visible(False)

    plt.suptitle('Cell Type Composition Across LUAD Progression Stages',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig1_celltype_composition.png', dpi=300)
    fig.savefig(output_dir / 'fig1_celltype_composition.pdf')
    plt.close(fig)
    print(f"Saved fig1_celltype_composition.png/pdf")


def figure_2_spatial_examples(base_dir: Path, output_dir: Path,
                              backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 2: Representative spatial plots for each disease stage.

    Shows one sample per stage with cell type coloring.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find one representative sample per stage
    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    n_stages = len([s for s in STAGE_ORDER if s in stage_samples])
    if n_stages == 0:
        print("No samples found!")
        return

    # Create figure
    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    for ax, stage in zip(axes, STAGE_ORDER):
        if stage not in stage_samples:
            ax.set_visible(False)
            continue

        sample_dir = stage_samples[stage]
        adata = load_spatial_data(sample_dir, backend)

        if adata is None:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        spatial_celltype_plot(adata, ax, celltype_col_prefix=f'{backend}_')
        ax.set_title(f'{stage}\n({sample_dir.name.split("_")[1]})',
                    fontweight='bold', color=STAGE_COLORS.get(stage, 'black'))

    plt.suptitle('Spatial Cell Type Distribution by Disease Stage',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig2_spatial_stages.png', dpi=300)
    fig.savefig(output_dir / 'fig2_spatial_stages.pdf')
    plt.close(fig)
    print(f"Saved fig2_spatial_stages.png/pdf")


def figure_3_at2_spatial(base_dir: Path, output_dir: Path,
                         backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 3: AT2 proportion across disease stages (key for LUAD).

    Spatial heatmaps showing AT2 cell proportion.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find representative samples
    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    n_stages = len([s for s in STAGE_ORDER if s in stage_samples])
    if n_stages == 0:
        print("No samples found!")
        return

    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    # Find global vmax for consistent coloring
    all_at2 = []
    for stage in STAGE_ORDER:
        if stage in stage_samples:
            adata = load_spatial_data(stage_samples[stage], backend)
            if adata is not None and f'{backend}_AT2' in adata.obs.columns:
                all_at2.extend(adata.obs[f'{backend}_AT2'].values)

    vmax = np.percentile(all_at2, 99) if all_at2 else 1.0

    for ax, stage in zip(axes, STAGE_ORDER):
        if stage not in stage_samples:
            ax.set_visible(False)
            continue

        adata = load_spatial_data(stage_samples[stage], backend)
        if adata is None or f'{backend}_AT2' not in adata.obs.columns:
            ax.text(0.5, 0.5, 'AT2 data not found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        spatial_proportion_plot(adata, ax, 'AT2',
                               celltype_col_prefix=f'{backend}_',
                               cmap='YlOrRd', vmax=vmax)
        ax.set_title(f'{stage}', fontweight='bold',
                    color=STAGE_COLORS.get(stage, 'black'))

    plt.suptitle('AT2 Cell Proportion Across Disease Stages',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig3_at2_spatial.png', dpi=300)
    fig.savefig(output_dir / 'fig3_at2_spatial.pdf')
    plt.close(fig)
    print(f"Saved fig3_at2_spatial.png/pdf")


def figure_4_macrophage_infiltration(data: pd.DataFrame, base_dir: Path,
                                     output_dir: Path, backend: str = 'tangram',
                                     label_source: str = 'hlca'):
    """
    Figure 4: Macrophage infiltration analysis.

    Combined violin plot + spatial examples.
    """
    fig = plt.figure(figsize=(14, 5))

    # Left panel: Violin plot
    ax1 = fig.add_subplot(1, 3, 1)
    if 'Macrophages' in data.columns:
        violin_box_jitter(data, x='stage', y='Macrophages', ax=ax1)
        ax1.set_ylabel('Macrophage proportion')
        ax1.set_xlabel('')
        ax1.set_title('Macrophage Infiltration', fontweight='bold')

    # Middle and right panels: Spatial examples (Normal vs LUAD)
    samples_dir = base_dir / label_source / backend / 'samples'
    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in ['Normal', 'LUAD'] and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    for i, (stage, panel_idx) in enumerate([('Normal', 2), ('LUAD', 3)]):
        ax = fig.add_subplot(1, 3, panel_idx)
        if stage not in stage_samples:
            ax.text(0.5, 0.5, f'{stage} not found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        adata = load_spatial_data(stage_samples[stage], backend)
        if adata is None or f'{backend}_Macrophages' not in adata.obs.columns:
            ax.text(0.5, 0.5, 'Macrophage data not found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        spatial_proportion_plot(adata, ax, 'Macrophages',
                               celltype_col_prefix=f'{backend}_',
                               cmap='Purples')
        ax.set_title(f'{stage}', fontweight='bold',
                    color=STAGE_COLORS.get(stage, 'black'))

    plt.suptitle('Macrophage Infiltration Across Disease Stages',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig4_macrophage_infiltration.png', dpi=300)
    fig.savefig(output_dir / 'fig4_macrophage_infiltration.pdf')
    plt.close(fig)
    print(f"Saved fig4_macrophage_infiltration.png/pdf")


def figure_5_celltype_summary_heatmap(data: pd.DataFrame, output_dir: Path):
    """
    Figure 5: Summary heatmap of cell type proportions by stage.
    """
    # Get cell type columns
    ct_cols = [c for c in data.columns if c not in ['sample', 'stage', 'patient']]

    # Calculate mean proportion per stage (aggregate by sample first to avoid spot bias)
    sample_means = data.groupby(['sample', 'stage'])[ct_cols].mean().reset_index()
    stage_means = sample_means.groupby('stage')[ct_cols].mean()

    # Reorder stages
    stage_order = [s for s in STAGE_ORDER if s in stage_means.index]
    stage_means = stage_means.loc[stage_order]

    # Z-score normalize across stages for better visualization
    stage_zscore = (stage_means - stage_means.mean()) / (stage_means.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Heatmap
    im = ax.imshow(stage_zscore.T.values, aspect='auto', cmap='RdBu_r',
                   vmin=-2, vmax=2)

    # Labels
    ax.set_xticks(range(len(stage_order)))
    ax.set_xticklabels(stage_order, fontweight='bold')
    ax.set_yticks(range(len(ct_cols)))
    ax.set_yticklabels(ct_cols)

    # Color x-tick labels by stage
    for i, stage in enumerate(stage_order):
        ax.get_xticklabels()[i].set_color(STAGE_COLORS.get(stage, 'black'))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Z-score', fontsize=10)

    # Add values - use scientific notation for small numbers
    for i in range(len(ct_cols)):
        for j in range(len(stage_order)):
            val = stage_means.iloc[j, i]
            zscore = stage_zscore.iloc[j, i]
            # Format based on magnitude
            if val < 0.001:
                val_str = f'{val:.1e}'
            else:
                val_str = f'{val:.3f}'
            text = ax.text(j, i, val_str, ha='center', va='center',
                          fontsize=7, color='white' if abs(zscore) > 1 else 'black')

    ax.set_title('Cell Type Proportions Across Disease Stages\n(values = mean proportion, color = z-score)',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig5_celltype_heatmap.png', dpi=300)
    fig.savefig(output_dir / 'fig5_celltype_heatmap.pdf')
    plt.close(fig)
    print(f"Saved fig5_celltype_heatmap.png/pdf")


def figure_6_backend_comparison(base_dir: Path, output_dir: Path):
    """
    Figure 6: Comparison of deconvolution backends for one sample.
    """
    # Find a sample with multiple backends
    backends = ['tangram', 'destvi', 'tacco', 'cell2location']
    label_source = 'hlca'

    # Find sample available in most backends
    sample_availability = defaultdict(set)
    for backend in backends:
        samples_dir = base_dir / label_source / backend / 'samples'
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if sample_dir.is_dir():
                    sample_availability[sample_dir.name].add(backend)

    # Pick sample with most backends
    best_sample = max(sample_availability.keys(),
                      key=lambda x: len(sample_availability[x]),
                      default=None)

    if not best_sample:
        print("No samples found for backend comparison")
        return

    available_backends = sample_availability[best_sample]
    n_backends = len(available_backends)

    fig, axes = plt.subplots(1, n_backends, figsize=(4 * n_backends, 4))
    if n_backends == 1:
        axes = [axes]

    stage = extract_stage(best_sample)

    for ax, backend in zip(axes, sorted(available_backends)):
        sample_dir = base_dir / label_source / backend / 'samples' / best_sample
        adata = load_spatial_data(sample_dir, backend)

        if adata is None:
            ax.text(0.5, 0.5, f'{backend} data not found', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue

        spatial_celltype_plot(adata, ax, celltype_col_prefix=f'{backend}_')
        ax.set_title(f'{backend.upper()}', fontweight='bold')

    plt.suptitle(f'Deconvolution Backend Comparison\n{best_sample} ({stage})',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig6_backend_comparison.png', dpi=300)
    fig.savefig(output_dir / 'fig6_backend_comparison.pdf')
    plt.close(fig)
    print(f"Saved fig6_backend_comparison.png/pdf")


def figure_7_multipanel_spatial(base_dir: Path, output_dir: Path,
                                backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 7: Multi-panel spatial cell type proportions (Cell2location style).

    Shows multiple cell types side-by-side for a single sample.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find a LUAD sample with good data
    target_sample = None
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage == 'LUAD':
            target_sample = sample_dir
            break

    if target_sample is None:
        print("No LUAD sample found!")
        return

    adata = load_spatial_data(target_sample, backend)
    if adata is None:
        print(f"Could not load data from {target_sample}")
        return

    # Key cell types relevant for LUAD
    celltypes = ['AT2', 'Macrophages', 'Fibroblast lineage', 'T cell lineage',
                 'Capillary', 'Basal', 'Secretory', 'Ciliated']

    # Filter to available cell types
    available = [ct for ct in celltypes if f'{backend}_{ct}' in adata.obs.columns]
    n_types = len(available)

    if n_types == 0:
        print("No cell type columns found!")
        return

    # Calculate grid
    ncols = 4
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    coords = adata.obsm['spatial']

    for i, ct in enumerate(available):
        ax = axes[i]
        col_name = f'{backend}_{ct}'
        values = adata.obs[col_name].values

        # Use magma colormap like Cell2location tutorial
        vmax = np.percentile(values, 99.2)
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=values, s=8, alpha=0.8,
            cmap='magma', vmin=0, vmax=vmax,
            edgecolors='none', rasterized=True
        )

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(ct, fontsize=11, fontweight='bold')

        # Small colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax in axes[len(available):]:
        ax.set_visible(False)

    stage = extract_stage(target_sample.name)
    plt.suptitle(f'Cell Type Proportions - {target_sample.name}\n({stage})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig7_multipanel_spatial.png', dpi=300,
                facecolor='white', bbox_inches='tight')
    fig.savefig(output_dir / 'fig7_multipanel_spatial.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig7_multipanel_spatial.png/pdf")


def figure_8_stage_comparison_grid(base_dir: Path, output_dir: Path,
                                   backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 8: Cell type × Stage comparison grid.

    Shows 3 key cell types across all 5 stages.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find one representative sample per stage
    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    # Key cell types
    celltypes = ['AT2', 'Macrophages', 'T cell lineage']
    stages_available = [s for s in STAGE_ORDER if s in stage_samples]

    n_cts = len(celltypes)
    n_stages = len(stages_available)

    if n_stages == 0:
        print("No samples found!")
        return

    fig, axes = plt.subplots(n_cts, n_stages, figsize=(3 * n_stages, 3 * n_cts))

    # Compute global vmax for each cell type
    vmax_dict = {}
    for ct in celltypes:
        all_vals = []
        for stage in stages_available:
            adata = load_spatial_data(stage_samples[stage], backend)
            if adata is not None and f'{backend}_{ct}' in adata.obs.columns:
                all_vals.extend(adata.obs[f'{backend}_{ct}'].values)
        vmax_dict[ct] = np.percentile(all_vals, 99) if all_vals else 1.0

    for i, ct in enumerate(celltypes):
        for j, stage in enumerate(stages_available):
            ax = axes[i, j] if n_cts > 1 else axes[j]

            adata = load_spatial_data(stage_samples[stage], backend)
            col_name = f'{backend}_{ct}'

            if adata is None or col_name not in adata.obs.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                continue

            coords = adata.obsm['spatial']
            values = adata.obs[col_name].values

            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=values, s=5, alpha=0.7,
                cmap='Reds', vmin=0, vmax=vmax_dict[ct],
                edgecolors='none', rasterized=True
            )

            ax.set_aspect('equal')
            ax.axis('off')

            # Row labels (cell types)
            if j == 0:
                ax.set_ylabel(ct, fontsize=11, fontweight='bold', rotation=90,
                             labelpad=10)
                ax.yaxis.set_label_coords(-0.1, 0.5)

            # Column labels (stages)
            if i == 0:
                ax.set_title(stage, fontsize=11, fontweight='bold',
                            color=STAGE_COLORS.get(stage, 'black'))

    # Add single colorbar
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Proportion (normalized)', fontsize=10)

    plt.suptitle('Cell Type Distribution Across Disease Stages',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig8_stage_comparison_grid.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / 'fig8_stage_comparison_grid.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig8_stage_comparison_grid.png/pdf")


def figure_9_violin_detailed(data: pd.DataFrame, output_dir: Path):
    """
    Figure 9: Detailed violin plots with significance annotations.

    More detailed version with pairwise comparisons.
    """
    # All available cell types
    ct_cols = [c for c in data.columns if c not in ['sample', 'stage', 'patient']]

    # Take top cell types by variance across stages
    sample_means = data.groupby(['sample', 'stage'])[ct_cols].mean()
    stage_means = sample_means.groupby('stage').mean()
    ct_variance = stage_means.var()
    top_cts = ct_variance.nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, ct in zip(axes, top_cts):
        # Aggregate by sample first
        sample_data = data.groupby(['sample', 'stage'])[ct].mean().reset_index()

        violin_box_jitter(sample_data, x='stage', y=ct, ax=ax, jitter_alpha=0.5, jitter_size=15)
        ax.set_ylabel(f'{ct}\n(mean per sample)')
        ax.set_xlabel('')
        ax.set_title(ct, fontweight='bold', fontsize=12)

        # Format y-axis for small values
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    plt.suptitle('Cell Type Proportions by Disease Stage\n(Sample-level means)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig9_violin_detailed.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig9_violin_detailed.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig9_violin_detailed.png/pdf")


# =============================================================================
# SPATIAL STATISTICS FUNCTIONS
# =============================================================================

def compute_spatial_weights(coords: np.ndarray, k: int = 8,
                           bandwidth: float = None) -> csr_matrix:
    """
    Compute spatial weights matrix using k-nearest neighbors.

    Parameters
    ----------
    coords : array (N, 2)
        Spatial coordinates
    k : int
        Number of nearest neighbors
    bandwidth : float
        Optional distance bandwidth (uses k-NN if None)

    Returns
    -------
    W : sparse matrix
        Row-normalized spatial weights
    """
    n = len(coords)
    distances = cdist(coords, coords)

    # K-nearest neighbors
    W = np.zeros((n, n))
    for i in range(n):
        # Get k nearest (excluding self)
        idx = np.argsort(distances[i])[1:k+1]
        W[i, idx] = 1.0

    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums

    return csr_matrix(W)


def local_morans_i(values: np.ndarray, W: csr_matrix) -> tuple:
    """
    Compute Local Moran's I statistic.

    Parameters
    ----------
    values : array
        Variable of interest
    W : sparse matrix
        Spatial weights matrix

    Returns
    -------
    I_local : array
        Local Moran's I for each observation
    z_scores : array
        Z-scores for significance
    quadrants : array
        HH, HL, LH, LL classification
    """
    n = len(values)

    # Standardize values
    z = (values - values.mean()) / (values.std() + 1e-10)

    # Spatial lag
    Wz = W.dot(z)

    # Local Moran's I
    I_local = z * Wz

    # Determine quadrants
    quadrants = np.empty(n, dtype='U2')
    quadrants[(z > 0) & (Wz > 0)] = 'HH'  # High-High (hot spot)
    quadrants[(z > 0) & (Wz < 0)] = 'HL'  # High-Low (spatial outlier)
    quadrants[(z < 0) & (Wz > 0)] = 'LH'  # Low-High (spatial outlier)
    quadrants[(z < 0) & (Wz < 0)] = 'LL'  # Low-Low (cold spot)

    # Approximate z-scores (simplified)
    # Full inference would require permutation testing
    E_I = -1 / (n - 1)
    z_scores = (I_local - E_I) / (np.std(I_local) + 1e-10)

    return I_local, z_scores, quadrants, z, Wz


def getis_ord_gi_star(values: np.ndarray, W: csr_matrix) -> tuple:
    """
    Compute Getis-Ord Gi* statistic for hotspot analysis.

    Parameters
    ----------
    values : array
        Variable of interest
    W : sparse matrix
        Spatial weights matrix (should include self-weights for Gi*)

    Returns
    -------
    Gi_star : array
        Gi* statistic for each location
    z_scores : array
        Z-scores for significance
    """
    n = len(values)
    x_bar = values.mean()
    s = values.std()

    # Add self-weights for Gi*
    W_star = W.toarray()
    np.fill_diagonal(W_star, 1)

    # Compute Gi*
    Gi_star = np.zeros(n)
    for i in range(n):
        wi = W_star[i]
        sum_wij_xj = np.sum(wi * values)
        sum_wij = np.sum(wi)
        sum_wij2 = np.sum(wi ** 2)

        numerator = sum_wij_xj - x_bar * sum_wij
        denominator = s * np.sqrt((n * sum_wij2 - sum_wij**2) / (n - 1) + 1e-10)

        Gi_star[i] = numerator / (denominator + 1e-10)

    return Gi_star, Gi_star  # Z-scores are the Gi* values themselves


def compute_correlogram(values: np.ndarray, coords: np.ndarray,
                       n_bins: int = 15) -> tuple:
    """
    Compute spatial correlogram (Moran's I at different distance lags).

    Returns
    -------
    distances : array
        Distance bin centers
    morans_i : array
        Moran's I at each distance lag
    """
    distances_matrix = cdist(coords, coords)
    max_dist = np.percentile(distances_matrix[distances_matrix > 0], 50)
    bins = np.linspace(0, max_dist, n_bins + 1)

    z = (values - values.mean()) / (values.std() + 1e-10)

    morans_i = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (distances_matrix > bins[i]) & (distances_matrix <= bins[i+1])

        if mask.sum() > 0:
            # Create weight matrix for this distance band
            W = mask.astype(float)
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums

            # Global Moran's I for this lag
            Wz = W.dot(z)
            I = np.mean(z * Wz)
            morans_i.append(I)
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    return np.array(bin_centers), np.array(morans_i)


# =============================================================================
# SPATIAL STATISTICS FIGURES
# =============================================================================

def figure_10_local_morans(base_dir: Path, output_dir: Path,
                           backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 10: Local Moran's I analysis for spatial clustering.

    Shows hotspots (HH), coldspots (LL), and spatial outliers (HL, LH).
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find one sample per stage
    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    stages_available = [s for s in STAGE_ORDER if s in stage_samples]
    n_stages = len(stages_available)

    if n_stages == 0:
        print("No samples found!")
        return

    # Cell type to analyze
    celltype = 'AT2'

    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 8))

    # Quadrant colors
    quad_colors = {
        'HH': '#922B21',  # Brick red - hot spots
        'HL': '#F39C12',  # Orange - high outliers
        'LH': '#3498DB',  # Blue - low outliers
        'LL': '#1B4F72',  # Navy - cold spots
    }

    for j, stage in enumerate(stages_available):
        adata = load_spatial_data(stage_samples[stage], backend)
        col_name = f'{backend}_{celltype}'

        if adata is None or col_name not in adata.obs.columns:
            for i in range(2):
                axes[i, j].text(0.5, 0.5, 'N/A', ha='center', va='center',
                               transform=axes[i, j].transAxes)
                axes[i, j].axis('off')
            continue

        coords = adata.obsm['spatial']
        values = adata.obs[col_name].values

        # Subsample for computational efficiency
        if len(values) > 5000:
            idx = np.random.choice(len(values), 5000, replace=False)
            coords_sub = coords[idx]
            values_sub = values[idx]
        else:
            coords_sub = coords
            values_sub = values

        # Compute spatial weights and Local Moran's I
        W = compute_spatial_weights(coords_sub, k=8)
        I_local, z_scores, quadrants, z, Wz = local_morans_i(values_sub, W)

        # Top row: Local Moran's I values
        ax1 = axes[0, j]
        scatter1 = ax1.scatter(
            coords_sub[:, 0], coords_sub[:, 1],
            c=I_local, s=5, alpha=0.7,
            cmap='RdBu_r', vmin=-2, vmax=2,
            edgecolors='none', rasterized=True
        )
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'{stage}\nLocal Moran\'s I',
                     fontweight='bold', color=STAGE_COLORS.get(stage, 'black'))
        if j == n_stages - 1:
            cbar = plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Local I', fontsize=8)

        # Bottom row: LISA clusters (quadrants)
        ax2 = axes[1, j]
        for quad in ['HH', 'HL', 'LH', 'LL']:
            mask = quadrants == quad
            if mask.sum() > 0:
                ax2.scatter(
                    coords_sub[mask, 0], coords_sub[mask, 1],
                    c=quad_colors[quad], s=5, alpha=0.7,
                    label=quad, edgecolors='none', rasterized=True
                )
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'LISA Clusters', fontsize=10)

        if j == 0:
            ax2.legend(loc='upper left', fontsize=7, frameon=False,
                      title='Cluster', title_fontsize=8)

    plt.suptitle(f'Local Moran\'s I Analysis - {celltype} Proportion\n'
                 f'(HH=Hot Spot, LL=Cold Spot, HL/LH=Spatial Outliers)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig10_local_morans.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig10_local_morans.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig10_local_morans.png/pdf")


def figure_11_moran_scatter(base_dir: Path, output_dir: Path,
                            backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 11: Moran scatter plots.

    Shows standardized value vs. spatial lag for each stage.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    stages_available = [s for s in STAGE_ORDER if s in stage_samples]
    n_stages = len(stages_available)

    if n_stages == 0:
        print("No samples found!")
        return

    celltype = 'AT2'

    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    if n_stages == 1:
        axes = [axes]

    for j, stage in enumerate(stages_available):
        ax = axes[j]
        adata = load_spatial_data(stage_samples[stage], backend)
        col_name = f'{backend}_{celltype}'

        if adata is None or col_name not in adata.obs.columns:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')
            continue

        coords = adata.obsm['spatial']
        values = adata.obs[col_name].values

        # Subsample
        if len(values) > 3000:
            idx = np.random.choice(len(values), 3000, replace=False)
            coords_sub = coords[idx]
            values_sub = values[idx]
        else:
            coords_sub = coords
            values_sub = values

        # Compute
        W = compute_spatial_weights(coords_sub, k=8)
        I_local, z_scores, quadrants, z, Wz = local_morans_i(values_sub, W)

        # Quadrant colors
        colors = [STAGE_COLORS.get(stage, '#666666')] * len(z)

        # Scatter plot
        ax.scatter(z, Wz, c=colors, s=3, alpha=0.3, edgecolors='none', rasterized=True)

        # Add quadrant lines
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

        # Regression line
        slope, intercept = np.polyfit(z, Wz, 1)
        x_line = np.array([z.min(), z.max()])
        ax.plot(x_line, slope * x_line + intercept, color='#922B21',
               linewidth=2, label=f'Moran\'s I = {slope:.3f}')

        ax.set_xlabel('Standardized Value (z)')
        ax.set_ylabel('Spatial Lag (Wz)')
        ax.set_title(stage, fontweight='bold', color=STAGE_COLORS.get(stage, 'black'))
        ax.legend(loc='upper left', fontsize=8, frameon=False)

        # Quadrant labels
        ax.text(0.95, 0.95, 'HH', transform=ax.transAxes, fontsize=10,
               fontweight='bold', ha='right', va='top', color='#922B21')
        ax.text(0.05, 0.95, 'LH', transform=ax.transAxes, fontsize=10,
               fontweight='bold', ha='left', va='top', color='#3498DB')
        ax.text(0.05, 0.05, 'LL', transform=ax.transAxes, fontsize=10,
               fontweight='bold', ha='left', va='bottom', color='#1B4F72')
        ax.text(0.95, 0.05, 'HL', transform=ax.transAxes, fontsize=10,
               fontweight='bold', ha='right', va='bottom', color='#F39C12')

    plt.suptitle(f'Moran Scatter Plots - {celltype} Proportion\n'
                 f'(Slope = Global Moran\'s I)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig11_moran_scatter.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig11_moran_scatter.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig11_moran_scatter.png/pdf")


def figure_12_getis_ord(base_dir: Path, output_dir: Path,
                        backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 12: Getis-Ord Gi* hotspot analysis.

    Shows statistically significant hot and cold spots.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    stages_available = [s for s in STAGE_ORDER if s in stage_samples]
    n_stages = len(stages_available)

    if n_stages == 0:
        print("No samples found!")
        return

    celltypes = ['AT2', 'Macrophages']
    n_cts = len(celltypes)

    fig, axes = plt.subplots(n_cts, n_stages, figsize=(4 * n_stages, 4 * n_cts))

    for i, ct in enumerate(celltypes):
        for j, stage in enumerate(stages_available):
            ax = axes[i, j] if n_cts > 1 else axes[j]
            adata = load_spatial_data(stage_samples[stage], backend)
            col_name = f'{backend}_{ct}'

            if adata is None or col_name not in adata.obs.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes)
                ax.axis('off')
                continue

            coords = adata.obsm['spatial']
            values = adata.obs[col_name].values

            # Subsample
            if len(values) > 5000:
                idx = np.random.choice(len(values), 5000, replace=False)
                coords_sub = coords[idx]
                values_sub = values[idx]
            else:
                coords_sub = coords
                values_sub = values

            # Compute Gi*
            W = compute_spatial_weights(coords_sub, k=8)
            Gi_star, z_scores = getis_ord_gi_star(values_sub, W)

            # Plot with diverging colormap
            scatter = ax.scatter(
                coords_sub[:, 0], coords_sub[:, 1],
                c=Gi_star, s=5, alpha=0.7,
                cmap='RdBu_r', vmin=-3, vmax=3,
                edgecolors='none', rasterized=True
            )

            ax.set_aspect('equal')
            ax.axis('off')

            # Row labels
            if j == 0:
                ax.set_ylabel(ct, fontsize=11, fontweight='bold')

            # Column labels
            if i == 0:
                ax.set_title(stage, fontweight='bold',
                            color=STAGE_COLORS.get(stage, 'black'))

    # Shared colorbar
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=Normalize(-3, 3))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Gi* Z-score\n(>1.96 = Hot, <-1.96 = Cold)', fontsize=10)

    plt.suptitle('Getis-Ord Gi* Hotspot Analysis\n'
                 '(Red = Hot Spots, Blue = Cold Spots)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig12_getis_ord.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig12_getis_ord.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig12_getis_ord.png/pdf")


def figure_13_correlogram(base_dir: Path, output_dir: Path,
                          backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 13: Spatial correlograms.

    Shows how spatial autocorrelation changes with distance.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    stage_samples = {}
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage in STAGE_ORDER and stage not in stage_samples:
            stage_samples[stage] = sample_dir

    stages_available = [s for s in STAGE_ORDER if s in stage_samples]

    if not stages_available:
        print("No samples found!")
        return

    celltypes = ['AT2', 'Macrophages', 'Fibroblast lineage']

    fig, axes = plt.subplots(1, len(celltypes), figsize=(5 * len(celltypes), 4))
    if len(celltypes) == 1:
        axes = [axes]

    for ax, ct in zip(axes, celltypes):
        for stage in stages_available:
            adata = load_spatial_data(stage_samples[stage], backend)
            col_name = f'{backend}_{ct}'

            if adata is None or col_name not in adata.obs.columns:
                continue

            coords = adata.obsm['spatial']
            values = adata.obs[col_name].values

            # Subsample heavily for correlogram
            if len(values) > 2000:
                idx = np.random.choice(len(values), 2000, replace=False)
                coords_sub = coords[idx]
                values_sub = values[idx]
            else:
                coords_sub = coords
                values_sub = values

            # Compute correlogram
            dist_bins, moran_vals = compute_correlogram(values_sub, coords_sub, n_bins=12)

            ax.plot(dist_bins, moran_vals, 'o-', color=STAGE_COLORS.get(stage, '#666666'),
                   label=stage, linewidth=2, markersize=4)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Moran\'s I')
        ax.set_title(ct, fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.set_ylim(-0.5, 1.0)

    plt.suptitle('Spatial Correlograms\n(Moran\'s I vs Distance Lag)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig13_correlogram.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig13_correlogram.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig13_correlogram.png/pdf")


def figure_14_colocalization(base_dir: Path, output_dir: Path,
                             backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 14: Cell type co-localization analysis.

    Bivariate Local Moran's I between cell type pairs.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    # Find LUAD sample
    target_sample = None
    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        stage = extract_stage(sample_dir.name)
        if stage == 'LUAD':
            target_sample = sample_dir
            break

    if target_sample is None:
        print("No LUAD sample found!")
        return

    adata = load_spatial_data(target_sample, backend)
    if adata is None:
        print("Could not load data!")
        return

    coords = adata.obsm['spatial']

    # Cell type pairs to analyze
    pairs = [
        ('AT2', 'Macrophages'),
        ('AT2', 'Fibroblast lineage'),
        ('Macrophages', 'T cell lineage'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (ct1, ct2) in enumerate(pairs):
        col1 = f'{backend}_{ct1}'
        col2 = f'{backend}_{ct2}'

        if col1 not in adata.obs.columns or col2 not in adata.obs.columns:
            for row in range(2):
                axes[row, idx].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                   transform=axes[row, idx].transAxes)
                axes[row, idx].axis('off')
            continue

        values1 = adata.obs[col1].values
        values2 = adata.obs[col2].values

        # Subsample
        if len(values1) > 5000:
            sub_idx = np.random.choice(len(values1), 5000, replace=False)
            coords_sub = coords[sub_idx]
            values1_sub = values1[sub_idx]
            values2_sub = values2[sub_idx]
        else:
            coords_sub = coords
            values1_sub = values1
            values2_sub = values2

        # Compute bivariate Local Moran
        W = compute_spatial_weights(coords_sub, k=8)
        z1 = (values1_sub - values1_sub.mean()) / (values1_sub.std() + 1e-10)
        z2 = (values2_sub - values2_sub.mean()) / (values2_sub.std() + 1e-10)
        Wz2 = W.dot(z2)
        I_bivariate = z1 * Wz2

        # Top row: Bivariate Local Moran's I
        ax1 = axes[0, idx]
        scatter = ax1.scatter(
            coords_sub[:, 0], coords_sub[:, 1],
            c=I_bivariate, s=5, alpha=0.7,
            cmap='PiYG', vmin=-2, vmax=2,
            edgecolors='none', rasterized=True
        )
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'{ct1} - {ct2}\nBivariate Local Moran\'s I',
                     fontweight='bold', fontsize=10)
        plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)

        # Bottom row: Scatter plot of proportions
        ax2 = axes[1, idx]
        ax2.scatter(values1_sub, values2_sub, c=STAGE_COLORS['LUAD'],
                   s=3, alpha=0.3, edgecolors='none', rasterized=True)

        # Correlation
        corr = np.corrcoef(values1_sub, values2_sub)[0, 1]
        ax2.set_xlabel(ct1)
        ax2.set_ylabel(ct2)
        ax2.set_title(f'Correlation: r = {corr:.3f}', fontsize=10)
        ax2.ticklabel_format(style='scientific', scilimits=(-3, 3))

    plt.suptitle('Cell Type Co-localization Analysis (LUAD)\n'
                 '(Green = positive co-localization, Pink = negative)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig14_colocalization.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig14_colocalization.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig14_colocalization.png/pdf")


# =============================================================================
# BENCHMARK-SPECIFIC METRICS (JSD, RMSE, Correlation)
# =============================================================================

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon Divergence between two distributions.

    JSD is symmetric and bounded [0, 1] when using log2.
    """
    # Normalize to proper distributions
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)

    # Midpoint distribution
    m = 0.5 * (p + q)

    # KL divergences (with epsilon to avoid log(0))
    eps = 1e-10
    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps)))

    return 0.5 * (kl_pm + kl_qm)


def compute_benchmark_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute standard deconvolution benchmark metrics.

    Parameters
    ----------
    pred : array (n_spots, n_celltypes)
        Predicted cell type proportions
    true : array (n_spots, n_celltypes)
        True/reference cell type proportions

    Returns
    -------
    dict with JSD, RMSE, Pearson correlation
    """
    n_spots, n_cts = pred.shape

    # Per-spot JSD
    jsd_per_spot = np.array([
        jensen_shannon_divergence(pred[i], true[i])
        for i in range(n_spots)
    ])

    # Per-celltype RMSE
    rmse_per_ct = np.sqrt(np.mean((pred - true) ** 2, axis=0))

    # Total RMSE
    rmse_total = np.sqrt(np.mean((pred - true) ** 2))

    # Pearson correlation (flattened)
    corr, pval = stats.pearsonr(pred.flatten(), true.flatten())

    # Per-celltype correlation
    corr_per_ct = np.array([
        stats.pearsonr(pred[:, i], true[:, i])[0]
        for i in range(n_cts)
    ])

    return {
        'jsd_median': np.median(jsd_per_spot),
        'jsd_mean': np.mean(jsd_per_spot),
        'jsd_per_spot': jsd_per_spot,
        'rmse_total': rmse_total,
        'rmse_per_ct': rmse_per_ct,
        'pearson_r': corr,
        'pearson_pval': pval,
        'pearson_per_ct': corr_per_ct,
    }


def figure_15_spatial_summary(data: pd.DataFrame, base_dir: Path, output_dir: Path,
                              backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 15: Comprehensive spatial statistics summary.

    Bar chart comparing Global Moran's I across stages for multiple cell types.
    """
    samples_dir = base_dir / label_source / backend / 'samples'

    celltypes = ['AT2', 'Macrophages', 'Fibroblast lineage', 'T cell lineage']

    # Collect Global Moran's I for each sample
    results = []

    for sample_dir in samples_dir.iterdir():
        if not sample_dir.is_dir():
            continue

        stage = extract_stage(sample_dir.name)
        if stage not in STAGE_ORDER:
            continue

        adata = load_spatial_data(sample_dir, backend)
        if adata is None:
            continue

        coords = adata.obsm['spatial']

        # Subsample for speed
        if len(coords) > 3000:
            idx = np.random.choice(len(coords), 3000, replace=False)
            coords_sub = coords[idx]
        else:
            coords_sub = coords
            idx = slice(None)

        W = compute_spatial_weights(coords_sub, k=8)

        for ct in celltypes:
            col_name = f'{backend}_{ct}'
            if col_name not in adata.obs.columns:
                continue

            values = adata.obs[col_name].values
            if isinstance(idx, np.ndarray):
                values = values[idx]

            # Global Moran's I
            z = (values - values.mean()) / (values.std() + 1e-10)
            Wz = W.dot(z)
            I_global = np.mean(z * Wz)

            results.append({
                'sample': sample_dir.name,
                'stage': stage,
                'celltype': ct,
                'morans_i': I_global
            })

    if not results:
        print("No results computed!")
        return

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by stage and celltype
    x_pos = np.arange(len(STAGE_ORDER))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(celltypes))

    ct_colors = ['#922B21', '#1B4F72', '#1D6F42', '#D4A03C']

    for i, ct in enumerate(celltypes):
        ct_data = df[df['celltype'] == ct]

        means = []
        sems = []
        for stage in STAGE_ORDER:
            stage_vals = ct_data[ct_data['stage'] == stage]['morans_i'].values
            if len(stage_vals) > 0:
                means.append(np.mean(stage_vals))
                sems.append(stats.sem(stage_vals) if len(stage_vals) > 1 else 0)
            else:
                means.append(0)
                sems.append(0)

        ax.bar(x_pos + offsets[i], means, width, yerr=sems,
               label=ct, color=ct_colors[i], capsize=3, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Global Moran\'s I')
    ax.set_xlabel('Disease Stage')
    ax.legend(loc='upper right', frameon=False)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    # Color x-tick labels
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(STAGE_ORDER[i], 'black'))
        label.set_fontweight('bold')

    ax.set_title('Spatial Autocorrelation (Global Moran\'s I) Across Disease Stages\n'
                 '(Higher = more spatially clustered)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig15_spatial_summary.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig15_spatial_summary.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig15_spatial_summary.png/pdf")


# =============================================================================
# BENCHMARK COMPARISON FIGURES
# =============================================================================

def figure_16_backend_agreement(base_dir: Path, output_dir: Path, label_source: str = 'hlca'):
    """
    Figure 16: Backend method agreement heatmap.

    Shows correlation between different deconvolution methods.
    """
    backends = ['tangram', 'destvi', 'tacco', 'cell2location']

    # Find a sample with multiple backends
    sample_availability = defaultdict(dict)
    for backend in backends:
        samples_dir = base_dir / label_source / backend / 'samples'
        if not samples_dir.exists():
            continue
        for sample_dir in samples_dir.iterdir():
            if sample_dir.is_dir():
                props_file = sample_dir / 'cell_type_proportions.parquet'
                if props_file.exists():
                    sample_availability[sample_dir.name][backend] = props_file

    # Find sample with most backends
    best_sample = max(sample_availability.keys(),
                      key=lambda x: len(sample_availability[x]),
                      default=None)

    if not best_sample or len(sample_availability[best_sample]) < 2:
        print("Need at least 2 backends for agreement analysis")
        return

    # Load data from each backend
    backend_data = {}
    common_celltypes = None

    for backend, props_file in sample_availability[best_sample].items():
        df = pd.read_parquet(props_file)
        backend_data[backend] = df
        if common_celltypes is None:
            common_celltypes = set(df.columns)
        else:
            common_celltypes &= set(df.columns)

    common_celltypes = sorted(list(common_celltypes))
    available_backends = list(backend_data.keys())
    n_backends = len(available_backends)

    # Compute pairwise correlations
    corr_matrix = np.zeros((n_backends, n_backends))

    for i, b1 in enumerate(available_backends):
        for j, b2 in enumerate(available_backends):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Align indices
                df1 = backend_data[b1][common_celltypes]
                df2 = backend_data[b2][common_celltypes]
                common_idx = df1.index.intersection(df2.index)

                if len(common_idx) > 10:  # Need enough data points
                    v1 = df1.loc[common_idx].values.flatten()
                    v2 = df2.loc[common_idx].values.flatten()
                    # Check for variance
                    if v1.std() > 1e-10 and v2.std() > 1e-10:
                        corr_matrix[i, j] = stats.pearsonr(v1, v2)[0]
                    else:
                        corr_matrix[i, j] = 0.0
                else:
                    corr_matrix[i, j] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n_backends))
    ax.set_yticks(range(n_backends))
    ax.set_xticklabels([b.upper() for b in available_backends], fontsize=11, fontweight='bold')
    ax.set_yticklabels([b.upper() for b in available_backends], fontsize=11, fontweight='bold')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add correlation values
    for i in range(n_backends):
        for j in range(n_backends):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=12,
                          color='white' if corr_matrix[i, j] > 0.7 else 'black',
                          fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation', fontsize=11)

    ax.set_title(f'Backend Method Agreement\n{best_sample}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig16_backend_agreement.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig16_backend_agreement.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig16_backend_agreement.png/pdf")


def figure_17_embedding_umap(base_dir: Path, output_dir: Path,
                             backend: str = 'tangram', label_source: str = 'hlca'):
    """
    Figure 17: UMAP embedding of deconvolution results.

    Shows spots in reduced dimension space colored by stage and dominant cell type.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn not available for embedding")
        return

    samples_dir = base_dir / label_source / backend / 'samples'

    # Collect data from multiple samples
    all_props = []
    all_stages = []
    all_samples = []

    for sample_dir in sorted(samples_dir.iterdir())[:20]:  # Limit for speed
        if not sample_dir.is_dir():
            continue

        props_file = sample_dir / 'cell_type_proportions.parquet'
        if not props_file.exists():
            continue

        df = pd.read_parquet(props_file)
        stage = extract_stage(sample_dir.name)

        # Subsample
        if len(df) > 500:
            df = df.sample(500, random_state=42)

        all_props.append(df.values)
        all_stages.extend([stage] * len(df))
        all_samples.extend([sample_dir.name] * len(df))

    if not all_props:
        print("No data found for embedding")
        return

    # Combine
    X = np.vstack(all_props)
    stages = np.array(all_stages)

    # PCA then t-SNE (faster than UMAP for this demo)
    print("  Computing PCA...")
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X)

    print("  Computing t-SNE...")
    # n_iter renamed to max_iter in newer sklearn versions
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    X_embed = tsne.fit_transform(X_pca)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Color by stage
    ax1 = axes[0]
    for stage in STAGE_ORDER:
        mask = stages == stage
        if mask.sum() > 0:
            ax1.scatter(X_embed[mask, 0], X_embed[mask, 1],
                       c=STAGE_COLORS.get(stage, '#666666'),
                       s=5, alpha=0.5, label=stage,
                       edgecolors='none', rasterized=True)

    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('Colored by Disease Stage', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right: Color by dominant cell type
    ax2 = axes[1]
    dominant_ct = np.argmax(X, axis=1)

    # Get cell type names (from first sample's columns)
    first_props = pd.read_parquet(list(samples_dir.iterdir())[0] / 'cell_type_proportions.parquet')
    ct_names = first_props.columns.tolist()

    scatter = ax2.scatter(X_embed[:, 0], X_embed[:, 1],
                         c=dominant_ct, s=5, alpha=0.5,
                         cmap='tab10', edgecolors='none', rasterized=True)

    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('Colored by Dominant Cell Type', fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Colorbar with cell type names
    cbar = plt.colorbar(scatter, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(ct_names)))
    cbar.set_ticklabels(ct_names, fontsize=7)

    plt.suptitle(f'Deconvolution Embedding ({backend.upper()})\n'
                 f'{len(np.unique(all_samples))} samples, {len(X)} spots',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig17_embedding_umap.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig17_embedding_umap.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig17_embedding_umap.png/pdf")


def figure_18_multibackend_spatial(base_dir: Path, output_dir: Path, label_source: str = 'hlca'):
    """
    Figure 18: Multi-backend spatial comparison (benchmark style).

    Side-by-side spatial plots from different backends for same sample.
    """
    backends = ['tangram', 'destvi', 'tacco', 'cell2location']
    celltype = 'AT2'

    # Find sample with most backends
    sample_availability = defaultdict(set)
    for backend in backends:
        samples_dir = base_dir / label_source / backend / 'samples'
        if samples_dir.exists():
            for sample_dir in samples_dir.iterdir():
                if sample_dir.is_dir():
                    sample_availability[sample_dir.name].add(backend)

    best_sample = max(sample_availability.keys(),
                      key=lambda x: len(sample_availability[x]),
                      default=None)

    if not best_sample:
        print("No samples found!")
        return

    available_backends = sorted(sample_availability[best_sample])
    n_backends = len(available_backends)

    fig, axes = plt.subplots(1, n_backends + 1, figsize=(4 * (n_backends + 1), 4))

    # First panel: combined marker or dominant cell type from first backend
    first_backend = available_backends[0]
    first_adata = load_spatial_data(
        base_dir / label_source / first_backend / 'samples' / best_sample,
        first_backend
    )

    if first_adata is not None:
        coords = first_adata.obsm['spatial']
        ct_cols = [c for c in first_adata.obs.columns if c.startswith(f'{first_backend}_')]
        ct_data = first_adata.obs[ct_cols].values
        dominant = np.argmax(ct_data, axis=1)

        axes[0].scatter(coords[:, 0], coords[:, 1], c=dominant, s=5, alpha=0.7,
                       cmap='tab10', edgecolors='none', rasterized=True)
        axes[0].set_aspect('equal')
        axes[0].axis('off')
        axes[0].set_title('Dominant\nCell Type', fontweight='bold')

    # Remaining panels: each backend's proportion for target cell type
    for i, backend in enumerate(available_backends, 1):
        ax = axes[i]
        adata = load_spatial_data(
            base_dir / label_source / backend / 'samples' / best_sample,
            backend
        )

        if adata is None:
            ax.text(0.5, 0.5, f'{backend}\nN/A', ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')
            continue

        col_name = f'{backend}_{celltype}'
        if col_name not in adata.obs.columns:
            ax.text(0.5, 0.5, f'{backend}\n{celltype} N/A', ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')
            continue

        coords = adata.obsm['spatial']
        values = adata.obs[col_name].values

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=5, alpha=0.7,
                            cmap='Reds', vmin=0, vmax=np.percentile(values, 99),
                            edgecolors='none', rasterized=True)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{backend.upper()}\n{celltype}', fontweight='bold')

        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

    stage = extract_stage(best_sample)
    plt.suptitle(f'Multi-Backend Comparison - {best_sample} ({stage})',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig18_multibackend_spatial.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig18_multibackend_spatial.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig18_multibackend_spatial.png/pdf")


def figure_19_celltype_performance(base_dir: Path, output_dir: Path, label_source: str = 'hlca'):
    """
    Figure 19: Per-cell-type performance comparison across backends.

    Bar chart showing mean proportion and variance per cell type per backend.
    """
    backends = ['tangram', 'destvi', 'tacco', 'cell2location']

    # Collect statistics
    ct_stats = defaultdict(lambda: defaultdict(list))

    for backend in backends:
        samples_dir = base_dir / label_source / backend / 'samples'
        if not samples_dir.exists():
            continue

        for sample_dir in samples_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            props_file = sample_dir / 'cell_type_proportions.parquet'
            if not props_file.exists():
                continue

            df = pd.read_parquet(props_file)

            for ct in df.columns:
                ct_stats[ct][backend].append(df[ct].mean())

    if not ct_stats:
        print("No data found!")
        return

    # Get common cell types
    celltypes = sorted(ct_stats.keys())[:8]  # Top 8
    available_backends = [b for b in backends if any(b in ct_stats[ct] for ct in celltypes)]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(celltypes))
    width = 0.2
    offsets = np.linspace(-0.3, 0.3, len(available_backends))

    backend_colors = {
        'tangram': '#922B21',
        'destvi': '#1B4F72',
        'tacco': '#1D6F42',
        'cell2location': '#D4A03C',
    }

    for i, backend in enumerate(available_backends):
        means = []
        stds = []
        for ct in celltypes:
            values = ct_stats[ct].get(backend, [0])
            means.append(np.mean(values))
            stds.append(np.std(values))

        ax.bar(x + offsets[i], means, width, yerr=stds,
               label=backend.upper(), color=backend_colors.get(backend, '#666666'),
               capsize=2, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(celltypes, rotation=45, ha='right')
    ax.set_ylabel('Mean Proportion')
    ax.set_xlabel('Cell Type')
    ax.legend(loc='upper right', frameon=False)
    ax.set_title('Per-Cell-Type Predictions Across Backends',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig19_celltype_performance.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig19_celltype_performance.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig19_celltype_performance.png/pdf")


def figure_20_ridgeline(data: pd.DataFrame, output_dir: Path):
    """
    Figure 20: Ridgeline/Joy plot of cell type proportions by stage.

    Beautiful visualization of distribution shifts across disease progression.
    """
    ct_cols = [c for c in data.columns if c not in ['sample', 'stage', 'patient']]

    # Take top 4 cell types by overall proportion
    ct_means = data[ct_cols].mean().nlargest(4)
    top_cts = ct_means.index.tolist()

    fig, axes = plt.subplots(len(top_cts), 1, figsize=(10, 3 * len(top_cts)), sharex=True)
    if len(top_cts) == 1:
        axes = [axes]

    for ax, ct in zip(axes, top_cts):
        # Get data per stage
        y_offset = 0
        max_density = 0

        for stage in STAGE_ORDER:
            stage_data = data[data['stage'] == stage][ct].dropna().values

            if len(stage_data) < 10:
                continue

            # Compute KDE
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(stage_data)
                x_range = np.linspace(0, np.percentile(stage_data, 99), 200)
                density = kde(x_range)
                max_density = max(max_density, density.max())
            except:
                continue

        # Now plot with scaling
        y_offset = 0
        for i, stage in enumerate(STAGE_ORDER):
            stage_data = data[data['stage'] == stage][ct].dropna().values

            if len(stage_data) < 10:
                continue

            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(stage_data)
                x_range = np.linspace(0, np.percentile(data[ct].dropna(), 99), 200)
                density = kde(x_range)

                # Normalize and offset
                density_scaled = density / (max_density + 1e-10) * 0.8

                # Fill
                ax.fill_between(x_range, y_offset, y_offset + density_scaled,
                               color=STAGE_COLORS.get(stage, '#666666'),
                               alpha=0.7, label=stage)

                # Outline
                ax.plot(x_range, y_offset + density_scaled,
                       color=STAGE_COLORS.get(stage, '#666666'),
                       linewidth=1.5)

                y_offset += 1
            except:
                continue

        ax.set_ylabel(ct, fontsize=11, fontweight='bold')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=5)

    axes[-1].set_xlabel('Proportion')
    plt.suptitle('Cell Type Proportion Distributions Across Disease Stages\n(Ridgeline Plot)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'fig20_ridgeline.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig20_ridgeline.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fig20_ridgeline.png/pdf")


def main():
    """Generate all publication figures."""
    base_dir = Path("results/spatial_benchmark")
    output_dir = Path("results/spatial_benchmark/figures")
    backend = 'tangram'
    label_source = 'hlca'

    print("=" * 60)
    print("Generating Publication-Quality Spatial Figures")
    print("=" * 60)

    # Load all data
    print("\nLoading data...")
    data = collect_all_data(base_dir, backend, label_source)
    print(f"Loaded {len(data)} spots from {data['sample'].nunique()} samples")
    print(f"Stages: {data['stage'].value_counts().to_dict()}")

    # Generate figures
    print("\n" + "-" * 40)
    print("Figure 1: Cell Type Composition (Violin plots)")
    figure_1_celltype_composition(data, output_dir)

    print("\n" + "-" * 40)
    print("Figure 2: Spatial Cell Types by Stage")
    figure_2_spatial_examples(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 3: AT2 Spatial Distribution")
    figure_3_at2_spatial(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 4: Macrophage Infiltration")
    figure_4_macrophage_infiltration(data, base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 5: Cell Type Summary Heatmap")
    figure_5_celltype_summary_heatmap(data, output_dir)

    print("\n" + "-" * 40)
    print("Figure 6: Backend Comparison")
    figure_6_backend_comparison(base_dir, output_dir)

    print("\n" + "-" * 40)
    print("Figure 7: Multi-panel Spatial (Cell2location style)")
    figure_7_multipanel_spatial(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 8: Stage Comparison Grid")
    figure_8_stage_comparison_grid(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 9: Detailed Violin Plots")
    figure_9_violin_detailed(data, output_dir)

    # Spatial Statistics Figures
    print("\n" + "=" * 40)
    print("SPATIAL STATISTICS ANALYSIS")
    print("=" * 40)

    print("\n" + "-" * 40)
    print("Figure 10: Local Moran's I (LISA)")
    figure_10_local_morans(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 11: Moran Scatter Plots")
    figure_11_moran_scatter(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 12: Getis-Ord Gi* Hotspots")
    figure_12_getis_ord(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 13: Spatial Correlograms")
    figure_13_correlogram(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 14: Co-localization Analysis")
    figure_14_colocalization(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 15: Spatial Summary Statistics")
    figure_15_spatial_summary(data, base_dir, output_dir, backend, label_source)

    # Benchmark-specific figures
    print("\n" + "=" * 40)
    print("BENCHMARK COMPARISON FIGURES")
    print("=" * 40)

    print("\n" + "-" * 40)
    print("Figure 16: Backend Method Agreement")
    figure_16_backend_agreement(base_dir, output_dir, label_source)

    print("\n" + "-" * 40)
    print("Figure 17: Embedding Visualization (UMAP)")
    figure_17_embedding_umap(base_dir, output_dir, backend, label_source)

    print("\n" + "-" * 40)
    print("Figure 18: Multi-Backend Spatial Grid")
    figure_18_multibackend_spatial(base_dir, output_dir, label_source)

    print("\n" + "-" * 40)
    print("Figure 19: Per-Cell-Type Performance")
    figure_19_celltype_performance(base_dir, output_dir, label_source)

    print("\n" + "-" * 40)
    print("Figure 20: Proportion Density Ridgeline")
    figure_20_ridgeline(data, output_dir)

    print("\n" + "=" * 60)
    print(f"Generated 20 publication figures in: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
