#!/usr/bin/env python3
"""Publication-quality dual-reference embedding figures.

Creates sophisticated, Nature Methods-style visualizations from cells.parquet:
- Multi-panel embedding overviews with proper annotations
- Biological feature integration (pathways, cell cycle, mutations)
- Statistical comparisons with effect sizes
- Density-based visualizations
- Trajectory/progression analysis
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.ndimage import gaussian_filter
from collections import defaultdict

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# =============================================================================
# PUBLICATION SETTINGS
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
})

# Dark, rich stage colors
STAGE_COLORS = {
    'Normal': '#1B4F72',
    'AAH': '#2E86AB',
    'AIS': '#1D6F42',
    'MIA': '#D4A03C',
    'LUAD': '#922B21',
}
STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Cell type palette
CELLTYPE_COLORS = {
    'AT2': '#E64B35',
    'Basal': '#00A087',
    'Ciliated': '#3C5488',
    'Secretory': '#F39B7F',
    'T cell lineage': '#91D1C2',
    'Macrophages': '#7E6148',
    'Mast cells': '#E18727',
    'Fibroblast lineage': '#7876B1',
    'Capillary': '#6F99AD',
    'mixed': '#CCCCCC',
}

# Cell cycle colors
CYCLE_COLORS = {
    'G1': '#3498DB',
    'S': '#E74C3C',
    'G2M': '#2ECC71',
}


def get_celltype_color(ct):
    return CELLTYPE_COLORS.get(ct, '#999999')


def format_pvalue(p):
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir: Path):
    """Load cells and neighborhoods."""
    cells = pd.read_parquet(data_dir / "cells.parquet")
    print(f"Loaded {len(cells):,} cells")
    return cells


def get_embeddings(df, prefix):
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    return df[cols].values.astype(np.float32) if cols else None


def compute_umap(X, n_neighbors=30, min_dist=0.3, seed=42):
    """Compute UMAP or fallback to PCA."""
    if HAS_UMAP:
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=seed, n_jobs=1).fit_transform(X)
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def sample_balanced(df, n_per_stage=5000, seed=42):
    """Sample balanced across stages."""
    np.random.seed(seed)
    samples = []
    for stage in STAGE_ORDER:
        stage_df = df[df['stage'] == stage]
        n = min(len(stage_df), n_per_stage)
        samples.append(stage_df.sample(n, random_state=seed))
    return pd.concat(samples, ignore_index=True)


# =============================================================================
# FIGURE 1: COMPREHENSIVE EMBEDDING OVERVIEW
# =============================================================================

def figure1_embedding_overview(cells, output_dir):
    """Publication Figure 1: Comprehensive embedding overview."""
    print("\nGenerating Figure 1: Embedding Overview...")

    # Sample for visualization
    cells_s = sample_balanced(cells, n_per_stage=6000)
    fused = get_embeddings(cells_s, "z_fused_")

    print("  Computing UMAP...")
    umap_coords = compute_umap(fused)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # A: Main UMAP by stage (large panel)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_main.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=STAGE_COLORS[stage], s=8, alpha=0.6,
                       label=f'{stage} (n={mask.sum():,})', rasterized=True)
    ax_main.legend(loc='upper right', fontsize=10, markerscale=2, framealpha=0.9)
    ax_main.set_xlabel('UMAP 1', fontsize=12)
    ax_main.set_ylabel('UMAP 2', fontsize=12)
    ax_main.set_title('A. Fused Dual-Reference Embeddings by Disease Stage', fontsize=13)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # B: Density contours per stage
    ax_density = fig.add_subplot(gs[0, 2])
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        if mask.sum() > 100:
            sns.kdeplot(x=umap_coords[mask, 0], y=umap_coords[mask, 1],
                       color=STAGE_COLORS[stage], levels=3, linewidths=1.5,
                       ax=ax_density, alpha=0.8)
    ax_density.set_xlabel('UMAP 1')
    ax_density.set_ylabel('UMAP 2')
    ax_density.set_title('B. Stage Density Contours', fontsize=11)
    ax_density.set_xticks([])
    ax_density.set_yticks([])

    # C: Cell type overlay
    ax_ct = fig.add_subplot(gs[0, 3])
    for ct in cells_s['cell_type'].unique():
        if ct == 'mixed':
            continue
        mask = cells_s['cell_type'] == ct
        ax_ct.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                     c=get_celltype_color(ct), s=5, alpha=0.5, label=ct, rasterized=True)
    ax_ct.legend(loc='upper right', fontsize=7, markerscale=2, ncol=1)
    ax_ct.set_xlabel('UMAP 1')
    ax_ct.set_ylabel('UMAP 2')
    ax_ct.set_title('C. Cell Type Distribution', fontsize=11)
    ax_ct.set_xticks([])
    ax_ct.set_yticks([])

    # D: Cell cycle
    ax_cycle = fig.add_subplot(gs[1, 2])
    if 'phase' in cells_s.columns:
        for phase in ['G1', 'S', 'G2M']:
            mask = cells_s['phase'] == phase
            if mask.sum() > 0:
                ax_cycle.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                               c=CYCLE_COLORS.get(phase, '#999999'), s=5, alpha=0.5,
                               label=f'{phase} (n={mask.sum():,})', rasterized=True)
        ax_cycle.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_cycle.set_xlabel('UMAP 1')
    ax_cycle.set_ylabel('UMAP 2')
    ax_cycle.set_title('D. Cell Cycle Phase', fontsize=11)
    ax_cycle.set_xticks([])
    ax_cycle.set_yticks([])

    # E: Proliferation score
    ax_prolif = fig.add_subplot(gs[1, 3])
    if 'S_score' in cells_s.columns and 'G2M_score' in cells_s.columns:
        prolif_score = cells_s['S_score'] + cells_s['G2M_score']
        scatter = ax_prolif.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                   c=prolif_score, s=5, alpha=0.6,
                                   cmap='RdYlBu_r', vmin=-0.5, vmax=0.5, rasterized=True)
        plt.colorbar(scatter, ax=ax_prolif, shrink=0.7, label='Proliferation')
    ax_prolif.set_xlabel('UMAP 1')
    ax_prolif.set_ylabel('UMAP 2')
    ax_prolif.set_title('E. Proliferation Score', fontsize=11)
    ax_prolif.set_xticks([])
    ax_prolif.set_yticks([])

    # F: Stage composition bar chart
    ax_bar = fig.add_subplot(gs[2, 0])
    stage_counts = cells['stage'].value_counts().reindex(STAGE_ORDER)
    bars = ax_bar.bar(STAGE_ORDER, stage_counts.values,
                     color=[STAGE_COLORS[s] for s in STAGE_ORDER],
                     edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, stage_counts.values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count/1000:.0f}k', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_bar.set_ylabel('Cell Count')
    ax_bar.set_title('F. Stage Distribution', fontsize=11)
    ax_bar.set_ylim(0, stage_counts.max() * 1.15)
    ax_bar.tick_params(axis='x', rotation=45)

    # G: Cell type by stage heatmap
    ax_heat = fig.add_subplot(gs[2, 1])
    ct_stage = pd.crosstab(cells['stage'], cells['cell_type'], normalize='index')
    ct_stage = ct_stage.reindex(STAGE_ORDER)
    ct_stage = ct_stage[[c for c in ct_stage.columns if c != 'mixed']]
    im = ax_heat.imshow(ct_stage.values, aspect='auto', cmap='YlOrRd')
    ax_heat.set_xticks(range(len(ct_stage.columns)))
    ax_heat.set_xticklabels(ct_stage.columns, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(STAGE_ORDER)))
    ax_heat.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_heat.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_heat.set_title('G. Cell Type by Stage', fontsize=11)
    plt.colorbar(im, ax=ax_heat, shrink=0.7, label='Proportion')

    # H: Mean embedding heatmap
    ax_emb = fig.add_subplot(gs[2, 2])
    fused_all = get_embeddings(cells, "z_fused_")
    stage_means = np.array([fused_all[cells['stage'] == s].mean(axis=0) for s in STAGE_ORDER])
    im2 = ax_emb.imshow(stage_means, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax_emb.set_yticks(range(len(STAGE_ORDER)))
    ax_emb.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_emb.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_emb.set_xlabel('Embedding Dimension')
    ax_emb.set_title('H. Mean Fused Embedding', fontsize=11)
    plt.colorbar(im2, ax=ax_emb, shrink=0.7)

    # I: Dimension importance
    ax_var = fig.add_subplot(gs[2, 3])
    # Compute between-stage variance for each dimension
    between_var = np.var(stage_means, axis=0)
    within_var = np.array([fused_all[cells['stage'] == s].var(axis=0).mean() for s in STAGE_ORDER]).mean()
    f_ratio = between_var / (within_var + 1e-10)
    top_dims = np.argsort(f_ratio)[-15:][::-1]
    ax_var.barh(range(len(top_dims)), f_ratio[top_dims], color='#1B4F72', alpha=0.8)
    ax_var.set_yticks(range(len(top_dims)))
    ax_var.set_yticklabels([f'D{d}' for d in top_dims], fontsize=8)
    ax_var.set_xlabel('F-ratio (Between/Within)')
    ax_var.set_title('I. Stage-Discriminative Dims', fontsize=11)
    ax_var.invert_yaxis()

    plt.suptitle('Dual-Reference Embedding Analysis of LUAD Progression',
                fontsize=15, fontweight='bold', y=1.01)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig1_embedding_overview.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig1_embedding_overview.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig1_embedding_overview.png/pdf")


# =============================================================================
# FIGURE 2: BIOLOGICAL FEATURE ANALYSIS
# =============================================================================

def figure2_biological_features(cells, output_dir):
    """Publication Figure 2: Biological feature analysis."""
    print("\nGenerating Figure 2: Biological Features...")

    cells_s = sample_balanced(cells, n_per_stage=5000)
    fused = get_embeddings(cells_s, "z_fused_")
    umap_coords = compute_umap(fused)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Pathway columns
    pathway_cols = [c for c in cells_s.columns if c.startswith('pathway_')]

    # A-D: Top 4 pathway activities
    pathway_names = ['EMT', 'Hypoxia', 'Inflammation', 'Proliferation',
                    'Apoptosis', 'Angiogenesis', 'DNA Repair', 'Metabolism',
                    'WNT', 'NOTCH', 'TGFb', 'JAK-STAT', 'PI3K', 'MAPK']

    for i, (col, name) in enumerate(zip(pathway_cols[:4], pathway_names[:4])):
        ax = fig.add_subplot(gs[0, i])
        values = cells_s[col].values
        vmin, vmax = np.percentile(values, [5, 95])
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=values, s=5, alpha=0.6, cmap='viridis',
                           vmin=vmin, vmax=vmax, rasterized=True)
        plt.colorbar(scatter, ax=ax, shrink=0.7)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{chr(65+i)}. {name} Score', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # E: Pathway score violin by stage
    ax_violin = fig.add_subplot(gs[1, 0:2])
    pathway_data = []
    for stage in STAGE_ORDER:
        for col, name in zip(pathway_cols[:6], pathway_names[:6]):
            stage_vals = cells[cells['stage'] == stage][col].values
            for v in np.random.choice(stage_vals, min(500, len(stage_vals)), replace=False):
                pathway_data.append({'Stage': stage, 'Pathway': name, 'Score': v})

    pathway_df = pd.DataFrame(pathway_data)

    # Create grouped violin
    positions = []
    for i, pathway in enumerate(pathway_names[:6]):
        for j, stage in enumerate(STAGE_ORDER):
            positions.append(i * 6 + j)

    for i, pathway in enumerate(pathway_names[:6]):
        for j, stage in enumerate(STAGE_ORDER):
            data = pathway_df[(pathway_df['Pathway'] == pathway) & (pathway_df['Stage'] == stage)]['Score']
            if len(data) > 10:
                parts = ax_violin.violinplot([data.values], positions=[i * 6 + j],
                                            showmeans=False, showmedians=True, widths=0.8)
                for pc in parts['bodies']:
                    pc.set_facecolor(STAGE_COLORS[stage])
                    pc.set_alpha(0.7)
                parts['cmedians'].set_color('black')

    ax_violin.set_xticks([2.5 + i*6 for i in range(6)])
    ax_violin.set_xticklabels(pathway_names[:6], fontsize=9)
    ax_violin.set_ylabel('Pathway Score')
    ax_violin.set_title('E. Pathway Activity by Disease Stage', fontsize=11)

    # Legend for stages
    handles = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER]
    ax_violin.legend(handles=handles, loc='upper right', fontsize=8, ncol=5)

    # F: Mutation status
    ax_mut = fig.add_subplot(gs[1, 2])
    mut_cols = ['kras_mut', 'egfr_mut', 'tp53_mut', 'stk11_mut']
    mut_names = ['KRAS', 'EGFR', 'TP53', 'STK11']

    mut_by_stage = []
    for stage in STAGE_ORDER:
        stage_cells = cells[cells['stage'] == stage]
        mut_rates = [stage_cells[c].mean() * 100 for c in mut_cols]
        mut_by_stage.append(mut_rates)

    mut_array = np.array(mut_by_stage)
    im = ax_mut.imshow(mut_array, aspect='auto', cmap='Reds', vmin=0, vmax=100)
    ax_mut.set_xticks(range(len(mut_names)))
    ax_mut.set_xticklabels(mut_names, fontsize=9)
    ax_mut.set_yticks(range(len(STAGE_ORDER)))
    ax_mut.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_mut.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')

    # Add percentages
    for i in range(len(STAGE_ORDER)):
        for j in range(len(mut_names)):
            ax_mut.text(j, i, f'{mut_array[i,j]:.0f}%', ha='center', va='center',
                       fontsize=9, color='white' if mut_array[i,j] > 50 else 'black')

    plt.colorbar(im, ax=ax_mut, shrink=0.7, label='% Mutated')
    ax_mut.set_title('F. Mutation Frequency', fontsize=11)

    # G: Evolutionary clonal patterns (from paper Fig 2D)
    ax_clonal = fig.add_subplot(gs[1, 3])

    # Load paper patterns if available
    paper_patterns_path = Path("data/paper/clonal_patterns.json")
    if paper_patterns_path.exists():
        import json
        with open(paper_patterns_path) as f:
            paper_patterns = json.load(f)

        # Map donor to pattern
        patient_to_pattern = paper_patterns.get('patient_to_pattern', {})
        cells_with_pattern = cells.copy()
        cells_with_pattern['evo_pattern'] = cells_with_pattern['donor_id'].map(patient_to_pattern)
        cells_with_pattern = cells_with_pattern.dropna(subset=['evo_pattern'])

        # Pattern colors and labels matching paper
        pattern_colors = {'1a': '#2E86AB', '1b': '#A23B72', '2': '#F18F01'}
        pattern_labels = {
            '1a': '1a: Direct lineage',
            '1b': '1b: Branched',
            '2': '2: Independent'
        }

        # Compute proportions by stage
        pattern_by_stage = pd.crosstab(
            cells_with_pattern['stage'],
            cells_with_pattern['evo_pattern'],
            normalize='index'
        )
        pattern_by_stage = pattern_by_stage.reindex(STAGE_ORDER).reindex(columns=['1a', '1b', '2'])

        # Stacked bar
        bottom = np.zeros(len(STAGE_ORDER))
        x = np.arange(len(STAGE_ORDER))
        for pattern in ['1a', '1b', '2']:
            if pattern in pattern_by_stage.columns:
                values = pattern_by_stage[pattern].values
                ax_clonal.bar(x, values, bottom=bottom, color=pattern_colors[pattern],
                            label=pattern_labels[pattern], edgecolor='white', linewidth=0.5)
                bottom += values

        ax_clonal.set_xticks(x)
        ax_clonal.set_xticklabels(STAGE_ORDER, fontsize=9)
        for i, label in enumerate(ax_clonal.get_xticklabels()):
            label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
            label.set_fontweight('bold')
        ax_clonal.set_ylabel('Proportion')
        ax_clonal.set_xlabel('')
        ax_clonal.legend(title='Evolution', fontsize=7, loc='upper right')
        ax_clonal.tick_params(axis='x', rotation=45)
    else:
        ax_clonal.text(0.5, 0.5, 'Clonal patterns\nnot available',
                      ha='center', va='center', transform=ax_clonal.transAxes)
    ax_clonal.set_title('G. Clonal Evolution', fontsize=11)

    plt.suptitle('Biological Feature Integration with Embeddings',
                fontsize=15, fontweight='bold', y=1.01)

    fig.savefig(output_dir / "fig2_biological_features.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig2_biological_features.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig2_biological_features.png/pdf")


# =============================================================================
# FIGURE 2B: CLONAL EVOLUTION ANALYSIS (like Peng et al. Figure 2)
# =============================================================================

def figure2b_clonal_evolution(cells, output_dir):
    """Publication Figure 2B: Comprehensive clonal evolution analysis.

    Inspired by Peng et al. Figure 2D showing evolutionary patterns:
    - Pattern 1a: Direct lineage (monoclonal precursor shared with invasive)
    - Pattern 1b: Branched evolution (polyclonal precursor partially shared)
    - Pattern 2: Independent origins (precursor and invasive not related)
    """
    print("\nGenerating Figure 2B: Clonal Evolution Analysis...")

    # Load paper patterns
    paper_patterns_path = Path("data/paper/clonal_patterns.json")
    if not paper_patterns_path.exists():
        print("  WARNING: clonal_patterns.json not found, skipping figure")
        return

    import json
    with open(paper_patterns_path) as f:
        paper_patterns = json.load(f)

    patient_to_pattern = paper_patterns.get('patient_to_pattern', {})
    pattern_info = paper_patterns.get('patterns', {})

    # Map patterns to cells
    cells_with_pattern = cells.copy()
    cells_with_pattern['evo_pattern'] = cells_with_pattern['donor_id'].map(patient_to_pattern)
    cells_with_pattern = cells_with_pattern.dropna(subset=['evo_pattern'])

    # Pattern styling
    pattern_colors = {'1a': '#2E86AB', '1b': '#A23B72', '2': '#F18F01'}
    pattern_names = {
        '1a': 'Direct Lineage',
        '1b': 'Branched Evolution',
        '2': 'Independent Origins'
    }

    # Sample for UMAP
    cells_s = sample_balanced(cells_with_pattern, n_per_stage=3000)
    fused = get_embeddings(cells_s, "z_fused_")
    umap_coords = compute_umap(fused)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # A: UMAP colored by evolutionary pattern
    ax_umap = fig.add_subplot(gs[0, 0:2])
    for pattern in ['1a', '1b', '2']:
        mask = cells_s['evo_pattern'] == pattern
        if mask.sum() > 0:
            ax_umap.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                          c=pattern_colors[pattern], s=8, alpha=0.5,
                          label=f'{pattern}: {pattern_names[pattern]}',
                          rasterized=True)
    ax_umap.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_umap.set_xlabel('UMAP 1')
    ax_umap.set_ylabel('UMAP 2')
    ax_umap.set_title('A. Evolutionary Patterns in Embedding Space', fontsize=11)
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])

    # B: Patient counts per pattern (pie chart)
    ax_pie = fig.add_subplot(gs[0, 2])
    pattern_counts = [pattern_info.get(p, {}).get('n', 0) for p in ['1a', '1b', '2']]
    colors = [pattern_colors[p] for p in ['1a', '1b', '2']]
    wedges, texts, autotexts = ax_pie.pie(
        pattern_counts, colors=colors, autopct='%1.0f%%',
        startangle=90, explode=[0.02, 0.02, 0.02],
        textprops={'fontsize': 10}
    )
    ax_pie.set_title('B. Patient Distribution', fontsize=11)
    # Legend with pattern names
    ax_pie.legend(wedges, [f'{p}: {pattern_names[p]}' for p in ['1a', '1b', '2']],
                 loc='lower center', fontsize=8, bbox_to_anchor=(0.5, -0.15))

    # C: Cell counts per pattern (bar)
    ax_bar = fig.add_subplot(gs[0, 3])
    cell_counts = cells_with_pattern['evo_pattern'].value_counts()
    bars = ax_bar.bar(['1a', '1b', '2'],
                     [cell_counts.get(p, 0) for p in ['1a', '1b', '2']],
                     color=[pattern_colors[p] for p in ['1a', '1b', '2']],
                     edgecolor='white', linewidth=1)
    ax_bar.set_ylabel('Cell Count')
    ax_bar.set_xlabel('Pattern')
    ax_bar.set_title('C. Cells per Pattern', fontsize=11)
    # Add count labels
    for bar, p in zip(bars, ['1a', '1b', '2']):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    # D: Stage composition within each pattern
    ax_comp = fig.add_subplot(gs[1, 0:2])
    stage_by_pattern = pd.crosstab(
        cells_with_pattern['evo_pattern'],
        cells_with_pattern['stage'],
        normalize='index'
    )
    stage_by_pattern = stage_by_pattern.reindex(['1a', '1b', '2']).reindex(columns=STAGE_ORDER)

    x = np.arange(3)
    width = 0.15
    for i, stage in enumerate(STAGE_ORDER):
        offset = (i - 2) * width
        if stage in stage_by_pattern.columns:
            vals = stage_by_pattern[stage].values
            ax_comp.bar(x + offset, vals, width, label=stage,
                       color=STAGE_COLORS[stage], edgecolor='white')
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([f'{p}\n{pattern_names[p]}' for p in ['1a', '1b', '2']], fontsize=9)
    ax_comp.set_ylabel('Proportion')
    ax_comp.set_title('D. Stage Distribution by Evolution Pattern', fontsize=11)
    ax_comp.legend(loc='upper right', fontsize=8, ncol=5)

    # E: Pattern distribution by stage (inverse view)
    ax_inv = fig.add_subplot(gs[1, 2:4])
    pattern_by_stage = pd.crosstab(
        cells_with_pattern['stage'],
        cells_with_pattern['evo_pattern'],
        normalize='index'
    )
    pattern_by_stage = pattern_by_stage.reindex(STAGE_ORDER).reindex(columns=['1a', '1b', '2'])

    x = np.arange(len(STAGE_ORDER))
    width = 0.25
    for i, pattern in enumerate(['1a', '1b', '2']):
        offset = (i - 1) * width
        if pattern in pattern_by_stage.columns:
            vals = pattern_by_stage[pattern].values
            ax_inv.bar(x + offset, vals, width, label=f'{pattern}: {pattern_names[pattern]}',
                      color=pattern_colors[pattern], edgecolor='white')
    ax_inv.set_xticks(x)
    ax_inv.set_xticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_inv.get_xticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_inv.set_ylabel('Proportion')
    ax_inv.set_title('E. Evolution Pattern by Stage', fontsize=11)
    ax_inv.legend(loc='upper right', fontsize=8)

    # F: Schematic of evolutionary patterns
    ax_schema = fig.add_subplot(gs[2, 0:2])
    ax_schema.set_xlim(0, 10)
    ax_schema.set_ylim(0, 6)
    ax_schema.axis('off')

    # Pattern 1a: Direct lineage
    ax_schema.add_patch(plt.Circle((1.5, 4.5), 0.4, color=pattern_colors['1a'], alpha=0.8))
    ax_schema.text(1.5, 4.5, 'N', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.annotate('', xy=(2.5, 4.5), xytext=(2, 4.5),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((3, 4.5), 0.4, color=pattern_colors['1a'], alpha=0.8))
    ax_schema.text(3, 4.5, 'P', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.annotate('', xy=(4, 4.5), xytext=(3.5, 4.5),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((4.5, 4.5), 0.4, color=pattern_colors['1a'], alpha=0.8))
    ax_schema.text(4.5, 4.5, 'I', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.text(6, 4.5, '1a: Direct Lineage\nMonoclonal precursor', fontsize=9, va='center')

    # Pattern 1b: Branched
    ax_schema.add_patch(plt.Circle((1.5, 2.5), 0.4, color=pattern_colors['1b'], alpha=0.8))
    ax_schema.text(1.5, 2.5, 'N', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.annotate('', xy=(2.3, 3), xytext=(1.9, 2.7),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.annotate('', xy=(2.3, 2), xytext=(1.9, 2.3),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((2.8, 3.2), 0.35, color=pattern_colors['1b'], alpha=0.6))
    ax_schema.add_patch(plt.Circle((2.8, 1.8), 0.35, color=pattern_colors['1b'], alpha=0.6))
    ax_schema.text(2.8, 3.2, 'P1', ha='center', va='center', fontsize=8, color='white')
    ax_schema.text(2.8, 1.8, 'P2', ha='center', va='center', fontsize=8, color='white')
    ax_schema.annotate('', xy=(4, 2.5), xytext=(3.2, 3),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((4.5, 2.5), 0.4, color=pattern_colors['1b'], alpha=0.8))
    ax_schema.text(4.5, 2.5, 'I', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.text(6, 2.5, '1b: Branched Evolution\nPolyclonal precursor', fontsize=9, va='center')

    # Pattern 2: Independent
    ax_schema.add_patch(plt.Circle((1.5, 0.5), 0.4, color=pattern_colors['2'], alpha=0.8))
    ax_schema.text(1.5, 0.5, 'N', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.annotate('', xy=(2.5, 1), xytext=(1.9, 0.7),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((3, 1.2), 0.35, color=pattern_colors['2'], alpha=0.6))
    ax_schema.text(3, 1.2, 'P', ha='center', va='center', fontsize=9, color='white')
    ax_schema.add_patch(plt.Circle((3, -0.2), 0.35, color='#888888', alpha=0.6))
    ax_schema.text(3, -0.2, '?', ha='center', va='center', fontsize=9, color='white')
    ax_schema.annotate('', xy=(4, 0.3), xytext=(3.4, -0.1),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_schema.add_patch(plt.Circle((4.5, 0.5), 0.4, color=pattern_colors['2'], alpha=0.8))
    ax_schema.text(4.5, 0.5, 'I', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax_schema.text(6, 0.5, '2: Independent Origins\nUnrelated precursor', fontsize=9, va='center')

    ax_schema.set_title('F. Evolutionary Pattern Schematics', fontsize=11)
    ax_schema.text(0.5, 5.5, 'N=Normal  P=Precursor  I=Invasive', fontsize=8, style='italic')

    # G: Per-patient pattern assignment
    ax_patients = fig.add_subplot(gs[2, 2:4])
    patient_df = pd.DataFrame([
        {'Patient': p, 'Pattern': pat}
        for p, pat in patient_to_pattern.items()
    ])
    patient_df['Pattern_num'] = patient_df['Pattern'].map({'1a': 0, '1b': 1, '2': 2})
    patient_df = patient_df.sort_values(['Pattern_num', 'Patient'])

    y_pos = 0
    for pattern in ['1a', '1b', '2']:
        patients = patient_df[patient_df['Pattern'] == pattern]['Patient'].tolist()
        for i, patient in enumerate(patients):
            ax_patients.add_patch(plt.Rectangle((i * 0.8, y_pos), 0.7, 0.8,
                                               color=pattern_colors[pattern], alpha=0.8))
            ax_patients.text(i * 0.8 + 0.35, y_pos + 0.4, patient,
                           ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        ax_patients.text(-0.5, y_pos + 0.4, f'{pattern}:', ha='right', va='center',
                        fontsize=10, fontweight='bold', color=pattern_colors[pattern])
        y_pos += 1.2

    ax_patients.set_xlim(-1, 12)
    ax_patients.set_ylim(-0.2, 4)
    ax_patients.axis('off')
    ax_patients.set_title('G. Patient Assignments (n=23)', fontsize=11)

    plt.suptitle('Clonal Evolution Patterns in LUAD Progression',
                fontsize=15, fontweight='bold', y=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig2b_clonal_evolution.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig2b_clonal_evolution.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig2b_clonal_evolution.png/pdf")


# =============================================================================
# FIGURE 3: STAGE TRANSITION ANALYSIS
# =============================================================================

def figure3_stage_transitions(cells, output_dir):
    """Publication Figure 3: Stage transition and trajectory analysis."""
    print("\nGenerating Figure 3: Stage Transitions...")

    cells_s = sample_balanced(cells, n_per_stage=4000)
    fused = get_embeddings(cells_s, "z_fused_")
    umap_coords = compute_umap(fused)

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # A: Progression pseudotime (stage index)
    ax_pseudo = fig.add_subplot(gs[0, 0:2])
    stage_numeric = cells_s['stage'].map({s: i for i, s in enumerate(STAGE_ORDER)})
    scatter = ax_pseudo.scatter(umap_coords[:, 0], umap_coords[:, 1],
                               c=stage_numeric, s=8, alpha=0.6,
                               cmap='coolwarm', rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax_pseudo, shrink=0.7)
    cbar.set_ticks(range(5))
    cbar.set_ticklabels(STAGE_ORDER)
    ax_pseudo.set_xlabel('UMAP 1')
    ax_pseudo.set_ylabel('UMAP 2')
    ax_pseudo.set_title('A. Disease Progression Pseudotime', fontsize=11)
    ax_pseudo.set_xticks([])
    ax_pseudo.set_yticks([])

    # Add stage centroids with arrows
    centroids = {}
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        centroids[stage] = umap_coords[mask].mean(axis=0)

    for i in range(len(STAGE_ORDER) - 1):
        s1, s2 = STAGE_ORDER[i], STAGE_ORDER[i+1]
        ax_pseudo.annotate('', xy=centroids[s2], xytext=centroids[s1],
                          arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # B: Per-stage density plots
    ax_dens = fig.add_subplot(gs[0, 2:4])
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        if mask.sum() > 100:
            sns.kdeplot(x=umap_coords[mask, 0], y=umap_coords[mask, 1],
                       color=STAGE_COLORS[stage], levels=5, linewidths=1.5,
                       alpha=0.8, ax=ax_dens)

    # Add centroid labels
    for stage, centroid in centroids.items():
        ax_dens.annotate(stage, xy=centroid, fontsize=10, fontweight='bold',
                        color=STAGE_COLORS[stage], ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor=STAGE_COLORS[stage], alpha=0.9))
    ax_dens.set_xlabel('UMAP 1')
    ax_dens.set_ylabel('UMAP 2')
    ax_dens.set_title('B. Stage Density Landscapes', fontsize=11)
    ax_dens.set_xticks([])
    ax_dens.set_yticks([])

    # C: Embedding distance between stages
    ax_dist = fig.add_subplot(gs[1, 0])
    fused_all = get_embeddings(cells, "z_fused_")
    dist_matrix = np.zeros((5, 5))
    for i, s1 in enumerate(STAGE_ORDER):
        for j, s2 in enumerate(STAGE_ORDER):
            mean1 = fused_all[cells['stage'] == s1].mean(axis=0)
            mean2 = fused_all[cells['stage'] == s2].mean(axis=0)
            dist_matrix[i, j] = np.linalg.norm(mean1 - mean2)

    im = ax_dist.imshow(dist_matrix, cmap='Blues')
    ax_dist.set_xticks(range(5))
    ax_dist.set_xticklabels(STAGE_ORDER, rotation=45, ha='right', fontsize=9)
    ax_dist.set_yticks(range(5))
    ax_dist.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax_dist.text(j, i, f'{dist_matrix[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if dist_matrix[i,j] > dist_matrix.max()/2 else 'black')
    plt.colorbar(im, ax=ax_dist, shrink=0.7, label='Euclidean Distance')
    ax_dist.set_title('C. Inter-Stage Distance', fontsize=11)

    # D: PC1 by stage (violin)
    ax_pc = fig.add_subplot(gs[1, 1])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(fused_all)
    cells_temp = cells.copy()
    cells_temp['PC1'] = pcs[:, 0]

    for i, stage in enumerate(STAGE_ORDER):
        data = cells_temp[cells_temp['stage'] == stage]['PC1'].values
        parts = ax_pc.violinplot([data], positions=[i], showmeans=False, showmedians=True, widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(STAGE_COLORS[stage])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

    ax_pc.set_xticks(range(5))
    ax_pc.set_xticklabels(STAGE_ORDER, fontsize=9)
    ax_pc.set_ylabel('PC1 Score')
    ax_pc.set_title(f'D. PC1 by Stage ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=11)
    # Color x labels
    for i, label in enumerate(ax_pc.get_xticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')

    # E: Transition zone cells
    ax_trans = fig.add_subplot(gs[1, 2])
    # Find cells near stage boundaries (intermediate PC1)
    stage_means = [cells_temp[cells_temp['stage'] == s]['PC1'].mean() for s in STAGE_ORDER]
    transition_mask = np.zeros(len(cells_s), dtype=bool)
    for i in range(len(STAGE_ORDER) - 1):
        boundary = (stage_means[i] + stage_means[i+1]) / 2
        width = abs(stage_means[i+1] - stage_means[i]) * 0.2
        cells_s_temp = cells_s.copy()
        cells_s_temp['PC1'] = pca.transform(fused)[:, 0]
        near_boundary = (cells_s_temp['PC1'] > boundary - width) & (cells_s_temp['PC1'] < boundary + width)
        transition_mask |= near_boundary.values

    ax_trans.scatter(umap_coords[~transition_mask, 0], umap_coords[~transition_mask, 1],
                    c='#CCCCCC', s=3, alpha=0.3, rasterized=True, label='Core')
    ax_trans.scatter(umap_coords[transition_mask, 0], umap_coords[transition_mask, 1],
                    c='#E74C3C', s=5, alpha=0.6, rasterized=True, label='Transition')
    ax_trans.legend(loc='upper right', fontsize=9, markerscale=2)
    ax_trans.set_xlabel('UMAP 1')
    ax_trans.set_ylabel('UMAP 2')
    ax_trans.set_title(f'E. Transition Zone ({transition_mask.sum():,} cells)', fontsize=11)
    ax_trans.set_xticks([])
    ax_trans.set_yticks([])

    # F: Stage progression statistics
    ax_stats = fig.add_subplot(gs[1, 3])
    # Compute consecutive stage distances
    consecutive_dist = []
    all_dist = []
    for i in range(len(STAGE_ORDER) - 1):
        consecutive_dist.append(dist_matrix[i, i+1])
    for i in range(len(STAGE_ORDER)):
        for j in range(i+1, len(STAGE_ORDER)):
            all_dist.append(dist_matrix[i, j])

    ax_stats.bar(['Adjacent\nStages', 'All\nPairs'],
                [np.mean(consecutive_dist), np.mean(all_dist)],
                yerr=[np.std(consecutive_dist), np.std(all_dist)],
                color=['#1B4F72', '#922B21'], edgecolor='white', linewidth=1.5, capsize=5)
    ax_stats.set_ylabel('Mean Embedding Distance')
    ax_stats.set_title('F. Stage Distance Statistics', fontsize=11)

    plt.suptitle('Disease Progression and Trajectory Analysis',
                fontsize=15, fontweight='bold', y=1.01)

    fig.savefig(output_dir / "fig3_stage_transitions.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig3_stage_transitions.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig3_stage_transitions.png/pdf")


# =============================================================================
# FIGURE 4: REFERENCE COMPARISON
# =============================================================================

def figure4_reference_comparison(cells, output_dir):
    """Publication Figure 4: HLCA vs LuCA reference comparison."""
    print("\nGenerating Figure 4: Reference Comparison...")

    cells_s = sample_balanced(cells, n_per_stage=4000)

    hlca = get_embeddings(cells_s, "z_hlca_")
    luca = get_embeddings(cells_s, "z_luca_")
    fused = get_embeddings(cells_s, "z_fused_")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # A: HLCA PCA
    ax_hlca = fig.add_subplot(gs[0, 0])
    pca_hlca = PCA(n_components=2).fit_transform(hlca)
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_hlca.scatter(pca_hlca[mask, 0], pca_hlca[mask, 1],
                       c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)
    ax_hlca.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_hlca.set_xlabel('PC1')
    ax_hlca.set_ylabel('PC2')
    ax_hlca.set_title('A. HLCA Reference (30D)', fontsize=11)

    # B: LuCA PCA
    ax_luca = fig.add_subplot(gs[0, 1])
    pca_luca = PCA(n_components=2).fit_transform(luca)
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_luca.scatter(pca_luca[mask, 0], pca_luca[mask, 1],
                       c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)
    ax_luca.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_luca.set_xlabel('PC1')
    ax_luca.set_ylabel('PC2')
    ax_luca.set_title('B. LuCA Reference (10D)', fontsize=11)

    # C: Fused PCA
    ax_fused = fig.add_subplot(gs[0, 2])
    pca_fused = PCA(n_components=2).fit_transform(fused)
    for stage in STAGE_ORDER:
        mask = cells_s['stage'] == stage
        ax_fused.scatter(pca_fused[mask, 0], pca_fused[mask, 1],
                        c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)
    ax_fused.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_fused.set_xlabel('PC1')
    ax_fused.set_ylabel('PC2')
    ax_fused.set_title('C. Fused Embedding (40D)', fontsize=11)

    # D: Variance explained comparison
    ax_var = fig.add_subplot(gs[0, 3])
    pca_h = PCA(n_components=10).fit(hlca)
    pca_l = PCA(n_components=10).fit(luca)
    pca_f = PCA(n_components=10).fit(fused)

    x = np.arange(1, 11)
    ax_var.plot(x, np.cumsum(pca_h.explained_variance_ratio_) * 100,
               'o-', color='#1B4F72', label='HLCA', linewidth=2)
    ax_var.plot(x, np.cumsum(pca_l.explained_variance_ratio_) * 100,
               's-', color='#922B21', label='LuCA', linewidth=2)
    ax_var.plot(x, np.cumsum(pca_f.explained_variance_ratio_) * 100,
               '^-', color='#1D6F42', label='Fused', linewidth=2)
    ax_var.set_xlabel('Number of PCs')
    ax_var.set_ylabel('Cumulative Variance Explained (%)')
    ax_var.set_title('D. Variance Explained', fontsize=11)
    ax_var.legend(loc='lower right', fontsize=9)
    ax_var.set_xlim(0.5, 10.5)
    ax_var.set_ylim(0, 100)

    # E: HLCA mean embedding
    ax_hlca_mean = fig.add_subplot(gs[1, 0])
    hlca_all = get_embeddings(cells, "z_hlca_")
    hlca_means = np.array([hlca_all[cells['stage'] == s].mean(axis=0) for s in STAGE_ORDER])
    im = ax_hlca_mean.imshow(hlca_means, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax_hlca_mean.set_yticks(range(5))
    ax_hlca_mean.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_hlca_mean.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_hlca_mean.set_xlabel('HLCA Dimension')
    ax_hlca_mean.set_title('E. HLCA Mean by Stage', fontsize=11)
    plt.colorbar(im, ax=ax_hlca_mean, shrink=0.7)

    # F: LuCA mean embedding
    ax_luca_mean = fig.add_subplot(gs[1, 1])
    luca_all = get_embeddings(cells, "z_luca_")
    luca_means = np.array([luca_all[cells['stage'] == s].mean(axis=0) for s in STAGE_ORDER])
    im2 = ax_luca_mean.imshow(luca_means, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax_luca_mean.set_yticks(range(5))
    ax_luca_mean.set_yticklabels(STAGE_ORDER, fontsize=9)
    for i, label in enumerate(ax_luca_mean.get_yticklabels()):
        label.set_color(STAGE_COLORS[STAGE_ORDER[i]])
        label.set_fontweight('bold')
    ax_luca_mean.set_xlabel('LuCA Dimension')
    ax_luca_mean.set_title('F. LuCA Mean by Stage', fontsize=11)
    plt.colorbar(im2, ax=ax_luca_mean, shrink=0.7)

    # G: Stage separability comparison
    ax_sep = fig.add_subplot(gs[1, 2])

    def compute_silhouette_approx(X, labels):
        """Approximate silhouette using centroids."""
        unique_labels = np.unique(labels)
        centroids = {l: X[labels == l].mean(axis=0) for l in unique_labels}
        scores = []
        for l in unique_labels:
            mask = labels == l
            intra = np.mean([np.linalg.norm(x - centroids[l]) for x in X[mask][:500]])
            inter = np.min([np.linalg.norm(centroids[l] - centroids[l2])
                           for l2 in unique_labels if l2 != l])
            scores.append((inter - intra) / max(inter, intra))
        return np.mean(scores)

    stages_arr = cells_s['stage'].values
    sil_hlca = compute_silhouette_approx(hlca, stages_arr)
    sil_luca = compute_silhouette_approx(luca, stages_arr)
    sil_fused = compute_silhouette_approx(fused, stages_arr)

    bars = ax_sep.bar(['HLCA', 'LuCA', 'Fused'], [sil_hlca, sil_luca, sil_fused],
                     color=['#1B4F72', '#922B21', '#1D6F42'], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, [sil_hlca, sil_luca, sil_fused]):
        ax_sep.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_sep.set_ylabel('Stage Separability Score')
    ax_sep.set_title('G. Stage Discriminability', fontsize=11)
    ax_sep.set_ylim(0, max(sil_hlca, sil_luca, sil_fused) * 1.2)

    # H: Reference correlation
    ax_corr = fig.add_subplot(gs[1, 3])
    # Compute correlation between HLCA and LuCA PC1s
    hlca_pc1 = PCA(n_components=1).fit_transform(hlca).flatten()
    luca_pc1 = PCA(n_components=1).fit_transform(luca).flatten()

    # Subsample for visualization
    idx = np.random.choice(len(hlca_pc1), min(5000, len(hlca_pc1)), replace=False)

    for stage in STAGE_ORDER:
        mask = cells_s['stage'].values[idx] == stage
        ax_corr.scatter(hlca_pc1[idx][mask], luca_pc1[idx][mask],
                       c=STAGE_COLORS[stage], s=5, alpha=0.5, label=stage, rasterized=True)

    r, p = stats.pearsonr(hlca_pc1, luca_pc1)
    ax_corr.set_xlabel('HLCA PC1')
    ax_corr.set_ylabel('LuCA PC1')
    ax_corr.set_title(f'H. Reference Correlation (r={r:.3f})', fontsize=11)
    ax_corr.legend(loc='upper left', fontsize=8, markerscale=2)

    plt.suptitle('Dual-Reference Embedding Comparison: HLCA vs LuCA',
                fontsize=15, fontweight='bold', y=1.01)

    fig.savefig(output_dir / "fig4_reference_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / "fig4_reference_comparison.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig4_reference_comparison.png/pdf")


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
    print("Generating Publication-Quality Embedding Figures")
    print("=" * 60)

    cells = load_data(args.data_dir)

    print(f"\nData: {len(cells):,} cells, {cells['donor_id'].nunique()} donors")
    print(f"Stages: {dict(cells['stage'].value_counts())}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure1_embedding_overview(cells, args.output_dir)
    figure2_biological_features(cells, args.output_dir)
    figure2b_clonal_evolution(cells, args.output_dir)
    figure3_stage_transitions(cells, args.output_dir)
    figure4_reference_comparison(cells, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
