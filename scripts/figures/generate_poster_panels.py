#!/usr/bin/env python3
"""
Generate individual poster panels with proper scaling and annotations.

Outputs to: figures/poster/panels/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
import torch
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/home/booka/projects/StageBridge/data')
FIG_DIR = Path('/home/booka/projects/StageBridge/figures/poster/panels')
FIG_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path('/home/booka/projects/StageBridge/results/v1/full/fold_0_seed_44_best_checkpoint.pt')

# Stage colors
STAGE_COLORS = {'Normal': '#228B22', 'Preinvasive': '#4169E1', 'Invasive': '#8B1A1A'}
STAGE_ORDER = ['Normal', 'Preinvasive', 'Invasive']

# Cell type colors (tab20)
CT_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# Publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def save_panel(fig, name, dpi=300):
    for fmt in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')


def load_data():
    print('Loading data...')
    cells = pd.read_parquet(DATA_DIR / 'cells.parquet')
    cells['stage_3'] = cells['stage']

    # Capitalize 'mixed' for consistency
    if 'cell_type' in cells.columns:
        cells['cell_type'] = cells['cell_type'].replace({'mixed': 'Mixed'})

    print(f'  Cells: {len(cells):,}')
    return cells


# =============================================================================
# UMAP PANELS
# =============================================================================
def panel_umap_by_stage(cells, n_sample=50000):
    """UMAP colored by stage."""
    print('Generating: UMAP by stage...')

    z_cols = [f'z_fused_{i}' for i in range(40)]
    if not all(c in cells.columns for c in z_cols):
        print('  SKIP: z_fused not found')
        return

    sample = cells.sample(min(n_sample, len(cells)), random_state=42)
    Z = sample[z_cols].values

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        emb = reducer.fit_transform(Z)
        sample['umap_1'] = emb[:, 0]
        sample['umap_2'] = emb[:, 1]
    except ImportError:
        print('  SKIP: umap not installed')
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        ax.scatter(sample.loc[mask, 'umap_1'], sample.loc[mask, 'umap_2'],
                  c=STAGE_COLORS[stage], label=stage, s=3, alpha=0.5, rasterized=True)

    ax.legend(markerscale=4, frameon=False)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embedding Space by Stage')

    save_panel(fig, 'umap_stage')
    return sample  # Return for reuse


def panel_umap_by_celltype(cells, sample_with_umap=None, n_sample=50000):
    """UMAP colored by cell type."""
    print('Generating: UMAP by cell type...')

    ct_col = 'cell_type'
    if ct_col not in cells.columns:
        print('  SKIP: cell_type not found')
        return

    z_cols = [f'z_fused_{i}' for i in range(40)]

    if sample_with_umap is not None and 'umap_1' in sample_with_umap.columns:
        sample = sample_with_umap
    else:
        sample = cells.sample(min(n_sample, len(cells)), random_state=42)
        Z = sample[z_cols].values
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
            emb = reducer.fit_transform(Z)
            sample['umap_1'] = emb[:, 0]
            sample['umap_2'] = emb[:, 1]
        except ImportError:
            print('  SKIP: umap not installed')
            return

    fig, ax = plt.subplots(figsize=(10, 8))

    cell_types = [ct for ct in sample[ct_col].unique() if ct is not None and pd.notna(ct)]
    cell_types = sorted(cell_types)
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))

    for i, ct in enumerate(cell_types):
        mask = sample[ct_col] == ct
        ax.scatter(sample.loc[mask, 'umap_1'], sample.loc[mask, 'umap_2'],
                  c=[colors[i]], label=ct, s=3, alpha=0.5, rasterized=True)

    ax.legend(markerscale=4, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Embedding Space by Cell Type')

    save_panel(fig, 'umap_celltype')


def panel_umap_by_il1b(cells, sample_with_umap=None, n_sample=50000):
    """UMAP colored by IL1B expression."""
    print('Generating: UMAP by IL1B...')

    if 'il1b_raw' not in cells.columns:
        print('  SKIP: il1b_raw not found')
        return

    z_cols = [f'z_fused_{i}' for i in range(40)]

    if sample_with_umap is not None and 'umap_1' in sample_with_umap.columns:
        sample = sample_with_umap
    else:
        sample = cells.sample(min(n_sample, len(cells)), random_state=42)
        Z = sample[z_cols].values
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
            emb = reducer.fit_transform(Z)
            sample['umap_1'] = emb[:, 0]
            sample['umap_2'] = emb[:, 1]
        except ImportError:
            print('  SKIP: umap not installed')
            return

    fig, ax = plt.subplots(figsize=(8, 7))

    vmin = sample['il1b_raw'].quantile(0.01)
    vmax = sample['il1b_raw'].quantile(0.99)

    scatter = ax.scatter(sample['umap_1'], sample['umap_2'],
                        c=sample['il1b_raw'], cmap='Reds', s=3, alpha=0.6,
                        vmin=vmin, vmax=vmax, rasterized=True)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('IL1B Expression')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('IL1B Expression in Embedding Space')

    save_panel(fig, 'umap_il1b')


# =============================================================================
# SPATIAL PANELS
# =============================================================================
def panel_spatial_by_stage(cells, donor_id=None):
    """Spatial plot colored by stage for one donor."""
    print('Generating: Spatial by stage...')

    if 'x_spatial' not in cells.columns:
        print('  SKIP: no spatial coords')
        return

    if donor_id is None:
        donor_id = cells.groupby('donor_id').size().idxmax()

    donor_cells = cells[cells['donor_id'] == donor_id]

    fig, ax = plt.subplots(figsize=(8, 8))

    for stage in STAGE_ORDER:
        mask = donor_cells['stage_3'] == stage
        ax.scatter(donor_cells.loc[mask, 'x_spatial'],
                  donor_cells.loc[mask, 'y_spatial'],
                  c=STAGE_COLORS[stage], label=stage, s=2, alpha=0.6, rasterized=True)

    ax.legend(markerscale=5, frameon=False, loc='upper right')
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Spatial Distribution by Stage\n(Donor: {donor_id}, n={len(donor_cells):,})')
    ax.set_aspect('equal')

    save_panel(fig, 'spatial_stage')
    return donor_id


def panel_spatial_by_celltype(cells, donor_id=None):
    """Spatial plot colored by cell type for one donor."""
    print('Generating: Spatial by cell type...')

    if 'x_spatial' not in cells.columns or 'cell_type' not in cells.columns:
        print('  SKIP: required columns not found')
        return

    if donor_id is None:
        donor_id = cells.groupby('donor_id').size().idxmax()

    donor_cells = cells[cells['donor_id'] == donor_id]
    cell_types = [ct for ct in donor_cells['cell_type'].unique() if ct is not None and pd.notna(ct)]
    cell_types = sorted(cell_types)
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, ct in enumerate(cell_types):
        mask = donor_cells['cell_type'] == ct
        ax.scatter(donor_cells.loc[mask, 'x_spatial'],
                  donor_cells.loc[mask, 'y_spatial'],
                  c=[colors[i]], label=ct, s=2, alpha=0.6, rasterized=True)

    ax.legend(markerscale=5, frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9)
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Spatial Distribution by Cell Type\n(Donor: {donor_id})')
    ax.set_aspect('equal')

    save_panel(fig, 'spatial_celltype')


def panel_spatial_by_il1b(cells, donor_id=None):
    """Spatial plot colored by IL1B for one donor."""
    print('Generating: Spatial by IL1B...')

    if 'x_spatial' not in cells.columns or 'il1b_raw' not in cells.columns:
        print('  SKIP: required columns not found')
        return

    if donor_id is None:
        donor_id = cells.groupby('donor_id').size().idxmax()

    donor_cells = cells[cells['donor_id'] == donor_id]

    fig, ax = plt.subplots(figsize=(8, 8))

    vmin = donor_cells['il1b_raw'].quantile(0.01)
    vmax = donor_cells['il1b_raw'].quantile(0.99)

    scatter = ax.scatter(donor_cells['x_spatial'], donor_cells['y_spatial'],
                        c=donor_cells['il1b_raw'], cmap='Reds', s=2, alpha=0.7,
                        vmin=vmin, vmax=vmax, rasterized=True)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('IL1B Expression')
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'IL1B Spatial Distribution\n(Donor: {donor_id})')
    ax.set_aspect('equal')

    save_panel(fig, 'spatial_il1b')


# =============================================================================
# IL1B PANELS
# =============================================================================
def pval_to_stars(pval):
    """Convert p-value to significance stars."""
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'


def panel_il1b_violin(cells):
    """IL1B violin plot by stage - violin + box + points overlay."""
    print('Generating: IL1B violin...')

    if 'il1b_raw' not in cells.columns:
        print('  SKIP: il1b_raw not found')
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Trim outliers for visualization (keep 1st-99th percentile)
    raw_data = cells[['stage_3', 'il1b_raw']].dropna().copy()
    q_low, q_high = raw_data['il1b_raw'].quantile([0.01, 0.99])
    plot_data = raw_data[(raw_data['il1b_raw'] >= q_low) & (raw_data['il1b_raw'] <= q_high)].copy()

    # Sample points for jitter (too many points otherwise)
    n_points = 200
    sampled = plot_data.groupby('stage_3').apply(
        lambda x: x.sample(min(n_points, len(x)), random_state=42)
    ).reset_index(drop=True)

    # Violin (light fill)
    sns.violinplot(data=plot_data, x='stage_3', y='il1b_raw', order=STAGE_ORDER,
                   palette=STAGE_COLORS, ax=ax, inner=None, cut=0, alpha=0.4)

    # Box plot overlay (darker, narrower)
    sns.boxplot(data=plot_data, x='stage_3', y='il1b_raw', order=STAGE_ORDER,
                ax=ax, width=0.2, showcaps=True, boxprops={'alpha': 0.8},
                whiskerprops={'linewidth': 1.5}, medianprops={'color': 'black', 'linewidth': 2},
                fliersize=0, palette=STAGE_COLORS)

    # Jittered points
    sns.stripplot(data=sampled, x='stage_3', y='il1b_raw', order=STAGE_ORDER,
                  ax=ax, alpha=0.5, size=3, jitter=0.15, palette=STAGE_COLORS,
                  edgecolor='gray', linewidth=0.5)

    # Significance test (use full data, not trimmed)
    normal = cells[cells['stage_3'] == 'Normal']['il1b_raw'].dropna()
    invasive = cells[cells['stage_3'] == 'Invasive']['il1b_raw'].dropna()
    stat, pval = stats.mannwhitneyu(normal, invasive, alternative='less')
    stars = pval_to_stars(pval)

    # Y limits based on trimmed data
    data_min, data_max = q_low, q_high
    data_range = data_max - data_min
    y_bottom = data_min - 0.15 * data_range
    y_top = data_max + 0.20 * data_range

    ax.set_ylim(y_bottom, y_top)

    # Sample size labels (report full counts)
    counts = raw_data.groupby('stage_3').size()
    for i, stage in enumerate(STAGE_ORDER):
        n = counts.get(stage, 0)
        ax.text(i, y_bottom + 0.02 * data_range, f'(n={n:,})', ha='center', va='top', fontsize=10)

    ax.set_xlabel('')
    ax.set_ylabel('IL1B Expression')
    ax.set_title('IL1B Expression by Stage')

    # Draw bracket between Normal (0) and Invasive (2)
    bracket_y = data_max + 0.08 * data_range
    ax.plot([0, 0, 2, 2], [bracket_y - 0.02 * data_range, bracket_y, bracket_y, bracket_y - 0.02 * data_range],
            'k-', linewidth=1.5)
    ax.text(1, bracket_y + 0.02 * data_range, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    save_panel(fig, 'il1b_violin')




# =============================================================================
# CELL TYPE PANELS
# =============================================================================
def panel_celltype_composition(cells):
    """Cell type composition stacked bar by stage."""
    print('Generating: Cell type composition...')

    ct_col = 'cell_type'
    if ct_col not in cells.columns:
        print('  SKIP: cell_type not found')
        return

    # Compute proportions
    props = cells.groupby(['stage_3', ct_col]).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0)
    props = props.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6))

    props.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.7)

    ax.set_ylabel('Proportion')
    ax.set_xlabel('')
    ax.set_xticklabels(STAGE_ORDER, rotation=0)
    ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_title('Cell Type Composition by Stage')

    save_panel(fig, 'celltype_composition')


def panel_celltype_enrichment(cells):
    """Cell type log2 fold change vs Normal."""
    print('Generating: Cell type enrichment...')

    ct_col = 'cell_type'
    if ct_col not in cells.columns:
        print('  SKIP: cell_type not found')
        return

    # Compute proportions per stage
    props = cells.groupby(['stage_3', ct_col]).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0)
    props = props.reindex(STAGE_ORDER)

    normal_props = props.loc['Normal']

    fc_data = []
    for stage in ['Preinvasive', 'Invasive']:
        stage_props = props.loc[stage]
        log2fc = np.log2((stage_props + 1e-6) / (normal_props + 1e-6))
        for ct, fc in log2fc.items():
            fc_data.append({'Cell Type': ct, 'Stage': stage, 'Log2FC': fc})

    fc_df = pd.DataFrame(fc_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    pivot = fc_df.pivot(index='Cell Type', columns='Stage', values='Log2FC')
    pivot = pivot[['Preinvasive', 'Invasive']]

    x = np.arange(len(pivot))
    width = 0.35

    ax.bar(x - width/2, pivot['Preinvasive'], width, label='Preinvasive',
           color=STAGE_COLORS['Preinvasive'], alpha=0.8)
    ax.bar(x + width/2, pivot['Invasive'], width, label='Invasive',
           color=STAGE_COLORS['Invasive'], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.set_ylabel('Log2 Fold Change vs Normal')
    ax.set_xlabel('')
    ax.legend(frameon=False)
    ax.set_title('Cell Type Enrichment by Stage')

    save_panel(fig, 'celltype_enrichment')


# =============================================================================
# DESTVI PANELS
# =============================================================================
def panel_destvi_heatmap(cells):
    """DestVI gamma heatmap by stage."""
    print('Generating: DestVI heatmap...')

    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if not all(c in cells.columns for c in gamma_cols):
        print('  SKIP: gamma columns not found')
        return

    gamma_by_stage = cells.groupby('stage_3')[gamma_cols].mean()
    gamma_by_stage = gamma_by_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(gamma_by_stage, cmap='viridis', annot=True, fmt='.3f', ax=ax,
                xticklabels=[f'CT{i}' for i in range(10)],
                cbar_kws={'label': 'Mean Gamma'})
    ax.set_ylabel('')
    ax.set_title('DestVI Cell Type Proportions by Stage')

    save_panel(fig, 'destvi_heatmap')


def panel_niche_entropy(cells):
    """Niche entropy (Shannon) by stage - violin + box + points."""
    print('Generating: Niche entropy...')

    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if not all(c in cells.columns for c in gamma_cols):
        print('  SKIP: gamma columns not found')
        return

    # Compute entropy
    gamma_matrix = cells[gamma_cols].values.astype(float)
    p = gamma_matrix / (gamma_matrix.sum(axis=1, keepdims=True) + 1e-10)
    p = np.clip(p, 1e-10, 1)
    cells['ct_entropy'] = -np.sum(p * np.log2(p), axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Trim outliers for visualization (keep 1st-99th percentile)
    raw_data = cells[['stage_3', 'ct_entropy']].dropna().copy()
    q_low, q_high = raw_data['ct_entropy'].quantile([0.01, 0.99])
    plot_data = raw_data[(raw_data['ct_entropy'] >= q_low) & (raw_data['ct_entropy'] <= q_high)].copy()

    # Sample for points
    n_points = 200
    sampled = plot_data.groupby('stage_3').apply(
        lambda x: x.sample(min(n_points, len(x)), random_state=42)
    ).reset_index(drop=True)

    # Violin
    sns.violinplot(data=plot_data, x='stage_3', y='ct_entropy', order=STAGE_ORDER,
                   palette=STAGE_COLORS, ax=ax, inner=None, cut=0, alpha=0.4)

    # Box
    sns.boxplot(data=plot_data, x='stage_3', y='ct_entropy', order=STAGE_ORDER,
                ax=ax, width=0.2, showcaps=True, boxprops={'alpha': 0.8},
                whiskerprops={'linewidth': 1.5}, medianprops={'color': 'black', 'linewidth': 2},
                fliersize=0, palette=STAGE_COLORS)

    # Points
    sns.stripplot(data=sampled, x='stage_3', y='ct_entropy', order=STAGE_ORDER,
                  ax=ax, alpha=0.5, size=3, jitter=0.15, palette=STAGE_COLORS,
                  edgecolor='gray', linewidth=0.5)

    # Stats - test Normal vs Invasive (use full data)
    normal = cells[cells['stage_3'] == 'Normal']['ct_entropy'].dropna()
    invasive = cells[cells['stage_3'] == 'Invasive']['ct_entropy'].dropna()
    stat, pval = stats.mannwhitneyu(normal, invasive)
    stars = pval_to_stars(pval)

    # Y limits based on trimmed data
    data_min, data_max = q_low, q_high
    data_range = data_max - data_min
    y_bottom = data_min - 0.15 * data_range
    y_top = data_max + 0.20 * data_range

    ax.set_ylim(y_bottom, y_top)

    # Sample sizes (report full counts)
    counts = raw_data.groupby('stage_3').size()
    for i, stage in enumerate(STAGE_ORDER):
        n = counts.get(stage, 0)
        ax.text(i, y_bottom + 0.02 * data_range, f'(n={n:,})', ha='center', va='top', fontsize=10)

    ax.set_xlabel('')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Niche Complexity by Stage')

    # Draw bracket between Normal (0) and Invasive (2)
    bracket_y = data_max + 0.08 * data_range
    ax.plot([0, 0, 2, 2], [bracket_y - 0.02 * data_range, bracket_y, bracket_y, bracket_y - 0.02 * data_range],
            'k-', linewidth=1.5)
    ax.text(1, bracket_y + 0.02 * data_range, stars, ha='center', va='bottom', fontsize=14, fontweight='bold')

    save_panel(fig, 'niche_entropy')


# =============================================================================
# MUTATION PANELS
# =============================================================================
def panel_mutation_prevalence(cells):
    """Mutation prevalence by stage."""
    print('Generating: Mutation prevalence...')

    mut_cols = ['kras_mut', 'egfr_mut', 'tp53_mut', 'stk11_mut', 'keap1_mut']
    available = [c for c in mut_cols if c in cells.columns]

    if not available:
        print('  SKIP: no mutation columns')
        return

    # Compute prevalence by stage
    prev_data = []
    for stage in STAGE_ORDER:
        stage_cells = cells[cells['stage_3'] == stage]
        for mut in available:
            prev = stage_cells[mut].mean()
            prev_data.append({
                'Stage': stage,
                'Mutation': mut.replace('_mut', '').upper(),
                'Prevalence': prev
            })

    prev_df = pd.DataFrame(prev_data)
    pivot = prev_df.pivot(index='Mutation', columns='Stage', values='Prevalence')
    pivot = pivot[STAGE_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(pivot))
    width = 0.25

    for i, stage in enumerate(STAGE_ORDER):
        ax.bar(x + i*width, pivot[stage], width, label=stage, color=STAGE_COLORS[stage])

    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel('Cell Prevalence')
    ax.set_xlabel('')
    ax.legend(frameon=False)
    ax.set_title('Driver Mutation Prevalence by Stage')

    save_panel(fig, 'mutation_prevalence')


# =============================================================================
# MODEL WEIGHT PANELS
# =============================================================================
def panel_drift_gate_weights(ckpt):
    """Drift head context gate input importance."""
    print('Generating: Drift gate weights...')

    if ckpt is None:
        print('  SKIP: no checkpoint')
        return

    state = ckpt['model_state_dict']
    gate_w = state['drift_head.context_gate.0.weight'].numpy()  # [256, 544]

    input_importance = np.abs(gate_w).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(range(len(input_importance)), input_importance, alpha=0.7, width=1.0)
    ax.axvline(x=256, color='red', linestyle='--', linewidth=1.5, label='Receiver | Context')
    ax.set_xlabel('Input Dimension')
    ax.set_ylabel('Mean |Weight|')
    ax.set_title('Drift Head: Context Gate Input Importance')
    ax.legend(frameon=False)
    ax.set_xlim(0, len(input_importance))

    save_panel(fig, 'drift_gate_weights')


def panel_stage_embedding(ckpt):
    """Stage embedding similarity matrix."""
    print('Generating: Stage embedding...')

    if ckpt is None:
        print('  SKIP: no checkpoint')
        return

    state = ckpt['model_state_dict']
    stage_emb = state['stage_embedding.weight'].numpy()  # [9, 32]

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(stage_emb)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xlabel('Stage Index')
    ax.set_ylabel('Stage Index')
    ax.set_title('Stage Embedding Cosine Similarity')
    plt.colorbar(im, ax=ax, shrink=0.8)

    save_panel(fig, 'stage_embedding')


def panel_ring_pooler_variance(ckpt):
    """Ring pooler inducing point variance."""
    print('Generating: Ring pooler variance...')

    if ckpt is None:
        print('  SKIP: no checkpoint')
        return

    state = ckpt['model_state_dict']

    ring_vars = []
    ring_labels = []
    for i in range(4):
        key = f'niche_tokenizer.ring_poolers.{i}.isab.inducing_points'
        if key in state:
            pts = state[key].numpy()
            ring_vars.append(pts.std())
            ring_labels.append(f'Ring {i+1}')

    fig, ax = plt.subplots(figsize=(5, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax.bar(ring_labels, ring_vars, color=colors[:len(ring_labels)])
    ax.set_ylabel('Inducing Point Std Dev')
    ax.set_title('Spatial Ring Representation Variability')

    save_panel(fig, 'ring_pooler_variance')


# =============================================================================
# TRAJECTORY PANEL
# =============================================================================
def panel_trajectory_projection(cells):
    """Projection onto Normal->Invasive axis."""
    print('Generating: Trajectory projection...')

    z_cols = [f'z_fused_{i}' for i in range(40)]
    if not all(c in cells.columns for c in z_cols):
        print('  SKIP: z_fused not found')
        return

    sample = cells.sample(min(50000, len(cells)), random_state=42)
    Z = sample[z_cols].values

    # Compute centroids
    centroids = {}
    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        centroids[stage] = Z[mask].mean(axis=0)

    # Trajectory vector
    trajectory_vec = centroids['Invasive'] - centroids['Normal']
    trajectory_vec = trajectory_vec / np.linalg.norm(trajectory_vec)

    # Project
    projections = (Z - centroids['Normal']) @ trajectory_vec

    fig, ax = plt.subplots(figsize=(7, 5))

    for stage in STAGE_ORDER:
        mask = sample['stage_3'].values == stage
        ax.hist(projections[mask], bins=50, alpha=0.6, label=stage,
                color=STAGE_COLORS[stage], density=True)

    ax.set_xlabel('Projection onto Normal -> Invasive Axis')
    ax.set_ylabel('Density')
    ax.set_title('Cell Distribution Along Progression Trajectory')
    ax.legend(frameon=False)

    save_panel(fig, 'trajectory_projection')


def main():
    print('=' * 60)
    print('GENERATING INDIVIDUAL POSTER PANELS')
    print('=' * 60)
    print(f'Output: {FIG_DIR}')
    print()

    cells = load_data()

    # Load model
    ckpt = None
    if CHECKPOINT.exists():
        print('Loading model weights...')
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)

    print()

    # UMAP panels (reuse embedding)
    sample_with_umap = panel_umap_by_stage(cells)
    panel_umap_by_celltype(cells, sample_with_umap)
    panel_umap_by_il1b(cells, sample_with_umap)

    # Spatial panels
    donor_id = panel_spatial_by_stage(cells)
    panel_spatial_by_celltype(cells, donor_id)
    panel_spatial_by_il1b(cells, donor_id)

    # IL1B panel
    panel_il1b_violin(cells)

    # Cell type panels
    panel_celltype_composition(cells)
    panel_celltype_enrichment(cells)

    # DestVI panels
    panel_destvi_heatmap(cells)
    panel_niche_entropy(cells)

    # Mutation panel
    panel_mutation_prevalence(cells)

    # Model weight panels
    panel_drift_gate_weights(ckpt)
    panel_stage_embedding(ckpt)
    panel_ring_pooler_variance(ckpt)

    # Trajectory panel
    panel_trajectory_projection(cells)

    print()
    print('=' * 60)
    print('DONE')
    print('=' * 60)
    print(f'Panels saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
