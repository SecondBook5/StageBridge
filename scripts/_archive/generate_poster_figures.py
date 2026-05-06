#!/usr/bin/env python3
"""
Generate poster figures from local parquets and model weights.

Outputs to: figures/poster/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import torch
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/home/booka/projects/StageBridge/data')
FIG_DIR = Path('/home/booka/projects/StageBridge/figures/poster')
FIG_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path('/home/booka/projects/StageBridge/results/v1/full/fold_0_seed_44_best_checkpoint.pt')

# Stage colors
STAGE_COLORS = {
    'Normal': '#228B22',
    'Preinvasive': '#4169E1',
    'Invasive': '#8B1A1A',
}
STAGE_ORDER = ['Normal', 'Preinvasive', 'Invasive']

def save_fig(fig, name, dpi=300):
    for fmt in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')


def load_data():
    """Load cells and neighborhoods parquets."""
    print('Loading data...')
    cells = pd.read_parquet(DATA_DIR / 'cells.parquet')
    print(f'  Cells: {len(cells):,}')

    # Use stage column directly (already 3-class: Normal, Preinvasive, Invasive)
    if 'stage' in cells.columns:
        cells['stage_3'] = cells['stage']

    return cells


def fig_cell_type_by_stage(cells):
    """Cell type composition by stage."""
    print('Generating: cell type by stage...')

    # Use luca_cell_type if available, else cell_type
    ct_col = 'luca_cell_type' if 'luca_cell_type' in cells.columns else 'cell_type'

    # Compute proportions
    props = cells.groupby(['stage_3', ct_col]).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0)
    props = props.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    props.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('')
    ax.set_xticklabels(STAGE_ORDER, rotation=0)
    ax.legend(title='Cell Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_title('Cell Type Composition by Stage')

    save_fig(fig, 'cell_type_by_stage')


def fig_il1b_by_stage(cells):
    """IL1B expression by stage."""
    print('Generating: IL1B by stage...')

    if 'il1b_raw' not in cells.columns:
        print('  SKIP: il1b_raw not in data')
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    data = [cells[cells['stage_3'] == s]['il1b_raw'].dropna() for s in STAGE_ORDER]
    parts = ax.violinplot(data, positions=range(len(STAGE_ORDER)), showmeans=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(STAGE_COLORS.values())[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('IL1B Expression (log-normalized)')
    ax.set_title('IL1B Expression Increases with Progression')

    # Stats
    normal = cells[cells['stage_3'] == 'Normal']['il1b_raw'].dropna()
    invasive = cells[cells['stage_3'] == 'Invasive']['il1b_raw'].dropna()
    if len(normal) > 0 and len(invasive) > 0:
        stat, pval = stats.mannwhitneyu(normal, invasive, alternative='less')
        fc = invasive.mean() / (normal.mean() + 1e-10)
        ax.text(0.95, 0.95, f'FC={fc:.2f}\np={pval:.2e}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    save_fig(fig, 'il1b_by_stage')


def fig_destvi_gammas(cells):
    """DestVI gamma (cell type proportion) heatmap by stage."""
    print('Generating: DestVI gammas by stage...')

    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if not all(c in cells.columns for c in gamma_cols):
        print('  SKIP: gamma columns not found')
        return

    # Mean gammas per stage
    gamma_by_stage = cells.groupby('stage_3')[gamma_cols].mean()
    gamma_by_stage = gamma_by_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(gamma_by_stage, cmap='viridis', annot=True, fmt='.3f', ax=ax)
    ax.set_ylabel('')
    ax.set_title('DestVI Cell Type Proportions by Stage')

    save_fig(fig, 'destvi_gammas_by_stage')


def fig_embedding_umap(cells, n_sample=50000):
    """UMAP of fused embeddings colored by stage."""
    print('Generating: embedding UMAP...')

    z_cols = [f'z_fused_{i}' for i in range(40)]
    if not all(c in cells.columns for c in z_cols):
        print('  SKIP: z_fused columns not found')
        return

    # Sample for speed
    if len(cells) > n_sample:
        sample = cells.sample(n_sample, random_state=42)
    else:
        sample = cells

    Z = sample[z_cols].values

    # Run UMAP
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        emb = reducer.fit_transform(Z)
    except ImportError:
        print('  SKIP: umap not installed')
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        ax.scatter(emb[mask, 0], emb[mask, 1],
                  c=STAGE_COLORS[stage], label=stage, s=1, alpha=0.5)

    ax.legend(markerscale=5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('Fused Embedding Space')
    ax.set_xticks([])
    ax.set_yticks([])

    save_fig(fig, 'embedding_umap')


def fig_mutation_by_stage(cells):
    """Mutation prevalence by stage."""
    print('Generating: mutations by stage...')

    mut_cols = ['kras_mut', 'egfr_mut', 'tp53_mut', 'stk11_mut', 'keap1_mut']
    available = [c for c in mut_cols if c in cells.columns]

    if not available:
        print('  SKIP: no mutation columns')
        return

    # Prevalence by stage (donor-level to avoid double counting)
    mut_data = []
    for stage in STAGE_ORDER:
        stage_cells = cells[cells['stage_3'] == stage]
        donors = stage_cells['donor_id'].unique()
        for mut in available:
            # Proportion of donors with mutation
            donor_mut = stage_cells.groupby('donor_id')[mut].max()
            prev = donor_mut.mean()
            mut_data.append({'stage': stage, 'mutation': mut.replace('_mut', '').upper(), 'prevalence': prev})

    mut_df = pd.DataFrame(mut_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = mut_df.pivot(index='mutation', columns='stage', values='prevalence')
    pivot = pivot[STAGE_ORDER]
    pivot.plot(kind='bar', ax=ax, color=[STAGE_COLORS[s] for s in STAGE_ORDER])
    ax.set_ylabel('Donor Prevalence')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Stage')
    ax.set_title('Driver Mutation Prevalence by Stage')

    save_fig(fig, 'mutations_by_stage')


def fig_pathway_heatmap(cells):
    """Pathway activity heatmap by stage."""
    print('Generating: pathway heatmap...')

    pw_cols = [f'pathway_raw_{i}' for i in range(14)]
    available = [c for c in pw_cols if c in cells.columns]

    if len(available) < 5:
        print('  SKIP: insufficient pathway columns')
        return

    # Mean pathway activity by stage
    pw_by_stage = cells.groupby('stage_3')[available].mean()
    pw_by_stage = pw_by_stage.reindex(STAGE_ORDER)

    # Z-score normalize across stages
    pw_z = (pw_by_stage - pw_by_stage.mean()) / (pw_by_stage.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pw_z.T, cmap='RdBu_r', center=0, ax=ax,
                yticklabels=[f'Pathway {i}' for i in range(len(available))])
    ax.set_title('Pathway Activity (z-scored) by Stage')

    save_fig(fig, 'pathway_heatmap')


def fig_spatial_stage_distribution(cells, n_donors=6):
    """Spatial distribution of cells by stage for select donors."""
    print('Generating: spatial stage distribution...')

    if 'x_spatial' not in cells.columns or 'y_spatial' not in cells.columns:
        print('  SKIP: no spatial coordinates')
        return

    # Select donors with most cells
    top_donors = cells['donor_id'].value_counts().head(n_donors).index

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, donor in enumerate(top_donors):
        ax = axes[i]
        donor_cells = cells[cells['donor_id'] == donor]

        for stage in STAGE_ORDER:
            mask = donor_cells['stage_3'] == stage
            ax.scatter(donor_cells.loc[mask, 'x_spatial'],
                      donor_cells.loc[mask, 'y_spatial'],
                      c=STAGE_COLORS[stage], label=stage, s=1, alpha=0.5)

        ax.set_title(f'{donor} (n={len(donor_cells):,})')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(markerscale=5, loc='upper left')

    plt.suptitle('Spatial Distribution by Stage', fontsize=14)
    plt.tight_layout()
    save_fig(fig, 'spatial_stage_distribution')


def fig_model_attention_weights(checkpoint_path):
    """Analyze model attention weights if checkpoint available."""
    print('Generating: model weight analysis...')

    if not checkpoint_path.exists():
        print(f'  SKIP: checkpoint not found at {checkpoint_path}')
        return

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    # Find attention weights
    attn_keys = [k for k in state_dict.keys() if 'attn' in k.lower() and 'weight' in k.lower()]

    if not attn_keys:
        print('  SKIP: no attention weights found')
        return

    # Analyze weight magnitudes
    weight_stats = []
    for key in attn_keys[:10]:  # First 10
        w = state_dict[key]
        weight_stats.append({
            'layer': key.split('.')[1] if '.' in key else key,
            'mean': w.abs().mean().item(),
            'std': w.std().item(),
            'shape': str(list(w.shape)),
        })

    df = pd.DataFrame(weight_stats)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(df)), df['mean'], xerr=df['std'], capsize=3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['layer'])
    ax.set_xlabel('Mean |Weight|')
    ax.set_title('Attention Weight Magnitudes')

    save_fig(fig, 'model_attention_weights')


def fig_donor_summary(cells):
    """Summary statistics per donor - cells by stage within each donor."""
    print('Generating: donor summary...')

    # Count cells per donor per stage
    donor_stage = cells.groupby(['donor_id', 'stage_3']).size().unstack(fill_value=0)
    donor_stage = donor_stage.reindex(columns=STAGE_ORDER, fill_value=0)
    donor_stage['total'] = donor_stage.sum(axis=1)
    donor_stage = donor_stage.sort_values('total', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stacked bar: cells per donor by stage
    ax = axes[0]
    bottom = np.zeros(len(donor_stage))
    for stage in STAGE_ORDER:
        ax.bar(range(len(donor_stage)), donor_stage[stage], bottom=bottom,
               label=stage, color=STAGE_COLORS[stage], alpha=0.8)
        bottom += donor_stage[stage].values
    ax.set_xlabel('Donor')
    ax.set_ylabel('Number of Cells')
    ax.set_title(f'Cells per Donor by Stage (n={len(donor_stage)} donors)')
    ax.legend()
    ax.set_xticks([])

    # Overall cell stage distribution (pie)
    ax = axes[1]
    stage_counts = cells['stage_3'].value_counts().reindex(STAGE_ORDER)
    ax.pie(stage_counts, labels=STAGE_ORDER, colors=[STAGE_COLORS[s] for s in STAGE_ORDER],
           autopct='%1.1f%%')
    ax.set_title(f'Cell Stage Distribution (n={len(cells):,})')

    plt.tight_layout()
    save_fig(fig, 'donor_summary')


def main():
    print('=' * 60)
    print('GENERATING POSTER FIGURES')
    print('=' * 60)
    print(f'Output: {FIG_DIR}')
    print()

    cells = load_data()
    print()

    # Generate all figures
    fig_cell_type_by_stage(cells)
    fig_il1b_by_stage(cells)
    fig_destvi_gammas(cells)
    fig_embedding_umap(cells)
    fig_mutation_by_stage(cells)
    fig_pathway_heatmap(cells)
    fig_spatial_stage_distribution(cells)
    fig_model_attention_weights(CHECKPOINT)
    fig_donor_summary(cells)

    print()
    print('=' * 60)
    print('DONE')
    print('=' * 60)
    print(f'Figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
