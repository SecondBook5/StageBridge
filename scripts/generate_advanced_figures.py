#!/usr/bin/env python3
"""
Advanced poster figures using model weights, embeddings, and DestVI outputs.

Outputs to: figures/poster/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/home/booka/projects/StageBridge/data')
FIG_DIR = Path('/home/booka/projects/StageBridge/figures/poster')
FIG_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT = Path('/home/booka/projects/StageBridge/results/v1/full/fold_0_seed_44_best_checkpoint.pt')

# Stage colors
STAGE_COLORS = {'Normal': '#228B22', 'Preinvasive': '#4169E1', 'Invasive': '#8B1A1A'}
STAGE_ORDER = ['Normal', 'Preinvasive', 'Invasive']

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})


def save_fig(fig, name, dpi=300):
    for fmt in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {name}')


def load_data():
    """Load cells parquet."""
    print('Loading data...')
    cells = pd.read_parquet(DATA_DIR / 'cells.parquet')
    cells['stage_3'] = cells['stage']
    print(f'  Cells: {len(cells):,}')
    return cells


def load_model_weights():
    """Load model checkpoint."""
    print('Loading model weights...')
    if not CHECKPOINT.exists():
        print(f'  SKIP: checkpoint not found')
        return None
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    return ckpt


# =============================================================================
# FIGURE 1: Drift Head Analysis - How context gates velocity prediction
# =============================================================================
def fig_drift_head_analysis(ckpt):
    """Analyze drift head weights - context gating mechanism."""
    print('Generating: drift head analysis...')

    if ckpt is None:
        print('  SKIP: no checkpoint')
        return

    state = ckpt['model_state_dict']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Context gate input weights (what features it looks at)
    ax = axes[0, 0]
    gate_w = state['drift_head.context_gate.0.weight'].numpy()  # [256, 544]
    # Average absolute weight per input dimension
    input_importance = np.abs(gate_w).mean(axis=0)
    ax.bar(range(len(input_importance)), input_importance, alpha=0.7)
    ax.set_xlabel('Input Dimension (receiver + context)')
    ax.set_ylabel('Mean |Weight|')
    ax.set_title('Drift Head: Context Gate Input Importance')
    ax.axvline(x=256, color='red', linestyle='--', label='receiver|context boundary')
    ax.legend()

    # 2. Latent-only pathway weights
    ax = axes[0, 1]
    latent_w = state['drift_head.latent_only.0.weight'].numpy()  # [256, 104]
    latent_importance = np.abs(latent_w).mean(axis=0)
    ax.bar(range(len(latent_importance)), latent_importance, alpha=0.7, color='orange')
    ax.set_xlabel('Input Dimension (fused + hlca + luca + stats)')
    ax.set_ylabel('Mean |Weight|')
    ax.set_title('Drift Head: Latent-Only Pathway Input Importance')
    # Annotate sections
    ax.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=70, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5)
    ax.text(20, ax.get_ylim()[1]*0.9, 'z_fused', ha='center', fontsize=9)
    ax.text(55, ax.get_ylim()[1]*0.9, 'z_hlca', ha='center', fontsize=9)
    ax.text(75, ax.get_ylim()[1]*0.9, 'z_luca', ha='center', fontsize=9)
    ax.text(92, ax.get_ylim()[1]*0.9, 'stats', ha='center', fontsize=9)

    # 3. Stage embedding similarity
    ax = axes[1, 0]
    stage_emb = state['stage_embedding.weight'].numpy()  # [9, 32]
    # Compute cosine similarity between stage embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(stage_emb)
    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_title('Stage Embedding Cosine Similarity')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 4. Ring pooler inducing point variance (which rings are most variable)
    ax = axes[1, 1]
    ring_vars = []
    for i in range(4):
        key = f'niche_tokenizer.ring_poolers.{i}.isab.inducing_points'
        if key in state:
            pts = state[key].numpy()  # [1, 4, 256]
            ring_vars.append(pts.std())
    ax.bar(['Ring 1', 'Ring 2', 'Ring 3', 'Ring 4'], ring_vars, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Inducing Point Std Dev')
    ax.set_title('Spatial Ring Pooler Variability')

    plt.tight_layout()
    save_fig(fig, 'drift_head_analysis')


# =============================================================================
# FIGURE 2: DestVI Gamma Analysis - Cell type deconvolution by stage
# =============================================================================
def fig_destvi_advanced(cells):
    """Advanced DestVI gamma analysis with clustering."""
    print('Generating: advanced DestVI analysis...')

    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if not all(c in cells.columns for c in gamma_cols):
        print('  SKIP: gamma columns not found')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Hierarchical clustering of cell type proportions
    ax = axes[0, 0]
    gamma_by_stage = cells.groupby('stage_3')[gamma_cols].mean()
    gamma_by_stage = gamma_by_stage.reindex(STAGE_ORDER)

    # Cluster columns (cell types)
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist

    gamma_vals = gamma_by_stage.values.T  # [10, 3] - cell types x stages
    Z = linkage(gamma_vals, method='ward')
    order = leaves_list(Z)

    gamma_ordered = gamma_by_stage.iloc[:, order]
    sns.heatmap(gamma_ordered.T, cmap='viridis', annot=True, fmt='.3f', ax=ax,
                xticklabels=STAGE_ORDER, yticklabels=[f'CT{i}' for i in order])
    ax.set_title('DestVI Cell Type Proportions (clustered)')

    # 2. Stage-specific cell type enrichment (log2 fold change vs Normal)
    ax = axes[0, 1]
    normal_gamma = gamma_by_stage.loc['Normal']
    fc_data = []
    for stage in ['Preinvasive', 'Invasive']:
        stage_gamma = gamma_by_stage.loc[stage]
        log2fc = np.log2((stage_gamma + 1e-6) / (normal_gamma + 1e-6))
        for i, fc in enumerate(log2fc):
            fc_data.append({'Cell Type': f'CT{i}', 'Stage': stage, 'Log2FC': fc})

    fc_df = pd.DataFrame(fc_data)
    pivot = fc_df.pivot(index='Cell Type', columns='Stage', values='Log2FC')

    colors = [STAGE_COLORS['Preinvasive'], STAGE_COLORS['Invasive']]
    pivot.plot(kind='bar', ax=ax, color=colors, width=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Log2 Fold Change vs Normal')
    ax.set_xlabel('')
    ax.set_title('Cell Type Enrichment by Stage')
    ax.legend(title='Stage')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 3. Cell type diversity (Shannon entropy) by stage
    ax = axes[1, 0]
    def compute_entropy(gamma_matrix):
        """Compute Shannon entropy for each row."""
        p = gamma_matrix / (gamma_matrix.sum(axis=1, keepdims=True) + 1e-10)
        p = np.clip(p, 1e-10, 1)  # Avoid log(0)
        return -np.sum(p * np.log2(p), axis=1)

    gamma_matrix = cells[gamma_cols].values.astype(float)
    cells['ct_entropy'] = compute_entropy(gamma_matrix)

    entropy_data = [cells[cells['stage_3'] == s]['ct_entropy'].dropna() for s in STAGE_ORDER]
    parts = ax.violinplot(entropy_data, positions=range(len(STAGE_ORDER)), showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(STAGE_COLORS.values())[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel('Shannon Entropy (cell type diversity)')
    ax.set_title('Niche Complexity Increases with Progression')

    # Stats
    stat, pval = stats.kruskal(*entropy_data)
    ax.text(0.95, 0.95, f'Kruskal p={pval:.2e}', transform=ax.transAxes, ha='right', va='top')

    # 4. PCA of gamma profiles colored by stage
    ax = axes[1, 1]
    gamma_matrix = cells[gamma_cols].values
    pca = PCA(n_components=2)
    # Sample for speed
    n_sample = min(50000, len(cells))
    idx = np.random.choice(len(cells), n_sample, replace=False)
    pca_emb = pca.fit_transform(gamma_matrix[idx])

    for stage in STAGE_ORDER:
        mask = cells.iloc[idx]['stage_3'].values == stage
        ax.scatter(pca_emb[mask, 0], pca_emb[mask, 1], c=STAGE_COLORS[stage],
                  label=stage, s=1, alpha=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('DestVI Gamma PCA by Stage')
    ax.legend(markerscale=5)

    plt.tight_layout()
    save_fig(fig, 'destvi_advanced')


# =============================================================================
# FIGURE 3: Embedding Space Analysis - Drift vectors and stage transitions
# =============================================================================
def fig_embedding_analysis(cells):
    """Analyze embedding space structure and stage transitions."""
    print('Generating: embedding space analysis...')

    z_cols = [f'z_fused_{i}' for i in range(40)]
    if not all(c in cells.columns for c in z_cols):
        print('  SKIP: z_fused columns not found')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sample for computation
    n_sample = 50000
    sample = cells.sample(min(n_sample, len(cells)), random_state=42)
    Z = sample[z_cols].values

    # 1. Stage centroids and inter-stage distances
    ax = axes[0, 0]
    centroids = {}
    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        centroids[stage] = Z[mask].mean(axis=0)

    # Compute pairwise distances
    dist_matrix = np.zeros((3, 3))
    for i, s1 in enumerate(STAGE_ORDER):
        for j, s2 in enumerate(STAGE_ORDER):
            dist_matrix[i, j] = np.linalg.norm(centroids[s1] - centroids[s2])

    im = ax.imshow(dist_matrix, cmap='YlOrRd')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_yticklabels(STAGE_ORDER)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{dist_matrix[i, j]:.2f}', ha='center', va='center', fontsize=11)
    ax.set_title('Stage Centroid Distances in Embedding Space')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Pseudotrajectory: project onto Normal→Invasive axis
    ax = axes[0, 1]
    trajectory_vec = centroids['Invasive'] - centroids['Normal']
    trajectory_vec = trajectory_vec / np.linalg.norm(trajectory_vec)

    # Project all cells
    projections = (Z - centroids['Normal']) @ trajectory_vec
    sample['trajectory_proj'] = projections

    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        proj_vals = projections[mask]
        ax.hist(proj_vals, bins=50, alpha=0.6, label=stage, color=STAGE_COLORS[stage], density=True)

    ax.set_xlabel('Projection onto Normal→Invasive axis')
    ax.set_ylabel('Density')
    ax.set_title('Cell Distribution Along Progression Trajectory')
    ax.legend()

    # 3. Within-stage variance (embedding spread)
    ax = axes[1, 0]
    variances = []
    for stage in STAGE_ORDER:
        mask = sample['stage_3'] == stage
        stage_z = Z[mask]
        # Total variance = trace of covariance
        var = np.var(stage_z, axis=0).sum()
        variances.append(var)

    bars = ax.bar(STAGE_ORDER, variances, color=[STAGE_COLORS[s] for s in STAGE_ORDER])
    ax.set_ylabel('Total Embedding Variance')
    ax.set_title('Stage Heterogeneity in Embedding Space')

    # 4. t-SNE colored by IL1B expression
    ax = axes[1, 1]
    if 'il1b_raw' in sample.columns:
        # Subsample further for t-SNE
        tsne_n = min(10000, len(sample))
        tsne_idx = np.random.choice(len(sample), tsne_n, replace=False)
        tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Z[tsne_idx])

        il1b_vals = sample.iloc[tsne_idx]['il1b_raw'].values
        scatter = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=il1b_vals,
                           cmap='Reds', s=2, alpha=0.6, vmin=0, vmax=np.percentile(il1b_vals, 95))
        plt.colorbar(scatter, ax=ax, label='IL1B Expression')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Embedding t-SNE Colored by IL1B')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, 'IL1B not available', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_fig(fig, 'embedding_analysis')


# =============================================================================
# FIGURE 4: Mutation-Niche Interaction
# =============================================================================
def fig_mutation_niche(cells):
    """Analyze mutation-niche interactions."""
    print('Generating: mutation-niche analysis...')

    mut_cols = ['kras_mut', 'egfr_mut', 'tp53_mut', 'stk11_mut', 'keap1_mut']
    gamma_cols = [f'gamma_{i}' for i in range(10)]

    available_muts = [c for c in mut_cols if c in cells.columns]
    if not available_muts or not all(c in cells.columns for c in gamma_cols):
        print('  SKIP: required columns not found')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Cell type composition by mutation status (using TP53 as example)
    ax = axes[0, 0]
    if 'tp53_mut' in cells.columns:
        # Per-cell gamma means stratified by TP53 status
        gamma_wt = cells[cells['tp53_mut'] == 0][gamma_cols].mean()
        gamma_mut = cells[cells['tp53_mut'] == 1][gamma_cols].mean()

        x = np.arange(10)
        width = 0.35
        ax.bar(x - width/2, gamma_wt, width, label='TP53 WT', color='#4CAF50', alpha=0.8)
        ax.bar(x + width/2, gamma_mut, width, label='TP53 Mut', color='#F44336', alpha=0.8)
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Mean Gamma')
        ax.set_title('Niche Composition: TP53 WT vs Mutant')
        ax.set_xticks(x)
        ax.set_xticklabels([f'CT{i}' for i in range(10)])
        ax.legend()

    # 2. Mutation co-occurrence with stage
    ax = axes[0, 1]
    mut_stage_data = []
    for mut in available_muts:
        for stage in STAGE_ORDER:
            stage_cells = cells[cells['stage_3'] == stage]
            if len(stage_cells) > 0:
                prev = stage_cells[mut].mean()
                mut_stage_data.append({
                    'Mutation': mut.replace('_mut', '').upper(),
                    'Stage': stage,
                    'Prevalence': prev
                })

    mut_df = pd.DataFrame(mut_stage_data)
    pivot = mut_df.pivot(index='Mutation', columns='Stage', values='Prevalence')
    pivot = pivot[STAGE_ORDER]

    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.3f', ax=ax)
    ax.set_title('Mutation Prevalence by Stage')

    # 3. IL1B expression by TP53 status and stage
    ax = axes[1, 0]
    if 'il1b_raw' in cells.columns and 'tp53_mut' in cells.columns:
        plot_data = []
        for stage in STAGE_ORDER:
            for tp53_status in [0, 1]:
                mask = (cells['stage_3'] == stage) & (cells['tp53_mut'] == tp53_status)
                il1b_vals = cells.loc[mask, 'il1b_raw'].values
                for v in np.random.choice(il1b_vals, min(1000, len(il1b_vals)), replace=False):
                    plot_data.append({
                        'Stage': stage,
                        'TP53': 'Mutant' if tp53_status == 1 else 'WT',
                        'IL1B': v
                    })

        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(data=plot_df, x='Stage', y='IL1B', hue='TP53', ax=ax,
                   order=STAGE_ORDER, palette={'WT': '#4CAF50', 'Mutant': '#F44336'})
        ax.set_title('IL1B Expression by Stage and TP53 Status')
        ax.set_ylabel('IL1B Expression')

    # 4. Embedding distance to Normal centroid by mutation status
    ax = axes[1, 1]
    z_cols = [f'z_fused_{i}' for i in range(40)]
    if all(c in cells.columns for c in z_cols):
        # Compute Normal centroid
        normal_centroid = cells[cells['stage_3'] == 'Normal'][z_cols].mean().values

        # Distance for each cell
        Z = cells[z_cols].values
        cells['dist_to_normal'] = np.linalg.norm(Z - normal_centroid, axis=1)

        # Compare distances by TP53 status within Invasive stage
        if 'tp53_mut' in cells.columns:
            invasive = cells[cells['stage_3'] == 'Invasive']
            wt_dist = invasive[invasive['tp53_mut'] == 0]['dist_to_normal']
            mut_dist = invasive[invasive['tp53_mut'] == 1]['dist_to_normal']

            ax.hist(wt_dist, bins=50, alpha=0.6, label='TP53 WT', color='#4CAF50', density=True)
            ax.hist(mut_dist, bins=50, alpha=0.6, label='TP53 Mut', color='#F44336', density=True)
            ax.set_xlabel('Distance from Normal Centroid')
            ax.set_ylabel('Density')
            ax.set_title('Invasive Cells: Distance to Normal by TP53')
            ax.legend()

            # Stats
            stat, pval = stats.mannwhitneyu(wt_dist, mut_dist)
            ax.text(0.95, 0.95, f'MWU p={pval:.2e}', transform=ax.transAxes, ha='right', va='top')

    plt.tight_layout()
    save_fig(fig, 'mutation_niche')


# =============================================================================
# FIGURE 5: IL1B Axis Deep Dive
# =============================================================================
def fig_il1b_axis(cells):
    """Deep analysis of IL1B pathway."""
    print('Generating: IL1B axis analysis...')

    if 'il1b_raw' not in cells.columns:
        print('  SKIP: il1b_raw not found')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. IL1B distribution by stage (density with stage overlap visualization)
    ax = axes[0, 0]
    for stage in STAGE_ORDER:
        data = cells[cells['stage_3'] == stage]['il1b_raw']
        ax.hist(data, bins=50, alpha=0.5, label=stage, color=STAGE_COLORS[stage], density=True)
    ax.set_xlabel('IL1B Expression')
    ax.set_ylabel('Density')
    ax.set_title('IL1B Expression Distribution by Stage')
    ax.legend()

    # Stats table
    stats_text = []
    for stage in STAGE_ORDER:
        data = cells[cells['stage_3'] == stage]['il1b_raw']
        stats_text.append(f'{stage}: mean={data.mean():.3f}, median={data.median():.3f}')
    ax.text(0.95, 0.75, '\n'.join(stats_text), transform=ax.transAxes,
            ha='right', va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. IL1B vs KAC correlation
    ax = axes[0, 1]
    if 'kac_raw' in cells.columns:
        # Sample for scatterplot
        sample = cells.sample(min(10000, len(cells)), random_state=42)
        for stage in STAGE_ORDER:
            mask = sample['stage_3'] == stage
            ax.scatter(sample.loc[mask, 'il1b_raw'], sample.loc[mask, 'kac_raw'],
                      c=STAGE_COLORS[stage], label=stage, s=5, alpha=0.3)

        # Correlation
        r, p = stats.pearsonr(cells['il1b_raw'].dropna(), cells['kac_raw'].dropna())
        ax.set_xlabel('IL1B Expression')
        ax.set_ylabel('KAC Expression')
        ax.set_title(f'IL1B vs KAC (r={r:.3f}, p={p:.2e})')
        ax.legend(markerscale=3)
    else:
        ax.text(0.5, 0.5, 'KAC not available', ha='center', va='center', transform=ax.transAxes)

    # 3. IL1B high vs low cells - niche composition
    ax = axes[1, 0]
    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if all(c in cells.columns for c in gamma_cols):
        # Define IL1B high/low (top/bottom quartile)
        il1b_q75 = cells['il1b_raw'].quantile(0.75)
        il1b_q25 = cells['il1b_raw'].quantile(0.25)

        gamma_high = cells[cells['il1b_raw'] >= il1b_q75][gamma_cols].mean()
        gamma_low = cells[cells['il1b_raw'] <= il1b_q25][gamma_cols].mean()

        x = np.arange(10)
        width = 0.35
        ax.bar(x - width/2, gamma_low, width, label='IL1B Low (Q1)', color='#2196F3', alpha=0.8)
        ax.bar(x + width/2, gamma_high, width, label='IL1B High (Q4)', color='#FF5722', alpha=0.8)
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Mean Gamma')
        ax.set_title('Niche Composition: IL1B Low vs High Cells')
        ax.set_xticks(x)
        ax.set_xticklabels([f'CT{i}' for i in range(10)])
        ax.legend()

    # 4. IL1B by donor
    ax = axes[1, 1]
    donor_il1b = cells.groupby(['donor_id', 'stage_3'])['il1b_raw'].mean().reset_index()

    # Order donors by mean IL1B
    donor_order = donor_il1b.groupby('donor_id')['il1b_raw'].mean().sort_values().index

    for stage in STAGE_ORDER:
        stage_data = donor_il1b[donor_il1b['stage_3'] == stage]
        stage_data = stage_data.set_index('donor_id').reindex(donor_order)
        ax.scatter(range(len(donor_order)), stage_data['il1b_raw'],
                  c=STAGE_COLORS[stage], label=stage, s=50, alpha=0.7)

    ax.set_xlabel('Donor (ordered by mean IL1B)')
    ax.set_ylabel('Mean IL1B Expression')
    ax.set_title('IL1B Expression by Donor and Stage')
    ax.legend()
    ax.set_xticks([])

    plt.tight_layout()
    save_fig(fig, 'il1b_axis')


# =============================================================================
# FIGURE 6: Spatial Niche Structure
# =============================================================================
def fig_spatial_niche(cells):
    """Spatial analysis of niche structure."""
    print('Generating: spatial niche analysis...')

    if 'x_spatial' not in cells.columns or 'y_spatial' not in cells.columns:
        print('  SKIP: no spatial coordinates')
        return

    # Pick one donor with good coverage
    donor_counts = cells.groupby('donor_id').size()
    top_donor = donor_counts.idxmax()
    donor_cells = cells[cells['donor_id'] == top_donor].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Spatial plots colored by different features
    features = [
        ('stage_3', 'Stage', STAGE_COLORS, 'categorical'),
        ('il1b_raw', 'IL1B Expression', 'Reds', 'continuous'),
        ('gamma_0', 'Cell Type 0 (gamma_0)', 'viridis', 'continuous'),
    ]

    for ax, (col, title, cmap, ftype) in zip(axes[0], features):
        if col not in donor_cells.columns:
            ax.text(0.5, 0.5, f'{col} not available', ha='center', va='center', transform=ax.transAxes)
            continue

        if ftype == 'categorical':
            for cat in STAGE_ORDER:
                mask = donor_cells['stage_3'] == cat
                ax.scatter(donor_cells.loc[mask, 'x_spatial'], donor_cells.loc[mask, 'y_spatial'],
                          c=cmap[cat], label=cat, s=1, alpha=0.5)
            ax.legend(markerscale=5, loc='upper right')
        else:
            scatter = ax.scatter(donor_cells['x_spatial'], donor_cells['y_spatial'],
                               c=donor_cells[col], cmap=cmap, s=1, alpha=0.5,
                               vmin=donor_cells[col].quantile(0.05),
                               vmax=donor_cells[col].quantile(0.95))
            plt.colorbar(scatter, ax=ax, shrink=0.8)

        ax.set_title(f'{title}\n(Donor: {top_donor})')
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 2: More features
    features2 = [
        ('tp53_mut', 'TP53 Mutation', {0: '#4CAF50', 1: '#F44336'}, 'binary'),
        ('ct_entropy', 'Cell Type Entropy', 'plasma', 'continuous'),
        ('dist_to_normal', 'Distance to Normal', 'magma', 'continuous'),
    ]

    # Compute entropy if not present
    gamma_cols = [f'gamma_{i}' for i in range(10)]
    if 'ct_entropy' not in donor_cells.columns and all(c in donor_cells.columns for c in gamma_cols):
        gamma_matrix = donor_cells[gamma_cols].values.astype(float)
        p = gamma_matrix / (gamma_matrix.sum(axis=1, keepdims=True) + 1e-10)
        p = np.clip(p, 1e-10, 1)
        donor_cells['ct_entropy'] = -np.sum(p * np.log2(p), axis=1)

    # Compute distance if not present
    z_cols = [f'z_fused_{i}' for i in range(40)]
    if 'dist_to_normal' not in donor_cells.columns and all(c in donor_cells.columns for c in z_cols):
        normal_centroid = cells[cells['stage_3'] == 'Normal'][z_cols].mean().values
        Z = donor_cells[z_cols].values
        donor_cells['dist_to_normal'] = np.linalg.norm(Z - normal_centroid, axis=1)

    for ax, (col, title, cmap, ftype) in zip(axes[1], features2):
        if col not in donor_cells.columns:
            ax.text(0.5, 0.5, f'{col} not available', ha='center', va='center', transform=ax.transAxes)
            continue

        if ftype == 'binary':
            for val, color in cmap.items():
                mask = donor_cells[col] == val
                label = 'Mut' if val == 1 else 'WT'
                ax.scatter(donor_cells.loc[mask, 'x_spatial'], donor_cells.loc[mask, 'y_spatial'],
                          c=color, label=label, s=1, alpha=0.5)
            ax.legend(markerscale=5, loc='upper right')
        else:
            scatter = ax.scatter(donor_cells['x_spatial'], donor_cells['y_spatial'],
                               c=donor_cells[col], cmap=cmap, s=1, alpha=0.5,
                               vmin=donor_cells[col].quantile(0.05),
                               vmax=donor_cells[col].quantile(0.95))
            plt.colorbar(scatter, ax=ax, shrink=0.8)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Spatial Niche Analysis - Donor: {top_donor} (n={len(donor_cells):,} cells)', fontsize=14)
    plt.tight_layout()
    save_fig(fig, 'spatial_niche')


def main():
    print('=' * 60)
    print('GENERATING ADVANCED POSTER FIGURES')
    print('=' * 60)
    print(f'Output: {FIG_DIR}')
    print()

    cells = load_data()
    ckpt = load_model_weights()
    print()

    # Generate all figures
    fig_drift_head_analysis(ckpt)
    fig_destvi_advanced(cells)
    fig_embedding_analysis(cells)
    fig_mutation_niche(cells)
    fig_il1b_axis(cells)
    fig_spatial_niche(cells)

    print()
    print('=' * 60)
    print('DONE')
    print('=' * 60)
    print(f'Figures saved to: {FIG_DIR}')


if __name__ == '__main__':
    main()
