#!/usr/bin/env python
"""
Generate publication-quality EDA figures from StageBridge data prep outputs.
Style inspired by Mayr et al. Sci Adv 2024 (IPF spatial niches).

Usage:
    python scripts/generate_eda_figures.py --data-dir /path/to/canonical --output-dir figures/eda

Outputs individual figures (not panels) as requested.
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Publication style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palettes (matching Mayr et al. style)
NICHE_COLORS = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
    '#F39B7F', '#8491B4', '#91D1C2', '#DC0000',
    '#7E6148', '#B09C85'
]

STAGE_COLORS = {
    'Normal': '#4DAF4A',
    'AAH': '#984EA3',
    'AIS': '#FF7F00',
    'MIA': '#E41A1C',
    'LUAD': '#377EB8',
    'Control': '#4DAF4A',
    'IPF': '#E41A1C',
}

CELLTYPE_CMAP = 'tab20'


def load_parquet_safe(path):
    """Load parquet with error handling."""
    if path.exists():
        return pd.read_parquet(path)
    print(f"  Warning: {path.name} not found")
    return None


# =============================================================================
# Figure 1: Spatial cell type abundance maps
# =============================================================================

def fig_spatial_celltype_abundance(data_dir, output_dir, sample=None):
    """
    Spatial plots of cell type abundances from deconvolution.
    Like Mayr et al. Fig 1B.
    """
    print("Generating spatial cell type abundance plots...")

    # Try cell2location first, then DestVI
    deconv = load_parquet_safe(data_dir / 'visium/spot_deconvolution_cell2location.parquet')
    method = 'cell2location'
    if deconv is None:
        deconv = load_parquet_safe(data_dir / 'visium/spot_deconvolution_destvi.parquet')
        method = 'destvi'

    if deconv is None:
        print("  Skipping: no deconvolution data")
        return

    # Get cell type columns
    exclude = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id', 'batch']
    ct_cols = [c for c in deconv.columns if c not in exclude and deconv[c].dtype in ['float64', 'float32']]

    # Get coordinates
    if 'x' not in deconv.columns:
        spatial = load_parquet_safe(data_dir / 'visium/spatial_gene_expression.parquet')
        if spatial is not None and 'x' in spatial.columns:
            deconv['x'] = spatial['x'].values[:len(deconv)]
            deconv['y'] = spatial['y'].values[:len(deconv)]

    if 'x' not in deconv.columns:
        print("  Skipping: no spatial coordinates")
        return

    # Filter to one sample if specified, or use first sample
    if 'sample' in deconv.columns:
        samples = deconv['sample'].unique()
        if sample and sample in samples:
            deconv = deconv[deconv['sample'] == sample]
        else:
            sample = samples[0]
            deconv = deconv[deconv['sample'] == sample]

    # Top cell types by abundance
    top_cts = deconv[ct_cols].mean().nlargest(9).index.tolist()

    out_subdir = output_dir / 'spatial_abundance'
    out_subdir.mkdir(exist_ok=True, parents=True)

    for ct in top_cts:
        fig, ax = plt.subplots(figsize=(5, 4.5))

        scatter = ax.scatter(
            deconv['x'], deconv['y'],
            c=deconv[ct],
            cmap='Reds',
            s=8,
            alpha=0.9,
            edgecolors='none',
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Abundance', fontsize=10)

        ax.set_aspect('equal')
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
        ax.set_title(f'{ct}', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Clean spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        safe_name = ct.replace('/', '_').replace(' ', '_').replace('+', 'pos')
        fig.savefig(out_subdir / f'spatial_abundance_{safe_name}.png')
        fig.savefig(out_subdir / f'spatial_abundance_{safe_name}.pdf')
        plt.close(fig)

    print(f"  Saved {len(top_cts)} spatial abundance plots")


# =============================================================================
# Figure 2: Cell type co-localization correlation matrix
# =============================================================================

def fig_colocalization_heatmap(data_dir, output_dir):
    """
    Clustered heatmap of cell type co-localization correlations.
    Like Mayr et al. Fig 1D.
    """
    print("Generating co-localization heatmap...")

    corr = load_parquet_safe(data_dir / 'visium/celltype_colocalization_corr.parquet')
    if corr is None:
        print("  Skipping: no co-localization data")
        return

    # Cluster the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Hierarchical clustering for ordering
    dist = 1 - corr.values
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)

    try:
        linkage_matrix = linkage(squareform(dist), method='ward')
        order = leaves_list(linkage_matrix)
        corr_ordered = corr.iloc[order, order]
    except:
        corr_ordered = corr

    fig, ax = plt.subplots(figsize=(10, 9))

    sns.heatmap(
        corr_ordered,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': 'Pearson correlation'},
        ax=ax,
    )

    ax.set_title('Cell type co-localization', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.savefig(output_dir / 'colocalization_heatmap.png')
    fig.savefig(output_dir / 'colocalization_heatmap.pdf')
    plt.close(fig)
    print("  Saved colocalization_heatmap")


# =============================================================================
# Figure 3: Spatial vs scRNA-seq frequency comparison
# =============================================================================

def fig_spatial_vs_snrna_frequencies(data_dir, output_dir):
    """
    Bar plot comparing cell type frequencies between spatial and scRNA-seq.
    Like Mayr et al. Fig 1C.
    """
    print("Generating spatial vs scRNA-seq frequency comparison...")

    comp = load_parquet_safe(data_dir / 'visium/spatial_vs_snrna_frequencies.parquet')
    if comp is None:
        print("  Skipping: no frequency comparison data")
        return

    # Sort by scRNA-seq frequency
    comp = comp.sort_values('snrna_frequency', ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(comp) * 0.25)))

    y_pos = np.arange(len(comp))
    bar_height = 0.35

    # Spatial bars
    ax.barh(y_pos - bar_height/2, comp['spatial_frequency'], bar_height,
            label='Spatial', color='#E64B35', alpha=0.9)

    # scRNA-seq bars
    ax.barh(y_pos + bar_height/2, comp['snrna_frequency'], bar_height,
            label='scRNA-seq', color='#4DBBD5', alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp.index, fontsize=9)
    ax.set_xlabel('Normalized cell type frequency')
    ax.set_title('Spatial vs scRNA-seq cell type frequencies', fontweight='bold')
    ax.legend(loc='lower right', frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'spatial_vs_snrna_frequencies.png')
    fig.savefig(output_dir / 'spatial_vs_snrna_frequencies.pdf')
    plt.close(fig)
    print("  Saved spatial_vs_snrna_frequencies")


# =============================================================================
# Figure 4: Niche composition heatmap
# =============================================================================

def fig_niche_composition_heatmap(data_dir, output_dir):
    """
    Heatmap showing cell type composition of each niche/phenotype.
    Like Mayr et al. Fig 2A.
    """
    print("Generating niche composition heatmap...")

    centers = load_parquet_safe(data_dir / 'niche_phenotypes/phenotype_centers.parquet')
    if centers is None:
        print("  Skipping: no niche phenotype data")
        return

    # Get cell type columns only
    exclude = ['n_spots', 'phenotype_name', 'top1_celltype', 'top1_prop',
               'top2_celltype', 'top2_prop', 'top3_celltype', 'top3_prop']
    ct_cols = [c for c in centers.columns if c not in exclude]

    if not ct_cols:
        print("  Skipping: no cell type columns found")
        return

    # Normalize each niche to sum to 100%
    comp = centers[ct_cols].copy()
    comp = comp.div(comp.sum(axis=1), axis=0) * 100

    # Use phenotype names if available
    if 'phenotype_name' in centers.columns:
        comp.index = centers['phenotype_name'].values
    else:
        comp.index = [f'Niche {i}' for i in comp.index]

    fig, ax = plt.subplots(figsize=(12, max(4, len(comp) * 0.6)))

    sns.heatmap(
        comp,
        cmap='YlOrRd',
        annot=True,
        fmt='.0f',
        annot_kws={'fontsize': 7},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': 'Cell type abundance (%)'},
        ax=ax,
    )

    ax.set_title('Niche cell type composition', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_ylabel('')

    fig.savefig(output_dir / 'niche_composition_heatmap.png')
    fig.savefig(output_dir / 'niche_composition_heatmap.pdf')
    plt.close(fig)
    print("  Saved niche_composition_heatmap")


# =============================================================================
# Figure 5: Niche frequency by stage
# =============================================================================

def fig_niche_frequency_by_stage(data_dir, output_dir):
    """
    Stacked bar plot showing niche frequencies across disease stages.
    Like Mayr et al. Fig 2B.
    """
    print("Generating niche frequency by stage...")

    freq = load_parquet_safe(data_dir / 'niche_phenotypes/phenotype_by_stage.parquet')
    if freq is None:
        print("  Skipping: no niche by stage data")
        return

    # Ensure columns are numeric
    freq = freq.apply(pd.to_numeric, errors='coerce')

    # Sort stages if known order
    stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD', 'Control', 'IPF']
    available_stages = [s for s in stage_order if s in freq.index]
    other_stages = [s for s in freq.index if s not in stage_order]
    freq = freq.reindex(available_stages + other_stages)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = NICHE_COLORS[:len(freq.columns)]
    freq.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.7, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Relative frequency')
    ax.set_xlabel('')
    ax.set_title('Niche distribution across stages', fontweight='bold')
    ax.legend(title='Niche', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'niche_frequency_by_stage.png')
    fig.savefig(output_dir / 'niche_frequency_by_stage.pdf')
    plt.close(fig)
    print("  Saved niche_frequency_by_stage")


# =============================================================================
# Figure 6: Spatial niche maps
# =============================================================================

def fig_spatial_niche_maps(data_dir, output_dir):
    """
    Spatial plots colored by niche assignment.
    Like Mayr et al. Fig 2C.
    """
    print("Generating spatial niche maps...")

    phenotypes = load_parquet_safe(data_dir / 'niche_phenotypes/spot_niche_phenotypes.parquet')
    if phenotypes is None:
        print("  Skipping: no niche phenotype data")
        return

    if 'x' not in phenotypes.columns or 'y' not in phenotypes.columns:
        print("  Skipping: no spatial coordinates")
        return

    out_subdir = output_dir / 'spatial_niches'
    out_subdir.mkdir(exist_ok=True, parents=True)

    # If we have samples, make one plot per sample
    if 'sample' in phenotypes.columns:
        samples = phenotypes['sample'].unique()
    else:
        samples = [None]
        phenotypes['sample'] = 'all'

    n_niches = phenotypes['niche_phenotype'].nunique()
    colors = NICHE_COLORS[:n_niches]

    for sample in samples[:6]:  # Max 6 samples
        if sample:
            df = phenotypes[phenotypes['sample'] == sample]
            title = f'Sample: {sample}'
            fname = f'spatial_niches_{sample}'
        else:
            df = phenotypes
            title = 'Spatial niche distribution'
            fname = 'spatial_niches_all'

        fig, ax = plt.subplots(figsize=(6, 5.5))

        for i, niche in enumerate(sorted(df['niche_phenotype'].unique())):
            mask = df['niche_phenotype'] == niche
            label = df.loc[mask, 'phenotype_name'].iloc[0] if 'phenotype_name' in df.columns else f'Niche {niche}'
            ax.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=[colors[i % len(colors)]],
                s=10,
                alpha=0.8,
                label=label,
                edgecolors='none',
            )

        ax.set_aspect('equal')
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
        ax.set_title(title, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.savefig(out_subdir / f'{fname}.png')
        fig.savefig(out_subdir / f'{fname}.pdf')
        plt.close(fig)

    print(f"  Saved spatial niche maps for {min(len(samples), 6)} samples")


# =============================================================================
# Figure 7: Pathway activity spatial plots
# =============================================================================

def fig_pathway_activity_spatial(data_dir, output_dir):
    """
    Spatial plots of pathway activity (PROGENy).
    Like Mayr et al. Fig 2F/G.
    """
    print("Generating pathway activity spatial plots...")

    activity = load_parquet_safe(data_dir / 'activity/pathway_activity_progeny.parquet')
    if activity is None:
        print("  Skipping: no pathway activity data")
        return

    # Get spatial coordinates
    spatial = load_parquet_safe(data_dir / 'visium/spatial_gene_expression.parquet')
    if spatial is None or 'x' not in spatial.columns:
        # Try from embeddings
        spatial = load_parquet_safe(data_dir / 'embeddings/umap_embedding.parquet')

    if spatial is None:
        print("  Skipping: no spatial coordinates")
        return

    # Match indices
    if len(activity) != len(spatial):
        print(f"  Warning: length mismatch ({len(activity)} vs {len(spatial)}), using min")
        n = min(len(activity), len(spatial))
        activity = activity.iloc[:n]
        spatial = spatial.iloc[:n]

    # Key pathways to plot
    key_pathways = ['TGFb', 'TNFa', 'NFkB', 'JAK-STAT', 'Hypoxia', 'EGFR', 'VEGF', 'PI3K', 'MAPK']
    available = [p for p in key_pathways if p in activity.columns]

    if not available:
        available = activity.columns[:6].tolist()

    out_subdir = output_dir / 'pathway_activity'
    out_subdir.mkdir(exist_ok=True, parents=True)

    x_col = 'x' if 'x' in spatial.columns else 'UMAP1'
    y_col = 'y' if 'y' in spatial.columns else 'UMAP2'

    for pathway in available:
        fig, ax = plt.subplots(figsize=(5, 4.5))

        values = activity[pathway].values
        vmax = np.percentile(np.abs(values), 95)

        scatter = ax.scatter(
            spatial[x_col], spatial[y_col],
            c=values,
            cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            s=8,
            alpha=0.9,
            edgecolors='none',
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Activity score', fontsize=10)

        ax.set_aspect('equal')
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
        ax.set_title(f'{pathway} activity', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.savefig(out_subdir / f'pathway_{pathway}.png')
        fig.savefig(out_subdir / f'pathway_{pathway}.pdf')
        plt.close(fig)

    print(f"  Saved {len(available)} pathway activity plots")


# =============================================================================
# Figure 8: Signature score spatial plots
# =============================================================================

def fig_signature_spatial(data_dir, output_dir):
    """
    Spatial plots of gene signature scores.
    Like Mayr et al. Fig 3E (senescence).
    """
    print("Generating signature score spatial plots...")

    sigs = load_parquet_safe(data_dir / 'signatures/gene_signatures.parquet')
    if sigs is None:
        print("  Skipping: no signature data")
        return

    # Get spatial coordinates
    spatial = load_parquet_safe(data_dir / 'visium/spatial_gene_expression.parquet')
    if spatial is None or 'x' not in spatial.columns:
        spatial = load_parquet_safe(data_dir / 'embeddings/umap_embedding.parquet')

    if spatial is None:
        print("  Skipping: no spatial coordinates")
        return

    n = min(len(sigs), len(spatial))
    sigs = sigs.iloc[:n]
    spatial = spatial.iloc[:n]

    # Key signatures
    key_sigs = ['hypoxia', 'stemness', 'il1b_pathway', 'tgfb_response', 'ap1_stress',
                'ifn_gamma', 'glycolysis', 'nfkb']
    exclude = ['cell_id', 'stage', 'cell_type']
    available = [s for s in key_sigs if s in sigs.columns]
    if not available:
        available = [c for c in sigs.columns if c not in exclude][:6]

    out_subdir = output_dir / 'signature_spatial'
    out_subdir.mkdir(exist_ok=True, parents=True)

    x_col = 'x' if 'x' in spatial.columns else 'UMAP1'
    y_col = 'y' if 'y' in spatial.columns else 'UMAP2'

    for sig in available:
        fig, ax = plt.subplots(figsize=(5, 4.5))

        values = sigs[sig].values

        scatter = ax.scatter(
            spatial[x_col], spatial[y_col],
            c=values,
            cmap='YlOrRd',
            s=8,
            alpha=0.9,
            edgecolors='none',
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Score', fontsize=10)

        ax.set_aspect('equal')
        ax.set_xlabel('Spatial 1')
        ax.set_ylabel('Spatial 2')
        title = sig.replace('_', ' ').title()
        ax.set_title(f'{title} signature', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.savefig(out_subdir / f'signature_{sig}.png')
        fig.savefig(out_subdir / f'signature_{sig}.pdf')
        plt.close(fig)

    print(f"  Saved {len(available)} signature spatial plots")


# =============================================================================
# Figure 9: Cell-cell communication network
# =============================================================================

def fig_communication_network(data_dir, output_dir):
    """
    Network diagram showing cell-cell interactions.
    Like Mayr et al. Fig 3I.
    """
    print("Generating cell-cell communication network...")

    comm = load_parquet_safe(data_dir / 'communication/communication_matrix.parquet')
    if comm is None:
        print("  Skipping: no communication matrix")
        return

    try:
        import networkx as nx
    except ImportError:
        print("  Skipping: networkx not installed")
        return

    # Create network
    G = nx.DiGraph()

    # Add nodes
    cell_types = list(comm.index)
    G.add_nodes_from(cell_types)

    # Add edges (threshold to top interactions)
    threshold = np.percentile(comm.values.flatten(), 90)

    for source in comm.index:
        for target in comm.columns:
            weight = comm.loc[source, target]
            if weight > threshold and source != target:
                G.add_edge(source, target, weight=weight)

    if G.number_of_edges() == 0:
        print("  Skipping: no edges above threshold")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes by degree
    degrees = dict(G.degree())
    node_sizes = [300 + degrees[n] * 100 for n in G.nodes()]

    # Edge widths by weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + 4 * w / max_weight for w in edge_weights]

    # Colors
    node_colors = [NICHE_COLORS[i % len(NICHE_COLORS)] for i in range(len(G.nodes()))]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                           edge_color='gray', arrows=True,
                           arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title('Cell-cell communication network', fontweight='bold', fontsize=14)
    ax.axis('off')

    fig.savefig(output_dir / 'communication_network.png')
    fig.savefig(output_dir / 'communication_network.pdf')
    plt.close(fig)
    print("  Saved communication_network")


# =============================================================================
# Figure 10: L-R interaction heatmap
# =============================================================================

def fig_lr_interaction_heatmap(data_dir, output_dir):
    """
    Heatmap of ligand-receptor interactions.
    Like Mayr et al. Fig 3K, 4I, 5G.
    """
    print("Generating L-R interaction heatmap...")

    interactions = load_parquet_safe(data_dir / 'communication/top_interactions.parquet')
    if interactions is None:
        interactions = load_parquet_safe(data_dir / 'liana_interactions.parquet')

    if interactions is None:
        print("  Skipping: no interaction data")
        return

    # Check for required columns
    required = ['source', 'target']
    if not all(c in interactions.columns for c in required):
        print("  Skipping: missing source/target columns")
        return

    # Get ligand-receptor info
    lr_col = None
    for col in ['ligand_complex', 'ligand', 'interaction']:
        if col in interactions.columns:
            lr_col = col
            break

    if lr_col is None:
        print("  Skipping: no ligand column found")
        return

    receptor_col = None
    for col in ['receptor_complex', 'receptor']:
        if col in interactions.columns:
            receptor_col = col
            break

    # Score column
    score_col = None
    for col in ['specificity_rank', 'magnitude_rank', 'pvalue', 'score']:
        if col in interactions.columns:
            score_col = col
            break

    if score_col is None:
        print("  Skipping: no score column found")
        return

    # Create L-R pair label
    if receptor_col:
        interactions['lr_pair'] = interactions[lr_col] + ' - ' + interactions[receptor_col]
    else:
        interactions['lr_pair'] = interactions[lr_col]

    # Create source-target pair
    interactions['ct_pair'] = interactions['source'] + ' -> ' + interactions['target']

    # Pivot to heatmap format (top 30 L-R pairs x top 20 cell type pairs)
    top_lr = interactions.groupby('lr_pair')[score_col].mean().nsmallest(30).index
    top_ct = interactions.groupby('ct_pair')[score_col].mean().nsmallest(20).index

    subset = interactions[interactions['lr_pair'].isin(top_lr) & interactions['ct_pair'].isin(top_ct)]

    if len(subset) < 5:
        print("  Skipping: not enough data for heatmap")
        return

    pivot = subset.pivot_table(index='ct_pair', columns='lr_pair', values=score_col, aggfunc='mean')

    fig, ax = plt.subplots(figsize=(14, 8))

    # Transform scores (lower = more significant for ranks)
    if 'rank' in score_col or 'pvalue' in score_col:
        plot_data = -np.log10(pivot + 0.001)
        cbar_label = '-log10(score)'
    else:
        plot_data = pivot
        cbar_label = 'Score'

    sns.heatmap(
        plot_data,
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': cbar_label},
        ax=ax,
    )

    ax.set_title('Ligand-receptor interactions', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel('Ligand - Receptor')
    ax.set_ylabel('Source -> Target')

    fig.savefig(output_dir / 'lr_interaction_heatmap.png')
    fig.savefig(output_dir / 'lr_interaction_heatmap.pdf')
    plt.close(fig)
    print("  Saved lr_interaction_heatmap")


# =============================================================================
# Figure 11: Cell type proportions by stage
# =============================================================================

def fig_celltype_proportions_by_stage(data_dir, output_dir):
    """
    Stacked bar or grouped bar plot of cell type proportions by stage.
    """
    print("Generating cell type proportions by stage...")

    props = load_parquet_safe(data_dir / 'summary_stats/celltype_proportions_by_stage.parquet')
    if props is None:
        print("  Skipping: no proportion data")
        return

    # Sort stages
    stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD', 'Control', 'IPF']
    available_stages = [s for s in stage_order if s in props.index]
    other_stages = [s for s in props.index if s not in stage_order]
    props = props.reindex(available_stages + other_stages)

    # Top cell types
    top_cts = props.mean().nlargest(12).index
    props_top = props[top_cts]

    fig, ax = plt.subplots(figsize=(10, 6))

    props_top.plot(kind='bar', ax=ax, width=0.8, colormap='tab20')

    ax.set_ylabel('Proportion')
    ax.set_xlabel('')
    ax.set_title('Cell type composition by disease stage', fontweight='bold')
    ax.legend(title='Cell type', bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=False, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'celltype_proportions_by_stage.png')
    fig.savefig(output_dir / 'celltype_proportions_by_stage.pdf')
    plt.close(fig)
    print("  Saved celltype_proportions_by_stage")


# =============================================================================
# Figure 12: UMAP with cell types
# =============================================================================

def fig_umap_celltype(data_dir, output_dir):
    """
    UMAP colored by cell type.
    """
    print("Generating UMAP by cell type...")

    umap = load_parquet_safe(data_dir / 'embeddings/umap_embedding.parquet')
    if umap is None:
        print("  Skipping: no UMAP data")
        return

    if 'UMAP1' not in umap.columns:
        print("  Skipping: no UMAP coordinates")
        return

    ct_col = 'cell_type' if 'cell_type' in umap.columns else None
    if ct_col is None:
        print("  Skipping: no cell type column")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    cell_types = umap[ct_col].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))

    for i, ct in enumerate(sorted(cell_types)):
        mask = umap[ct_col] == ct
        ax.scatter(
            umap.loc[mask, 'UMAP1'],
            umap.loc[mask, 'UMAP2'],
            c=[colors[i]],
            s=2,
            alpha=0.6,
            label=ct,
        )

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP by cell type', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
              fontsize=7, markerscale=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'umap_celltype.png')
    fig.savefig(output_dir / 'umap_celltype.pdf')
    plt.close(fig)
    print("  Saved umap_celltype")


# =============================================================================
# Figure 13: UMAP by stage
# =============================================================================

def fig_umap_stage(data_dir, output_dir):
    """
    UMAP colored by disease stage.
    """
    print("Generating UMAP by stage...")

    umap = load_parquet_safe(data_dir / 'embeddings/umap_embedding.parquet')
    if umap is None or 'stage' not in umap.columns:
        print("  Skipping: no UMAP or stage data")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    stages = umap['stage'].unique()

    for stage in stages:
        mask = umap['stage'] == stage
        color = STAGE_COLORS.get(stage, '#808080')
        ax.scatter(
            umap.loc[mask, 'UMAP1'],
            umap.loc[mask, 'UMAP2'],
            c=[color],
            s=2,
            alpha=0.5,
            label=stage,
        )

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP by disease stage', fontweight='bold')
    ax.legend(frameon=False, markerscale=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'umap_stage.png')
    fig.savefig(output_dir / 'umap_stage.pdf')
    plt.close(fig)
    print("  Saved umap_stage")


# =============================================================================
# Figure 14: QC violin plots
# =============================================================================

def fig_qc_violins(data_dir, output_dir):
    """
    Violin plots of QC metrics by stage.
    """
    print("Generating QC violin plots...")

    qc = load_parquet_safe(data_dir / 'qc/snrna_qc_metrics.parquet')
    if qc is None:
        print("  Skipping: no QC data")
        return

    if 'stage' not in qc.columns:
        print("  Skipping: no stage column")
        return

    metrics = ['pct_counts_mt', 'pct_counts_ribo', 'n_genes_by_counts', 'total_counts']
    available = [m for m in metrics if m in qc.columns]

    out_subdir = output_dir / 'qc'
    out_subdir.mkdir(exist_ok=True, parents=True)

    for metric in available:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Order stages
        stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
        available_stages = [s for s in stage_order if s in qc['stage'].unique()]
        other_stages = [s for s in qc['stage'].unique() if s not in stage_order]
        order = available_stages + other_stages

        palette = {s: STAGE_COLORS.get(s, '#808080') for s in order}

        sns.violinplot(data=qc, x='stage', y=metric, order=order, palette=palette,
                       ax=ax, inner='box', cut=0)

        title_map = {
            'pct_counts_mt': 'Mitochondrial %',
            'pct_counts_ribo': 'Ribosomal %',
            'n_genes_by_counts': 'Genes detected',
            'total_counts': 'Total counts',
        }

        ax.set_xlabel('')
        ax.set_ylabel(title_map.get(metric, metric))
        ax.set_title(f'{title_map.get(metric, metric)} by stage', fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.savefig(out_subdir / f'qc_{metric}.png')
        fig.savefig(out_subdir / f'qc_{metric}.pdf')
        plt.close(fig)

    print(f"  Saved {len(available)} QC violin plots")


# =============================================================================
# Figure 15: GO/KEGG enrichment bar plots
# =============================================================================

def fig_pathway_enrichment(data_dir, output_dir):
    """
    Bar plots of pathway enrichment results.
    Like Mayr et al. Fig 2D/E.
    """
    print("Generating pathway enrichment plots...")

    pathway_dir = data_dir / 'pathways'
    if not pathway_dir.exists():
        print("  Skipping: no pathway directory")
        return

    out_subdir = output_dir / 'enrichment'
    out_subdir.mkdir(exist_ok=True, parents=True)

    # Find enrichment files
    enrichment_files = list(pathway_dir.glob('gsea_*.parquet')) + list(pathway_dir.glob('ora_*.parquet'))

    for f in enrichment_files[:10]:  # Max 10 files
        df = pd.read_parquet(f)

        # Find term and score columns
        term_col = None
        for col in ['Term', 'term', 'Name', 'name', 'pathway']:
            if col in df.columns:
                term_col = col
                break

        if term_col is None:
            continue

        score_col = None
        for col in ['NES', 'nes', 'Combined Score', 'combined_score', '-log10(pvalue)']:
            if col in df.columns:
                score_col = col
                break

        if score_col is None:
            # Try to compute from p-value
            for col in ['pvalue', 'P-value', 'FDR', 'fdr', 'padj']:
                if col in df.columns:
                    df['score'] = -np.log10(df[col] + 1e-10)
                    score_col = 'score'
                    break

        if score_col is None:
            continue

        # Top terms
        if 'NES' in score_col.upper():
            # GSEA: sort by absolute NES
            df['abs_score'] = df[score_col].abs()
            top = df.nlargest(15, 'abs_score')
        else:
            top = df.nlargest(15, score_col)

        if len(top) < 3:
            continue

        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.3)))

        # Color by direction if NES
        if 'NES' in score_col.upper():
            colors = ['#E64B35' if x > 0 else '#4DBBD5' for x in top[score_col]]
        else:
            colors = '#E64B35'

        ax.barh(range(len(top)), top[score_col], color=colors, alpha=0.9)
        ax.set_yticks(range(len(top)))

        # Truncate long names
        labels = [t[:50] + '...' if len(t) > 50 else t for t in top[term_col]]
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_xlabel(score_col)
        ax.invert_yaxis()

        title = f.stem.replace('gsea_', '').replace('ora_', '').replace('_', ' ').title()
        ax.set_title(title, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.savefig(out_subdir / f'{f.stem}.png')
        fig.savefig(out_subdir / f'{f.stem}.pdf')
        plt.close(fig)

    print(f"  Saved enrichment plots")


# =============================================================================
# Figure 16: Rare cell signature dot plot
# =============================================================================

def fig_rare_cell_dotplot(data_dir, output_dir):
    """
    Dot plot of rare cell signature scores by cell type.
    """
    print("Generating rare cell signature dot plot...")

    sigs = load_parquet_safe(data_dir / 'rare_cells/rare_signatures_by_celltype.parquet')
    if sigs is None:
        print("  Skipping: no rare cell signature data")
        return

    # Select key signatures
    key_sigs = ['cDC1', 'LAMP3_DC', 'Treg', 'exhausted_CD8', 'plasma_cell',
                'hypoxic_tumor', 'EMT_tumor', 'myCAF', 'iCAF', 'lymphatic_endo']
    available = [s for s in key_sigs if s in sigs.columns]

    if len(available) < 3:
        available = sigs.columns[:10].tolist()

    data = sigs[available]

    fig, ax = plt.subplots(figsize=(12, max(5, len(data) * 0.4)))

    # Normalize for dot size
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)

    for i, ct in enumerate(data.index):
        for j, sig in enumerate(available):
            size = data_norm.loc[ct, sig] * 200 + 20
            color = data.loc[ct, sig]
            ax.scatter(j, i, s=size, c=[color], cmap='YlOrRd',
                      vmin=data.min().min(), vmax=data.max().max())

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([s.replace('_', '\n') for s in available], fontsize=9, rotation=0)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_title('Rare cell signatures by cell type', fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=plt.Normalize(vmin=data.min().min(), vmax=data.max().max()))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Mean score', fontsize=10)

    ax.set_xlim(-0.5, len(available) - 0.5)
    ax.set_ylim(-0.5, len(data) - 0.5)

    fig.savefig(output_dir / 'rare_cell_signatures_dotplot.png')
    fig.savefig(output_dir / 'rare_cell_signatures_dotplot.pdf')
    plt.close(fig)
    print("  Saved rare_cell_signatures_dotplot")


# =============================================================================
# Figure 17: Neighborhood enrichment heatmap
# =============================================================================

def fig_neighborhood_enrichment(data_dir, output_dir):
    """
    Heatmap of cell type neighborhood enrichment (Squidpy).
    """
    print("Generating neighborhood enrichment heatmap...")

    nhood = load_parquet_safe(data_dir / 'spatial_stats/nhood_enrichment.parquet')
    if nhood is None:
        print("  Skipping: no neighborhood enrichment data")
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    # Cluster for better visualization
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        dist = pdist(nhood.values)
        link = linkage(dist, method='ward')
        order = leaves_list(link)
        nhood = nhood.iloc[order, order]
    except:
        pass

    vmax = np.percentile(np.abs(nhood.values), 95)

    sns.heatmap(
        nhood,
        cmap='RdBu_r',
        center=0,
        vmin=-vmax, vmax=vmax,
        square=True,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': 'Z-score'},
        ax=ax,
    )

    ax.set_title('Neighborhood enrichment', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.savefig(output_dir / 'neighborhood_enrichment.png')
    fig.savefig(output_dir / 'neighborhood_enrichment.pdf')
    plt.close(fig)
    print("  Saved neighborhood_enrichment")


# =============================================================================
# Figure 18: Moran's I spatial autocorrelation
# =============================================================================

def fig_morans_i(data_dir, output_dir):
    """
    Bar plot of Moran's I for spatially variable genes.
    """
    print("Generating Moran's I plot...")

    morans = load_parquet_safe(data_dir / 'spatial_stats/morans_i.parquet')
    if morans is None:
        print("  Skipping: no Moran's I data")
        return

    # Sort by I statistic
    if 'I' in morans.columns:
        morans = morans.sort_values('I', ascending=False).head(30)
        y_col = 'I'
    else:
        print("  Skipping: no I column found")
        return

    fig, ax = plt.subplots(figsize=(8, max(5, len(morans) * 0.25)))

    colors = ['#E64B35' if p < 0.05 else '#808080'
              for p in morans.get('pval_norm', [0.01] * len(morans))]

    ax.barh(range(len(morans)), morans[y_col], color=colors, alpha=0.9)
    ax.set_yticks(range(len(morans)))
    ax.set_yticklabels(morans.index, fontsize=9)
    ax.set_xlabel("Moran's I")
    ax.set_title('Spatially variable genes', fontweight='bold')
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(output_dir / 'morans_i.png')
    fig.savefig(output_dir / 'morans_i.pdf')
    plt.close(fig)
    print("  Saved morans_i")


# =============================================================================
# Figure 19: Key gene expression dotplot
# =============================================================================

def fig_key_genes_dotplot(data_dir, output_dir):
    """
    Dot plot of key gene expression by stage and cell type.
    """
    print("Generating key genes dot plot...")

    expr = load_parquet_safe(data_dir / 'expression/key_genes_mean_by_stage_celltype.parquet')
    if expr is None:
        print("  Skipping: no key genes expression data")
        return

    # Key genes of interest
    key_genes = ['IL1B', 'IL1R1', 'VIM', 'CDH1', 'KRT17', 'SOX9', 'ACTA2',
                 'CD68', 'CD3D', 'MKI67', 'CD274', 'PDCD1']
    available = [g for g in key_genes if g in expr.columns]

    if len(available) < 3:
        available = expr.columns[:10].tolist()

    data = expr[available]

    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.3)))

    # Normalize for visualization
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)

    for i, idx in enumerate(data.index):
        for j, gene in enumerate(available):
            size = data_norm.loc[idx, gene] * 200 + 20
            color = data.loc[idx, gene]
            ax.scatter(j, i, s=size, c=[color], cmap='Reds',
                      vmin=data.min().min(), vmax=data.max().max())

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(data)))

    # Format index labels
    labels = [f'{idx[0]} - {idx[1][:15]}' if isinstance(idx, tuple) else str(idx)
              for idx in data.index]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title('Key gene expression', fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap='Reds',
                                norm=plt.Normalize(vmin=data.min().min(), vmax=data.max().max()))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Mean expression', fontsize=10)

    ax.set_xlim(-0.5, len(available) - 0.5)
    ax.set_ylim(-0.5, len(data) - 0.5)

    fig.savefig(output_dir / 'key_genes_dotplot.png')
    fig.savefig(output_dir / 'key_genes_dotplot.pdf')
    plt.close(fig)
    print("  Saved key_genes_dotplot")


# =============================================================================
# Figure 20: TF activity heatmap
# =============================================================================

def fig_tf_activity_heatmap(data_dir, output_dir):
    """
    Heatmap of transcription factor activity by stage.
    """
    print("Generating TF activity heatmap...")

    tf_act = load_parquet_safe(data_dir / 'activity/tf_activity_by_stage.parquet')
    if tf_act is None:
        print("  Skipping: no TF activity data")
        return

    # Top variable TFs
    tf_var = tf_act.var().nlargest(30).index
    data = tf_act[tf_var].T

    # Sort stages
    stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
    available = [s for s in stage_order if s in data.columns]
    other = [s for s in data.columns if s not in stage_order]
    data = data[available + other]

    fig, ax = plt.subplots(figsize=(8, 10))

    sns.heatmap(
        data,
        cmap='RdBu_r',
        center=0,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': 'Activity'},
        ax=ax,
    )

    ax.set_title('Transcription factor activity', fontweight='bold', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.savefig(output_dir / 'tf_activity_heatmap.png')
    fig.savefig(output_dir / 'tf_activity_heatmap.pdf')
    plt.close(fig)
    print("  Saved tf_activity_heatmap")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate publication EDA figures')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Path to canonical data directory')
    parser.add_argument('--output-dir', type=Path, default=Path('figures/eda'),
                        help='Output directory for figures')
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("StageBridge EDA Figure Generation")
    print("=" * 60)
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print()

    # Generate all figures
    fig_spatial_celltype_abundance(args.data_dir, args.output_dir)
    fig_colocalization_heatmap(args.data_dir, args.output_dir)
    fig_spatial_vs_snrna_frequencies(args.data_dir, args.output_dir)
    fig_niche_composition_heatmap(args.data_dir, args.output_dir)
    fig_niche_frequency_by_stage(args.data_dir, args.output_dir)
    fig_spatial_niche_maps(args.data_dir, args.output_dir)
    fig_pathway_activity_spatial(args.data_dir, args.output_dir)
    fig_signature_spatial(args.data_dir, args.output_dir)
    fig_communication_network(args.data_dir, args.output_dir)
    fig_lr_interaction_heatmap(args.data_dir, args.output_dir)
    fig_celltype_proportions_by_stage(args.data_dir, args.output_dir)
    fig_umap_celltype(args.data_dir, args.output_dir)
    fig_umap_stage(args.data_dir, args.output_dir)
    fig_qc_violins(args.data_dir, args.output_dir)
    fig_pathway_enrichment(args.data_dir, args.output_dir)
    fig_rare_cell_dotplot(args.data_dir, args.output_dir)
    fig_neighborhood_enrichment(args.data_dir, args.output_dir)
    fig_morans_i(args.data_dir, args.output_dir)
    fig_key_genes_dotplot(args.data_dir, args.output_dir)
    fig_tf_activity_heatmap(args.data_dir, args.output_dir)

    print()
    print("=" * 60)
    print("Figure generation complete!")
    print("=" * 60)
    print(f"\nOutputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
