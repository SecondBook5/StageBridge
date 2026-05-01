#!/bin/bash
# Full pySCENIC regulon analysis - run separately due to environment conflicts
#
# Setup (one-time):
#   conda create -n pyscenic python=3.10 -y
#   conda activate pyscenic
#   pip install pyscenic scanpy anndata pandas numpy pyarrow matplotlib seaborn networkx
#
# Run: bash scripts/run_scenic.sh [--figures]
#
# This runs the FULL pipeline:
#   1. GRNBoost2 - infer TF-target relationships from expression
#   2. cistarget - prune with motif enrichment (requires databases)
#   3. AUCell - score regulon activity per cell
#   4. (optional) Generate publication-quality figures

set -e

# Parse args
MAKE_FIGURES=false
for arg in "$@"; do
    case $arg in
        --figures) MAKE_FIGURES=true ;;
    esac
done

DATA=/data1/chaunzt1/stagebridge/processed/luad_evo
CANONICAL=$DATA/canonical
SNRNA=$DATA/snrna_with_celltypes.h5ad

# cistarget database paths
DB_DIR=$DATA/scenic_dbs
MOTIF_DB=$DB_DIR/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
ANNOTATIONS=$DB_DIR/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
TF_LIST=$DB_DIR/allTFs_hg38.txt

echo "=============================================="
echo "pySCENIC Full Regulon Analysis"
echo "=============================================="

# Check/download databases
if [ ! -f "$MOTIF_DB" ]; then
    echo "Downloading motif database (~1.1GB)..."
    mkdir -p $DB_DIR
    wget -q --show-progress -O $MOTIF_DB \
        "https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
fi

if [ ! -f "$ANNOTATIONS" ]; then
    echo "Downloading motif annotations..."
    wget -q --show-progress -O $ANNOTATIONS \
        "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
fi

if [ ! -f "$TF_LIST" ]; then
    echo "Downloading TF list..."
    wget -q --show-progress -O $TF_LIST \
        "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt"
fi

echo "Databases ready in: $DB_DIR"

mkdir -p $CANONICAL/scenic

# =============================================================================
# Step 1: Convert h5ad to loom (pySCENIC CLI requires loom format)
# =============================================================================
LOOM_INPUT=$CANONICAL/scenic/input.loom
LOOM_OUTPUT=$CANONICAL/scenic/output.loom
ADJ_FILE=$CANONICAL/scenic/adjacencies.csv
REG_FILE=$CANONICAL/scenic/regulons.csv

if [ ! -f "$LOOM_INPUT" ]; then
    echo "Converting h5ad to loom..."
    python << CONVERT_END
import scanpy as sc
import loompy as lp
import numpy as np
import h5py
import pandas as pd
from scipy import sparse

SNRNA = '$SNRNA'
LOOM_PATH = '$LOOM_INPUT'

print(f'Loading {SNRNA}...')
# Load minimal data to avoid uns issues
with h5py.File(SNRNA, 'r') as f:
    # Get gene names
    var_names = f['var']['gene'][:].astype(str)
    # Get cell IDs
    obs_names = f['obs']['cell_id'][:].astype(str)
    # Get X (sparse)
    X_grp = f['X']
    if 'data' in X_grp:
        data = X_grp['data'][:]
        indices = X_grp['indices'][:]
        indptr = X_grp['indptr'][:]
        shape = (len(obs_names), len(var_names))
        X = sparse.csr_matrix((data, indices, indptr), shape=shape)
    else:
        X = X_grp[:]

if sparse.issparse(X):
    X = X.toarray()

print(f'  {X.shape[0]:,} cells x {X.shape[1]:,} genes')

# Create loom file
row_attrs = {'Gene': np.array(var_names)}
col_attrs = {
    'CellID': np.array(obs_names),
    'nGene': np.array(np.sum(X > 0, axis=1)).flatten(),
    'nUMI': np.array(np.sum(X, axis=1)).flatten(),
}

print(f'Writing loom file...')
lp.create(LOOM_PATH, X.T, row_attrs, col_attrs)
print(f'  Saved {LOOM_PATH}')
CONVERT_END
fi

# =============================================================================
# Step 2: GRN inference with GRNBoost2 (pyscenic grn)
# =============================================================================
if [ ! -f "$ADJ_FILE" ]; then
    echo ""
    echo "Step 1/3: Running GRNBoost2 (this takes 1-2 hours)..."
    pyscenic grn $LOOM_INPUT $TF_LIST \
        -o $ADJ_FILE \
        --num_workers 8 \
        --seed 42
    echo "  Saved adjacencies to $ADJ_FILE"
else
    echo "Step 1/3: Adjacencies exist, skipping GRN inference"
fi

# =============================================================================
# Step 3: Motif pruning with cistarget (pyscenic ctx)
# =============================================================================
if [ ! -f "$REG_FILE" ]; then
    echo ""
    echo "Step 2/3: Pruning with cistarget motifs (this takes 30-60 min)..."
    pyscenic ctx $ADJ_FILE $MOTIF_DB \
        --annotations_fname $ANNOTATIONS \
        --expression_mtx_fname $LOOM_INPUT \
        --output $REG_FILE \
        --mask_dropouts \
        --num_workers 8
    echo "  Saved regulons to $REG_FILE"
else
    echo "Step 2/3: Regulons exist, skipping motif pruning"
fi

# =============================================================================
# Step 4: AUCell scoring (pyscenic aucell)
# =============================================================================
if [ ! -f "$LOOM_OUTPUT" ]; then
    echo ""
    echo "Step 3/3: Computing AUCell scores..."
    pyscenic aucell $LOOM_INPUT $REG_FILE \
        --output $LOOM_OUTPUT \
        --num_workers 8
    echo "  Saved AUCell results to $LOOM_OUTPUT"
else
    echo "Step 3/3: AUCell results exist, skipping"
fi

# =============================================================================
# Step 5: Convert results to parquet
# =============================================================================
echo ""
echo "Converting results to parquet..."
python << EXPORT_END
import loompy as lp
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('$CANONICAL/scenic')

# Load AUCell results from loom
lf = lp.connect('$LOOM_OUTPUT', mode='r', validate=False)
auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
lf.close()

print(f'  {auc_mtx.shape[0]:,} cells x {auc_mtx.shape[1]} regulons')

# Save as parquet
auc_mtx.to_parquet(OUTPUT_DIR / 'aucell_scores.parquet')
print(f'  Saved aucell_scores.parquet')

# Save regulon summary
summary = pd.DataFrame({
    'regulon': auc_mtx.columns,
    'mean_activity': auc_mtx.mean().values,
    'std_activity': auc_mtx.std().values,
})
summary.to_parquet(OUTPUT_DIR / 'regulon_scores.parquet')
print(f'  Saved regulon_scores.parquet')

# Convert adjacencies CSV to parquet
adj = pd.read_csv(OUTPUT_DIR / 'adjacencies.csv')
adj.to_parquet(OUTPUT_DIR / 'adjacencies.parquet')
print(f'  Saved adjacencies.parquet ({len(adj):,} TF-target pairs)')
EXPORT_END

echo ""
echo "pySCENIC complete! Results in: $CANONICAL/scenic/"
echo "  - adjacencies.parquet: TF-target network"
echo "  - aucell_scores.parquet: Per-cell regulon activity"
echo "  - regulon_scores.parquet: Regulon summary stats"

# =============================================================================
# Step 6: Generate figures (optional)
# =============================================================================
if [ "$MAKE_FIGURES" = true ]; then
    echo ""
    echo "=============================================="
    echo "Generating SCENIC figures..."
    echo "=============================================="
    python scripts/run_scenic.py --figures-only
fi

exit 0

# OLD INLINE CODE (kept for reference, not executed)
python << PYTHON_END_UNUSED
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from arboreto.algo import grnboost2
from arboreto.utils import load_tf_names
from pyscenic.aucell import aucell
from pyscenic.prune import prune2df, df2regulons
from ctxcore.rnkdb import FeatherRankingDatabase

SNRNA = '$SNRNA'
OUTPUT_DIR = Path('$CANONICAL/scenic')
MOTIF_DB = Path('$MOTIF_DB')
ANNOTATIONS = Path('$ANNOTATIONS')
TF_LIST = '$TF_LIST'
N_JOBS = 8

print('Running full pySCENIC pipeline...')
print('  This takes 2-4 hours on 800k cells')

# Load data (skip problematic uns fields by reading only X, obs, var)
print(f'Loading {SNRNA}...')
import anndata as ad
import h5py
from scipy import sparse

with h5py.File(SNRNA, 'r') as f:
    # Debug: print structure
    print(f'  Keys: {list(f.keys())}')
    print(f'  var keys: {list(f["var"].keys()) if "var" in f else "N/A"}')
    print(f'  obs keys: {list(f["obs"].keys()) if "obs" in f else "N/A"}')

    # Read var names (try multiple possible keys)
    var_names = None
    for key in ['gene', 'gene_ids', 'gene_names', '_index', 'index']:
        if 'var' in f and key in f['var']:
            data = f['var'][key]
            if isinstance(data, h5py.Dataset):
                var_names = data[:].astype(str)
                print(f'  Using var key: {key}')
                break

    # Read obs names
    obs_names = None
    for key in ['cell_id', 'barcode', 'cell_ids', '_index', 'index']:
        if 'obs' in f and key in f['obs']:
            data = f['obs'][key]
            if isinstance(data, h5py.Dataset):
                obs_names = data[:].astype(str)
                print(f'  Using obs key: {key}')
                break

    # Read X (handle sparse or dense)
    if 'X' in f:
        X_grp = f['X']
        if isinstance(X_grp, h5py.Dataset):
            X = X_grp[:]
        elif 'data' in X_grp:  # sparse format
            data = X_grp['data'][:]
            indices = X_grp['indices'][:]
            indptr = X_grp['indptr'][:]
            shape = tuple(X_grp.attrs['shape']) if 'shape' in X_grp.attrs else (len(obs_names), len(var_names))
            X = sparse.csr_matrix((data, indices, indptr), shape=shape)
        else:
            raise ValueError(f'Unknown X format: {list(X_grp.keys())}')

if not sparse.issparse(X):
    X = sparse.csr_matrix(X)

adata = ad.AnnData(X=X)
if obs_names is not None:
    adata.obs_names = pd.Index(obs_names)
if var_names is not None:
    adata.var_names = pd.Index(var_names)

print(f'  {adata.n_obs:,} cells x {adata.n_vars:,} genes')

# Get expression matrix
X = adata.X
if hasattr(X, 'toarray'):
    X = X.toarray()
expr_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

# Step 1: GRN inference
adj_path = OUTPUT_DIR / 'adjacencies.parquet'
if adj_path.exists():
    print(f'Loading existing adjacencies from {adj_path}')
    adjacencies = pd.read_parquet(adj_path)
else:
    print('Step 1: Running GRNBoost2...')
    tf_list = load_tf_names(TF_LIST)
    tf_list = [tf for tf in tf_list if tf in adata.var_names]
    print(f'  Using {len(tf_list)} TFs present in data')

    adjacencies = grnboost2(
        expression_data=expr_df,
        tf_names=tf_list,
        verbose=True,
        client_or_address='local',
        seed=42,
    )
    adjacencies.to_parquet(adj_path)
    print(f'  Found {len(adjacencies):,} TF-target pairs')

# Step 2: Motif pruning
print('Step 2: Pruning with cistarget motif enrichment...')
dbs = [FeatherRankingDatabase(MOTIF_DB)]
df_motifs = prune2df(dbs, adjacencies, ANNOTATIONS, num_workers=N_JOBS)
regulons = df2regulons(df_motifs)
print(f'  Found {len(regulons)} regulons')

# Step 3: AUCell scoring
print('Step 3: Computing AUCell scores...')
auc_mtx = aucell(expr_df, regulons, num_workers=N_JOBS)
auc_path = OUTPUT_DIR / 'aucell_scores.parquet'
auc_mtx.to_parquet(auc_path)
print(f'  Computed activity for {auc_mtx.shape[0]:,} cells x {auc_mtx.shape[1]} regulons')

# Save regulon summary
print('Saving regulon summary...')
regulon_dict = {r.name: list(r.genes) for r in regulons}
regulon_sizes = {name: len(genes) for name, genes in regulon_dict.items()}
summary = pd.DataFrame({
    'regulon': list(regulon_sizes.keys()),
    'n_genes': list(regulon_sizes.values()),
    'mean_activity': [auc_mtx[r].mean() for r in regulon_sizes.keys()],
    'std_activity': [auc_mtx[r].std() for r in regulon_sizes.keys()],
})
summary.to_parquet(OUTPUT_DIR / 'regulon_scores.parquet')

print('pySCENIC complete!')
print(f'  adjacencies: {adj_path}')
print(f'  aucell: {auc_path}')
print(f'  summary: {OUTPUT_DIR / "regulon_scores.parquet"}')
PYTHON_END

echo ""
echo "pySCENIC complete! Results in: $CANONICAL/scenic/"
echo "  - adjacencies.parquet: TF-target network"
echo "  - aucell_scores.parquet: Per-cell regulon activity"
echo "  - regulon_scores.parquet: Regulon summary stats"

# =============================================================================
# Generate figures (optional)
# =============================================================================
if [ "$MAKE_FIGURES" = true ]; then
    echo ""
    echo "=============================================="
    echo "Generating SCENIC figures..."
    echo "=============================================="

    FIGURE_DIR=$CANONICAL/scenic/figures
    mkdir -p $FIGURE_DIR

python << PYTHON_END
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

CANONICAL = '$CANONICAL'
SNRNA = '$SNRNA'
FIGURE_DIR = Path('$FIGURE_DIR')

print('Loading data...')
auc = pd.read_parquet(f'{CANONICAL}/scenic/aucell_scores.parquet')
adata = sc.read_h5ad(SNRNA)

stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
stages = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
cell_types = adata.obs.get('cell_type_luca', adata.obs.get('cell_type'))

# Align indices
common = auc.index.intersection(adata.obs.index)
auc = auc.loc[common]
if stages is not None:
    stages = adata.obs.loc[common, 'stage']
if cell_types is not None:
    cell_types = adata.obs.loc[common, cell_types.name if hasattr(cell_types, 'name') else 'cell_type']

print(f'  {len(auc)} cells, {auc.shape[1]} regulons')

# -----------------------------------------------------------------------------
# Figure 1: Top regulons heatmap by stage
# -----------------------------------------------------------------------------
print('Figure 1: Regulon activity heatmap by stage...')
if stages is not None:
    stage_means = auc.groupby(stages).mean()
    stage_means = stage_means.reindex([s for s in stage_order if s in stage_means.index])

    # Top variable regulons
    regulon_var = auc.var().sort_values(ascending=False)
    top_regulons = regulon_var.head(40).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(stage_means[top_regulons].T, cmap='RdBu_r', center=0,
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={'label': 'Mean AUCell Score'})
    ax.set_xlabel('Stage')
    ax.set_ylabel('Regulon (TF)')
    ax.set_title('Top 40 Variable Regulons Across Disease Stages')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_heatmap_by_stage.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_heatmap_by_stage.png')

# -----------------------------------------------------------------------------
# Figure 2: Regulon specificity dotplot
# -----------------------------------------------------------------------------
print('Figure 2: Stage-specific regulon dotplot...')
if stages is not None:
    stage_means = auc.groupby(stages).mean()
    global_mean = auc.mean()
    global_std = auc.std()
    zscore = (stage_means - global_mean) / (global_std + 1e-8)
    zscore = zscore.reindex([s for s in stage_order if s in zscore.index])

    # Find most specific regulons per stage
    top_per_stage = []
    for stage in zscore.index:
        top = zscore.loc[stage].nlargest(8).index.tolist()
        top_per_stage.extend(top)
    top_specific = list(dict.fromkeys(top_per_stage))[:35]

    # Prepare dotplot data
    plot_data = []
    for stage in zscore.index:
        for reg in top_specific:
            plot_data.append({
                'Stage': stage,
                'Regulon': reg,
                'Z-score': zscore.loc[stage, reg],
                'Mean Activity': stage_means.loc[stage, reg],
            })
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        plot_df['Stage'].map({s: i for i, s in enumerate(stage_order)}),
        plot_df['Regulon'],
        c=plot_df['Z-score'],
        s=np.abs(plot_df['Z-score']) * 50 + 20,
        cmap='RdBu_r',
        vmin=-3, vmax=3,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
    )
    ax.set_xticks(range(len([s for s in stage_order if s in zscore.index])))
    ax.set_xticklabels([s for s in stage_order if s in zscore.index])
    ax.set_xlabel('Stage')
    ax.set_ylabel('Regulon (TF)')
    ax.set_title('Stage-Specific Regulon Activity')
    cbar = plt.colorbar(scatter, ax=ax, label='Z-score')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_specificity_dotplot.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_specificity_dotplot.png')

# -----------------------------------------------------------------------------
# Figure 3: Top regulons UMAP
# -----------------------------------------------------------------------------
print('Figure 3: UMAP colored by top regulons...')
if 'X_umap' in adata.obsm:
    umap = adata.obsm['X_umap'][adata.obs.index.isin(common)]

    # Top 6 most variable regulons
    top6 = auc.var().nlargest(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, reg in zip(axes.flat, top6):
        scores = auc[reg].values
        vmax = np.percentile(scores, 99)
        scatter = ax.scatter(umap[:, 0], umap[:, 1], c=scores, s=0.5,
                           cmap='viridis', vmin=0, vmax=vmax, alpha=0.7)
        ax.set_title(f'{reg} regulon', fontsize=12)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax, shrink=0.8)
    plt.suptitle('Top Variable Regulon Activity on UMAP', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_umap_top6.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_umap_top6.png')

# -----------------------------------------------------------------------------
# Figure 4: Regulon activity by cell type
# -----------------------------------------------------------------------------
print('Figure 4: Regulon activity by cell type...')
if cell_types is not None:
    ct_means = auc.groupby(cell_types).mean()

    # Filter to cell types with enough cells
    ct_counts = cell_types.value_counts()
    major_cts = ct_counts[ct_counts >= 100].index.tolist()[:20]
    ct_means = ct_means.loc[ct_means.index.isin(major_cts)]

    # Top variable across cell types
    ct_var = ct_means.var().sort_values(ascending=False)
    top_ct_regs = ct_var.head(30).index.tolist()

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(ct_means[top_ct_regs].T, cmap='YlOrRd',
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={'label': 'Mean AUCell Score'})
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Regulon (TF)')
    ax.set_title('Top Regulons by Cell Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_heatmap_by_celltype.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_heatmap_by_celltype.png')

# -----------------------------------------------------------------------------
# Figure 5: TF-target network (top regulons)
# -----------------------------------------------------------------------------
print('Figure 5: TF-target network...')
adj_path = Path(f'{CANONICAL}/scenic/adjacencies.parquet')
if adj_path.exists():
    try:
        import networkx as nx
        adj = pd.read_parquet(adj_path)

        # Top TFs by total importance
        tf_importance = adj.groupby('TF')['importance'].sum().nlargest(15)
        top_tfs = tf_importance.index.tolist()

        # Build network
        G = nx.DiGraph()
        for tf in top_tfs:
            G.add_node(tf, node_type='TF')
            targets = adj[adj['TF'] == tf].nlargest(5, 'importance')
            for _, row in targets.iterrows():
                G.add_node(row['target'], node_type='target')
                G.add_edge(tf, row['target'], weight=row['importance'])

        fig, ax = plt.subplots(figsize=(14, 14))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw TFs
        tf_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'TF']
        target_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']

        nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_color='#E74C3C',
                              node_size=800, alpha=0.9, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='#3498DB',
                              node_size=300, alpha=0.7, ax=ax)

        # Edges
        edges = G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        max_w = max(weights) if weights else 1
        edge_widths = [2 * w / max_w + 0.5 for w in weights]
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=edge_widths,
                              edge_color='gray', arrows=True,
                              arrowsize=15, ax=ax)

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        ax.set_title('Top TF-Target Regulatory Network\n(Red=TF, Blue=Target)', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / 'tf_target_network.png', bbox_inches='tight')
        plt.close()
        print('  Saved tf_target_network.png')
    except ImportError:
        print('  networkx not installed, skipping network figure')

# -----------------------------------------------------------------------------
# Figure 6: Binarized regulon activity
# -----------------------------------------------------------------------------
print('Figure 6: Binarized regulon states...')
if stages is not None:
    # Binarize using Otsu-like threshold per regulon
    binary = pd.DataFrame(index=auc.index)
    for col in auc.columns[:30]:  # Top 30
        threshold = auc[col].quantile(0.75)
        binary[col] = (auc[col] > threshold).astype(int)

    # Proportion "on" per stage
    binary['stage'] = stages.values
    prop_on = binary.groupby('stage').mean()
    prop_on = prop_on.reindex([s for s in stage_order if s in prop_on.index])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(prop_on.T, cmap='YlGnBu', vmin=0, vmax=1,
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={'label': 'Proportion Active'})
    ax.set_xlabel('Stage')
    ax.set_ylabel('Regulon (TF)')
    ax.set_title('Proportion of Cells with Active Regulon (>75th percentile)')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_binary_heatmap.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_binary_heatmap.png')

# -----------------------------------------------------------------------------
# Figure 7: Progression-associated regulons
# -----------------------------------------------------------------------------
print('Figure 7: Progression trajectory regulons...')
if stages is not None:
    stage_num = stages.map({'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4})
    valid = ~stage_num.isna()

    # Correlation with progression
    correlations = {}
    for col in auc.columns:
        if valid.sum() > 100:
            correlations[col] = np.corrcoef(stage_num[valid], auc.loc[valid.values, col])[0, 1]

    corr_df = pd.Series(correlations).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top increasing
    top_increasing = corr_df.tail(15)
    axes[0].barh(range(len(top_increasing)), top_increasing.values, color='#E74C3C')
    axes[0].set_yticks(range(len(top_increasing)))
    axes[0].set_yticklabels(top_increasing.index)
    axes[0].set_xlabel('Correlation with Progression')
    axes[0].set_title('Regulons INCREASING with Progression')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)

    # Top decreasing
    top_decreasing = corr_df.head(15)
    axes[1].barh(range(len(top_decreasing)), top_decreasing.values, color='#3498DB')
    axes[1].set_yticks(range(len(top_decreasing)))
    axes[1].set_yticklabels(top_decreasing.index)
    axes[1].set_xlabel('Correlation with Progression')
    axes[1].set_title('Regulons DECREASING with Progression')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('Regulons Associated with Disease Progression', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'regulon_progression_correlation.png', bbox_inches='tight')
    plt.close()
    print('  Saved regulon_progression_correlation.png')

print('')
print(f'All figures saved to: {FIGURE_DIR}')
PYTHON_END

    echo ""
    echo "Figures saved to: $FIGURE_DIR"
fi
