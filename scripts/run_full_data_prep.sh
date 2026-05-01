#!/bin/bash
# Full data preparation for StageBridge (with skip-if-exists logic)
# Run with: srun --partition=gpu --gres=gpu:1 --mem=256G --time=8:00:00 --account=chaunzt1 --pty bash
# Then: bash scripts/run_full_data_prep.sh

set -e

# Paths
DATA=/data1/chaunzt1/stagebridge/processed/luad_evo
CANONICAL=$DATA/canonical
SNRNA=$DATA/snrna_with_celltypes.h5ad
SPATIAL=$DATA/spatial_merged.h5ad

echo "=============================================="
echo "StageBridge Full Data Preparation"
echo "=============================================="

# =============================================================================
# Skip logic helper
# =============================================================================
run_step() {
    local step_num=$1
    local step_name=$2
    local check_file=$3
    shift 3
    
    echo ""
    echo "[$step_num] $step_name..."
    
    if [ -f "$check_file" ]; then
        echo "  SKIP: $check_file exists"
        return 0
    fi
    
    "$@"
}

# =============================================================================
# Step 1: Main data prep with LIANA
# =============================================================================
echo ""
echo "[1/26] Main data prep with LIANA..."
if [ -f "$CANONICAL/liana_interactions.parquet" ]; then
    echo "  SKIP: LIANA results exist"
else
    python -m stagebridge.pipelines.prepare_data \
        --cells $CANONICAL/cells.parquet \
        --output-dir $CANONICAL \
        --destvi-luca /data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet \
        --destvi-hlca /data1/chaunzt1/stagebridge/results/spatial_benchmark/hlca/destvi/cell_type_proportions.parquet \
        --progression $CANONICAL/progression/progression_scores.parquet \
        --h5ad $SNRNA \
        --run-liana \
        --figures
fi

# =============================================================================
# Step 2: SCENIC - run separately
# =============================================================================
echo ""
echo "[2/26] SCENIC regulon analysis..."
echo "  SKIP: Run separately with: bash scripts/run_scenic.sh"

# =============================================================================
# Step 3: Squidpy spatial statistics
# =============================================================================
echo ""
echo "[3/26] Squidpy spatial statistics..."
if [ -f "$CANONICAL/spatial_stats/nhood_enrichment.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/spatial_stats
python << 'PYTHON_END'
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading spatial data...')
adata = sc.read_h5ad(SPATIAL)
print(f'  {adata.n_obs} spots')

out = Path(CANONICAL) / 'spatial_stats'

print('Computing spatial neighbors...')
sq.gr.spatial_neighbors(adata, coord_type='generic')

print('Computing neighborhood enrichment...')
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    sq.gr.nhood_enrichment(adata, cluster_key=cell_type_key)
    zscore = adata.uns[f'{cell_type_key}_nhood_enrichment']['zscore']
    pd.DataFrame(zscore, index=adata.obs[cell_type_key].cat.categories,
                 columns=adata.obs[cell_type_key].cat.categories).to_parquet(out / 'nhood_enrichment.parquet')
    print('  Saved nhood_enrichment.parquet')

print('Computing Morans I...')
key_genes = ['IL1B', 'IL1R1', 'CXCL12', 'CXCR4', 'EGFR', 'SOX9', 'KRT17', 'VIM', 'CDH1', 'ACTA2', 'COL1A1', 'CD274', 'PDCD1']
available_genes = [g for g in key_genes if g in adata.var_names]
if available_genes:
    sq.gr.spatial_autocorr(adata, genes=available_genes, mode='moran')
    adata.uns['moranI'].to_parquet(out / 'morans_i.parquet')
    print(f'  Saved morans_i.parquet')

print('Squidpy complete')
PYTHON_END
fi

# =============================================================================
# Step 4: Differential expression (sequential with progress)
# =============================================================================
echo ""
echo "[4/26] Differential expression..."
# Check all 5 stages exist, not just Normal
if [ -f "$CANONICAL/de_analysis/de_stage_Normal.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_AAH.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_AIS.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_MIA.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_LUAD.parquet" ]; then
    echo "  SKIP: all stages exist"
else
mkdir -p $CANONICAL/de_analysis
python -u << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...', flush=True)
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs} cells x {adata.n_vars} genes', flush=True)

out = Path(CANONICAL) / 'de_analysis'
stages = adata.obs['stage'].unique().tolist()
print(f'  Stages: {stages}', flush=True)

# Run DE for each stage vs rest
for stage in tqdm(stages, desc='DE by stage'):
    out_file = out / f'de_stage_{stage}.parquet'

    if out_file.exists():
        tqdm.write(f'  {stage}: SKIP (exists)')
        continue

    # Subset to speed up: this stage + random sample of others
    stage_mask = adata.obs['stage'] == stage
    n_stage = stage_mask.sum()

    # For "rest", sample min(available, 50k) for speed
    rest_mask = ~stage_mask
    n_rest = min(rest_mask.sum(), 50000)

    rest_idx = np.random.choice(np.where(rest_mask)[0], size=n_rest, replace=False)
    stage_idx = np.where(stage_mask)[0]

    subset_idx = np.concatenate([stage_idx, rest_idx])
    adata_sub = adata[subset_idx].copy()
    adata_sub.obs['_group'] = (adata_sub.obs['stage'] == stage).map({True: stage, False: 'rest'})

    tqdm.write(f'  {stage}: {n_stage} vs {n_rest} cells')

    # Run wilcoxon
    sc.tl.rank_genes_groups(adata_sub, groupby='_group', groups=[stage],
                            reference='rest', method='wilcoxon', n_genes=500)

    df = sc.get.rank_genes_groups_df(adata_sub, group=stage)
    df.to_parquet(out_file)
    tqdm.write(f'  {stage}: saved {len(df)} genes')

print('DE complete', flush=True)
PYTHON_END
fi

# =============================================================================
# Step 5: Summary statistics
# =============================================================================
echo ""
echo "[5/26] Summary statistics..."
if [ -f "$CANONICAL/summary_stats/celltype_proportions_by_stage.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/summary_stats
python << 'PYTHON_END'
import pandas as pd
import numpy as np
from pathlib import Path
import os

CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading neighborhoods...')
df = pd.read_parquet(f'{CANONICAL}/neighborhoods.parquet')
print(f'  {len(df)} cells')

out = Path(CANONICAL) / 'summary_stats'

cell_type_key = 'cell_type_luca' if 'cell_type_luca' in df.columns else 'cell_type'
if cell_type_key in df.columns and 'stage' in df.columns:
    props = df.groupby(['stage', cell_type_key]).size().unstack(fill_value=0)
    props_norm = props.div(props.sum(axis=1), axis=0)
    props_norm.to_parquet(out / 'celltype_proportions_by_stage.parquet')
    print('  Saved celltype_proportions_by_stage.parquet')

if cell_type_key in df.columns and 'donor_id' in df.columns:
    props = df.groupby(['donor_id', cell_type_key]).size().unstack(fill_value=0)
    props_norm = props.div(props.sum(axis=1), axis=0)
    props_norm.to_parquet(out / 'celltype_proportions_by_donor.parquet')
    print('  Saved celltype_proportions_by_donor.parquet')

numeric_cols = ['caf_fraction', 'immune_fraction', 'diversity', 'malignant_fraction',
                'S_score', 'G2M_score', 'emt_score', 'senescence_score', 'sasp_score']
available = [c for c in numeric_cols if c in df.columns]
if available and 'stage' in df.columns:
    stage_means = df.groupby('stage')[available].agg(['mean', 'std', 'median'])
    stage_means.to_parquet(out / 'feature_stats_by_stage.parquet')
    print('  Saved feature_stats_by_stage.parquet')

print('Summary stats complete')
PYTHON_END
fi

# =============================================================================
# Step 6: Gene signatures
# =============================================================================
echo ""
echo "[6/26] Gene signatures..."
if [ -f "$CANONICAL/signatures/gene_signatures.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/signatures
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs} cells')

out = Path(CANONICAL) / 'signatures'

signatures = {
    'ap1_stress': ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND', 'ATF3', 'ATF4', 'EGR1'],
    'hypoxia': ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA', 'CA9', 'BNIP3', 'PDK1'],
    'glycolysis': ['HK1', 'HK2', 'GPI', 'PFKP', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'PKM', 'LDHA'],
    'ifn_gamma': ['STAT1', 'IRF1', 'GBP1', 'GBP2', 'IDO1', 'CXCL9', 'CXCL10', 'CXCL11'],
    'tgfb_response': ['TGFBI', 'SERPINE1', 'CTGF', 'COL1A1', 'COL3A1', 'FN1', 'ACTA2', 'TAGLN'],
    'stemness': ['SOX2', 'SOX9', 'NANOG', 'POU5F1', 'KLF4', 'MYC', 'LGR5', 'ALDH1A1', 'CD44', 'PROM1'],
    'il1b_pathway': ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN', 'NLRP3', 'CASP1', 'PYCARD'],
    'nfkb': ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL', 'IKBKB', 'IKBKG', 'NFKBIA'],
}

results = {'cell_id': adata.obs.index.tolist()}
for name, genes in signatures.items():
    available = [g for g in genes if g in adata.var_names]
    if len(available) >= 3:
        print(f'  Scoring {name}...')
        sc.tl.score_genes(adata, gene_list=available, score_name=name)
        results[name] = adata.obs[name].tolist()

sig_df = pd.DataFrame(results)
sig_df.to_parquet(out / 'gene_signatures.parquet')
print('Signatures complete')
PYTHON_END
fi

# =============================================================================
# Step 7: Key gene expression
# =============================================================================
echo ""
echo "[7/26] Key gene expression..."
if [ -f "$CANONICAL/expression/key_genes_expression.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/expression
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
out = Path(CANONICAL) / 'expression'

key_genes = ['IL1B', 'IL1R1', 'VIM', 'CDH1', 'KRT17', 'SOX9', 'ACTA2', 'FAP', 'COL1A1',
             'CD68', 'CD163', 'CD3D', 'CD8A', 'FOXP3', 'MKI67', 'CD274', 'PDCD1', 'EGFR', 'VEGFA']
available = [g for g in key_genes if g in adata.var_names]
print(f'  Extracting {len(available)} genes')

if hasattr(adata.X, 'toarray'):
    expr = pd.DataFrame(adata[:, available].X.toarray(), index=adata.obs.index, columns=available)
else:
    expr = pd.DataFrame(adata[:, available].X, index=adata.obs.index, columns=available)

expr['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    expr['cell_type'] = adata.obs[cell_type_key].values

expr.to_parquet(out / 'key_genes_expression.parquet')

if 'stage' in expr.columns and 'cell_type' in expr.columns:
    agg = expr.groupby(['stage', 'cell_type'])[available].mean()
    agg.to_parquet(out / 'key_genes_mean_by_stage_celltype.parquet')

print('Key genes complete')
PYTHON_END
fi

# =============================================================================
# Step 8: GSEA
# =============================================================================
echo ""
echo "[8/26] GSEA pathway enrichment..."
if [ -f "$CANONICAL/pathways/gsea_hallmark_Normal.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/pathways
python << 'PYTHON_END'
import pandas as pd
import numpy as np
from pathlib import Path
import os

CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'pathways'
de_dir = Path(CANONICAL) / 'de_analysis'

try:
    import gseapy as gp
    print('Running GSEA...')
    
    de_files = list(de_dir.glob('de_stage_*.parquet'))
    for de_file in de_files:
        stage = de_file.stem.replace('de_stage_', '')
        print(f'  {stage}...')
        
        de_df = pd.read_parquet(de_file)
        de_df['rank'] = -np.log10(de_df['pvals'] + 1e-300) * np.sign(de_df['logfoldchanges'])
        ranked = de_df.set_index('names')['rank'].sort_values(ascending=False)
        
        try:
            hallmark = gp.prerank(rnk=ranked, gene_sets='MSigDB_Hallmark_2020',
                                 outdir=None, min_size=15, max_size=500, permutation_num=100, verbose=False)
            hallmark.res2d.to_parquet(out / f'gsea_hallmark_{stage}.parquet')
        except Exception as e:
            print(f'    Hallmark failed: {e}')
    
    print('GSEA complete')
except ImportError:
    print('gseapy not installed')
PYTHON_END
fi

# =============================================================================
# Step 9: decoupleR
# =============================================================================
echo ""
echo "[9/26] decoupleR TF/pathway activity..."
if [ -f "$CANONICAL/activity/pathway_activity_progeny.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/activity
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'activity'

try:
    import decoupler as dc
    print('Running decoupleR...')

    adata = sc.read_h5ad(SNRNA)
    print(f'  {adata.n_obs} cells')

    print('  TF activity (CollecTRI)...')
    collectri = dc.op.collectri(organism='human')
    dc.mt.ulm(data=adata, net=collectri)
    tf_acts = dc.pp.get_obsm(adata=adata, key='score_ulm')
    pd.DataFrame(tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index).to_parquet(out / 'tf_activity_collectri.parquet')

    print('  Pathway activity (PROGENy)...')
    progeny = dc.op.progeny(organism='human')
    dc.mt.ulm(data=adata, net=progeny)
    pathway_acts = dc.pp.get_obsm(adata=adata, key='score_ulm')
    pd.DataFrame(pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index).to_parquet(out / 'pathway_activity_progeny.parquet')

    if 'stage' in adata.obs.columns:
        tf_df = pd.DataFrame(tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index)
        tf_df['stage'] = adata.obs['stage'].values
        tf_df.groupby('stage').mean().to_parquet(out / 'tf_activity_by_stage.parquet')

        pw_df = pd.DataFrame(pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index)
        pw_df['stage'] = adata.obs['stage'].values
        pw_df.groupby('stage').mean().to_parquet(out / 'pathway_activity_by_stage.parquet')

    print('decoupleR complete')
except ImportError:
    print('decoupleR not installed')
PYTHON_END
fi

# =============================================================================
# Step 10: Trajectories
# =============================================================================
echo ""
echo "[10/26] Diffusion pseudotime / PAGA..."
if [ -f "$CANONICAL/trajectories/diffusion_pseudotime.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/trajectories
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
out = Path(CANONICAL) / 'trajectories'

if 'X_pca' not in adata.obsm:
    sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')

print('Computing diffusion map...')
sc.tl.diffmap(adata, n_comps=15)
pd.DataFrame(adata.obsm['X_diffmap'], index=adata.obs.index).to_parquet(out / 'diffmap_embedding.parquet')

if 'stage' in adata.obs.columns and 'Normal' in adata.obs['stage'].values:
    print('Computing diffusion pseudotime...')
    root_mask = adata.obs['stage'] == 'Normal'
    root_indices = np.where(root_mask)[0]
    diffmap = adata.obsm['X_diffmap']
    centroid = diffmap[root_mask].mean(axis=0)
    distances = np.linalg.norm(diffmap[root_mask] - centroid, axis=1)
    root_cell = root_indices[np.argmin(distances)]
    adata.uns['iroot'] = root_cell
    sc.tl.dpt(adata, n_branchings=0)
    pd.DataFrame({
        'cell_id': adata.obs.index,
        'dpt_pseudotime': adata.obs['dpt_pseudotime'].values,
        'stage': adata.obs['stage'].values,
    }).to_parquet(out / 'diffusion_pseudotime.parquet')

print('Trajectory complete')
PYTHON_END
fi

# =============================================================================
# Step 11: Embeddings (UMAP/PHATE) + clustering
# =============================================================================
echo ""
echo "[11/26] UMAP/PHATE embeddings + clustering..."
if [ -f "$CANONICAL/embeddings/umap_embedding.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/embeddings
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
out = Path(CANONICAL) / 'embeddings'

cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'

if 'X_pca' not in adata.obsm:
    print('Computing PCA...')
    sc.pp.pca(adata, n_comps=50)

if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')

print('Computing UMAP...')
sc.tl.umap(adata)
umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs.index, columns=['UMAP1', 'UMAP2'])
umap_df['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
if cell_type_key in adata.obs.columns:
    umap_df['cell_type'] = adata.obs[cell_type_key].values

# PHATE
phate_emb = None
try:
    import phate
    print('Computing PHATE...')
    phate_op = phate.PHATE(n_components=2, n_jobs=-1, random_state=42)
    phate_emb = phate_op.fit_transform(adata.obsm['X_pca'][:, :50])
    adata.obsm['X_phate'] = phate_emb
    phate_df = pd.DataFrame(phate_emb, index=adata.obs.index, columns=['PHATE1', 'PHATE2'])
    phate_df['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
    if cell_type_key in adata.obs.columns:
        phate_df['cell_type'] = adata.obs[cell_type_key].values
    phate_df.to_parquet(out / 'phate_embedding.parquet')
except ImportError:
    print('PHATE not installed')
except Exception as e:
    print(f'PHATE failed: {e}')

# Clustering (PCA-based)
print('Computing Leiden/Louvain clustering...')
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    sc.tl.louvain(adata, resolution=res, key_added=f'louvain_{res}')

# PHATE-based clustering
if 'X_phate' in adata.obsm:
    print('Computing PHATE-based clustering...')
    sc.pp.neighbors(adata, use_rep='X_phate', n_neighbors=30, key_added='phate_neighbors')
    for res in [0.5, 1.0]:
        sc.tl.leiden(adata, resolution=res, neighbors_key='phate_neighbors', key_added=f'leiden_phate_{res}')
        sc.tl.louvain(adata, resolution=res, neighbors_key='phate_neighbors', key_added=f'louvain_phate_{res}')

# Save clustering
cluster_df = pd.DataFrame({'cell_id': adata.obs.index})
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    cluster_df[f'leiden_{res}'] = adata.obs[f'leiden_{res}'].values
    cluster_df[f'louvain_{res}'] = adata.obs[f'louvain_{res}'].values
if 'leiden_phate_0.5' in adata.obs.columns:
    for res in [0.5, 1.0]:
        cluster_df[f'leiden_phate_{res}'] = adata.obs[f'leiden_phate_{res}'].values
        cluster_df[f'louvain_phate_{res}'] = adata.obs[f'louvain_phate_{res}'].values
cluster_df.to_parquet(out / 'clustering.parquet')

# Add clusters to UMAP
for res in [0.5, 1.0]:
    umap_df[f'leiden_{res}'] = adata.obs[f'leiden_{res}'].values
    if f'leiden_phate_{res}' in adata.obs.columns:
        umap_df[f'leiden_phate_{res}'] = adata.obs[f'leiden_phate_{res}'].values
umap_df.to_parquet(out / 'umap_embedding.parquet')

print('Embeddings complete')
PYTHON_END
fi

# =============================================================================
# Step 12: Communication summary
# =============================================================================
echo ""
echo "[12/26] Cell-cell communication summary..."
if [ -f "$CANONICAL/communication/communication_matrix.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/communication
python << 'PYTHON_END'
import pandas as pd
import numpy as np
from pathlib import Path
import os

CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'communication'
liana_path = Path(CANONICAL) / 'liana_interactions.parquet'

if liana_path.exists():
    print('Processing LIANA results...')
    liana = pd.read_parquet(liana_path)
    
    rank_col = 'specificity_rank' if 'specificity_rank' in liana.columns else 'pvalue'
    liana.nsmallest(100, rank_col).to_parquet(out / 'top_interactions.parquet')
    
    if 'source' in liana.columns and 'target' in liana.columns:
        ct_comm = liana.groupby(['source', 'target']).size().reset_index(name='n_interactions')
        ct_comm.to_parquet(out / 'celltype_communication_counts.parquet')
        
        comm_matrix = ct_comm.pivot(index='source', columns='target', values='n_interactions').fillna(0)
        comm_matrix.to_parquet(out / 'communication_matrix.parquet')
        
        outgoing = comm_matrix.sum(axis=1)
        incoming = comm_matrix.sum(axis=0)
        centrality = pd.DataFrame({
            'outgoing_strength': outgoing,
            'incoming_strength': incoming,
            'total_strength': outgoing + incoming,
        })
        centrality.to_parquet(out / 'communication_centrality.parquet')
    
    print('Communication summary complete')
else:
    print('LIANA results not found')
PYTHON_END
fi

# =============================================================================
# Step 13: Visium analysis
# =============================================================================
echo ""
echo "[13/26] Visium spatial analysis..."
if [ -f "$CANONICAL/visium/celltype_colocalization_corr.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/visium
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os

SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading spatial data...')
adata = sc.read_h5ad(SPATIAL)
out = Path(CANONICAL) / 'visium'

# DestVI results
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
if destvi_path.exists():
    print('Processing DestVI...')
    deconv = pd.read_parquet(destvi_path)
    deconv.to_parquet(out / 'spot_deconvolution_destvi.parquet')
    
    exclude_cols = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id', 'batch']
    ct_cols = [c for c in deconv.columns if c not in exclude_cols and deconv[c].dtype in ['float64', 'float32']]
    
    # Co-localization
    print('  Computing co-localization...')
    ct_corr = deconv[ct_cols].corr(method='pearson')
    ct_corr.to_parquet(out / 'celltype_colocalization_corr.parquet')
    
    # Clustering
    dist_matrix = 1 - ct_corr.values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 2)
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    modules = fcluster(linkage_matrix, t=4, criterion='maxclust')
    pd.DataFrame({'cell_type': ct_cols, 'colocalization_module': modules}).to_parquet(out / 'celltype_colocalization_modules.parquet')
    
    if 'stage' in adata.obs.columns:
        deconv['stage'] = adata.obs['stage'].values[:len(deconv)]
        deconv.groupby('stage')[ct_cols].mean().to_parquet(out / 'deconv_proportions_by_stage_destvi.parquet')

print('Visium complete')
PYTHON_END
fi

# =============================================================================
# Step 14: Rare cells
# =============================================================================
echo ""
echo "[14/26] Rare cell signatures..."
if [ -f "$CANONICAL/rare_cells/rare_cell_signatures.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/rare_cells
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
out = Path(CANONICAL) / 'rare_cells'

cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'

# Rare cell signatures
signatures = {
    'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'IRF8', 'CADM1'],
    'LAMP3_DC': ['LAMP3', 'CCR7', 'CCL19', 'CCL22', 'FSCN1'],
    'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2'],
    'exhausted_CD8': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX'],
    'plasma_cell': ['JCHAIN', 'MZB1', 'XBP1', 'SDC1'],
    'hypoxic_tumor': ['CA9', 'VEGFA', 'SLC2A1', 'LDHA', 'HIF1A'],
    'EMT_tumor': ['VIM', 'SNAI1', 'SNAI2', 'ZEB1', 'TWIST1'],
    'myCAF': ['ACTA2', 'TAGLN', 'MYL9', 'POSTN'],
    'iCAF': ['IL6', 'CXCL12', 'PDGFRA', 'CFD'],
    'lymphatic_endo': ['PROX1', 'LYVE1', 'PDPN', 'FLT4'],
}

results = {'cell_id': adata.obs.index.tolist()}
for name, genes in signatures.items():
    available = [g for g in genes if g in adata.var_names]
    if len(available) >= 2:
        print(f'  Scoring {name}...')
        sc.tl.score_genes(adata, gene_list=available, score_name=name)
        results[name] = adata.obs[name].tolist()

sig_df = pd.DataFrame(results)
sig_df.to_parquet(out / 'rare_cell_signatures.parquet')

if 'stage' in adata.obs.columns:
    sig_df['stage'] = adata.obs['stage'].values
    score_cols = [c for c in sig_df.columns if c not in ['cell_id', 'stage']]
    sig_df.groupby('stage')[score_cols].mean().to_parquet(out / 'rare_signatures_by_stage.parquet')

if cell_type_key in adata.obs.columns:
    sig_df['cell_type'] = adata.obs[cell_type_key].values
    score_cols = [c for c in sig_df.columns if c not in ['cell_id', 'stage', 'cell_type']]
    sig_df.groupby('cell_type')[score_cols].mean().to_parquet(out / 'rare_signatures_by_celltype.parquet')

print('Rare cells complete')
PYTHON_END
fi

# =============================================================================
# Step 15: Niche phenotyping
# =============================================================================
echo ""
echo "[15/26] Spatial niche phenotyping..."
if [ -f "$CANONICAL/niche_phenotypes/spot_niche_phenotypes.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/niche_phenotypes
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture
import os

SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Loading spatial data...')
adata = sc.read_h5ad(SPATIAL)
out = Path(CANONICAL) / 'niche_phenotypes'

# Load DestVI proportions
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
props = pd.read_parquet(destvi_path) if destvi_path.exists() else None

# Get coordinates
if 'spatial' in adata.obsm:
    coords = adata.obsm['spatial']
else:
    coords = np.column_stack([adata.obs['x_spatial'].values, adata.obs['y_spatial'].values])

# Use DestVI or PCA
if props is not None and len(props) == adata.n_obs:
    exclude_cols = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id']
    prop_cols = [c for c in props.columns if c not in exclude_cols and props[c].dtype in ['float64', 'float32']]
    X = props[prop_cols].values
else:
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)
    X = adata.obsm['X_pca'][:, :20]

# HMRF via ICM
print('Running HMRF-ICM...')
tree = KDTree(coords)
_, neighbor_idx = tree.query(coords, k=7)
neighbor_idx = neighbor_idx[:, 1:]

n_phenotypes = 8
beta = 2.0

gmm = GaussianMixture(n_components=n_phenotypes, random_state=42, n_init=5)
gmm.fit(X)
z = gmm.predict(X)

for iteration in range(50):
    z_old = z.copy()
    changes = 0
    for i in np.random.permutation(len(z)):
        energies = np.zeros(n_phenotypes)
        for k in range(n_phenotypes):
            energies[k] = -gmm._estimate_weighted_log_prob(X[i:i+1])[0, k]
            energies[k] += beta * np.sum(z[neighbor_idx[i]] != k)
        new_label = np.argmin(energies)
        if new_label != z[i]:
            changes += 1
        z[i] = new_label
    if changes < len(z) * 0.001:
        print(f'  Converged at iteration {iteration + 1}')
        break

adata.obs['niche_phenotype'] = z

# Save
phenotype_df = pd.DataFrame({
    'spot_id': adata.obs.index,
    'niche_phenotype': z,
    'x': coords[:, 0],
    'y': coords[:, 1],
})
if 'stage' in adata.obs.columns:
    phenotype_df['stage'] = adata.obs['stage'].values
if 'sample' in adata.obs.columns:
    phenotype_df['sample'] = adata.obs['sample'].values
phenotype_df.to_parquet(out / 'spot_niche_phenotypes.parquet')

# Phenotype centers
if props is not None:
    props['niche_phenotype'] = z
    centers = props.groupby('niche_phenotype')[prop_cols].mean()
    centers['n_spots'] = pd.Series(z).value_counts().sort_index()
    
    phenotype_names = []
    for idx in centers.index:
        top3 = centers.loc[idx, prop_cols].nlargest(3).index.tolist()
        name = '_'.join([t[:10] for t in top3])
        phenotype_names.append(f'P{idx}_{name}')
    centers['phenotype_name'] = phenotype_names
    centers.to_parquet(out / 'phenotype_centers.parquet')
    
    phenotype_df['phenotype_name'] = phenotype_df['niche_phenotype'].map(dict(zip(centers.index, phenotype_names)))
    phenotype_df.to_parquet(out / 'spot_niche_phenotypes.parquet')

# By stage
if 'stage' in phenotype_df.columns:
    pd.crosstab(phenotype_df['stage'], phenotype_df['niche_phenotype'], normalize='index').to_parquet(out / 'phenotype_by_stage.parquet')

print('Niche phenotyping complete')
PYTHON_END
fi

# =============================================================================
# Step 16: QC metrics
# =============================================================================
echo ""
echo "[16/26] QC metrics..."
if [ -f "$CANONICAL/qc/snrna_qc_metrics.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/qc
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')

print('Computing QC metrics...')
adata = sc.read_h5ad(SNRNA)
out = Path(CANONICAL) / 'qc'

if 'pct_counts_mt' not in adata.obs.columns:
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

if 'pct_counts_ribo' not in adata.obs.columns:
    adata.var['ribo'] = adata.var_names.str.match('^RP[SL]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)

qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']
qc_df = adata.obs[qc_cols].copy()
qc_df['cell_id'] = adata.obs.index
if 'stage' in adata.obs.columns:
    qc_df['stage'] = adata.obs['stage'].values

qc_df.to_parquet(out / 'snrna_qc_metrics.parquet')

if 'stage' in qc_df.columns:
    qc_df.groupby('stage')[qc_cols].agg(['mean', 'median', 'std']).to_parquet(out / 'qc_by_stage.parquet')

print('QC complete')
PYTHON_END
fi

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Outputs in: $CANONICAL/"
echo ""
echo "Generate figures with:"
echo "  python scripts/generate_eda_figures.py --data-dir $CANONICAL --output-dir figures/eda"
