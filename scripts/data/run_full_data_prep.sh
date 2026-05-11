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
# Step 1: Main data prep with LIANA
# =============================================================================
echo ""
echo "[1/14] Main data prep with LIANA..."
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
# Step 2: Squidpy spatial statistics
# =============================================================================
echo ""
echo "[2/14] Squidpy spatial statistics..."
if [ -f "$CANONICAL/spatial_stats/nhood_enrichment.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.preprocessing.spatial_stats \
        --spatial $SPATIAL \
        --output $CANONICAL/spatial_stats
fi

# =============================================================================
# Step 3: Differential expression
# =============================================================================
echo ""
echo "[3/14] Differential expression..."
if [ -f "$CANONICAL/de_analysis/de_stage_Normal.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_AAH.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_AIS.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_MIA.parquet" ] && \
   [ -f "$CANONICAL/de_analysis/de_stage_LUAD.parquet" ]; then
    echo "  SKIP: all stages exist"
else
    python -m stagebridge.preprocessing.de_analysis \
        --h5ad $SNRNA \
        --output $CANONICAL/de_analysis
fi

# =============================================================================
# Step 4: Summary statistics
# =============================================================================
echo ""
echo "[4/14] Summary statistics..."
if [ -f "$CANONICAL/summary_stats/celltype_proportions_by_stage.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.preprocessing.summary_stats \
        --neighborhoods $CANONICAL/neighborhoods.parquet \
        --output $CANONICAL/summary_stats
fi

# =============================================================================
# Step 5: Gene signatures (EMT, senescence, key genes)
# =============================================================================
echo ""
echo "[5/14] Gene signatures..."
if [ -f "$CANONICAL/signatures/gene_signatures.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/signatures
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

from stagebridge.biology import (
    compute_emt_score,
    compute_senescence_score,
    compute_sasp_score,
)

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'signatures'

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs} cells')

print('Computing gene signatures...')
signatures = pd.DataFrame({'cell_id': adata.obs.index})

signatures['emt_score'] = compute_emt_score(adata)
signatures['senescence_score'] = compute_senescence_score(adata)
signatures['sasp_score'] = compute_sasp_score(adata)

if 'stage' in adata.obs.columns:
    signatures['stage'] = adata.obs['stage'].values

signatures.to_parquet(out / 'gene_signatures.parquet')
print(f'  Saved gene_signatures.parquet')

# Stage summaries
if 'stage' in signatures.columns:
    summary = signatures.groupby('stage')[['emt_score', 'senescence_score', 'sasp_score']].mean()
    summary.to_parquet(out / 'signatures_by_stage.parquet')
    print(f'  Saved signatures_by_stage.parquet')

print('Signatures complete')
PYTHON_END
fi

# =============================================================================
# Step 6: GSEA pathway enrichment
# =============================================================================
echo ""
echo "[6/14] GSEA pathway enrichment..."
if [ -f "$CANONICAL/gsea/gsea_summary.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.biology.gsea \
        --de-dir $CANONICAL/de_analysis \
        --output $CANONICAL/gsea
fi

# =============================================================================
# Step 7: decoupleR TF/pathway activity
# =============================================================================
echo ""
echo "[7/14] decoupleR TF/pathway activity..."
if [ -f "$CANONICAL/activity/pathway_activity_progeny.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/activity
mkdir -p $CANONICAL/spatial_activity
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
from pathlib import Path
import os

from stagebridge.biology import (
    compute_tf_activity,
    compute_pathway_activity,
    compute_hallmark_activity,
    rank_by_progression,
    rank_by_group,
)

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'activity'
spatial_out = Path(CANONICAL) / 'spatial_activity'

print('Running decoupleR activity analysis...')

# Single-cell enrichment
print('\n[1/4] Single-cell enrichment...')
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs} cells')

print('  TF activity (CollecTRI)...')
tf_df = compute_tf_activity(adata)
tf_df.to_parquet(out / 'tf_activity_collectri.parquet')

print('  Pathway activity (PROGENy)...')
pw_df = compute_pathway_activity(adata)
pw_df.to_parquet(out / 'pathway_activity_progeny.parquet')

print('  Hallmark gene sets...')
hm_df = compute_hallmark_activity(adata)
hm_df.to_parquet(out / 'hallmark_activity.parquet')

# Stage summaries
if 'stage' in adata.obs.columns:
    for name, df in [('tf', tf_df), ('pathway', pw_df), ('hallmark', hm_df)]:
        df_copy = df.copy()
        df_copy['stage'] = adata.obs['stage'].values
        df_copy.groupby('stage').mean().to_parquet(out / f'{name}_activity_by_stage.parquet')

# Pseudotime enrichment
print('\n[2/4] Pseudotime enrichment...')
if 'stage' in adata.obs.columns:
    import decoupler as dc

    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4}
    adata.obs['stage_num'] = adata.obs['stage'].map(stage_map)

    print('  Ranking TFs by progression...')
    tf_prog = rank_by_progression(adata, order_col='stage_num')
    tf_prog.to_parquet(out / 'tf_progression_correlation.parquet')

    print('  Ranking pathways by progression...')
    progeny = dc.op.progeny(organism='human')
    dc.mt.ulm(data=adata, net=progeny)
    pw_prog = rank_by_progression(adata, order_col='stage_num')
    pw_prog.to_parquet(out / 'pathway_progression_correlation.parquet')

    print('  Finding stage-specific TFs...')
    collectri = dc.op.collectri(organism='human')
    dc.mt.ulm(data=adata, net=collectri)
    tf_markers = rank_by_group(adata, groupby='stage')
    tf_markers.to_parquet(out / 'tf_markers_by_stage.parquet')

# Pseudobulk enrichment
print('\n[3/4] Pseudobulk enrichment...')
sample_col = 'donor_id' if 'donor_id' in adata.obs.columns else 'sample_id' if 'sample_id' in adata.obs.columns else None
if sample_col:
    import decoupler as dc

    print(f'  Aggregating by {sample_col}...')
    pdata = dc.pp.pseudobulk(adata=adata, sample_col=sample_col,
                             groups_col='stage' if 'stage' in adata.obs.columns else None, mode='sum',
                             skip_checks=True)
    dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000)
    print(f'  {pdata.n_obs} pseudobulk samples')

    pdata.layers['counts'] = pdata.X.copy()
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)

    pb_tf = compute_tf_activity(pdata)
    pb_tf.to_parquet(out / 'pseudobulk_tf_activity.parquet')

    pb_pw = compute_pathway_activity(pdata)
    pb_pw.to_parquet(out / 'pseudobulk_pathway_activity.parquet')

    pb_hm = compute_hallmark_activity(pdata)
    pb_hm.to_parquet(out / 'pseudobulk_hallmark_activity.parquet')
else:
    print('  SKIP: no sample column')

# Spatial enrichment
print('\n[4/4] Spatial enrichment...')
if Path(SPATIAL).exists():
    sdata = sc.read_h5ad(SPATIAL)
    print(f'  {sdata.n_obs} spots')

    if 'spatial' in sdata.obsm:
        import decoupler as dc
        print('  Applying spatial smoothing...')
        dc.pp.knn(sdata, key='spatial', bw=100, cutoff=0.1)
        sdata.X = sdata.obsp['spatial_connectivities'].dot(sdata.X)

    sp_tf = compute_tf_activity(sdata)
    sp_tf.to_parquet(spatial_out / 'spatial_tf_activity.parquet')

    sp_pw = compute_pathway_activity(sdata)
    sp_pw.to_parquet(spatial_out / 'spatial_pathway_activity.parquet')

    sp_hm = compute_hallmark_activity(sdata)
    sp_hm.to_parquet(spatial_out / 'spatial_hallmark_activity.parquet')

    if 'stage' in sdata.obs.columns:
        for name, df in [('tf', sp_tf), ('pathway', sp_pw), ('hallmark', sp_hm)]:
            df_copy = df.copy()
            df_copy['stage'] = sdata.obs['stage'].values
            df_copy.groupby('stage').mean().to_parquet(spatial_out / f'spatial_{name}_by_stage.parquet')
else:
    print('  SKIP: spatial data not found')

print('\ndecoupleR complete')
PYTHON_END
fi

# =============================================================================
# Step 8: Trajectory analysis (DPT, PAGA)
# =============================================================================
echo ""
echo "[8/14] Trajectory analysis..."
if [ -f "$CANONICAL/trajectories/diffusion_pseudotime.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.biology.trajectories \
        --h5ad $SNRNA \
        --output $CANONICAL/trajectories \
        --root Normal
fi

# =============================================================================
# Step 9: UMAP/PHATE embeddings + clustering
# =============================================================================
echo ""
echo "[9/14] UMAP/PHATE embeddings + clustering..."
if [ -f "$CANONICAL/embeddings/umap_embedding.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.preprocessing.embeddings \
        --h5ad $SNRNA \
        --output $CANONICAL/embeddings
fi

# =============================================================================
# Step 10: Cell-cell communication summary
# =============================================================================
echo ""
echo "[10/14] Cell-cell communication summary..."
if [ -f "$CANONICAL/communication/communication_matrix.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/communication
python << 'PYTHON_END'
import pandas as pd
from pathlib import Path
import os

CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'communication'

print('Loading LIANA results...')
liana_path = Path(CANONICAL) / 'liana_interactions.parquet'
if not liana_path.exists():
    print('  SKIP: LIANA not run')
    exit(0)

liana = pd.read_parquet(liana_path)
print(f'  {len(liana)} interactions')

# Communication matrix (sender x receiver)
print('Computing communication matrix...')
comm_matrix = liana.groupby(['source', 'target']).size().unstack(fill_value=0)
comm_matrix.to_parquet(out / 'communication_matrix.parquet')

# Top interactions by stage
if 'stage' in liana.columns:
    print('Top interactions by stage...')
    for stage in liana['stage'].unique():
        stage_df = liana[liana['stage'] == stage].nlargest(100, 'magnitude_rank' if 'magnitude_rank' in liana.columns else 'score')
        stage_df.to_parquet(out / f'top_interactions_{stage}.parquet')

# IL1B-specific interactions
print('IL1B interactions...')
ligand_col = 'ligand_complex' if 'ligand_complex' in liana.columns else 'ligand'
receptor_col = 'receptor_complex' if 'receptor_complex' in liana.columns else 'receptor'
il1b = liana[liana[ligand_col].str.contains('IL1', na=False) | liana[receptor_col].str.contains('IL1R', na=False)]
if len(il1b) > 0:
    il1b.to_parquet(out / 'il1b_interactions.parquet')
    print(f'  {len(il1b)} IL1B-related interactions')

print('Communication summary complete')
PYTHON_END
fi

# =============================================================================
# Step 11: Key gene expression
# =============================================================================
echo ""
echo "[11/14] Key gene expression..."
if [ -f "$CANONICAL/key_genes/key_genes_by_stage.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/key_genes
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SNRNA = os.environ.get('SNRNA', '/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'key_genes'

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)

KEY_GENES = [
    # IL1B pathway
    'IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN',
    # Epithelial markers
    'EPCAM', 'KRT7', 'KRT18', 'KRT19', 'CDH1',
    # AT2 / progenitor
    'SFTPC', 'SFTPA1', 'SFTPA2', 'NKX2-1', 'HOPX', 'AGER',
    # EMT
    'VIM', 'CDH2', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1',
    # Cancer genes
    'TP53', 'KRAS', 'EGFR', 'MYC', 'SOX2', 'SOX9',
    # Immune
    'CD274', 'PDCD1', 'CTLA4', 'CD8A', 'CD4', 'FOXP3',
    # CAF
    'ACTA2', 'COL1A1', 'COL1A2', 'FAP', 'PDGFRA', 'PDGFRB',
]

available = [g for g in KEY_GENES if g in adata.var_names]
print(f'  Found {len(available)}/{len(KEY_GENES)} key genes')

if available:
    X = adata[:, available].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    expr_df = pd.DataFrame(X, index=adata.obs.index, columns=available)

    if 'stage' in adata.obs.columns:
        expr_df['stage'] = adata.obs['stage'].values
        summary = expr_df.groupby('stage')[available].agg(['mean', 'std', lambda x: (x > 0).mean()])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.to_parquet(out / 'key_genes_by_stage.parquet')
        print(f'  Saved key_genes_by_stage.parquet')

print('Key genes complete')
PYTHON_END
fi

# =============================================================================
# Step 12: Visium spatial analysis
# =============================================================================
echo ""
echo "[12/14] Visium spatial analysis..."
if [ -f "$CANONICAL/visium/spot_deconvolution_summary.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/visium
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os

SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'visium'

if not Path(SPATIAL).exists():
    print('  SKIP: no spatial data')
    exit(0)

print('Loading spatial data...')
adata = sc.read_h5ad(SPATIAL)
print(f'  {adata.n_obs} spots')

# Check for deconvolution results
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
if destvi_path.exists():
    print('Loading DestVI deconvolution...')
    props = pd.read_parquet(destvi_path)

    # Summary by stage
    if 'stage' in adata.obs.columns:
        props['stage'] = adata.obs['stage'].values
        prop_cols = [c for c in props.columns if c not in ['stage', 'spot_id', 'cell_id']]
        summary = props.groupby('stage')[prop_cols].mean()
        summary.to_parquet(out / 'spot_deconvolution_summary.parquet')
        print(f'  Saved spot_deconvolution_summary.parquet')
else:
    print('  No deconvolution results found')
    pd.DataFrame().to_parquet(out / 'spot_deconvolution_summary.parquet')

print('Visium analysis complete')
PYTHON_END
fi

# =============================================================================
# Step 13: Spatial niche phenotyping
# =============================================================================
echo ""
echo "[13/14] Spatial niche phenotyping..."
if [ -f "$CANONICAL/niches/spot_niche_phenotypes.parquet" ]; then
    echo "  SKIP: exists"
else
mkdir -p $CANONICAL/niches
python << 'PYTHON_END'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.cluster import KMeans

SPATIAL = os.environ.get('SPATIAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad')
CANONICAL = os.environ.get('CANONICAL', '/data1/chaunzt1/stagebridge/processed/luad_evo/canonical')
out = Path(CANONICAL) / 'niches'

if not Path(SPATIAL).exists():
    print('  SKIP: no spatial data')
    exit(0)

print('Loading spatial data...')
adata = sc.read_h5ad(SPATIAL)
print(f'  {adata.n_obs} spots')

# Load deconvolution for niche definition
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
if destvi_path.exists():
    print('Loading deconvolution...')
    props = pd.read_parquet(destvi_path)
    prop_cols = [c for c in props.columns if c not in ['stage', 'spot_id', 'cell_id']]

    if prop_cols:
        print('Clustering niches...')
        X = props[prop_cols].values
        n_clusters = min(8, len(props) // 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        niches = kmeans.fit_predict(X)

        niche_df = pd.DataFrame({
            'spot_id': props.index if 'spot_id' not in props.columns else props['spot_id'],
            'niche': niches,
        })

        if 'stage' in adata.obs.columns:
            niche_df['stage'] = adata.obs['stage'].values

        niche_df.to_parquet(out / 'spot_niche_phenotypes.parquet')
        print(f'  Saved spot_niche_phenotypes.parquet with {n_clusters} niches')

        # Niche composition
        niche_comp = props[prop_cols].copy()
        niche_comp['niche'] = niches
        niche_summary = niche_comp.groupby('niche')[prop_cols].mean()
        niche_summary.to_parquet(out / 'niche_composition.parquet')
        print(f'  Saved niche_composition.parquet')
else:
    print('  No deconvolution for niche clustering')
    pd.DataFrame().to_parquet(out / 'spot_niche_phenotypes.parquet')

print('Niche phenotyping complete')
PYTHON_END
fi

# =============================================================================
# Step 14: QC metrics
# =============================================================================
echo ""
echo "[14/14] QC metrics..."
if [ -f "$CANONICAL/qc/snrna_qc_metrics.parquet" ]; then
    echo "  SKIP: exists"
else
    python -m stagebridge.preprocessing.qc \
        --h5ad $SNRNA \
        --output $CANONICAL/qc
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
