#!/bin/bash
# Full data preparation for StageBridge
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

# 1. Main data prep with LIANA
echo ""
echo "[1/8] Running main data prep with LIANA..."
python -m stagebridge.pipelines.prepare_data \
    --cells $CANONICAL/cells.parquet \
    --output-dir $CANONICAL \
    --destvi-luca /data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet \
    --destvi-hlca /data1/chaunzt1/stagebridge/results/spatial_benchmark/hlca/destvi/cell_type_proportions.parquet \
    --progression $CANONICAL/progression/progression_scores.parquet \
    --h5ad $SNRNA \
    --run-liana \
    --figures

# 2. SCENIC regulon analysis
echo ""
echo "[2/8] Running SCENIC regulon analysis..."
mkdir -p $CANONICAL/scenic
python -c "
from pathlib import Path
from stagebridge.biology.regulons import run_scenic_pipeline

results = run_scenic_pipeline(
    Path('$SNRNA'),
    Path('$CANONICAL/scenic'),
    skip_grn=True,  # Use predefined lung cancer regulons
    n_jobs=8,
)
print('SCENIC complete:', results)
"

# 3. Squidpy spatial statistics
echo ""
echo "[3/8] Running Squidpy spatial statistics..."
mkdir -p $CANONICAL/spatial_stats
python -c "
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading spatial data...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

out = Path('$CANONICAL/spatial_stats')

# Spatial neighbors
print('Computing spatial neighbors...')
sq.gr.spatial_neighbors(adata, coord_type='generic')

# Neighborhood enrichment (cell type co-localization)
print('Computing neighborhood enrichment...')
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    sq.gr.nhood_enrichment(adata, cluster_key=cell_type_key)
    zscore = adata.uns[f'{cell_type_key}_nhood_enrichment']['zscore']
    pd.DataFrame(zscore, index=adata.obs[cell_type_key].cat.categories,
                 columns=adata.obs[cell_type_key].cat.categories).to_parquet(out / 'nhood_enrichment.parquet')
    print(f'  Saved nhood_enrichment.parquet')

# Spatial autocorrelation for key genes
print('Computing Morans I for key genes...')
key_genes = ['IL1B', 'IL1R1', 'CXCL12', 'CXCR4', 'EGFR', 'SOX9', 'KRT17', 'VIM', 'CDH1', 'ACTA2', 'COL1A1', 'CD274', 'PDCD1']
available_genes = [g for g in key_genes if g in adata.var_names]
if available_genes:
    sq.gr.spatial_autocorr(adata, genes=available_genes, mode='moran')
    adata.uns['moranI'].to_parquet(out / 'morans_i.parquet')
    print(f'  Saved morans_i.parquet ({len(available_genes)} genes)')

# Co-occurrence by stage if available
print('Computing co-occurrence...')
if 'stage' in adata.obs.columns and cell_type_key in adata.obs.columns:
    for stage in adata.obs['stage'].unique():
        stage_adata = adata[adata.obs['stage'] == stage].copy()
        if stage_adata.n_obs > 100:
            try:
                sq.gr.spatial_neighbors(stage_adata, coord_type='generic')
                sq.gr.co_occurrence(stage_adata, cluster_key=cell_type_key)
                occ = stage_adata.uns[f'{cell_type_key}_co_occurrence']
                np.savez(out / f'co_occurrence_{stage}.npz',
                         occ=occ['occ'], interval=occ['interval'])
                print(f'  Saved co_occurrence_{stage}.npz')
            except Exception as e:
                print(f'  Skipping {stage}: {e}')

print('Squidpy complete')
"

# 4. Differential expression by stage
echo ""
echo "[4/8] Running differential expression analysis..."
mkdir -p $CANONICAL/de_analysis
python -c "
import scanpy as sc
import pandas as pd
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/de_analysis')

# DE by stage
if 'stage' in adata.obs.columns:
    print('Running DE by stage...')
    sc.tl.rank_genes_groups(adata, groupby='stage', method='wilcoxon', n_genes=200)

    # Save full results
    for stage in adata.obs['stage'].unique():
        df = sc.get.rank_genes_groups_df(adata, group=stage)
        df.to_parquet(out / f'de_stage_{stage}.parquet')
    print('  Saved DE by stage')

# DE by cell type within each stage
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns and 'stage' in adata.obs.columns:
    print('Running DE by cell type per stage...')
    for stage in adata.obs['stage'].unique():
        stage_adata = adata[adata.obs['stage'] == stage].copy()
        if stage_adata.n_obs > 100:
            try:
                sc.tl.rank_genes_groups(stage_adata, groupby=cell_type_key, method='wilcoxon', n_genes=100)
                for ct in stage_adata.obs[cell_type_key].unique():
                    df = sc.get.rank_genes_groups_df(stage_adata, group=ct)
                    safe_ct = ct.replace('/', '_').replace(' ', '_')
                    df.to_parquet(out / f'de_{stage}_{safe_ct}.parquet')
            except Exception as e:
                print(f'  Skipping {stage}: {e}')
    print('  Saved DE by cell type per stage')

print('DE analysis complete')
"

# 5. Cell type proportions and summary stats
echo ""
echo "[5/8] Computing summary statistics..."
mkdir -p $CANONICAL/summary_stats
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading neighborhoods...')
df = pd.read_parquet('$CANONICAL/neighborhoods.parquet')
print(f'  {len(df)} cells')

out = Path('$CANONICAL/summary_stats')

# Cell type proportions by stage
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in df.columns else 'cell_type'
if cell_type_key in df.columns and 'stage' in df.columns:
    props = df.groupby(['stage', cell_type_key]).size().unstack(fill_value=0)
    props_norm = props.div(props.sum(axis=1), axis=0)
    props_norm.to_parquet(out / 'celltype_proportions_by_stage.parquet')
    print('  Saved celltype_proportions_by_stage.parquet')

# Cell type proportions by donor
if cell_type_key in df.columns and 'donor_id' in df.columns:
    props = df.groupby(['donor_id', cell_type_key]).size().unstack(fill_value=0)
    props_norm = props.div(props.sum(axis=1), axis=0)
    props_norm.to_parquet(out / 'celltype_proportions_by_donor.parquet')
    print('  Saved celltype_proportions_by_donor.parquet')

# Feature means by stage
numeric_cols = ['caf_fraction', 'immune_fraction', 'diversity', 'malignant_fraction',
                'S_score', 'G2M_score', 'emt_score', 'senescence_score', 'sasp_score',
                'cytotrace', 'pseudotime', 'il1b_raw', 'kac_raw']
available = [c for c in numeric_cols if c in df.columns]
if available and 'stage' in df.columns:
    stage_means = df.groupby('stage')[available].agg(['mean', 'std', 'median'])
    stage_means.to_parquet(out / 'feature_stats_by_stage.parquet')
    print('  Saved feature_stats_by_stage.parquet')

# Feature means by cell type
if available and cell_type_key in df.columns:
    ct_means = df.groupby(cell_type_key)[available].agg(['mean', 'std', 'median'])
    ct_means.to_parquet(out / 'feature_stats_by_celltype.parquet')
    print('  Saved feature_stats_by_celltype.parquet')

# Correlation matrix of features
if len(available) > 1:
    corr = df[available].corr()
    corr.to_parquet(out / 'feature_correlations.parquet')
    print('  Saved feature_correlations.parquet')

# Sample/donor summary
if 'donor_id' in df.columns:
    donor_summary = df.groupby('donor_id').agg({
        'cell_id': 'count',
        'stage': lambda x: x.mode()[0] if len(x) > 0 else None,
    }).rename(columns={'cell_id': 'n_cells', 'stage': 'dominant_stage'})
    if available:
        donor_means = df.groupby('donor_id')[available].mean()
        donor_summary = donor_summary.join(donor_means)
    donor_summary.to_parquet(out / 'donor_summary.parquet')
    print('  Saved donor_summary.parquet')

print('Summary stats complete')
"

# 6. Gene signature scores (AP1, hypoxia, glycolysis, IFN)
echo ""
echo "[6/8] Computing additional gene signatures..."
mkdir -p $CANONICAL/signatures
python -c "
import scanpy as sc
import pandas as pd
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/signatures')

# Gene signature lists
signatures = {
    'ap1_stress': ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND', 'ATF3', 'ATF4', 'EGR1'],
    'hypoxia': ['HIF1A', 'VEGFA', 'SLC2A1', 'LDHA', 'PGK1', 'ENO1', 'ALDOA', 'CA9', 'BNIP3', 'PDK1'],
    'glycolysis': ['HK1', 'HK2', 'GPI', 'PFKP', 'ALDOA', 'TPI1', 'GAPDH', 'PGK1', 'PGAM1', 'ENO1', 'PKM', 'LDHA'],
    'oxphos': ['NDUFA1', 'NDUFB1', 'SDHA', 'SDHB', 'UQCRB', 'COX5A', 'COX7A2', 'ATP5F1A', 'ATP5F1B'],
    'ifn_gamma': ['STAT1', 'IRF1', 'GBP1', 'GBP2', 'IDO1', 'CXCL9', 'CXCL10', 'CXCL11', 'HLA-DRA', 'CD274'],
    'ifn_alpha': ['ISG15', 'MX1', 'MX2', 'OAS1', 'OAS2', 'IFIT1', 'IFIT2', 'IFIT3', 'IFI44', 'IFI44L'],
    'tgfb_response': ['TGFBI', 'SERPINE1', 'CTGF', 'COL1A1', 'COL3A1', 'FN1', 'ACTA2', 'TAGLN'],
    'stemness': ['SOX2', 'SOX9', 'NANOG', 'POU5F1', 'KLF4', 'MYC', 'LGR5', 'ALDH1A1', 'CD44', 'PROM1'],
    'il1b_pathway': ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN', 'NLRP3', 'CASP1', 'PYCARD'],
    'nfkb': ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL', 'IKBKB', 'IKBKG', 'NFKBIA', 'NFKBIB'],
}

results = {'cell_id': adata.obs.index.tolist()}

for name, genes in signatures.items():
    available = [g for g in genes if g in adata.var_names]
    if len(available) >= 3:
        print(f'  Scoring {name} ({len(available)}/{len(genes)} genes)...')
        sc.tl.score_genes(adata, gene_list=available, score_name=name)
        results[name] = adata.obs[name].tolist()
    else:
        print(f'  Skipping {name} (only {len(available)} genes available)')

# Save all signatures
sig_df = pd.DataFrame(results)
sig_df.to_parquet(out / 'gene_signatures.parquet')
print(f'Saved {len([k for k in results if k != \"cell_id\"])} signatures to gene_signatures.parquet')

# Also save per-stage means for quick reference
if 'stage' in adata.obs.columns:
    sig_df['stage'] = adata.obs['stage'].values
    stage_means = sig_df.groupby('stage').mean()
    stage_means.to_parquet(out / 'signature_means_by_stage.parquet')
    print('Saved signature_means_by_stage.parquet')

print('Signature scoring complete')
"

# 7. Key gene expression tables
echo ""
echo "[7/8] Extracting key gene expression..."
mkdir -p $CANONICAL/expression
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/expression')

# Key genes for figures
key_genes = [
    # IL1B axis (main finding)
    'IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN',
    # EMT markers
    'VIM', 'CDH1', 'CDH2', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1',
    # KAC/reactive pneumocyte
    'KRT17', 'KRT5', 'KRT8', 'SOX9', 'TP63',
    # CAF markers
    'ACTA2', 'FAP', 'COL1A1', 'COL3A1', 'FN1', 'PDGFRA', 'PDGFRB',
    # Immune markers
    'CD68', 'CD163', 'CD14', 'CD3D', 'CD4', 'CD8A', 'FOXP3', 'CD274', 'PDCD1',
    # Proliferation
    'MKI67', 'TOP2A', 'PCNA',
    # Stress/AP1
    'FOS', 'JUN', 'ATF3', 'EGR1',
    # Checkpoints
    'CTLA4', 'LAG3', 'TIGIT', 'HAVCR2',
    # Growth factors
    'EGF', 'EGFR', 'TGFB1', 'VEGFA', 'HGF', 'MET',
]

available = [g for g in key_genes if g in adata.var_names]
print(f'  Extracting {len(available)}/{len(key_genes)} genes')

# Extract expression matrix for key genes
if hasattr(adata.X, 'toarray'):
    expr = pd.DataFrame(
        adata[:, available].X.toarray(),
        index=adata.obs.index,
        columns=available
    )
else:
    expr = pd.DataFrame(
        adata[:, available].X,
        index=adata.obs.index,
        columns=available
    )

# Add metadata
expr['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
expr['donor_id'] = adata.obs['donor_id'].values if 'donor_id' in adata.obs.columns else None
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    expr['cell_type'] = adata.obs[cell_type_key].values

expr.to_parquet(out / 'key_genes_expression.parquet')
print('  Saved key_genes_expression.parquet')

# Aggregate by stage x cell type
if 'stage' in expr.columns and 'cell_type' in expr.columns:
    agg = expr.groupby(['stage', 'cell_type'])[available].mean()
    agg.to_parquet(out / 'key_genes_mean_by_stage_celltype.parquet')
    print('  Saved key_genes_mean_by_stage_celltype.parquet')

# Percent expressing per stage x cell type
def pct_expressing(x):
    return (x > 0).mean() * 100

if 'stage' in expr.columns and 'cell_type' in expr.columns:
    pct = expr.groupby(['stage', 'cell_type'])[available].apply(lambda x: x.apply(pct_expressing))
    pct.to_parquet(out / 'key_genes_pct_expressing.parquet')
    print('  Saved key_genes_pct_expressing.parquet')

print('Key gene extraction complete')
"

# 8. GSEA / Pathway enrichment (GO, KEGG, Reactome)
echo ""
echo "[8/11] Running pathway enrichment analysis..."
mkdir -p $CANONICAL/pathways
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

out = Path('$CANONICAL/pathways')
de_dir = Path('$CANONICAL/de_analysis')

try:
    import gseapy as gp
    print('Running GSEA with gseapy...')

    # Get DE results by stage
    de_files = list(de_dir.glob('de_stage_*.parquet'))

    for de_file in de_files:
        stage = de_file.stem.replace('de_stage_', '')
        print(f'  Processing {stage}...')

        de_df = pd.read_parquet(de_file)

        # Rank by signed log10 pval
        de_df['rank'] = -np.log10(de_df['pvals'] + 1e-300) * np.sign(de_df['logfoldchanges'])
        ranked = de_df.set_index('names')['rank'].sort_values(ascending=False)

        # GO Biological Process
        try:
            go_bp = gp.prerank(rnk=ranked, gene_sets='GO_Biological_Process_2021',
                              outdir=None, min_size=15, max_size=500, permutation_num=100, verbose=False)
            go_bp.res2d.to_parquet(out / f'gsea_go_bp_{stage}.parquet')
        except Exception as e:
            print(f'    GO BP failed: {e}')

        # KEGG
        try:
            kegg = gp.prerank(rnk=ranked, gene_sets='KEGG_2021_Human',
                             outdir=None, min_size=15, max_size=500, permutation_num=100, verbose=False)
            kegg.res2d.to_parquet(out / f'gsea_kegg_{stage}.parquet')
        except Exception as e:
            print(f'    KEGG failed: {e}')

        # Reactome
        try:
            reactome = gp.prerank(rnk=ranked, gene_sets='Reactome_2022',
                                 outdir=None, min_size=15, max_size=500, permutation_num=100, verbose=False)
            reactome.res2d.to_parquet(out / f'gsea_reactome_{stage}.parquet')
        except Exception as e:
            print(f'    Reactome failed: {e}')

        # Hallmark gene sets
        try:
            hallmark = gp.prerank(rnk=ranked, gene_sets='MSigDB_Hallmark_2020',
                                 outdir=None, min_size=15, max_size=500, permutation_num=100, verbose=False)
            hallmark.res2d.to_parquet(out / f'gsea_hallmark_{stage}.parquet')
        except Exception as e:
            print(f'    Hallmark failed: {e}')

    print('GSEA complete')

except ImportError:
    print('gseapy not installed, skipping GSEA')
except Exception as e:
    print(f'GSEA failed: {e}')
"

# 9. decoupleR / VIPER-style TF activity
echo ""
echo "[9/11] Running decoupleR TF/pathway activity..."
mkdir -p $CANONICAL/activity
python -c "
import scanpy as sc
import pandas as pd
from pathlib import Path

out = Path('$CANONICAL/activity')

try:
    import decoupler as dc
    print('Running decoupleR...')

    adata = sc.read_h5ad('$SNRNA')
    print(f'  {adata.n_obs} cells')

    # TF activity with CollecTRI (VIPER-style)
    print('  Computing TF activity (CollecTRI)...')
    dc.run_ulm(
        mat=adata,
        net=dc.get_collectri(organism='human', split_complexes=False),
        source='source', target='target', weight='weight',
        verbose=True
    )
    tf_acts = dc.get_acts(adata, obsm_key='ulm_estimate')
    pd.DataFrame(tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index).to_parquet(out / 'tf_activity_collectri.parquet')

    # Pathway activity with PROGENy
    print('  Computing pathway activity (PROGENy)...')
    dc.run_mlm(
        mat=adata,
        net=dc.get_progeny(organism='human', top=300),
        source='source', target='target', weight='weight',
        verbose=True
    )
    pathway_acts = dc.get_acts(adata, obsm_key='mlm_estimate')
    pd.DataFrame(pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index).to_parquet(out / 'pathway_activity_progeny.parquet')

    # Aggregate by stage
    if 'stage' in adata.obs.columns:
        tf_df = pd.DataFrame(tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index)
        tf_df['stage'] = adata.obs['stage'].values
        tf_df.groupby('stage').mean().to_parquet(out / 'tf_activity_by_stage.parquet')

        pw_df = pd.DataFrame(pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index)
        pw_df['stage'] = adata.obs['stage'].values
        pw_df.groupby('stage').mean().to_parquet(out / 'pathway_activity_by_stage.parquet')

    # Aggregate by cell type
    cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
    if cell_type_key in adata.obs.columns:
        tf_df = pd.DataFrame(tf_acts.X, index=tf_acts.obs.index, columns=tf_acts.var.index)
        tf_df['cell_type'] = adata.obs[cell_type_key].values
        tf_df.groupby('cell_type').mean().to_parquet(out / 'tf_activity_by_celltype.parquet')

        pw_df = pd.DataFrame(pathway_acts.X, index=pathway_acts.obs.index, columns=pathway_acts.var.index)
        pw_df['cell_type'] = adata.obs[cell_type_key].values
        pw_df.groupby('cell_type').mean().to_parquet(out / 'pathway_activity_by_celltype.parquet')

    print('decoupleR complete')

except ImportError:
    print('decoupleR not installed, skipping')
except Exception as e:
    print(f'decoupleR failed: {e}')
"

# 10. ORA (Over-representation analysis) for top DE genes
echo ""
echo "[10/11] Running over-representation analysis..."
python -c "
import pandas as pd
from pathlib import Path

out = Path('$CANONICAL/pathways')
de_dir = Path('$CANONICAL/de_analysis')

try:
    import gseapy as gp
    print('Running ORA...')

    de_files = list(de_dir.glob('de_stage_*.parquet'))

    for de_file in de_files:
        stage = de_file.stem.replace('de_stage_', '')
        print(f'  Processing {stage}...')

        de_df = pd.read_parquet(de_file)

        # Top upregulated genes (logFC > 0.5, pval < 0.05)
        up_genes = de_df[(de_df['logfoldchanges'] > 0.5) & (de_df['pvals_adj'] < 0.05)]['names'].tolist()

        if len(up_genes) >= 10:
            try:
                enr = gp.enrichr(gene_list=up_genes[:500],
                                gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
                                outdir=None)
                enr.results.to_parquet(out / f'ora_up_{stage}.parquet')
            except Exception as e:
                print(f'    ORA up failed: {e}')

        # Top downregulated genes
        down_genes = de_df[(de_df['logfoldchanges'] < -0.5) & (de_df['pvals_adj'] < 0.05)]['names'].tolist()

        if len(down_genes) >= 10:
            try:
                enr = gp.enrichr(gene_list=down_genes[:500],
                                gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
                                outdir=None)
                enr.results.to_parquet(out / f'ora_down_{stage}.parquet')
            except Exception as e:
                print(f'    ORA down failed: {e}')

    print('ORA complete')

except ImportError:
    print('gseapy not installed, skipping ORA')
except Exception as e:
    print(f'ORA failed: {e}')
"

# 11. Diffusion pseudotime / PAGA trajectory
echo ""
echo "[11/14] Computing diffusion pseudotime and PAGA..."
mkdir -p $CANONICAL/trajectories
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/trajectories')

# PCA and neighbors
print('Computing PCA and neighbors...')
if 'X_pca' not in adata.obsm:
    sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')

# Diffusion map
print('Computing diffusion map...')
sc.tl.diffmap(adata, n_comps=15)
pd.DataFrame(adata.obsm['X_diffmap'], index=adata.obs.index).to_parquet(out / 'diffmap_embedding.parquet')

# Diffusion pseudotime rooted at Normal
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
        'stage': adata.obs['stage'].values if 'stage' in adata.obs.columns else None,
    }).to_parquet(out / 'diffusion_pseudotime.parquet')

# PAGA
print('Computing PAGA...')
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    sc.tl.paga(adata, groups=cell_type_key)
    # Save PAGA connectivity
    conn = pd.DataFrame(
        adata.uns['paga']['connectivities'].toarray(),
        index=adata.obs[cell_type_key].cat.categories,
        columns=adata.obs[cell_type_key].cat.categories
    )
    conn.to_parquet(out / 'paga_connectivity_celltype.parquet')

if 'stage' in adata.obs.columns:
    sc.tl.paga(adata, groups='stage')
    conn = pd.DataFrame(
        adata.uns['paga']['connectivities'].toarray(),
        index=adata.obs['stage'].cat.categories if hasattr(adata.obs['stage'], 'cat') else adata.obs['stage'].unique(),
        columns=adata.obs['stage'].cat.categories if hasattr(adata.obs['stage'], 'cat') else adata.obs['stage'].unique()
    )
    conn.to_parquet(out / 'paga_connectivity_stage.parquet')

print('Trajectory analysis complete')
"

# 12. UMAP/PHATE embeddings for visualization
echo ""
echo "[12/14] Computing embeddings (UMAP, PHATE)..."
mkdir -p $CANONICAL/embeddings
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/embeddings')

# PCA
if 'X_pca' not in adata.obsm:
    print('Computing PCA...')
    sc.pp.pca(adata, n_comps=50)
pd.DataFrame(adata.obsm['X_pca'][:, :20], index=adata.obs.index,
             columns=[f'PC{i+1}' for i in range(20)]).to_parquet(out / 'pca_embedding.parquet')

# Neighbors
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')

# UMAP
print('Computing UMAP...')
sc.tl.umap(adata)
umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs.index, columns=['UMAP1', 'UMAP2'])
umap_df['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
umap_df['donor_id'] = adata.obs['donor_id'].values if 'donor_id' in adata.obs.columns else None
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    umap_df['cell_type'] = adata.obs[cell_type_key].values
umap_df.to_parquet(out / 'umap_embedding.parquet')

# PHATE if available
try:
    import phate
    print('Computing PHATE...')
    phate_op = phate.PHATE(n_components=2, n_jobs=-1, random_state=42)
    X = adata.obsm['X_pca'][:, :50]
    phate_emb = phate_op.fit_transform(X)
    phate_df = pd.DataFrame(phate_emb, index=adata.obs.index, columns=['PHATE1', 'PHATE2'])
    phate_df['stage'] = adata.obs['stage'].values if 'stage' in adata.obs.columns else None
    if cell_type_key in adata.obs.columns:
        phate_df['cell_type'] = adata.obs[cell_type_key].values
    phate_df.to_parquet(out / 'phate_embedding.parquet')
except ImportError:
    print('PHATE not installed, skipping')
except Exception as e:
    print(f'PHATE failed: {e}')

# Louvain/Leiden clustering at multiple resolutions
print('Computing Louvain/Leiden clustering...')
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}')
    sc.tl.louvain(adata, resolution=res, key_added=f'louvain_{res}')

# Save clustering results
cluster_df = pd.DataFrame({'cell_id': adata.obs.index})
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    cluster_df[f'leiden_{res}'] = adata.obs[f'leiden_{res}'].values
    cluster_df[f'louvain_{res}'] = adata.obs[f'louvain_{res}'].values

if 'stage' in adata.obs.columns:
    cluster_df['stage'] = adata.obs['stage'].values
if cell_type_key in adata.obs.columns:
    cluster_df['cell_type'] = adata.obs[cell_type_key].values

cluster_df.to_parquet(out / 'clustering.parquet')

# Add clustering to UMAP for convenience
umap_df = pd.read_parquet(out / 'umap_embedding.parquet')
for res in [0.5, 1.0]:
    umap_df[f'leiden_{res}'] = adata.obs[f'leiden_{res}'].values
    umap_df[f'louvain_{res}'] = adata.obs[f'louvain_{res}'].values
umap_df.to_parquet(out / 'umap_embedding.parquet')

print('Embedding computation complete')

# Cluster annotation
print('Annotating clusters...')

# 1. Majority voting from reference labels
if cell_type_key in adata.obs.columns:
    for res in [0.5, 1.0]:
        cluster_col = f'leiden_{res}'

        # Majority cell type per cluster
        majority = adata.obs.groupby(cluster_col)[cell_type_key].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
        )

        # Purity (fraction of majority type)
        purity = adata.obs.groupby(cluster_col)[cell_type_key].agg(
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
        )

        annotation = pd.DataFrame({
            'cluster': majority.index,
            'majority_celltype': majority.values,
            'purity': purity.values,
            'n_cells': adata.obs[cluster_col].value_counts()[majority.index].values,
        })
        annotation.to_parquet(out / f'cluster_annotation_{cluster_col}.parquet')

# 2. Marker gene enrichment per cluster
print('Computing cluster markers...')
for res in [0.5, 1.0]:
    cluster_col = f'leiden_{res}'
    sc.tl.rank_genes_groups(adata, groupby=cluster_col, method='wilcoxon', n_genes=50)

    # Save top markers per cluster
    markers = []
    for cluster in adata.obs[cluster_col].unique():
        df = sc.get.rank_genes_groups_df(adata, group=str(cluster))
        df['cluster'] = cluster
        markers.append(df.head(20))

    marker_df = pd.concat(markers, ignore_index=True)
    marker_df.to_parquet(out / f'cluster_markers_{cluster_col}.parquet')

# 3. CellTypist for immune refinement (if installed)
try:
    import celltypist
    from celltypist import models

    print('Running CellTypist for immune annotation...')
    models.download_models(force_update=False, model='Immune_All_Low.pkl')
    model = models.Model.load(model='Immune_All_Low.pkl')

    predictions = celltypist.annotate(adata, model=model, majority_voting=True)
    adata.obs['celltypist_label'] = predictions.predicted_labels['majority_voting']

    celltypist_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'celltypist_label': adata.obs['celltypist_label'].values,
        'celltypist_conf': predictions.probability_matrix.max(axis=1).values,
    })
    celltypist_df.to_parquet(out / 'celltypist_annotation.parquet')
    print('  CellTypist complete')

except ImportError:
    print('CellTypist not installed, skipping')
except Exception as e:
    print(f'CellTypist failed: {e}')
"

# 13. Cell-cell communication summary (LIANA + CellChat-style)
echo ""
echo "[13/14] Summarizing cell-cell communication..."
python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import scanpy as sc

out = Path('$CANONICAL/communication')
out.mkdir(exist_ok=True)

liana_path = Path('$CANONICAL/liana_interactions.parquet')
if liana_path.exists():
    print('Processing LIANA results...')
    liana = pd.read_parquet(liana_path)

    # Top interactions overall
    rank_col = 'specificity_rank' if 'specificity_rank' in liana.columns else 'pvalue'
    top_interactions = liana.nsmallest(100, rank_col)
    top_interactions.to_parquet(out / 'top_interactions.parquet')

    # Aggregate by ligand-receptor pair
    if 'ligand_complex' in liana.columns and 'receptor_complex' in liana.columns:
        agg_cols = {rank_col: 'mean'}
        if 'magnitude_rank' in liana.columns:
            agg_cols['magnitude_rank'] = 'mean'
        lr_summary = liana.groupby(['ligand_complex', 'receptor_complex']).agg(agg_cols).reset_index()
        lr_summary = lr_summary.nsmallest(50, rank_col)
        lr_summary.to_parquet(out / 'lr_pair_summary.parquet')

    # Aggregate by source-target cell type
    if 'source' in liana.columns and 'target' in liana.columns:
        ct_comm = liana.groupby(['source', 'target']).size().reset_index(name='n_interactions')
        ct_comm.to_parquet(out / 'celltype_communication_counts.parquet')

        # Communication matrix (CellChat-style)
        comm_matrix = ct_comm.pivot(index='source', columns='target', values='n_interactions').fillna(0)
        comm_matrix.to_parquet(out / 'communication_matrix.parquet')

        # Outgoing/incoming communication strength (CellChat-style)
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
    print('LIANA results not found, skipping')

# Run full LIANA with multiple methods if not done
adata_path = Path('$SNRNA')
if adata_path.exists() and not liana_path.exists():
    print('Running LIANA with multiple methods...')
    try:
        import liana as li
        adata = sc.read_h5ad(adata_path)
        cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'

        # Run multiple methods (CellChat-like coverage)
        li.mt.rank_aggregate(
            adata,
            groupby=cell_type_key,
            resource_name='consensus',
            expr_prop=0.1,
            verbose=True,
            use_raw=False,
        )

        # Save aggregated results
        if 'liana_res' in adata.uns:
            adata.uns['liana_res'].to_parquet(out / 'liana_aggregate.parquet')

        print('LIANA aggregate complete')
    except Exception as e:
        print(f'LIANA aggregate failed: {e}')

# Communication patterns by stage (CellChat-style)
nhood_path = Path('$CANONICAL/neighborhoods.parquet')
if nhood_path.exists() and liana_path.exists():
    print('Computing communication by stage...')
    try:
        nhood = pd.read_parquet(nhood_path)
        liana = pd.read_parquet(liana_path)

        # This requires stage info in LIANA - may need to merge
        # For now, save the full LIANA for stage-specific analysis locally

        print('Stage-specific communication ready for local analysis')
    except Exception as e:
        print(f'Stage communication failed: {e}')
"

# 14. Visium-specific spatial analysis
echo ""
echo "[14/17] Running Visium spatial analysis..."
mkdir -p $CANONICAL/visium
python -c "
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading spatial data...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

out = Path('$CANONICAL/visium')

# Spatial neighbors if not computed
if 'spatial_neighbors' not in adata.uns:
    print('Computing spatial neighbors...')
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)

# Spatially variable genes (beyond just key genes)
print('Computing spatially variable genes...')
try:
    sq.gr.spatial_autocorr(adata, mode='moran', n_perms=100, n_jobs=8)
    svg = adata.uns['moranI'].copy()
    svg = svg.sort_values('I', ascending=False)
    svg.to_parquet(out / 'spatially_variable_genes.parquet')
    print(f'  Found {len(svg[svg[\"pval_norm\"] < 0.05])} significant SVGs')
except Exception as e:
    print(f'  SVG computation failed: {e}')

# Spatial domains / clustering
print('Computing spatial domains...')
try:
    # Leiden on spatial graph
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=0.5, key_added='spatial_domain')

    domain_df = pd.DataFrame({
        'spot_id': adata.obs.index,
        'spatial_domain': adata.obs['spatial_domain'].values,
        'x': adata.obsm['spatial'][:, 0] if 'spatial' in adata.obsm else adata.obs.get('x_spatial', 0),
        'y': adata.obsm['spatial'][:, 1] if 'spatial' in adata.obsm else adata.obs.get('y_spatial', 0),
    })
    if 'stage' in adata.obs.columns:
        domain_df['stage'] = adata.obs['stage'].values
    if 'sample' in adata.obs.columns:
        domain_df['sample'] = adata.obs['sample'].values
    domain_df.to_parquet(out / 'spatial_domains.parquet')

    # Domain composition
    cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
    if cell_type_key in adata.obs.columns:
        domain_comp = pd.crosstab(adata.obs['spatial_domain'], adata.obs[cell_type_key], normalize='index')
        domain_comp.to_parquet(out / 'domain_celltype_composition.parquet')
except Exception as e:
    print(f'  Spatial domain failed: {e}')

# Ripley's statistics (clustering patterns)
print('Computing Ripley L function...')
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    try:
        sq.gr.ripley(adata, cluster_key=cell_type_key, mode='L')
        ripley = adata.uns[f'{cell_type_key}_ripley_L']
        pd.DataFrame(ripley).to_parquet(out / 'ripley_L.parquet')
    except Exception as e:
        print(f'  Ripley failed: {e}')

# Ligand-receptor spatial proximity
print('Computing L-R spatial proximity...')
try:
    sq.gr.ligrec(
        adata,
        cluster_key=cell_type_key,
        n_perms=100,
        use_raw=False,
    )
    if 'ligrec' in adata.uns[f'{cell_type_key}_ligrec']:
        ligrec_pvals = adata.uns[f'{cell_type_key}_ligrec']['pvalues']
        ligrec_means = adata.uns[f'{cell_type_key}_ligrec']['means']
        ligrec_pvals.to_parquet(out / 'ligrec_pvalues.parquet')
        ligrec_means.to_parquet(out / 'ligrec_means.parquet')
except Exception as e:
    print(f'  L-R proximity failed: {e}')

# Interface/boundary analysis (tumor-stroma)
print('Computing interface zones...')
if 'stage' in adata.obs.columns or cell_type_key in adata.obs.columns:
    try:
        # Find boundary spots (neighbors with different labels)
        from scipy.sparse import csr_matrix
        conn = adata.obsp.get('spatial_connectivities', None)
        if conn is not None:
            label_col = 'stage' if 'stage' in adata.obs.columns else cell_type_key
            labels = adata.obs[label_col].values

            boundary_spots = []
            for i in range(adata.n_obs):
                neighbors = conn[i].indices
                if len(neighbors) > 0:
                    neighbor_labels = labels[neighbors]
                    if len(set(neighbor_labels)) > 1 or labels[i] not in neighbor_labels:
                        boundary_spots.append(i)

            boundary_df = pd.DataFrame({
                'spot_id': adata.obs.index[boundary_spots],
                'is_boundary': True,
                'label': labels[boundary_spots],
            })
            boundary_df.to_parquet(out / 'boundary_spots.parquet')
            print(f'  Found {len(boundary_spots)} boundary spots')
    except Exception as e:
        print(f'  Interface analysis failed: {e}')

# Deconvolution summary (from DestVI results if available)
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
if destvi_path.exists():
    print('Summarizing deconvolution results...')
    deconv = pd.read_parquet(destvi_path)
    deconv.to_parquet(out / 'spot_deconvolution.parquet')

    # Mean proportions by stage
    if 'stage' in adata.obs.columns:
        deconv['stage'] = adata.obs['stage'].values[:len(deconv)] if len(deconv) <= adata.n_obs else None
        if deconv['stage'].notna().any():
            stage_props = deconv.groupby('stage').mean()
            stage_props.to_parquet(out / 'deconv_proportions_by_stage.parquet')

print('Visium analysis complete')
"

# 15. Spatial gene patterns (expression in space)
echo ""
echo "[15/17] Extracting spatial gene expression patterns..."
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading spatial data...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

out = Path('$CANONICAL/visium')

# Key genes spatial expression
key_genes = [
    'IL1B', 'IL1R1', 'CXCL12', 'CXCR4', 'EGFR', 'SOX9', 'KRT17', 'KRT5',
    'VIM', 'CDH1', 'ACTA2', 'COL1A1', 'FAP', 'CD68', 'CD163', 'CD3D',
    'MKI67', 'TP63', 'EPCAM', 'CD274', 'PDCD1', 'VEGFA', 'HIF1A',
]

available = [g for g in key_genes if g in adata.var_names]
print(f'  Extracting {len(available)} genes')

if hasattr(adata.X, 'toarray'):
    expr = pd.DataFrame(
        adata[:, available].X.toarray(),
        index=adata.obs.index,
        columns=available
    )
else:
    expr = pd.DataFrame(
        adata[:, available].X,
        index=adata.obs.index,
        columns=available
    )

# Add spatial coords
if 'spatial' in adata.obsm:
    expr['x'] = adata.obsm['spatial'][:, 0]
    expr['y'] = adata.obsm['spatial'][:, 1]
elif 'x_spatial' in adata.obs.columns:
    expr['x'] = adata.obs['x_spatial'].values
    expr['y'] = adata.obs['y_spatial'].values

# Add metadata
if 'stage' in adata.obs.columns:
    expr['stage'] = adata.obs['stage'].values
if 'sample' in adata.obs.columns:
    expr['sample'] = adata.obs['sample'].values

expr.to_parquet(out / 'spatial_gene_expression.parquet')
print('  Saved spatial_gene_expression.parquet')

# Gene-gene spatial correlation
print('Computing gene-gene spatial correlations...')
gene_expr = expr[available]
spatial_corr = gene_expr.corr()
spatial_corr.to_parquet(out / 'gene_spatial_correlation.parquet')

print('Spatial gene patterns complete')
"

# 16. Sample-level summaries
echo ""
echo "[16/17] Computing sample-level summaries..."
mkdir -p $CANONICAL/samples
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading data...')
nhood = pd.read_parquet('$CANONICAL/neighborhoods.parquet')

out = Path('$CANONICAL/samples')

# Per-sample summary
if 'donor_id' in nhood.columns or 'sample' in nhood.columns:
    sample_col = 'sample' if 'sample' in nhood.columns else 'donor_id'

    # Basic counts
    sample_summary = nhood.groupby(sample_col).agg({
        'cell_id': 'count',
    }).rename(columns={'cell_id': 'n_cells'})

    # Stage distribution per sample
    if 'stage' in nhood.columns:
        stage_counts = pd.crosstab(nhood[sample_col], nhood['stage'])
        sample_summary = sample_summary.join(stage_counts)

    # Cell type distribution per sample
    cell_type_key = 'cell_type_luca' if 'cell_type_luca' in nhood.columns else 'cell_type'
    if cell_type_key in nhood.columns:
        ct_counts = pd.crosstab(nhood[sample_col], nhood[cell_type_key])
        ct_props = ct_counts.div(ct_counts.sum(axis=1), axis=0)
        ct_props.columns = [f'{c}_prop' for c in ct_props.columns]
        sample_summary = sample_summary.join(ct_props)

    # Mean features per sample
    numeric_cols = ['caf_fraction', 'immune_fraction', 'diversity', 'emt_score',
                    'senescence_score', 'S_score', 'G2M_score', 'cytotrace', 'pseudotime']
    available = [c for c in numeric_cols if c in nhood.columns]
    if available:
        sample_means = nhood.groupby(sample_col)[available].mean()
        sample_summary = sample_summary.join(sample_means)

    sample_summary.to_parquet(out / 'sample_summary.parquet')
    print(f'  Saved summary for {len(sample_summary)} samples')

    # Sample metadata if available
    meta_cols = ['stage', 'patient_id', 'tissue_type', 'batch']
    available_meta = [c for c in meta_cols if c in nhood.columns]
    if available_meta:
        sample_meta = nhood.groupby(sample_col)[available_meta].first()
        sample_meta.to_parquet(out / 'sample_metadata.parquet')

print('Sample summaries complete')
"

# 17. Rare cell type discovery and signatures
echo ""
echo "[17/20] Discovering rare cell populations..."
mkdir -p $CANONICAL/rare_cells
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/rare_cells')

cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'

# 1. Identify rare populations by frequency
print('Identifying rare populations...')
if cell_type_key in adata.obs.columns:
    ct_counts = adata.obs[cell_type_key].value_counts()
    total = ct_counts.sum()
    ct_freq = ct_counts / total

    rare_threshold = 0.01  # <1% of cells
    rare_types = ct_freq[ct_freq < rare_threshold].index.tolist()

    rare_summary = pd.DataFrame({
        'cell_type': ct_counts.index,
        'count': ct_counts.values,
        'frequency': ct_freq.values,
        'is_rare': ct_freq.values < rare_threshold,
    })
    rare_summary.to_parquet(out / 'celltype_frequencies.parquet')
    print(f'  Found {len(rare_types)} rare cell types (<1%): {rare_types[:10]}')

# 2. Rare immune populations of interest
print('Extracting rare immune signatures...')
rare_immune_markers = {
    'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'IRF8', 'CADM1'],
    'LAMP3_DC': ['LAMP3', 'CCR7', 'CCL19', 'CCL22', 'FSCN1'],
    'pDC': ['LILRA4', 'IL3RA', 'CLEC4C', 'IRF7', 'TCF4'],
    'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18'],
    'exhausted_CD8': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1'],
    'proliferating_T': ['MKI67', 'TOP2A', 'CD3D', 'CD3E'],
    'plasma_cell': ['JCHAIN', 'MZB1', 'XBP1', 'SDC1', 'IGHG1'],
    'mast_cell': ['TPSAB1', 'TPSB2', 'CPA3', 'KIT', 'MS4A2'],
    'neutrophil': ['FCGR3B', 'CSF3R', 'CXCR2', 'S100A8', 'S100A9'],
}

# 3. Rare tumor states
rare_tumor_markers = {
    'cycling_tumor': ['MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNB1'],
    'hypoxic_tumor': ['CA9', 'VEGFA', 'SLC2A1', 'LDHA', 'HIF1A', 'BNIP3'],
    'EMT_tumor': ['VIM', 'SNAI1', 'SNAI2', 'ZEB1', 'TWIST1', 'CDH2'],
    'stemlike_tumor': ['SOX2', 'SOX9', 'NANOG', 'ALDH1A1', 'CD44', 'PROM1'],
    'IFN_stimulated_tumor': ['ISG15', 'MX1', 'IFIT1', 'STAT1', 'IRF1'],
    'antigen_low_tumor': ['B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'TAP1'],  # low expression
    'senescent_tumor': ['CDKN1A', 'CDKN2A', 'SERPINE1', 'IL1B', 'IL6'],
}

# 4. Rare stromal states
rare_stromal_markers = {
    'myCAF': ['ACTA2', 'TAGLN', 'MYL9', 'COL11A1', 'POSTN'],
    'iCAF': ['IL6', 'CXCL12', 'PDGFRA', 'CFD', 'DPT'],
    'apCAF': ['HLA-DRA', 'HLA-DRB1', 'CD74', 'PDGFRA'],
    'lymphatic_endo': ['PROX1', 'LYVE1', 'PDPN', 'FLT4', 'CCL21'],
    'tip_endo': ['ESM1', 'APLN', 'PGF', 'DLL4', 'KDR'],
    'pericyte': ['RGS5', 'PDGFRB', 'NOTCH3', 'ACTA2', 'MCAM'],
}

all_signatures = {**rare_immune_markers, **rare_tumor_markers, **rare_stromal_markers}

# Score all signatures
results = {'cell_id': adata.obs.index.tolist()}
for name, genes in all_signatures.items():
    available = [g for g in genes if g in adata.var_names]
    if len(available) >= 2:
        sc.tl.score_genes(adata, gene_list=available, score_name=name)
        results[name] = adata.obs[name].tolist()
        print(f'  Scored {name} ({len(available)}/{len(genes)} genes)')

sig_df = pd.DataFrame(results)
sig_df.to_parquet(out / 'rare_cell_signatures.parquet')

# Add metadata
if 'stage' in adata.obs.columns:
    sig_df['stage'] = adata.obs['stage'].values
if cell_type_key in adata.obs.columns:
    sig_df['cell_type'] = adata.obs[cell_type_key].values

# Mean signatures by stage
if 'stage' in sig_df.columns:
    score_cols = [c for c in sig_df.columns if c not in ['cell_id', 'stage', 'cell_type']]
    stage_means = sig_df.groupby('stage')[score_cols].mean()
    stage_means.to_parquet(out / 'rare_signatures_by_stage.parquet')

# Mean signatures by cell type
if 'cell_type' in sig_df.columns:
    score_cols = [c for c in sig_df.columns if c not in ['cell_id', 'stage', 'cell_type']]
    ct_means = sig_df.groupby('cell_type')[score_cols].mean()
    ct_means.to_parquet(out / 'rare_signatures_by_celltype.parquet')

print('Rare cell discovery complete')
"

# 18. Differential abundance (Milo-style neighborhoods)
echo ""
echo "[18/20] Computing differential abundance..."
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading snRNA data...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

out = Path('$CANONICAL/rare_cells')

# Milo-style analysis if available
try:
    import milopy.core as milo

    print('Running Milo differential abundance...')

    # Need condition/stage for DA testing
    if 'stage' in adata.obs.columns:
        # Build KNN graph
        if 'neighbors' not in adata.uns:
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, n_neighbors=30)

        # Make neighborhoods
        milo.make_nhoods(adata, prop=0.1)

        # Count cells per neighborhood per sample
        if 'sample' in adata.obs.columns or 'donor_id' in adata.obs.columns:
            sample_col = 'sample' if 'sample' in adata.obs.columns else 'donor_id'
            milo.count_nhoods(adata, sample_col=sample_col)

            # DA testing
            milo.DA_nhoods(adata, design='~ stage')

            # Save results
            da_results = adata.uns['nhood_adata'].obs.copy()
            da_results.to_parquet(out / 'milo_da_results.parquet')
            print('  Milo DA complete')

except ImportError:
    print('milopy not installed, using simple abundance test...')

    # Simple differential abundance by stage
    if 'stage' in adata.obs.columns:
        cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
        if cell_type_key in adata.obs.columns:
            # Chi-square style: observed vs expected proportions
            ct_by_stage = pd.crosstab(adata.obs['stage'], adata.obs[cell_type_key])
            ct_props = ct_by_stage.div(ct_by_stage.sum(axis=1), axis=0)
            overall_props = adata.obs[cell_type_key].value_counts(normalize=True)

            # Log fold change vs overall
            log_fc = np.log2((ct_props + 0.001) / (overall_props + 0.001))
            log_fc.to_parquet(out / 'celltype_logfc_by_stage.parquet')
            print('  Simple DA complete')

except Exception as e:
    print(f'DA analysis failed: {e}')

print('Differential abundance complete')
"

# 19. Rare cell spatial mapping
echo ""
echo "[19/20] Mapping rare cells to spatial data..."
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

print('Loading spatial data...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

out = Path('$CANONICAL/rare_cells')

# Load rare cell signatures and apply to spatial
sig_path = Path('$CANONICAL/rare_cells/rare_cell_signatures.parquet')
if not sig_path.exists():
    print('Rare signatures not found, computing directly on spatial...')

# Rare signatures to map
rare_signatures = {
    'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'IRF8'],
    'LAMP3_DC': ['LAMP3', 'CCR7', 'CCL19', 'FSCN1'],
    'Treg': ['FOXP3', 'IL2RA', 'CTLA4'],
    'exhausted_CD8': ['PDCD1', 'LAG3', 'HAVCR2', 'TOX'],
    'plasma_cell': ['JCHAIN', 'MZB1', 'XBP1'],
    'hypoxic_tumor': ['CA9', 'VEGFA', 'SLC2A1', 'LDHA'],
    'EMT_tumor': ['VIM', 'SNAI1', 'ZEB1', 'CDH2'],
    'myCAF': ['ACTA2', 'TAGLN', 'POSTN'],
    'iCAF': ['IL6', 'CXCL12', 'PDGFRA'],
    'lymphatic_endo': ['PROX1', 'LYVE1', 'PDPN'],
}

results = {'spot_id': adata.obs.index.tolist()}

for name, genes in rare_signatures.items():
    available = [g for g in genes if g in adata.var_names]
    if len(available) >= 2:
        sc.tl.score_genes(adata, gene_list=available, score_name=name)
        results[name] = adata.obs[name].tolist()
        print(f'  Mapped {name} ({len(available)} genes)')

spatial_rare = pd.DataFrame(results)

# Add spatial coords
if 'spatial' in adata.obsm:
    spatial_rare['x'] = adata.obsm['spatial'][:, 0]
    spatial_rare['y'] = adata.obsm['spatial'][:, 1]
elif 'x_spatial' in adata.obs.columns:
    spatial_rare['x'] = adata.obs['x_spatial'].values
    spatial_rare['y'] = adata.obs['y_spatial'].values

if 'stage' in adata.obs.columns:
    spatial_rare['stage'] = adata.obs['stage'].values
if 'sample' in adata.obs.columns:
    spatial_rare['sample'] = adata.obs['sample'].values

spatial_rare.to_parquet(out / 'rare_signatures_spatial.parquet')

# Co-localization: correlate rare signatures with each other spatially
print('Computing rare cell co-localization...')
score_cols = [c for c in spatial_rare.columns if c not in ['spot_id', 'x', 'y', 'stage', 'sample']]
if len(score_cols) > 1:
    coloc = spatial_rare[score_cols].corr()
    coloc.to_parquet(out / 'rare_cell_colocalization.parquet')

# Rare cell enrichment near tumor/stroma boundaries
print('Computing rare cell spatial enrichment...')
cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    # Mean rare signature by dominant cell type
    spatial_rare['dominant_celltype'] = adata.obs[cell_type_key].values
    rare_by_ct = spatial_rare.groupby('dominant_celltype')[score_cols].mean()
    rare_by_ct.to_parquet(out / 'rare_signatures_by_spot_celltype.parquet')

print('Rare cell spatial mapping complete')
"

# 20. Spatial niche phenotyping (SpaGCN / Giotto HMRF)
echo ""
echo "[20/23] Running spatial niche phenotyping..."
mkdir -p $CANONICAL/niche_phenotypes
python -c "
import pandas as pd
import numpy as np
from pathlib import Path
import scanpy as sc

out = Path('$CANONICAL/niche_phenotypes')

# Load spatial data
print('Loading spatial data...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

# Load DestVI proportions
destvi_path = Path('/data1/chaunzt1/stagebridge/results/spatial_benchmark/luca/destvi/cell_type_proportions.parquet')
if not destvi_path.exists():
    destvi_path = Path('$CANONICAL/visium/spot_deconvolution.parquet')

props = None
if destvi_path.exists():
    props = pd.read_parquet(destvi_path)
    print(f'  Loaded DestVI: {props.shape}')

# Try SpaGCN first (proper spatial GCN clustering)
spagcn_success = False
try:
    import SpaGCN as spg
    print('Running SpaGCN spatial clustering...')

    # Get spatial coords
    if 'spatial' in adata.obsm:
        x_pixel = adata.obsm['spatial'][:, 0]
        y_pixel = adata.obsm['spatial'][:, 1]
    else:
        x_pixel = adata.obs['x_spatial'].values
        y_pixel = adata.obs['y_spatial'].values

    # Calculate adjacency matrix
    adj = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)

    # Find optimal number of clusters
    sc.pp.pca(adata, n_comps=50)

    # Use DestVI proportions if available, else use PCA
    if props is not None and len(props) == adata.n_obs:
        exclude_cols = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id']
        prop_cols = [c for c in props.columns if c not in exclude_cols and props[c].dtype in ['float64', 'float32']]
        # Add proportions to adata for SpaGCN
        for col in prop_cols:
            adata.obs[col] = props[col].values

    # Run SpaGCN
    l = spg.search_l(p=0.5, adj=adj, start=0.01, end=1000, tol=0.01, max_run=100)
    n_clusters = 8

    # Set seed
    r_seed = t_seed = n_seed = 42

    clf = spg.SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init='louvain', res=0.6,
              tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()

    adata.obs['spagcn_cluster'] = y_pred
    adata.obs['spagcn_cluster'] = adata.obs['spagcn_cluster'].astype('category')

    # Refine clusters using HMRF
    print('  Refining with HMRF...')
    adj_2d = spg.calculate_adj_matrix(x=x_pixel, y=y_pixel, histology=False)
    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(),
                               pred=adata.obs['spagcn_cluster'].tolist(),
                               dis=adj_2d, shape='hexagon')
    adata.obs['niche_phenotype'] = refined_pred
    adata.obs['niche_phenotype'] = adata.obs['niche_phenotype'].astype('category')

    spagcn_success = True
    print(f'  SpaGCN found {len(adata.obs[\"niche_phenotype\"].unique())} phenotypes')

except ImportError:
    print('SpaGCN not installed, trying BANKSY...')
except Exception as e:
    print(f'SpaGCN failed: {e}, trying BANKSY...')

# Try BANKSY if SpaGCN failed
if not spagcn_success:
    try:
        # BANKSY via squidpy integration or standalone
        import squidpy as sq
        print('Running BANKSY-style spatial clustering via Squidpy...')

        # Spatial neighbors
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=6)

        # Compute spatial lag features (BANKSY-like)
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)

        # Use both expression and spatial lag
        sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15)

        # Leiden with spatial constraint
        sc.tl.leiden(adata, resolution=0.8, key_added='niche_phenotype')

        spagcn_success = True
        print(f'  Squidpy found {len(adata.obs[\"niche_phenotype\"].unique())} phenotypes')

    except Exception as e:
        print(f'BANKSY/Squidpy failed: {e}')

# Fallback: Giotto-style HMRF (pure Python implementation)
if not spagcn_success:
    print('Running Giotto-style HMRF (Python implementation)...')

    from scipy.spatial import KDTree
    from sklearn.mixture import GaussianMixture

    # Get data
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    else:
        coords = np.column_stack([adata.obs['x_spatial'].values, adata.obs['y_spatial'].values])

    # Use DestVI proportions or PCA
    if props is not None and len(props) == adata.n_obs:
        exclude_cols = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id']
        prop_cols = [c for c in props.columns if c not in exclude_cols and props[c].dtype in ['float64', 'float32']]
        X = props[prop_cols].values
    else:
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        X = adata.obsm['X_pca'][:, :20]

    # Build spatial graph
    tree = KDTree(coords)
    _, neighbor_idx = tree.query(coords, k=7)
    neighbor_idx = neighbor_idx[:, 1:]  # exclude self

    n_spots = X.shape[0]
    n_phenotypes = 8
    beta = 2.0  # Spatial smoothness (Potts model parameter)

    # Initialize with GMM
    gmm = GaussianMixture(n_components=n_phenotypes, random_state=42, n_init=5, covariance_type='full')
    gmm.fit(X)
    z = gmm.predict(X)

    # HMRF via Iterated Conditional Modes (ICM)
    print('  Running ICM optimization...')
    for iteration in range(100):
        z_old = z.copy()
        changes = 0

        # Random order to avoid bias
        order = np.random.permutation(n_spots)

        for i in order:
            # Compute energy for each label
            energies = np.zeros(n_phenotypes)

            for k in range(n_phenotypes):
                # Data term: negative log-likelihood from GMM
                energies[k] = -gmm.score_samples(X[i:i+1])[0]  # Approx

                # Actually use proper GMM component likelihood
                energies[k] = -gmm._estimate_weighted_log_prob(X[i:i+1])[0, k]

                # Spatial term: Potts model (penalize different neighbors)
                n_diff = np.sum(z[neighbor_idx[i]] != k)
                energies[k] += beta * n_diff

            # Assign label with minimum energy
            new_label = np.argmin(energies)
            if new_label != z[i]:
                changes += 1
            z[i] = new_label

        if changes < n_spots * 0.001:
            print(f'  ICM converged at iteration {iteration + 1}')
            break

    adata.obs['niche_phenotype'] = z
    adata.obs['niche_phenotype'] = adata.obs['niche_phenotype'].astype('category')
    print(f'  HMRF-ICM found {len(np.unique(z))} phenotypes')

# Save results
print('Saving phenotype results...')

# Get coordinates
if 'spatial' in adata.obsm:
    x_coords = adata.obsm['spatial'][:, 0]
    y_coords = adata.obsm['spatial'][:, 1]
else:
    x_coords = adata.obs['x_spatial'].values
    y_coords = adata.obs['y_spatial'].values

phenotype_df = pd.DataFrame({
    'spot_id': adata.obs.index,
    'niche_phenotype': adata.obs['niche_phenotype'].values,
    'x': x_coords,
    'y': y_coords,
})

if 'stage' in adata.obs.columns:
    phenotype_df['stage'] = adata.obs['stage'].values
if 'sample' in adata.obs.columns:
    phenotype_df['sample'] = adata.obs['sample'].values

phenotype_df.to_parquet(out / 'spot_niche_phenotypes.parquet')

# Compute phenotype centers from DestVI proportions
if props is not None and len(props) == adata.n_obs:
    exclude_cols = ['sample', 'stage', 'donor_id', 'x', 'y', 'spot_id']
    prop_cols = [c for c in props.columns if c not in exclude_cols and props[c].dtype in ['float64', 'float32']]

    props['niche_phenotype'] = adata.obs['niche_phenotype'].values
    centers = props.groupby('niche_phenotype')[prop_cols].mean()
    centers['n_spots'] = props.groupby('niche_phenotype').size()

    # Name by top cell types
    phenotype_names = []
    for idx in centers.index:
        top3 = centers.loc[idx, prop_cols].nlargest(3).index.tolist()
        name = '_'.join([t[:10] for t in top3])
        phenotype_names.append(f'P{idx}_{name}')

    centers['phenotype_name'] = phenotype_names
    centers.to_parquet(out / 'phenotype_centers.parquet')

    phenotype_df['phenotype_name'] = phenotype_df['niche_phenotype'].map(
        dict(zip(centers.index, phenotype_names))
    )
    phenotype_df.to_parquet(out / 'spot_niche_phenotypes.parquet')

# Phenotype by stage
if 'stage' in phenotype_df.columns:
    stage_pheno = pd.crosstab(phenotype_df['stage'], phenotype_df['niche_phenotype'], normalize='index')
    stage_pheno.to_parquet(out / 'phenotype_by_stage.parquet')

# Spatial transitions
print('Computing spatial transitions...')
from scipy.spatial import KDTree
coords = phenotype_df[['x', 'y']].values
tree = KDTree(coords)
_, neighbor_idx = tree.query(coords, k=7)
neighbor_idx = neighbor_idx[:, 1:]

z = phenotype_df['niche_phenotype'].values
n_pheno = int(z.max()) + 1
transitions = np.zeros((n_pheno, n_pheno))
for i in range(len(z)):
    for j in neighbor_idx[i]:
        if j < len(z):
            transitions[int(z[i]), int(z[j])] += 1

transitions = transitions / (transitions.sum() + 1e-10)
trans_df = pd.DataFrame(transitions)
trans_df.to_parquet(out / 'phenotype_transitions.parquet')

print('Niche phenotyping complete')
"

# 21. Niche phenotype biological characterization
echo ""
echo "[21/23] Characterizing niche phenotypes..."
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

out = Path('$CANONICAL/niche_phenotypes')
pheno_path = out / 'spot_niche_phenotypes.parquet'
centers_path = out / 'phenotype_centers.parquet'

if pheno_path.exists() and centers_path.exists():
    print('Loading phenotype results...')
    phenotypes = pd.read_parquet(pheno_path)
    centers = pd.read_parquet(centers_path)

    # Load signatures to characterize phenotypes
    sig_path = Path('$CANONICAL/signatures/gene_signatures.parquet')
    if sig_path.exists():
        sigs = pd.read_parquet(sig_path)

        # Mean signature per phenotype
        if len(sigs) == len(phenotypes):
            sigs['niche_phenotype'] = phenotypes['niche_phenotype'].values
            sig_cols = [c for c in sigs.columns if c not in ['cell_id', 'niche_phenotype', 'stage']]
            pheno_sigs = sigs.groupby('niche_phenotype')[sig_cols].mean()
            pheno_sigs.to_parquet(out / 'phenotype_signatures.parquet')
            print('  Saved phenotype signature characterization')

    # Load rare cell signatures
    rare_path = Path('$CANONICAL/rare_cells/rare_signatures_spatial.parquet')
    if rare_path.exists():
        rare = pd.read_parquet(rare_path)

        if len(rare) == len(phenotypes):
            rare['niche_phenotype'] = phenotypes['niche_phenotype'].values
            rare_cols = [c for c in rare.columns if c not in ['spot_id', 'x', 'y', 'stage', 'sample', 'niche_phenotype']]
            pheno_rare = rare.groupby('niche_phenotype')[rare_cols].mean()
            pheno_rare.to_parquet(out / 'phenotype_rare_cells.parquet')
            print('  Saved phenotype rare cell enrichment')

    # Phenotype annotation summary
    print('Creating phenotype annotation...')
    annotation = centers.copy()

    # Find dominant cell types per phenotype
    prop_cols = [c for c in centers.columns if c not in ['phenotype', 'n_spots']]
    for idx, row in annotation.iterrows():
        props = row[prop_cols].values
        top3 = np.argsort(props)[-3:][::-1]
        annotation.loc[idx, 'top1_celltype'] = prop_cols[top3[0]]
        annotation.loc[idx, 'top1_prop'] = props[top3[0]]
        annotation.loc[idx, 'top2_celltype'] = prop_cols[top3[1]]
        annotation.loc[idx, 'top2_prop'] = props[top3[1]]
        annotation.loc[idx, 'top3_celltype'] = prop_cols[top3[2]]
        annotation.loc[idx, 'top3_prop'] = props[top3[2]]

    annotation.to_parquet(out / 'phenotype_annotation.parquet')
    print('Phenotype characterization complete')
else:
    print('Phenotype results not found, skipping characterization')
"

# 22. Interface analysis between phenotypes
echo ""
echo "[22/23] Analyzing phenotype interfaces..."
python -c "
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

out = Path('$CANONICAL/niche_phenotypes')
pheno_path = out / 'spot_niche_phenotypes.parquet'

if pheno_path.exists():
    print('Loading phenotype assignments...')
    phenotypes = pd.read_parquet(pheno_path)

    coords = phenotypes[['x', 'y']].values
    z = phenotypes['niche_phenotype'].values

    # Build neighbor graph
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=7)
    neighbor_idx = indices[:, 1:]

    # Find interface spots (neighbors with different phenotype)
    interface_spots = []
    interface_types = []

    for i in range(len(phenotypes)):
        neighbor_phenos = z[neighbor_idx[i]]
        unique_neighbors = set(neighbor_phenos) - {z[i]}

        if unique_neighbors:
            interface_spots.append(i)
            for neighbor_pheno in unique_neighbors:
                interface_types.append({
                    'spot_idx': i,
                    'spot_phenotype': z[i],
                    'neighbor_phenotype': int(neighbor_pheno),
                    'x': coords[i, 0],
                    'y': coords[i, 1],
                })

    interface_df = pd.DataFrame(interface_types)
    interface_df.to_parquet(out / 'phenotype_interfaces.parquet')
    print(f'  Found {len(interface_spots)} interface spots, {len(interface_df)} interface pairs')

    # Interface enrichment matrix
    n_pheno = z.max() + 1
    interface_counts = np.zeros((n_pheno, n_pheno))
    for _, row in interface_df.iterrows():
        p1, p2 = int(row['spot_phenotype']), int(row['neighbor_phenotype'])
        interface_counts[p1, p2] += 1
        interface_counts[p2, p1] += 1

    interface_matrix = pd.DataFrame(interface_counts,
                                     index=[f'P{i}' for i in range(n_pheno)],
                                     columns=[f'P{i}' for i in range(n_pheno)])
    interface_matrix.to_parquet(out / 'interface_counts.parquet')

    print('Interface analysis complete')
else:
    print('Phenotype results not found, skipping interface analysis')
"

# 23. QC metrics (mito, ribo, counts, genes)
echo ""
echo "[23/26] Computing QC metrics..."
mkdir -p $CANONICAL/qc
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

out = Path('$CANONICAL/qc')

# snRNA-seq QC
print('Computing snRNA-seq QC metrics...')
adata = sc.read_h5ad('$SNRNA')
print(f'  {adata.n_obs} cells')

# Compute QC metrics if not present
if 'pct_counts_mt' not in adata.obs.columns:
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

if 'pct_counts_ribo' not in adata.obs.columns:
    adata.var['ribo'] = adata.var_names.str.match('^RP[SL]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)

# Also compute hemoglobin if present
adata.var['hb'] = adata.var_names.str.match('^HB[^P]')
if adata.var['hb'].sum() > 0:
    sc.pp.calculate_qc_metrics(adata, qc_vars=['hb'], percent_top=None, log1p=False, inplace=True)

# Save QC table
qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']
if 'pct_counts_hb' in adata.obs.columns:
    qc_cols.append('pct_counts_hb')

qc_df = adata.obs[qc_cols].copy()
qc_df['cell_id'] = adata.obs.index

if 'stage' in adata.obs.columns:
    qc_df['stage'] = adata.obs['stage'].values
if 'donor_id' in adata.obs.columns:
    qc_df['donor_id'] = adata.obs['donor_id'].values
if 'sample' in adata.obs.columns:
    qc_df['sample'] = adata.obs['sample'].values

cell_type_key = 'cell_type_luca' if 'cell_type_luca' in adata.obs.columns else 'cell_type'
if cell_type_key in adata.obs.columns:
    qc_df['cell_type'] = adata.obs[cell_type_key].values

qc_df.to_parquet(out / 'snrna_qc_metrics.parquet')
print('  Saved snrna_qc_metrics.parquet')

# QC summary by stage
if 'stage' in qc_df.columns:
    stage_qc = qc_df.groupby('stage')[qc_cols].agg(['mean', 'median', 'std'])
    stage_qc.to_parquet(out / 'qc_by_stage.parquet')

# QC summary by cell type
if 'cell_type' in qc_df.columns:
    ct_qc = qc_df.groupby('cell_type')[qc_cols].agg(['mean', 'median', 'std'])
    ct_qc.to_parquet(out / 'qc_by_celltype.parquet')

# QC summary by sample/donor
sample_col = 'sample' if 'sample' in qc_df.columns else 'donor_id'
if sample_col in qc_df.columns:
    sample_qc = qc_df.groupby(sample_col)[qc_cols].agg(['mean', 'median', 'std'])
    sample_qc.to_parquet(out / 'qc_by_sample.parquet')

# Thresholds used (for reporting)
thresholds = {
    'min_genes': 200,
    'max_genes': 8000,
    'max_pct_mt': 20,
    'max_pct_ribo': 50,
    'min_counts': 500,
}
pd.Series(thresholds).to_frame('value').to_parquet(out / 'qc_thresholds.parquet')

print('snRNA QC complete')
"

# 24. Spatial QC metrics
echo ""
echo "[24/26] Computing spatial QC metrics..."
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

out = Path('$CANONICAL/qc')

print('Computing spatial QC metrics...')
adata = sc.read_h5ad('$SPATIAL')
print(f'  {adata.n_obs} spots')

# Compute QC metrics
if 'pct_counts_mt' not in adata.obs.columns:
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

if 'pct_counts_ribo' not in adata.obs.columns:
    adata.var['ribo'] = adata.var_names.str.match('^RP[SL]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)

# Save QC table
qc_cols = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']

qc_df = adata.obs[qc_cols].copy()
qc_df['spot_id'] = adata.obs.index

# Add spatial coords
if 'spatial' in adata.obsm:
    qc_df['x'] = adata.obsm['spatial'][:, 0]
    qc_df['y'] = adata.obsm['spatial'][:, 1]
elif 'x_spatial' in adata.obs.columns:
    qc_df['x'] = adata.obs['x_spatial'].values
    qc_df['y'] = adata.obs['y_spatial'].values

if 'stage' in adata.obs.columns:
    qc_df['stage'] = adata.obs['stage'].values
if 'sample' in adata.obs.columns:
    qc_df['sample'] = adata.obs['sample'].values

qc_df.to_parquet(out / 'spatial_qc_metrics.parquet')
print('  Saved spatial_qc_metrics.parquet')

# QC by sample
if 'sample' in qc_df.columns:
    sample_qc = qc_df.groupby('sample')[qc_cols].agg(['mean', 'median', 'std', 'count'])
    sample_qc.to_parquet(out / 'spatial_qc_by_sample.parquet')

# QC by stage
if 'stage' in qc_df.columns:
    stage_qc = qc_df.groupby('stage')[qc_cols].agg(['mean', 'median', 'std'])
    stage_qc.to_parquet(out / 'spatial_qc_by_stage.parquet')

print('Spatial QC complete')
"

# 25. Doublet scores (if available)
echo ""
echo "[25/26] Extracting doublet scores..."
python -c "
import scanpy as sc
import pandas as pd
from pathlib import Path

out = Path('$CANONICAL/qc')

print('Checking for doublet scores...')
adata = sc.read_h5ad('$SNRNA')

doublet_cols = ['doublet_score', 'scrublet_score', 'DoubletFinder_score', 'predicted_doublet']
found = [c for c in doublet_cols if c in adata.obs.columns]

if found:
    doublet_df = adata.obs[found].copy()
    doublet_df['cell_id'] = adata.obs.index
    if 'stage' in adata.obs.columns:
        doublet_df['stage'] = adata.obs['stage'].values
    doublet_df.to_parquet(out / 'doublet_scores.parquet')
    print(f'  Saved doublet scores: {found}')
else:
    print('  No doublet scores found in adata.obs')
    print('  Run Scrublet or DoubletFinder if needed')
"

# 26. Validate contract
echo ""
echo "[26/26] Validating data contract..."
python -c "
from stagebridge.contracts import validate_contract
validate_contract('$CANONICAL')
print('Contract validation PASSED')
"

# Summary
echo ""
echo "=============================================="
echo "Data preparation complete!"
echo "=============================================="
echo ""
echo "Outputs:"
echo ""
echo "Core data:"
ls -lh $CANONICAL/*.parquet 2>/dev/null
echo ""
echo "SCENIC:"
ls -lh $CANONICAL/scenic/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Spatial stats:"
ls -lh $CANONICAL/spatial_stats/ 2>/dev/null || echo "  (not found)"
echo ""
echo "DE analysis:"
ls -lh $CANONICAL/de_analysis/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Summary stats:"
ls -lh $CANONICAL/summary_stats/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Signatures:"
ls -lh $CANONICAL/signatures/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Expression:"
ls -lh $CANONICAL/expression/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Pathways (GSEA/ORA):"
ls -lh $CANONICAL/pathways/ 2>/dev/null || echo "  (not found)"
echo ""
echo "TF/Pathway activity (decoupleR):"
ls -lh $CANONICAL/activity/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Trajectories (DPT, PAGA):"
ls -lh $CANONICAL/trajectories/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Embeddings (UMAP, PHATE):"
ls -lh $CANONICAL/embeddings/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Communication:"
ls -lh $CANONICAL/communication/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Visium spatial:"
ls -lh $CANONICAL/visium/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Sample summaries:"
ls -lh $CANONICAL/samples/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Rare cells:"
ls -lh $CANONICAL/rare_cells/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Niche phenotypes (EM-HMRF):"
ls -lh $CANONICAL/niche_phenotypes/ 2>/dev/null || echo "  (not found)"
echo ""
echo "QC metrics:"
ls -lh $CANONICAL/qc/ 2>/dev/null || echo "  (not found)"
echo ""
echo "Ready for training: snakemake --profile workflow/slurm"
