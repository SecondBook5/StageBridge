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

# 11. Validate contract
echo ""
echo "[11/11] Validating data contract..."
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
echo "Ready for training: snakemake --profile workflow/slurm"
