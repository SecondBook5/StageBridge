#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/booka/StageBridge/logs/full_pipeline_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/full_pipeline_%j.err

# Complete StageBridge data preparation + figure generation pipeline
# 1. Use pre-computed LuCA labels (luca_cell_type column)
# 2. Run LIANA with LuCA cell types
# 3. Full data prep (signatures, DE, activity, trajectories, niches)
# 4. Generate advanced publication figures (UMAP, PHATE, GSEA, Ricci, etc.)
#
# ENVIRONMENT:
#   Data prep (Parts 1-5): stagebridge_env
#   Figures (Part 6): stagebridge_figures (has PHATE, GSEA, GraphRicciCurvature)
#
# To create figures env:
#   conda env create -f /home/booka/StageBridge/envs/figures_env.yaml \
#       -p /scratch/chaunzt1/envs/stagebridge_figures

set -e

echo "=============================================="
echo "StageBridge Full Pipeline"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

module load miniforge3
eval "$(conda shell.bash hook)"

# Use main env for data processing
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

# =============================================================================
# Paths
# =============================================================================
DATA="/data1/chaunzt1/stagebridge/processed/luad_evo"
CANONICAL="$DATA/canonical"
SNRNA="$DATA/snrna_with_celltypes.h5ad"
SPATIAL="$DATA/spatial_merged.h5ad"
LUCA_REF="/scratch/chaunzt1/stagebridge/references/luca/luca_core_atlas.h5ad"
FIGURES="/home/booka/StageBridge/figures/publication"
LOCAL="/home/booka/StageBridge/data"

mkdir -p $CANONICAL/liana_luca
mkdir -p $CANONICAL/signatures
mkdir -p $CANONICAL/activity
mkdir -p $CANONICAL/spatial_activity
mkdir -p $CANONICAL/trajectories
mkdir -p $CANONICAL/embeddings
mkdir -p $CANONICAL/communication
mkdir -p $CANONICAL/key_genes
mkdir -p $CANONICAL/visium
mkdir -p $CANONICAL/niches
mkdir -p $CANONICAL/qc
mkdir -p $FIGURES
mkdir -p $LOCAL
mkdir -p /home/booka/StageBridge/logs

export SNRNA SPATIAL CANONICAL

# =============================================================================
# PART 1: LuCA Label Transfer + LIANA (33 cell types)
# =============================================================================
echo ""
echo "=============================================="
echo "PART 1: LuCA Label Transfer + LIANA"
echo "=============================================="

if [ -f "$CANONICAL/liana_luca/liana_interactions_luca.parquet" ]; then
    echo "  SKIP: LuCA LIANA results exist"
else
python << 'PYTHON_LIANA'
import sys
import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

SNRNA = os.environ['SNRNA']
CANONICAL = Path(os.environ['CANONICAL'])
OUTPUT = CANONICAL / "liana_luca"
LOCAL = Path("/home/booka/StageBridge/data")

print("=" * 60)
print("Step 1.1: Load snRNA data")
print("=" * 60)
sys.stdout.flush()

adata = sc.read_h5ad(SNRNA)
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"HLCA cell types: {adata.obs['cell_type'].nunique()} types")
print(f"LuCA cell types: {adata.obs['luca_cell_type'].nunique()} types (pre-computed)")
sys.stdout.flush()

# Verify luca_cell_type exists
if 'luca_cell_type' not in adata.obs.columns:
    raise ValueError("luca_cell_type column not found! Expected pre-computed labels.")

print("\nLuCA cell type distribution:")
print(adata.obs['luca_cell_type'].value_counts().head(15))
sys.stdout.flush()

print("\n" + "=" * 60)
print("Step 1.2: Run LIANA with LuCA cell types")
print("=" * 60)
sys.stdout.flush()

import liana as li

print("Running LIANA (this takes ~30 min)...")
sys.stdout.flush()

li.mt.rank_aggregate(
    adata,
    groupby="luca_cell_type",
    resource_name="consensus",
    expr_prop=0.1,
    n_perms=100,
    verbose=True,
    use_raw=False,
)

liana_results = adata.uns["liana_res"].copy()
print(f"\nLIANA found {len(liana_results):,} interactions")
print(f"  Sources: {liana_results['source'].nunique()}")
print(f"  Targets: {liana_results['target'].nunique()}")
sys.stdout.flush()

print("\n" + "=" * 60)
print("Step 1.4: Save results")
print("=" * 60)
sys.stdout.flush()

liana_out = OUTPUT / "liana_interactions_luca.parquet"
liana_results.to_parquet(liana_out)
print(f"Saved: {liana_out}")

celltypes_out = OUTPUT / "luca_cell_type_mapping.parquet"
adata.obs[["cell_type", "luca_cell_type"]].to_parquet(celltypes_out)
print(f"Saved: {celltypes_out}")

import json
summary = {
    "n_cells": int(adata.n_obs),
    "n_luca_types": int(adata.obs["luca_cell_type"].nunique()),
    "n_hlca_types": int(adata.obs["cell_type"].nunique()),
    "n_interactions": int(len(liana_results)),
    "n_sources": int(liana_results["source"].nunique()),
    "n_targets": int(liana_results["target"].nunique()),
}
with open(OUTPUT / "liana_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT / 'liana_summary.json'}")

import shutil
shutil.copy(liana_out, LOCAL / "liana_interactions_luca.parquet")
print(f"Copied to local: {LOCAL / 'liana_interactions_luca.parquet'}")

print("\nLIANA with LuCA complete!")
PYTHON_LIANA
fi

# =============================================================================
# PART 2: Gene Signatures
# =============================================================================
echo ""
echo "=============================================="
echo "PART 2: Gene Signatures (EMT, senescence, SASP)"
echo "=============================================="

if [ -f "$CANONICAL/signatures/gene_signatures.parquet" ]; then
    echo "  SKIP: exists"
else
python << 'PYTHON_SIG'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

from stagebridge.biology import (
    compute_emt_score,
    compute_senescence_score,
    compute_sasp_score,
)

SNRNA = os.environ['SNRNA']
CANONICAL = Path(os.environ['CANONICAL'])
out = CANONICAL / 'signatures'

print('Loading snRNA data...')
sys.stdout.flush()
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs:,} cells')

print('Computing gene signatures...')
sys.stdout.flush()
signatures = pd.DataFrame({'cell_id': adata.obs.index})
signatures['emt_score'] = compute_emt_score(adata)
signatures['senescence_score'] = compute_senescence_score(adata)
signatures['sasp_score'] = compute_sasp_score(adata)

if 'stage' in adata.obs.columns:
    signatures['stage'] = adata.obs['stage'].values
if 'luca_cell_type' in adata.obs.columns:
    signatures['luca_cell_type'] = adata.obs['luca_cell_type'].values
if 'cell_type' in adata.obs.columns:
    signatures['cell_type'] = adata.obs['cell_type'].values

signatures.to_parquet(out / 'gene_signatures.parquet')
print(f'  Saved gene_signatures.parquet')

score_cols = ['emt_score', 'senescence_score', 'sasp_score']

if 'stage' in signatures.columns:
    summary = signatures.groupby('stage')[score_cols].mean()
    summary.to_parquet(out / 'signatures_by_stage.parquet')
    print(f'  Saved signatures_by_stage.parquet')

if 'luca_cell_type' in signatures.columns:
    summary = signatures.groupby('luca_cell_type')[score_cols].mean()
    summary.to_parquet(out / 'signatures_by_luca_celltype.parquet')
    print(f'  Saved signatures_by_luca_celltype.parquet')

    # Stage x cell type crosstab
    if 'stage' in signatures.columns:
        crosstab = signatures.groupby(['stage', 'luca_cell_type'])[score_cols].mean()
        crosstab.to_parquet(out / 'signatures_by_stage_celltype.parquet')
        print(f'  Saved signatures_by_stage_celltype.parquet')

print('Signatures complete')
PYTHON_SIG
fi

# =============================================================================
# PART 3: decoupleR TF/Pathway Activity
# =============================================================================
echo ""
echo "=============================================="
echo "PART 3: decoupleR TF/Pathway Activity"
echo "=============================================="

if [ -f "$CANONICAL/activity/pathway_activity_progeny.parquet" ]; then
    echo "  SKIP: exists"
else
python << 'PYTHON_ACTIVITY'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from scipy import stats

SNRNA = os.environ['SNRNA']
CANONICAL = Path(os.environ['CANONICAL'])
out = CANONICAL / 'activity'

print('Running decoupleR activity analysis...')
sys.stdout.flush()

print('Loading snRNA data...')
adata = sc.read_h5ad(SNRNA)
print(f'  {adata.n_obs:,} cells')
sys.stdout.flush()

import decoupler as dc

# =========================================================================
# CollecTRI TF activity (with retry and timeout fallback)
# =========================================================================
import time

def extract_activity_df(adata, net, method='ulm'):
    """Extract activity scores from adata.obsm as DataFrame."""
    # Try different key patterns used by decoupler versions
    possible_keys = [
        f'{method}_estimate',      # newer versions
        f'{method}estimate',       # some versions
        f'X_{method}',             # alternative
        method,                    # simple
    ]

    key = None
    for k in possible_keys:
        if k in adata.obsm:
            key = k
            break

    if key is None:
        print(f"  Available obsm keys: {list(adata.obsm.keys())}")
        raise KeyError(f"Could not find {method} results in adata.obsm")

    print(f"  Using obsm key: {key}")
    estimates = adata.obsm[key]
    if hasattr(estimates, 'toarray'):
        estimates = estimates.toarray()
    sources = net['source'].unique()
    return pd.DataFrame(estimates, index=adata.obs_names, columns=sources)

print('Computing TF activity (CollecTRI)...')
sys.stdout.flush()
tf_df = None

max_retries = 3
for attempt in range(max_retries):
    try:
        print(f'  Attempt {attempt + 1}/{max_retries}: downloading CollecTRI...')
        sys.stdout.flush()
        collectri = dc.op.collectri(organism='human')
        dc.mt.ulm(data=adata, net=collectri)
        tf_df = extract_activity_df(adata, collectri)
        tf_df.to_parquet(out / 'tf_activity_collectri.parquet')
        print(f'  CollecTRI TF activity saved: {tf_df.shape[1]} TFs')
        break
    except Exception as e:
        print(f'  Attempt {attempt + 1} failed: {e}')
        if attempt < max_retries - 1:
            wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s
            print(f'  Retrying in {wait_time}s...')
            sys.stdout.flush()
            time.sleep(wait_time)
        else:
            print('  WARNING: CollecTRI failed after all retries, using expression fallback')
sys.stdout.flush()

# =========================================================================
# PROGENy pathway activity
# =========================================================================
print('Computing pathway activity (PROGENy)...')
sys.stdout.flush()
progeny = dc.op.progeny(organism='human')
dc.mt.ulm(data=adata, net=progeny)
pw_df = extract_activity_df(adata, progeny)
pw_df.to_parquet(out / 'pathway_activity_progeny.parquet')
print(f'  PROGENy pathway activity saved: {pw_df.shape[1]} pathways')

# =========================================================================
# AP-1 SCORING - use CollecTRI TF ACTIVITIES (not just expression)
# =========================================================================
print('Computing AP-1 transcription factor scores...')
sys.stdout.flush()

AP1_TFS = ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND', 'ATF3', 'ATF4', 'ATF6', 'BATF', 'BATF2', 'BATF3']

ap1_results = pd.DataFrame(index=adata.obs_names)
ap1_results['stage'] = adata.obs['stage'].values
if 'luca_cell_type' in adata.obs.columns:
    ap1_results['luca_cell_type'] = adata.obs['luca_cell_type'].values
if 'cell_type' in adata.obs.columns:
    ap1_results['cell_type'] = adata.obs['cell_type'].values
if 'donor_id' in adata.obs.columns:
    ap1_results['donor_id'] = adata.obs['donor_id'].values

# Primary: Use CollecTRI TF ACTIVITY scores (target-based inference)
if tf_df is not None:
    ap1_in_collectri = [tf for tf in AP1_TFS if tf in tf_df.columns]
    print(f'  AP-1 TFs in CollecTRI: {ap1_in_collectri}')

    if ap1_in_collectri:
        # Add individual TF activities
        for tf in ap1_in_collectri:
            ap1_results[f'{tf}_activity'] = tf_df[tf].values

        # Composite AP-1 activity score (mean of all AP-1 TF activities)
        ap1_results['ap1_activity_score'] = tf_df[ap1_in_collectri].mean(axis=1).values
        print(f'  Created ap1_activity_score from {len(ap1_in_collectri)} TFs')
else:
    print('  CollecTRI not available, using expression-based fallback')

# Fallback/supplement: Expression-based scoring
ap1_tfs_found = [g for g in AP1_TFS if g in adata.var_names]
print(f'  AP-1 TF genes in expression: {ap1_tfs_found}')

if ap1_tfs_found:
    sc.tl.score_genes(adata, gene_list=ap1_tfs_found, score_name='ap1_expr_score')
    ap1_results['ap1_expr_score'] = adata.obs['ap1_expr_score'].values

    # Add raw expression for each TF
    for g in ap1_tfs_found:
        expr = adata[:, g].X
        ap1_results[f'{g}_expr'] = expr.toarray().flatten() if hasattr(expr, 'toarray') else expr.flatten()

# Add IL1B for correlation analysis
if 'IL1B' in adata.var_names:
    expr = adata[:, 'IL1B'].X
    ap1_results['IL1B'] = expr.toarray().flatten() if hasattr(expr, 'toarray') else expr.flatten()

ap1_results.to_parquet(out / 'ap1_scores.parquet')
print(f'  Saved AP-1 scores for {len(ap1_results):,} cells')

# Summary statistics
print('\n  === AP-1 Activity by Stage ===')
score_col = 'ap1_activity_score' if 'ap1_activity_score' in ap1_results.columns else 'ap1_expr_score'
if score_col in ap1_results.columns:
    print(ap1_results.groupby('stage')[score_col].agg(['mean', 'std', 'count']))

    # Stage correlation
    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4}
    valid_stages = ap1_results['stage'].isin(stage_map.keys())
    if valid_stages.sum() > 0:
        stage_numeric = ap1_results.loc[valid_stages, 'stage'].map(stage_map)
        r, p = stats.spearmanr(ap1_results.loc[valid_stages, score_col], stage_numeric)
        print(f'\n  AP-1 vs Stage correlation: r={r:.3f}, p={p:.2e}')

    if 'IL1B' in ap1_results.columns:
        r, p = stats.spearmanr(ap1_results[score_col], ap1_results['IL1B'])
        print(f'  AP-1 vs IL1B correlation: r={r:.3f}, p={p:.2e}')
sys.stdout.flush()

# =========================================================================
# SENESCENCE / SASP SCORING
# =========================================================================
print('Computing Senescence and SASP scores...')
sys.stdout.flush()

# Core senescence markers (CDKN2A=p16, CDKN1A=p21)
SENESCENCE_CORE = ['CDKN2A', 'CDKN1A', 'TP53', 'RB1', 'GLB1', 'SERPINE1', 'IGFBP7']

# SASP (Senescence-Associated Secretory Phenotype) - inflammatory cytokines/chemokines
SASP_GENES = [
    'IL1A', 'IL1B', 'IL6', 'IL8', 'CXCL8', 'CXCL1', 'CXCL2', 'CXCL3',
    'CCL2', 'CCL3', 'CCL5', 'CCL20',
    'MMP1', 'MMP3', 'MMP9', 'MMP10', 'MMP12',
    'SERPINE1', 'SERPINE2', 'IGFBP3', 'IGFBP5', 'IGFBP7',
    'VEGFA', 'FGF2', 'HGF', 'AREG',
    'TGFB1', 'TGFB2', 'TNFA', 'TNF',
]

# Extended senescence signature (SenMayo - curated from literature)
SENMAYO_GENES = [
    'CDKN1A', 'CDKN2A', 'CDKN2B', 'TP53', 'RB1',
    'SERPINE1', 'IGFBP7', 'GLB1', 'LMNB1',
    'IL1A', 'IL1B', 'IL6', 'IL8', 'CXCL8',
    'MMP1', 'MMP3', 'MMP9',
    'VEGFA', 'FGF2', 'CCL2', 'TGFB1',
]

sen_core_found = [g for g in SENESCENCE_CORE if g in adata.var_names]
sasp_found = [g for g in SASP_GENES if g in adata.var_names]
senmayo_found = [g for g in SENMAYO_GENES if g in adata.var_names]

print(f'  Senescence core found: {sen_core_found}')
print(f'  SASP genes found: {len(sasp_found)}/{len(SASP_GENES)}')
print(f'  SenMayo genes found: {len(senmayo_found)}/{len(SENMAYO_GENES)}')

sen_results = pd.DataFrame(index=adata.obs_names)
sen_results['stage'] = adata.obs['stage'].values
if 'luca_cell_type' in adata.obs.columns:
    sen_results['luca_cell_type'] = adata.obs['luca_cell_type'].values
if 'cell_type' in adata.obs.columns:
    sen_results['cell_type'] = adata.obs['cell_type'].values
if 'donor_id' in adata.obs.columns:
    sen_results['donor_id'] = adata.obs['donor_id'].values

if sen_core_found:
    sc.tl.score_genes(adata, gene_list=sen_core_found, score_name='senescence_core_score')
    sen_results['senescence_core_score'] = adata.obs['senescence_core_score'].values

if sasp_found:
    sc.tl.score_genes(adata, gene_list=sasp_found, score_name='sasp_score')
    sen_results['sasp_score'] = adata.obs['sasp_score'].values

if senmayo_found:
    sc.tl.score_genes(adata, gene_list=senmayo_found, score_name='senmayo_score')
    sen_results['senmayo_score'] = adata.obs['senmayo_score'].values

# Add individual key markers
for g in ['CDKN2A', 'CDKN1A', 'IL1B', 'IL6', 'SERPINE1', 'GLB1']:
    if g in adata.var_names:
        expr = adata[:, g].X
        sen_results[g] = expr.toarray().flatten() if hasattr(expr, 'toarray') else expr.flatten()

sen_results.to_parquet(out / 'senescence_sasp_scores.parquet')
print(f'  Saved senescence/SASP scores for {len(sen_results):,} cells')

if 'sasp_score' in sen_results.columns:
    print('\n  === SASP Score by Stage ===')
    print(sen_results.groupby('stage')['sasp_score'].agg(['mean', 'std', 'count']))
if 'senescence_core_score' in sen_results.columns:
    print('\n  === Senescence Core Score by Stage ===')
    print(sen_results.groupby('stage')['senescence_core_score'].agg(['mean', 'std', 'count']))
sys.stdout.flush()

# =========================================================================
# Summary statistics by stage and cell type
# =========================================================================
print('Computing summaries by stage and cell type...')
sys.stdout.flush()
for name, df in [('tf', tf_df), ('pathway', pw_df)]:
    if df is None:
        continue
    df_copy = df.copy()

    if 'stage' in adata.obs.columns:
        df_copy['stage'] = adata.obs['stage'].values
        df_copy.groupby('stage').mean(numeric_only=True).to_parquet(out / f'{name}_activity_by_stage.parquet')

    if 'luca_cell_type' in adata.obs.columns:
        df_copy['luca_cell_type'] = adata.obs['luca_cell_type'].values
        df_copy.groupby('luca_cell_type').mean(numeric_only=True).to_parquet(out / f'{name}_activity_by_luca_celltype.parquet')

        if 'stage' in adata.obs.columns:
            df_copy.groupby(['stage', 'luca_cell_type']).mean(numeric_only=True).to_parquet(out / f'{name}_activity_by_stage_celltype.parquet')

print('decoupleR + AP-1 + Senescence/SASP complete')
PYTHON_ACTIVITY
fi

# =============================================================================
# PART 4: Communication Summary (using LuCA LIANA)
# =============================================================================
echo ""
echo "=============================================="
echo "PART 4: Communication Summary"
echo "=============================================="

if [ -f "$CANONICAL/communication/communication_matrix_luca.parquet" ]; then
    echo "  SKIP: exists"
else
python << 'PYTHON_COMM'
import pandas as pd
from pathlib import Path
import os
import sys

CANONICAL = Path(os.environ['CANONICAL'])
out = CANONICAL / 'communication'
out.mkdir(exist_ok=True)

print('Loading LuCA LIANA results...')
sys.stdout.flush()
liana_path = CANONICAL / 'liana_luca' / 'liana_interactions_luca.parquet'
if not liana_path.exists():
    print('  ERROR: LuCA LIANA not found')
    sys.exit(1)

liana = pd.read_parquet(liana_path)
print(f'  {len(liana):,} interactions')

print('Computing communication matrix...')
sys.stdout.flush()
comm_matrix = liana.groupby(['source', 'target']).size().unstack(fill_value=0)
comm_matrix.to_parquet(out / 'communication_matrix_luca.parquet')

print('Extracting IL1B interactions...')
sys.stdout.flush()
il1b_mask = (
    liana['ligand_complex'].str.contains('IL1B', case=False, na=False) |
    liana['receptor_complex'].str.contains('IL1R', case=False, na=False)
)
il1b = liana[il1b_mask]
if len(il1b) > 0:
    il1b.to_parquet(out / 'il1b_interactions_luca.parquet')
    print(f'  {len(il1b)} IL1B-related interactions')

print('Communication summary complete')
PYTHON_COMM
fi

# =============================================================================
# PART 5: Key Gene Expression
# =============================================================================
echo ""
echo "=============================================="
echo "PART 5: Key Gene Expression"
echo "=============================================="

if [ -f "$CANONICAL/key_genes/key_genes_by_stage.parquet" ]; then
    echo "  SKIP: exists"
else
python << 'PYTHON_GENES'
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

SNRNA = os.environ['SNRNA']
CANONICAL = Path(os.environ['CANONICAL'])
out = CANONICAL / 'key_genes'
out.mkdir(exist_ok=True)

print('Loading snRNA data...')
sys.stdout.flush()
adata = sc.read_h5ad(SNRNA)

# Comprehensive gene sets for biological exploration
GENE_SETS = {
    # IL1B signaling axis (core hypothesis)
    'il1b_pathway': ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN', 'IL1A', 'IL18', 'NLRP3', 'CASP1', 'PYCARD'],

    # Pro-inflammatory cytokines
    'inflammatory': ['IL6', 'TNF', 'IL8', 'CXCL8', 'CCL2', 'CCL5', 'CXCL10', 'CXCL12', 'IL10', 'TGFB1'],

    # CAF markers and activation
    'caf_markers': ['ACTA2', 'FAP', 'PDGFRA', 'PDGFRB', 'COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'POSTN', 'TAGLN',
                   'THY1', 'S100A4', 'VIM', 'DCN', 'LUM', 'SPARC', 'MMP2', 'MMP9', 'MMP11', 'LOXL2'],

    # CAF subtypes (iCAF vs myCAF)
    'caf_icaf': ['IL6', 'CXCL12', 'PDGFRA', 'CFD', 'DPT', 'LMNA', 'CLU'],
    'caf_mycaf': ['ACTA2', 'TAGLN', 'MYL9', 'TPM1', 'TPM2', 'MYLK', 'CNN1'],

    # Senescence markers (CDKN genes, SASP)
    'senescence': ['CDKN1A', 'CDKN2A', 'CDKN2B', 'TP53', 'RB1', 'GLB1', 'SERPINE1', 'IGFBP3', 'IGFBP7'],

    # SASP factors
    'sasp': ['IL1A', 'IL1B', 'IL6', 'IL8', 'CXCL1', 'CXCL2', 'CCL2', 'CCL20', 'MMP1', 'MMP3', 'MMP10',
            'SERPINE1', 'IGFBP2', 'IGFBP3', 'IGFBP4', 'IGFBP5', 'IGFBP6', 'IGFBP7', 'GDF15', 'VEGFA'],

    # AP-1 complex and targets
    'ap1_complex': ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND', 'ATF3', 'ATF4', 'BATF'],
    'ap1_targets': ['MMP1', 'MMP9', 'IL8', 'VEGFA', 'CCND1', 'BCL2', 'CDKN1A'],

    # Epithelial markers and differentiation
    'epithelial': ['EPCAM', 'KRT7', 'KRT8', 'KRT18', 'KRT19', 'CDH1', 'CLDN1', 'CLDN4', 'OCLN', 'TJP1'],

    # Alveolar type markers
    'at2_markers': ['SFTPC', 'SFTPA1', 'SFTPA2', 'SFTPB', 'SFTPD', 'ABCA3', 'LAMP3', 'SLC34A2'],
    'at1_markers': ['AGER', 'PDPN', 'CAV1', 'AQP5', 'HOPX', 'CLIC5'],

    # EMT markers
    'emt_epithelial': ['CDH1', 'EPCAM', 'CLDN1', 'OCLN', 'KRT8', 'KRT18', 'KRT19'],
    'emt_mesenchymal': ['VIM', 'CDH2', 'FN1', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1', 'TWIST2'],

    # Stemness and progenitor
    'stemness': ['SOX2', 'SOX9', 'KLF4', 'MYC', 'NANOG', 'POU5F1', 'ALDH1A1', 'CD44', 'CD133', 'LGR5'],

    # Oncogenes and tumor suppressors
    'oncogenes': ['KRAS', 'EGFR', 'MYC', 'ERBB2', 'MET', 'ALK', 'ROS1', 'BRAF', 'PIK3CA'],
    'tumor_suppressors': ['TP53', 'RB1', 'CDKN2A', 'STK11', 'KEAP1', 'NF1', 'PTEN', 'APC', 'SMAD4'],

    # Immune checkpoint
    'immune_checkpoint': ['CD274', 'PDCD1LG2', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'IDO1', 'SIGLEC15'],

    # T cell markers
    't_cell': ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B', 'FOXP3', 'IL2RA', 'GZMB', 'PRF1', 'IFNG'],
    't_exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'EOMES'],

    # Macrophage polarization
    'mac_m1': ['CD68', 'CD80', 'CD86', 'NOS2', 'IL1B', 'IL6', 'TNF', 'CXCL9', 'CXCL10'],
    'mac_m2': ['CD163', 'MRC1', 'CD206', 'ARG1', 'IL10', 'TGFB1', 'CCL18', 'CCL22'],

    # Hypoxia response
    'hypoxia': ['HIF1A', 'EPAS1', 'VEGFA', 'SLC2A1', 'LDHA', 'PDK1', 'PGK1', 'ENO1', 'CA9', 'BNIP3'],

    # Wnt signaling
    'wnt': ['CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'TCF7', 'LEF1', 'WNT3A', 'WNT5A', 'DKK1', 'SFRP1'],

    # Notch signaling
    'notch': ['NOTCH1', 'NOTCH2', 'NOTCH3', 'JAG1', 'JAG2', 'DLL1', 'DLL4', 'HES1', 'HEY1', 'RBPJ'],

    # TGFb signaling
    'tgfb': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4', 'SMAD7'],

    # NF-kB signaling
    'nfkb': ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL', 'IKBKB', 'IKBKG', 'NFKBIA', 'NFKBIB'],

    # Dendritic cells
    'dendritic_cells': ['ITGAX', 'CD1C', 'CLEC9A', 'XCR1', 'BATF3', 'IRF8', 'CD83', 'CD86', 'CCR7',
                        'LAMP3', 'CCL17', 'CCL19', 'CCL22', 'FSCN1', 'CD40', 'HLA-DRA', 'HLA-DRB1'],
    'cdc1': ['CLEC9A', 'XCR1', 'BATF3', 'IRF8', 'CADM1', 'THBD'],
    'cdc2': ['CD1C', 'FCER1A', 'CLEC10A', 'CD1E', 'SIRPA'],
    'pdc': ['LILRA4', 'IL3RA', 'CLEC4C', 'TCF4', 'IRF7', 'GZMB'],

    # Neutrophils
    'neutrophils': ['FCGR3B', 'CXCR1', 'CXCR2', 'CSF3R', 'S100A8', 'S100A9', 'S100A12',
                   'CEACAM8', 'MMP8', 'MMP9', 'ELANE', 'MPO', 'CTSG', 'LCN2'],
    'neutrophil_nets': ['MPO', 'ELANE', 'CTSG', 'PRTN3', 'AZU1', 'PADI4'],

    # B cells and plasma
    'b_cells': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'PAX5', 'BANK1', 'CD22', 'CD72'],
    'plasma_cells': ['JCHAIN', 'MZB1', 'XBP1', 'PRDM1', 'SDC1', 'IGHG1', 'IGHA1'],

    # NK cells
    'nk_cells': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRB1', 'KLRC1', 'KLRF1', 'NCR1',
                'NCR3', 'FCGR3A', 'PRF1', 'GZMB', 'GZMH', 'IFNG'],

    # Mast cells
    'mast_cells': ['KIT', 'TPSAB1', 'TPSB2', 'CPA3', 'MS4A2', 'HDC', 'HPGDS', 'CMA1'],

    # Stromal - Endothelial
    'endothelial': ['PECAM1', 'CDH5', 'VWF', 'KDR', 'FLT1', 'CLDN5', 'PLVAP', 'ACKR1',
                   'SELE', 'VCAM1', 'ICAM1', 'NOS3', 'EMCN'],
    'lymphatic': ['LYVE1', 'PROX1', 'PDPN', 'FLT4', 'CCL21'],

    # Stromal - Pericytes and SMC
    'pericytes': ['RGS5', 'PDGFRB', 'CSPG4', 'MCAM', 'NOTCH3', 'DES', 'ACTA2'],
    'smooth_muscle': ['MYH11', 'ACTG2', 'CNN1', 'TAGLN', 'DES', 'ACTA2', 'MYOCD'],

    # Mesothelial
    'mesothelial': ['MSLN', 'WT1', 'UPK3B', 'KRT19', 'CALB2', 'BNC1'],

    # Tregs specifically
    'tregs': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2', 'TNFRSF18', 'CCR8', 'TIGIT', 'LAYN', 'BATF'],

    # CD8 subtypes
    'cd8_naive': ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R'],
    'cd8_effector': ['GZMB', 'PRF1', 'IFNG', 'TBX21', 'NKG7', 'GNLY'],
    'cd8_memory': ['IL7R', 'EOMES', 'GZMK', 'CCR7'],
    'cd8_resident': ['ITGAE', 'CD69', 'ZNF683', 'CXCR6'],

    # Monocytes
    'monocytes': ['CD14', 'FCGR3A', 'VCAN', 'S100A8', 'S100A9', 'FCN1', 'MNDA'],
    'classical_mono': ['CD14', 'S100A8', 'S100A9', 'S100A12', 'FCN1', 'VCAN'],
    'nonclassical_mono': ['FCGR3A', 'CDKN1C', 'MTSS1', 'CX3CR1', 'LST1'],
}

# Flatten all genes for bulk extraction
ALL_GENES = sorted(set(g for genes in GENE_SETS.values() for g in genes))
available = [g for g in ALL_GENES if g in adata.var_names]
print(f'  Found {len(available)}/{len(ALL_GENES)} genes across all pathways')

if available:
    X = adata[:, available].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    expr_df = pd.DataFrame(X, index=adata.obs.index, columns=available)

    # Add metadata
    if 'stage' in adata.obs.columns:
        expr_df['stage'] = adata.obs['stage'].values
    if 'luca_cell_type' in adata.obs.columns:
        expr_df['luca_cell_type'] = adata.obs['luca_cell_type'].values
    if 'cell_type' in adata.obs.columns:
        expr_df['cell_type'] = adata.obs['cell_type'].values

    # Save full expression matrix
    expr_df.to_parquet(out / 'all_key_genes_expression.parquet')
    print(f'  Saved all_key_genes_expression.parquet ({len(available)} genes)')

    # Summary by stage
    if 'stage' in expr_df.columns:
        stage_summary = expr_df.groupby('stage')[available].agg(['mean', 'std', lambda x: (x > 0).mean()])
        stage_summary.columns = ['_'.join(col).strip() for col in stage_summary.columns.values]
        stage_summary.to_parquet(out / 'key_genes_by_stage.parquet')
        print(f'  Saved key_genes_by_stage.parquet')

    # Summary by LuCA cell type
    if 'luca_cell_type' in expr_df.columns:
        ct_summary = expr_df.groupby('luca_cell_type')[available].agg(['mean', 'std', lambda x: (x > 0).mean()])
        ct_summary.columns = ['_'.join(col).strip() for col in ct_summary.columns.values]
        ct_summary.to_parquet(out / 'key_genes_by_luca_celltype.parquet')
        print(f'  Saved key_genes_by_luca_celltype.parquet')

    # Per-pathway summaries
    print('  Computing per-pathway summaries...')
    pathway_dir = out / 'pathways'
    pathway_dir.mkdir(exist_ok=True)

    for pathway_name, pathway_genes in GENE_SETS.items():
        pathway_available = [g for g in pathway_genes if g in available]
        if len(pathway_available) >= 2:
            # By stage
            if 'stage' in expr_df.columns:
                pw_stage = expr_df.groupby('stage')[pathway_available].mean()
                pw_stage.to_parquet(pathway_dir / f'{pathway_name}_by_stage.parquet')

            # By cell type
            if 'luca_cell_type' in expr_df.columns:
                pw_ct = expr_df.groupby('luca_cell_type')[pathway_available].mean()
                pw_ct.to_parquet(pathway_dir / f'{pathway_name}_by_luca_celltype.parquet')

            # Stage x cell type
            if 'stage' in expr_df.columns and 'luca_cell_type' in expr_df.columns:
                pw_crosstab = expr_df.groupby(['stage', 'luca_cell_type'])[pathway_available].mean()
                pw_crosstab.to_parquet(pathway_dir / f'{pathway_name}_by_stage_celltype.parquet')

            print(f'    {pathway_name}: {len(pathway_available)} genes')

    # Pathway scores (mean of z-scored genes)
    print('  Computing pathway scores...')
    from scipy.stats import zscore

    expr_z = expr_df[available].apply(zscore, nan_policy='omit')
    pathway_scores = pd.DataFrame(index=expr_df.index)

    for pathway_name, pathway_genes in GENE_SETS.items():
        pathway_available = [g for g in pathway_genes if g in available]
        if len(pathway_available) >= 2:
            pathway_scores[pathway_name] = expr_z[pathway_available].mean(axis=1)

    pathway_scores['stage'] = expr_df['stage'].values if 'stage' in expr_df.columns else None
    pathway_scores['luca_cell_type'] = expr_df['luca_cell_type'].values if 'luca_cell_type' in expr_df.columns else None

    pathway_scores.to_parquet(out / 'pathway_scores.parquet')
    print(f'  Saved pathway_scores.parquet ({len(GENE_SETS)} pathways)')

    # Pathway scores by stage
    if 'stage' in pathway_scores.columns:
        pw_names = [c for c in pathway_scores.columns if c not in ['stage', 'luca_cell_type']]
        pw_by_stage = pathway_scores.groupby('stage')[pw_names].mean()
        pw_by_stage.to_parquet(out / 'pathway_scores_by_stage.parquet')
        print(f'  Saved pathway_scores_by_stage.parquet')

    # Pathway scores by cell type
    if 'luca_cell_type' in pathway_scores.columns:
        pw_by_ct = pathway_scores.groupby('luca_cell_type')[pw_names].mean()
        pw_by_ct.to_parquet(out / 'pathway_scores_by_luca_celltype.parquet')
        print(f'  Saved pathway_scores_by_luca_celltype.parquet')

print('Key genes complete')
PYTHON_GENES
fi

# =============================================================================
# PART 6: Advanced Single-Cell Visualizations (Scanpy/Squidpy style)
# =============================================================================
echo ""
echo "=============================================="
echo "Switching to figures environment..."
echo "=============================================="

# Switch to figures environment (has PHATE, GSEA, GraphRicciCurvature, etc.)
FIGURES_ENV="/scratch/chaunzt1/envs/stagebridge_figures"
if [ -d "$FIGURES_ENV" ]; then
    conda activate "$FIGURES_ENV"
    echo "  Activated: stagebridge_figures"
else
    echo "  WARNING: stagebridge_figures not found, using stagebridge_env"
    echo "  To create: conda env create -f /home/booka/StageBridge/envs/figures_env.yaml -p $FIGURES_ENV"
fi

echo ""
echo "=============================================="
echo "PART 6A: Advanced SC Visualizations (UMAP, PHATE, etc.)"
echo "=============================================="

python << 'PYTHON_ADVANCED_SC'
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False)

SNRNA = os.environ['SNRNA']
CANONICAL = Path(os.environ['CANONICAL'])
FIGURES = Path("/home/booka/StageBridge/figures/publication")
ADV = FIGURES / 'advanced'
ADV.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("ADVANCED SINGLE-CELL VISUALIZATIONS")
print("=" * 60)
sys.stdout.flush()

# Deep color palette
STAGE_COLORS = {
    'Normal': '#228B22', 'AAH': '#4169E1', 'AIS': '#6B3FA0',
    'MIA': '#DAA520', 'LUAD': '#8B1A1A',
}

def save_fig(fig, path, name):
    for fmt in ['png', 'pdf']:
        fig.savefig(path / f'{name}.{fmt}', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    plt.close(fig)

print("\n[LOAD] Loading snRNA data...")
sys.stdout.flush()
adata = sc.read_h5ad(SNRNA)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

# Add pathway scores to adata.obs
print("\n[SCORES] Adding pathway scores to adata...")
sys.stdout.flush()
try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores.parquet')
    score_cols = [c for c in pw_scores.columns if c not in ['stage', 'luca_cell_type']]
    for col in score_cols:
        if col in pw_scores.columns:
            adata.obs[col] = pw_scores[col].values
    print(f"  Added {len(score_cols)} pathway scores")
except Exception as e:
    print(f"  Could not load pathway scores: {e}")

# =============================================================================
# UMAP - Compute if not present
# =============================================================================
print("\n[UMAP] Computing UMAP embedding...")
sys.stdout.flush()

if 'X_umap' not in adata.obsm:
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, min_dist=0.3)
    print("  Computed fresh UMAP")
else:
    print("  Using existing UMAP")

# =============================================================================
# UMAP by Stage - Publication quality
# =============================================================================
print("\n[UMAP] Stage distribution...")
sys.stdout.flush()

try:
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color='stage', palette=STAGE_COLORS, ax=ax, show=False,
              title='', frameon=False, size=3, alpha=0.7)
    ax.set_title('UMAP by Disease Stage', fontsize=14, fontweight='bold', pad=10)
    save_fig(fig, ADV, 'umap_stage')
    print("  Saved umap_stage.png/pdf")
except Exception as e:
    print(f"  UMAP stage failed: {e}")

# =============================================================================
# UMAP by Cell Type
# =============================================================================
print("\n[UMAP] Cell type distribution...")
sys.stdout.flush()

try:
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.umap(adata, color='luca_cell_type', ax=ax, show=False,
              title='', frameon=False, size=3, alpha=0.7, legend_loc='on data',
              legend_fontsize=6, legend_fontoutline=2)
    ax.set_title('UMAP by LuCA Cell Type', fontsize=14, fontweight='bold', pad=10)
    save_fig(fig, ADV, 'umap_celltype')
    print("  Saved umap_celltype.png/pdf")
except Exception as e:
    print(f"  UMAP celltype failed: {e}")

# =============================================================================
# UMAP Feature Plots - Key Markers
# =============================================================================
print("\n[UMAP] Key marker expression...")
sys.stdout.flush()

key_genes = ['IL1B', 'IL1R1', 'SFTPC', 'EPCAM', 'VIM', 'ACTA2', 'CD3D', 'CD68', 'PECAM1']
available_genes = [g for g in key_genes if g in adata.var_names]

for gene in available_genes:
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc.pl.umap(adata, color=gene, ax=ax, show=False, cmap='magma',
                  title='', frameon=False, size=5, alpha=0.8, vmin=0)
        ax.set_title(f'{gene} Expression', fontsize=14, fontweight='bold', pad=10)
        save_fig(fig, ADV, f'umap_{gene.lower()}')
        print(f"  Saved umap_{gene.lower()}.png/pdf")
    except Exception as e:
        print(f"  UMAP {gene} failed: {e}")

# =============================================================================
# UMAP Pathway Scores
# =============================================================================
print("\n[UMAP] Pathway score overlays...")
sys.stdout.flush()

pathway_scores_to_plot = ['il1b_pathway', 'caf_markers', 'senescence', 'emt_mesenchymal',
                          'mac_m1', 'mac_m2', 't_exhaustion', 'hypoxia']

for pw in pathway_scores_to_plot:
    if pw in adata.obs.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            sc.pl.umap(adata, color=pw, ax=ax, show=False, cmap='plasma',
                      title='', frameon=False, size=5, alpha=0.8)
            ax.set_title(f'{pw.replace("_", " ").title()} Score', fontsize=14, fontweight='bold', pad=10)
            save_fig(fig, ADV, f'umap_{pw}')
            print(f"  Saved umap_{pw}.png/pdf")
        except Exception as e:
            print(f"  UMAP {pw} failed: {e}")

# =============================================================================
# PHATE Embedding
# =============================================================================
print("\n[PHATE] Computing PHATE embedding...")
sys.stdout.flush()

try:
    import phate
    phate_op = phate.PHATE(n_jobs=-1, random_state=42, verbose=0)

    # Use PCA for speed
    if 'X_pca' not in adata.obsm:
        sc.pp.pca(adata, n_comps=50)

    X_phate = phate_op.fit_transform(adata.obsm['X_pca'][:, :30])
    adata.obsm['X_phate'] = X_phate
    print("  Computed PHATE embedding")

    # PHATE by stage
    fig, ax = plt.subplots(figsize=(10, 8))
    for stage in ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']:
        mask = adata.obs['stage'] == stage
        ax.scatter(X_phate[mask, 0], X_phate[mask, 1], s=3, alpha=0.6,
                  c=STAGE_COLORS[stage], label=stage)
    ax.set_xlabel('PHATE 1', fontweight='bold')
    ax.set_ylabel('PHATE 2', fontweight='bold')
    ax.set_title('PHATE by Disease Stage', fontsize=14, fontweight='bold', pad=10)
    ax.legend(markerscale=3, frameon=False)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig(fig, ADV, 'phate_stage')
    print("  Saved phate_stage.png/pdf")

    # PHATE colored by pathway scores
    for pw in ['il1b_pathway', 'senescence', 'emt_mesenchymal']:
        if pw in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(9, 7))
            scatter = ax.scatter(X_phate[:, 0], X_phate[:, 1], s=3, alpha=0.7,
                               c=adata.obs[pw], cmap='inferno')
            ax.set_xlabel('PHATE 1', fontweight='bold')
            ax.set_ylabel('PHATE 2', fontweight='bold')
            ax.set_title(f'PHATE: {pw.replace("_", " ").title()}', fontsize=14, fontweight='bold', pad=10)
            plt.colorbar(scatter, ax=ax, shrink=0.6, label='Score')
            for spine in ax.spines.values():
                spine.set_visible(False)
            save_fig(fig, ADV, f'phate_{pw}')
            print(f"  Saved phate_{pw}.png/pdf")

except ImportError:
    print("  PHATE not installed, skipping")
except Exception as e:
    print(f"  PHATE failed: {e}")

# =============================================================================
# Diffusion Pseudotime
# =============================================================================
print("\n[DPT] Diffusion pseudotime...")
sys.stdout.flush()

try:
    sc.tl.diffmap(adata)

    # Find root cell (likely Normal epithelial)
    normal_mask = (adata.obs['stage'] == 'Normal')
    if normal_mask.sum() > 0:
        adata.uns['iroot'] = np.where(normal_mask)[0][0]
        sc.tl.dpt(adata)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False, cmap='viridis',
                  title='', frameon=False, size=5, alpha=0.8)
        ax.set_title('Diffusion Pseudotime', fontsize=14, fontweight='bold', pad=10)
        save_fig(fig, ADV, 'umap_pseudotime')
        print("  Saved umap_pseudotime.png/pdf")

        # Pseudotime by stage boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        stage_order = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']
        colors = [STAGE_COLORS[s] for s in stage_order]
        data_to_plot = [adata.obs[adata.obs['stage'] == s]['dpt_pseudotime'].dropna() for s in stage_order]
        bp = ax.boxplot(data_to_plot, labels=stage_order, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Diffusion Pseudotime', fontweight='bold')
        ax.set_title('Pseudotime Distribution by Stage', fontsize=14, fontweight='bold', pad=10)
        save_fig(fig, ADV, 'pseudotime_boxplot')
        print("  Saved pseudotime_boxplot.png/pdf")

except Exception as e:
    print(f"  DPT failed: {e}")

# =============================================================================
# Scanpy Dotplot - Marker Genes
# =============================================================================
print("\n[DOTPLOT] Cell type marker dotplot...")
sys.stdout.flush()

try:
    marker_genes = {
        'Epithelial': ['EPCAM', 'KRT18', 'CDH1'],
        'AT2': ['SFTPC', 'SFTPA1'],
        'AT1': ['AGER', 'HOPX'],
        'Basal': ['KRT5', 'TP63'],
        'EMT': ['VIM', 'SNAI1', 'ZEB1'],
        'CAF': ['ACTA2', 'FAP', 'COL1A1'],
        'Macrophage': ['CD68', 'CD163', 'IL1B'],
        'T cell': ['CD3D', 'CD8A', 'CD4'],
        'Endothelial': ['PECAM1', 'VWF'],
    }

    available_markers = {}
    for cat, genes in marker_genes.items():
        avail = [g for g in genes if g in adata.var_names]
        if avail:
            available_markers[cat] = avail

    if available_markers:
        sc.pl.dotplot(adata, available_markers, groupby='luca_cell_type',
                     dendrogram=True, standard_scale='var', cmap='Reds',
                     save='_celltype_markers.png', show=False)
        print("  Saved dotplot (via scanpy)")
except Exception as e:
    print(f"  Dotplot failed: {e}")

# =============================================================================
# Scanpy Matrixplot - Pathway Heatmap
# =============================================================================
print("\n[MATRIXPLOT] Pathway heatmap by cell type...")
sys.stdout.flush()

try:
    pathway_cols = [c for c in adata.obs.columns if c in
                   ['il1b_pathway', 'caf_markers', 'senescence', 'sasp', 'emt_mesenchymal',
                    'mac_m1', 'mac_m2', 't_exhaustion', 'hypoxia', 'inflammatory']]

    if pathway_cols:
        # Create a fake gene matrix from pathway scores for matrixplot
        pathway_df = adata.obs.groupby('luca_cell_type')[pathway_cols].mean()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pathway_df, cmap='inferno', ax=ax, annot=True, fmt='.2f',
                   linewidths=0.5, cbar_kws={'shrink': 0.6, 'label': 'Mean Score'})
        ax.set_title('Pathway Scores by Cell Type', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Pathway', fontweight='bold')
        ax.set_ylabel('Cell Type', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        save_fig(fig, ADV, 'pathway_celltype_heatmap')
        print("  Saved pathway_celltype_heatmap.png/pdf")

except Exception as e:
    print(f"  Matrixplot failed: {e}")

# =============================================================================
# Stacked Violin Plot
# =============================================================================
print("\n[STACKED VIOLIN] Key markers...")
sys.stdout.flush()

try:
    genes_for_violin = ['IL1B', 'IL1R1', 'SFTPC', 'VIM', 'ACTA2', 'CD68', 'CD3D']
    avail_violin = [g for g in genes_for_violin if g in adata.var_names]

    if avail_violin:
        sc.pl.stacked_violin(adata, avail_violin, groupby='stage',
                            swap_axes=False, cmap='plasma',
                            save='_stage_markers.png', show=False)
        print("  Saved stacked violin (via scanpy)")
except Exception as e:
    print(f"  Stacked violin failed: {e}")

# =============================================================================
# Rank Genes Groups - Stage DEG
# =============================================================================
print("\n[DEG] Differential expression by stage...")
sys.stdout.flush()

try:
    sc.tl.rank_genes_groups(adata, 'stage', method='wilcoxon', n_genes=50)

    fig, ax = plt.subplots(figsize=(14, 8))
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, ax=ax, show=False,
                                   cmap='RdBu_r', values_to_plot='logfoldchanges')
    ax.set_title('Top DEG by Stage', fontsize=14, fontweight='bold')
    save_fig(fig, ADV, 'deg_stage_dotplot')
    print("  Saved deg_stage_dotplot.png/pdf")

    sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, cmap='viridis',
                                   save='_deg_stage.png', show=False)
    print("  Saved DEG heatmap (via scanpy)")

except Exception as e:
    print(f"  DEG analysis failed: {e}")

# =============================================================================
# Partition-based Graph Abstraction (PAGA)
# =============================================================================
print("\n[PAGA] Trajectory inference...")
sys.stdout.flush()

try:
    sc.tl.paga(adata, groups='stage')

    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.paga(adata, ax=ax, show=False, color='stage', cmap=None,
              node_size_scale=2, edge_width_scale=1.5, fontsize=12,
              frameon=False)
    ax.set_title('PAGA: Stage Connectivity', fontsize=14, fontweight='bold', pad=10)
    save_fig(fig, ADV, 'paga_stage')
    print("  Saved paga_stage.png/pdf")

    # PAGA on UMAP
    sc.tl.draw_graph(adata, init_pos='paga')
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.draw_graph(adata, color='stage', palette=STAGE_COLORS, ax=ax, show=False,
                    title='', frameon=False, size=3, alpha=0.6)
    ax.set_title('PAGA-initialized Force-directed Layout', fontsize=14, fontweight='bold', pad=10)
    save_fig(fig, ADV, 'paga_umap')
    print("  Saved paga_umap.png/pdf")

except Exception as e:
    print(f"  PAGA failed: {e}")

# =============================================================================
# Density plots
# =============================================================================
print("\n[DENSITY] Stage density on UMAP...")
sys.stdout.flush()

try:
    for stage in ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc.pl.embedding_density(adata, basis='umap', groupby='stage', group=stage,
                               ax=ax, show=False, cmap='magma')
        ax.set_title(f'{stage} Cell Density', fontsize=14, fontweight='bold', pad=10)
        save_fig(fig, ADV, f'density_{stage.lower()}')
        print(f"  Saved density_{stage.lower()}.png/pdf")
except Exception as e:
    print(f"  Density plots failed: {e}")

# =============================================================================
# GSEA - Gene Set Enrichment Analysis
# =============================================================================
print("\n[GSEA] Gene set enrichment analysis...")
sys.stdout.flush()

try:
    import gseapy as gp

    # Get DEG for GSEA
    if 'rank_genes_groups' in adata.uns:
        for stage in ['LUAD', 'MIA', 'AIS']:
            try:
                # Get ranked gene list
                genes_df = sc.get.rank_genes_groups_df(adata, group=stage)
                genes_df = genes_df.dropna(subset=['scores'])
                rnk = genes_df.set_index('names')['scores'].sort_values(ascending=False)

                # Run prerank GSEA
                pre_res = gp.prerank(
                    rnk=rnk,
                    gene_sets='KEGG_2021_Human',
                    processes=4,
                    permutation_num=100,
                    seed=42,
                    no_plot=True,
                )

                # Plot top enriched pathways
                res_df = pre_res.res2d
                res_df['abs_nes'] = res_df['NES'].abs()
                top_paths = res_df.nlargest(15, 'abs_nes')

                fig, ax = plt.subplots(figsize=(10, 8))
                colors = ['#8B1A1A' if x > 0 else '#4169E1' for x in top_paths['NES']]
                bars = ax.barh(range(len(top_paths)), top_paths['NES'], color=colors, alpha=0.85)
                ax.set_yticks(range(len(top_paths)))
                ax.set_yticklabels([p[:40] for p in top_paths['Term']], fontsize=9)
                ax.axvline(x=0, color='gray', linewidth=1)
                ax.set_xlabel('Normalized Enrichment Score', fontweight='bold')
                ax.set_title(f'GSEA: {stage} vs Rest (KEGG)', fontsize=14, fontweight='bold', pad=10)
                ax.invert_yaxis()

                for i, (nes, fdr) in enumerate(zip(top_paths['NES'], top_paths['FDR q-val'])):
                    ax.text(nes + 0.05 if nes > 0 else nes - 0.05, i,
                           f'FDR={fdr:.2e}', va='center', fontsize=7,
                           ha='left' if nes > 0 else 'right')

                save_fig(fig, ADV, f'gsea_{stage.lower()}_kegg')
                print(f"  Saved gsea_{stage.lower()}_kegg.png/pdf")

            except Exception as e:
                print(f"  GSEA {stage} failed: {e}")

except ImportError:
    print("  gseapy not installed, skipping GSEA")
except Exception as e:
    print(f"  GSEA failed: {e}")

# =============================================================================
# GSEA Enrichment Plot (classic running enrichment)
# =============================================================================
print("\n[GSEA] Enrichment plots...")
sys.stdout.flush()

try:
    import gseapy as gp

    if 'rank_genes_groups' in adata.uns:
        genes_df = sc.get.rank_genes_groups_df(adata, group='LUAD')
        rnk = genes_df.set_index('names')['scores'].sort_values(ascending=False)

        # Target specific pathways
        target_pathways = ['IL-17 signaling pathway', 'TNF signaling pathway',
                          'NF-kappa B signaling pathway', 'Cell cycle']

        pre_res = gp.prerank(
            rnk=rnk,
            gene_sets='KEGG_2021_Human',
            processes=4,
            permutation_num=100,
            seed=42,
            no_plot=True,
        )

        for pathway in target_pathways:
            try:
                if pathway in pre_res.res2d['Term'].values:
                    gp.gseaplot(
                        rank_metric=pre_res.ranking,
                        term=pathway,
                        ofname=str(ADV / f'gsea_enrichment_{pathway.replace(" ", "_").replace("-", "_").lower()}.png'),
                        **pre_res.results[pathway]
                    )
                    print(f"  Saved enrichment plot: {pathway}")
            except:
                pass

except ImportError:
    pass
except Exception as e:
    print(f"  GSEA enrichment plots failed: {e}")

# =============================================================================
# Ricci Curvature Network - Cell-Cell Communication
# =============================================================================
print("\n[RICCI] Ricci curvature network...")
sys.stdout.flush()

try:
    import networkx as nx
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    # Load LIANA communication matrix
    comm_path = CANONICAL / 'communication' / 'communication_matrix_luca.parquet'
    if comm_path.exists():
        comm = pd.read_parquet(comm_path)

        # Build graph from communication matrix
        G = nx.DiGraph()
        for source in comm.index:
            for target in comm.columns:
                weight = comm.loc[source, target]
                if weight > 0:
                    G.add_edge(source, target, weight=weight)

        if len(G.edges()) > 0:
            # Compute Ollivier-Ricci curvature
            orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
            orc.compute_ricci_curvature()
            G_ricci = orc.G

            # Extract curvature values
            edge_curvatures = [G_ricci[u][v]['ricciCurvature'] for u, v in G_ricci.edges()]

            # Plot network with curvature coloring
            fig, ax = plt.subplots(figsize=(14, 12))

            pos = nx.spring_layout(G_ricci, k=2, iterations=50, seed=42)

            # Node sizes by degree
            node_sizes = [300 + G_ricci.degree(n) * 50 for n in G_ricci.nodes()]

            # Edge colors by curvature
            edge_colors = [G_ricci[u][v]['ricciCurvature'] for u, v in G_ricci.edges()]

            # Draw network
            nx.draw_networkx_nodes(G_ricci, pos, ax=ax, node_size=node_sizes,
                                  node_color='#4169E1', alpha=0.8, edgecolors='white', linewidths=2)
            edges = nx.draw_networkx_edges(G_ricci, pos, ax=ax, edge_color=edge_colors,
                                          edge_cmap=plt.cm.RdYlBu_r, width=1.5, alpha=0.7,
                                          arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
            nx.draw_networkx_labels(G_ricci, pos, ax=ax, font_size=7, font_weight='bold')

            # Colorbar for curvature
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r,
                                       norm=plt.Normalize(min(edge_curvatures), max(edge_curvatures)))
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
            cbar.set_label('Ollivier-Ricci Curvature', fontweight='bold')

            ax.set_title('Cell-Cell Communication Network\n(Edge color = Ricci Curvature)',
                        fontsize=14, fontweight='bold', pad=15)
            ax.axis('off')

            save_fig(fig, ADV, 'ricci_communication_network')
            print("  Saved ricci_communication_network.png/pdf")

            # Curvature distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(edge_curvatures, bins=40, color='#6B3FA0', edgecolor='white', alpha=0.8)
            ax.axvline(x=0, color='#8B1A1A', linestyle='--', linewidth=2, label='Zero curvature')
            ax.axvline(x=np.mean(edge_curvatures), color='#DAA520', linestyle='-', linewidth=2,
                      label=f'Mean: {np.mean(edge_curvatures):.3f}')
            ax.set_xlabel('Ollivier-Ricci Curvature', fontweight='bold')
            ax.set_ylabel('Edge Count', fontweight='bold')
            ax.set_title('Ricci Curvature Distribution', fontsize=14, fontweight='bold', pad=10)
            ax.legend(frameon=False)
            save_fig(fig, ADV, 'ricci_curvature_distribution')
            print("  Saved ricci_curvature_distribution.png/pdf")

except ImportError:
    print("  GraphRicciCurvature not installed, skipping Ricci analysis")
except Exception as e:
    print(f"  Ricci curvature failed: {e}")

# =============================================================================
# Network Centrality Analysis
# =============================================================================
print("\n[NETWORK] Centrality analysis...")
sys.stdout.flush()

try:
    import networkx as nx

    comm_path = CANONICAL / 'communication' / 'communication_matrix_luca.parquet'
    if comm_path.exists():
        comm = pd.read_parquet(comm_path)

        G = nx.DiGraph()
        for source in comm.index:
            for target in comm.columns:
                weight = comm.loc[source, target]
                if weight > 10:  # Filter weak connections
                    G.add_edge(source, target, weight=weight)

        if len(G.nodes()) > 0:
            # Compute centralities
            in_deg = dict(G.in_degree(weight='weight'))
            out_deg = dict(G.out_degree(weight='weight'))
            pagerank = nx.pagerank(G, weight='weight')
            betweenness = nx.betweenness_centrality(G, weight='weight')

            centrality_df = pd.DataFrame({
                'In-degree': in_deg,
                'Out-degree': out_deg,
                'PageRank': pagerank,
                'Betweenness': betweenness,
            })

            # Plot centrality comparison
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            metrics = ['In-degree', 'Out-degree', 'PageRank', 'Betweenness']
            cmaps = ['rocket', 'mako', 'viridis', 'plasma']

            for ax, metric, cmap in zip(axes.flatten(), metrics, cmaps):
                data = centrality_df[metric].nlargest(15)
                colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.9, len(data)))
                ax.barh(range(len(data)), data.values, color=colors)
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data.index, fontsize=9)
                ax.set_title(f'{metric} Centrality', fontweight='bold')
                ax.invert_yaxis()

            fig.suptitle('Network Centrality Analysis', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            save_fig(fig, ADV, 'network_centrality')
            print("  Saved network_centrality.png/pdf")

except ImportError:
    print("  networkx not available")
except Exception as e:
    print(f"  Centrality analysis failed: {e}")

# =============================================================================
# Volcano Plots - DEG
# =============================================================================
print("\n[VOLCANO] Differential expression volcano plots...")
sys.stdout.flush()

try:
    if 'rank_genes_groups' in adata.uns:
        for stage in ['LUAD', 'MIA']:
            try:
                df = sc.get.rank_genes_groups_df(adata, group=stage)

                fig, ax = plt.subplots(figsize=(10, 8))

                # Define significance
                df['significance'] = 'NS'
                df.loc[(df['pvals_adj'] < 0.05) & (df['logfoldchanges'] > 0.5), 'significance'] = 'Up'
                df.loc[(df['pvals_adj'] < 0.05) & (df['logfoldchanges'] < -0.5), 'significance'] = 'Down'

                colors = {'NS': '#CCCCCC', 'Up': '#8B1A1A', 'Down': '#4169E1'}

                for sig, color in colors.items():
                    mask = df['significance'] == sig
                    ax.scatter(df.loc[mask, 'logfoldchanges'],
                              -np.log10(df.loc[mask, 'pvals_adj'] + 1e-300),
                              c=color, s=10, alpha=0.6, label=sig)

                # Label top genes
                top_up = df[df['significance'] == 'Up'].nlargest(10, 'logfoldchanges')
                top_down = df[df['significance'] == 'Down'].nsmallest(10, 'logfoldchanges')

                for _, row in pd.concat([top_up, top_down]).iterrows():
                    ax.annotate(row['names'], (row['logfoldchanges'], -np.log10(row['pvals_adj'] + 1e-300)),
                               fontsize=7, alpha=0.8)

                ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
                ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)

                ax.set_xlabel('Log2 Fold Change', fontweight='bold')
                ax.set_ylabel('-Log10(Adjusted P-value)', fontweight='bold')
                ax.set_title(f'Volcano Plot: {stage} vs Rest', fontsize=14, fontweight='bold', pad=10)
                ax.legend(frameon=False)

                save_fig(fig, ADV, f'volcano_{stage.lower()}')
                print(f"  Saved volcano_{stage.lower()}.png/pdf")

            except Exception as e:
                print(f"  Volcano {stage} failed: {e}")

except Exception as e:
    print(f"  Volcano plots failed: {e}")

# =============================================================================
# MA Plots
# =============================================================================
print("\n[MA] MA plots...")
sys.stdout.flush()

try:
    if 'rank_genes_groups' in adata.uns:
        for stage in ['LUAD']:
            df = sc.get.rank_genes_groups_df(adata, group=stage)

            # Compute mean expression (pseudo-bulk)
            stage_mask = adata.obs['stage'] == stage
            mean_expr = np.array(adata[stage_mask].X.mean(axis=0)).flatten()
            df['mean_expr'] = mean_expr[adata.var_names.get_indexer(df['names'])]

            fig, ax = plt.subplots(figsize=(10, 8))

            sig_mask = df['pvals_adj'] < 0.05
            ax.scatter(df.loc[~sig_mask, 'mean_expr'], df.loc[~sig_mask, 'logfoldchanges'],
                      c='#CCCCCC', s=5, alpha=0.5, label='NS')
            ax.scatter(df.loc[sig_mask, 'mean_expr'], df.loc[sig_mask, 'logfoldchanges'],
                      c='#8B1A1A', s=10, alpha=0.6, label='Significant')

            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('Mean Expression', fontweight='bold')
            ax.set_ylabel('Log2 Fold Change', fontweight='bold')
            ax.set_title(f'MA Plot: {stage}', fontsize=14, fontweight='bold', pad=10)
            ax.legend(frameon=False)

            save_fig(fig, ADV, f'ma_plot_{stage.lower()}')
            print(f"  Saved ma_plot_{stage.lower()}.png/pdf")

except Exception as e:
    print(f"  MA plots failed: {e}")

# =============================================================================
# Chord Diagram - Communication
# =============================================================================
print("\n[CHORD] Communication chord diagram...")
sys.stdout.flush()

try:
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patches as mpatches

    comm_path = CANONICAL / 'communication' / 'communication_matrix_luca.parquet'
    if comm_path.exists():
        comm = pd.read_parquet(comm_path)

        # Get top communicating pairs
        top_ct = comm.sum(axis=1).nlargest(10).index.tolist()
        comm_top = comm.loc[top_ct, top_ct]

        # Circular layout
        n = len(top_ct)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})

        # Draw arcs for connections
        for i, source in enumerate(top_ct):
            for j, target in enumerate(top_ct):
                if i != j:
                    weight = comm_top.loc[source, target]
                    if weight > comm_top.values.mean():
                        alpha = min(weight / comm_top.values.max(), 0.8)
                        width = 1 + 3 * (weight / comm_top.values.max())
                        ax.plot([angles[i], angles[j]], [1, 1], alpha=alpha,
                               linewidth=width, color=plt.cm.plasma(j/n))

        # Draw nodes
        ax.scatter(angles, [1]*n, s=500, c=[plt.cm.viridis(i/n) for i in range(n)],
                  zorder=10, edgecolors='white', linewidths=2)

        # Labels
        for angle, ct in zip(angles, top_ct):
            rotation = np.degrees(angle)
            if angle > np.pi/2 and angle < 3*np.pi/2:
                rotation += 180
                ha = 'right'
            else:
                ha = 'left'
            ax.text(angle, 1.15, ct[:15], rotation=rotation, ha=ha, va='center',
                   fontsize=9, fontweight='bold', rotation_mode='anchor')

        ax.set_ylim(0, 1.3)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.set_title('Cell-Cell Communication Network', fontsize=14, fontweight='bold', y=1.05)

        save_fig(fig, ADV, 'chord_communication')
        print("  Saved chord_communication.png/pdf")

except Exception as e:
    print(f"  Chord diagram failed: {e}")

print("\n[ADVANCED SC] Complete!")
print("=" * 60)
PYTHON_ADVANCED_SC

# =============================================================================
# PART 6B: Generate Publication Figures
# =============================================================================
echo ""
echo "=============================================="
echo "PART 6B: Publication Figures"
echo "=============================================="

python << 'PYTHON_FIGURES'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')

CANONICAL = Path(os.environ['CANONICAL'])
FIGURES = Path("/home/booka/StageBridge/figures/publication")
FIGURES.mkdir(exist_ok=True, parents=True)

print("Generating publication figures...")
sys.stdout.flush()

# =============================================================================
# PUBLICATION STYLE SETUP
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

# Deep, rich color palette
STAGE_COLORS = {
    'Normal': '#228B22',   # Forest green
    'AAH': '#4169E1',      # Royal blue
    'AIS': '#6B3FA0',      # Deep violet
    'MIA': '#DAA520',      # Goldenrod
    'LUAD': '#8B1A1A',     # Brick red / dark crimson
}

# Additional deep hue palettes for various plots
DEEP_PALETTE = {
    'brick_red': '#8B1A1A',
    'royal_blue': '#4169E1',
    'navy': '#1B2F5E',
    'gold': '#DAA520',
    'emerald': '#046307',
    'olive': '#556B2F',
    'violet': '#6B3FA0',
    'teal': '#115E59',
    'burgundy': '#722F37',
    'slate': '#4A5568',
}

# For sequential data
DEEP_SEQUENTIAL = ['#F7F7F7', '#FFF3CD', '#DAA520', '#C65D07', '#8B1A1A']

# For diverging data
DEEP_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'deep_diverging', ['#1B2F5E', '#4169E1', '#F7F7F7', '#C65D07', '#8B1A1A']
)

# For categorical data (10 deep hues)
DEEP_CATEGORICAL = [
    '#8B1A1A',  # Brick red
    '#4169E1',  # Royal blue
    '#046307',  # Emerald
    '#DAA520',  # Gold
    '#6B3FA0',  # Violet
    '#115E59',  # Teal
    '#556B2F',  # Olive
    '#1B2F5E',  # Navy
    '#722F37',  # Burgundy
    '#4A5568',  # Slate
]
STAGE_PALETTE = [STAGE_COLORS[s] for s in STAGE_ORDER]

def save_figure(fig, path, name, formats=['png', 'pdf']):
    for fmt in formats:
        fig.savefig(path / f'{name}.{fmt}', format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

# =============================================================================
# LIANA Figures (using stagebridge.viz.liana)
# =============================================================================
print("\n[LIANA] Advanced cell-cell communication figures...")
sys.stdout.flush()

try:
    from stagebridge.viz import liana as liana_viz

    liana_path = CANONICAL / 'liana_luca' / 'liana_interactions_luca.parquet'
    if liana_path.exists():
        df = liana_viz.load_liana_data(liana_path)
        print(f"  Loaded {len(df):,} interactions")
        liana_viz.generate_all_figures(df, FIGURES / 'liana', show=False)
        print("  Generated LIANA figure suite")
    else:
        print("  SKIP: LIANA data not found")
except Exception as e:
    print(f"  LIANA figures failed: {e}")

# =============================================================================
# FIGURE 1: Signature Ridge Plot - Elegant density distributions
# =============================================================================
print("\n[SIGNATURES] Ridge plot of EMT/Senescence/SASP distributions...")
sys.stdout.flush()

try:
    sig = pd.read_parquet(CANONICAL / 'signatures' / 'gene_signatures.parquet')

    if 'stage' in sig.columns:
        sig_melt = sig.melt(
            id_vars=['stage'],
            value_vars=['emt_score', 'senescence_score', 'sasp_score'],
            var_name='Signature', value_name='Score'
        )
        sig_melt['Signature'] = sig_melt['Signature'].map({
            'emt_score': 'EMT',
            'senescence_score': 'Senescence',
            'sasp_score': 'SASP'
        })
        sig_melt = sig_melt[sig_melt['stage'].isin(STAGE_ORDER)]

        for signature in ['EMT', 'Senescence', 'SASP']:
            fig, axes = plt.subplots(len(STAGE_ORDER), 1, figsize=(10, 8), sharex=True)
            fig.subplots_adjust(hspace=-0.5)

            subset = sig_melt[sig_melt['Signature'] == signature]

            for idx, (stage, ax) in enumerate(zip(STAGE_ORDER, axes)):
                stage_data = subset[subset['stage'] == stage]['Score'].dropna()
                if len(stage_data) > 10:
                    color = STAGE_COLORS[stage]
                    sns.kdeplot(data=stage_data, ax=ax, fill=True, alpha=0.7,
                               linewidth=2, color=color, bw_adjust=0.8)
                    ax.set_ylabel('')
                    ax.set_yticks([])
                    ax.text(-0.02, 0.5, stage, transform=ax.transAxes, fontsize=12,
                           fontweight='bold', va='center', ha='right', color=color)
                    ax.axhline(y=0, color='white', linewidth=1.5)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if idx < len(STAGE_ORDER) - 1:
                        ax.set_xlabel('')
                        ax.tick_params(labelbottom=False)

            axes[-1].set_xlabel(f'{signature} Score', fontsize=12, fontweight='bold')
            fig.suptitle(f'{signature} Score Distribution by Stage', fontsize=14,
                        fontweight='bold', y=0.98)
            save_figure(fig, FIGURES, f'{signature.lower()}_ridgeplot')
            plt.close()
            print(f"  Saved {signature.lower()}_ridgeplot.png/pdf")

except Exception as e:
    print(f"  Ridge plots failed: {e}")

# =============================================================================
# FIGURE 2: Combined Signature Heatmap with Dendrograms
# =============================================================================
print("\n[SIGNATURES] Clustered heatmap by cell type...")
sys.stdout.flush()

try:
    sig_ct = pd.read_parquet(CANONICAL / 'signatures' / 'signatures_by_luca_celltype.parquet')

    if len(sig_ct) > 3:
        sig_ct.columns = ['EMT', 'Senescence', 'SASP']

        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist

        sig_norm = (sig_ct - sig_ct.mean()) / sig_ct.std()

        g = sns.clustermap(
            sig_norm,
            cmap='RdBu_r',
            center=0,
            figsize=(10, 12),
            dendrogram_ratio=(0.15, 0.15),
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            linewidths=0.5,
            linecolor='white',
            yticklabels=True,
            xticklabels=True,
            row_cluster=True,
            col_cluster=False,
            annot=True,
            fmt='.1f',
            annot_kws={'size': 8},
        )
        g.ax_heatmap.set_xlabel('')
        g.ax_heatmap.set_ylabel('')
        g.ax_heatmap.tick_params(axis='y', labelsize=9, rotation=0)
        g.ax_heatmap.tick_params(axis='x', labelsize=11, rotation=0)
        g.fig.suptitle('Cell Type Signature Profiles (Z-scored)', fontsize=14,
                      fontweight='bold', y=1.02)
        g.cax.set_ylabel('Z-score', fontsize=10)

        save_figure(g.fig, FIGURES, 'signatures_clustermap')
        plt.close()
        print("  Saved signatures_clustermap.png/pdf")

except Exception as e:
    print(f"  Clustermap failed: {e}")

# =============================================================================
# FIGURE 3: Pathway Activity Radar/Spider Chart
# =============================================================================
print("\n[ACTIVITY] Pathway activity radar chart...")
sys.stdout.flush()

try:
    pw = pd.read_parquet(CANONICAL / 'activity' / 'pathway_activity_by_stage.parquet')
    pw = pw.reindex(STAGE_ORDER)

    categories = list(pw.columns)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontweight='bold')

    pw_norm = (pw - pw.min().min()) / (pw.max().max() - pw.min().min())

    for idx, stage in enumerate(STAGE_ORDER):
        values = pw_norm.loc[stage].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=stage,
               color=STAGE_COLORS[stage], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=STAGE_COLORS[stage])

    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
             fancybox=True, shadow=True, fontsize=10)
    ax.set_title('Pathway Activity by Disease Stage', fontsize=14,
                fontweight='bold', pad=20)

    save_figure(fig, FIGURES, 'pathway_radar')
    plt.close()
    print("  Saved pathway_radar.png/pdf")

except Exception as e:
    print(f"  Radar chart failed: {e}")

# =============================================================================
# FIGURE 4: TF Activity Dot Plot (like scanpy dotplot)
# =============================================================================
print("\n[ACTIVITY] TF activity dot plot...")
sys.stdout.flush()

try:
    tf = pd.read_parquet(CANONICAL / 'activity' / 'tf_activity_by_stage.parquet')
    tf = tf.reindex(STAGE_ORDER)
    tf_var = tf.var().nlargest(20).index.tolist()
    tf_top = tf[tf_var]

    tf_long = tf_top.reset_index().melt(id_vars='index', var_name='TF', value_name='Activity')
    tf_long.columns = ['Stage', 'TF', 'Activity']

    tf_scaled = tf_top.copy()
    for col in tf_scaled.columns:
        vmin, vmax = tf_scaled[col].min(), tf_scaled[col].max()
        if vmax > vmin:
            tf_scaled[col] = (tf_scaled[col] - vmin) / (vmax - vmin)

    tf_long['Size'] = tf_scaled.reset_index().melt(id_vars='index')['value'].values

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, stage in enumerate(STAGE_ORDER):
        stage_data = tf_long[tf_long['Stage'] == stage]
        for j, tf_name in enumerate(tf_var):
            row = stage_data[stage_data['TF'] == tf_name]
            if len(row) > 0:
                activity = row['Activity'].values[0]
                size = row['Size'].values[0]
                color = plt.cm.RdBu_r((activity - tf_long['Activity'].min()) /
                                      (tf_long['Activity'].max() - tf_long['Activity'].min()))
                ax.scatter(j, i, s=size * 400 + 50, c=[color], edgecolors='white',
                          linewidths=1.5, zorder=3)

    ax.set_xticks(range(len(tf_var)))
    ax.set_xticklabels(tf_var, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER, fontsize=11, fontweight='bold')

    for i, stage in enumerate(STAGE_ORDER):
        ax.axhline(i, color='#EEEEEE', linewidth=0.8, zorder=1)
    for j in range(len(tf_var)):
        ax.axvline(j, color='#EEEEEE', linewidth=0.8, zorder=1)

    ax.set_xlim(-0.5, len(tf_var) - 0.5)
    ax.set_ylim(-0.5, len(STAGE_ORDER) - 0.5)
    ax.set_xlabel('Transcription Factor', fontsize=12, fontweight='bold')
    ax.set_title('TF Activity by Stage (size = relative magnitude)', fontsize=14,
                fontweight='bold', pad=15)

    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(tf_long['Activity'].min(), tf_long['Activity'].max()))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Activity Score', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, FIGURES, 'tf_dotplot')
    plt.close()
    print("  Saved tf_dotplot.png/pdf")

except Exception as e:
    print(f"  Dot plot failed: {e}")

# =============================================================================
# FIGURE 5: Cell Type Lollipop Chart
# =============================================================================
print("\n[CELLTYPE] Lollipop chart...")
sys.stdout.flush()

try:
    ct_map = pd.read_parquet(CANONICAL / 'liana_luca' / 'luca_cell_type_mapping.parquet')
    ct_counts = ct_map['luca_cell_type'].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = range(len(ct_counts))
    colors = [DEEP_CATEGORICAL[i % len(DEEP_CATEGORICAL)] for i in range(len(ct_counts))]

    ax.hlines(y=y_pos, xmin=0, xmax=ct_counts.values, color=colors, linewidth=3, alpha=0.8)
    ax.scatter(ct_counts.values, y_pos, s=200, c=colors, zorder=3, edgecolors='white', linewidths=2)

    for i, (ct, val) in enumerate(ct_counts.items()):
        ax.text(val + ct_counts.max() * 0.02, i, f'{val:,}', va='center', fontsize=10,
               fontweight='bold', color='#333333')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ct_counts.index, fontsize=10)
    ax.set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Composition (LuCA Annotations)', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.set_xlim(0, ct_counts.max() * 1.18)

    ax.axvline(x=ct_counts.median(), color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(ct_counts.median(), -0.8, f'Median: {ct_counts.median():,.0f}',
           ha='center', fontsize=10, color='#FF6B6B', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, FIGURES, 'celltype_lollipop')
    plt.close()
    print("  Saved celltype_lollipop.png/pdf")

except Exception as e:
    print(f"  Lollipop chart failed: {e}")

# =============================================================================
# FIGURE 6: IL1B Pathway Bump Chart (rank changes across stages)
# =============================================================================
print("\n[GENES] IL1B pathway bump chart...")
sys.stdout.flush()

try:
    genes = pd.read_parquet(CANONICAL / 'key_genes' / 'key_genes_by_stage.parquet')

    il1_genes = ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN']
    available = [g for g in il1_genes if f'{g}_mean' in genes.columns]

    if available:
        gene_colors = {
            'IL1B': '#8B1A1A',   # Brick red
            'IL1R1': '#4169E1',  # Royal blue
            'IL1R2': '#046307',  # Emerald
            'IL1RAP': '#6B3FA0', # Violet
            'IL1RN': '#DAA520',  # Gold
        }

        fig, ax = plt.subplots(figsize=(12, 7))

        for gene in available:
            col = f'{gene}_mean'
            if col in genes.columns:
                vals = genes[col].reindex(STAGE_ORDER)
                color = gene_colors.get(gene, '#666666')

                ax.plot(range(len(STAGE_ORDER)), vals, marker='o', markersize=14,
                       linewidth=4, color=color, label=gene, markeredgecolor='white',
                       markeredgewidth=2.5, solid_capstyle='round', zorder=3)

                for i, v in enumerate(vals):
                    ax.annotate(f'{v:.2f}', (i, v), textcoords="offset points",
                               xytext=(0, 12), ha='center', fontsize=9,
                               fontweight='bold', color=color)

        ax.fill_between(range(len(STAGE_ORDER)),
                       ax.get_ylim()[0], ax.get_ylim()[1],
                       where=[True if s in ['AAH', 'AIS'] else False for s in STAGE_ORDER],
                       alpha=0.1, color='#FFD700', zorder=1)
        ax.text(1.5, ax.get_ylim()[1] * 0.95, 'Therapeutic Window',
               ha='center', fontsize=10, fontstyle='italic', color='#B8860B', fontweight='bold')

        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, fontsize=12, fontweight='bold')
        for i, stage in enumerate(STAGE_ORDER):
            ax.get_xticklabels()[i].set_color(STAGE_COLORS[stage])

        ax.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Expression', fontsize=12, fontweight='bold')
        ax.set_title('IL1B Signaling Pathway Expression', fontsize=14, fontweight='bold', pad=15)

        ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)

        legend = ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True,
                          shadow=True, borderpad=1)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#CCCCCC')

        plt.tight_layout()
        save_figure(fig, FIGURES, 'il1b_pathway_expression')
        plt.close()
        print("  Saved il1b_pathway_expression.png/pdf")

except Exception as e:
    print(f"  IL1B chart failed: {e}")

# =============================================================================
# FIGURE 7: Stage x Cell Type Heatmap with Annotations
# =============================================================================
print("\n[CROSSTAB] Stage x Cell Type signature heatmap...")
sys.stdout.flush()

try:
    crosstab_path = CANONICAL / 'signatures' / 'signatures_by_stage_celltype.parquet'
    if crosstab_path.exists():
        ct_stage = pd.read_parquet(crosstab_path)

        emt_pivot = ct_stage['emt_score'].unstack(level=0)
        emt_pivot = emt_pivot.reindex(columns=STAGE_ORDER)

        top_celltypes = emt_pivot.var(axis=1).nlargest(15).index
        emt_top = emt_pivot.loc[top_celltypes]

        fig, ax = plt.subplots(figsize=(10, 10))

        sns.heatmap(emt_top, cmap='RdYlBu_r', center=0, ax=ax,
                   linewidths=0.5, linecolor='white',
                   cbar_kws={'label': 'EMT Score', 'shrink': 0.6},
                   annot=True, fmt='.2f', annot_kws={'size': 8})

        ax.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_title('EMT Score by Stage and Cell Type', fontsize=14, fontweight='bold', pad=15)

        for i, label in enumerate(ax.get_xticklabels()):
            label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))
            label.set_fontweight('bold')

        plt.tight_layout()
        save_figure(fig, FIGURES, 'emt_stage_celltype_heatmap')
        plt.close()
        print("  Saved emt_stage_celltype_heatmap.png/pdf")

except Exception as e:
    print(f"  Stage x Cell Type heatmap failed: {e}")

# =============================================================================
# FIGURE 8: IL1B Cell Type Expression (Key for paper)
# =============================================================================
print("\n[IL1B] IL1B pathway by cell type...")
sys.stdout.flush()

try:
    il1_ct_path = CANONICAL / 'key_genes' / 'il1b_pathway_by_luca_celltype.parquet'
    if il1_ct_path.exists():
        il1_ct = pd.read_parquet(il1_ct_path)

        il1_ct_sorted = il1_ct.loc[il1_ct.sum(axis=1).nlargest(20).index]

        fig, ax = plt.subplots(figsize=(12, 10))

        custom_cmap = LinearSegmentedColormap.from_list('il1b',
            ['#F7F7F7', '#FFF3CD', '#DAA520', '#C65D07', '#8B1A1A'])

        sns.heatmap(il1_ct_sorted, cmap=custom_cmap, ax=ax,
                   linewidths=0.8, linecolor='white',
                   cbar_kws={'label': 'Mean Expression', 'shrink': 0.6},
                   annot=True, fmt='.2f', annot_kws={'size': 9, 'fontweight': 'bold'})

        ax.set_xlabel('IL1B Pathway Genes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_title('IL1B Pathway Expression by Cell Type', fontsize=14, fontweight='bold', pad=15)

        ax.axvline(x=1, color='#333333', linewidth=2, linestyle='-')
        ax.text(0.5, -0.8, 'Ligand', ha='center', fontsize=10, fontweight='bold', color='#8B1A1A')
        ax.text(2.5, -0.8, 'Receptors', ha='center', fontsize=10, fontweight='bold', color='#4169E1')

        plt.tight_layout()
        save_figure(fig, FIGURES, 'il1b_celltype_heatmap')
        plt.close()
        print("  Saved il1b_celltype_heatmap.png/pdf")

except Exception as e:
    print(f"  IL1B cell type heatmap failed: {e}")

# =============================================================================
# FIGURE 9: Communication Network Summary (Sankey-like)
# =============================================================================
print("\n[COMM] Communication flow visualization...")
sys.stdout.flush()

try:
    comm_path = CANONICAL / 'communication' / 'communication_matrix_luca.parquet'
    if comm_path.exists():
        comm = pd.read_parquet(comm_path)

        total_out = comm.sum(axis=1).nlargest(10)
        total_in = comm.sum(axis=0).nlargest(10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        ax1 = axes[0]
        colors = [f'#{int(139 + (200-139)*i/len(total_out)):02x}{int(26 + (80-26)*i/len(total_out)):02x}{int(26 + (80-26)*i/len(total_out)):02x}' for i in range(len(total_out))]  # Brick red gradient
        bars = ax1.barh(range(len(total_out)), total_out.values, color=colors,
                       edgecolor='white', linewidth=1.5)
        ax1.set_yticks(range(len(total_out)))
        ax1.set_yticklabels(total_out.index, fontsize=10)
        ax1.set_xlabel('Outgoing Interactions', fontsize=11, fontweight='bold')
        ax1.set_title('Top Signal Senders', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        for bar, val in zip(bars, total_out.values):
            ax1.text(val + total_out.max() * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9, fontweight='bold')

        ax2 = axes[1]
        colors = [f'#{int(27 + (100-27)*i/len(total_in)):02x}{int(47 + (120-47)*i/len(total_in)):02x}{int(94 + (180-94)*i/len(total_in)):02x}' for i in range(len(total_in))]  # Navy gradient
        bars = ax2.barh(range(len(total_in)), total_in.values, color=colors,
                       edgecolor='white', linewidth=1.5)
        ax2.set_yticks(range(len(total_in)))
        ax2.set_yticklabels(total_in.index, fontsize=10)
        ax2.set_xlabel('Incoming Interactions', fontsize=11, fontweight='bold')
        ax2.set_title('Top Signal Receivers', fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        for bar, val in zip(bars, total_in.values):
            ax2.text(val + total_in.max() * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=9, fontweight='bold')

        fig.suptitle('Cell-Cell Communication Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, FIGURES, 'communication_summary')
        plt.close()
        print("  Saved communication_summary.png/pdf")

except Exception as e:
    print(f"  Communication summary failed: {e}")

# Multi-panel figures removed per user preference - individual advanced figures only

# =============================================================================
# EXPLORATORY DATA ANALYSIS - Full Biological Landscape
# =============================================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Create EDA subdirectory
EDA = FIGURES / 'eda'
EDA.mkdir(exist_ok=True)

# =============================================================================
# EDA 1: All Pathway Scores Heatmap (comprehensive view)
# =============================================================================
print("\n[EDA] Pathway scores landscape...")
sys.stdout.flush()

try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_stage.parquet')
    pw_scores = pw_scores.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(16, 10))

    pw_z = (pw_scores - pw_scores.mean()) / pw_scores.std()

    sns.heatmap(pw_z.T, cmap='RdBu_r', center=0, ax=ax,
               linewidths=0.5, linecolor='white',
               cbar_kws={'label': 'Z-score', 'shrink': 0.6},
               annot=True, fmt='.2f', annot_kws={'size': 8})

    ax.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biological Pathway', fontsize=12, fontweight='bold')
    ax.set_title('Pathway Activity Landscape Across LUAD Progression', fontsize=14, fontweight='bold', pad=15)

    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))
        label.set_fontweight('bold')

    plt.tight_layout()
    save_figure(fig, EDA, 'pathway_landscape_heatmap')
    plt.close()
    print("  Saved pathway_landscape_heatmap.png/pdf")

except Exception as e:
    print(f"  Pathway landscape failed: {e}")

# =============================================================================
# EDA 2: CAF Subtype Analysis (iCAF vs myCAF)
# =============================================================================
print("\n[EDA] CAF subtype analysis...")
sys.stdout.flush()

try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_luca_celltype.parquet')

    if 'caf_icaf' in pw_scores.columns and 'caf_mycaf' in pw_scores.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        caf_data = pw_scores[['caf_icaf', 'caf_mycaf', 'caf_markers']].dropna()

        ax1 = axes[0]
        caf_sorted = caf_data.sort_values('caf_markers', ascending=True).tail(20)
        y_pos = range(len(caf_sorted))
        ax1.barh(y_pos, caf_sorted['caf_icaf'], height=0.4, label='iCAF', color='#4169E1', alpha=0.9)
        ax1.barh([y + 0.4 for y in y_pos], caf_sorted['caf_mycaf'], height=0.4, label='myCAF', color='#8B1A1A', alpha=0.9)
        ax1.set_yticks([y + 0.2 for y in y_pos])
        ax1.set_yticklabels(caf_sorted.index, fontsize=9)
        ax1.set_xlabel('Pathway Score', fontweight='bold')
        ax1.set_title('CAF Subtypes by Cell Type', fontweight='bold')
        ax1.legend(loc='lower right')

        ax2 = axes[1]
        ax2.scatter(caf_data['caf_icaf'], caf_data['caf_mycaf'], s=100, alpha=0.7,
                   c=caf_data['caf_markers'], cmap='inferno', edgecolors='white', linewidth=1.5)
        for ct in caf_data.nlargest(10, 'caf_markers').index:
            ax2.annotate(ct[:15], (caf_data.loc[ct, 'caf_icaf'], caf_data.loc[ct, 'caf_mycaf']),
                        fontsize=8, alpha=0.8)
        ax2.set_xlabel('iCAF Score', fontweight='bold')
        ax2.set_ylabel('myCAF Score', fontweight='bold')
        ax2.set_title('iCAF vs myCAF Polarization', fontweight='bold')
        ax2.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5)

        sm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(caf_data['caf_markers'].min(),
                                                                       caf_data['caf_markers'].max()))
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.6)
        cbar.set_label('CAF Marker Score', fontweight='bold')

        fig.suptitle('Cancer-Associated Fibroblast Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, EDA, 'caf_subtype_analysis')
        plt.close()
        print("  Saved caf_subtype_analysis.png/pdf")

except Exception as e:
    print(f"  CAF analysis failed: {e}")

# =============================================================================
# EDA 3: Senescence and SASP Deep Dive
# =============================================================================
print("\n[EDA] Senescence/SASP analysis...")
sys.stdout.flush()

try:
    pw_scores_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_stage.parquet')
    pw_scores_ct = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_luca_celltype.parquet')

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if 'senescence' in pw_scores_stage.columns:
        sen_stage = pw_scores_stage['senescence'].reindex(STAGE_ORDER)
        bars = ax1.bar(STAGE_ORDER, sen_stage, color=STAGE_PALETTE, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Senescence Score', fontweight='bold')
        ax1.set_title('Senescence by Stage', fontweight='bold')
        for bar, val in zip(bars, sen_stage):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                    ha='center', fontsize=9, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    if 'sasp' in pw_scores_stage.columns:
        sasp_stage = pw_scores_stage['sasp'].reindex(STAGE_ORDER)
        bars = ax2.bar(STAGE_ORDER, sasp_stage, color=STAGE_PALETTE, edgecolor='white', linewidth=2)
        ax2.set_ylabel('SASP Score', fontweight='bold')
        ax2.set_title('SASP by Stage', fontweight='bold')
        for bar, val in zip(bars, sasp_stage):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                    ha='center', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    if 'senescence' in pw_scores_ct.columns:
        sen_ct = pw_scores_ct['senescence'].nlargest(15)
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(sen_ct)))
        ax3.barh(range(len(sen_ct)), sen_ct.values, color=colors)
        ax3.set_yticks(range(len(sen_ct)))
        ax3.set_yticklabels(sen_ct.index, fontsize=9)
        ax3.set_xlabel('Senescence Score', fontweight='bold')
        ax3.set_title('Top Senescent Cell Types', fontweight='bold')
        ax3.invert_yaxis()

    ax4 = fig.add_subplot(gs[1, 1])
    if 'sasp' in pw_scores_ct.columns and 'senescence' in pw_scores_ct.columns:
        ax4.scatter(pw_scores_ct['senescence'], pw_scores_ct['sasp'], s=100, alpha=0.7,
                   c='#8B1A1A', edgecolors='white', linewidth=1.5)
        for ct in pw_scores_ct.nlargest(8, 'sasp').index:
            ax4.annotate(ct[:20], (pw_scores_ct.loc[ct, 'senescence'], pw_scores_ct.loc[ct, 'sasp']),
                        fontsize=8, alpha=0.8)
        ax4.set_xlabel('Senescence Score', fontweight='bold')
        ax4.set_ylabel('SASP Score', fontweight='bold')
        ax4.set_title('Senescence vs SASP Correlation', fontweight='bold')

        from scipy.stats import pearsonr
        r, p = pearsonr(pw_scores_ct['senescence'].dropna(), pw_scores_ct['sasp'].dropna())
        ax4.text(0.05, 0.95, f'r = {r:.2f}\np = {p:.2e}', transform=ax4.transAxes,
                fontsize=10, fontweight='bold', va='top')

    fig.suptitle('Cellular Senescence and SASP Analysis', fontsize=14, fontweight='bold', y=0.98)
    save_figure(fig, EDA, 'senescence_sasp_analysis')
    plt.close()
    print("  Saved senescence_sasp_analysis.png/pdf")

except Exception as e:
    print(f"  Senescence analysis failed: {e}")

# =============================================================================
# EDA 4: AP-1 Complex and Targets
# =============================================================================
print("\n[EDA] AP-1 signaling analysis...")
sys.stdout.flush()

try:
    ap1_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'ap1_complex_by_stage.parquet')
    ap1_stage = ap1_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ap1_z = (ap1_stage - ap1_stage.mean()) / ap1_stage.std()
    sns.heatmap(ap1_z.T, cmap='RdYlBu_r', center=0, ax=ax1,
               annot=True, fmt='.2f', annot_kws={'size': 9},
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('AP-1 Complex Members by Stage', fontweight='bold')
    for i, label in enumerate(ax1.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))

    ax2 = axes[1]
    ap1_mean = ap1_stage.mean(axis=1)
    for gene in ap1_stage.columns:
        ax2.plot(range(len(STAGE_ORDER)), ap1_stage[gene], 'o-', linewidth=2, markersize=8,
                label=gene, alpha=0.8)
    ax2.set_xticks(range(len(STAGE_ORDER)))
    ax2.set_xticklabels(STAGE_ORDER)
    ax2.set_ylabel('Expression', fontweight='bold')
    ax2.set_title('AP-1 Member Expression Trajectories', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=2)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle('AP-1 Transcription Factor Complex Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'ap1_analysis')
    plt.close()
    print("  Saved ap1_analysis.png/pdf")

except Exception as e:
    print(f"  AP-1 analysis failed: {e}")

# =============================================================================
# EDA 5: EMT Trajectory Analysis
# =============================================================================
print("\n[EDA] EMT trajectory analysis...")
sys.stdout.flush()

try:
    emt_epi = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'emt_epithelial_by_stage.parquet')
    emt_mes = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'emt_mesenchymal_by_stage.parquet')
    emt_epi = emt_epi.reindex(STAGE_ORDER)
    emt_mes = emt_mes.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax1 = axes[0]
    epi_mean = emt_epi.mean(axis=1)
    mes_mean = emt_mes.mean(axis=1)
    x = np.arange(len(STAGE_ORDER))
    width = 0.35
    ax1.bar(x - width/2, epi_mean, width, label='Epithelial', color='#4169E1', alpha=0.9)
    ax1.bar(x + width/2, mes_mean, width, label='Mesenchymal', color='#8B1A1A', alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGE_ORDER)
    ax1.set_ylabel('Mean Expression', fontweight='bold')
    ax1.set_title('Epithelial vs Mesenchymal Markers', fontweight='bold')
    ax1.legend()

    ax2 = axes[1]
    emt_ratio = mes_mean / (epi_mean + 0.01)
    colors = ['#4169E1' if r < 1 else '#8B1A1A' for r in emt_ratio]
    bars = ax2.bar(STAGE_ORDER, emt_ratio, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=2)
    ax2.set_ylabel('Mesenchymal/Epithelial Ratio', fontweight='bold')
    ax2.set_title('EMT Ratio by Stage', fontweight='bold')
    ax2.text(0.5, 1.05, 'Mesenchymal', ha='center', color='#8B1A1A', fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, -0.1, 'Epithelial', ha='center', color='#4169E1', fontweight='bold', transform=ax2.transAxes)

    ax3 = axes[2]
    for gene in emt_mes.columns:
        ax3.plot(range(len(STAGE_ORDER)), emt_mes[gene], 'o-', linewidth=2, markersize=6, label=gene, alpha=0.8)
    ax3.set_xticks(range(len(STAGE_ORDER)))
    ax3.set_xticklabels(STAGE_ORDER)
    ax3.set_ylabel('Expression', fontweight='bold')
    ax3.set_title('EMT-TF Expression', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    fig.suptitle('Epithelial-Mesenchymal Transition Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'emt_trajectory_analysis')
    plt.close()
    print("  Saved emt_trajectory_analysis.png/pdf")

except Exception as e:
    print(f"  EMT trajectory failed: {e}")

# =============================================================================
# EDA 6: Macrophage Polarization
# =============================================================================
print("\n[EDA] Macrophage polarization...")
sys.stdout.flush()

try:
    pw_ct = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_luca_celltype.parquet')

    if 'mac_m1' in pw_ct.columns and 'mac_m2' in pw_ct.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        ax1 = axes[0]
        mac_data = pw_ct[['mac_m1', 'mac_m2']].dropna()
        ax1.scatter(mac_data['mac_m1'], mac_data['mac_m2'], s=150, alpha=0.7,
                   c=mac_data['mac_m1'] - mac_data['mac_m2'], cmap='RdBu_r',
                   edgecolors='white', linewidth=2)
        for ct in mac_data.nlargest(10, 'mac_m1').index.union(mac_data.nlargest(10, 'mac_m2').index):
            ax1.annotate(ct[:20], (mac_data.loc[ct, 'mac_m1'], mac_data.loc[ct, 'mac_m2']),
                        fontsize=8, alpha=0.9)
        ax1.axline((0, 0), slope=1, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('M1 Score (Pro-inflammatory)', fontweight='bold')
        ax1.set_ylabel('M2 Score (Anti-inflammatory)', fontweight='bold')
        ax1.set_title('Macrophage Polarization by Cell Type', fontweight='bold')

        ax2 = axes[1]
        mac_diff = (mac_data['mac_m1'] - mac_data['mac_m2']).sort_values()
        colors = ['#8B1A1A' if v > 0 else '#1B2F5E' for v in mac_diff]
        ax2.barh(range(len(mac_diff)), mac_diff.values, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(mac_diff)))
        ax2.set_yticklabels(mac_diff.index, fontsize=8)
        ax2.axvline(x=0, color='gray', linewidth=2)
        ax2.set_xlabel('M1 - M2 Score', fontweight='bold')
        ax2.set_title('M1/M2 Balance', fontweight='bold')
        ax2.text(mac_diff.max() * 0.5, len(mac_diff) * 0.95, 'M1 dominant', color='#8B1A1A', fontweight='bold')
        ax2.text(mac_diff.min() * 0.5, len(mac_diff) * 0.95, 'M2 dominant', color='#1B2F5E', fontweight='bold')

        fig.suptitle('Tumor-Associated Macrophage Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, EDA, 'macrophage_polarization')
        plt.close()
        print("  Saved macrophage_polarization.png/pdf")

except Exception as e:
    print(f"  Macrophage analysis failed: {e}")

# =============================================================================
# EDA 7: T Cell Exhaustion
# =============================================================================
print("\n[EDA] T cell exhaustion analysis...")
sys.stdout.flush()

try:
    t_exh_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 't_exhaustion_by_stage.parquet')
    t_exh_stage = t_exh_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    t_exh_z = (t_exh_stage - t_exh_stage.mean()) / t_exh_stage.std()
    sns.heatmap(t_exh_z.T, cmap='magma', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Z-score'})
    ax1.set_title('T Cell Exhaustion Markers by Stage', fontweight='bold')

    ax2 = axes[1]
    for gene in t_exh_stage.columns:
        ax2.plot(range(len(STAGE_ORDER)), t_exh_stage[gene], 'o-', linewidth=2.5,
                markersize=10, label=gene, alpha=0.85)
    ax2.set_xticks(range(len(STAGE_ORDER)))
    ax2.set_xticklabels(STAGE_ORDER)
    ax2.set_ylabel('Expression', fontweight='bold')
    ax2.set_title('Exhaustion Marker Trajectories', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle('T Cell Exhaustion Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 't_cell_exhaustion')
    plt.close()
    print("  Saved t_cell_exhaustion.png/pdf")

except Exception as e:
    print(f"  T cell exhaustion failed: {e}")

# =============================================================================
# EDA 8: Immune Checkpoint Landscape
# =============================================================================
print("\n[EDA] Immune checkpoint landscape...")
sys.stdout.flush()

try:
    ckpt_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'immune_checkpoint_by_stage.parquet')
    ckpt_ct = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'immune_checkpoint_by_luca_celltype.parquet')
    ckpt_stage = ckpt_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax1 = axes[0]
    ckpt_z = (ckpt_stage - ckpt_stage.mean()) / ckpt_stage.std()
    sns.heatmap(ckpt_z.T, cmap='viridis', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.7})
    ax1.set_title('Checkpoint Expression by Stage', fontweight='bold')

    ax2 = axes[1]
    ckpt_top = ckpt_ct.sum(axis=1).nlargest(15)
    ckpt_subset = ckpt_ct.loc[ckpt_top.index]
    ckpt_norm = (ckpt_subset - ckpt_subset.min()) / (ckpt_subset.max() - ckpt_subset.min() + 0.01)
    sns.heatmap(ckpt_norm, cmap='magma', ax=ax2, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Normalized'})
    ax2.set_title('Checkpoint Expression by Cell Type (Top 15)', fontweight='bold')

    fig.suptitle('Immune Checkpoint Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'immune_checkpoint_landscape')
    plt.close()
    print("  Saved immune_checkpoint_landscape.png/pdf")

except Exception as e:
    print(f"  Checkpoint analysis failed: {e}")

# =============================================================================
# EDA 9: Signaling Pathway Comparison (Wnt/Notch/TGFb/NFkB)
# =============================================================================
print("\n[EDA] Major signaling pathways...")
sys.stdout.flush()

try:
    pathways = ['wnt', 'notch', 'tgfb', 'nfkb', 'hypoxia']
    pw_data = {}

    for pw in pathways:
        try:
            df = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / f'{pw}_by_stage.parquet')
            pw_data[pw] = df.reindex(STAGE_ORDER).mean(axis=1)
        except:
            pass

    if pw_data:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (pw_name, pw_vals) in enumerate(pw_data.items()):
            ax = axes[idx]
            bars = ax.bar(STAGE_ORDER, pw_vals, color=STAGE_PALETTE, edgecolor='white', linewidth=2)
            ax.set_title(f'{pw_name.upper()} Signaling', fontweight='bold')
            ax.set_ylabel('Mean Expression')
            for bar, val in zip(bars, pw_vals):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
                       ha='center', fontsize=8, fontweight='bold')

        ax_radar = axes[5]
        if len(pw_data) >= 4:
            categories = list(pw_data.keys())
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            ax_radar.remove()
            ax_radar = fig.add_subplot(2, 3, 6, polar=True)
            ax_radar.set_theta_offset(np.pi / 2)
            ax_radar.set_theta_direction(-1)

            for stage in STAGE_ORDER:
                values = [pw_data[pw].loc[stage] if stage in pw_data[pw].index else 0 for pw in categories]
                values_norm = [(v - min(values)) / (max(values) - min(values) + 0.01) for v in values]
                values_norm += values_norm[:1]
                ax_radar.plot(angles, values_norm, 'o-', linewidth=2, label=stage, color=STAGE_COLORS[stage])
                ax_radar.fill(angles, values_norm, alpha=0.1, color=STAGE_COLORS[stage])

            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels([p.upper() for p in categories], fontsize=9)
            ax_radar.set_title('Pathway Comparison', fontweight='bold', pad=15)
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

        fig.suptitle('Major Signaling Pathway Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_figure(fig, EDA, 'signaling_pathways')
        plt.close()
        print("  Saved signaling_pathways.png/pdf")

except Exception as e:
    print(f"  Signaling pathways failed: {e}")

# =============================================================================
# EDA 10: Alveolar Cell Type Markers (AT1 vs AT2)
# =============================================================================
print("\n[EDA] Alveolar cell type analysis...")
sys.stdout.flush()

try:
    at1 = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'at1_markers_by_stage.parquet')
    at2 = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'at2_markers_by_stage.parquet')
    at1 = at1.reindex(STAGE_ORDER)
    at2 = at2.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax1 = axes[0]
    at1_mean = at1.mean(axis=1)
    at2_mean = at2.mean(axis=1)
    x = np.arange(len(STAGE_ORDER))
    width = 0.35
    ax1.bar(x - width/2, at1_mean, width, label='AT1', color='#6B3FA0', alpha=0.9)
    ax1.bar(x + width/2, at2_mean, width, label='AT2', color='#046307', alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGE_ORDER)
    ax1.set_ylabel('Mean Expression', fontweight='bold')
    ax1.set_title('AT1 vs AT2 Markers', fontweight='bold')
    ax1.legend()

    ax2 = axes[1]
    at_ratio = at2_mean / (at1_mean + 0.01)
    bars = ax2.bar(STAGE_ORDER, at_ratio, color=STAGE_PALETTE, edgecolor='white', linewidth=2)
    ax2.set_ylabel('AT2/AT1 Ratio', fontweight='bold')
    ax2.set_title('Alveolar Type Ratio', fontweight='bold')
    ax2.axhline(y=1, color='gray', linestyle='--')

    ax3 = axes[2]
    for gene in at2.columns[:5]:
        ax3.plot(range(len(STAGE_ORDER)), at2[gene], 'o-', linewidth=2, markersize=8, label=gene)
    ax3.set_xticks(range(len(STAGE_ORDER)))
    ax3.set_xticklabels(STAGE_ORDER)
    ax3.set_ylabel('Expression', fontweight='bold')
    ax3.set_title('AT2 Marker Trajectories', fontweight='bold')
    ax3.legend(loc='best', fontsize=8)

    fig.suptitle('Alveolar Epithelial Cell Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'alveolar_analysis')
    plt.close()
    print("  Saved alveolar_analysis.png/pdf")

except Exception as e:
    print(f"  Alveolar analysis failed: {e}")

# =============================================================================
# EDA 11: Oncogene/Tumor Suppressor Landscape
# =============================================================================
print("\n[EDA] Oncogene/tumor suppressor analysis...")
sys.stdout.flush()

try:
    onco = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'oncogenes_by_stage.parquet')
    ts = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'tumor_suppressors_by_stage.parquet')
    onco = onco.reindex(STAGE_ORDER)
    ts = ts.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    onco_z = (onco - onco.mean()) / onco.std()
    sns.heatmap(onco_z.T, cmap='rocket', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('Oncogene Expression by Stage', fontweight='bold')

    ax2 = axes[0, 1]
    ts_z = (ts - ts.mean()) / ts.std()
    sns.heatmap(ts_z.T, cmap='mako', ax=ax2, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax2.set_title('Tumor Suppressor Expression by Stage', fontweight='bold')

    ax3 = axes[1, 0]
    for gene in ['KRAS', 'EGFR', 'MYC']:
        if gene in onco.columns:
            ax3.plot(range(len(STAGE_ORDER)), onco[gene], 'o-', linewidth=2.5,
                    markersize=10, label=gene)
    ax3.set_xticks(range(len(STAGE_ORDER)))
    ax3.set_xticklabels(STAGE_ORDER)
    ax3.set_ylabel('Expression', fontweight='bold')
    ax3.set_title('Key Oncogene Trajectories', fontweight='bold')
    ax3.legend()
    ax3.yaxis.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    for gene in ['TP53', 'RB1', 'STK11', 'KEAP1']:
        if gene in ts.columns:
            ax4.plot(range(len(STAGE_ORDER)), ts[gene], 'o-', linewidth=2.5,
                    markersize=10, label=gene)
    ax4.set_xticks(range(len(STAGE_ORDER)))
    ax4.set_xticklabels(STAGE_ORDER)
    ax4.set_ylabel('Expression', fontweight='bold')
    ax4.set_title('Key Tumor Suppressor Trajectories', fontweight='bold')
    ax4.legend()
    ax4.yaxis.grid(True, alpha=0.3)

    fig.suptitle('Driver Gene Landscape', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, EDA, 'driver_genes')
    plt.close()
    print("  Saved driver_genes.png/pdf")

except Exception as e:
    print(f"  Driver gene analysis failed: {e}")

# =============================================================================
# EDA 12: Correlation Matrix of All Pathways
# =============================================================================
print("\n[EDA] Pathway correlation analysis...")
sys.stdout.flush()

try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores.parquet')
    pw_cols = [c for c in pw_scores.columns if c not in ['stage', 'luca_cell_type']]

    corr = pw_scores[pw_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax,
               square=True, linewidths=0.5,
               cbar_kws={'shrink': 0.6, 'label': 'Correlation'},
               annot=True, fmt='.2f', annot_kws={'size': 7})
    ax.set_title('Pathway Score Correlations', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    save_figure(fig, EDA, 'pathway_correlations')
    plt.close()
    print("  Saved pathway_correlations.png/pdf")

except Exception as e:
    print(f"  Correlation analysis failed: {e}")

# =============================================================================
# EDA 13: Stage-Specific Marker Discovery (Top variable genes)
# =============================================================================
print("\n[EDA] Stage-specific marker discovery...")
sys.stdout.flush()

try:
    genes = pd.read_parquet(CANONICAL / 'key_genes' / 'key_genes_by_stage.parquet')

    mean_cols = [c for c in genes.columns if c.endswith('_mean')]
    gene_names = [c.replace('_mean', '') for c in mean_cols]

    genes_mean = genes[mean_cols].copy()
    genes_mean.columns = gene_names

    gene_var = genes_mean.var()
    top_var = gene_var.nlargest(30).index.tolist()

    genes_top = genes_mean[top_var].reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(18, 10))
    genes_z = (genes_top - genes_top.mean()) / genes_top.std()

    sns.heatmap(genes_z.T, cmap='RdBu_r', center=0, ax=ax,
               linewidths=0.5, cbar_kws={'shrink': 0.6, 'label': 'Z-score'},
               yticklabels=True)

    ax.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax.set_title('Top 30 Most Variable Genes Across Stages', fontsize=14, fontweight='bold', pad=15)

    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))
        label.set_fontweight('bold')

    plt.tight_layout()
    save_figure(fig, EDA, 'top_variable_genes')
    plt.close()
    print("  Saved top_variable_genes.png/pdf")

except Exception as e:
    print(f"  Marker discovery failed: {e}")

# =============================================================================
# EDA 14: Inflammatory Cytokine Network
# =============================================================================
print("\n[EDA] Inflammatory cytokine analysis...")
sys.stdout.flush()

try:
    infl = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'inflammatory_by_stage.parquet')
    infl = infl.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    infl_z = (infl - infl.mean()) / infl.std()
    sns.heatmap(infl_z.T, cmap='magma', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('Inflammatory Cytokines by Stage', fontweight='bold')

    ax2 = axes[1]
    key_cytokines = ['IL6', 'TNF', 'IL1B', 'CCL2', 'CXCL10']
    for gene in key_cytokines:
        if gene in infl.columns:
            ax2.plot(range(len(STAGE_ORDER)), infl[gene], 'o-', linewidth=2.5,
                    markersize=10, label=gene)
    ax2.set_xticks(range(len(STAGE_ORDER)))
    ax2.set_xticklabels(STAGE_ORDER)
    ax2.set_ylabel('Expression', fontweight='bold')
    ax2.set_title('Key Inflammatory Cytokine Trajectories', fontweight='bold')
    ax2.legend(loc='best')
    ax2.yaxis.grid(True, alpha=0.3)

    fig.suptitle('Inflammatory Microenvironment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'inflammatory_cytokines')
    plt.close()
    print("  Saved inflammatory_cytokines.png/pdf")

except Exception as e:
    print(f"  Inflammatory analysis failed: {e}")

# =============================================================================
# EDA 15: Dendritic Cell Subtypes
# =============================================================================
print("\n[EDA] Dendritic cell analysis...")
sys.stdout.flush()

try:
    dc_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'dendritic_cells_by_stage.parquet')
    dc_stage = dc_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(14, 8))
    dc_z = (dc_stage - dc_stage.mean()) / dc_stage.std()
    sns.heatmap(dc_z.T, cmap='plasma', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 8},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('Dendritic Cell Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))
        label.set_fontweight('bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'dendritic_cells')
    plt.close()
    print("  Saved dendritic_cells.png/pdf")

except Exception as e:
    print(f"  DC analysis failed: {e}")

# =============================================================================
# EDA 16: Neutrophil Infiltration
# =============================================================================
print("\n[EDA] Neutrophil analysis...")
sys.stdout.flush()

try:
    neut_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'neutrophils_by_stage.parquet')
    neut_stage = neut_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(14, 8))
    neut_z = (neut_stage - neut_stage.mean()) / neut_stage.std()
    sns.heatmap(neut_z.T, cmap='inferno', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 8},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('Neutrophil Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(STAGE_COLORS.get(label.get_text(), 'black'))
        label.set_fontweight('bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'neutrophils')
    plt.close()
    print("  Saved neutrophils.png/pdf")

except Exception as e:
    print(f"  Neutrophil analysis failed: {e}")

# =============================================================================
# EDA 17: B Cell and Plasma Cell Analysis
# =============================================================================
print("\n[EDA] B cell/Plasma cell analysis...")
sys.stdout.flush()

try:
    b_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'b_cells_by_stage.parquet')
    plasma_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'plasma_cells_by_stage.parquet')
    b_stage = b_stage.reindex(STAGE_ORDER)
    plasma_stage = plasma_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    b_z = (b_stage - b_stage.mean()) / b_stage.std()
    sns.heatmap(b_z.T, cmap='viridis', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('B Cell Markers', fontsize=13, fontweight='bold')

    ax2 = axes[1]
    plasma_z = (plasma_stage - plasma_stage.mean()) / plasma_stage.std()
    sns.heatmap(plasma_z.T, cmap='cividis', ax=ax2, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax2.set_title('Plasma Cell Markers', fontsize=13, fontweight='bold')

    fig.suptitle('B Cell Lineage Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'b_cells_plasma')
    plt.close()
    print("  Saved b_cells_plasma.png/pdf")

except Exception as e:
    print(f"  B cell analysis failed: {e}")

# =============================================================================
# EDA 18: NK Cell Analysis
# =============================================================================
print("\n[EDA] NK cell analysis...")
sys.stdout.flush()

try:
    nk_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'nk_cells_by_stage.parquet')
    nk_stage = nk_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(12, 8))
    nk_z = (nk_stage - nk_stage.mean()) / nk_stage.std()
    sns.heatmap(nk_z.T, cmap='magma', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 9},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('NK Cell Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'nk_cells')
    plt.close()
    print("  Saved nk_cells.png/pdf")

except Exception as e:
    print(f"  NK cell analysis failed: {e}")

# =============================================================================
# EDA 19: Stromal Compartment - Endothelial and Lymphatic
# =============================================================================
print("\n[EDA] Stromal/Endothelial analysis...")
sys.stdout.flush()

try:
    endo_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'endothelial_by_stage.parquet')
    lymph_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'lymphatic_by_stage.parquet')
    endo_stage = endo_stage.reindex(STAGE_ORDER)
    lymph_stage = lymph_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    endo_z = (endo_stage - endo_stage.mean()) / endo_stage.std()
    sns.heatmap(endo_z.T, cmap='mako', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('Endothelial Markers', fontsize=13, fontweight='bold')

    ax2 = axes[1]
    if len(lymph_stage.columns) > 0:
        lymph_z = (lymph_stage - lymph_stage.mean()) / lymph_stage.std()
        sns.heatmap(lymph_z.T, cmap='rocket', ax=ax2, annot=True, fmt='.2f',
                   linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax2.set_title('Lymphatic Markers', fontsize=13, fontweight='bold')

    fig.suptitle('Vascular Compartment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'endothelial_lymphatic')
    plt.close()
    print("  Saved endothelial_lymphatic.png/pdf")

except Exception as e:
    print(f"  Stromal analysis failed: {e}")

# =============================================================================
# EDA 20: Pericytes and Smooth Muscle
# =============================================================================
print("\n[EDA] Pericyte/SMC analysis...")
sys.stdout.flush()

try:
    peri_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'pericytes_by_stage.parquet')
    smc_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'smooth_muscle_by_stage.parquet')
    peri_stage = peri_stage.reindex(STAGE_ORDER)
    smc_stage = smc_stage.reindex(STAGE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    peri_z = (peri_stage - peri_stage.mean()) / peri_stage.std()
    sns.heatmap(peri_z.T, cmap='flare', ax=ax1, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax1.set_title('Pericyte Markers', fontsize=13, fontweight='bold')

    ax2 = axes[1]
    smc_z = (smc_stage - smc_stage.mean()) / smc_stage.std()
    sns.heatmap(smc_z.T, cmap='crest', ax=ax2, annot=True, fmt='.2f',
               linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax2.set_title('Smooth Muscle Markers', fontsize=13, fontweight='bold')

    fig.suptitle('Mural Cell Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, EDA, 'pericytes_smc')
    plt.close()
    print("  Saved pericytes_smc.png/pdf")

except Exception as e:
    print(f"  Pericyte/SMC analysis failed: {e}")

# =============================================================================
# EDA 21: Treg Analysis
# =============================================================================
print("\n[EDA] Treg analysis...")
sys.stdout.flush()

try:
    treg_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'tregs_by_stage.parquet')
    treg_stage = treg_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(12, 8))
    treg_z = (treg_stage - treg_stage.mean()) / treg_stage.std()
    sns.heatmap(treg_z.T, cmap='RdPu', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 9},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('Regulatory T Cell Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'tregs')
    plt.close()
    print("  Saved tregs.png/pdf")

except Exception as e:
    print(f"  Treg analysis failed: {e}")

# =============================================================================
# EDA 22: CD8 T Cell Subtypes
# =============================================================================
print("\n[EDA] CD8 T cell subtypes...")
sys.stdout.flush()

try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_stage.parquet')
    pw_scores = pw_scores.reindex(STAGE_ORDER)

    cd8_cols = ['cd8_naive', 'cd8_effector', 'cd8_memory', 'cd8_resident']
    cd8_avail = [c for c in cd8_cols if c in pw_scores.columns]

    if cd8_avail:
        fig, ax = plt.subplots(figsize=(10, 7))
        for col in cd8_avail:
            label = col.replace('cd8_', '').title()
            ax.plot(range(len(STAGE_ORDER)), pw_scores[col], 'o-', linewidth=3,
                   markersize=12, label=label, markeredgecolor='white', markeredgewidth=2)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, fontsize=11)
        for i, stage in enumerate(STAGE_ORDER):
            ax.get_xticklabels()[i].set_color(STAGE_COLORS[stage])
            ax.get_xticklabels()[i].set_fontweight('bold')
        ax.set_ylabel('Pathway Score', fontweight='bold')
        ax.set_title('CD8+ T Cell Subtype Trajectories', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, EDA, 'cd8_subtypes')
        plt.close()
        print("  Saved cd8_subtypes.png/pdf")

except Exception as e:
    print(f"  CD8 subtype analysis failed: {e}")

# =============================================================================
# EDA 23: Monocyte Subtypes
# =============================================================================
print("\n[EDA] Monocyte analysis...")
sys.stdout.flush()

try:
    mono_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'monocytes_by_stage.parquet')
    mono_stage = mono_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(14, 8))
    mono_z = (mono_stage - mono_stage.mean()) / mono_stage.std()
    sns.heatmap(mono_z.T, cmap='icefire', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 9},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('Monocyte Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'monocytes')
    plt.close()
    print("  Saved monocytes.png/pdf")

except Exception as e:
    print(f"  Monocyte analysis failed: {e}")

# =============================================================================
# EDA 24: Immune Landscape Overview - Scanpy-style Dotplot
# =============================================================================
print("\n[EDA] Immune landscape dotplot...")
sys.stdout.flush()

try:
    pw_ct = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores_by_luca_celltype.parquet')

    immune_pathways = ['t_cell', 'mac_m1', 'mac_m2', 'dendritic_cells', 'neutrophils',
                      'b_cells', 'nk_cells', 'tregs', 't_exhaustion']
    avail_pw = [p for p in immune_pathways if p in pw_ct.columns]

    if avail_pw:
        pw_immune = pw_ct[avail_pw]
        top_ct = pw_immune.sum(axis=1).nlargest(20).index
        pw_top = pw_immune.loc[top_ct]

        pw_scaled = (pw_top - pw_top.min()) / (pw_top.max() - pw_top.min() + 0.001)

        fig, ax = plt.subplots(figsize=(14, 12))

        for i, ct in enumerate(pw_top.index):
            for j, pw in enumerate(avail_pw):
                val = pw_top.loc[ct, pw]
                size = pw_scaled.loc[ct, pw]
                color = plt.cm.plasma(size)
                ax.scatter(j, i, s=size * 400 + 30, c=[color], edgecolors='#333333',
                          linewidths=0.5, zorder=3)

        ax.set_xticks(range(len(avail_pw)))
        ax.set_xticklabels([p.replace('_', ' ').title() for p in avail_pw],
                          rotation=45, ha='right', fontsize=10)
        ax.set_yticks(range(len(pw_top.index)))
        ax.set_yticklabels(pw_top.index, fontsize=9)

        for i in range(len(pw_top.index)):
            ax.axhline(i, color='#EEEEEE', linewidth=0.5, zorder=1)
        for j in range(len(avail_pw)):
            ax.axvline(j, color='#EEEEEE', linewidth=0.5, zorder=1)

        ax.set_xlim(-0.5, len(avail_pw) - 0.5)
        ax.set_ylim(-0.5, len(pw_top.index) - 0.5)

        ax.set_title('Immune Pathway Enrichment by Cell Type', fontsize=14, fontweight='bold', pad=15)

        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Relative Enrichment', fontweight='bold')

        plt.tight_layout()
        save_figure(fig, EDA, 'immune_landscape_dotplot')
        plt.close()
        print("  Saved immune_landscape_dotplot.png/pdf")

except Exception as e:
    print(f"  Immune dotplot failed: {e}")

# =============================================================================
# EDA 25: Mast Cell Analysis
# =============================================================================
print("\n[EDA] Mast cell analysis...")
sys.stdout.flush()

try:
    mast_stage = pd.read_parquet(CANONICAL / 'key_genes' / 'pathways' / 'mast_cells_by_stage.parquet')
    mast_stage = mast_stage.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(12, 7))
    mast_z = (mast_stage - mast_stage.mean()) / mast_stage.std()
    sns.heatmap(mast_z.T, cmap='YlOrBr', ax=ax, annot=True, fmt='.2f', annot_kws={'size': 10},
               linewidths=0.5, cbar_kws={'shrink': 0.7, 'label': 'Z-score'})
    ax.set_title('Mast Cell Markers by Disease Stage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Disease Stage', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, EDA, 'mast_cells')
    plt.close()
    print("  Saved mast_cells.png/pdf")

except Exception as e:
    print(f"  Mast cell analysis failed: {e}")

# =============================================================================
# EDA 26: Full Pathway Correlation Clustermap
# =============================================================================
print("\n[EDA] Pathway correlation clustermap...")
sys.stdout.flush()

try:
    pw_scores = pd.read_parquet(CANONICAL / 'key_genes' / 'pathway_scores.parquet')
    pw_cols = [c for c in pw_scores.columns if c not in ['stage', 'luca_cell_type']]

    corr = pw_scores[pw_cols].corr()

    g = sns.clustermap(
        corr,
        cmap='RdBu_r',
        center=0,
        figsize=(16, 14),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        linewidths=0.3,
        linecolor='white',
        method='ward',
        annot=False,
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.tick_params(axis='both', labelsize=8)
    g.fig.suptitle('Pathway Score Correlations (Hierarchically Clustered)',
                  fontsize=14, fontweight='bold', y=1.02)
    g.cax.set_ylabel('Correlation', fontsize=10)

    save_figure(g.fig, EDA, 'pathway_clustermap')
    plt.close()
    print("  Saved pathway_clustermap.png/pdf")

except Exception as e:
    print(f"  Pathway clustermap failed: {e}")

print("\n[EDA] Complete!")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("FIGURES COMPLETE")
print("=" * 60)
print(f"Output directory: {FIGURES}")
print("\nGenerated figures:")
for f in sorted(FIGURES.glob('*.png')):
    print(f"  - {f.name}")
PYTHON_FIGURES

# =============================================================================
# Copy figures to local
# =============================================================================
echo ""
echo "=============================================="
echo "Copying outputs to local..."
echo "=============================================="

cp -r $FIGURES/* /home/booka/StageBridge/figures/publication/ 2>/dev/null || true
cp $CANONICAL/liana_luca/*.parquet /home/booka/StageBridge/data/ 2>/dev/null || true

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE!"
echo "=============================================="
echo "Date: $(date)"
echo ""
echo "Outputs:"
echo "  Data: $CANONICAL/"
echo "  Figures: $FIGURES/"
echo "  Local: /home/booka/StageBridge/data/"
echo ""
echo "Key files:"
echo "  - liana_luca/liana_interactions_luca.parquet (33 cell types)"
echo "  - signatures/gene_signatures.parquet"
echo "  - activity/pathway_activity_progeny.parquet"
echo "  - communication/il1b_interactions_luca.parquet"
echo ""
echo "Figures (11 single-panel PNGs):"
ls -1 $FIGURES/*.png 2>/dev/null || echo "  (figures pending)"
