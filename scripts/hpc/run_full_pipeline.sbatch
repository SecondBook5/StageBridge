#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/booka/StageBridge/logs/full_pipeline_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/full_pipeline_%j.err

# Complete StageBridge data preparation pipeline
# 1. Transfer LuCA labels (33 cell types)
# 2. Run LIANA with LuCA cell types
# 3. Full data prep (signatures, DE, activity, trajectories, niches)
# 4. Generate single-panel figures for each analysis

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
LUCA_REF = "/scratch/chaunzt1/stagebridge/references/luca/luca_core_atlas.h5ad"
CANONICAL = Path(os.environ['CANONICAL'])
OUTPUT = CANONICAL / "liana_luca"
LOCAL = Path("/home/booka/StageBridge/data")

print("=" * 60)
print("Step 1.1: Load snRNA data")
print("=" * 60)
sys.stdout.flush()

adata = sc.read_h5ad(SNRNA)
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Existing cell type column: {adata.obs['cell_type'].nunique()} types (HLCA)")
sys.stdout.flush()

print("\n" + "=" * 60)
print("Step 1.2: Transfer LuCA labels via Ridge + k-NN")
print("=" * 60)
sys.stdout.flush()

print("Loading LuCA reference atlas...")
sys.stdout.flush()
luca_ref = sc.read_h5ad(LUCA_REF)
print(f"  Reference: {luca_ref.n_obs:,} cells")

latent_key = "X_scANVI" if "X_scANVI" in luca_ref.obsm else "X_scVI"
ref_latent = luca_ref.obsm[latent_key]
ref_labels = luca_ref.obs["cell_type"].values
print(f"  Latent: {latent_key} ({ref_latent.shape[1]}d)")
print(f"  Cell types: {np.unique(ref_labels).shape[0]}")
sys.stdout.flush()

print("\nProjecting query cells to reference latent space...")
sys.stdout.flush()
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

shared_genes = sorted(set(adata.var_names) & set(luca_ref.var_names))
print(f"  Shared genes: {len(shared_genes)}")
sys.stdout.flush()

query_expr = adata[:, shared_genes].X
if hasattr(query_expr, "toarray"):
    query_expr = query_expr.toarray()
query_expr = np.log1p(query_expr).astype(np.float32)

ref_expr = luca_ref[:, shared_genes].X
if hasattr(ref_expr, "toarray"):
    ref_expr = ref_expr.toarray()
ref_expr = np.log1p(ref_expr).astype(np.float32)

print("  Learning expression -> latent mapping (subsampled)...")
sys.stdout.flush()
n_subsample = min(50000, ref_expr.shape[0])
subsample_idx = np.random.choice(ref_expr.shape[0], n_subsample, replace=False)

ridge = Ridge(alpha=1.0)
ridge.fit(ref_expr[subsample_idx], ref_latent[subsample_idx])
query_latent = ridge.predict(query_expr)
print(f"  Query latent: {query_latent.shape}")
sys.stdout.flush()

print("\nFinding 15 nearest neighbors...")
sys.stdout.flush()
knn = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1)
knn.fit(ref_latent)
distances, indices = knn.kneighbors(query_latent)

print("Transferring labels by majority vote...")
sys.stdout.flush()
from tqdm import tqdm
transferred_labels = []
confidence = []
for i in tqdm(range(len(query_latent)), desc="Label transfer"):
    neighbor_labels = ref_labels[indices[i]]
    unique, counts = np.unique(neighbor_labels, return_counts=True)
    best = unique[counts.argmax()]
    transferred_labels.append(best)
    confidence.append(counts.max() / 15)

adata.obs["cell_type_luca"] = transferred_labels
adata.obs["cell_type_luca_confidence"] = confidence

print(f"\nTransferred {len(np.unique(transferred_labels))} LuCA cell types:")
print(pd.Series(transferred_labels).value_counts().head(15))
sys.stdout.flush()

del luca_ref, ref_latent, ref_labels, query_latent, ref_expr
import gc
gc.collect()

print("\n" + "=" * 60)
print("Step 1.3: Run LIANA with LuCA cell types")
print("=" * 60)
sys.stdout.flush()

import liana as li

print("Running LIANA (this takes ~30 min)...")
sys.stdout.flush()

li.mt.rank_aggregate(
    adata,
    groupby="cell_type_luca",
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

celltypes_out = OUTPUT / "cell_type_luca_mapping.parquet"
adata.obs[["cell_type", "cell_type_luca", "cell_type_luca_confidence"]].to_parquet(celltypes_out)
print(f"Saved: {celltypes_out}")

import json
summary = {
    "n_cells": int(adata.n_obs),
    "n_luca_types": int(adata.obs["cell_type_luca"].nunique()),
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

signatures.to_parquet(out / 'gene_signatures.parquet')
print(f'  Saved gene_signatures.parquet')

if 'stage' in signatures.columns:
    summary = signatures.groupby('stage')[['emt_score', 'senescence_score', 'sasp_score']].mean()
    summary.to_parquet(out / 'signatures_by_stage.parquet')
    print(f'  Saved signatures_by_stage.parquet')

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

print('Computing TF activity (CollecTRI)...')
sys.stdout.flush()
collectri = dc.op.collectri(organism='human')
dc.mt.ulm(data=adata, net=collectri)
tf_df = dc.extract(adata)
tf_df.to_parquet(out / 'tf_activity_collectri.parquet')

print('Computing pathway activity (PROGENy)...')
sys.stdout.flush()
progeny = dc.op.progeny(organism='human')
dc.mt.ulm(data=adata, net=progeny)
pw_df = dc.extract(adata)
pw_df.to_parquet(out / 'pathway_activity_progeny.parquet')

if 'stage' in adata.obs.columns:
    print('Computing stage summaries...')
    for name, df in [('tf', tf_df), ('pathway', pw_df)]:
        df_copy = df.copy()
        df_copy['stage'] = adata.obs['stage'].values
        df_copy.groupby('stage').mean().to_parquet(out / f'{name}_activity_by_stage.parquet')

print('decoupleR complete')
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

KEY_GENES = [
    'IL1B', 'IL1R1', 'IL1R2', 'IL1RAP', 'IL1RN',
    'EPCAM', 'KRT7', 'KRT18', 'KRT19', 'CDH1',
    'SFTPC', 'SFTPA1', 'SFTPA2', 'NKX2-1', 'HOPX', 'AGER',
    'VIM', 'CDH2', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'TWIST1',
    'TP53', 'KRAS', 'EGFR', 'MYC', 'SOX2', 'SOX9',
    'CD274', 'PDCD1', 'CTLA4', 'CD8A', 'CD4', 'FOXP3',
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
PYTHON_GENES
fi

# =============================================================================
# PART 6: Generate Publication Figures (using stagebridge.viz)
# =============================================================================
echo ""
echo "=============================================="
echo "PART 6: Generate Publication Figures"
echo "=============================================="

python << 'PYTHON_FIGURES'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

CANONICAL = Path(os.environ['CANONICAL'])
FIGURES = Path("/home/booka/StageBridge/figures/publication")
FIGURES.mkdir(exist_ok=True, parents=True)

print("Generating publication figures...")
sys.stdout.flush()

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

        # Generate all LIANA figures (Ricci network, heatmap, chord, IL1B, etc.)
        liana_viz.generate_all_figures(df, FIGURES / 'liana', show=False)
        print("  Generated LIANA figure suite")
    else:
        print("  SKIP: LIANA data not found")
except Exception as e:
    print(f"  LIANA figures failed: {e}")

# =============================================================================
# Signature Figures (EMT, senescence, SASP with publication styling)
# =============================================================================
print("\n[SIGNATURES] Gene signature figures...")
sys.stdout.flush()

# Use StageBridge theme for consistency
from stagebridge.viz.theme import configure_publication_style, STAGE_COLORS, save_figure
configure_publication_style()

STAGE_ORDER = ['Normal', 'AAH', 'AIS', 'MIA', 'LUAD']

try:
    sig = pd.read_parquet(CANONICAL / 'signatures' / 'signatures_by_stage.parquet')
    sig = sig.reindex(STAGE_ORDER)
    colors = [STAGE_COLORS[s] for s in sig.index]

    # EMT score - clean publication style
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(sig)), sig['emt_score'], color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels(sig.index, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean EMT Score')
    ax.set_title('Epithelial-Mesenchymal Transition', pad=12)
    ax.axhline(0, color='#CCCCCC', linewidth=1, linestyle='-', zorder=0)
    # Position values inside bars if negative, outside if positive
    for bar, val in zip(bars, sig['emt_score']):
        y_pos = val + 0.02 * (1 if val >= 0 else -1) * abs(sig['emt_score']).max()
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylim(min(0, sig['emt_score'].min() * 1.3), sig['emt_score'].max() * 1.15)
    plt.tight_layout()
    save_figure(fig, FIGURES, 'emt_by_stage')
    plt.close()
    print("  Saved emt_by_stage.png/pdf")

    # Senescence score
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(sig)), sig['senescence_score'], color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels(sig.index, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Senescence Score')
    ax.set_title('Cellular Senescence', pad=12)
    ymax = sig['senescence_score'].max()
    for bar, val in zip(bars, sig['senescence_score']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 * ymax, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylim(0, ymax * 1.15)
    plt.tight_layout()
    save_figure(fig, FIGURES, 'senescence_by_stage')
    plt.close()
    print("  Saved senescence_by_stage.png/pdf")

    # SASP score
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(sig)), sig['sasp_score'], color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(sig)))
    ax.set_xticklabels(sig.index, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean SASP Score')
    ax.set_title('Senescence-Associated Secretory Phenotype', pad=12)
    ymax = sig['sasp_score'].max()
    for bar, val in zip(bars, sig['sasp_score']):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02 * ymax, f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylim(0, ymax * 1.15)
    plt.tight_layout()
    save_figure(fig, FIGURES, 'sasp_by_stage')
    plt.close()
    print("  Saved sasp_by_stage.png/pdf")

except Exception as e:
    print(f"  Signature figures failed: {e}")

# =============================================================================
# Pathway Activity Heatmap
# =============================================================================
print("\n[ACTIVITY] Pathway activity heatmap...")
sys.stdout.flush()

try:
    pw = pd.read_parquet(CANONICAL / 'activity' / 'pathway_activity_by_stage.parquet')
    pw = pw.reindex(STAGE_ORDER)

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(pw.values.min()), abs(pw.values.max()))
    im = ax.imshow(pw.T.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(pw.columns)))
    ax.set_yticklabels(pw.columns, fontsize=10)
    ax.set_xlabel('Disease Stage')
    ax.set_ylabel('Pathway')
    ax.set_title('PROGENy Pathway Activity', pad=12)

    # Add values in cells
    for i in range(len(pw.columns)):
        for j in range(len(STAGE_ORDER)):
            val = pw.T.values[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Activity Score', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, FIGURES, 'pathway_activity_heatmap')
    plt.close()
    print("  Saved pathway_activity_heatmap.png/pdf")
except Exception as e:
    print(f"  Pathway heatmap failed: {e}")

# =============================================================================
# TF Activity Heatmap (top 15 variable)
# =============================================================================
print("\n[ACTIVITY] TF activity heatmap...")
sys.stdout.flush()

try:
    tf = pd.read_parquet(CANONICAL / 'activity' / 'tf_activity_by_stage.parquet')
    tf = tf.reindex(STAGE_ORDER)
    tf_var = tf.var().nlargest(15).index
    tf_top = tf[tf_var]

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = max(abs(tf_top.values.min()), abs(tf_top.values.max()))
    im = ax.imshow(tf_top.T.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(tf_var)))
    ax.set_yticklabels(tf_var, fontsize=10)
    ax.set_xlabel('Disease Stage')
    ax.set_ylabel('Transcription Factor')
    ax.set_title('Top Variable TF Activity (CollecTRI)', pad=12)

    # Add values in cells
    for i in range(len(tf_var)):
        for j in range(len(STAGE_ORDER)):
            val = tf_top.T.values[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Activity Score', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, FIGURES, 'tf_activity_heatmap')
    plt.close()
    print("  Saved tf_activity_heatmap.png/pdf")
except Exception as e:
    print(f"  TF heatmap failed: {e}")

# =============================================================================
# Cell Type Distribution (LuCA 33 types) - horizontal bar with gradient
# =============================================================================
print("\n[CELLTYPE] Cell type distribution...")
sys.stdout.flush()

try:
    from stagebridge.viz.theme import CELLTYPE_COLORS

    ct_map = pd.read_parquet(CANONICAL / 'liana_luca' / 'cell_type_luca_mapping.parquet')
    ct_counts = ct_map['cell_type_luca'].value_counts().head(15)  # Top 15 for readability

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use cell type colors where available, gradient for others
    bar_colors = [CELLTYPE_COLORS.get(ct, plt.cm.viridis(0.3 + 0.5 * i / len(ct_counts)))
                  for i, ct in enumerate(ct_counts.index)]

    bars = ax.barh(range(len(ct_counts)), ct_counts.values, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.set_yticks(range(len(ct_counts)))
    ax.set_yticklabels(ct_counts.index, fontsize=10)
    ax.set_xlabel('Number of Cells')
    ax.set_title('Cell Type Distribution (LuCA)', pad=12)
    ax.invert_yaxis()

    # Values at end of bars - outside with padding
    max_val = ct_counts.max()
    for bar, val in zip(bars, ct_counts.values):
        ax.text(val + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9, fontweight='bold', color='#333333')

    ax.set_xlim(0, max_val * 1.15)  # Leave room for labels
    plt.tight_layout()
    save_figure(fig, FIGURES, 'celltype_distribution_luca')
    plt.close()
    print("  Saved celltype_distribution_luca.png/pdf")
except Exception as e:
    print(f"  Cell type distribution failed: {e}")

# =============================================================================
# Label Transfer Confidence - clean histogram
# =============================================================================
print("\n[QC] Label transfer confidence...")
sys.stdout.flush()

try:
    ct_map = pd.read_parquet(CANONICAL / 'liana_luca' / 'cell_type_luca_mapping.parquet')
    conf = ct_map['cell_type_luca_confidence']

    fig, ax = plt.subplots(figsize=(7, 5))
    n, bins, patches = ax.hist(conf, bins=40, color='#4169E1', edgecolor='white', linewidth=0.8)

    # Color bars by confidence level
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge >= 0.8:
            patch.set_facecolor('#228B22')  # Green for high confidence
        elif left_edge >= 0.6:
            patch.set_facecolor('#DAA520')  # Gold for medium
        else:
            patch.set_facecolor('#CD5C5C')  # Red for low

    ax.axvline(conf.median(), color='#000000', linewidth=2.5, linestyle='--', zorder=10)
    ax.text(conf.median() + 0.02, ax.get_ylim()[1] * 0.95,
            f'Median: {conf.median():.2f}', fontsize=11, fontweight='bold', va='top')

    ax.set_xlabel('k-NN Confidence (majority vote fraction)')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Label Transfer Quality', pad=12)

    # Add annotation about quality
    high_conf = (conf >= 0.8).sum() / len(conf) * 100
    ax.text(0.98, 0.98, f'{high_conf:.0f}% cells with >80% confidence',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            fontweight='bold', color='#228B22')

    plt.tight_layout()
    save_figure(fig, FIGURES, 'label_transfer_confidence')
    plt.close()
    print("  Saved label_transfer_confidence.png/pdf")
except Exception as e:
    print(f"  Confidence histogram failed: {e}")

# =============================================================================
# IL1B Expression Line Plot - publication quality
# =============================================================================
print("\n[GENES] IL1B pathway expression...")
sys.stdout.flush()

try:
    genes = pd.read_parquet(CANONICAL / 'key_genes' / 'key_genes_by_stage.parquet')

    il1_genes = ['IL1B', 'IL1R1', 'IL1R2', 'IL1RAP']
    available = [g for g in il1_genes if f'{g}_mean' in genes.columns]

    if available:
        # Deep, distinguishable colors
        gene_colors = {'IL1B': '#8B0000', 'IL1R1': '#4169E1', 'IL1R2': '#228B22', 'IL1RAP': '#9932CC'}

        fig, ax = plt.subplots(figsize=(8, 5))

        for gene in available:
            col = f'{gene}_mean'
            vals = genes[col].reindex(STAGE_ORDER)
            color = gene_colors.get(gene, '#666666')
            ax.plot(range(len(STAGE_ORDER)), vals, marker='o', markersize=10, linewidth=3,
                   color=color, label=gene, markeredgecolor='white', markeredgewidth=1.5)

        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, fontsize=11, fontweight='bold')
        ax.set_xlabel('Disease Stage')
        ax.set_ylabel('Mean Expression')
        ax.set_title('IL1B Signaling Pathway', pad=12)

        # Legend outside to avoid overlap
        ax.legend(frameon=False, loc='upper left', fontsize=11)

        # Subtle grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')

        plt.tight_layout()
        save_figure(fig, FIGURES, 'il1b_pathway_expression')
        plt.close()
        print("  Saved il1b_pathway_expression.png/pdf")
except Exception as e:
    print(f"  IL1B expression failed: {e}")

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
