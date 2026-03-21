#!/bin/bash
#SBATCH --job-name=retrain_luca
#SBATCH --output=logs/retrain_luca_%j.out
#SBATCH --error=logs/retrain_luca_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL

# Retrain LuCA scANVI model with current environment
# Uses 4x H100 GPUs with Distributed Data Parallel (DDP)
# Goal: Identical quality to original, compatible with current pandas/anndata

set -e

export DATA=/scratch/chaunzt1/stagebridge
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p logs

echo "=============================================="
echo "Retrain LuCA scANVI Model (Multi-GPU)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x H100 (DDP)"
echo "=============================================="

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/stagebridge_env

cd /home/booka/StageBridge

python << 'PYTHON_SCRIPT'
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
from pathlib import Path
from datetime import datetime

# Paths
DATA = os.environ.get("DATA", "/scratch/chaunzt1/stagebridge")
luca_path = Path(DATA) / "references/luca/luca_core_atlas.h5ad"
output_dir = Path(DATA) / "references/luca/retrained_model"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 1: Load and inspect LuCA Core Atlas")
print("=" * 60)

# Load atlas
print(f"Loading: {luca_path}")
adata = ad.read_h5ad(luca_path)
print(f"Shape: {adata.shape}")
print(f"Obs columns: {list(adata.obs.columns)}")

# Find batch and label keys
print("\nSearching for batch key...")
batch_candidates = ['dataset', 'batch', 'study', 'sample', 'patient', 'donor']
batch_key = None
for col in batch_candidates:
    if col in adata.obs.columns:
        n_batches = adata.obs[col].nunique()
        print(f"  Found '{col}': {n_batches} unique values")
        if batch_key is None and n_batches > 1 and n_batches < len(adata):
            batch_key = col

print(f"\nSearching for cell type label key...")
label_candidates = ['cell_type', 'cell_type_fine', 'cell_type_coarse', 'celltype',
                    'ann_level_1', 'ann_level_2', 'ann_level_3', 'annotation']
labels_key = None
for col in label_candidates:
    if col in adata.obs.columns:
        n_labels = adata.obs[col].nunique()
        print(f"  Found '{col}': {n_labels} unique values")
        if labels_key is None:
            labels_key = col

# Check for existing latent representation to understand original model
print("\nChecking for original model info...")
if 'X_scANVI' in adata.obsm:
    original_n_latent = adata.obsm['X_scANVI'].shape[1]
    print(f"  Original latent dim: {original_n_latent}")
elif 'X_scVI' in adata.obsm:
    original_n_latent = adata.obsm['X_scVI'].shape[1]
    print(f"  Original latent dim (scVI): {original_n_latent}")
else:
    original_n_latent = 10  # LuCA default
    print(f"  Using default latent dim: {original_n_latent}")

# Check for raw counts
print("\nChecking for count data...")
if adata.raw is not None:
    print("  Using adata.raw for counts")
    # Create a new adata with raw counts
    adata_train = ad.AnnData(
        X=adata.raw.X,
        obs=adata.obs.copy(),
        var=adata.raw.var.copy(),
    )
elif 'counts' in adata.layers:
    print("  Using adata.layers['counts']")
    adata_train = adata.copy()
else:
    print("  WARNING: No raw counts found, using X (may be normalized)")
    adata_train = adata.copy()

# Use batch_key and labels_key or fail with helpful error
if batch_key is None:
    print("\nAvailable obs columns for manual selection:")
    for col in adata.obs.columns:
        print(f"  {col}: {adata.obs[col].dtype}, {adata.obs[col].nunique()} unique")
    raise ValueError("Could not auto-detect batch_key. Please specify manually.")

if labels_key is None:
    print("\nAvailable obs columns for manual selection:")
    for col in adata.obs.columns:
        print(f"  {col}: {adata.obs[col].dtype}, {adata.obs[col].nunique()} unique")
    raise ValueError("Could not auto-detect labels_key. Please specify manually.")

print(f"\n>>> Using batch_key='{batch_key}', labels_key='{labels_key}'")
print(f">>> Using n_latent={original_n_latent}")

# Filter genes if needed (highly variable)
print("\n" + "=" * 60)
print("Step 2: Prepare data for training")
print("=" * 60)

# Check if HVG already computed
if 'highly_variable' in adata_train.var.columns:
    n_hvg = adata_train.var['highly_variable'].sum()
    print(f"Using existing HVG selection: {n_hvg} genes")
    adata_train = adata_train[:, adata_train.var['highly_variable']].copy()
else:
    # Compute HVG - EXACTLY as original LuCA (6000 genes, seurat_v3)
    print("Computing highly variable genes (matching original LuCA: 6000, seurat_v3)...")
    sc.pp.highly_variable_genes(
        adata_train,
        n_top_genes=6000,  # Original LuCA used 6000
        batch_key=batch_key,
        flavor='seurat_v3',  # Original LuCA used seurat_v3
        subset=True
    )
    print(f"Selected {adata_train.n_vars} HVG")

print(f"Training data shape: {adata_train.shape}")

# Setup scVI
print("\n" + "=" * 60)
print("Step 3: Train scVI base model (4 GPUs)")
print("=" * 60)

scvi.model.SCVI.setup_anndata(
    adata_train,
    batch_key=batch_key,
)

# Model architecture EXACTLY matching original LuCA (from build_atlas.config --arches_params)
# These are the scArches-optimized parameters used in the published model
vae = scvi.model.SCVI(
    adata_train,
    n_latent=original_n_latent,
    n_layers=2,                    # From arches_params
    use_layer_norm="both",         # From arches_params
    use_batch_norm="none",         # From arches_params
    encode_covariates=True,        # From arches_params
    dropout_rate=0.2,              # From arches_params
    gene_likelihood="nb",          # negative binomial (scvi default)
)

print(f"scVI model: {vae}")
print(f"Training on {adata_train.n_obs} cells, {adata_train.n_vars} genes")
print("Using scArches-optimized parameters from original LuCA build")

# Train with multi-GPU DDP
# Original used vae.train(use_gpu=True) with defaults, but we add early stopping for safety
vae.train(
    max_epochs=400,
    early_stopping=True,
    early_stopping_patience=30,
    early_stopping_monitor="elbo_validation",
    train_size=0.9,
    validation_size=0.1,
    batch_size=1024,
    accelerator="gpu",
    devices=4,
    strategy="ddp_find_unused_parameters_true",
    check_val_every_n_epoch=1,
)

print(f"scVI training complete. Best epoch: {vae.history['elbo_validation'].idxmin()}")

# Convert to scANVI
print("\n" + "=" * 60)
print("Step 4: Train scANVI with cell type labels (4 GPUs)")
print("=" * 60)

# Setup scANVI - MUST use "unknown" (lowercase) to match original LuCA
scvi.model.SCANVI.setup_anndata(
    adata_train,
    batch_key=batch_key,
    labels_key=labels_key,
    unlabeled_category="unknown",  # Original LuCA used lowercase
)

scanvi = scvi.model.SCANVI.from_scvi_model(
    vae,
    unlabeled_category="unknown",  # Must match setup
    labels_key=labels_key,
)

print(f"scANVI model: {scanvi}")

# Train scANVI
scanvi.train(
    max_epochs=200,
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_monitor="elbo_validation",
    train_size=0.9,
    validation_size=0.1,
    batch_size=1024,
    accelerator="gpu",
    devices=4,
    strategy="ddp_find_unused_parameters_true",
    check_val_every_n_epoch=1,
)

print(f"scANVI training complete. Best epoch: {scanvi.history['elbo_validation'].idxmin()}")

# Save model
print("\n" + "=" * 60)
print("Step 5: Save model and validate")
print("=" * 60)

model_path = output_dir / "scanvi_model"
scanvi.save(model_path, overwrite=True)
print(f"Saved model to: {model_path}")

# Get latent representation
print("Computing latent representation...")
latent = scanvi.get_latent_representation()
adata_train.obsm['X_scANVI_retrained'] = latent

# Validate quality with scIB metrics
print("\nValidating model quality with scIB benchmarks...")
print("(This is the gold standard for integration evaluation)")

from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scanpy as sc

# Sample for speed (50k cells is enough for reliable metrics)
n_sample = min(50000, len(latent))
np.random.seed(42)
idx = np.random.choice(len(latent), n_sample, replace=False)

# Create evaluation adata
adata_eval = adata_train[idx].copy()
adata_eval.obsm['X_retrained'] = latent[idx]

# Add original latent if available
has_original = 'X_scANVI' in adata.obsm
if has_original:
    adata_eval.obsm['X_original'] = adata.obsm['X_scANVI'][idx]

def compute_scib_metrics(adata_sub, embed_key, batch_key, label_key):
    """Compute scIB-style metrics for integration quality."""
    X = adata_sub.obsm[embed_key]
    batch = adata_sub.obs[batch_key].values
    labels = adata_sub.obs[label_key].values

    metrics = {}

    # === BATCH CORRECTION METRICS (higher = better mixing) ===

    # 1. Batch ASW (Average Silhouette Width) - inverted so higher = better
    batch_asw = silhouette_score(X, batch)
    metrics['batch_ASW'] = (1 - batch_asw) / 2  # Scale to [0,1], higher = better mixing

    # 2. Graph connectivity (batch) - are batches connected in kNN graph?
    nn = NearestNeighbors(n_neighbors=15).fit(X)
    knn_graph = nn.kneighbors_graph(X, mode='connectivity')

    # For each cell, what fraction of neighbors are from different batches?
    batch_codes = pd.Categorical(batch).codes
    n_cells = len(batch_codes)
    cross_batch_neighbors = 0
    total_neighbors = 0
    for i in range(n_cells):
        neighbors = knn_graph[i].nonzero()[1]
        same_batch = (batch_codes[neighbors] == batch_codes[i]).sum()
        cross_batch_neighbors += len(neighbors) - same_batch
        total_neighbors += len(neighbors)
    metrics['batch_kNN_mixing'] = cross_batch_neighbors / total_neighbors

    # === BIOLOGICAL CONSERVATION METRICS (higher = better separation) ===

    # 3. Cell type ASW - higher = better cell type separation
    label_asw = silhouette_score(X, labels)
    metrics['celltype_ASW'] = (label_asw + 1) / 2  # Scale to [0,1]

    # 4. NMI (Normalized Mutual Information) with Leiden clustering
    # Cluster the embedding and compare to true labels
    adata_temp = ad.AnnData(X=X, obs=adata_sub.obs.copy())
    sc.pp.neighbors(adata_temp, use_rep='X', n_neighbors=15)
    sc.tl.leiden(adata_temp, resolution=1.0, key_added='leiden')
    metrics['NMI'] = normalized_mutual_info_score(labels, adata_temp.obs['leiden'])

    # 5. ARI (Adjusted Rand Index) with Leiden clustering
    metrics['ARI'] = adjusted_rand_score(labels, adata_temp.obs['leiden'])

    # 6. Cell type kNN purity - what fraction of neighbors share cell type?
    same_label_neighbors = 0
    total_neighbors = 0
    label_codes = pd.Categorical(labels).codes
    for i in range(n_cells):
        neighbors = knn_graph[i].nonzero()[1]
        same_label = (label_codes[neighbors] == label_codes[i]).sum()
        same_label_neighbors += same_label
        total_neighbors += len(neighbors)
    metrics['celltype_kNN_purity'] = same_label_neighbors / total_neighbors

    # === OVERALL SCORE ===
    # scIB uses weighted average: 40% batch correction, 60% bio conservation
    batch_score = (metrics['batch_ASW'] + metrics['batch_kNN_mixing']) / 2
    bio_score = (metrics['celltype_ASW'] + metrics['NMI'] + metrics['ARI'] + metrics['celltype_kNN_purity']) / 4
    metrics['overall_score'] = 0.4 * batch_score + 0.6 * bio_score

    return metrics

print("\nComputing metrics for RETRAINED model...")
retrained_metrics = compute_scib_metrics(adata_eval, 'X_retrained', batch_key, labels_key)

print("\n" + "=" * 50)
print("RETRAINED MODEL METRICS")
print("=" * 50)
print(f"  Batch ASW:           {retrained_metrics['batch_ASW']:.4f}  (higher = better batch mixing)")
print(f"  Batch kNN mixing:    {retrained_metrics['batch_kNN_mixing']:.4f}  (higher = better)")
print(f"  Cell type ASW:       {retrained_metrics['celltype_ASW']:.4f}  (higher = better separation)")
print(f"  NMI:                 {retrained_metrics['NMI']:.4f}  (higher = better)")
print(f"  ARI:                 {retrained_metrics['ARI']:.4f}  (higher = better)")
print(f"  Cell type kNN purity:{retrained_metrics['celltype_kNN_purity']:.4f}  (higher = better)")
print("-" * 50)
print(f"  OVERALL SCORE:       {retrained_metrics['overall_score']:.4f}")

if has_original:
    print("\nComputing metrics for ORIGINAL model...")
    original_metrics = compute_scib_metrics(adata_eval, 'X_original', batch_key, labels_key)

    print("\n" + "=" * 50)
    print("ORIGINAL MODEL METRICS")
    print("=" * 50)
    print(f"  Batch ASW:           {original_metrics['batch_ASW']:.4f}")
    print(f"  Batch kNN mixing:    {original_metrics['batch_kNN_mixing']:.4f}")
    print(f"  Cell type ASW:       {original_metrics['celltype_ASW']:.4f}")
    print(f"  NMI:                 {original_metrics['NMI']:.4f}")
    print(f"  ARI:                 {original_metrics['ARI']:.4f}")
    print(f"  Cell type kNN purity:{original_metrics['celltype_kNN_purity']:.4f}")
    print("-" * 50)
    print(f"  OVERALL SCORE:       {original_metrics['overall_score']:.4f}")

    print("\n" + "=" * 50)
    print("COMPARISON: RETRAINED vs ORIGINAL")
    print("=" * 50)
    for key in retrained_metrics:
        diff = retrained_metrics[key] - original_metrics[key]
        symbol = "✓" if diff >= -0.01 else "✗"  # Allow 1% tolerance
        direction = "+" if diff > 0 else ""
        print(f"  {key:22s}: {direction}{diff:.4f}  {symbol}")

    overall_diff = retrained_metrics['overall_score'] - original_metrics['overall_score']
    if overall_diff >= -0.01:
        print("\n>>> PASS: Retrained model is comparable or better than original!")
    else:
        print(f"\n>>> WARNING: Retrained model underperforms by {-overall_diff:.4f}")
        print("    Consider: more epochs, different hyperparameters, or checking data quality")

    # Store comparison
    comparison = {
        'retrained': retrained_metrics,
        'original': original_metrics,
        'difference': {k: retrained_metrics[k] - original_metrics[k] for k in retrained_metrics},
        'verdict': 'PASS' if overall_diff >= -0.01 else 'NEEDS_REVIEW'
    }
else:
    print("\n>>> No original X_scANVI found - cannot compare to original model")
    print("    Metrics look reasonable if:")
    print("    - overall_score > 0.5")
    print("    - celltype_ASW > 0.3")
    print("    - NMI > 0.4")
    comparison = {'retrained': retrained_metrics, 'original': None}

# Save metadata with full benchmark results
metadata = {
    "trained_at": datetime.now().isoformat(),
    "n_cells": int(adata_train.n_obs),
    "n_genes": int(adata_train.n_vars),
    "n_latent": original_n_latent,
    "batch_key": batch_key,
    "labels_key": labels_key,
    "n_batches": int(adata_train.obs[batch_key].nunique()),
    "n_cell_types": int(adata_train.obs[labels_key].nunique()),
    "scvi_epochs": len(vae.history['elbo_train']),
    "scanvi_epochs": len(scanvi.history['elbo_train']),
    "scib_metrics": {k: float(v) for k, v in retrained_metrics.items()},
    "comparison_to_original": comparison if has_original else None,
    "verdict": comparison.get('verdict', 'NO_ORIGINAL_TO_COMPARE'),
    "environment": {
        "scvi_version": scvi.__version__,
        "pandas_version": pd.__version__,
        "anndata_version": ad.__version__,
    }
}

# Save gene list for scArches compatibility
gene_list = adata_train.var_names.tolist()
with open(output_dir / "gene_list.json", "w") as f:
    json.dump(gene_list, f)
print(f"Saved gene list ({len(gene_list)} genes) for scArches compatibility")

with open(output_dir / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved metadata to: {output_dir / 'model_metadata.json'}")

# Save training history and comparison plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure 1: Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(vae.history['elbo_train'], label='Train')
ax.plot(vae.history['elbo_validation'], label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('ELBO')
ax.set_title('scVI Training')
ax.legend()

ax = axes[1]
ax.plot(scanvi.history['elbo_train'], label='Train')
ax.plot(scanvi.history['elbo_validation'], label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('ELBO')
ax.set_title('scANVI Training')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'training_curves.png', dpi=150)
plt.close()
print(f"Saved training curves to: {output_dir / 'training_curves.png'}")

# Figure 2: scIB metrics comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
metric_names = ['batch_ASW', 'batch_kNN_mixing', 'celltype_ASW', 'NMI', 'ARI', 'celltype_kNN_purity', 'overall_score']
x = np.arange(len(metric_names))
width = 0.35

retrained_vals = [retrained_metrics[m] for m in metric_names]
bars1 = ax.bar(x - width/2 if has_original else x, retrained_vals, width, label='Retrained', color='steelblue')

if has_original:
    original_vals = [original_metrics[m] for m in metric_names]
    bars2 = ax.bar(x + width/2, original_vals, width, label='Original', color='coral')

ax.set_ylabel('Score (higher = better)')
ax.set_title('scIB Benchmark: Retrained vs Original Model')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'scib_comparison.png', dpi=150)
plt.close()
print(f"Saved scIB comparison to: {output_dir / 'scib_comparison.png'}")

# Figure 3: UMAP comparison (if original exists)
if has_original:
    print("Generating UMAP comparison (original vs retrained)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subsample for UMAP
    n_umap = min(20000, len(adata_eval))
    umap_idx = np.random.choice(len(adata_eval), n_umap, replace=False)
    adata_umap = adata_eval[umap_idx].copy()

    # UMAP on original
    sc.pp.neighbors(adata_umap, use_rep='X_original', n_neighbors=15, key_added='original')
    sc.tl.umap(adata_umap, neighbors_key='original')
    adata_umap.obsm['X_umap_original'] = adata_umap.obsm['X_umap'].copy()

    # UMAP on retrained
    sc.pp.neighbors(adata_umap, use_rep='X_retrained', n_neighbors=15, key_added='retrained')
    sc.tl.umap(adata_umap, neighbors_key='retrained')
    adata_umap.obsm['X_umap_retrained'] = adata_umap.obsm['X_umap'].copy()

    # Plot original
    ax = axes[0]
    for ct in adata_umap.obs[labels_key].unique():
        mask = adata_umap.obs[labels_key] == ct
        ax.scatter(adata_umap.obsm['X_umap_original'][mask, 0],
                   adata_umap.obsm['X_umap_original'][mask, 1],
                   s=1, alpha=0.3, label=ct if mask.sum() > 100 else None)
    ax.set_title(f'Original Model (score: {original_metrics["overall_score"]:.3f})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    # Plot retrained
    ax = axes[1]
    for ct in adata_umap.obs[labels_key].unique():
        mask = adata_umap.obs[labels_key] == ct
        ax.scatter(adata_umap.obsm['X_umap_retrained'][mask, 0],
                   adata_umap.obsm['X_umap_retrained'][mask, 1],
                   s=1, alpha=0.3, label=ct if mask.sum() > 100 else None)
    ax.set_title(f'Retrained Model (score: {retrained_metrics["overall_score"]:.3f})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    plt.tight_layout()
    plt.savefig(output_dir / 'umap_comparison.png', dpi=150)
    plt.close()
    print(f"Saved UMAP comparison to: {output_dir / 'umap_comparison.png'}")

print("\n" + "=" * 60)
print("DONE! Model ready for scArches surgery.")
print("=" * 60)
print(f"\nModel saved to: {model_path}")
print(f"\nTo use this model for reference mapping (scArches surgery):")
print("""
# Load model
scanvi_ref = scvi.model.SCANVI.load(model_path, adata_ref)

# Load query (must have same genes - see gene_list.json)
vae_q = scvi.model.SCANVI.load_query_data(query_hvg, scanvi_ref)

# Train with original LuCA scArches parameters
vae_q.train(
    max_epochs=100,
    plan_kwargs=dict(weight_decay=0.0),
    check_val_every_n_epoch=10,
)

# Get latent
query.obsm['X_scANVI'] = vae_q.get_latent_representation()
""")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "LuCA Model Retraining Complete"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Model saved to: $DATA/references/luca/retrained_model/"
ls -la $DATA/references/luca/retrained_model/
