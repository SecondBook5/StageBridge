#!/bin/bash
#SBATCH --job-name=liana_luca
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=/home/booka/StageBridge/logs/liana_luca_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/liana_luca_%j.err

# Complete LIANA pipeline with LuCA cell types (33 types)
# 1. Transfer LuCA labels to snRNA via k-NN
# 2. Run LIANA with 33 cell types
# 3. Save results for local figure generation

set -e

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

# Paths
DATA="/data1/chaunzt1/stagebridge/processed/luad_evo"
SNRNA="$DATA/snrna_with_celltypes.h5ad"
OUTPUT="$DATA/canonical/liana_luca"
LOCAL_COPY="/home/booka/StageBridge/data"

mkdir -p $OUTPUT
mkdir -p $LOCAL_COPY
mkdir -p /home/booka/StageBridge/logs

echo ""
echo "=============================================="
echo "LIANA with LuCA Cell Types (33 types)"
echo "=============================================="
echo "  Input: $SNRNA"
echo "  Output: $OUTPUT"
echo ""

python << 'PYTHON_SCRIPT'
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SNRNA = "/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad"
LUCA_REF = "/scratch/chaunzt1/stagebridge/references/luca/luca_core_atlas.h5ad"
OUTPUT = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/canonical/liana_luca")
LOCAL = Path("/home/booka/StageBridge/data")

print("=" * 60)
print("Step 1: Load snRNA data")
print("=" * 60)
sys.stdout.flush()

adata = sc.read_h5ad(SNRNA)
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Existing cell type column: {adata.obs['cell_type'].nunique()} types (HLCA)")
sys.stdout.flush()

print("\n" + "=" * 60)
print("Step 2: Transfer LuCA labels via k-NN")
print("=" * 60)
sys.stdout.flush()

# Load LuCA reference
print("Loading LuCA reference atlas...")
sys.stdout.flush()
luca_ref = sc.read_h5ad(LUCA_REF)
print(f"  Reference: {luca_ref.n_obs:,} cells")

# Get reference latent and labels
latent_key = "X_scANVI" if "X_scANVI" in luca_ref.obsm else "X_scVI"
ref_latent = luca_ref.obsm[latent_key]
ref_labels = luca_ref.obs["cell_type"].values
print(f"  Latent: {latent_key} ({ref_latent.shape[1]}d)")
print(f"  Cell types: {np.unique(ref_labels).shape[0]}")
sys.stdout.flush()

# Project query to reference scANVI space via shared genes + PCA
print("\nProjecting query cells to reference latent space...")
sys.stdout.flush()
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import CCA

# Find shared genes
shared_genes = sorted(set(adata.var_names) & set(luca_ref.var_names))
print(f"  Shared genes: {len(shared_genes)}")
sys.stdout.flush()

# Extract expression for shared genes from both
query_expr = adata[:, shared_genes].X
if hasattr(query_expr, "toarray"):
    query_expr = query_expr.toarray()
query_expr = np.log1p(query_expr).astype(np.float32)

ref_expr = luca_ref[:, shared_genes].X
if hasattr(ref_expr, "toarray"):
    ref_expr = ref_expr.toarray()
ref_expr = np.log1p(ref_expr).astype(np.float32)

# Learn mapping from expression -> scANVI latent using reference
# Subsample reference for faster fitting
print("  Learning expression -> latent mapping (subsampled)...")
sys.stdout.flush()
n_subsample = min(50000, ref_expr.shape[0])
subsample_idx = np.random.choice(ref_expr.shape[0], n_subsample, replace=False)
ref_expr_sub = ref_expr[subsample_idx]
ref_latent_sub = ref_latent[subsample_idx]

# Use ridge regression to map expression to latent
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(ref_expr_sub, ref_latent_sub)
query_latent = ridge.predict(query_expr)
print(f"  Query latent: {query_latent.shape}")
print(f"  Reference latent: {ref_latent.shape}")
sys.stdout.flush()

# k-NN label transfer
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

# Clean up reference to save memory
del luca_ref, ref_latent, ref_labels, query_latent
import gc
gc.collect()

print("\n" + "=" * 60)
print("Step 3: Run LIANA with LuCA cell types")
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
print("Step 4: Save results")
print("=" * 60)
sys.stdout.flush()

# Save LIANA results
liana_out = OUTPUT / "liana_interactions_luca.parquet"
liana_results.to_parquet(liana_out)
print(f"Saved: {liana_out}")

# Save cell type mapping
celltypes_out = OUTPUT / "cell_type_luca_mapping.parquet"
adata.obs[["cell_type", "cell_type_luca", "cell_type_luca_confidence"]].to_parquet(celltypes_out)
print(f"Saved: {celltypes_out}")

# Save summary stats
summary = {
    "n_cells": int(adata.n_obs),
    "n_luca_types": int(adata.obs["cell_type_luca"].nunique()),
    "n_hlca_types": int(adata.obs["cell_type"].nunique()),
    "n_interactions": int(len(liana_results)),
    "n_sources": int(liana_results["source"].nunique()),
    "n_targets": int(liana_results["target"].nunique()),
    "top_interactions": liana_results.nsmallest(20, "magnitude_rank")[
        ["source", "target", "ligand_complex", "receptor_complex", "magnitude_rank"]
    ].to_dict("records"),
}

import json
with open(OUTPUT / "liana_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUTPUT / 'liana_summary.json'}")

# Copy to local for figures
print("\nCopying to local...")
import shutil
shutil.copy(liana_out, LOCAL / "liana_interactions_luca.parquet")
print(f"  -> {LOCAL / 'liana_interactions_luca.parquet'}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
print(f"\nTo generate figures locally:")
print(f"  python scripts/generate_liana_figures.py --input data/liana_interactions_luca.parquet")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "Complete: $(date)"
echo "=============================================="
