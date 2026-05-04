#!/bin/bash
#SBATCH --job-name=liana_luca
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=/home/booka/StageBridge/logs/liana_luca_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/liana_luca_%j.err

# Complete LIANA pipeline with LuCA cell types (33 types)
# 1. Use pre-computed LuCA labels (luca_cell_type column)
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
OUTPUT = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/canonical/liana_luca")
LOCAL = Path("/home/booka/StageBridge/data")

print("=" * 60)
print("Step 1: Load snRNA data")
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
print("Step 2: Run LIANA with LuCA cell types")
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
print("Step 4: Save results")
print("=" * 60)
sys.stdout.flush()

# Save LIANA results
liana_out = OUTPUT / "liana_interactions_luca.parquet"
liana_results.to_parquet(liana_out)
print(f"Saved: {liana_out}")

# Save cell type mapping
celltypes_out = OUTPUT / "luca_cell_type_mapping.parquet"
adata.obs[["cell_type", "luca_cell_type"]].to_parquet(celltypes_out)
print(f"Saved: {celltypes_out}")

# Save summary stats
summary = {
    "n_cells": int(adata.n_obs),
    "n_luca_types": int(adata.obs["luca_cell_type"].nunique()),
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
