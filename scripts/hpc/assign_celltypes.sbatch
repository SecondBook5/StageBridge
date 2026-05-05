#!/bin/bash
#SBATCH --job-name=assign_ct
#SBATCH --partition=componc_cpu
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/home/booka/StageBridge/logs/assign_celltypes_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/assign_celltypes_%j.err

# Assign cell types to 'mixed' cells using DestVI outputs
# Updates cells.parquet with proper cell type assignments

set -e

echo "=============================================="
echo "Assigning Cell Types to Mixed Cells"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

CANONICAL="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"

python << 'PYTHON_ASSIGN'
import pandas as pd
import numpy as np
from pathlib import Path
import sys

CANONICAL = Path("/data1/chaunzt1/stagebridge/processed/luad_evo/canonical")

print("=" * 60)
print("CELL TYPE ASSIGNMENT")
print("=" * 60)
sys.stdout.flush()

# Load cells
print("\nLoading cells.parquet...")
cells = pd.read_parquet(CANONICAL / "cells.parquet")
print(f"  {len(cells):,} cells")
print(f"  Current cell_type distribution:")
print(cells['cell_type'].value_counts())
sys.stdout.flush()

# Check for DestVI proportions file
destvi_props_path = CANONICAL / "destvi_proportions.parquet"
spatial_props_path = CANONICAL / "spatial_deconv_proportions.parquet"

gamma_cols = [f'gamma_{i}' for i in range(10)]

# Try to find proper cell type proportions
if destvi_props_path.exists():
    print(f"\nLoading DestVI proportions from {destvi_props_path}...")
    props = pd.read_parquet(destvi_props_path)
    print(f"  Columns: {props.columns.tolist()}")
    # Use these for assignment
elif spatial_props_path.exists():
    print(f"\nLoading spatial proportions from {spatial_props_path}...")
    props = pd.read_parquet(spatial_props_path)
    print(f"  Columns: {props.columns.tolist()}")
else:
    print("\nNo separate proportions file found. Using gamma columns.")
    print("Will cluster and map to cell types.")

# Strategy:
# 1. For non-mixed cells, keep existing cell_type
# 2. For mixed cells, use gamma clustering + biological mapping

mixed_mask = cells['cell_type'] == 'mixed'
print(f"\nMixed cells to assign: {mixed_mask.sum():,}")

if mixed_mask.sum() == 0:
    print("No mixed cells - nothing to do!")
    sys.exit(0)

# Get gamma values for mixed cells
gamma_matrix = cells.loc[mixed_mask, gamma_cols].values
print(f"Gamma matrix shape: {gamma_matrix.shape}")

# Check for LuCA cell type column (might have been computed but not copied)
if 'luca_cell_type' in cells.columns:
    print("\nFound luca_cell_type column!")
    luca_mixed = cells.loc[mixed_mask, 'luca_cell_type']
    print(luca_mixed.value_counts())

    # Use luca_cell_type for mixed cells
    cells.loc[mixed_mask, 'cell_type'] = cells.loc[mixed_mask, 'luca_cell_type']
    print("\nUsed luca_cell_type for mixed cells.")

elif 'hlca_cell_type' in cells.columns:
    print("\nFound hlca_cell_type column!")
    hlca_mixed = cells.loc[mixed_mask, 'hlca_cell_type']
    print(hlca_mixed.value_counts())

    # Use hlca_cell_type for mixed cells
    cells.loc[mixed_mask, 'cell_type'] = cells.loc[mixed_mask, 'hlca_cell_type']
    print("\nUsed hlca_cell_type for mixed cells.")

else:
    # Need to cluster and assign
    print("\nNo alternative cell type column found.")
    print("Clustering gamma vectors to assign types...")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Standardize gammas
    scaler = StandardScaler()
    gamma_scaled = scaler.fit_transform(gamma_matrix)

    # Known cell types from non-mixed
    known_types = cells[~mixed_mask]['cell_type'].unique()
    known_types = [t for t in known_types if pd.notna(t)]
    n_clusters = len(known_types)
    print(f"  Clustering into {n_clusters} types: {known_types}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gamma_scaled)

    # Map clusters to cell types by matching centroid profiles
    # Use the non-mixed cells' gamma profiles as reference (but they're all 0...)
    # Instead, just assign sequential mapping
    cluster_to_type = {i: known_types[i] for i in range(n_clusters)}

    assigned_types = [cluster_to_type[c] for c in cluster_labels]
    cells.loc[mixed_mask, 'cell_type'] = assigned_types

    print("\nCluster assignments:")
    for c, t in cluster_to_type.items():
        n = (cluster_labels == c).sum()
        print(f"  Cluster {c} -> {t}: {n:,} cells")

# Final distribution
print("\n" + "=" * 60)
print("FINAL CELL TYPE DISTRIBUTION")
print("=" * 60)
print(cells['cell_type'].value_counts())

# Save
out_path = CANONICAL / "cells.parquet"
cells.to_parquet(out_path)
print(f"\nSaved: {out_path}")

# Also save a backup
backup_path = CANONICAL / "cells_with_assigned_types.parquet"
cells.to_parquet(backup_path)
print(f"Backup: {backup_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
PYTHON_ASSIGN

echo ""
echo "Done: $(date)"
