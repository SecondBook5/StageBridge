#!/bin/bash
#SBATCH --job-name=spatial_tangram
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/spatial_tangram_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/spatial_tangram_%j.err

# =============================================================================
# Tangram Spatial Backend
# =============================================================================
#
# Runs Tangram deconvolution on spatial data using snRNA reference.
# Part of parallel spatial backend benchmark (can run alongside other backends).
#
# Prerequisites:
#   - snRNA data with cell_type column (from reference mapping)
#   - Spatial data with spatial coordinates
#
# Outputs:
#   - cell_type_proportions.parquet
#   - mapping_confidence.parquet
#   - upstream_metrics.json
#   - backend_metadata.json
#
# Skip-if-exists: Will skip if upstream_metrics.json already exists
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Tangram Spatial Backend"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge || cd $HOME/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA="${DATA:-/scratch/chaunzt1/stagebridge}"
SNRNA="$DATA/processed/luad_evo/snrna_qc_normalized.h5ad"
SPATIAL="$DATA/processed/luad_evo/spatial_merged.h5ad"
OUTPUT_DIR="$DATA/runs/spatial_benchmark/tangram"

# Reference geometry (for cell_type check)
REF_GEOMETRY_DIR="$DATA/processed/luad_evo/reference_geometry"

# GPU setup
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  DATA: $DATA"
echo "  snRNA: $SNRNA"
echo "  Spatial: $SPATIAL"
echo "  Output: $OUTPUT_DIR"
echo "  Reference Geometry: $REF_GEOMETRY_DIR"
echo ""

# =============================================================================
# Skip-if-exists check
# =============================================================================
COMPLETION_MARKER="$OUTPUT_DIR/upstream_metrics.json"
if [ -f "$COMPLETION_MARKER" ]; then
    echo "=============================================="
    echo "SKIPPING: Tangram already completed"
    echo "  Found: $COMPLETION_MARKER"
    echo "  To re-run, delete: $OUTPUT_DIR"
    echo "=============================================="

    # Show existing metrics
    echo ""
    echo "Existing metrics:"
    cat "$COMPLETION_MARKER"
    exit 0
fi

# =============================================================================
# Prerequisite checks
# =============================================================================
echo "Checking prerequisites..."

# Check snRNA file exists
if [ ! -f "$SNRNA" ]; then
    echo "ERROR: snRNA file not found: $SNRNA"
    exit 1
fi

# Check spatial file exists
if [ ! -f "$SPATIAL" ]; then
    echo "ERROR: Spatial file not found: $SPATIAL"
    exit 1
fi

# Check for cell_type column in snRNA (required for spatial deconvolution)
echo "Checking for cell_type column in snRNA..."
python3 << 'PYTHON_CHECK'
import sys
import anndata as ad
import os

snrna_path = os.environ.get('SNRNA', '/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_qc_normalized.h5ad')

try:
    # Read just the obs to check columns (memory efficient)
    adata = ad.read_h5ad(snrna_path, backed='r')

    if 'cell_type' not in adata.obs.columns:
        print("ERROR: snRNA data missing 'cell_type' column")
        print("  Available columns:", list(adata.obs.columns)[:10], "...")
        print("")
        print("  cell_type is added by reference mapping (run_reference.py)")
        print("  Please ensure reference mapping completed successfully before")
        print("  running spatial backends.")
        sys.exit(1)

    n_celltypes = adata.obs['cell_type'].nunique()
    print(f"  cell_type column found: {n_celltypes} unique cell types")

    # Check for NaN cell types
    n_nan = adata.obs['cell_type'].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} cells have NaN cell_type")

    adata.file.close()
    sys.exit(0)

except Exception as e:
    print(f"ERROR checking snRNA: {e}")
    sys.exit(1)
PYTHON_CHECK

CELL_TYPE_CHECK=$?
if [ $CELL_TYPE_CHECK -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "FAILED: cell_type column not found in snRNA"
    echo "=============================================="
    echo ""
    echo "The spatial backends require cell_type annotations from"
    echo "reference mapping. Please ensure these steps completed:"
    echo ""
    echo "  1. HLCA mapping: hpc_step1_hlca_mapping.sbatch"
    echo "  2. LuCA mapping: hpc_step2_luca_mapping.sbatch"
    echo ""
    echo "Or check if cell_types.parquet exists in reference_geometry:"
    echo "  ls -la $REF_GEOMETRY_DIR/"
    exit 1
fi

# Show GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "  No GPU available"
echo ""

# =============================================================================
# Run Tangram
# =============================================================================
mkdir -p "$OUTPUT_DIR"

echo "Running Tangram..."
echo ""

python3 << 'PYTHON_RUN'
import os
import sys
import json
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# Get paths from environment
DATA = os.environ.get('DATA', '/scratch/chaunzt1/stagebridge')
SNRNA = os.environ.get('SNRNA', f'{DATA}/processed/luad_evo/snrna_qc_normalized.h5ad')
SPATIAL = os.environ.get('SPATIAL', f'{DATA}/processed/luad_evo/spatial_merged.h5ad')
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', f'{DATA}/runs/spatial_benchmark/tangram'))

start_time = time.time()

try:
    from stagebridge.spatial_mapping import TangramBackend

    print("Loading data...")
    snrna = ad.read_h5ad(SNRNA)
    spatial = ad.read_h5ad(SPATIAL)

    print(f"  snRNA: {snrna.shape[0]:,} cells x {snrna.shape[1]:,} genes")
    print(f"  Spatial: {spatial.shape[0]:,} spots x {spatial.shape[1]:,} genes")
    print(f"  Cell types: {snrna.obs['cell_type'].nunique()}")
    print("")

    # Initialize and run Tangram
    print("Initializing Tangram backend...")
    backend = TangramBackend(
        mode='clusters',  # Map at cluster level for robustness
        n_epochs=1000,
        device='cuda',
    )

    print("Running Tangram mapping...")
    result = backend.map(snrna, spatial, output_dir=OUTPUT_DIR)

    # Save results
    print("")
    print("Saving results...")
    result.save(OUTPUT_DIR)

    runtime = time.time() - start_time

    # Add runtime to metrics
    metrics = result.upstream_metrics.copy()
    metrics['runtime_seconds'] = runtime
    metrics['runtime_minutes'] = runtime / 60

    with open(OUTPUT_DIR / 'upstream_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("")
    print("=" * 60)
    print("Tangram completed successfully!")
    print("=" * 60)
    print(f"  Runtime: {runtime/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    print("")
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    sys.exit(0)

except Exception as e:
    runtime = time.time() - start_time

    print("")
    print("=" * 60)
    print("ERROR: Tangram failed")
    print("=" * 60)
    print(f"  Error: {e}")
    print(f"  Runtime before failure: {runtime/60:.1f} minutes")
    print("")
    traceback.print_exc()

    # Save failure log
    failure_log = {
        'status': 'failed',
        'error': str(e),
        'traceback': traceback.format_exc(),
        'runtime_seconds': runtime,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'failure_log.json', 'w') as f:
        json.dump(failure_log, f, indent=2)

    sys.exit(1)
PYTHON_RUN

TANGRAM_EXIT=$?

echo ""
echo "=============================================="
echo "Tangram Backend Complete: $(date)"
echo "Exit code: $TANGRAM_EXIT"
echo "=============================================="

# List outputs
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/ 2>/dev/null || echo "  (no output directory)"

exit $TANGRAM_EXIT
