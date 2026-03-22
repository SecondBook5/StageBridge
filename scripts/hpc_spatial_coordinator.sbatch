#!/bin/bash
#SBATCH --job-name=spatial_coord
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/spatial_coordinator_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/spatial_coordinator_%j.err

# =============================================================================
# Spatial Backend Coordinator
# =============================================================================
#
# Orchestrates parallel execution of all 4 spatial backends:
#   1. Tangram  (GPU, ~2-4 hours)
#   2. DestVI   (GPU, ~4-6 hours)
#   3. TACCO    (GPU, ~1-2 hours)
#   4. Cell2location (GPU, ~6-12 hours)
#
# All 4 backends run IN PARALLEL on separate GPUs.
# After all complete, runs comparison to select canonical backend.
#
# Prerequisites:
#   - snRNA with cell_type column (from reference mapping)
#   - Spatial data with coordinates
#
# Usage:
#   # With dependency on reference mapping:
#   JOB_REF=$(sbatch --parsable scripts/hpc_step2_luca_mapping.sbatch)
#   sbatch --dependency=afterok:$JOB_REF scripts/hpc_spatial_coordinator.sbatch
#
#   # Or standalone (if reference mapping already done):
#   sbatch scripts/hpc_spatial_coordinator.sbatch
#
# Outputs:
#   - $DATA/runs/spatial_benchmark/{tangram,destvi,tacco,cell2location}/
#   - $DATA/runs/spatial_benchmark/backend_comparison.json
#   - $DATA/runs/spatial_benchmark/backend_selection_report.md
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Spatial Backend Coordinator"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge || cd $HOME/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
export DATA="${DATA:-/scratch/chaunzt1/stagebridge}"
export SNRNA="$DATA/processed/luad_evo/snrna_qc_normalized.h5ad"
export SPATIAL="$DATA/processed/luad_evo/spatial_merged.h5ad"
OUTPUT_BASE="$DATA/runs/spatial_benchmark"
SCRIPTS_DIR="/home/booka/StageBridge/scripts"
LOGS_DIR="/home/booka/StageBridge/logs"

mkdir -p "$OUTPUT_BASE" "$LOGS_DIR"

echo ""
echo "Configuration:"
echo "  DATA: $DATA"
echo "  snRNA: $SNRNA"
echo "  Spatial: $SPATIAL"
echo "  Output: $OUTPUT_BASE"
echo ""

# =============================================================================
# Prerequisite check: Verify cell_type column exists
# =============================================================================
echo "Checking prerequisites..."

if [ ! -f "$SNRNA" ]; then
    echo "ERROR: snRNA file not found: $SNRNA"
    echo "  Run data preparation first (run_data_prep.py)"
    exit 1
fi

if [ ! -f "$SPATIAL" ]; then
    echo "ERROR: Spatial file not found: $SPATIAL"
    echo "  Run data preparation first (run_data_prep.py)"
    exit 1
fi

# Check cell_type column
echo "Verifying cell_type column in snRNA..."
python3 << 'PYTHON_CHECK'
import sys
import anndata as ad
import os

snrna_path = os.environ.get('SNRNA')

try:
    adata = ad.read_h5ad(snrna_path, backed='r')

    if 'cell_type' not in adata.obs.columns:
        print("")
        print("=" * 60)
        print("ERROR: cell_type column NOT FOUND in snRNA")
        print("=" * 60)
        print("")
        print("The spatial backends require cell_type annotations from")
        print("reference mapping. Please run reference mapping first:")
        print("")
        print("  sbatch scripts/hpc_step1_hlca_mapping.sbatch")
        print("  sbatch scripts/hpc_step2_luca_mapping.sbatch")
        print("")
        print("Or use --dependency to wait for reference mapping:")
        print("  JOB_REF=<ref_mapping_job_id>")
        print("  sbatch --dependency=afterok:$JOB_REF scripts/hpc_spatial_coordinator.sbatch")
        print("")
        adata.file.close()
        sys.exit(1)

    n_celltypes = adata.obs['cell_type'].nunique()
    n_cells = adata.n_obs
    n_nan = adata.obs['cell_type'].isna().sum()

    print(f"  OK: cell_type found")
    print(f"      {n_cells:,} cells, {n_celltypes} cell types")
    if n_nan > 0:
        print(f"      WARNING: {n_nan:,} cells with NaN cell_type")

    adata.file.close()
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYTHON_CHECK

PREREQ_CHECK=$?
if [ $PREREQ_CHECK -ne 0 ]; then
    exit 1
fi

echo ""
echo "All prerequisites satisfied."
echo ""

# =============================================================================
# Check which backends already completed (skip-if-exists)
# =============================================================================
echo "Checking existing backend results..."

BACKENDS_TO_RUN=()
BACKENDS_DONE=()

for backend in tangram destvi tacco cell2location; do
    MARKER="$OUTPUT_BASE/$backend/upstream_metrics.json"
    if [ -f "$MARKER" ]; then
        echo "  $backend: DONE (found upstream_metrics.json)"
        BACKENDS_DONE+=("$backend")
    else
        echo "  $backend: PENDING"
        BACKENDS_TO_RUN+=("$backend")
    fi
done

echo ""

if [ ${#BACKENDS_TO_RUN[@]} -eq 0 ]; then
    echo "=============================================="
    echo "All backends already completed!"
    echo "=============================================="
    echo ""
    echo "Skipping to comparison step..."
    echo ""
else
    echo "Will run: ${BACKENDS_TO_RUN[*]}"
    echo ""
fi

# =============================================================================
# Submit backend jobs in parallel
# =============================================================================
declare -A JOB_IDS

if [[ " ${BACKENDS_TO_RUN[*]} " =~ " tangram " ]]; then
    echo "Submitting Tangram..."
    JOB_IDS[tangram]=$(sbatch --parsable "$SCRIPTS_DIR/hpc_spatial_tangram.sbatch")
    echo "  Job ID: ${JOB_IDS[tangram]}"
fi

if [[ " ${BACKENDS_TO_RUN[*]} " =~ " destvi " ]]; then
    echo "Submitting DestVI..."
    JOB_IDS[destvi]=$(sbatch --parsable "$SCRIPTS_DIR/hpc_spatial_destvi.sbatch")
    echo "  Job ID: ${JOB_IDS[destvi]}"
fi

if [[ " ${BACKENDS_TO_RUN[*]} " =~ " tacco " ]]; then
    echo "Submitting TACCO..."
    JOB_IDS[tacco]=$(sbatch --parsable "$SCRIPTS_DIR/hpc_spatial_tacco.sbatch")
    echo "  Job ID: ${JOB_IDS[tacco]}"
fi

if [[ " ${BACKENDS_TO_RUN[*]} " =~ " cell2location " ]]; then
    echo "Submitting Cell2location..."
    JOB_IDS[cell2location]=$(sbatch --parsable "$SCRIPTS_DIR/hpc_spatial_cell2location.sbatch")
    echo "  Job ID: ${JOB_IDS[cell2location]}"
fi

echo ""

# =============================================================================
# Submit comparison job with dependency on all backends
# =============================================================================
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    # Build dependency string
    DEP_STRING=""
    for job_id in "${JOB_IDS[@]}"; do
        if [ -z "$DEP_STRING" ]; then
            DEP_STRING="afterany:$job_id"
        else
            DEP_STRING="$DEP_STRING:$job_id"
        fi
    done

    echo "Submitting comparison job (depends on: $DEP_STRING)..."
    JOB_COMPARE=$(sbatch --parsable --dependency="$DEP_STRING" "$SCRIPTS_DIR/hpc_spatial_compare.sbatch")
    echo "  Job ID: $JOB_COMPARE"
else
    # All backends done, run comparison immediately
    echo "Running comparison immediately (all backends done)..."
    sbatch "$SCRIPTS_DIR/hpc_spatial_compare.sbatch"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "Spatial Backend Jobs Submitted"
echo "=============================================="
echo ""
echo "Backend Jobs (running in parallel):"
for backend in tangram destvi tacco cell2location; do
    if [[ -v "JOB_IDS[$backend]" ]]; then
        echo "  $backend: ${JOB_IDS[$backend]}"
    elif [[ " ${BACKENDS_DONE[*]} " =~ " $backend " ]]; then
        echo "  $backend: ALREADY DONE"
    fi
done
echo ""
echo "Comparison job: Will run after all backends complete"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 60 'squeue -u \$USER'"
echo ""
echo "Check logs:"
echo "  tail -f $LOGS_DIR/spatial_tangram_*.log"
echo "  tail -f $LOGS_DIR/spatial_destvi_*.log"
echo "  tail -f $LOGS_DIR/spatial_tacco_*.log"
echo "  tail -f $LOGS_DIR/spatial_cell2location_*.log"
echo ""

# Save job IDs for reference
cat > "$OUTPUT_BASE/current_jobs.txt" << EOF
# Spatial Backend Jobs - $(date)
# Run by: hpc_spatial_coordinator.sbatch
EOF

for backend in tangram destvi tacco cell2location; do
    if [[ -v "JOB_IDS[$backend]" ]]; then
        echo "JOB_${backend^^}=${JOB_IDS[$backend]}" >> "$OUTPUT_BASE/current_jobs.txt"
    fi
done

echo ""
echo "Job IDs saved to: $OUTPUT_BASE/current_jobs.txt"
echo ""
echo "Coordinator complete: $(date)"
