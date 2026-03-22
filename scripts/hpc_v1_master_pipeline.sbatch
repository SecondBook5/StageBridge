#!/bin/bash
#SBATCH --job-name=sb_master
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/master_pipeline_%j.log

# =============================================================================
# StageBridge V1 Master Pipeline Orchestrator
# =============================================================================
#
# This script submits all pipeline stages with proper SLURM dependencies.
# Each stage waits for its prerequisites to complete successfully.
#
# Execution Order:
# 1. HLCA Reference Mapping (--hlca-only)
# 2. LuCA Reference Mapping (--luca-only) [parallel with HLCA if separate env]
# 3. Spatial Backend Benchmark (Tangram, DestVI, TACCO, Cell2location)
# 4. Complete Data Preparation (canonical format)
# 5. Full Model Training (SSL + Transition)
# 6. Ablation Studies (5-fold CV)
# 7. Publication Figure Generation
#
# Usage:
#   sbatch scripts/hpc_v1_master_pipeline.sbatch
#
# Or run stages individually:
#   JOB1=$(sbatch --parsable scripts/hpc_step1_hlca_mapping.sbatch)
#   JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/hpc_step2_luca_mapping.sbatch)
#   ...
# =============================================================================

set -euo pipefail

# Environment setup
cd /home/booka/StageBridge || cd $HOME/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
export DATA=/scratch/chaunzt1/stagebridge
export LOGS_DIR=/home/booka/StageBridge/logs
export SCRIPTS_DIR=/home/booka/StageBridge/scripts

mkdir -p $LOGS_DIR

echo "============================================================"
echo "StageBridge V1 Master Pipeline Orchestrator"
echo "============================================================"
echo "Start time: $(date)"
echo "DATA: $DATA"
echo "LOGS: $LOGS_DIR"
echo ""

# Verify prerequisites exist
echo "Verifying prerequisites..."
REQUIRED_FILES=(
    "$DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad"
    "$DATA/processed/luad_evo/spatial_merged.h5ad"
    "$DATA/references/hlca/hlca_reference.h5ad"
    "$DATA/references/luca/luca_core_atlas.h5ad"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        echo "ERROR: Prerequisites not met. Run data preparation first."
        exit 1
    fi
done

echo ""
echo "Submitting pipeline stages..."
echo ""

# =============================================================================
# Stage 1: HLCA Reference Mapping
# =============================================================================
echo "Stage 1: HLCA Reference Mapping"
JOB_HLCA=$(sbatch --parsable $SCRIPTS_DIR/hpc_step1_hlca_mapping.sbatch)
echo "  Submitted job: $JOB_HLCA"

# =============================================================================
# Stage 2: LuCA Reference Mapping (depends on HLCA for fused output)
# =============================================================================
echo "Stage 2: LuCA Reference Mapping (after HLCA)"
JOB_LUCA=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_step2_luca_mapping.sbatch)
echo "  Submitted job: $JOB_LUCA (depends on $JOB_HLCA)"

# =============================================================================
# Stage 3: Spatial Backend Benchmark (4 backends in PARALLEL)
# Each backend runs on separate GPU, all wait for HLCA to have cell types
# =============================================================================
echo "Stage 3: Spatial Backend Benchmark (4 parallel backends after HLCA)"

# Check if individual backend scripts exist, otherwise use combined script
if [ -f "$SCRIPTS_DIR/hpc_spatial_tangram.sbatch" ]; then
    echo "  Using parallel backend execution..."
    JOB_TANGRAM=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_spatial_tangram.sbatch)
    JOB_DESTVI=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_spatial_destvi.sbatch)
    JOB_TACCO=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_spatial_tacco.sbatch)
    JOB_CELL2LOC=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_spatial_cell2location.sbatch)
    echo "    Tangram:      $JOB_TANGRAM"
    echo "    DestVI:       $JOB_DESTVI"
    echo "    TACCO:        $JOB_TACCO"
    echo "    Cell2location: $JOB_CELL2LOC"
    # Combined dependency for next stage
    JOB_SPATIAL="$JOB_TANGRAM:$JOB_DESTVI:$JOB_TACCO:$JOB_CELL2LOC"
else
    echo "  Using sequential spatial benchmark..."
    JOB_SPATIAL=$(sbatch --parsable --dependency=afterok:$JOB_HLCA $SCRIPTS_DIR/hpc_step3_spatial_benchmark.sbatch)
    echo "    Combined job: $JOB_SPATIAL"
fi

# =============================================================================
# Stage 4: Complete Data Preparation (depends on ref mapping AND spatial)
# =============================================================================
echo "Stage 4: Complete Data Preparation (after ref mapping + spatial)"
JOB_DATAPREP=$(sbatch --parsable --dependency=afterok:$JOB_LUCA:$JOB_SPATIAL $SCRIPTS_DIR/hpc_step4_complete_data_prep.sbatch)
echo "  Submitted job: $JOB_DATAPREP (depends on $JOB_LUCA, $JOB_SPATIAL)"

# =============================================================================
# Stage 5: Full Model Training (depends on data prep)
# =============================================================================
echo "Stage 5: Full Model Training (after data prep)"
JOB_TRAIN=$(sbatch --parsable --dependency=afterok:$JOB_DATAPREP $SCRIPTS_DIR/hpc_step5_training.sbatch)
echo "  Submitted job: $JOB_TRAIN (depends on $JOB_DATAPREP)"

# =============================================================================
# Stage 6: Ablation Studies (depends on training)
# =============================================================================
echo "Stage 6: Ablation Studies (after training)"
JOB_ABLATION=$(sbatch --parsable --dependency=afterok:$JOB_TRAIN $SCRIPTS_DIR/hpc_step6_ablations.sbatch)
echo "  Submitted job: $JOB_ABLATION (depends on $JOB_TRAIN)"

# =============================================================================
# Stage 7: Publication Figures (depends on ablations)
# =============================================================================
echo "Stage 7: Publication Figures (after ablations)"
JOB_FIGURES=$(sbatch --parsable --dependency=afterok:$JOB_ABLATION $SCRIPTS_DIR/hpc_step7_figures.sbatch)
echo "  Submitted job: $JOB_FIGURES (depends on $JOB_ABLATION)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Pipeline Submitted Successfully"
echo "============================================================"
echo ""
echo "Job Chain:"
echo "  1. HLCA Mapping:      $JOB_HLCA"
echo "  2. LuCA Mapping:      $JOB_LUCA (after $JOB_HLCA)"
echo "  3. Spatial Benchmark: $JOB_SPATIAL (parallel)"
echo "  4. Data Preparation:  $JOB_DATAPREP (after $JOB_LUCA, $JOB_SPATIAL)"
echo "  5. Model Training:    $JOB_TRAIN (after $JOB_DATAPREP)"
echo "  6. Ablation Studies:  $JOB_ABLATION (after $JOB_TRAIN)"
echo "  7. Publication Figs:  $JOB_FIGURES (after $JOB_ABLATION)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Check logs:"
echo "  tail -f $LOGS_DIR/hlca_mapping_\$JOB_HLCA.log"
echo ""
echo "Cancel all:"
echo "  scancel $JOB_HLCA $JOB_LUCA $JOB_SPATIAL $JOB_DATAPREP $JOB_TRAIN $JOB_ABLATION $JOB_FIGURES"
echo ""

# Save job IDs for later reference
cat > $DATA/runs/current_pipeline_jobs.txt << EOF
# StageBridge V1 Pipeline Jobs - $(date)
HLCA_MAPPING=$JOB_HLCA
LUCA_MAPPING=$JOB_LUCA
SPATIAL_BENCHMARK=$JOB_SPATIAL
DATA_PREP=$JOB_DATAPREP
TRAINING=$JOB_TRAIN
ABLATIONS=$JOB_ABLATION
FIGURES=$JOB_FIGURES
EOF

echo "Job IDs saved to: $DATA/runs/current_pipeline_jobs.txt"
echo ""
echo "Master orchestrator complete: $(date)"
