#!/bin/bash
#SBATCH --job-name=sb_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# StageBridge V1 Training
# =============================================================================
#
# Full training pipeline:
# 1. SSL Pretraining (receiver-centered niche learning)
# 2. Transition Modeling (flow matching)
#
# Uses single GPU (1 GPU is more efficient than DDP for this model size)
#
# Usage:
#   sbatch scripts/hpc/run_training.sbatch              # Fresh training
#   sbatch scripts/hpc/run_training.sbatch --resume     # Resume from checkpoint
#   sbatch scripts/hpc/run_training.sbatch --force      # Force retrain
#   sbatch scripts/hpc/run_training.sbatch --smoke      # Quick test
# =============================================================================

set -euo pipefail

# Parse command-line arguments
FORCE_RETRAIN=false
RESUME_MODE=false
SMOKE_MODE=false
FOLD=${FOLD:-0}
SEED=${SEED:-42}

for arg in "$@"; do
    case $arg in
        --force)
            FORCE_RETRAIN=true
            ;;
        --resume)
            RESUME_MODE=true
            ;;
        --smoke)
            SMOKE_MODE=true
            ;;
        --fold=*)
            FOLD="${arg#*=}"
            ;;
        --seed=*)
            SEED="${arg#*=}"
            ;;
    esac
done

echo "=============================================="
echo "StageBridge V1 Training"
echo "=============================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Fold: $FOLD, Seed: $SEED"
echo "Force retrain: $FORCE_RETRAIN"
echo "Resume mode: $RESUME_MODE"
echo "Smoke mode: $SMOKE_MODE"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

cd ~/StageBridge
mkdir -p logs

# Load environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

# Data paths
DATA="${DATA:-/scratch/chaunzt1/stagebridge}"
CANONICAL="$DATA/processed/luad_evo/canonical"
OUTPUT="$DATA/runs/v1_complete"

# =============================================================================
# Training Configuration
# =============================================================================

if [ "$SMOKE_MODE" = true ]; then
    SSL_EPOCHS=2
    TRANSITION_EPOCHS=3
    BATCH_SIZE=32
    CHECKPOINT_EVERY=1
    echo "SMOKE MODE: Using reduced parameters for quick test"
else
    SSL_EPOCHS=100
    TRANSITION_EPOCHS=50
    BATCH_SIZE=256
    CHECKPOINT_EVERY=10
fi

LATENT_DIM=40  # HLCA (30) + LuCA (10)
LEARNING_RATE=1e-3

echo ""
echo "Training Configuration:"
echo "  Latent dim: $LATENT_DIM"
echo "  SSL epochs: $SSL_EPOCHS"
echo "  Transition epochs: $TRANSITION_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Checkpoint every: $CHECKPOINT_EVERY epochs"
echo ""

# =============================================================================
# Verify Prerequisites
# =============================================================================

echo "Verifying prerequisites..."
REQUIRED=(
    "$CANONICAL/cells.parquet"
    "$CANONICAL/neighborhoods.parquet"
    "$CANONICAL/split_manifest.json"
)

PREREQS_OK=true
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        PREREQS_OK=false
    fi
done

if [ "$PREREQS_OK" = false ]; then
    echo ""
    echo "ERROR: Prerequisites not met. Run complete_data_prep first."
    exit 1
fi

# Verify PyTorch can see GPU
echo ""
echo "Verifying PyTorch GPU access..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# =============================================================================
# Skip-if-Complete Logic
# =============================================================================

RUN_DIR="$OUTPUT/fold${FOLD}_seed${SEED}"
FINAL_CHECKPOINT="$RUN_DIR/checkpoints/final_checkpoint.pt"

if [ "$FORCE_RETRAIN" = false ] && [ -f "$FINAL_CHECKPOINT" ]; then
    echo ""
    echo "=============================================="
    echo "TRAINING ALREADY COMPLETE - SKIPPING"
    echo "=============================================="
    echo "Found: $FINAL_CHECKPOINT"
    echo "To force retraining, run with --force flag"
    exit 0
fi

# =============================================================================
# Run Training
# =============================================================================

mkdir -p "$RUN_DIR"

echo ""
echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Output: $RUN_DIR"
echo ""

RESUME_ARG=""
if [ "$RESUME_MODE" = true ] && [ -d "$RUN_DIR/checkpoints" ]; then
    LATEST=$(ls -t "$RUN_DIR/checkpoints"/checkpoint_epoch_*.pt 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume_checkpoint $LATEST"
        echo "Resuming from: $LATEST"
    fi
fi

# Single GPU via torchrun (keeps same interface, no DDP overhead with nproc=1)
torchrun \
    --nproc_per_node=1 \
    --master_port=$((29500 + FOLD * 10 + SEED % 10)) \
    -m stagebridge.pipelines.run_v1_ddp \
    --data_dir "$CANONICAL" \
    --output_dir "$RUN_DIR" \
    --latent_dim $LATENT_DIM \
    --ssl_epochs $SSL_EPOCHS \
    --transition_epochs $TRANSITION_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --validation_fold $FOLD \
    --seed $SEED \
    --checkpoint_every $CHECKPOINT_EVERY \
    $RESUME_ARG

echo ""
echo "=============================================="
echo "Training Complete: $(date)"
echo "=============================================="
echo "Output: $RUN_DIR"
ls -lh "$RUN_DIR/"
