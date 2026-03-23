#!/bin/bash
#SBATCH --job-name=sb_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/training_%j.log

# =============================================================================
# Stage 5: Full Model Training (Multi-GPU DDP)
# =============================================================================
#
# Enhanced training script with:
# - Multi-GPU DDP via torchrun (4x H100)
# - Checkpoint/resume logic (auto-resume from last checkpoint)
# - Skip-if-complete (skips training if final model exists and is valid)
# - Donor-held-out cross-validation splits
# - Intermediate checkpoints every N epochs
# - All publication artifacts saved
#
# Training stages:
# 1. SSL Pretraining (receiver-centered niche learning, 70% weight)
# 2. Transition Modeling (flow matching)
#
# Uses ReceiverCenteredNicheEncoder per doctrine:
# - Receiver as query, neighbors as keys/values
# - Distance-aware attention
# - Sparsity regularization
#
# Produces:
# - checkpoints/checkpoint_epoch_*.pt (intermediate)
# - checkpoints/final_checkpoint.pt
# - checkpoints/best_checkpoint.pt
# - metrics/training_metrics.csv
# - metrics/validation_metrics.csv
# - diagnostics/runtime_summary.json
# - attention/attention_summaries.parquet
# - trajectories/sample_predictions.parquet
# - uncertainty/uncertainty_summary.json
# - figures/training_curves.png
# - manifest.json (artifact manifest)
#
# Prerequisites:
# - Stage 4 (Data Preparation) completed
#
# Runtime: ~12-24 hours (SSL + Transition)
#
# Usage:
#   sbatch scripts/hpc_step5_training.sbatch              # Fresh training
#   sbatch scripts/hpc_step5_training.sbatch --resume     # Resume from checkpoint
#   sbatch scripts/hpc_step5_training.sbatch --force      # Force retrain
# =============================================================================

set -euo pipefail

# Parse command-line arguments
FORCE_RETRAIN=false
RESUME_MODE=false
SMOKE_MODE=false
for arg in "$@"; do
    case $arg in
        --force)
            FORCE_RETRAIN=true
            shift
            ;;
        --resume)
            RESUME_MODE=true
            shift
            ;;
        --smoke)
            SMOKE_MODE=true
            shift
            ;;
    esac
done

echo "=============================================="
echo "Stage 5: Full Model Training (Multi-GPU DDP)"
echo "=============================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Force retrain: $FORCE_RETRAIN"
echo "Resume mode: $RESUME_MODE"
echo "Smoke mode: $SMOKE_MODE"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

cd /home/booka/StageBridge

# Conda activation (handle both interactive and batch)
if [ -f /admin/software/miniforge3/etc/profile.d/conda.sh ]; then
    source /admin/software/miniforge3/etc/profile.d/conda.sh
    conda activate /scratch/chaunzt1/stagebridge_env
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate stagebridge
fi

# Data paths
DATA="${DATA:-/scratch/chaunzt1/stagebridge}"
CANONICAL="$DATA/processed/luad_evo/canonical"
HLCA="$DATA/references/hlca/hlca_reference.h5ad"
LUCA="$DATA/references/luca/luca_core_atlas.h5ad"
OUTPUT="$DATA/runs/v1_complete"

# =============================================================================
# GPU Configuration for Multi-GPU DDP
# =============================================================================

# Set visible GPUs - DDP will use all of them
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Count available GPUs
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo ""
echo "GPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Number of GPUs: $N_GPUS"

# DDP settings
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12355}"
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo ""

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    echo ""
fi

# =============================================================================
# Training Configuration
# =============================================================================

# Default training parameters
if [ "$SMOKE_MODE" = true ]; then
    SSL_EPOCHS=2
    TRANSITION_EPOCHS=3
    BATCH_SIZE=32
    HPO_TRIALS=2
    CHECKPOINT_EVERY=1
    echo "SMOKE MODE: Using reduced parameters for quick test"
else
    SSL_EPOCHS=50
    TRANSITION_EPOCHS=100
    BATCH_SIZE=128
    HPO_TRIALS=30
    CHECKPOINT_EVERY=10
fi

LATENT_DIM=32
LEARNING_RATE=1e-3
SEED=42
N_FOLDS=5

echo ""
echo "Training Configuration:"
echo "  Latent dim: $LATENT_DIM"
echo "  SSL epochs: $SSL_EPOCHS"
echo "  Transition epochs: $TRANSITION_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Seed: $SEED"
echo "  N folds: $N_FOLDS"
echo "  HPO trials: $HPO_TRIALS"
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
    echo "ERROR: Prerequisites not met. Run Stage 4 (Data Preparation) first."
    echo "Command: sbatch scripts/hpc_step4_complete_data_prep.sbatch"
    exit 1
fi

# Verify PyTorch can see GPUs
echo ""
echo "Verifying PyTorch GPU access..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)')
" || {
    echo "ERROR: PyTorch cannot access GPUs"
    exit 1
}

# =============================================================================
# Skip-if-Complete Logic
# =============================================================================

FINAL_CHECKPOINT="$OUTPUT/checkpoints/final_checkpoint.pt"
FINAL_MODEL="$OUTPUT/weights/final_model.pt"
RESULTS_JSON="$OUTPUT/results.json"
MANIFEST_JSON="$OUTPUT/manifest.json"

check_training_complete() {
    # Check if all required outputs exist and are valid
    if [ ! -f "$FINAL_CHECKPOINT" ] && [ ! -f "$FINAL_MODEL" ]; then
        return 1
    fi

    if [ ! -f "$RESULTS_JSON" ]; then
        return 1
    fi

    # Validate checkpoint is loadable
    python -c "
import torch
import json
import sys

# Check checkpoint
ckpt_path = '$FINAL_CHECKPOINT' if '$FINAL_CHECKPOINT' != '' else '$FINAL_MODEL'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state_dict' not in ckpt and 'state_dict' not in ckpt:
        print('WARNING: Checkpoint missing model state dict')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: Cannot load checkpoint: {e}')
    sys.exit(1)

# Check results
try:
    with open('$RESULTS_JSON') as f:
        results = json.load(f)
    if 'duration_seconds' not in results:
        print('WARNING: Results file incomplete')
        sys.exit(1)
except Exception as e:
    print(f'ERROR: Cannot load results: {e}')
    sys.exit(1)

print('Training outputs verified successfully')
sys.exit(0)
" 2>/dev/null
    return $?
}

if [ "$FORCE_RETRAIN" = false ]; then
    echo ""
    echo "Checking if training is already complete..."

    if check_training_complete; then
        echo ""
        echo "=============================================="
        echo "TRAINING ALREADY COMPLETE - SKIPPING"
        echo "=============================================="
        echo ""
        echo "Found valid training outputs:"
        [ -f "$FINAL_CHECKPOINT" ] && echo "  - $FINAL_CHECKPOINT"
        [ -f "$FINAL_MODEL" ] && echo "  - $FINAL_MODEL"
        [ -f "$RESULTS_JSON" ] && echo "  - $RESULTS_JSON"
        echo ""
        echo "To force retraining, run with --force flag:"
        echo "  sbatch scripts/hpc_step5_training.sbatch --force"
        echo ""
        echo "Stage 5 Complete (skipped): $(date)"
        exit 0
    fi

    echo "  Training not complete - will proceed"
fi

# =============================================================================
# Checkpoint/Resume Logic
# =============================================================================

RESUME_CHECKPOINT=""

find_latest_checkpoint() {
    # Find the most recent checkpoint to resume from
    local ckpt_dir="$OUTPUT/checkpoints"

    if [ ! -d "$ckpt_dir" ]; then
        return 1
    fi

    # Find checkpoint with highest epoch number
    local latest=$(ls -t "$ckpt_dir"/checkpoint_epoch_*.pt 2>/dev/null | head -1)

    if [ -n "$latest" ] && [ -f "$latest" ]; then
        # Validate it's loadable
        if python -c "import torch; torch.load('$latest', map_location='cpu')" 2>/dev/null; then
            echo "$latest"
            return 0
        fi
    fi

    return 1
}

if [ "$RESUME_MODE" = true ] || [ "$FORCE_RETRAIN" = false ]; then
    echo ""
    echo "Checking for resumable checkpoint..."

    LATEST_CKPT=$(find_latest_checkpoint || echo "")

    if [ -n "$LATEST_CKPT" ]; then
        RESUME_CHECKPOINT="$LATEST_CKPT"

        # Extract epoch info
        RESUME_EPOCH=$(python -c "
import torch
ckpt = torch.load('$RESUME_CHECKPOINT', map_location='cpu')
print(ckpt.get('epoch', ckpt.get('ssl_epoch', 0) + ckpt.get('transition_epoch', 0)))
" 2>/dev/null || echo "0")

        echo "  Found checkpoint: $RESUME_CHECKPOINT"
        echo "  Resume from epoch: $RESUME_EPOCH"
    else
        echo "  No checkpoint found - starting fresh"
    fi
fi

# =============================================================================
# Create Output Directories
# =============================================================================

mkdir -p "$OUTPUT"/{checkpoints,weights,metrics,diagnostics,attention,trajectories,influence,uncertainty,logs,figures,config}

echo ""
echo "Output directory: $OUTPUT"
ls -la "$OUTPUT/"

# =============================================================================
# Run Training
# =============================================================================

echo ""
echo "=============================================="
echo "Starting V1 Complete Training Pipeline"
echo "=============================================="
echo ""

# Determine which training module to use
# Prefer DDP module for multi-GPU, fallback to original for single GPU
USE_DDP_MODULE=false
if [ "$N_GPUS" -gt 1 ]; then
    # Check if DDP module exists
    if python -c "from stagebridge.pipelines.run_v1_ddp import train" 2>/dev/null; then
        USE_DDP_MODULE=true
        echo "Using DDP-aware training module (run_v1_ddp)"
    else
        echo "DDP module not available, using run_v1_complete with DataParallel fallback"
    fi
fi

# Build training command
if [ "$USE_DDP_MODULE" = true ]; then
    # Use torchrun with DDP module
    echo "Using torchrun for multi-GPU DDP training..."
    echo "  Number of processes: $N_GPUS"
    echo ""

    TRAIN_CMD="torchrun \
        --nproc_per_node=$N_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m stagebridge.pipelines.run_v1_ddp \
        --data_dir \"$CANONICAL\" \
        --output_dir \"$OUTPUT\" \
        --hlca_path \"$HLCA\" \
        --luca_path \"$LUCA\" \
        --latent_dim $LATENT_DIM \
        --ssl_epochs $SSL_EPOCHS \
        --transition_epochs $TRANSITION_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --seed $SEED \
        --hpo_trials $HPO_TRIALS \
        --checkpoint_every $CHECKPOINT_EVERY"

    if [ "$USE_BEST_HPARAMS" = true ] 2>/dev/null || [ -n "$HPO_TRIALS" ]; then
        TRAIN_CMD="$TRAIN_CMD --use_best_hparams"
    fi

    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume_checkpoint \"$RESUME_CHECKPOINT\""
    fi
else
    # Use original module (single GPU or fallback)
    TRAIN_CMD="python -m stagebridge.pipelines.run_v1_complete \
        --data_dir \"$CANONICAL\" \
        --output_dir \"$OUTPUT\" \
        --hlca_path \"$HLCA\" \
        --luca_path \"$LUCA\" \
        --latent_dim $LATENT_DIM \
        --ssl_epochs $SSL_EPOCHS \
        --transition_epochs $TRANSITION_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --seed $SEED \
        --device auto \
        --hpo_trials $HPO_TRIALS \
        --use_best_hparams"

    # Add checkpoint-related flags if the pipeline supports them
    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume_checkpoint \"$RESUME_CHECKPOINT\""
    fi
fi

echo "Training command:"
echo "$TRAIN_CMD"
echo ""

# Run training with error handling
TRAINING_START=$(date +%s)

eval "$TRAIN_CMD" 2>&1 | tee "$OUTPUT/logs/training_${SLURM_JOB_ID:-local}.log"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

TRAINING_END=$(date +%s)
TRAINING_DURATION=$((TRAINING_END - TRAINING_START))

echo ""
echo "Training duration: $((TRAINING_DURATION / 3600))h $((TRAINING_DURATION % 3600 / 60))m $((TRAINING_DURATION % 60))s"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=============================================="
    echo "TRAINING FAILED (exit code: $TRAIN_EXIT_CODE)"
    echo "=============================================="
    echo ""
    echo "Check logs: $OUTPUT/logs/training_${SLURM_JOB_ID:-local}.log"
    echo ""
    echo "To resume from last checkpoint:"
    echo "  sbatch scripts/hpc_step5_training.sbatch --resume"
    exit $TRAIN_EXIT_CODE
fi

# =============================================================================
# Create Symlinks for Compatibility
# =============================================================================

# Create symlinks for backward compatibility with old paths
if [ -f "$OUTPUT/checkpoints/final_checkpoint.pt" ] && [ ! -f "$OUTPUT/weights/final_model.pt" ]; then
    ln -sf "$OUTPUT/checkpoints/final_checkpoint.pt" "$OUTPUT/weights/final_model.pt"
fi

if [ -f "$OUTPUT/checkpoints/best_checkpoint.pt" ]; then
    for dataset in semi_synthetic real; do
        if [ ! -f "$OUTPUT/weights/best_model_${dataset}.pt" ]; then
            ln -sf "$OUTPUT/checkpoints/best_checkpoint.pt" "$OUTPUT/weights/best_model_${dataset}.pt"
        fi
    done
fi

# =============================================================================
# Verify Outputs
# =============================================================================

echo ""
echo "Verifying outputs..."

EXPECTED_FILES=(
    "$OUTPUT/weights/final_model.pt"
    "$OUTPUT/results.json"
    "$OUTPUT/config.json"
)

OPTIONAL_FILES=(
    "$OUTPUT/checkpoints/final_checkpoint.pt"
    "$OUTPUT/checkpoints/best_checkpoint.pt"
    "$OUTPUT/figures/fig1_training_curves.png"
    "$OUTPUT/figures/fig2_ablation.png"
    "$OUTPUT/figures/fig3_trajectories.png"
    "$OUTPUT/metrics/training_metrics.csv"
    "$OUTPUT/hpo_results.json"
    "$OUTPUT/ablation_results.csv"
)

ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [REQUIRED OK] $f ($SIZE)"
    else
        echo "  [REQUIRED MISSING] $f"
        ALL_OK=false
    fi
done

for f in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OPTIONAL OK] $f ($SIZE)"
    else
        echo "  [OPTIONAL SKIP] $f"
    fi
done

# =============================================================================
# Generate Artifact Manifest
# =============================================================================

echo ""
echo "Generating artifact manifest..."

python -c "
import json
import os
from pathlib import Path
from datetime import datetime

output_dir = Path('$OUTPUT')
manifest = {
    'run_id': 'v1_complete',
    'timestamp': datetime.now().isoformat(),
    'job_id': '${SLURM_JOB_ID:-local}',
    'duration_seconds': $TRAINING_DURATION,
    'n_gpus': $N_GPUS,
    'artifacts': {}
}

# Scan for artifacts
artifact_dirs = ['checkpoints', 'weights', 'metrics', 'diagnostics',
                 'attention', 'trajectories', 'influence', 'uncertainty',
                 'figures', 'config', 'logs']

for dir_name in artifact_dirs:
    dir_path = output_dir / dir_name
    if dir_path.exists():
        files = []
        for f in dir_path.iterdir():
            if f.is_file():
                files.append({
                    'name': f.name,
                    'size_bytes': f.stat().st_size,
                    'modified': datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        manifest['artifacts'][dir_name] = files

# Add root-level files
root_files = []
for f in output_dir.iterdir():
    if f.is_file() and f.suffix in ['.json', '.csv', '.yaml']:
        root_files.append({
            'name': f.name,
            'size_bytes': f.stat().st_size
        })
manifest['artifacts']['root'] = root_files

# Write manifest
with open(output_dir / 'manifest.json', 'w') as fh:
    json.dump(manifest, fh, indent=2)

print(f'Manifest written with {sum(len(v) for v in manifest[\"artifacts\"].values())} artifacts')
"

# =============================================================================
# Show Training Summary
# =============================================================================

if [ -f "$OUTPUT/results.json" ]; then
    echo ""
    echo "Training Summary:"
    echo "================"
    python -c "
import json

with open('$OUTPUT/results.json') as f:
    results = json.load(f)

print('Semi-Synthetic:')
ssl = results.get('history_semi_synthetic', {}).get('ssl_loss', [])
val = results.get('history_semi_synthetic', {}).get('val_loss', [])
if ssl: print(f'  Final SSL loss: {ssl[-1]:.4f}')
if val: print(f'  Final Val loss: {val[-1]:.4f}')

print('Real Data:')
ssl = results.get('history_real', {}).get('ssl_loss', [])
val = results.get('history_real', {}).get('val_loss', [])
if ssl: print(f'  Final SSL loss: {ssl[-1]:.4f}')
if val: print(f'  Final Val loss: {val[-1]:.4f}')

print(f'Duration: {results.get(\"duration_seconds\", 0)/60:.1f} minutes')
print(f'Parameters: {results.get(\"n_parameters\", 0):,}')

# Show ablation summary if available
ablation = results.get('ablation_results', {})
if ablation and 'Model' in ablation:
    print()
    print('Ablation Summary:')
    models = ablation.get('Model', {})
    losses = ablation.get('loss', {})
    for idx in sorted(models.keys()):
        print(f'  {models[idx]}: {losses.get(idx, \"N/A\")}')
"
fi

# =============================================================================
# List All Outputs
# =============================================================================

echo ""
echo "Model Weights:"
ls -lh "$OUTPUT/weights/" 2>/dev/null || echo "  No weights directory"

echo ""
echo "Checkpoints:"
ls -lh "$OUTPUT/checkpoints/" 2>/dev/null || echo "  No checkpoints directory"

echo ""
echo "Figures:"
ls -lh "$OUTPUT/figures/" 2>/dev/null || echo "  No figures directory"

# =============================================================================
# Final Status
# =============================================================================

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "=============================================="
    echo "WARNING: Some required outputs are missing"
    echo "=============================================="
    echo ""
    echo "Training may have partially completed."
    echo "Check logs: $OUTPUT/logs/"
    echo ""
    echo "To resume: sbatch scripts/hpc_step5_training.sbatch --resume"
    exit 1
fi

echo ""
echo "=============================================="
echo "Stage 5 Complete: $(date)"
echo "=============================================="
echo ""
echo "Training completed successfully!"
echo ""
echo "Outputs:"
echo "  Final model:  $OUTPUT/weights/final_model.pt"
echo "  Best models:  $OUTPUT/weights/best_model_*.pt"
echo "  Checkpoints:  $OUTPUT/checkpoints/"
echo "  Metrics:      $OUTPUT/metrics/"
echo "  Figures:      $OUTPUT/figures/"
echo "  Results:      $OUTPUT/results.json"
echo "  Manifest:     $OUTPUT/manifest.json"
echo ""
echo "GPU hours used: $(echo \"scale=2; $TRAINING_DURATION * $N_GPUS / 3600\" | bc) GPU-hours"
echo ""
echo "Next: Stage 6 (Ablation Studies)"
echo "  sbatch scripts/hpc_step6_ablations.sbatch"
