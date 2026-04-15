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
# StageBridge V1 Complete Training Pipeline
# =============================================================================
#
# Full pipeline:
# 1. Generate semi-synthetic benchmark (if not exists)
# 2. Run HPO (if not exists)
# 3. SSL Pretraining
# 4. Transition Modeling
#
# Usage:
#   sbatch scripts/hpc/run_training.sbatch
#   sbatch scripts/hpc/run_training.sbatch --fold=1 --seed=123
#   sbatch scripts/hpc/run_training.sbatch --smoke   # Quick test
#   sbatch scripts/hpc/run_training.sbatch --skip-hpo  # Use defaults
# =============================================================================

set -euo pipefail

# Parse command-line arguments
FORCE_RETRAIN=false
RESUME_MODE=false
SMOKE_MODE=false
SKIP_HPO=false
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
        --skip-hpo)
            SKIP_HPO=true
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
echo "StageBridge V1 Complete Training Pipeline"
echo "=============================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Fold: $FOLD, Seed: $SEED"
echo "Smoke mode: $SMOKE_MODE"
echo "Skip HPO: $SKIP_HPO"
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
HPO_DIR="$OUTPUT/hpo"
BENCHMARK_DIR="$CANONICAL/benchmark"

# =============================================================================
# Training Configuration
# =============================================================================

if [ "$SMOKE_MODE" = true ]; then
    SSL_EPOCHS=2
    TRANSITION_EPOCHS=3
    BATCH_SIZE=32
    HPO_TRIALS=2
    CHECKPOINT_EVERY=1
    echo "SMOKE MODE: Using reduced parameters for quick test"
else
    SSL_EPOCHS=100
    TRANSITION_EPOCHS=50
    BATCH_SIZE=256
    HPO_TRIALS=30
    CHECKPOINT_EVERY=10
fi

LATENT_DIM=40  # HLCA (30) + LuCA (10)

echo ""
echo "Training Configuration:"
echo "  Latent dim: $LATENT_DIM"
echo "  SSL epochs: $SSL_EPOCHS"
echo "  Transition epochs: $TRANSITION_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  HPO trials: $HPO_TRIALS"
echo ""

# =============================================================================
# Step 0: Verify Prerequisites
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

# Verify GPU
echo ""
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# =============================================================================
# Step 1: Generate Semi-Synthetic Benchmark (if not exists)
# =============================================================================

SEMI_SYNTHETIC="$BENCHMARK_DIR/semi_synthetic.pt"
GROUND_TRUTH="$BENCHMARK_DIR/ground_truth.json"

if [ -f "$SEMI_SYNTHETIC" ] && [ -f "$GROUND_TRUTH" ]; then
    echo ""
    echo "[Step 1] Semi-synthetic benchmark already exists - SKIPPING"
    echo "  $SEMI_SYNTHETIC"
else
    echo ""
    echo "=============================================="
    echo "[Step 1] Generating Semi-Synthetic Benchmark"
    echo "=============================================="

    mkdir -p "$BENCHMARK_DIR"

    python -c "
import torch
import json
import numpy as np
from pathlib import Path
from stagebridge.benchmarks.semi_synthetic import SemiSyntheticBenchmarkGenerator

print('Generating semi-synthetic benchmark with ground truth...')

# Load real expression if available
canonical = Path('$CANONICAL')
manifest_path = canonical / 'data_manifest.json'

expression_source = None
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    snrna_path = manifest.get('snrna_path')
    if snrna_path and Path(snrna_path).exists():
        expression_source = Path(snrna_path)
        print(f'Using expression from: {expression_source}')

# Generate benchmark
generator = SemiSyntheticBenchmarkGenerator(
    expression_source=expression_source,
    n_worlds=10 if '$SMOKE_MODE' != 'true' else 2,
    cells_per_world=1000 if '$SMOKE_MODE' != 'true' else 100,
)

benchmark = generator.generate()

# Save
output_dir = Path('$BENCHMARK_DIR')
torch.save(benchmark['tensors'], output_dir / 'semi_synthetic.pt')
with open(output_dir / 'ground_truth.json', 'w') as f:
    json.dump(benchmark['ground_truth'], f, indent=2, default=str)

print(f'Saved semi-synthetic benchmark to {output_dir}')
print(f'  Tensors: {list(benchmark[\"tensors\"].keys())}')
"

    echo "Semi-synthetic benchmark generated."
fi

# =============================================================================
# Step 2: Hyperparameter Optimization (if not exists)
# =============================================================================

HPO_PARAMS="$HPO_DIR/best_params.json"

if [ "$SKIP_HPO" = true ]; then
    echo ""
    echo "[Step 2] HPO skipped (--skip-hpo flag)"
    HPO_ARG=""
elif [ -f "$HPO_PARAMS" ]; then
    echo ""
    echo "[Step 2] HPO results already exist - SKIPPING"
    echo "  $HPO_PARAMS"
    HPO_ARG="--hpo_params $HPO_PARAMS --use_best_hparams"
else
    echo ""
    echo "=============================================="
    echo "[Step 2] Running Hyperparameter Optimization"
    echo "=============================================="

    mkdir -p "$HPO_DIR"

    python -m stagebridge.pipelines.run_hpo \
        --data_dir "$CANONICAL" \
        --output_dir "$HPO_DIR" \
        --n_trials $HPO_TRIALS \
        --n_epochs 10 \
        --batch_size 64 \
        --latent_dim $LATENT_DIM \
        --seed $SEED

    echo "HPO complete."
    HPO_ARG="--hpo_params $HPO_PARAMS --use_best_hparams"
fi

# =============================================================================
# Step 3: Skip-if-Complete Check
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
# Step 4: SSL Pretraining + Transition Training
# =============================================================================

echo ""
echo "=============================================="
echo "[Step 3-4] SSL Pretraining + Transition Training"
echo "=============================================="
echo "Output: $RUN_DIR"
echo ""

mkdir -p "$RUN_DIR"

RESUME_ARG=""
if [ "$RESUME_MODE" = true ] && [ -d "$RUN_DIR/checkpoints" ]; then
    LATEST=$(ls -t "$RUN_DIR/checkpoints"/checkpoint_epoch_*.pt 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST" ]; then
        RESUME_ARG="--resume_checkpoint $LATEST"
        echo "Resuming from: $LATEST"
    fi
fi

# Single GPU via torchrun (no DDP overhead with nproc=1)
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
    --validation_fold $FOLD \
    --seed $SEED \
    --checkpoint_every $CHECKPOINT_EVERY \
    $HPO_ARG \
    $RESUME_ARG

echo ""
echo "=============================================="
echo "Training Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  Run directory: $RUN_DIR"
echo "  Semi-synthetic: $BENCHMARK_DIR"
echo "  HPO params: $HPO_PARAMS"
echo ""
ls -lh "$RUN_DIR/"
