#!/bin/bash
#SBATCH --job-name=sb_hpo
#SBATCH --output=/data1/chaunzt1/stagebridge/outputs/v1.2/hpo/hpo_%j.log
#SBATCH --error=/data1/chaunzt1/stagebridge/outputs/v1.2/hpo/hpo_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# HPO for StageBridge v1.2 with fixed SSL reconstruction
# Uses DataParallel across 4 H200 GPUs
# Expected runtime: ~18 hours for 100 trials

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/data/v1"
OUTPUT_DIR="/data1/chaunzt1/stagebridge/outputs/v1.2/hpo"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "StageBridge HPO v1.2"
echo "============================================================"
echo "Start time: $(date)"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

cd /home/booka/projects/StageBridge

# Run HPO with:
# - 30 trials
# - 15 epochs per trial
# - batch size 128 per GPU (512 effective with 4 GPUs)
# - Optuna storage for resumability
python -m stagebridge.pipelines.run_hpo_dp \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-trials 30 \
    --n-epochs 15 \
    --batch-size 128 \
    --seed 42 \
    --storage "sqlite:///$OUTPUT_DIR/optuna.db" \
    --study-name "stagebridge_v1.2_fixed"

echo "============================================================"
echo "HPO Complete"
echo "End time: $(date)"
echo "============================================================"

# Print best params
if [ -f "$OUTPUT_DIR/best_params.json" ]; then
    echo "Best params:"
    cat "$OUTPUT_DIR/best_params.json"
fi
