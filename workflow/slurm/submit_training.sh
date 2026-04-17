#!/bin/bash
#SBATCH -A chaunzt1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH -C h100
#SBATCH --mem=512G
#SBATCH -t 720
#SBATCH --nodes=1
#SBATCH -o /data1/chaunzt1/stagebridge/runs/logs/training/fold%a_seed%j.out
#SBATCH -e /data1/chaunzt1/stagebridge/runs/logs/training/fold%a_seed%j.err

# Usage: sbatch --array=0-4 submit_training.sh <seed>
# Example: sbatch --array=0-4 submit_training.sh 42

SEED=${1:-42}
FOLD=$SLURM_ARRAY_TASK_ID

DATA_DIR=/data1/chaunzt1/stagebridge/processed/luad_evo/canonical
OUTPUT_DIR=/data1/chaunzt1/stagebridge/runs/v1_complete/fold${FOLD}_seed${SEED}
HPO_PARAMS=/data1/chaunzt1/stagebridge/runs/v1_complete/hpo/best_params.json

echo "=============================================="
echo "StageBridge V1 Training"
echo "  Fold: ${FOLD}/5"
echo "  Seed: ${SEED}"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L | wc -l)"
echo "=============================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/metrics
mkdir -p /data1/chaunzt1/stagebridge/runs/logs/training

# Run training with torchrun for DDP
torchrun \
    --nproc_per_node=4 \
    --master_port=$((29500 + FOLD * 10 + SEED % 10)) \
    -m stagebridge.pipelines.run_v1_ddp \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --validation_fold ${FOLD} \
    --seed ${SEED} \
    --ssl_epochs 100 \
    --transition_epochs 50 \
    --batch_size 256 \
    --checkpoint_every 10 \
    --hpo_params ${HPO_PARAMS} \
    --use_best_hparams

echo "Training complete for fold ${FOLD}, seed ${SEED}"
