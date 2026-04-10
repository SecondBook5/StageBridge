#!/bin/bash
#SBATCH -A chaunzt1
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH -C h100
#SBATCH --mem=512G
#SBATCH -t 720
#SBATCH --nodes=1
#SBATCH -o /scratch/chaunzt1/stagebridge/runs/logs/training_frozen/fold%a_seed%j.out
#SBATCH -e /scratch/chaunzt1/stagebridge/runs/logs/training_frozen/fold%a_seed%j.err

# Usage: sbatch --array=0-4 submit_training_frozen.sh <seed>
# Example: sbatch --array=0-4 submit_training_frozen.sh 42

SEED=${1:-42}
FOLD=$SLURM_ARRAY_TASK_ID

DATA_DIR=/scratch/chaunzt1/stagebridge/processed/luad_evo/canonical
OUTPUT_DIR=/scratch/chaunzt1/stagebridge/runs/v1_complete/frozen/fold${FOLD}_seed${SEED}
HPO_PARAMS=/scratch/chaunzt1/stagebridge/runs/v1_complete/hpo/best_params.json

echo "=============================================="
echo "StageBridge V1 Training (FROZEN ENCODER ABLATION)"
echo "  Fold: ${FOLD}/5"
echo "  Seed: ${SEED}"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi -L | wc -l)"
echo "  Purpose: Test SSL representation transfer quality"
echo "=============================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/metrics
mkdir -p /scratch/chaunzt1/stagebridge/runs/logs/training_frozen

# Run training with torchrun for DDP (frozen encoder)
torchrun \
    --nproc_per_node=4 \
    --master_port=$((29600 + FOLD * 10 + SEED % 10)) \
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
    --use_best_hparams \
    --freeze_encoder

echo "Frozen encoder training complete for fold ${FOLD}, seed ${SEED}"
