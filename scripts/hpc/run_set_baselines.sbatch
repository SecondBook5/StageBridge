#!/bin/bash
#SBATCH --job-name=set_baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-24
#SBATCH --output=baselines_%A_%a.out
#SBATCH --error=baselines_%A_%a.err

# 25 jobs: 5 baselines x 5 folds
# Array index maps to (baseline, fold) pairs

BASELINES=(pooling_mlp deep_sets set_transformer graph_sage receiver_centered)
FOLDS=(0 1 2 3 4)

# Calculate baseline and fold from array index
BASELINE_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % 5))

BASELINE=${BASELINES[$BASELINE_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}

echo "Running baseline: $BASELINE, fold: $FOLD"

# Data paths
DATA=/scratch/chaunzt1/stagebridge
CANONICAL=$DATA/processed/luad_evo/canonical
OUTPUT=$DATA/runs/baselines

mkdir -p $OUTPUT

# Load environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd ~/StageBridge

python -m stagebridge.baselines.evaluate_set_baselines \
    --data_dir $CANONICAL \
    --output_dir $OUTPUT \
    --baseline $BASELINE \
    --fold $FOLD \
    --seed 42 \
    --epochs 50 \
    --batch_size 256
