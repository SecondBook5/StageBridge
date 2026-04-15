#!/bin/bash
#SBATCH --job-name=set_baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=baselines_%A_%a.out
#SBATCH --error=baselines_%A_%a.err

# 20 jobs: 4 baselines x 5 folds
# Array index maps to (baseline, fold) pairs

BASELINES=(pooling_mlp deep_sets set_transformer graph_sage)
FOLDS=(0 1 2 3 4)

# Calculate baseline and fold from array index
# 4 baselines x 5 folds = 20 jobs
BASELINE_IDX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD_IDX=$((SLURM_ARRAY_TASK_ID % 5))

BASELINE=${BASELINES[$BASELINE_IDX]}
FOLD=${FOLDS[$FOLD_IDX]}
SEED=42

echo "Running baseline: $BASELINE, fold: $FOLD, seed: $SEED"

# Data paths
DATA=/scratch/chaunzt1/stagebridge
CANONICAL=$DATA/processed/luad_evo/canonical
OUTPUT_BASE=$DATA/runs/v1_complete/baselines

# Output will be: OUTPUT_BASE/{baseline}_fold{fold}_seed{seed}/
# Skip logic is in the Python script

mkdir -p $OUTPUT_BASE

# Load environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd ~/StageBridge

python -m stagebridge.baselines.evaluate \
    --data_dir $CANONICAL \
    --output_dir $OUTPUT_BASE \
    --baseline $BASELINE \
    --fold $FOLD \
    --seed $SEED \
    --epochs 50 \
    --batch_size 256
