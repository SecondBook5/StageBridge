#!/bin/bash
#SBATCH --job-name=entropy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/booka/StageBridge/logs/entropy_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/entropy_%j.err

# Compute entropy for all completed StageBridge checkpoints

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Activate environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

python scripts/compute_entropy.py

echo "Done: $(date)"
