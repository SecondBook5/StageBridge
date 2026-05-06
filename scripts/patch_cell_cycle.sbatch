#!/bin/bash
#SBATCH --job-name=patch_cell_cycle
#SBATCH --partition=gpu
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --output=patch_cell_cycle_%j.log
#SBATCH --error=patch_cell_cycle_%j.log

source ~/.bashrc
conda activate stagebridge_env

cd /home/booka/projects/StageBridge

python scripts/patch_cell_cycle.py --use-config

echo "Done!"
