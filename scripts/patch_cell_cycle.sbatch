#!/bin/bash
#SBATCH --job-name=patch_and_run
#SBATCH --partition=gpu
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=patch_and_run_%j.log
#SBATCH --error=patch_and_run_%j.log

source ~/.bashrc
conda activate stagebridge_env

cd /home/booka/projects/StageBridge

echo "=== Patching cell cycle and cell types ==="
python scripts/patch_cell_cycle.py --use-config

if [ $? -ne 0 ]; then
    echo "Patch failed, not starting snakemake"
    exit 1
fi

echo ""
echo "=== Starting Snakemake pipeline ==="
snakemake --profile workflow/slurm --jobs 50

echo "Done!"
