#!/bin/bash
#SBATCH --job-name=ref_mapping
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/reference_mapping_%j.log

cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

python -m stagebridge.pipelines.run_reference \
    --data-root /scratch/chaunzt1/stagebridge_data \
    --hpc \
    --luca /scratch/chaunzt1/stagebridge_data/references/luca/luca_core_atlas.h5ad
