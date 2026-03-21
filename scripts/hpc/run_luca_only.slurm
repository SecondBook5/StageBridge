#!/bin/bash
#SBATCH --job-name=luca_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/luca_map_%j.out
#SBATCH --error=logs/luca_map_%j.err

# LuCA-only mapping using luca_final environment
# This environment has compatible anndata/pandas/numpy versions

set -e

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /home/booka/luca_final

export DATA=/scratch/chaunzt1/stagebridge
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

echo "=============================================="
echo "LuCA Reference Mapping"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Environment: luca_final"
echo "=============================================="

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd /home/booka/StageBridge

python -m stagebridge.pipelines.run_reference \
    --data-root "$DATA" \
    --hpc \
    --luca-only \
    --snrna "$DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad"

echo ""
echo "=============================================="
echo "LuCA Mapping Complete"
echo "Finished: $(date)"
echo "=============================================="
