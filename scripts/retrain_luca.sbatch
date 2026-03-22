#!/bin/bash
#SBATCH --job-name=luca_retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/logs/luca_retrain_%j.log

# =============================================================================
# CRITICAL: Multi-GPU Configuration for PyTorch Lightning + SLURM
# =============================================================================
#
# APPROACH 1 (This script): Single task, Lightning spawns GPU processes
#   - Use --ntasks-per-node=1 (default, not explicitly set)
#   - Use --gres=gpu:4 to request GPUs
#   - In Python: strategy="ddp_spawn", devices=4
#   - Lightning internally spawns 4 processes
#
# APPROACH 2 (Alternative): SLURM manages GPU processes
#   - Use --ntasks-per-node=4
#   - Use --gres=gpu:4
#   - Launch with: srun python script.py
#   - In Python: devices=1 (each task sees one GPU)
#
# The error "devices=4 but ntasks-per-node=1 doesn't match" happens when
# using strategy="ddp" (not ddp_spawn) which expects SLURM to manage processes.
# =============================================================================

echo "=============================================="
echo "Retrain LuCA scANVI Model (Multi-GPU)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "GPUs: 4x H100 (ddp_spawn strategy)"
echo "=============================================="

# Environment setup
cd /home/booka/projects/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Verify GPU visibility
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Data paths
DATA=/scratch/chaunzt1/stagebridge
ATLAS=$DATA/references/luca/luca_core_atlas.h5ad
OUTPUT=$DATA/references/luca/retrained_model

# Run training with ddp_spawn strategy (compatible with ntasks=1)
python scripts/retrain_luca_multigpu.py \
    --atlas "$ATLAS" \
    --output "$OUTPUT" \
    --n-latent 10 \
    --n-hidden 128 \
    --n-hvg 6000 \
    --max-epochs 400 \
    --batch-key dataset \
    --labels-key cell_type \
    --n-gpus 4

echo ""
echo "=============================================="
echo "Training Complete: $(date)"
echo "=============================================="
