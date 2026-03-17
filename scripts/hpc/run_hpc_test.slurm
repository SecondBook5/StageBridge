#!/bin/bash
#SBATCH --job-name=stagebridge_test
#SBATCH --output=logs/stagebridge_test_%j.out
#SBATCH --error=logs/stagebridge_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

################################################################################
# StageBridge V1 Quick Test - HPC
################################################################################

echo "=========================================="
echo "StageBridge V1 Quick Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# Load modules
module purge
module load cuda/12.1
module load gcc/11.2.0

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stagebridge

# Verify GPU
nvidia-smi

# Create directories
mkdir -p logs
mkdir -p outputs/synthetic_test

# Change to project directory
cd $SLURM_SUBMIT_DIR

echo ""
echo "Running synthetic data test..."
echo ""

# Generate synthetic data
python -c "
from stagebridge.data.synthetic import generate_synthetic_dataset
from pathlib import Path

data_path = generate_synthetic_dataset(
    output_dir='outputs/synthetic_test',
    n_cells=500,
    n_donors=5,
    latent_dim=32,
    seed=42,
)
print(f' Synthetic data: {data_path}')
"

# Train one fold for 3 epochs
echo ""
echo "Training quick test (3 epochs, fold 0)..."
python stagebridge/pipelines/run_v1_full.py \
    --data_dir outputs/synthetic_test \
    --fold 0 \
    --n_epochs 3 \
    --batch_size 32 \
    --output_dir outputs/synthetic_test/training/fold_0 \
    --niche_encoder mlp \
    --use_wes

echo ""
echo "=========================================="
echo " Test Complete!"
echo "=========================================="

# Check results
if [ -f "outputs/synthetic_test/training/fold_0/results.json" ]; then
    echo " Training succeeded!"
    cat outputs/synthetic_test/training/fold_0/results.json
else
    echo " Training failed!"
    exit 1
fi
