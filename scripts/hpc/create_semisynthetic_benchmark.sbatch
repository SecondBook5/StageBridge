#!/bin/bash
#SBATCH --job-name=semisynthetic
#SBATCH --partition=cpu
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/home/booka/StageBridge/logs/semisynthetic_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/semisynthetic_%j.err

# Create semi-synthetic benchmark from real expression profiles
# AMICI-style: real expression + controlled spatial layout + known ground truth

echo "=============================================="
echo "Semi-Synthetic Benchmark Generation"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

# Paths
SNRNA="/data1/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad"
OUTPUT="/data1/chaunzt1/stagebridge/processed/luad_evo/semisynthetic_benchmark"
LOCAL="/home/booka/StageBridge/data/semisynthetic_benchmark"

mkdir -p $OUTPUT
mkdir -p $LOCAL
mkdir -p /home/booka/StageBridge/logs

# =============================================================================
# Generate multiple benchmark sizes
# =============================================================================

# Small (for quick local testing)
echo ""
echo "[1/3] Creating small benchmark (1000 cells)..."
python scripts/benchmarks/create_semisynthetic_benchmark.py \
    --h5ad "$SNRNA" \
    --output "${OUTPUT}_small" \
    --n-cells 1000 \
    --pattern gradient \
    --seed 42

# Medium (for development)
echo ""
echo "[2/3] Creating medium benchmark (5000 cells)..."
python scripts/benchmarks/create_semisynthetic_benchmark.py \
    --h5ad "$SNRNA" \
    --output "${OUTPUT}_medium" \
    --n-cells 5000 \
    --pattern gradient \
    --seed 42

# Large (for full evaluation)
echo ""
echo "[3/3] Creating large benchmark (20000 cells)..."
python scripts/benchmarks/create_semisynthetic_benchmark.py \
    --h5ad "$SNRNA" \
    --output "${OUTPUT}_large" \
    --n-cells 20000 \
    --pattern gradient \
    --seed 42

# =============================================================================
# Copy small benchmark locally for quick iteration
# =============================================================================
echo ""
echo "Copying small benchmark to local..."
cp -r "${OUTPUT}_small"/* "$LOCAL/"
echo "  Local copy: $LOCAL"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "COMPLETE"
echo "=============================================="
echo "Small (1k):  ${OUTPUT}_small"
echo "Medium (5k): ${OUTPUT}_medium"
echo "Large (20k): ${OUTPUT}_large"
echo "Local copy:  $LOCAL"
echo ""
echo "To train locally:"
echo "  python scripts/train.py --data-dir $LOCAL --epochs 10"
echo ""
echo "Done: $(date)"
