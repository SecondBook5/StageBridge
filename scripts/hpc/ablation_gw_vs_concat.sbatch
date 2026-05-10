#!/bin/bash
#SBATCH --job-name=ablation_gw
#SBATCH --output=logs/ablation_gw_%j.out
#SBATCH --error=logs/ablation_gw_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# GW vs Concat ablation - controlled comparison
# Runs both arms sequentially on same GPU for fair comparison

set -e

echo "=========================================="
echo "GW Fusion vs Concatenation Ablation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo ""

# Paths - adjust as needed
DATA_DIR="${DATA_DIR:-/data1/chaunzt1/stagebridge/processed/luad_evo/canonical}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/chaunzt1/stagebridge/outputs/ablation_gw}"

# Activate environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

# Create output dir
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

cd /home/booka/projects/StageBridge

echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run both arms
python scripts/benchmarks/ablation_gw_vs_concat.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --fold-idx 0 \
    --ssl-epochs 50 \
    --transition-epochs 100 \
    --batch-size 64 \
    --arm both

echo ""
echo "=========================================="
echo "Ablation complete: $(date)"
echo "Results in: $OUTPUT_DIR"
echo "=========================================="
