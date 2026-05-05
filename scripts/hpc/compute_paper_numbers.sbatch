#!/bin/bash
#SBATCH --job-name=paper_numbers
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/home/booka/StageBridge/logs/paper_numbers_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/paper_numbers_%j.err

# Compute ALL paper numbers with traceable provenance
# Output: /data1/chaunzt1/stagebridge/outputs/v1/paper_numbers.json
#         /data1/chaunzt1/stagebridge/outputs/v1/drift_alignment.json

set -e

echo "=============================================="
echo "Computing Paper Numbers"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

DATA="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT="/data1/chaunzt1/stagebridge/outputs/v1"
RESULTS="/home/booka/StageBridge/results/v1"

mkdir -p $OUTPUT

# ============================================
# Part 1: General paper numbers (cell composition, mutations, etc.)
# ============================================
echo ""
echo "Part 1: Computing general paper numbers..."
python scripts/compute_paper_numbers.py \
    --data-dir $DATA \
    --results-dir $RESULTS \
    --output $OUTPUT/paper_numbers.json

# ============================================
# Part 2: Drift alignment (requires GPU + model inference)
# ============================================
echo ""
echo "Part 2: Computing drift alignment..."
python scripts/compute_drift_alignment.py \
    --results-dir $RESULTS \
    --data-dir $DATA \
    --output $OUTPUT/drift_alignment.json

# ============================================
# Summary
# ============================================
echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
echo "Paper numbers: $OUTPUT/paper_numbers.json"
echo "Drift alignment: $OUTPUT/drift_alignment.json"
echo ""
echo "Date: $(date)"
