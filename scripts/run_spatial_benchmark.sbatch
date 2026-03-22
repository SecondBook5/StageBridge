#!/bin/bash
#SBATCH --job-name=spatial_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/booka/StageBridge/logs/spatial_benchmark_%j.out
#SBATCH --error=/home/booka/StageBridge/logs/spatial_benchmark_%j.err

# Spatial Backend Benchmark
# Compares Tangram, DestVI, TACCO, and Cell2location
# Selects canonical backend for downstream training

set -euo pipefail

# Setup
cd /home/booka/projects/StageBridge
source ~/.bashrc
conda activate stagebridge

export CUDA_VISIBLE_DEVICES=0

# Data paths
DATA="${DATA:-/home/booka/data}"
SNRNA="${DATA}/processed/luad_evo/snrna_qc_normalized.h5ad"
SPATIAL="${DATA}/processed/luad_evo/spatial_merged.h5ad"
OUTPUT_DIR="${DATA}/runs/spatial_benchmark"

# Ensure logs directory exists
mkdir -p /home/booka/StageBridge/logs

echo "============================================================"
echo "Spatial Backend Benchmark"
echo "============================================================"
echo "Start time: $(date)"
echo "SNRNA: ${SNRNA}"
echo "SPATIAL: ${SPATIAL}"
echo "OUTPUT: ${OUTPUT_DIR}"
echo ""

# Verify inputs exist
if [ ! -f "${SNRNA}" ]; then
    echo "ERROR: snRNA file not found: ${SNRNA}"
    exit 1
fi

if [ ! -f "${SPATIAL}" ]; then
    echo "ERROR: Spatial file not found: ${SPATIAL}"
    exit 1
fi

# Run benchmark
python stagebridge/pipelines/run_spatial_benchmark.py \
    --snrna "${SNRNA}" \
    --spatial "${SPATIAL}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Benchmark complete"
echo "End time: $(date)"
echo "Results: ${OUTPUT_DIR}/backend_comparison.json"
echo "============================================================"
