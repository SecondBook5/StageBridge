#!/bin/bash
#SBATCH --job-name=sb_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err

# StageBridge HPO with 4 parallel Optuna workers
# Each worker uses 1 GPU, Optuna DB coordinates trials

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_DIR="/data1/chaunzt1/stagebridge/runs/hpo"
GW_CHECKPOINT="${DATA_DIR}/gw_alignment"
STUDY_NAME="stagebridge_hpo_$(date +%Y%m%d)"
OPTUNA_DB="sqlite:///${OUTPUT_DIR}/optuna.db"
N_TRIALS=30
N_EPOCHS=15

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "Starting HPO: ${STUDY_NAME}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Trials: ${N_TRIALS}"
echo "  Workers: 4 (1 GPU each)"

# Activate environment
source ~/.bashrc
conda activate stagebridge_env

# Launch 4 workers, each pinned to 1 GPU
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.pipelines.run_hpo \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --gw-checkpoint "${GW_CHECKPOINT}" \
        --n-trials "${N_TRIALS}" \
        --n-epochs "${N_EPOCHS}" \
        --n-jobs 1 \
        --storage "${OPTUNA_DB}" \
        --study-name "${STUDY_NAME}" \
        --seed $((42 + gpu)) &
    echo "  Launched worker on GPU ${gpu}"
done

echo "Waiting for all workers..."
wait

echo "HPO complete. Best params saved to ${OUTPUT_DIR}/best_params.json"
