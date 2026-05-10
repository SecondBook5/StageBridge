#!/bin/bash
# HPO with DDP - single Optuna process, 4 GPUs via DDP for faster trials
# Usage: srun --gpus=4 --cpus-per-task=32 --mem=128G --time=24:00:00 --pty bash scripts/hpc/run_hpo_ddp.sh

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_DIR="/data1/chaunzt1/stagebridge/runs/hpo"
GW_CHECKPOINT="${DATA_DIR}/gw_alignment"
STUDY_NAME="stagebridge_hpo_$(date +%Y%m%d_%H%M%S)"
OPTUNA_DB="sqlite:///${OUTPUT_DIR}/optuna.db"
N_TRIALS=30
N_EPOCHS=15

mkdir -p "${OUTPUT_DIR}"

echo "Starting HPO with DDP: ${STUDY_NAME}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Trials: ${N_TRIALS}"
echo "  GPUs: 4 (DDP)"

# Activate environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

# Run with DDP - single process coordinates, 4 GPUs train each trial
python -m stagebridge.pipelines.run_hpo_ddp \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --gw-checkpoint "${GW_CHECKPOINT}" \
    --n-trials "${N_TRIALS}" \
    --n-epochs "${N_EPOCHS}" \
    --storage "${OPTUNA_DB}" \
    --study-name "${STUDY_NAME}" \
    --world-size 4 \
    --seed 42

echo "HPO complete. Best params: ${OUTPUT_DIR}/best_params.json"
