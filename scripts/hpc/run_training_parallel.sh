#!/bin/bash
# Train all folds x seeds in parallel across 4 GPUs
# Usage: bash scripts/hpc/run_training_parallel.sh

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_DIR="/data1/chaunzt1/stagebridge/runs/full"
HPO_PARAMS="/data1/chaunzt1/stagebridge/runs/hpo/best_params.json"

N_FOLDS=5
N_SEEDS=3
N_GPUS=4

SSL_EPOCHS=50
TRANSITION_EPOCHS=100

# Activate environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

mkdir -p "${OUTPUT_DIR}"

echo "Starting training: ${N_FOLDS} folds x ${N_SEEDS} seeds = $((N_FOLDS * N_SEEDS)) runs"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  HPO params: ${HPO_PARAMS}"
echo "  GPUs: ${N_GPUS}"

# Build array of all (fold, seed) pairs
declare -a JOBS=()
for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed in $(seq 0 $((N_SEEDS - 1))); do
        JOBS+=("${fold}:${seed}")
    done
done

# Process jobs in batches of N_GPUS
job_idx=0
while [ $job_idx -lt ${#JOBS[@]} ]; do
    # Launch up to N_GPUS jobs
    for gpu in $(seq 0 $((N_GPUS - 1))); do
        if [ $job_idx -lt ${#JOBS[@]} ]; then
            IFS=':' read -r fold seed <<< "${JOBS[$job_idx]}"
            run_dir="${OUTPUT_DIR}/fold_${fold}/seed_${seed}"

            echo "  GPU ${gpu}: fold=${fold} seed=${seed}"

            CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.training.trainer \
                --data-dir "${DATA_DIR}" \
                --output-dir "${run_dir}" \
                --fold-idx "${fold}" \
                --seed "${seed}" \
                --ssl-epochs "${SSL_EPOCHS}" \
                --transition-epochs "${TRANSITION_EPOCHS}" \
                --hpo-params "${HPO_PARAMS}" \
                --batch-size 64 &

            job_idx=$((job_idx + 1))
        fi
    done

    echo "Waiting for batch to complete..."
    wait
    echo "Batch complete."
done

echo "All training complete."
echo "Results in: ${OUTPUT_DIR}"
