#!/bin/bash
# Train all 15 models (5 folds x 3 seeds) on 4 GPUs
# v1.2: Fixed stats token shortcut
#
# Each GPU runs: train -> inference -> attention figure, then next job
#
# Usage: bash scripts/train_all_v1.2.sh

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_BASE="/data1/chaunzt1/stagebridge/outputs/v1.2"
NUM_GPUS=4

# All 15 jobs: fold:seed pairs
all_jobs=(
    "0:42" "1:42" "2:42" "3:42"
    "4:42" "0:43" "1:43" "2:43"
    "3:43" "4:43" "0:44" "1:44"
    "2:44" "3:44" "4:44"
)

echo "=============================================="
echo "StageBridge v1.2 Full Training"
echo "=============================================="
echo "Fix: Stats token removed from context refiner"
echo "Each GPU: train -> inference -> attention figure"
echo "Jobs: ${#all_jobs[@]} | GPUs: $NUM_GPUS"
echo "=============================================="

mkdir -p "$OUTPUT_BASE/logs"
mkdir -p "$OUTPUT_BASE/figures"

# Function to run one complete job on one GPU
run_job() {
    local gpu=$1
    local fold=$2
    local seed=$3

    local outdir="${OUTPUT_BASE}/full/fold_${fold}/seed_${seed}"
    local infdir="${OUTPUT_BASE}/inference/full/fold_${fold}/seed_${seed}"
    local logfile="${OUTPUT_BASE}/logs/fold_${fold}_seed_${seed}.log"
    local checkpoint="${outdir}/checkpoints/best_checkpoint.pt"
    local figpath="${OUTPUT_BASE}/figures/attention_fold_${fold}_seed_${seed}.png"

    # RESUME CHECK: Skip if attention figure exists (means job completed)
    if [[ -f "$figpath" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: SKIP (already complete)"
        return 0
    fi

    # RESUME CHECK: Skip training if checkpoint exists
    if [[ -f "$checkpoint" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Checkpoint exists, skipping train..."
    else
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Starting train..."

        mkdir -p "$outdir"

        # 1. Train
        CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.training.trainer \
            --data-dir "$DATA_DIR" \
            --output-dir "$outdir" \
            --fold-idx $fold \
            --seed $seed \
            --ssl-epochs 50 \
            --transition-epochs 100 \
            --batch-size 64 \
            >> "$logfile" 2>&1

        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Train done."
    fi

    # 2. Inference (if checkpoint exists)
    if [[ -f "$checkpoint" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Starting inference..."

        mkdir -p "$infdir"

        CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.pipelines.infer \
            --checkpoint "$checkpoint" \
            --data-dir "$DATA_DIR" \
            --output-dir "$infdir" \
            --fold-idx $fold \
            --save-embeddings \
            --save-attention \
            >> "$logfile" 2>&1

        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Inference done. Making figures..."

        # 3. Attention + drift figures
        python scripts/quick_attention_fig.py "$infdir" "$figpath" "$DATA_DIR" >> "$logfile" 2>&1

    else
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: No checkpoint found!" >> "$logfile"
    fi

    echo "[GPU $gpu] fold_${fold}/seed_${seed}: Complete"
}

# Distribute jobs across GPUs
# GPU 0 gets jobs 0,4,8,12
# GPU 1 gets jobs 1,5,9,13
# GPU 2 gets jobs 2,6,10,14
# GPU 3 gets jobs 3,7,11

gpu_jobs=()
for ((g=0; g<NUM_GPUS; g++)); do
    gpu_jobs[$g]=""
done

for ((i=0; i<${#all_jobs[@]}; i++)); do
    gpu=$((i % NUM_GPUS))
    gpu_jobs[$gpu]+="${all_jobs[$i]} "
done

# Launch one process per GPU that runs its jobs sequentially
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
        for job in ${gpu_jobs[$gpu]}; do
            fold=${job%:*}
            seed=${job#*:}
            run_job $gpu $fold $seed
        done
    ) &
    pids+=($!)
    echo "Launched GPU $gpu with jobs: ${gpu_jobs[$gpu]}"
done

echo ""
echo "All GPUs running. Logs: tail -f ${OUTPUT_BASE}/logs/*.log"
echo ""

# Wait for all GPUs to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "=============================================="
echo "All training complete!"
echo "=============================================="
echo ""
echo "Check attention results:"
echo "  grep -r 'Attention:' ${OUTPUT_BASE}/logs/*.log"
