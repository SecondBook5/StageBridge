#!/bin/bash
# Train all 15 models (5 folds x 3 seeds) on 4 GPUs
# v1.2: Fixed stats token shortcut
#
# Each job runs on 1 GPU. 4 jobs run in parallel.
#
# Usage: bash scripts/train_all_v1.2.sh
#
# Estimated time: ~12 hours total (4 batches, ~3h each)

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_BASE="/data1/chaunzt1/stagebridge/outputs/v1.2"
NUM_GPUS=4

# All 15 jobs: fold:seed pairs
jobs=(
    "0:42" "1:42" "2:42" "3:42"
    "4:42" "0:43" "1:43" "2:43"
    "3:43" "4:43" "0:44" "1:44"
    "2:44" "3:44" "4:44"
)

echo "=============================================="
echo "StageBridge v1.2 Full Training"
echo "=============================================="
echo "Fix: Stats token removed from context refiner"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "Jobs: ${#jobs[@]} (5 folds x 3 seeds)"
echo "GPUs: $NUM_GPUS (1 job per GPU)"
echo "=============================================="
echo ""

mkdir -p "$OUTPUT_BASE/logs"

# Process in batches of NUM_GPUS
batch_num=0
for ((i=0; i<${#jobs[@]}; i+=NUM_GPUS)); do
    batch_num=$((batch_num + 1))
    pids=()

    echo "========== Batch $batch_num / $(( (${#jobs[@]} + NUM_GPUS - 1) / NUM_GPUS )) =========="

    for ((j=0; j<NUM_GPUS && i+j<${#jobs[@]}; j++)); do
        job=${jobs[i+j]}
        fold=${job%:*}
        seed=${job#*:}
        gpu=$j

        outdir="${OUTPUT_BASE}/full/fold_${fold}/seed_${seed}"
        logfile="${OUTPUT_BASE}/logs/fold_${fold}_seed_${seed}.log"

        mkdir -p "$outdir"

        echo "  GPU $gpu: fold_${fold}/seed_${seed}"

        CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.training.trainer \
            --data-dir "$DATA_DIR" \
            --output-dir "$outdir" \
            --fold-idx $fold \
            --seed $seed \
            --ssl-epochs 50 \
            --transition-epochs 100 \
            --batch-size 64 \
            > "$logfile" 2>&1 &

        pids+=($!)
    done

    echo ""
    echo "  Logs: tail -f ${OUTPUT_BASE}/logs/*.log"
    echo "  Waiting..."
    echo ""

    # Wait for this batch
    for pid in "${pids[@]}"; do
        wait $pid || echo "  Process $pid failed"
    done

    echo "  Batch $batch_num complete"
    echo ""
done

echo "=============================================="
echo "All training complete!"
echo "=============================================="
