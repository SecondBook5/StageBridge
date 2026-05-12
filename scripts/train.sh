#!/bin/bash
# Train all 15 models (5 folds x 3 seeds) on 4 GPUs
# v1.2: Fixed stats token shortcut
#
# Each GPU runs: train -> inference -> figures, then next job
# Resumable: re-run to continue from where it stopped
#
# Usage: bash scripts/train.sh [--hpo-params path/to/best_params.json]
#        bash scripts/train.sh [--hpo-db path/to/optuna.db]

# Don't use set -e - we handle errors ourselves to continue on failure

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_BASE="/data1/chaunzt1/stagebridge/outputs/v1.2"
HPO_DIR="/data1/chaunzt1/stagebridge/outputs/v1.2/hpo"
NUM_GPUS=4
HPO_PARAMS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hpo-params)
            HPO_PARAMS="$2"
            shift 2
            ;;
        --hpo-dir)
            HPO_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use best_params.json from HPO dir if not explicitly specified
if [[ -z "$HPO_PARAMS" && -f "${HPO_DIR}/best_params.json" ]]; then
    HPO_PARAMS="${HPO_DIR}/best_params.json"
fi

# Validate HPO params exist
if [[ -n "$HPO_PARAMS" && ! -f "$HPO_PARAMS" ]]; then
    echo "ERROR: HPO params file not found: $HPO_PARAMS"
    exit 1
fi

if [[ -n "$HPO_PARAMS" ]]; then
    echo "Using HPO params: $HPO_PARAMS"
    cat "$HPO_PARAMS"
    echo ""
else
    echo "WARNING: No HPO params found, using trainer defaults"
fi

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
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "HPO params: ${HPO_PARAMS:-'(using defaults)'}"
echo "Jobs: ${#all_jobs[@]} | GPUs: $NUM_GPUS"
echo "Each GPU: train -> inference -> attention figure"
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

    # RESUME LOGIC:
    # - Figure exists -> fully complete, skip everything
    # - Checkpoint exists -> skip train, run inference + figures
    # - Neither -> run everything

    if [[ -f "$figpath" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: SKIP (figure exists)"
        return 0
    fi

    if [[ -f "$checkpoint" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Checkpoint exists, skipping to inference..."
    else
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Starting train..."

        mkdir -p "$outdir"

        # Build trainer command
        TRAIN_CMD="CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.training.trainer \
            --data-dir $DATA_DIR \
            --output-dir $outdir \
            --fold-idx $fold \
            --seed $seed \
            --ssl-epochs 50 \
            --transition-epochs 100 \
            --batch-size 256"

        # Add HPO params if provided
        if [[ -n "$HPO_PARAMS" ]]; then
            TRAIN_CMD="$TRAIN_CMD --hpo-params $HPO_PARAMS"
        fi

        # 1. Train (with error handling - continue on failure)
        if ! eval $TRAIN_CMD >> "$logfile" 2>&1; then
            echo "[GPU $gpu] fold_${fold}/seed_${seed}: TRAIN FAILED - see $logfile" | tee -a "$logfile"
            echo "FAILED" > "${outdir}/FAILED"
            return 1
        fi

        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Train done."
    fi

    # 2. Inference (if checkpoint exists)
    if [[ -f "$checkpoint" ]]; then
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Starting inference..."

        mkdir -p "$infdir"

        if ! CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.pipelines.infer \
            --checkpoint "$checkpoint" \
            --data-dir "$DATA_DIR" \
            --output-dir "$infdir" \
            --fold-idx $fold \
            --save-embeddings \
            --save-attention \
            >> "$logfile" 2>&1; then
            echo "[GPU $gpu] fold_${fold}/seed_${seed}: INFERENCE FAILED - see $logfile" | tee -a "$logfile"
            return 1
        fi

        echo "[GPU $gpu] fold_${fold}/seed_${seed}: Inference done. Making figures..."

        # 3. Attention + drift figures (non-fatal if fails)
        python scripts/quick_attention_fig.py "$infdir" "$figpath" "$DATA_DIR" >> "$logfile" 2>&1 || \
            echo "[GPU $gpu] fold_${fold}/seed_${seed}: Figure generation failed (non-fatal)" | tee -a "$logfile"

    else
        echo "[GPU $gpu] fold_${fold}/seed_${seed}: No checkpoint found!" | tee -a "$logfile"
        return 1
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
# Jobs continue even if one fails
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    (
        for job in ${gpu_jobs[$gpu]}; do
            fold=${job%:*}
            seed=${job#*:}
            run_job $gpu $fold $seed || true  # Continue even if job fails
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
echo "Check for failures:"
echo "  find ${OUTPUT_BASE} -name 'FAILED'"
echo "  grep -l 'FAILED\|Error\|Exception' ${OUTPUT_BASE}/logs/*.log"
echo ""
echo "Check attention results:"
echo "  grep -r 'Attention:' ${OUTPUT_BASE}/logs/*.log"
