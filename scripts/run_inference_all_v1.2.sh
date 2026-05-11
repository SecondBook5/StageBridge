#!/bin/bash
# Run inference on all v1.2 checkpoints
# 4 jobs in parallel (one per GPU)

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_BASE="/data1/chaunzt1/stagebridge/outputs/v1.2"
NUM_GPUS=4

jobs=()
for fold in 0 1 2 3 4; do
  for seed in 42 43 44; do
    jobs+=("${fold}:${seed}")
  done
done

echo "Running ${#jobs[@]} inference jobs across $NUM_GPUS GPUs"

for ((i=0; i<${#jobs[@]}; i+=NUM_GPUS)); do
  pids=()

  for ((j=0; j<NUM_GPUS && i+j<${#jobs[@]}; j++)); do
    job=${jobs[i+j]}
    fold=${job%:*}
    seed=${job#*:}
    gpu=$j

    checkpoint="${OUTPUT_BASE}/full/fold_${fold}/seed_${seed}/checkpoints/best_checkpoint.pt"
    outdir="${OUTPUT_BASE}/inference/full/fold_${fold}/seed_${seed}"

    if [[ -f "$checkpoint" ]]; then
      echo "GPU $gpu: fold_${fold}/seed_${seed}"
      CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.pipelines.infer \
        --checkpoint "$checkpoint" \
        --data-dir "$DATA_DIR" \
        --output-dir "$outdir" \
        --fold-idx $fold \
        --save-embeddings \
        --save-attention &
      pids+=($!)
    else
      echo "Skipping fold_${fold}/seed_${seed}: checkpoint not found"
    fi
  done

  for pid in "${pids[@]}"; do
    wait $pid
  done
  echo "Batch complete"
done

echo "All inference complete"
