#!/bin/bash
# Run inference on all completed full model checkpoints
# Cycles through available GPUs

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_BASE="/data1/chaunzt1/stagebridge/outputs/v1.1"
NUM_GPUS=4

gpu=0
for fold in 0 1 2 3 4; do
  for seed in 42 43 44; do
    # Skip fold_0/seed_44 if still training (remove this line when done)
    if [[ "$fold" == "0" && "$seed" == "44" ]]; then
      echo "Skipping fold_${fold}/seed_${seed} (still training)"
      continue
    fi

    checkpoint="${OUTPUT_BASE}/full/fold_${fold}/seed_${seed}/checkpoints/best_checkpoint.pt"
    outdir="${OUTPUT_BASE}/inference/full/fold_${fold}/seed_${seed}"

    if [[ -f "$checkpoint" ]]; then
      echo "GPU $gpu: Running inference for fold_${fold}/seed_${seed}"
      CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.pipelines.infer \
        --checkpoint "$checkpoint" \
        --data-dir "$DATA_DIR" \
        --output-dir "$outdir" \
        --fold-idx $fold \
        --save-embeddings \
        --save-attention

      if [[ $? -eq 0 ]]; then
        echo "  Done: $outdir"
      else
        echo "  FAILED: fold_${fold}/seed_${seed}"
      fi
    else
      echo "Skipping fold_${fold}/seed_${seed}: checkpoint not found"
    fi

    gpu=$(( (gpu + 1) % NUM_GPUS ))
  done
done

echo "All inference complete"
