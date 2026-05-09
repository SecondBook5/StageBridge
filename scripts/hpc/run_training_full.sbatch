#!/bin/bash
#SBATCH --job-name=sb_train
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# StageBridge full training: 5 folds × 3 seeds = 15 runs
# 4 runs in parallel (1 GPU each), 4 batches total

set -e

DATA_DIR="/data1/chaunzt1/stagebridge/processed/luad_evo/canonical"
OUTPUT_DIR="/data1/chaunzt1/stagebridge/runs/training"
HPO_DIR="/data1/chaunzt1/stagebridge/runs/hpo"
GW_CHECKPOINT="${DATA_DIR}/gw_alignment"

N_FOLDS=5
N_SEEDS=3
SEEDS=(42 123 456)

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "Starting full training: ${N_FOLDS} folds × ${N_SEEDS} seeds = $((N_FOLDS * N_SEEDS)) runs"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Using best params from: ${HPO_DIR}/best_params.json"

# Activate environment
source ~/.bashrc
conda activate stagebridge_env

# Build list of all (fold, seed) combinations
runs=()
for fold in $(seq 0 $((N_FOLDS - 1))); do
    for seed_idx in $(seq 0 $((N_SEEDS - 1))); do
        runs+=("${fold}:${SEEDS[$seed_idx]}")
    done
done

echo "Total runs: ${#runs[@]}"

# Process in batches of 4
batch_size=4
for ((i=0; i<${#runs[@]}; i+=batch_size)); do
    batch=("${runs[@]:i:batch_size}")
    echo ""
    echo "=== Batch $((i/batch_size + 1)): ${batch[*]} ==="

    gpu=0
    for run in "${batch[@]}"; do
        fold="${run%:*}"
        seed="${run#*:}"
        run_name="fold${fold}_seed${seed}"
        run_dir="${OUTPUT_DIR}/${run_name}"

        echo "  Launching ${run_name} on GPU ${gpu}"

        CUDA_VISIBLE_DEVICES=$gpu python -m stagebridge.training.trainer \
            --data-dir "${DATA_DIR}" \
            --output-dir "${run_dir}" \
            --hpo-params "${HPO_DIR}/best_params.json" \
            --fold-idx "${fold}" \
            --seed "${seed}" \
            --use-gw-fusion \
            --batch-size 64 &

        ((gpu++))
    done

    echo "  Waiting for batch to complete..."
    wait
done

echo ""
echo "All training runs complete."
echo "Results in: ${OUTPUT_DIR}"

# Aggregate results
echo "Aggregating results..."
python -c "
import json
from pathlib import Path
import numpy as np

output_dir = Path('${OUTPUT_DIR}')
results = []

for run_dir in sorted(output_dir.glob('fold*_seed*')):
    summary_path = run_dir / 'training_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        results.append({
            'run': run_dir.name,
            'best_val_loss': summary.get('best_val_loss'),
            'best_epoch': summary.get('best_epoch'),
        })

if results:
    val_losses = [r['best_val_loss'] for r in results if r['best_val_loss'] is not None]
    print(f'Completed runs: {len(results)}/15')
    print(f'Val loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}')

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump({
            'runs': results,
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)),
        }, f, indent=2)
"
