#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/booka/StageBridge/logs/backend_compare_%j.out
#SBATCH --error=/home/booka/StageBridge/logs/backend_compare_%j.err

set -euo pipefail
cd /home/booka/StageBridge
source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env

DATA="/scratch/chaunzt1/stagebridge"
OUTPUT_DIR="${DATA}/runs/spatial_benchmark"

echo "=== Backend Comparison ==="
echo "Start: $(date)"

python -c "
import json
import pandas as pd
import numpy as np
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
backends = ['tangram', 'destvi', 'tacco', 'cell2location']

results = {}
for backend in backends:
    backend_dir = output_dir / backend
    metrics_file = backend_dir / 'upstream_metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            results[backend] = json.load(f)
        print(f'{backend}: loaded metrics')
    else:
        print(f'{backend}: MISSING metrics')

# Compare and rank
if results:
    df = pd.DataFrame(results).T
    print('\n=== Metrics Comparison ===')
    print(df.to_string())

    # Simple ranking by mean_entropy (moderate ~0.6 is good) and coverage
    df['entropy_score'] = 1 - abs(df.get('mean_entropy', 0.5) - 0.6)
    df['coverage_score'] = df.get('coverage', 0)
    df['composite'] = 0.5 * df['entropy_score'] + 0.5 * df['coverage_score']

    best = df['composite'].idxmax()
    print(f'\n=== RECOMMENDATION ===')
    print(f'Best backend: {best.upper()}')
    print(f'Composite score: {df.loc[best, \"composite\"]:.3f}')

    # Save comparison
    comparison = {
        'metrics': df.to_dict(),
        'recommendation': best,
        'composite_scores': df['composite'].to_dict()
    }
    with open(output_dir / 'backend_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'\nSaved: {output_dir}/backend_comparison.json')
"

echo "End: $(date)"
echo ""
echo "Results: ${OUTPUT_DIR}/backend_comparison.json"
