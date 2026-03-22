#!/bin/bash
#SBATCH --job-name=tangram
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/home/booka/StageBridge/logs/tangram_%j.out
#SBATCH --error=/home/booka/StageBridge/logs/tangram_%j.err

set -euo pipefail
cd /home/booka/StageBridge
source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env
export CUDA_VISIBLE_DEVICES=0

DATA="/scratch/chaunzt1/stagebridge"
mkdir -p /home/booka/StageBridge/logs

echo "=== Tangram Backend ==="
echo "Start: $(date)"

python -c "
import anndata as ad
from pathlib import Path
from stagebridge.spatial_backends import TangramBackend

snrna = ad.read_h5ad('${DATA}/processed/luad_evo/snrna_merged.h5ad')
spatial = ad.read_h5ad('${DATA}/processed/luad_evo/spatial_merged.h5ad')

print(f'snRNA: {snrna.shape}, Spatial: {spatial.shape}')

backend = TangramBackend(mode='clusters', n_epochs=1000)
output_dir = Path('${DATA}/runs/spatial_benchmark/tangram')
output_dir.mkdir(parents=True, exist_ok=True)

result = backend.map(snrna, spatial, output_dir=output_dir)
result.save(output_dir)
print('Tangram complete!')
"

echo "End: $(date)"
