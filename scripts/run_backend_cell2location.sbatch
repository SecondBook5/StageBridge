#!/bin/bash
#SBATCH --job-name=cell2loc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/home/booka/StageBridge/logs/cell2location_%j.out
#SBATCH --error=/home/booka/StageBridge/logs/cell2location_%j.err

set -euo pipefail
cd /home/booka/StageBridge
source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env
export CUDA_VISIBLE_DEVICES=0

DATA="/scratch/chaunzt1/stagebridge"
mkdir -p /home/booka/StageBridge/logs

echo "=== Cell2location Backend ==="
echo "Start: $(date)"

# Check/install cell2location
python -c "import cell2location" || pip install cell2location --quiet

python -c "
import anndata as ad
from pathlib import Path
from stagebridge.spatial_backends import Cell2locationBackend

snrna = ad.read_h5ad('${DATA}/processed/luad_evo/snrna_merged.h5ad')
spatial = ad.read_h5ad('${DATA}/processed/luad_evo/spatial_merged.h5ad')

print(f'snRNA: {snrna.shape}, Spatial: {spatial.shape}')

backend = Cell2locationBackend(max_epochs_ref=250, max_epochs_spatial=2500)
output_dir = Path('${DATA}/runs/spatial_benchmark/cell2location')
output_dir.mkdir(parents=True, exist_ok=True)

result = backend.map(snrna, spatial, output_dir=output_dir)
result.save(output_dir)
print('Cell2location complete!')
"

echo "End: $(date)"
