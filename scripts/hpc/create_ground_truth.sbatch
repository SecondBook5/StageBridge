#!/bin/bash
#SBATCH --job-name=ground_truth
#SBATCH --partition=cpu
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/home/booka/StageBridge/logs/ground_truth_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/ground_truth_%j.err

# Create ground truth interaction labels for LUAD data
# Uses AMICI-style labeling: known sender-receiver rules + distance thresholds

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Activate environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd /home/booka/StageBridge

# Data paths
DATA_ROOT="/data1/chaunzt1/stagebridge"
SNRNA="${DATA_ROOT}/processed/luad_evo/snrna_with_celltypes.h5ad"
SPATIAL="${DATA_ROOT}/processed/luad_evo/spatial_merged.h5ad"
CELLS="${DATA_ROOT}/processed/luad_evo/raw/cells.parquet"
NEIGHBORHOODS="${DATA_ROOT}/processed/luad_evo/canonical/neighborhoods.parquet"
OUTPUT="${DATA_ROOT}/results/ground_truth"

echo ""
echo "Creating ground truth labels..."
echo "  snRNA: $SNRNA"
echo "  Spatial: $SPATIAL"
echo "  Cells: $CELLS"
echo "  Neighborhoods: $NEIGHBORHOODS"
echo "  Output: $OUTPUT"
echo ""

# Run the labeler on all available data
python scripts/create_ground_truth.py \
    --snrna "$SNRNA" \
    --spatial "$SPATIAL" \
    --cells "$CELLS" \
    --neighborhoods "$NEIGHBORHOODS" \
    --output "$OUTPUT" \
    --cell-type-key cell_type

echo ""
echo "Done: $(date)"
