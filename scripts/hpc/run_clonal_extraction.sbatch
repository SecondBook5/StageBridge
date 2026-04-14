#!/bin/bash
#SBATCH --job-name=clonal_cnv
#SBATCH --output=logs/clonal_%j.out
#SBATCH --error=logs/clonal_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --mail-type=END,FAIL

# InferCNV-based clonal pattern extraction
# Classifies donors into patterns: 1a (direct), 1b (branched), 2 (independent)

set -e

# Load environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /scratch/chaunzt1/envs/stagebridge_env

cd ~/StageBridge

# Paths
DATA=/scratch/chaunzt1/stagebridge
SNRNA=$DATA/processed/luad_evo/snrna_with_celltypes.h5ad
OUTPUT=$DATA/runs/clonal

mkdir -p $OUTPUT
mkdir -p logs

echo "============================================"
echo "Clonal Pattern Extraction (InferCNV)"
echo "============================================"
echo "Start: $(date)"
echo "snRNA: $SNRNA"
echo "Output: $OUTPUT"
echo ""

# Run clonal extraction
python -m stagebridge.pipelines.run_clonal_extraction \
    --spatial-h5ad $SNRNA \
    --output-dir $OUTPUT \
    --patient-key donor_id \
    --stage-key stage \
    --cell-type-key cell_type \
    --reference-types AT2 \
    --window-size 100 \
    --min-clone-cells 5

echo ""
echo "============================================"
echo "Clonal extraction complete: $(date)"
echo "============================================"

# Show results
if [ -f "$OUTPUT/clonal_patterns.json" ]; then
    echo "Clonal patterns:"
    cat $OUTPUT/clonal_patterns.json
fi
