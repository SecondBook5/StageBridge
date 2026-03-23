#!/bin/bash
# Submit smoke test jobs for all 4 spatial backends in parallel
# Usage: ./workflow/slurm/submit_smoke_test.sh

set -e

# Configuration
ENV="/scratch/chaunzt1/stagebridge_env"
DATA="/scratch/chaunzt1/stagebridge"
SNRNA="${DATA}/processed/luad_evo/snrna_with_celltypes.h5ad"
SPATIAL="${DATA}/processed/luad_evo/spatial_merged.h5ad"
OUTPUT="${DATA}/runs/spatial_benchmark/smoke_test_normal"
SAMPLE="GSM9226174_P4_Normal"
LOGS="${DATA}/runs/logs"

mkdir -p "${OUTPUT}"
mkdir -p "${LOGS}"

echo "Submitting smoke test jobs for 4 backends..."
echo "  Sample: ${SAMPLE}"
echo "  Output: ${OUTPUT}"
echo ""

# Tangram (GPU)
JOB_TG=$(sbatch --parsable \
    --job-name=smoke_tangram \
    --partition=gpu \
    --gres=gpu:1 \
    --time=4:00:00 \
    --mem=128G \
    --cpus-per-task=8 \
    --output="${LOGS}/smoke_tangram_%j.log" \
    --error="${LOGS}/smoke_tangram_%j.err" \
    --wrap="
module load miniforge3
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends tangram
")
echo "Tangram submitted: ${JOB_TG}"

# DestVI (GPU)
JOB_DV=$(sbatch --parsable \
    --job-name=smoke_destvi \
    --partition=gpu \
    --gres=gpu:2 \
    --time=4:00:00 \
    --mem=192G \
    --cpus-per-task=8 \
    --output="${LOGS}/smoke_destvi_%j.log" \
    --error="${LOGS}/smoke_destvi_%j.err" \
    --wrap="
module load miniforge3
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends destvi
")
echo "DestVI submitted: ${JOB_DV}"

# Cell2location (GPU, needs more time)
JOB_C2L=$(sbatch --parsable \
    --job-name=smoke_cell2loc \
    --partition=gpu \
    --gres=gpu:2 \
    --time=6:00:00 \
    --mem=256G \
    --cpus-per-task=8 \
    --output="${LOGS}/smoke_cell2loc_%j.log" \
    --error="${LOGS}/smoke_cell2loc_%j.err" \
    --wrap="
module load miniforge3
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends cell2location
")
echo "Cell2location submitted: ${JOB_C2L}"

# TACCO (CPU only)
JOB_TC=$(sbatch --parsable \
    --job-name=smoke_tacco \
    --partition=cpu \
    --time=2:00:00 \
    --mem=128G \
    --cpus-per-task=16 \
    --output="${LOGS}/smoke_tacco_%j.log" \
    --error="${LOGS}/smoke_tacco_%j.err" \
    --wrap="
module load miniforge3
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends tacco
")
echo "TACCO submitted: ${JOB_TC}"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f ${LOGS}/smoke_*.log"
