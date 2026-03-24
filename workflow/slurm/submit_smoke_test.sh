#!/bin/bash
# Submit smoke test jobs for all 4 spatial backends
# Supports label source ablation (HLCA vs LuCA)
# Usage: ./workflow/slurm/submit_smoke_test.sh [hlca|luca|both] [--debug]
#
# --debug: Use minimal epochs (2-5) just to verify code runs (~5 min per backend)

set -e

# Configuration
ENV="/scratch/chaunzt1/stagebridge_env"
DATA="/scratch/chaunzt1/stagebridge"
SNRNA="${DATA}/processed/luad_evo/snrna_with_celltypes.h5ad"
SPATIAL="${DATA}/processed/luad_evo/spatial_merged.h5ad"
LABELS="${DATA}/processed/luad_evo/reference_geometry/cell_types.parquet"
SAMPLE="GSM9226174_P4_Normal"
LOGS="${DATA}/runs/logs"

# Parse arguments
DEBUG_FLAG=""
LABEL_SOURCE="hlca"

for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_FLAG="--debug"
            echo "DEBUG MODE: Using minimal epochs (2-5)"
            ;;
        hlca|luca|both)
            LABEL_SOURCE="$arg"
            ;;
    esac
done

if [[ "$LABEL_SOURCE" == "both" ]]; then
    SOURCES=("hlca" "luca")
else
    SOURCES=("$LABEL_SOURCE")
fi

# Set time limits based on mode
if [[ -n "$DEBUG_FLAG" ]]; then
    TIME_GPU="0:30:00"    # 30 min for debug
    TIME_C2L="0:30:00"    # 30 min for debug
    TIME_CPU="0:30:00"    # 30 min for debug
else
    TIME_GPU="4:00:00"    # 4 hours for full
    TIME_C2L="6:00:00"    # 6 hours for cell2location
    TIME_CPU="2:00:00"    # 2 hours for TACCO
fi

mkdir -p "${LOGS}"

for SRC in "${SOURCES[@]}"; do
    OUTPUT="${DATA}/runs/spatial_benchmark/smoke_test_${SRC}"
    mkdir -p "${OUTPUT}"

    echo "Submitting smoke test jobs for 4 backends with ${SRC^^} labels..."
    echo "  Sample: ${SAMPLE}"
    echo "  Output: ${OUTPUT}"
    echo ""

    # Common args for label source
    if [[ "$SRC" == "luca" ]]; then
        LABEL_ARGS="--label-source luca --labels-parquet ${LABELS}"
    else
        LABEL_ARGS="--label-source hlca"
    fi

    # Tangram (GPU)
    JOB_TG=$(sbatch --parsable \
        --job-name=smoke_tangram_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_GPU} \
        --mem=256G \
        --cpus-per-task=8 \
        --output="${LOGS}/smoke_tangram_${SRC}_%j.log" \
        --error="${LOGS}/smoke_tangram_${SRC}_%j.err" \
        --wrap="
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends tangram \
    ${LABEL_ARGS} ${DEBUG_FLAG}
")
    echo "Tangram (${SRC^^}) submitted: ${JOB_TG}"

    # DestVI (GPU)
    JOB_DV=$(sbatch --parsable \
        --job-name=smoke_destvi_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_GPU} \
        --mem=256G \
        --cpus-per-task=8 \
        --output="${LOGS}/smoke_destvi_${SRC}_%j.log" \
        --error="${LOGS}/smoke_destvi_${SRC}_%j.err" \
        --wrap="
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends destvi \
    ${LABEL_ARGS} ${DEBUG_FLAG}
")
    echo "DestVI (${SRC^^}) submitted: ${JOB_DV}"

    # Cell2location (GPU, needs more time)
    JOB_C2L=$(sbatch --parsable \
        --job-name=smoke_cell2loc_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_C2L} \
        --mem=256G \
        --cpus-per-task=8 \
        --output="${LOGS}/smoke_cell2loc_${SRC}_%j.log" \
        --error="${LOGS}/smoke_cell2loc_${SRC}_%j.err" \
        --wrap="
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends cell2location \
    ${LABEL_ARGS} ${DEBUG_FLAG}
")
    echo "Cell2location (${SRC^^}) submitted: ${JOB_C2L}"

    # TACCO (CPU only)
    JOB_TC=$(sbatch --parsable \
        --job-name=smoke_tacco_${SRC} \
        --partition=cpu \
        --time=${TIME_CPU} \
        --mem=256G \
        --cpus-per-task=16 \
        --output="${LOGS}/smoke_tacco_${SRC}_%j.log" \
        --error="${LOGS}/smoke_tacco_${SRC}_%j.err" \
        --wrap="
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample ${SAMPLE} \
    --sample-col sample_id \
    --backends tacco \
    ${LABEL_ARGS} ${DEBUG_FLAG}
")
    echo "TACCO (${SRC^^}) submitted: ${JOB_TC}"
    echo ""
done

echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f ${LOGS}/smoke_*.log"
