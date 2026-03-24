#!/bin/bash
# Submit full stratified benchmark: 9 samples x 4 backends x 2 label sources
# Usage: ./workflow/slurm/submit_full_benchmark.sh [--dry-run]
#
# This runs the complete spatial deconvolution benchmark comparing:
# - 4 backends: Tangram, DestVI, TACCO, Cell2location
# - 2 label sources: HLCA vs LuCA cell type annotations
# - 9 Visium samples stratified across LUAD progression stages

set -e

# Configuration
ENV="/scratch/chaunzt1/stagebridge_env"
DATA="/scratch/chaunzt1/stagebridge"
SNRNA="${DATA}/processed/luad_evo/snrna_with_celltypes.h5ad"
SPATIAL="${DATA}/processed/luad_evo/spatial_merged.h5ad"
LABELS="${DATA}/processed/luad_evo/reference_geometry/cell_types.parquet"
OUTPUT_BASE="${DATA}/runs/spatial_benchmark/full_stratified"
LOGS="${DATA}/runs/logs/benchmark"

# 9 samples stratified by stage (from spatial_merged.h5ad sample_id)
SAMPLES=(
    "GSM9226174_P4_Normal"
    "GSM9226175_P4_AAH"
    "GSM9226176_P4_AIS"
    "GSM9226177_P5_Normal"
    "GSM9226178_P5_AAH"
    "GSM9226179_P5_MIA"
    "GSM9226180_P6_Normal"
    "GSM9226181_P6_AIS"
    "GSM9226182_P6_LUAD"
)

# Time limits per backend (production runs, not debug)
TIME_TANGRAM="2:00:00"
TIME_DESTVI="4:00:00"
TIME_TACCO="4:00:00"
TIME_C2L="8:00:00"

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "DRY RUN: Commands will be printed but not executed"
            ;;
    esac
done

mkdir -p "${LOGS}"

# Label sources to run
SOURCES=("hlca" "luca")

echo "=============================================="
echo "Full Stratified Spatial Benchmark"
echo "=============================================="
echo "Samples: ${#SAMPLES[@]}"
echo "Backends: Tangram, DestVI, TACCO, Cell2location"
echo "Labels: HLCA, LuCA"
echo "Total jobs: $((${#SAMPLES[@]} * 4 * 2)) = $((${#SAMPLES[@]} * 8))"
echo ""

# Submit array jobs per backend
for SRC in "${SOURCES[@]}"; do
    OUTPUT="${OUTPUT_BASE}/${SRC}"
    mkdir -p "${OUTPUT}"

    # Label args
    if [[ "$SRC" == "luca" ]]; then
        LABEL_ARGS="--label-source luca --labels-parquet ${LABELS}"
    else
        LABEL_ARGS="--label-source hlca"
    fi

    echo "--- ${SRC^^} Labels ---"

    # Tangram (GPU) - array job for all 9 samples
    CMD_TG="sbatch --parsable \
        --job-name=bench_tangram_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_TANGRAM} \
        --mem=256G \
        --cpus-per-task=8 \
        --array=0-8 \
        --output=${LOGS}/tangram_${SRC}_%A_%a.log \
        --error=${LOGS}/tangram_${SRC}_%A_%a.err \
        --wrap='
SAMPLES=(${SAMPLES[@]})
SAMPLE=\${SAMPLES[\$SLURM_ARRAY_TASK_ID]}
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample \$SAMPLE \
    --sample-col sample_id \
    --backends tangram \
    ${LABEL_ARGS}
'"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY] Tangram: $CMD_TG"
    else
        JOB_TG=$(eval $CMD_TG)
        echo "  Tangram submitted: ${JOB_TG} (array 0-8)"
    fi

    # DestVI (GPU)
    CMD_DV="sbatch --parsable \
        --job-name=bench_destvi_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_DESTVI} \
        --mem=256G \
        --cpus-per-task=8 \
        --array=0-8 \
        --output=${LOGS}/destvi_${SRC}_%A_%a.log \
        --error=${LOGS}/destvi_${SRC}_%A_%a.err \
        --wrap='
SAMPLES=(${SAMPLES[@]})
SAMPLE=\${SAMPLES[\$SLURM_ARRAY_TASK_ID]}
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample \$SAMPLE \
    --sample-col sample_id \
    --backends destvi \
    ${LABEL_ARGS}
'"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY] DestVI: $CMD_DV"
    else
        JOB_DV=$(eval $CMD_DV)
        echo "  DestVI submitted: ${JOB_DV} (array 0-8)"
    fi

    # Cell2location (GPU, longer time)
    CMD_C2L="sbatch --parsable \
        --job-name=bench_cell2loc_${SRC} \
        --partition=gpu \
        --gres=gpu:2 \
        --time=${TIME_C2L} \
        --mem=256G \
        --cpus-per-task=8 \
        --array=0-8 \
        --output=${LOGS}/cell2loc_${SRC}_%A_%a.log \
        --error=${LOGS}/cell2loc_${SRC}_%A_%a.err \
        --wrap='
SAMPLES=(${SAMPLES[@]})
SAMPLE=\${SAMPLES[\$SLURM_ARRAY_TASK_ID]}
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample \$SAMPLE \
    --sample-col sample_id \
    --backends cell2location \
    ${LABEL_ARGS}
'"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY] Cell2location: $CMD_C2L"
    else
        JOB_C2L=$(eval $CMD_C2L)
        echo "  Cell2location submitted: ${JOB_C2L} (array 0-8)"
    fi

    # TACCO (CPU only)
    CMD_TC="sbatch --parsable \
        --job-name=bench_tacco_${SRC} \
        --partition=cpu \
        --time=${TIME_TACCO} \
        --mem=256G \
        --cpus-per-task=16 \
        --array=0-8 \
        --output=${LOGS}/tacco_${SRC}_%A_%a.log \
        --error=${LOGS}/tacco_${SRC}_%A_%a.err \
        --wrap='
SAMPLES=(${SAMPLES[@]})
SAMPLE=\${SAMPLES[\$SLURM_ARRAY_TASK_ID]}
module load miniforge3 2>/dev/null
eval \"\$(conda shell.bash hook)\"
conda activate ${ENV}
cd /home/booka/StageBridge
python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna ${SNRNA} \
    --spatial ${SPATIAL} \
    --output_dir ${OUTPUT} \
    --sample \$SAMPLE \
    --sample-col sample_id \
    --backends tacco \
    ${LABEL_ARGS}
'"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY] TACCO: $CMD_TC"
    else
        JOB_TC=$(eval $CMD_TC)
        echo "  TACCO submitted: ${JOB_TC} (array 0-8)"
    fi

    echo ""
done

echo "=============================================="
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f ${LOGS}/*.log"
echo ""
echo "Expected completion times:"
echo "  Tangram:      ~2 hours per sample"
echo "  DestVI:       ~4 hours per sample"
echo "  TACCO:        ~4 hours per sample"
echo "  Cell2location: ~8 hours per sample"
echo "=============================================="
