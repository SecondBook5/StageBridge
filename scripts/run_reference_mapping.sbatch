#!/bin/bash
#SBATCH --job-name=dual_ref_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/dual_ref_map_%j.log

# =============================================================================
# Dual-Reference Mapping (HLCA + LuCA)
# =============================================================================
#
# Maps query cells to BOTH HLCA (healthy) and LuCA (cancer) reference spaces.
# Uses model-based scArches surgery for principled projection.
#
# Prerequisites:
# 1. HLCA reference downloaded (via download_references.py)
# 2. LuCA model trained (via retrain_luca.sbatch) - COMPLETED
# 3. Query data prepared (snrna_qc_normalized.h5ad)
#
# Output:
# - hlca_embedding.parquet (L2-normalized HLCA latents, 30 dims)
# - luca_embedding.parquet (L2-normalized LuCA latents, 10 dims)
# - fused_embedding.parquet (concatenated, 40 dims)
# - reference_confidence.parquet (calibrated confidence scores)
# =============================================================================

echo "=============================================="
echo "Dual-Reference Mapping (HLCA + LuCA)"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge || cd $HOME/StageBridge || echo "Warning: Could not cd to StageBridge"
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA=/scratch/chaunzt1/stagebridge
QUERY=$DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad
OUTPUT=$DATA/processed/luad_evo/reference_geometry

# HLCA reference and model
HLCA_REF=$DATA/references/hlca/hlca_reference.h5ad
HLCA_MODEL=$DATA/references/hlca/hub_cache

# LuCA reference and NEWLY TRAINED model
LUCA_REF=$DATA/references/luca/luca_core_atlas.h5ad
LUCA_MODEL=$DATA/references/luca/retrained_model/scanvi_model

# Create symlink so the mapping script finds the new LuCA model
# The script searches for: references/luca/luca_scanvi_model
mkdir -p $DATA/references/luca
ln -sfn $LUCA_MODEL $DATA/references/luca/luca_scanvi_model
echo "Created symlink: references/luca/luca_scanvi_model -> retrained_model/scanvi_model"

# Verify paths exist
echo ""
echo "Verifying paths..."
for path in "$QUERY" "$HLCA_REF" "$LUCA_REF" "$LUCA_MODEL"; do
    if [ -e "$path" ]; then
        echo "  [OK] $path"
    else
        echo "  [MISSING] $path"
        exit 1
    fi
done

# Check for HLCA HubModel
if [ -d "$HLCA_MODEL" ]; then
    echo "  [OK] HLCA HubModel cache exists"
else
    echo "  [WARN] HLCA HubModel cache not found - will use k-NN fallback for HLCA"
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set environment for GPU
export CUDA_VISIBLE_DEVICES=0

# Find HLCA model in HubModel cache
HLCA_MODEL_DIR=$(find $DATA/references/hlca/hub_cache -name "model.pt" -type f 2>/dev/null | head -1 | xargs dirname)
if [ -n "$HLCA_MODEL_DIR" ]; then
    echo "  [OK] Found HLCA model: $HLCA_MODEL_DIR"
else
    echo "  [WARN] HLCA model not found in hub_cache"
fi

# Run dual-reference mapping with model-based projection
# --hpc: memory-efficient chunked processing
# --use-model-based: use scArches surgery (not k-NN)
# --hlca-model / --luca-model: explicit model paths
python -m stagebridge.pipelines.run_reference \
    --data-root "$DATA" \
    --snrna "$QUERY" \
    --hlca "$HLCA_REF" \
    --luca "$LUCA_REF" \
    --output-dir "$OUTPUT" \
    --hpc \
    --use-model-based \
    --hlca-model "$HLCA_MODEL_DIR" \
    --luca-model "$LUCA_MODEL" \
    --luca-latent-key "X_scANVI" \
    --k-neighbors 50

echo ""
echo "=============================================="
echo "Reference Mapping Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs in: $OUTPUT"
echo "  - hlca_embedding.parquet"
echo "  - luca_embedding.parquet"
echo "  - fused_embedding.parquet"
echo "  - reference_confidence.parquet"
echo ""
echo "Next step: python -m stagebridge.pipelines.run_spatial_benchmark"
