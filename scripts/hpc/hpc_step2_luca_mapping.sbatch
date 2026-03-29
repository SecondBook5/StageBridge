#!/bin/bash
#SBATCH --job-name=luca_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/luca_mapping_%j.log

# =============================================================================
# Stage 2: LuCA Reference Mapping
# =============================================================================
#
# Maps query cells to LuCA (cancer) reference space using model-based
# scArches surgery. Produces:
# - luca_embedding.parquet (L2-normalized latents, 10 dims)
# - fused_embedding.parquet (HLCA + LuCA concatenated, 40 dims)
# - reference_confidence.parquet (calibrated confidence scores)
# - cell_types.parquet (HLCA + LuCA cell type predictions)
#
# IMPORTANT: LuCA model may require pandas 1.5.x environment.
# If this job fails with BlockPlacement error, use the pandas_compat env.
#
# Prerequisites:
# - Stage 1 (HLCA mapping) completed
# - LuCA core atlas and retrained scANVI model
#
# Runtime: ~2-4 hours for 800k cells
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 2: LuCA Reference Mapping"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh

# Try primary environment first, fall back to pandas_compat if needed
if conda activate /scratch/chaunzt1/stagebridge_env 2>/dev/null; then
    echo "Using primary stagebridge_env"
else
    echo "Primary env failed, trying pandas_compat..."
    conda activate /scratch/chaunzt1/luca_compat || {
        echo "ERROR: No compatible environment found"
        echo "Create one with: conda create -n luca_compat python=3.11 && pip install pandas==1.5.3 scvi-tools"
        exit 1
    }
fi

# Data paths
DATA=/scratch/chaunzt1/stagebridge
QUERY=$DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad
LUCA_REF=$DATA/references/luca/luca_core_atlas.h5ad
OUTPUT=$DATA/processed/luad_evo/reference_geometry

# LuCA model path (from retrain_luca.sbatch)
LUCA_MODEL=$DATA/references/luca/retrained_model/scanvi_model
if [ ! -d "$LUCA_MODEL" ]; then
    # Try alternative path
    LUCA_MODEL=$DATA/references/luca/luca_scanvi_model
fi

# GPU setup
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  Query: $QUERY"
echo "  LuCA Reference: $LUCA_REF"
echo "  LuCA Model: $LUCA_MODEL"
echo "  Output: $OUTPUT"
echo ""

# =============================================================================
# SKIP-IF-EXISTS: Check if valid LuCA output already exists
# (Fused embedding created separately after both HLCA and LuCA complete)
# =============================================================================
LUCA_LABELS="$OUTPUT/luca_mapping/luca_labels.parquet"
LUCA_LATENT="$OUTPUT/luca_mapping/luca_latent.h5ad"

check_luca_valid() {
    [[ -f "$LUCA_LABELS" ]] && [[ -f "$LUCA_LATENT" ]] && \
    python3 -c "
import pandas as pd
df = pd.read_parquet('$LUCA_LABELS')
assert 'luca_label' in df.columns, 'Missing luca_label'
n_valid = (df['luca_label'].notna() & (df['luca_label'] != 'unknown')).sum()
assert n_valid > 0, 'No valid cell type labels'
print(f'✓ LuCA valid: {len(df)} cells, {n_valid} with cell types')
" 2>/dev/null
}

if check_luca_valid; then
    echo "=============================================="
    echo "✓ LUCA MAPPING ALREADY EXISTS AND IS VALID"
    echo "  Labels: $LUCA_LABELS"
    echo "  Latent: $LUCA_LATENT"
    echo "  Fused: $FUSED"
    echo "SKIPPING - no work needed"
    echo "=============================================="
    exit 0
fi

echo "LuCA mapping needs to run (missing or invalid output)"
echo ""

# Note: HLCA runs in parallel - fused embedding created by separate step
HLCA_LATENT="$OUTPUT/hlca_mapping/hlca_latent.h5ad"

# Verify inputs
if [ ! -f "$LUCA_REF" ]; then
    echo "ERROR: LuCA reference not found: $LUCA_REF"
    exit 1
fi

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check pandas version (LuCA may need 1.5.x)
PANDAS_VERSION=$(python -c "import pandas; print(pandas.__version__)")
echo "Pandas version: $PANDAS_VERSION"

if [[ "$PANDAS_VERSION" == 2.* ]]; then
    echo "WARNING: Pandas 2.x detected. LuCA model may fail."
    echo "If job fails, create luca_compat environment with pandas 1.5.3"
fi

# Create symlink for model discovery (if needed)
if [ -d "$DATA/references/luca/retrained_model/scanvi_model" ]; then
    ln -sfn "$DATA/references/luca/retrained_model/scanvi_model" "$DATA/references/luca/luca_scanvi_model" 2>/dev/null || true
fi

# Run LuCA mapping (LuCA-only mode, will fuse with existing HLCA)
echo ""
echo "Running LuCA reference mapping..."
python -m stagebridge.pipelines.run_reference \
    --data-root "$DATA" \
    --snrna "$QUERY" \
    --luca "$LUCA_REF" \
    --output-dir "$OUTPUT" \
    --hpc \
    --luca-only \
    --use-model-based \
    --luca-latent-key "X_scANVI" \
    --k-neighbors 50

# Now run BOTH to create fused embedding
echo ""
echo "Creating fused embedding (HLCA + LuCA)..."
python -m stagebridge.pipelines.run_reference \
    --data-root "$DATA" \
    --snrna "$QUERY" \
    --output-dir "$OUTPUT" \
    --hpc \
    --use-model-based \
    --k-neighbors 50

# Verify outputs
echo ""
echo "Verifying outputs..."
EXPECTED_FILES=(
    "$OUTPUT/luca_embedding.parquet"
    "$OUTPUT/fused_embedding.parquet"
    "$OUTPUT/reference_confidence.parquet"
    "$OUTPUT/cell_types.parquet"
    "$OUTPUT/reference_manifest.json"
    "$OUTPUT/diagnostics_report.json"
)

ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK] $f ($SIZE)"
    else
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done

# Print diagnostics summary
if [ -f "$OUTPUT/diagnostics_report.json" ]; then
    echo ""
    echo "Diagnostics Summary:"
    python -c "
import json
with open('$OUTPUT/diagnostics_report.json') as f:
    d = json.load(f)
print(f\"  Status: {d.get('status', 'unknown')}\")
print(f\"  Mode readiness: {d.get('mode_readiness', {})}\")
if d.get('warnings'):
    print(f\"  Warnings: {len(d['warnings'])}\")
"
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some expected outputs are missing"
fi

echo ""
echo "=============================================="
echo "Stage 2 Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - $OUTPUT/hlca_embedding.parquet (30 dims)"
echo "  - $OUTPUT/luca_embedding.parquet (10 dims)"
echo "  - $OUTPUT/fused_embedding.parquet (40 dims)"
echo "  - $OUTPUT/reference_confidence.parquet"
echo "  - $OUTPUT/cell_types.parquet"
echo ""
echo "Next: Stage 4 (Data Preparation) after Stage 3 (Spatial) completes"
