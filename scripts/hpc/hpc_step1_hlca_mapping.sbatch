#!/bin/bash
#SBATCH --job-name=hlca_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/hlca_mapping_%j.log

# =============================================================================
# Stage 1: HLCA Reference Mapping
# =============================================================================
#
# Maps query cells to HLCA (healthy) reference space using model-based
# scArches surgery. Produces:
# - hlca_embedding.parquet (L2-normalized latents, 30 dims)
# - cell_type predictions from scANVI model
#
# Prerequisites:
# - snrna_qc_normalized_with_ensg.h5ad (from add_ensembl_ids.py)
# - HLCA reference and HubModel cache
#
# Runtime: ~2-4 hours for 800k cells
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 1: HLCA Reference Mapping"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA=/scratch/chaunzt1/stagebridge
QUERY=$DATA/processed/luad_evo/snrna_qc_normalized_with_ensg.h5ad
HLCA_REF=$DATA/references/hlca/hlca_reference.h5ad
OUTPUT=$DATA/processed/luad_evo/reference_geometry

# GPU setup
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  Query: $QUERY"
echo "  HLCA Reference: $HLCA_REF"
echo "  Output: $OUTPUT"
echo ""

# =============================================================================
# SKIP-IF-EXISTS: Check if valid output already exists
# =============================================================================
HLCA_LABELS="$OUTPUT/hlca_mapping/hlca_labels.parquet"
HLCA_LATENT="$OUTPUT/hlca_mapping/hlca_latent.h5ad"

check_hlca_valid() {
    [[ -f "$HLCA_LABELS" ]] && [[ -f "$HLCA_LATENT" ]] && \
    python3 -c "
import pandas as pd
df = pd.read_parquet('$HLCA_LABELS')
assert 'hlca_label' in df.columns, 'Missing hlca_label'
n_valid = (df['hlca_label'].notna() & (df['hlca_label'] != 'unknown')).sum()
assert n_valid > 0, 'No valid cell type labels'
print(f'✓ HLCA valid: {len(df)} cells, {n_valid} with cell types')
" 2>/dev/null
}

if check_hlca_valid; then
    echo "=============================================="
    echo "✓ HLCA MAPPING ALREADY EXISTS AND IS VALID"
    echo "  Labels: $HLCA_LABELS"
    echo "  Latent: $HLCA_LATENT"
    echo "SKIPPING - no work needed"
    echo "=============================================="
    exit 0
fi

echo "HLCA mapping needs to run (missing or invalid output)"
echo ""

# Verify inputs
if [ ! -f "$QUERY" ]; then
    echo "ERROR: Query file not found: $QUERY"
    echo "Run: python scripts/add_ensembl_ids.py first"
    exit 1
fi

if [ ! -f "$HLCA_REF" ]; then
    echo "ERROR: HLCA reference not found: $HLCA_REF"
    echo "Run: python -m stagebridge.pipelines.download_references --download_hlca"
    exit 1
fi

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Run HLCA mapping (HLCA-only mode)
echo ""
echo "Running HLCA reference mapping..."
python -m stagebridge.pipelines.run_reference \
    --data-root "$DATA" \
    --snrna "$QUERY" \
    --hlca "$HLCA_REF" \
    --output-dir "$OUTPUT" \
    --hpc \
    --hlca-only \
    --use-model-based \
    --k-neighbors 50

# Verify outputs
echo ""
echo "Verifying outputs..."
EXPECTED_FILES=(
    "$OUTPUT/hlca_embedding.parquet"
    "$OUTPUT/hlca_mapping/hlca_latent.h5ad"
    "$OUTPUT/hlca_mapping/hlca_labels.parquet"
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

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some expected outputs are missing"
fi

echo ""
echo "=============================================="
echo "Stage 1 Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - $OUTPUT/hlca_embedding.parquet"
echo "  - $OUTPUT/hlca_mapping/hlca_labels.parquet (cell types)"
echo ""
echo "Next: Stage 2 (LuCA Mapping)"
