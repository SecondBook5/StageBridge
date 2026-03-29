#!/bin/bash
#SBATCH --job-name=spatial_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/spatial_benchmark_%j.log

# =============================================================================
# Stage 3: Spatial Backend Benchmark
# =============================================================================
#
# Compares 4 spatial deconvolution backends:
# 1. Tangram (scvi-tools integration)
# 2. DestVI (probabilistic deconvolution)
# 3. TACCO (transfer learning)
# 4. Cell2location (reference-based mapping)
#
# Selects canonical backend based on:
# - Cell type concordance with snRNA
# - Spatial coherence metrics
# - Computational efficiency
#
# Produces:
# - cell_type_proportions.parquet (for selected backend)
# - backend_comparison.json (benchmark results)
# - comparison_figures/*.png
#
# This stage runs in PARALLEL with reference mapping (Stage 1-2).
#
# Runtime: ~12-24 hours (4 backends x ~3-6h each)
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 3: Spatial Backend Benchmark"
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
SNRNA=$DATA/processed/luad_evo/snrna_qc_normalized.h5ad
SPATIAL=$DATA/processed/luad_evo/spatial_merged.h5ad
OUTPUT_DIR=$DATA/processed/luad_evo/spatial_benchmark

# GPU setup
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "Configuration:"
echo "  snRNA: $SNRNA"
echo "  Spatial: $SPATIAL"
echo "  Output: $OUTPUT_DIR"
echo ""

# Verify inputs
if [ ! -f "$SNRNA" ]; then
    echo "ERROR: snRNA file not found: $SNRNA"
    exit 1
fi

if [ ! -f "$SPATIAL" ]; then
    echo "ERROR: Spatial file not found: $SPATIAL"
    exit 1
fi

# Show data sizes
echo "Data sizes:"
ls -lh "$SNRNA" "$SPATIAL"
echo ""

# Show GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check cell2location installation
echo "Checking backend installations..."
python -c "
import sys
backends = []
try:
    import tangram
    backends.append(f'tangram {tangram.__version__}')
except: backends.append('tangram MISSING')

try:
    import scvi
    backends.append(f'scvi-tools {scvi.__version__}')
except: backends.append('scvi-tools MISSING')

try:
    import tacco
    backends.append(f'tacco {tacco.__version__}')
except: backends.append('tacco MISSING')

try:
    import cell2location
    backends.append(f'cell2location {cell2location.__version__}')
except: backends.append('cell2location MISSING')

for b in backends:
    print(f'  {b}')
"

# Install missing backends
echo ""
echo "Installing any missing backends..."
pip install cell2location --quiet 2>/dev/null || echo "cell2location install skipped"

mkdir -p "$OUTPUT_DIR"

# Run spatial benchmark
echo ""
echo "Running spatial backend benchmark..."
echo "This will take several hours..."
echo ""

python -m stagebridge.pipelines.run_spatial_benchmark \
    --snrna "$SNRNA" \
    --spatial "$SPATIAL" \
    --output_dir "$OUTPUT_DIR"

# Verify outputs
echo ""
echo "Verifying outputs..."
EXPECTED_FILES=(
    "$OUTPUT_DIR/backend_comparison.json"
    "$OUTPUT_DIR/cell_type_proportions.parquet"
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

# Show benchmark summary
if [ -f "$OUTPUT_DIR/backend_comparison.json" ]; then
    echo ""
    echo "Benchmark Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/backend_comparison.json') as f:
    results = json.load(f)

print('Backend Comparison:')
for backend, metrics in results.get('backends', {}).items():
    score = metrics.get('overall_score', 'N/A')
    time = metrics.get('runtime_minutes', 'N/A')
    print(f'  {backend}: score={score}, time={time}min')

selected = results.get('selected_backend', 'unknown')
print(f'')
print(f'Selected Backend: {selected}')
"
fi

# List all backend outputs
echo ""
echo "Backend outputs:"
for backend in tangram destvi tacco cell2location; do
    DIR="$OUTPUT_DIR/$backend"
    if [ -d "$DIR" ]; then
        COUNT=$(ls -1 "$DIR" 2>/dev/null | wc -l)
        echo "  $backend: $COUNT files"
    else
        echo "  $backend: not run"
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some expected outputs are missing"
fi

echo ""
echo "=============================================="
echo "Stage 3 Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - $OUTPUT_DIR/backend_comparison.json"
echo "  - $OUTPUT_DIR/cell_type_proportions.parquet"
echo "  - $OUTPUT_DIR/{tangram,destvi,tacco,cell2location}/"
echo ""
echo "Next: Stage 4 (Data Preparation) when Stage 2 also completes"
