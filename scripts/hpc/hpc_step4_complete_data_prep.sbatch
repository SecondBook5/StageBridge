#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/data_prep_%j.log

# =============================================================================
# Stage 4: Complete Data Preparation
# =============================================================================
#
# Builds canonical training format from:
# - snRNA data with reference embeddings
# - Spatial data with backend deconvolution
# - WES features
#
# Produces:
# - cells.parquet (all cells with latents, cell types, WES features)
# - neighborhoods.parquet (9-token niche structure)
# - stage_edges.parquet (valid progression transitions)
# - split_manifest.json (5-fold donor-held-out CV)
# - feature_spec.yaml (documentation)
#
# Prerequisites:
# - Stage 2 (Reference mapping) completed
# - Stage 3 (Spatial benchmark) completed
#
# Runtime: ~1-2 hours
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 4: Complete Data Preparation"
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
WES=$DATA/processed/luad_evo/wes_features.parquet
SPATIAL_BACKEND=$DATA/processed/luad_evo/spatial_benchmark
REF_GEOMETRY=$DATA/processed/luad_evo/reference_geometry
OUTPUT=$DATA/processed/luad_evo/canonical

echo ""
echo "Configuration:"
echo "  snRNA: $SNRNA"
echo "  Spatial: $SPATIAL"
echo "  WES: $WES"
echo "  Spatial Backend: $SPATIAL_BACKEND"
echo "  Reference Geometry: $REF_GEOMETRY"
echo "  Output: $OUTPUT"
echo ""

# Verify prerequisites
echo "Verifying prerequisites..."
REQUIRED=(
    "$SNRNA"
    "$SPATIAL"
    "$SPATIAL_BACKEND/cell_type_proportions.parquet"
    "$REF_GEOMETRY/fused_embedding.parquet"
    "$REF_GEOMETRY/cell_types.parquet"
)

ALL_OK=true
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done

# WES is optional
if [ -f "$WES" ]; then
    echo "  [OK] $WES"
else
    echo "  [OPTIONAL] $WES (not found, will proceed without WES features)"
    WES=""  # Clear path
fi

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "ERROR: Prerequisites not met"
    exit 1
fi

mkdir -p "$OUTPUT"

# First, update snRNA with reference embeddings and cell types
echo ""
echo "Updating snRNA with reference embeddings..."
python -c "
import anndata as ad
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
print('  Loading snRNA...')
snrna = ad.read_h5ad('$SNRNA')
print(f'    {snrna.n_obs} cells, {snrna.n_vars} genes')

# Load reference embeddings
print('  Loading reference embeddings...')
fused = pd.read_parquet('$REF_GEOMETRY/fused_embedding.parquet')
cell_types = pd.read_parquet('$REF_GEOMETRY/cell_types.parquet')

print(f'    Fused: {len(fused)} cells')
print(f'    Cell types: {len(cell_types)} cells')

# Match cells
common_cells = snrna.obs_names.intersection(fused['cell_id'])
print(f'    Matched: {len(common_cells)} cells')

# Add cell types
cell_type_map = dict(zip(cell_types['cell_id'], cell_types['cell_type']))
snrna.obs['cell_type'] = snrna.obs_names.map(cell_type_map)
print(f'    Cell types mapped: {snrna.obs[\"cell_type\"].notna().sum()}')

# Add fused latent to obsm
fused_idx = fused.set_index('cell_id')
latent_cols = [c for c in fused.columns if c.startswith('fused_latent_')]
print(f'    Latent dims: {len(latent_cols)}')

# Create latent matrix
latent_matrix = np.zeros((snrna.n_obs, len(latent_cols)))
for i, cell_id in enumerate(snrna.obs_names):
    if cell_id in fused_idx.index:
        latent_matrix[i] = fused_idx.loc[cell_id, latent_cols].values

snrna.obsm['X_fused'] = latent_matrix
print(f'    Added X_fused to obsm')

# Save updated snRNA
output_path = Path('$OUTPUT') / 'snrna_with_latents.h5ad'
print(f'  Saving to {output_path}...')
snrna.write_h5ad(output_path)
print('  Done!')
"

# Run complete data prep
echo ""
echo "Running complete data preparation..."

# Check if WES file exists
if [ -n "$WES" ] && [ -f "$WES" ]; then
    WES_ARG="--wes $WES"
else
    # Create dummy WES file
    echo "  Creating placeholder WES features..."
    python -c "
import pandas as pd
import anndata as ad

snrna = ad.read_h5ad('$SNRNA')
donors = snrna.obs['donor_id'].unique() if 'donor_id' in snrna.obs else ['unknown']

wes_df = pd.DataFrame({
    'donor_id': donors,
    'tmb': [0.0] * len(donors),
    'smoking_signature': [0.0] * len(donors),
    'uv_signature': [0.0] * len(donors),
})
wes_df.to_parquet('$OUTPUT/wes_features_placeholder.parquet', index=False)
print(f'  Created placeholder WES for {len(donors)} donors')
"
    WES_ARG="--wes $OUTPUT/wes_features_placeholder.parquet"
fi

python -m stagebridge.pipelines.complete_data_prep \
    --snrna "$OUTPUT/snrna_with_latents.h5ad" \
    --spatial "$SPATIAL" \
    $WES_ARG \
    --spatial_backend_dir "$SPATIAL_BACKEND" \
    --output_dir "$OUTPUT" \
    --n_folds 5

# Verify outputs
echo ""
echo "Verifying outputs..."
EXPECTED_FILES=(
    "$OUTPUT/cells.parquet"
    "$OUTPUT/neighborhoods.parquet"
    "$OUTPUT/stage_edges.parquet"
    "$OUTPUT/split_manifest.json"
    "$OUTPUT/feature_spec.yaml"
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

# Show data summary
echo ""
echo "Data Summary:"
python -c "
import pandas as pd
import json
import yaml

cells = pd.read_parquet('$OUTPUT/cells.parquet')
print(f'  Cells: {len(cells):,}')
print(f'  Donors: {cells[\"donor_id\"].nunique()}')
print(f'  Stages: {sorted(cells[\"stage\"].unique().tolist())}')

try:
    neighborhoods = pd.read_parquet('$OUTPUT/neighborhoods.parquet')
    print(f'  Neighborhoods: {len(neighborhoods):,}')
except:
    print('  Neighborhoods: error loading')

with open('$OUTPUT/split_manifest.json') as f:
    splits = json.load(f)
print(f'  CV Folds: {len(splits[\"folds\"])}')

with open('$OUTPUT/feature_spec.yaml') as f:
    spec = yaml.safe_load(f)
print(f'  Latent dim: {spec.get(\"cells\", {}).get(\"latent_dim\", \"unknown\")}')
"

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: Some expected outputs are missing"
fi

echo ""
echo "=============================================="
echo "Stage 4 Complete: $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  - $OUTPUT/cells.parquet"
echo "  - $OUTPUT/neighborhoods.parquet"
echo "  - $OUTPUT/stage_edges.parquet"
echo "  - $OUTPUT/split_manifest.json"
echo "  - $OUTPUT/feature_spec.yaml"
echo ""
echo "Next: Stage 5 (Model Training)"
