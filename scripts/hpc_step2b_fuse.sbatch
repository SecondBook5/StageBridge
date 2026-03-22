#!/bin/bash
#SBATCH --job-name=fuse_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/fuse_embedding_%j.log

# =============================================================================
# Stage 2b: Fuse HLCA + LuCA Embeddings
# =============================================================================
# Creates fused embedding after both HLCA and LuCA mappings complete.
# This runs AFTER both Stage 1 (HLCA) and Stage 2 (LuCA) finish.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Stage 2b: Fuse HLCA + LuCA Embeddings"
echo "Started: $(date)"
echo "=============================================="

cd /home/booka/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/chaunzt1/stagebridge_env

DATA=/scratch/chaunzt1/stagebridge
OUTPUT=$DATA/processed/luad_evo/reference_geometry
FUSED="$OUTPUT/fused_embedding.parquet"
HLCA_LATENT="$OUTPUT/hlca_mapping/hlca_latent.h5ad"
LUCA_LATENT="$OUTPUT/luca_mapping/luca_latent.h5ad"

# Skip if fused already exists and is valid
if [ -f "$FUSED" ]; then
    python3 -c "
import pandas as pd
fused = pd.read_parquet('$FUSED')
hlca_cols = [c for c in fused.columns if c.startswith('hlca_latent_')]
luca_cols = [c for c in fused.columns if c.startswith('luca_latent_')]
assert len(hlca_cols) > 0, 'Missing HLCA'
assert len(luca_cols) > 0, 'Missing LuCA'
print(f'✓ Fused valid: {len(fused)} cells, {len(hlca_cols)} HLCA + {len(luca_cols)} LuCA dims')
" && {
        echo "✓ Fused embedding already exists and is VALID"
        echo "SKIPPING"
        exit 0
    }
fi

# Verify prerequisites
if [ ! -f "$HLCA_LATENT" ]; then
    echo "ERROR: HLCA latent not found: $HLCA_LATENT"
    exit 1
fi

if [ ! -f "$LUCA_LATENT" ]; then
    echo "ERROR: LuCA latent not found: $LUCA_LATENT"
    exit 1
fi

echo "Creating fused embedding..."
python3 << 'PYTHON_EOF'
import pandas as pd
import numpy as np
import anndata
from pathlib import Path

output_dir = Path("/scratch/chaunzt1/stagebridge/processed/luad_evo/reference_geometry")

# Load HLCA and LuCA latents
print("Loading HLCA latent...")
hlca = anndata.read_h5ad(output_dir / "hlca_mapping" / "hlca_latent.h5ad")
print(f"  HLCA: {hlca.n_obs} cells, {hlca.n_vars} dims")

print("Loading LuCA latent...")
luca = anndata.read_h5ad(output_dir / "luca_mapping" / "luca_latent.h5ad")
print(f"  LuCA: {luca.n_obs} cells, {luca.n_vars} dims")

# Verify same cells
assert list(hlca.obs.index) == list(luca.obs.index), 'Cell ID mismatch between HLCA and LuCA'

# L2 normalize each
print("L2-normalizing latents...")
hlca_norm = hlca.X / (np.linalg.norm(hlca.X, axis=1, keepdims=True) + 1e-8)
luca_norm = luca.X / (np.linalg.norm(luca.X, axis=1, keepdims=True) + 1e-8)

# Concatenate
fused = np.concatenate([hlca_norm, luca_norm], axis=1).astype(np.float32)
print(f"  Fused: {fused.shape[0]} cells, {fused.shape[1]} dims")

# Build DataFrame
df = pd.DataFrame(index=hlca.obs.index)
df.index.name = 'cell_id'

# Add metadata
for col in ['donor_id', 'stage', 'sample_id']:
    if col in hlca.obs.columns:
        df[col] = hlca.obs[col].values

# Add HLCA latent columns
for i in range(hlca.n_vars):
    df[f'hlca_latent_{i}'] = hlca_norm[:, i]

# Add LuCA latent columns
for i in range(luca.n_vars):
    df[f'luca_latent_{i}'] = luca_norm[:, i]

# Add fused latent columns
for i in range(fused.shape[1]):
    df[f'fused_latent_{i}'] = fused[:, i]

# Add reference mode
df['reference_mode_used'] = 'dual'

# Save
output_path = output_dir / 'fused_embedding.parquet'
df.to_parquet(output_path, index=True)
print(f"✓ Saved fused embedding: {output_path}")
print(f"  Shape: {len(df)} cells x {fused.shape[1]} dims")
PYTHON_EOF

echo ""
echo "=============================================="
echo "Fuse Complete: $(date)"
echo "Output: $FUSED"
echo "=============================================="
