#!/bin/bash
#SBATCH --job-name=download_hlca
#SBATCH --output=/scratch/chaunzt1/stagebridge/logs/download_hlca_%j.out
#SBATCH --error=/scratch/chaunzt1/stagebridge/logs/download_hlca_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# ============================================================================
# Download HLCA via scvi-tools HubModel
# ============================================================================

set -e

echo "========================================"
echo "Downloading HLCA (Human Lung Cell Atlas)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Time: $(date)"
echo ""

# Setup
SCRATCH_DIR="/scratch/chaunzt1/stagebridge"
HLCA_DIR="${SCRATCH_DIR}/processed/HLCA"
OUTPUT_FILE="${HLCA_DIR}/hlca_scanvi.h5ad"

mkdir -p ${HLCA_DIR}
mkdir -p ${SCRATCH_DIR}/logs

# Check if already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "HLCA already exists: $OUTPUT_FILE"
    echo "Delete it first if you want to re-download"
    exit 0
fi

# Load modules (adjust for your HPC)
# module load python/3.10
# module load cuda/11.8

# Activate conda environment (adjust for your setup)
# source ~/miniconda3/bin/activate stagebridge

echo "Downloading HLCA via scvi-tools..."
python3 << 'PYTHON_SCRIPT'
import os
os.environ['HF_HOME'] = '/scratch/chaunzt1/stagebridge/.cache/huggingface'

from scvi.hub import HubModel
import warnings
warnings.filterwarnings('ignore')

print("Pulling from HuggingFace Hub...")
hubmodel = HubModel.pull_from_huggingface_hub(
    repo_name="scvi-tools/human-lung-cell-atlas-scanvi",
)

print(f"Model loaded: {type(hubmodel.model)}")
print(f"AnnData: {hubmodel.adata.n_obs:,} cells × {hubmodel.adata.n_vars:,} genes")

# Get latent representation
print("Computing scANVI latent representation...")
hubmodel.adata.obsm["X_scANVI"] = hubmodel.model.get_latent_representation()
print(f"scANVI dims: {hubmodel.adata.obsm['X_scANVI'].shape[1]}")

# Save
output_path = "/data1/chaunzt1/stagebridge/processed/HLCA/hlca_scanvi.h5ad"
print(f"Saving to {output_path}...")
hubmodel.adata.write_h5ad(output_path)
print("Done!")
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "Download complete!"
echo "Output: $OUTPUT_FILE"
ls -lh $OUTPUT_FILE
echo "========================================"
