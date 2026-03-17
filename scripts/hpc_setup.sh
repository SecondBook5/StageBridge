#!/bin/bash
# ============================================================================
# StageBridge HPC Setup Script
# Run this on the HPC to set up the data directories and download HLCA
# ============================================================================

set -e

echo "========================================"
echo "StageBridge HPC Setup"
echo "========================================"

# Configuration
SCRATCH_DIR="/scratch/chaunzt1/stagebridge"
PROCESSED_DIR="${SCRATCH_DIR}/processed"

# Create directory structure
echo ""
echo "Creating directories..."
mkdir -p ${PROCESSED_DIR}/HLCA
mkdir -p ${PROCESSED_DIR}/LuCA
mkdir -p ${PROCESSED_DIR}/luad_evo
mkdir -p ${SCRATCH_DIR}/outputs
mkdir -p ${SCRATCH_DIR}/results
mkdir -p ${SCRATCH_DIR}/figures

echo "  ✓ ${PROCESSED_DIR}/HLCA"
echo "  ✓ ${PROCESSED_DIR}/LuCA"
echo "  ✓ ${PROCESSED_DIR}/luad_evo"
echo "  ✓ ${SCRATCH_DIR}/outputs"
echo "  ✓ ${SCRATCH_DIR}/results"
echo "  ✓ ${SCRATCH_DIR}/figures"

# Check for LuCA (should be transferred via scp)
echo ""
echo "Checking for LuCA..."
if [ -f "${PROCESSED_DIR}/LuCA/luca_extended.h5ad" ]; then
    echo "  ✓ LuCA found: luca_extended.h5ad"
else
    echo "  ✗ LuCA not found"
    echo "  Transfer from local machine:"
    echo "    scp /home/booka/data/stagebridge/processed/LuCA/luca_extended.h5ad booka@islogin01.mskcc.org:${PROCESSED_DIR}/LuCA/"
fi

# Check for evolutionary data TARs
echo ""
echo "Checking for evolutionary data..."
if ls ${PROCESSED_DIR}/luad_evo/*.tar* 1> /dev/null 2>&1; then
    echo "  ✓ TAR files found in luad_evo/"
    echo "  Run processing script to generate snrna_merged.h5ad"
else
    if [ -f "${PROCESSED_DIR}/luad_evo/snrna_merged.h5ad" ]; then
        echo "  ✓ snrna_merged.h5ad found"
    else
        echo "  ⚠ No evolutionary data found"
    fi
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Transfer LuCA (if not done): scp luca_extended.h5ad ..."
echo "  2. Download HLCA: sbatch scripts/download_hlca.sbatch"
echo "  3. Process evolutionary TARs (if needed)"
echo "  4. Run notebook: jupyter notebook StageBridge_V1.ipynb"
echo "========================================"
