#!/bin/bash
################################################################################
# HPC Setup Script for StageBridge V1
################################################################################

set -e

echo "=========================================="
echo "StageBridge HPC Environment Setup"
echo "=========================================="

# 1. Create conda environment
echo ""
echo "[1/6] Creating conda environment..."
if conda env list | grep -q "stagebridge"; then
    echo "  Environment 'stagebridge' already exists. Activating..."
else
    echo "  Creating new environment..."
    conda create -n stagebridge python=3.11 -y
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate stagebridge

# 2. Install PyTorch with GPU support
echo ""
echo "[2/6] Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install core scientific packages
echo ""
echo "[3/6] Installing scientific packages..."
conda install numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y

# 4. Install single-cell analysis tools
echo ""
echo "[4/6] Installing single-cell tools..."
pip install anndata scanpy scvi-tools squidpy

# 5. Install spatial mapping backends
echo ""
echo "[5/6] Installing spatial backends..."
pip install tangram-sc scvi-tools tacco

# 6. Install additional dependencies
echo ""
echo "[6/6] Installing additional packages..."
pip install umap-learn phate networkx pot tqdm pyyaml

# Install StageBridge in development mode
echo ""
echo "Installing StageBridge..."
pip install -e .

echo ""
echo "=========================================="
echo " HPC Environment Setup Complete!"
echo "=========================================="
echo ""
echo "To activate: conda activate stagebridge"
echo ""
