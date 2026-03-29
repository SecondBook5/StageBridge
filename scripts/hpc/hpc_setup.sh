#!/bin/bash
################################################################################
# HPC Setup Script for StageBridge V1 on Iris Cluster
################################################################################

set -e

echo "=========================================="
echo "StageBridge HPC Environment Setup (Iris)"
echo "=========================================="

# Load miniforge module (Iris-specific)
echo ""
echo "[0/7] Loading miniforge module..."
module load miniforge3

# 1. Create conda environment
echo ""
echo "[1/7] Creating conda environment..."
if conda env list | grep -q "stagebridge"; then
    echo "  Environment 'stagebridge' already exists. Activating..."
else
    echo "  Creating new environment..."
    conda create -n stagebridge python=3.11 -y
fi

eval "$(conda shell.bash hook)"
conda activate stagebridge

# 2. Install PyTorch with GPU support
echo ""
echo "[2/7] Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install core scientific packages
echo ""
echo "[3/7] Installing scientific packages..."
conda install numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y

# 4. Install single-cell analysis tools
echo ""
echo "[4/7] Installing single-cell tools..."
pip install anndata scanpy scvi-tools squidpy

# 5. Install spatial mapping backends
echo ""
echo "[5/7] Installing spatial backends..."
pip install tangram-sc scvi-tools tacco

# 6. Install additional dependencies
echo ""
echo "[6/7] Installing additional packages..."
pip install umap-learn phate networkx pot tqdm pyyaml

# 7. Install Jupyter kernel support (for Open OnDemand)
echo ""
echo "[7/7] Installing Jupyter kernel support..."
conda install ipykernel -y
python -m ipykernel install --user --name=stagebridge --display-name "StageBridge (Python 3.11)"

# Install StageBridge in development mode
echo ""
echo "Installing StageBridge..."
pip install -e .

echo ""
echo "=========================================="
echo " HPC Environment Setup Complete!"
echo "=========================================="
echo ""
echo "To activate: module load miniforge3 && conda activate stagebridge"
echo ""
echo "For Jupyter (Open OnDemand):"
echo "  1. Go to Iris Open OnDemand"
echo "  2. Launch Jupyter"
echo "  3. In Environment Setup field, add:"
echo "     module load miniforge3"
echo "     conda activate stagebridge"
echo "  4. Select 'StageBridge (Python 3.11)' kernel"
echo ""
