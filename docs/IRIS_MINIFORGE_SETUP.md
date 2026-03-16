# Miniforge Environment Setup for Iris HPC

Complete guide for setting up StageBridge with miniforge on Iris cluster.

---

## Understanding Miniforge on Iris

Iris provides `miniforge3` as a module (not Anaconda/Miniconda due to licensing).

**Key differences from local conda:**
- Must load module first: `module load miniforge3`
- Large ML environments should use `/data/` storage (not `~/.conda/`)
- Jupyter integration requires `ipykernel` registration

---

## Storage Strategy

### Home Directory (~/) - 5-10GB limit
**Use for:**
- Small environments
- Code
- Configuration files

**Default conda env location:** `~/.conda/envs/stagebridge`

### Lab Storage (/data/your_labname/) - Much larger
**Use for:**
- Large ML/AI environments (like StageBridge)
- Data files
- Model outputs

**Custom env location:** `/data/your_labname/envs/stagebridge`

---

## Option 1: Setup in Home Directory (Simple)

Good for: Testing, small projects

```bash
# SSH to Iris
ssh your_username@iris.mskcc.org
cd ~/StageBridge

# Run the standard setup
module load miniforge3
./hpc_setup.sh
```

This creates: `~/.conda/envs/stagebridge`

**Activate:**
```bash
module load miniforge3
conda activate stagebridge
```

---

## Option 2: Setup in Lab Storage (Recommended for StageBridge)

Good for: Large environments, production work

### Step 1: Create Modified Setup Script

Create `hpc_setup_custom.sh`:

```bash
#!/bin/bash
################################################################################
# StageBridge Setup with Custom Environment Location
################################################################################

set -e

# CONFIGURE THIS - Update with your lab name
LAB_NAME="your_labname"
ENV_PATH="/data/${LAB_NAME}/envs/stagebridge"

echo "=========================================="
echo "StageBridge Setup (Custom Location)"
echo "=========================================="
echo "Environment: $ENV_PATH"
echo ""

# Load miniforge
echo "[0/7] Loading miniforge module..."
module load miniforge3

# Create environment in custom location
echo ""
echo "[1/7] Creating conda environment at $ENV_PATH..."
if [ -d "$ENV_PATH" ]; then
    echo "  Environment already exists. Using existing..."
else
    echo "  Creating new environment..."
    conda create -p "$ENV_PATH" python=3.11 -y
fi

# Activate with full path
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# Install PyTorch with GPU support
echo ""
echo "[2/7] Installing PyTorch with CUDA..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core scientific packages
echo ""
echo "[3/7] Installing scientific packages..."
conda install numpy pandas scipy scikit-learn matplotlib seaborn -c conda-forge -y

# Install single-cell analysis tools
echo ""
echo "[4/7] Installing single-cell tools..."
pip install anndata scanpy scvi-tools squidpy

# Install spatial mapping backends
echo ""
echo "[5/7] Installing spatial backends..."
pip install tangram-sc scvi-tools tacco

# Install additional dependencies
echo ""
echo "[6/7] Installing additional packages..."
pip install umap-learn phate networkx pot tqdm pyyaml

# Install Jupyter kernel
echo ""
echo "[7/7] Installing Jupyter kernel..."
conda install ipykernel -y
python -m ipykernel install --user \
    --name=stagebridge \
    --display-name "StageBridge (Python 3.11)"

# Install StageBridge
echo ""
echo "Installing StageBridge..."
pip install -e .

# Create activation helper
echo ""
echo "Creating activation helper..."
cat > activate_stagebridge.sh << 'EOF'
#!/bin/bash
# Helper script to activate StageBridge environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate /data/your_labname/envs/stagebridge
EOF

sed -i "s|your_labname|${LAB_NAME}|g" activate_stagebridge.sh
chmod +x activate_stagebridge.sh

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "Environment location: $ENV_PATH"
echo ""
echo "To activate:"
echo "  source activate_stagebridge.sh"
echo ""
echo "Or manually:"
echo "  module load miniforge3"
echo "  conda activate $ENV_PATH"
echo ""
echo "For Jupyter, use in Environment Setup:"
echo "  module load miniforge3"
echo "  conda activate $ENV_PATH"
echo ""
```

### Step 2: Run Setup

```bash
# Edit with your lab name
nano hpc_setup_custom.sh
# Change: LAB_NAME="your_labname"

# Make executable and run
chmod +x hpc_setup_custom.sh
./hpc_setup_custom.sh
```

### Step 3: Activate Environment

```bash
# Easy way (using helper script)
source activate_stagebridge.sh

# Or manually
module load miniforge3
conda activate /data/your_labname/envs/stagebridge
```

---

## Jupyter Integration (Open OnDemand)

### For Home Directory Environment

**Environment Setup field:**
```bash
module load miniforge3
conda activate stagebridge
```

### For Custom Location Environment

**Environment Setup field:**
```bash
module load miniforge3
conda activate /data/your_labname/envs/stagebridge
```

Then select **"StageBridge (Python 3.11)"** kernel when notebook opens.

---

## Verify Installation

After setup, verify everything works:

```bash
# Activate environment
source activate_stagebridge.sh  # Or use your activation method

# Check Python location
which python
# Should show: /data/your_labname/envs/stagebridge/bin/python

# Test imports
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import stagebridge
print('StageBridge imported successfully!')
"

# Check installed packages
conda list | grep torch
pip list | grep anndata
```

---

## Managing Multiple Environments

If you need to switch between environments:

```bash
# List all environments
conda env list
# Or
conda info --envs

# Deactivate current environment
conda deactivate

# Activate different environment
conda activate stagebridge  # by name
conda activate /data/lab/envs/other_env  # by path
```

---

## Updating Environment

### Add new packages

```bash
source activate_stagebridge.sh

# Via conda
conda install package_name -y

# Via pip
pip install package_name
```

### Update existing packages

```bash
source activate_stagebridge.sh

# Update specific package
conda update package_name

# Update pip packages
pip install --upgrade package_name
```

### Rebuild environment

If something breaks:

```bash
# Remove old environment
conda remove -p /data/your_labname/envs/stagebridge --all -y

# Re-run setup
./hpc_setup_custom.sh
```

---

## Batch Job Template

For SLURM jobs using your custom environment:

```bash
#!/bin/bash
#SBATCH --job-name=stagebridge_job
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/job_%j.out

# Load miniforge
module load miniforge3

# Activate environment (use full path)
eval "$(conda shell.bash hook)"
conda activate /data/your_labname/envs/stagebridge

# Verify GPU
nvidia-smi

# Run your script
python your_script.py
```

---

## Troubleshooting

### "conda: command not found"

```bash
# Make sure module is loaded
module load miniforge3

# Check available modules
module avail miniforge
```

### "Environment not found"

```bash
# List environments
conda env list

# Check if path exists
ls -la /data/your_labname/envs/

# Recreate if needed
./hpc_setup_custom.sh
```

### Jupyter kernel not showing

```bash
# Re-register kernel
source activate_stagebridge.sh
python -m ipykernel install --user \
    --name=stagebridge \
    --display-name "StageBridge (Python 3.11)"

# List kernels
jupyter kernelspec list
```

### Out of space in home directory

If you see "No space left on device":

```bash
# Check usage
df -h ~
du -sh ~/.conda

# Clean conda cache
conda clean --all -y

# Use custom location (Option 2 above)
```

### Import errors in Jupyter

Make sure you:
1. Selected correct kernel ("StageBridge (Python 3.11)")
2. Used correct activation in Environment Setup
3. Kernel was registered from the right environment

---

## Quick Reference

```bash
# Load miniforge (always first!)
module load miniforge3

# Create environment in home
conda create -n stagebridge python=3.11 -y
conda activate stagebridge

# Create environment in /data (recommended)
conda create -p /data/labname/envs/stagebridge python=3.11 -y
conda activate /data/labname/envs/stagebridge

# List environments
conda env list

# Remove environment
conda remove -n stagebridge --all -y           # by name
conda remove -p /data/lab/envs/stagebridge --all -y  # by path

# Install packages
conda install package_name -y
pip install package_name

# Register for Jupyter
python -m ipykernel install --user --name=stagebridge
```

---

## Summary: What Setup Script Does

The `hpc_setup.sh` (or `hpc_setup_custom.sh`) script:

1. ✅ Loads miniforge3 module
2. ✅ Creates Python 3.11 environment
3. ✅ Installs PyTorch with CUDA 12.1 support
4. ✅ Installs scientific packages (numpy, pandas, sklearn, etc.)
5. ✅ Installs single-cell tools (scanpy, anndata, scvi-tools)
6. ✅ Installs spatial backends (tangram, destvi, tacco)
7. ✅ Installs analysis tools (umap, phate, pot)
8. ✅ Registers Jupyter kernel
9. ✅ Installs StageBridge package

**Total install time:** ~15-20 minutes
**Total disk space:** ~8-10GB

---

**Ready to set up! Choose Option 1 (simple) or Option 2 (custom location) based on your needs.**
