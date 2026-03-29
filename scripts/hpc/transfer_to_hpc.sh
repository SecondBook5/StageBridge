#!/bin/bash
################################################################################
# Transfer StageBridge to HPC
################################################################################

set -e

# CONFIGURE THESE - UPDATE WITH YOUR INFO
HPC_USER="YOUR_MSK_USERNAME"
HPC_HOST="isxfer01.mskcc.org"  # Transfer server for Iris
HPC_PATH="~/StageBridge"  # Or use /data/your_labname/StageBridge for more space

echo "=========================================="
echo "Transferring StageBridge to HPC"
echo "=========================================="
echo ""
echo "Target: $HPC_USER@$HPC_HOST:$HPC_PATH"
echo ""

# Check if SSH works
echo "Testing SSH connection..."
ssh -q $HPC_USER@$HPC_HOST exit
if [ $? -eq 0 ]; then
    echo " SSH connection successful"
else
    echo " SSH connection failed"
    echo "Please check your credentials and HPC host"
    exit 1
fi

# Transfer repository
echo ""
echo "[1/3] Transferring code repository..."
rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='data/raw/' \
    --exclude='data/processed/' \
    --exclude='data/references/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.egg-info/' \
    ./ \
    $HPC_USER@$HPC_HOST:$HPC_PATH/

# Transfer raw data if it exists
if [ -d "data/raw" ] && [ "$(ls -A data/raw)" ]; then
    echo ""
    echo "[2/3] Transferring raw data..."
    rsync -avz --progress \
        data/raw/ \
        $HPC_USER@$HPC_HOST:$HPC_PATH/data/raw/
else
    echo ""
    echo "[2/3] No raw data to transfer (data/raw/ is empty)"
    echo "     You'll need to download GEO datasets on HPC"
fi

# Create necessary directories on HPC
echo ""
echo "[3/3] Creating directory structure on HPC..."
ssh $HPC_USER@$HPC_HOST "
cd $HPC_PATH
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed/luad
mkdir -p data/references
mkdir -p outputs/luad_v1_comprehensive
chmod +x hpc_setup.sh
chmod +x run_hpc_test.slurm
chmod +x run_hpc_full.slurm
"

echo ""
echo "=========================================="
echo " Transfer Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to HPC: ssh $HPC_USER@$HPC_HOST"
echo "  2. cd $HPC_PATH"
echo "  3. Review HPC_README.md for full instructions"
echo "  4. Update SLURM scripts with your email/partition"
echo "  5. Run setup: ./hpc_setup.sh"
echo "  6. Submit test job: sbatch run_hpc_test.slurm"
echo ""
