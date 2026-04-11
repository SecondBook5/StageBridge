#!/bin/bash
#SBATCH --job-name=create_r_env
#SBATCH --output=/scratch/chaunzt1/stagebridge/logs/create_r_env_%j.out
#SBATCH --error=/scratch/chaunzt1/stagebridge/logs/create_r_env_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL

# Create R conda environment for spatial deconvolution backends
# (CARD, RCTD, SPOTlight)

set -e

SCRATCH_DIR="/scratch/chaunzt1/stagebridge"
ENV_PATH="/scratch/chaunzt1/envs/r_deconv"

mkdir -p ${SCRATCH_DIR}/logs

echo "========================================"
echo "Creating R Deconvolution Environment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Time: $(date)"
echo "Environment path: $ENV_PATH"
echo ""

# Load miniforge
module load miniforge3
eval "$(conda shell.bash hook)"

# Remove existing env if it exists (comment out to skip)
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment at $ENV_PATH"
    rm -rf "$ENV_PATH"
fi

# Create the environment
# Packages organized by purpose:
#   - Core: r-base, r-matrix, r-remotes, r-devtools, r-biocmanager
#   - Compilation: r-rcpp, r-rcpparmadillo, r-rcppeigen, r-rcppparallel
#   - CARD deps: r-fields, r-rann, r-pbmcapply, r-nnls, r-concaveman, r-sp, r-gtools, r-nmf
#   - RCTD deps: r-doparallel, r-foreach, r-quadprog
#   - Plotting: r-ggplot2, r-dplyr, r-tidyr, r-pheatmap, r-viridis, r-patchwork, etc.
#   - Seurat: r-seurat, r-seuratobject
#   - Bioconductor: SingleCellExperiment, SpatialExperiment, scater, scran, etc.
echo ""
echo "=== Creating conda environment ==="
conda create -p "$ENV_PATH" -y \
    -c conda-forge -c bioconda -c defaults \
    r-base=4.3 \
    r-matrix \
    r-remotes \
    r-devtools \
    r-biocmanager \
    r-rcpp \
    r-rcpparmadillo \
    r-rcppeigen \
    r-rcppparallel \
    r-fields \
    r-rann \
    r-pbmcapply \
    r-nnls \
    r-concaveman \
    r-sp \
    r-gtools \
    r-ggcorrplot \
    r-nmf \
    r-rcppml \
    r-doparallel \
    r-foreach \
    r-quadprog \
    r-ggplot2 \
    r-dplyr \
    r-tidyr \
    r-stringr \
    r-data.table \
    r-reshape2 \
    r-scales \
    r-viridis \
    r-patchwork \
    r-pheatmap \
    r-rcolorbrewer \
    r-gridextra \
    r-cowplot \
    r-seurat \
    r-seuratobject \
    bioconductor-singlecellexperiment \
    bioconductor-spatialexperiment \
    bioconductor-scater \
    bioconductor-scran \
    bioconductor-scuttle \
    bioconductor-summarizedexperiment \
    bioconductor-s4vectors \
    bioconductor-biocgenerics \
    bioconductor-biocparallel \
    bioconductor-delayedarray \
    bioconductor-hdf5array \
    bioconductor-rhdf5 \
    bioconductor-dropletutils \
    bioconductor-edger \
    bioconductor-limma \
    bioconductor-deseq2 \
    bioconductor-complexheatmap

echo ""
echo "Conda environment created successfully"

# Activate the environment
conda activate "$ENV_PATH"

echo "R version: $(R --version | head -1)"

# Install GitHub packages
echo ""
echo "=== Installing CARD, RCTD, SPOTlight ==="

Rscript -e '
cat("Installing CARD...\n")
remotes::install_github("YingMa0107/CARD", upgrade = "never", quiet = FALSE)

cat("\nInstalling spacexr (RCTD)...\n")
remotes::install_github("dmcable/spacexr", upgrade = "never", quiet = FALSE)

cat("\nInstalling SPOTlight...\n")
BiocManager::install("SPOTlight", update = FALSE, ask = FALSE)

cat("\n=== Verification ===\n")
for (pkg in c("CARD", "spacexr", "SPOTlight")) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("[OK] %s (%s)\n", pkg, packageVersion(pkg)))
    } else {
        cat(sprintf("[FAIL] %s\n", pkg))
    }
}
'

echo ""
echo "========================================"
echo "Done!"
echo "Finished: $(date)"
echo "========================================"
echo ""
echo "To use this environment:"
echo "  module load miniforge3"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate $ENV_PATH"
