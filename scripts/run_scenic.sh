#!/bin/bash
# SCENIC regulon analysis - run separately due to environment conflicts
#
# Recommended: Create a dedicated environment for pySCENIC:
#   conda create -n scenic_env python=3.10
#   conda activate scenic_env
#   pip install pyscenic scanpy anndata pandas numpy
#
# Then run: bash scripts/run_scenic.sh

set -e

DATA=/data1/chaunzt1/stagebridge/processed/luad_evo
CANONICAL=$DATA/canonical
SNRNA=$DATA/snrna_with_celltypes.h5ad

echo "=============================================="
echo "SCENIC Regulon Analysis"
echo "=============================================="

if [ -f "$CANONICAL/scenic/regulon_auc.parquet" ]; then
    echo "SCENIC results already exist, skipping"
    exit 0
fi

mkdir -p $CANONICAL/scenic

python -c "
from pathlib import Path
from stagebridge.biology.regulons import run_scenic_pipeline

print('Running SCENIC pipeline...')
results = run_scenic_pipeline(
    Path('$SNRNA'),
    Path('$CANONICAL/scenic'),
    skip_grn=True,  # Use predefined lung cancer regulons
    n_jobs=8,
)
print('SCENIC complete:', results)
"

echo ""
echo "SCENIC complete! Results in: $CANONICAL/scenic/"
