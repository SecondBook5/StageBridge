#!/bin/bash
# Full pySCENIC regulon analysis - run separately due to environment conflicts
#
# Setup (one-time):
#   conda create -n pyscenic python=3.10 -y
#   conda activate pyscenic
#   pip install pyscenic scanpy anndata pandas numpy pyarrow
#
# Run: bash scripts/run_scenic.sh
#
# This runs the FULL pipeline:
#   1. GRNBoost2 - infer TF-target relationships from expression
#   2. cistarget - prune with motif enrichment (requires databases)
#   3. AUCell - score regulon activity per cell

set -e

DATA=/data1/chaunzt1/stagebridge/processed/luad_evo
CANONICAL=$DATA/canonical
SNRNA=$DATA/snrna_with_celltypes.h5ad

# cistarget database paths
DB_DIR=$DATA/scenic_dbs
MOTIF_DB=$DB_DIR/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather
ANNOTATIONS=$DB_DIR/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl
TF_LIST=$DB_DIR/allTFs_hg38.txt

echo "=============================================="
echo "pySCENIC Full Regulon Analysis"
echo "=============================================="

# Check/download databases
if [ ! -f "$MOTIF_DB" ]; then
    echo "Downloading motif database (~1.1GB)..."
    mkdir -p $DB_DIR
    wget -q --show-progress -O $MOTIF_DB \
        "https://resources.aertslab.org/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather"
fi

if [ ! -f "$ANNOTATIONS" ]; then
    echo "Downloading motif annotations..."
    wget -q --show-progress -O $ANNOTATIONS \
        "https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
fi

if [ ! -f "$TF_LIST" ]; then
    echo "Downloading TF list..."
    wget -q --show-progress -O $TF_LIST \
        "https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt"
fi

echo "Databases ready in: $DB_DIR"

if [ -f "$CANONICAL/scenic/aucell_scores.parquet" ]; then
    echo "SCENIC results already exist, skipping"
    exit 0
fi

mkdir -p $CANONICAL/scenic

python -c "
from pathlib import Path
from stagebridge.biology.regulons import run_scenic_pipeline

print('Running full pySCENIC pipeline...')
print('  This takes 2-4 hours on 800k cells')

results = run_scenic_pipeline(
    Path('$SNRNA'),
    Path('$CANONICAL/scenic'),
    motif_db_path=Path('$MOTIF_DB'),
    annotation_path=Path('$ANNOTATIONS'),
    skip_grn=False,  # Run full GRNBoost2 + cistarget
    n_jobs=8,
)
print('pySCENIC complete:', results)
"

echo ""
echo "pySCENIC complete! Results in: $CANONICAL/scenic/"
echo "  - adjacencies.parquet: TF-target network"
echo "  - aucell_scores.parquet: Per-cell regulon activity"
echo "  - regulon_scores.parquet: Regulon summary stats"
