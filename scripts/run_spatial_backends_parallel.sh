#!/bin/bash
# Submit all 4 spatial backends in parallel, then compare results

DATA="/scratch/chaunzt1/stagebridge"
SNRNA="${DATA}/processed/luad_evo/snrna_qc_normalized.h5ad"
SPATIAL="${DATA}/processed/luad_evo/spatial_merged.h5ad"
OUTPUT_BASE="${DATA}/runs/spatial_benchmark"

# Submit each backend as separate job
JOB_TANGRAM=$(sbatch --parsable scripts/run_backend_tangram.sbatch)
JOB_DESTVI=$(sbatch --parsable scripts/run_backend_destvi.sbatch)
JOB_TACCO=$(sbatch --parsable scripts/run_backend_tacco.sbatch)
JOB_CELL2LOC=$(sbatch --parsable scripts/run_backend_cell2location.sbatch)

echo "Submitted jobs:"
echo "  Tangram:      $JOB_TANGRAM"
echo "  DestVI:       $JOB_DESTVI"
echo "  TACCO:        $JOB_TACCO"
echo "  Cell2location: $JOB_CELL2LOC"

# Submit comparison job that waits for all backends
sbatch --dependency=afterok:${JOB_TANGRAM}:${JOB_DESTVI}:${JOB_TACCO}:${JOB_CELL2LOC} \
    scripts/run_backend_compare.sbatch

echo ""
echo "Comparison job will run after all backends complete."
echo "Monitor with: squeue -u booka"
