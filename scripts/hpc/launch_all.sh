#!/bin/bash
# Launch all pending StageBridge jobs
# Run on HPC after: git pull

set -e
cd ~/StageBridge

echo "============================================"
echo "Launching StageBridge Pipeline Jobs"
echo "============================================"
echo ""

# 1. Clonal extraction (inferCNV) - ~6 hours
echo "[1/4] Submitting clonal extraction (inferCNV)..."
CLONAL_JOB=$(sbatch --parsable scripts/hpc/run_clonal_extraction.sbatch)
echo "  Job ID: $CLONAL_JOB"

# 2. Set baselines (if not already running)
echo "[2/4] Submitting set baselines..."
BASELINE_JOB=$(sbatch --parsable scripts/hpc/run_set_baselines.sbatch)
echo "  Job ID: $BASELINE_JOB"

# 3. Full Snakemake pipeline (will skip completed steps)
echo "[3/4] Launching Snakemake pipeline..."
snakemake --profile workflow/slurm --jobs 50 &
SNAKE_PID=$!
echo "  Snakemake PID: $SNAKE_PID"

echo ""
echo "============================================"
echo "All jobs submitted!"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/*.out"
echo ""
echo "Snakemake running in background (PID: $SNAKE_PID)"
echo "Check with: snakemake --profile workflow/slurm --summary"
