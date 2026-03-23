#!/bin/bash
# Spatial Backend Benchmark Pipeline
#
# Three-phase approach:
# 1. Smoke test: 1 Normal sample, all 4 backends (verify fixes work)
# 2. Stratified benchmark: 9 samples, all 4 backends (select best method)
# 3. Production run: 56 samples, best backend (full dataset)
#
# Usage:
#   ./spatial_benchmark_pipeline.sh smoke    # Run smoke test only
#   ./spatial_benchmark_pipeline.sh bench    # Run stratified benchmark
#   ./spatial_benchmark_pipeline.sh prod     # Run production (after selecting backend)
#   ./spatial_benchmark_pipeline.sh all      # Run smoke + bench sequentially

set -e

# Configuration
DATA_DIR="/home/booka/data/stagebridge/processed/luad_evo"
SNRNA="${DATA_DIR}/snrna_merged.h5ad"
SPATIAL="${DATA_DIR}/spatial_merged.h5ad"
OUTPUT_BASE="${DATA_DIR}/spatial_benchmark"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "${GREEN}==>${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

echo_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Phase 1: Smoke Test
run_smoke_test() {
    echo_step "Phase 1: Smoke Test (Normal sample, all 4 backends)"
    echo "Sample: GSM9226174_P4_Normal"
    echo "Backends: tangram, destvi, tacco, cell2location"
    echo ""

    python -m stagebridge.pipelines.run_spatial_benchmark \
        --snrna "$SNRNA" \
        --spatial "$SPATIAL" \
        --output_dir "${OUTPUT_BASE}/smoke_test_normal" \
        --sample GSM9226174_P4_Normal \
        --sample-col sample_id \
        --backends tangram destvi tacco cell2location

    if [ $? -eq 0 ]; then
        echo_step "Smoke test PASSED"
        echo "Results: ${OUTPUT_BASE}/smoke_test_normal/backend_comparison.json"
    else
        echo_error "Smoke test FAILED"
        exit 1
    fi
}

# Phase 2: Stratified Benchmark
run_stratified_benchmark() {
    echo_step "Phase 2: Stratified Benchmark (9 samples, all 4 backends)"
    echo "Samples:"
    echo "  - Normal: 1 (GSM9226174_P4_Normal)"
    echo "  - AAH: 2 (P1, P2)"
    echo "  - AIS: 2 (P3, P5)"
    echo "  - MIA: 2 (P10, P13)"
    echo "  - LUAD: 2 (P1, P3)"
    echo "Total: 9 samples x 4 backends = 36 runs"
    echo ""

    python workflow/scripts/run_stratified_benchmark.py \
        --output-dir "${OUTPUT_BASE}/stratified_benchmark" \
        --snrna "$SNRNA" \
        --spatial "$SPATIAL" \
        --backends tangram destvi tacco cell2location

    if [ $? -eq 0 ]; then
        echo_step "Stratified benchmark COMPLETE"
        echo "Results: ${OUTPUT_BASE}/stratified_benchmark/stratified_benchmark_results.json"

        # Show recommendation
        if [ -f "${OUTPUT_BASE}/stratified_benchmark/stratified_benchmark_results.json" ]; then
            echo ""
            echo_step "Backend Recommendation:"
            python -c "
import json
with open('${OUTPUT_BASE}/stratified_benchmark/stratified_benchmark_results.json') as f:
    r = json.load(f)
print(f\"  Recommended: {r.get('recommended_backend', 'N/A').upper()}\")
for i, b in enumerate(r.get('ranking', []), 1):
    s = r['aggregate_scores'][b]
    print(f\"  {i}. {b}: mean={s['mean']:.3f}, failures={s['n_failures']}\")
"
        fi
    else
        echo_error "Stratified benchmark FAILED"
        exit 1
    fi
}

# Phase 3: Production Run
run_production() {
    BACKEND=${1:-tangram}  # Default to tangram if not specified

    echo_step "Phase 3: Production Run (56 samples, ${BACKEND})"
    echo "Backend: $BACKEND"
    echo "Samples: All 56"
    echo ""

    # Get all sample IDs
    python -c "
import pandas as pd
manifest = pd.read_csv('${DATA_DIR}/spatial_manifest.csv')
for sample_id in manifest['sample_id']:
    print(sample_id)
" | while read sample_id; do
        echo_step "Processing: $sample_id"
        python -m stagebridge.pipelines.run_spatial_benchmark \
            --snrna "$SNRNA" \
            --spatial "$SPATIAL" \
            --output_dir "${OUTPUT_BASE}/production/${sample_id}" \
            --sample "$sample_id" \
            --sample-col sample_id \
            --backends "$BACKEND" || echo_warn "Failed: $sample_id"
    done

    echo_step "Production run COMPLETE"
    echo "Results: ${OUTPUT_BASE}/production/"
}

# Quick versions (reduced epochs)
run_smoke_test_quick() {
    echo_step "Phase 1: QUICK Smoke Test"
    python -m stagebridge.pipelines.run_spatial_benchmark \
        --snrna "$SNRNA" \
        --spatial "$SPATIAL" \
        --output_dir "${OUTPUT_BASE}/smoke_test_normal_quick" \
        --sample GSM9226174_P4_Normal \
        --sample-col sample_id \
        --backends tangram destvi tacco cell2location \
        --quick
}

run_stratified_benchmark_quick() {
    echo_step "Phase 2: QUICK Stratified Benchmark"
    python workflow/scripts/run_stratified_benchmark.py \
        --output-dir "${OUTPUT_BASE}/stratified_benchmark_quick" \
        --snrna "$SNRNA" \
        --spatial "$SPATIAL" \
        --backends tangram destvi tacco cell2location \
        --quick
}

# Main
case "${1:-help}" in
    smoke)
        run_smoke_test
        ;;
    smoke-quick)
        run_smoke_test_quick
        ;;
    bench)
        run_stratified_benchmark
        ;;
    bench-quick)
        run_stratified_benchmark_quick
        ;;
    prod)
        run_production "${2:-tangram}"
        ;;
    all)
        run_smoke_test
        echo ""
        run_stratified_benchmark
        ;;
    all-quick)
        run_smoke_test_quick
        echo ""
        run_stratified_benchmark_quick
        ;;
    *)
        echo "Spatial Backend Benchmark Pipeline"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  smoke         Run smoke test (1 sample, 4 backends)"
        echo "  smoke-quick   Run quick smoke test (reduced epochs)"
        echo "  bench         Run stratified benchmark (9 samples, 4 backends)"
        echo "  bench-quick   Run quick stratified benchmark"
        echo "  prod [backend] Run production (56 samples, 1 backend)"
        echo "  all           Run smoke + bench sequentially"
        echo "  all-quick     Run quick smoke + bench"
        echo ""
        echo "Recommended workflow:"
        echo "  1. $0 smoke-quick   # Verify fixes work (~30 min)"
        echo "  2. $0 bench         # Select best backend (~4-8 hours)"
        echo "  3. $0 prod tangram  # Full dataset with winner"
        ;;
esac
