#!/bin/bash
#SBATCH --job-name=spatial_compare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=chaunzt1
#SBATCH --output=/home/booka/StageBridge/logs/spatial_compare_%j.log
#SBATCH --error=/home/booka/StageBridge/logs/spatial_compare_%j.err

# =============================================================================
# Spatial Backend Comparison
# =============================================================================
#
# Compares results from all 4 spatial backends and selects canonical backend.
#
# Metrics compared:
#   - Upstream: entropy, coverage, sparsity, reconstruction error
#   - Runtime: wall-clock time, memory usage
#   - Confidence: mean/std of per-spot confidence
#
# Outputs:
#   - backend_comparison.json: Full metrics for all backends
#   - backend_selection_report.md: Human-readable comparison report
#   - canonical_backend.json: Selected backend with rationale
#   - backend_comparison.png: Visualization
#
# This job should run AFTER all 4 backend jobs complete.
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "Spatial Backend Comparison"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=============================================="

# Environment setup
cd /home/booka/StageBridge || cd $HOME/StageBridge
source /admin/software/miniforge3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate /scratch/chaunzt1/stagebridge_env

# Data paths
DATA="${DATA:-/scratch/chaunzt1/stagebridge}"
OUTPUT_DIR="$DATA/runs/spatial_benchmark"

echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# Check which backends completed
# =============================================================================
echo "Checking backend results..."

BACKENDS=("tangram" "destvi" "tacco" "cell2location")
SUCCESSFUL=()
FAILED=()

for backend in "${BACKENDS[@]}"; do
    METRICS_FILE="$OUTPUT_DIR/$backend/upstream_metrics.json"
    FAILURE_FILE="$OUTPUT_DIR/$backend/failure_log.json"

    if [ -f "$METRICS_FILE" ]; then
        echo "  $backend: SUCCESS"
        SUCCESSFUL+=("$backend")
    elif [ -f "$FAILURE_FILE" ]; then
        echo "  $backend: FAILED (see failure_log.json)"
        FAILED+=("$backend")
    else
        echo "  $backend: NOT RUN"
        FAILED+=("$backend")
    fi
done

echo ""
echo "Successful backends: ${SUCCESSFUL[*]:-none}"
echo "Failed backends: ${FAILED[*]:-none}"
echo ""

# Need at least 2 backends for meaningful comparison
if [ ${#SUCCESSFUL[@]} -lt 2 ]; then
    echo "ERROR: Need at least 2 successful backends for comparison"
    echo "  Only ${#SUCCESSFUL[@]} succeeded: ${SUCCESSFUL[*]:-none}"
    exit 1
fi

# =============================================================================
# Run comparison
# =============================================================================
echo "Running backend comparison..."
echo ""

python3 << 'PYTHON_COMPARE'
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
DATA = os.environ.get('DATA', '/scratch/chaunzt1/stagebridge')
OUTPUT_DIR = Path(f"{DATA}/runs/spatial_benchmark")

BACKENDS = ['tangram', 'destvi', 'tacco', 'cell2location']

print("=" * 70)
print("SPATIAL BACKEND COMPARISON")
print("=" * 70)
print("")

# =============================================================================
# Load metrics from each backend
# =============================================================================
results = {}
for backend in BACKENDS:
    backend_dir = OUTPUT_DIR / backend
    metrics_file = backend_dir / 'upstream_metrics.json'
    failure_file = backend_dir / 'failure_log.json'

    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Also load metadata if available
        metadata_file = backend_dir / 'backend_metadata.json'
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        results[backend] = {
            'status': 'success',
            'metrics': metrics,
            'metadata': metadata,
        }
        print(f"  {backend}: loaded metrics")

    elif failure_file.exists():
        with open(failure_file) as f:
            failure = json.load(f)
        results[backend] = {
            'status': 'failed',
            'error': failure.get('error', 'Unknown error'),
        }
        print(f"  {backend}: FAILED - {failure.get('error', 'Unknown')[:50]}...")

    else:
        results[backend] = {
            'status': 'not_run',
        }
        print(f"  {backend}: not run")

print("")

# =============================================================================
# Build comparison table
# =============================================================================
successful_backends = [b for b in BACKENDS if results[b]['status'] == 'success']

if len(successful_backends) < 2:
    print("ERROR: Need at least 2 successful backends for comparison")
    sys.exit(1)

# Extract metrics into DataFrame
rows = []
for backend in successful_backends:
    m = results[backend]['metrics']
    rows.append({
        'backend': backend,
        'mean_entropy': m.get('mean_entropy', np.nan),
        'coverage': m.get('coverage', np.nan),
        'sparsity': m.get('sparsity', np.nan),
        'runtime_minutes': m.get('runtime_minutes', m.get('runtime_seconds', 0) / 60),
    })

df = pd.DataFrame(rows)
df = df.set_index('backend')

print("Raw Metrics:")
print("-" * 70)
print(df.to_string())
print("")

# =============================================================================
# Compute scores (higher is better)
# =============================================================================
scores = pd.DataFrame(index=df.index)

# Entropy: moderate is good (0.5-0.7 ideal for mixed spots)
# Score = 1 - |entropy - 0.6|
scores['entropy_score'] = 1 - np.abs(df['mean_entropy'] - 0.6).clip(0, 1)

# Coverage: higher is better (fraction with confident mapping)
scores['coverage_score'] = df['coverage'].fillna(0).clip(0, 1)

# Sparsity: lower is better (more complete annotations)
scores['sparsity_score'] = (1 - df['sparsity'].fillna(1)).clip(0, 1)

# Runtime: faster is better (inverse, normalized to slowest)
runtime_normalized = df['runtime_minutes'] / df['runtime_minutes'].max()
scores['runtime_score'] = (1 - runtime_normalized).clip(0, 1)

print("Normalized Scores (higher is better):")
print("-" * 70)
print(scores.to_string())
print("")

# =============================================================================
# Compute composite score
# =============================================================================
weights = {
    'entropy_score': 0.25,     # Cell type diversity
    'coverage_score': 0.30,    # Confident mappings
    'sparsity_score': 0.25,    # Annotation completeness
    'runtime_score': 0.20,     # Efficiency
}

scores['composite'] = sum(scores[col] * w for col, w in weights.items())
scores = scores.sort_values('composite', ascending=False)

print("Composite Scores:")
print("-" * 70)
for backend in scores.index:
    print(f"  {backend:15s}: {scores.loc[backend, 'composite']:.4f}")
print("")

# =============================================================================
# Select canonical backend
# =============================================================================
best_backend = scores.index[0]
best_score = scores.loc[best_backend, 'composite']

# Get runner-up
if len(scores) > 1:
    runnerup_backend = scores.index[1]
    runnerup_score = scores.loc[runnerup_backend, 'composite']
else:
    runnerup_backend = None
    runnerup_score = None

print("=" * 70)
print(f"CANONICAL BACKEND: {best_backend.upper()}")
print(f"Composite Score: {best_score:.4f}")
print("=" * 70)
print("")

# =============================================================================
# Generate rationale
# =============================================================================
rationale_lines = [
    f"Selected {best_backend.upper()} as the canonical spatial backend based on",
    f"composite score ({best_score:.3f}) across multiple metrics.",
    "",
    "Scoring weights:",
]
for col, w in weights.items():
    metric_name = col.replace('_score', '')
    rationale_lines.append(f"  - {metric_name}: {w:.0%}")

rationale_lines.extend([
    "",
    f"Key factors for {best_backend.upper()}:",
])

# Highlight strengths
m = results[best_backend]['metrics']
if scores.loc[best_backend, 'entropy_score'] >= 0.7:
    rationale_lines.append(f"  - Balanced cell type diversity (entropy={m['mean_entropy']:.3f})")
if scores.loc[best_backend, 'coverage_score'] >= 0.7:
    rationale_lines.append(f"  - High mapping coverage ({m['coverage']:.1%})")
if scores.loc[best_backend, 'sparsity_score'] >= 0.7:
    rationale_lines.append(f"  - Low sparsity (sparsity={m['sparsity']:.3f})")
if scores.loc[best_backend, 'runtime_score'] >= 0.6:
    runtime_min = m.get('runtime_minutes', m.get('runtime_seconds', 0) / 60)
    rationale_lines.append(f"  - Efficient runtime ({runtime_min:.1f} minutes)")

if runnerup_backend:
    rationale_lines.extend([
        "",
        f"Alternative backend: {runnerup_backend.upper()} (score={runnerup_score:.3f})",
        "  Preserved for robustness comparisons.",
    ])

rationale = "\n".join(rationale_lines)
print(rationale)
print("")

# =============================================================================
# Save outputs
# =============================================================================
print("Saving outputs...")

# 1. Full comparison JSON
comparison = {
    'timestamp': datetime.now().isoformat(),
    'backends': results,
    'metrics_table': df.to_dict(),
    'scores_table': scores.to_dict(),
    'weights': weights,
    'canonical_backend': best_backend,
    'canonical_score': float(best_score),
    'alternative_backend': runnerup_backend,
    'alternative_score': float(runnerup_score) if runnerup_score else None,
}

with open(OUTPUT_DIR / 'backend_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"  Saved: backend_comparison.json")

# 2. Canonical backend JSON (simple format for downstream)
canonical = {
    'backend': best_backend,
    'score': float(best_score),
    'alternative': runnerup_backend,
    'path': str(OUTPUT_DIR / best_backend),
}

with open(OUTPUT_DIR / 'canonical_backend.json', 'w') as f:
    json.dump(canonical, f, indent=2)
print(f"  Saved: canonical_backend.json")

# 3. Markdown report
report_lines = [
    "# Spatial Backend Selection Report",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Summary",
    "",
    f"**Canonical Backend: {best_backend.upper()}**",
    "",
    rationale,
    "",
    "## Metrics Comparison",
    "",
    "| Backend | Entropy | Coverage | Sparsity | Runtime (min) | Composite |",
    "|---------|---------|----------|----------|---------------|-----------|",
]

for backend in scores.index:
    m = results[backend]['metrics']
    runtime = m.get('runtime_minutes', m.get('runtime_seconds', 0) / 60)
    report_lines.append(
        f"| {backend} | {m.get('mean_entropy', 0):.3f} | "
        f"{m.get('coverage', 0):.3f} | {m.get('sparsity', 0):.3f} | "
        f"{runtime:.1f} | **{scores.loc[backend, 'composite']:.3f}** |"
    )

report_lines.extend([
    "",
    "## Failed Backends",
    "",
])

failed = [b for b in BACKENDS if results[b]['status'] != 'success']
if failed:
    for backend in failed:
        status = results[backend]['status']
        if status == 'failed':
            error = results[backend].get('error', 'Unknown error')
            report_lines.append(f"- **{backend}**: {error[:100]}")
        else:
            report_lines.append(f"- **{backend}**: Not run")
else:
    report_lines.append("All backends completed successfully.")

report_lines.extend([
    "",
    "## Scoring Methodology",
    "",
    "Each backend is scored on 4 metrics:",
    "",
    "1. **Entropy Score**: Cell type diversity. Moderate entropy (~0.6) is ideal",
    "   for spatial spots that contain cell mixtures.",
    "",
    "2. **Coverage Score**: Fraction of spots with confident cell type mapping.",
    "   Higher coverage indicates more complete annotations.",
    "",
    "3. **Sparsity Score**: Inverse of zero-proportion fraction.",
    "   Lower sparsity means more cell types are represented per spot.",
    "",
    "4. **Runtime Score**: Computational efficiency.",
    "   Normalized inverse of runtime.",
    "",
    "Final composite score is a weighted average:",
    "",
])

for col, w in weights.items():
    metric_name = col.replace('_score', '').title()
    report_lines.append(f"- {metric_name}: {w:.0%}")

report_lines.extend([
    "",
    "## Output Files",
    "",
    f"- `{best_backend}/cell_type_proportions.parquet`: Spot-level cell type proportions",
    f"- `{best_backend}/mapping_confidence.parquet`: Per-spot confidence scores",
    f"- `{best_backend}/upstream_metrics.json`: Quality metrics",
    f"- `canonical_backend.json`: Selected backend metadata",
    f"- `backend_comparison.json`: Full comparison data",
    "",
])

report = "\n".join(report_lines)

with open(OUTPUT_DIR / 'backend_selection_report.md', 'w') as f:
    f.write(report)
print(f"  Saved: backend_selection_report.md")

# =============================================================================
# Generate comparison plot
# =============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Composite scores bar chart
    ax = axes[0, 0]
    colors = ['#2ecc71' if b == best_backend else '#3498db' for b in scores.index]
    ax.barh(scores.index, scores['composite'], color=colors)
    ax.set_xlabel('Composite Score')
    ax.set_title('Overall Performance')
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. Individual metric scores
    ax = axes[0, 1]
    metric_cols = ['entropy_score', 'coverage_score', 'sparsity_score', 'runtime_score']
    x = np.arange(len(scores.index))
    width = 0.2
    for i, col in enumerate(metric_cols):
        ax.bar(x + i * width, scores[col], width, label=col.replace('_score', ''))
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(scores.index)
    ax.set_ylabel('Score')
    ax.set_title('Individual Metrics')
    ax.legend()
    ax.set_ylim(0, 1)

    # 3. Runtime comparison
    ax = axes[1, 0]
    runtime_vals = [results[b]['metrics'].get('runtime_minutes',
                    results[b]['metrics'].get('runtime_seconds', 0) / 60)
                   for b in scores.index]
    colors = ['#e74c3c' if b == best_backend else '#95a5a6' for b in scores.index]
    ax.barh(scores.index, runtime_vals, color=colors)
    ax.set_xlabel('Runtime (minutes)')
    ax.set_title('Computational Cost')

    # 4. Entropy vs Coverage scatter
    ax = axes[1, 1]
    for backend in scores.index:
        m = results[backend]['metrics']
        color = '#2ecc71' if backend == best_backend else '#3498db'
        size = 200 if backend == best_backend else 100
        ax.scatter(m.get('mean_entropy', 0), m.get('coverage', 0),
                   s=size, c=[color], label=backend, edgecolors='black')
    ax.set_xlabel('Mean Entropy (Diversity)')
    ax.set_ylabel('Coverage (Confidence)')
    ax.set_title('Quality Trade-offs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0.6, color='gray', linestyle='--', alpha=0.5, label='Ideal entropy')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'backend_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: backend_comparison.png")

except Exception as e:
    print(f"  WARNING: Could not generate plot: {e}")

print("")
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
print(f"Canonical backend: {best_backend.upper()}")
print(f"Use: {OUTPUT_DIR / best_backend}/")
print("")

sys.exit(0)
PYTHON_COMPARE

COMPARE_EXIT=$?

echo ""
echo "=============================================="
echo "Spatial Backend Comparison Complete: $(date)"
echo "Exit code: $COMPARE_EXIT"
echo "=============================================="

if [ $COMPARE_EXIT -eq 0 ]; then
    echo ""
    echo "Summary files:"
    ls -lh "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.md "$OUTPUT_DIR"/*.png 2>/dev/null || true
    echo ""
    echo "To view report:"
    echo "  cat $OUTPUT_DIR/backend_selection_report.md"
    echo ""
    echo "Canonical backend for downstream:"
    cat "$OUTPUT_DIR/canonical_backend.json" 2>/dev/null || echo "  (not found)"
fi

exit $COMPARE_EXIT
