#!/usr/bin/env python3
"""Compare spatial backend results and select canonical backend.

Snakemake script - uses snakemake.input and snakemake.output.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Snakemake provides these
metric_files = snakemake.input
comparison_output = snakemake.output.comparison
canonical_output = snakemake.output.canonical
figure_output = snakemake.output.figure

print("=" * 60)
print("Comparing Spatial Backends")
print("=" * 60)

# Load all metrics
results = {}
for f in metric_files:
    path = Path(f)
    backend = path.parent.name
    with open(f) as fh:
        results[backend] = json.load(fh)
    print(f"  Loaded {backend}: {path}")

# Compute composite scores
# Metrics we care about (higher is better for all after transformation)
METRICS = [
    ('cell_type_accuracy', 1.0),      # Higher is better
    ('spatial_coherence', 1.0),       # Higher is better
    ('marker_preservation', 1.0),     # Higher is better
    ('computation_time', -1.0),       # Lower is better (invert)
]

scores = {}
for backend, metrics in results.items():
    score = 0.0
    n_metrics = 0
    for metric_name, direction in METRICS:
        if metric_name in metrics:
            val = metrics[metric_name]
            if direction < 0:
                val = 1.0 / (1.0 + val)  # Invert for "lower is better"
            score += val
            n_metrics += 1

    scores[backend] = score / max(n_metrics, 1)

print()
print("Composite Scores:")
for backend, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {backend}: {score:.4f}")

# Select canonical backend
canonical_backend = max(scores.keys(), key=lambda x: scores[x])
print(f"\nCanonical backend: {canonical_backend}")

# Save comparison
comparison = {
    'backends': list(results.keys()),
    'metrics': results,
    'composite_scores': scores,
    'canonical_backend': canonical_backend,
}

with open(comparison_output, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"Saved comparison: {comparison_output}")

# Save canonical selection
canonical = {
    'backend': canonical_backend,
    'score': scores[canonical_backend],
    'reason': f"Highest composite score ({scores[canonical_backend]:.4f})"
}
with open(canonical_output, 'w') as f:
    json.dump(canonical, f, indent=2)
print(f"Saved canonical: {canonical_output}")

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar plot of composite scores
ax = axes[0]
backends = list(scores.keys())
vals = [scores[b] for b in backends]
colors = ['#2ecc71' if b == canonical_backend else '#3498db' for b in backends]
ax.barh(backends, vals, color=colors)
ax.set_xlabel('Composite Score')
ax.set_title('Spatial Backend Comparison')
ax.axvline(max(vals), color='green', linestyle='--', alpha=0.5)

# Radar plot of individual metrics (if we have enough)
ax = axes[1]
if len(backends) >= 2:
    # Simple grouped bar for key metrics
    metric_names = [m[0] for m in METRICS if m[0] in results[backends[0]]]
    x = np.arange(len(metric_names))
    width = 0.8 / len(backends)

    for i, backend in enumerate(backends):
        vals = [results[backend].get(m, 0) for m in metric_names]
        offset = (i - len(backends)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=backend)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Individual Metrics')
    ax.legend()

plt.tight_layout()
plt.savefig(figure_output, dpi=150, bbox_inches='tight')
print(f"Saved figure: {figure_output}")

print()
print("=" * 60)
print("Comparison Complete")
print("=" * 60)
