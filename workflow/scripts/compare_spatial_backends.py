#!/usr/bin/env python3
"""Compare spatial backend results and select canonical backend.

Compares backends across label sources (HLCA vs LuCA) and selects the best
combination for full cohort deconvolution.

Snakemake script - uses snakemake.input, snakemake.output, snakemake.params.
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
label_ablation_output = snakemake.output.label_ablation

# Get params
spatial_dir = Path(snakemake.params.spatial_dir)
label_sources = snakemake.params.label_sources
backends = snakemake.params.backends

print("=" * 60)
print("Comparing Spatial Backends")
print(f"  Label sources: {label_sources}")
print(f"  Backends: {backends}")
print("=" * 60)

# Load all metrics: results[label_source][backend] = metrics
results = {}
for f in metric_files:
    path = Path(f)
    # Structure: {spatial_dir}/{label_source}/{backend}/upstream_metrics.json
    backend = path.parent.name
    label_source = path.parent.parent.name

    if label_source not in results:
        results[label_source] = {}

    with open(f) as fh:
        results[label_source][backend] = json.load(fh)
    print(f"  Loaded {label_source}/{backend}: {path}")

# Metrics we care about (from our backend wrappers)
# Higher is better for all after transformation
METRICS = [
    ('coverage', 1.0, 1.0),         # Higher is better, weight 1.0
    ('mean_entropy', -1.0, 0.5),    # Lower entropy = more confident (invert), weight 0.5
    ('sparsity', 1.0, 0.5),         # Higher sparsity = cleaner, weight 0.5
]

def compute_score(metrics_dict):
    """Compute weighted composite score."""
    score = 0.0
    total_weight = 0.0
    for metric_name, direction, weight in METRICS:
        if metric_name in metrics_dict:
            val = metrics_dict[metric_name]
            if val is not None and not np.isnan(val):
                if direction < 0:
                    val = 1.0 / (1.0 + val)  # Invert for "lower is better"
                score += val * weight
                total_weight += weight
    return score / max(total_weight, 1e-6)

# Compute scores for all combinations
scores = {}  # scores[(label_source, backend)] = score
for label_source, backend_results in results.items():
    for backend, metrics in backend_results.items():
        key = (label_source, backend)
        scores[key] = compute_score(metrics)

print()
print("Composite Scores:")
for (ls, be), score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {ls}/{be}: {score:.4f}")

# Select canonical combination
best_key = max(scores.keys(), key=lambda x: scores[x])
canonical_label_source, canonical_backend = best_key
print(f"\nCanonical: {canonical_label_source}/{canonical_backend} (score: {scores[best_key]:.4f})")

# Label source ablation: compare HLCA vs LuCA per backend
label_ablation = {}
for backend in backends:
    label_ablation[backend] = {}
    for ls in label_sources:
        key = (ls, backend)
        if key in scores:
            label_ablation[backend][ls] = {
                'score': scores[key],
                'metrics': results.get(ls, {}).get(backend, {})
            }

# Aggregate label source comparison
hlca_avg = np.mean([scores[(ls, be)] for ls, be in scores if ls == 'hlca'])
luca_avg = np.mean([scores[(ls, be)] for ls, be in scores if ls == 'luca'])
label_ablation['summary'] = {
    'hlca_mean_score': float(hlca_avg),
    'luca_mean_score': float(luca_avg),
    'better_label_source': 'hlca' if hlca_avg >= luca_avg else 'luca',
    'delta': float(abs(hlca_avg - luca_avg))
}

# Save comparison
comparison = {
    'label_sources': label_sources,
    'backends': backends,
    'results': {ls: {be: m for be, m in bm.items()} for ls, bm in results.items()},
    'composite_scores': {f"{ls}/{be}": s for (ls, be), s in scores.items()},
    'canonical': {
        'label_source': canonical_label_source,
        'backend': canonical_backend,
        'score': scores[best_key]
    }
}

with open(comparison_output, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"Saved comparison: {comparison_output}")

# Save canonical selection (used by downstream rules)
canonical = {
    'backend': canonical_backend,
    'label_source': canonical_label_source,
    'score': scores[best_key],
    'reason': f"Highest composite score ({scores[best_key]:.4f})"
}
with open(canonical_output, 'w') as f:
    json.dump(canonical, f, indent=2)
print(f"Saved canonical: {canonical_output}")

# Save label ablation
with open(label_ablation_output, 'w') as f:
    json.dump(label_ablation, f, indent=2)
print(f"Saved label ablation: {label_ablation_output}")

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Bar plot of composite scores by backend (grouped by label source)
ax = axes[0]
x = np.arange(len(backends))
width = 0.35
for i, ls in enumerate(label_sources):
    vals = [scores.get((ls, be), 0) for be in backends]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=ls.upper())
    # Highlight canonical
    for j, be in enumerate(backends):
        if ls == canonical_label_source and be == canonical_backend:
            bars[j].set_edgecolor('green')
            bars[j].set_linewidth(3)

ax.set_xlabel('Backend')
ax.set_ylabel('Composite Score')
ax.set_title('Backend Comparison by Label Source')
ax.set_xticks(x)
ax.set_xticklabels(backends)
ax.legend()

# 2. Label source comparison (aggregate)
ax = axes[1]
ls_scores = [hlca_avg, luca_avg]
colors = ['#3498db', '#e74c3c']
bars = ax.bar(['HLCA', 'LuCA'], ls_scores, color=colors)
ax.set_ylabel('Mean Score')
ax.set_title('Label Source Comparison\n(averaged across backends)')
# Highlight winner
winner_idx = 0 if hlca_avg >= luca_avg else 1
bars[winner_idx].set_edgecolor('green')
bars[winner_idx].set_linewidth(3)

# 3. Individual metrics for canonical backend
ax = axes[2]
canonical_metrics = results[canonical_label_source][canonical_backend]
metric_names = [m[0] for m in METRICS if m[0] in canonical_metrics]
metric_vals = [canonical_metrics[m] for m in metric_names]
ax.barh(metric_names, metric_vals, color='#2ecc71')
ax.set_xlabel('Value')
ax.set_title(f'Canonical: {canonical_label_source.upper()}/{canonical_backend}')

plt.tight_layout()
plt.savefig(figure_output, dpi=150, bbox_inches='tight')
print(f"Saved figure: {figure_output}")

print()
print("=" * 60)
print(f"Canonical backend selected: {canonical_label_source}/{canonical_backend}")
print("=" * 60)
