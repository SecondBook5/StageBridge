#!/usr/bin/env python3
"""Compare spatial backend results and select canonical backend.

Compares backends across label sources (HLCA vs LuCA) and selects the best
combination using comprehensive metrics.

Snakemake script - uses snakemake.input, snakemake.output, snakemake.params.

To force a specific backend, set FORCE_BACKEND environment variable:
    FORCE_BACKEND=destvi snakemake ...
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

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

# Check for forced backend
FORCE_BACKEND = os.environ.get("FORCE_BACKEND", "").lower().strip()
if FORCE_BACKEND:
    print(f"*** FORCE_BACKEND={FORCE_BACKEND} ***")

print("=" * 60)
print("Comparing Spatial Backends (Comprehensive Metrics)")
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


# =============================================================================
# NEW: Comprehensive metrics with proper weighting
# =============================================================================
# These are the metrics that actually matter for downstream niche modeling

METRICS_CONFIG = {
    # Higher is better
    "types_per_spot_mean": {"weight": 0.30, "higher_better": True},
    "effective_coverage": {"weight": 0.25, "higher_better": True},
    "global_type_coverage": {"weight": 0.20, "higher_better": True},
    "mean_entropy": {"weight": 0.15, "higher_better": True},  # Diversity is good
    # Lower is better
    "gini_coefficient_mean": {"weight": 0.10, "higher_better": False},
}

# Fallback to old metrics if new ones not available
FALLBACK_METRICS = {
    "coverage": {"weight": 0.4, "higher_better": True},
    "mean_entropy": {"weight": 0.3, "higher_better": True},
    "sparsity": {"weight": 0.3, "higher_better": False},
}


def compute_score(metrics_dict):
    """Compute weighted composite score using comprehensive metrics."""
    # Try new metrics first
    config = METRICS_CONFIG
    available = [m for m in config if m in metrics_dict]

    if len(available) < 2:
        # Fall back to old metrics
        config = FALLBACK_METRICS
        available = [m for m in config if m in metrics_dict]
        print(f"    (using fallback metrics: {available})")

    if not available:
        return 0.0

    score = 0.0
    total_weight = 0.0

    for metric_name in available:
        cfg = config[metric_name]
        val = metrics_dict.get(metric_name)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        # Normalize to [0, 1] range (approximate)
        if metric_name == "types_per_spot_mean":
            norm_val = min(val / 15.0, 1.0)  # Cap at 15 types
        elif metric_name in ["effective_coverage", "global_type_coverage", "coverage"]:
            norm_val = val  # Already [0, 1]
        elif metric_name == "mean_entropy":
            norm_val = val  # Already [0, 1]
        elif metric_name == "gini_coefficient_mean":
            norm_val = val  # Already [0, 1]
        elif metric_name == "sparsity":
            norm_val = val  # Already [0, 1]
        else:
            norm_val = val

        # Invert if lower is better
        if not cfg["higher_better"]:
            norm_val = 1.0 - norm_val

        score += norm_val * cfg["weight"]
        total_weight += cfg["weight"]

    return score / max(total_weight, 1e-6)


# Compute scores for all combinations
scores = {}  # scores[(label_source, backend)] = score
for label_source, backend_results in results.items():
    for backend, metrics in backend_results.items():
        key = (label_source, backend)
        scores[key] = compute_score(metrics)
        print(f"  {label_source}/{backend}: score={scores[key]:.4f}")

        # Print key metrics
        for m in ["types_per_spot_mean", "effective_coverage", "gini_coefficient_mean"]:
            if m in metrics:
                print(f"    {m}: {metrics[m]:.4f}")

print()
print("Composite Scores (ranked):")
for (ls, be), score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {ls}/{be}: {score:.4f}")

# =============================================================================
# Select canonical combination
# =============================================================================
if FORCE_BACKEND and FORCE_BACKEND in backends:
    # Forced selection - pick best label source for this backend
    forced_scores = {k: v for k, v in scores.items() if k[1] == FORCE_BACKEND}
    if forced_scores:
        best_key = max(forced_scores.keys(), key=lambda x: forced_scores[x])
        canonical_label_source, canonical_backend = best_key
        print(f"\n*** FORCED: {canonical_label_source}/{canonical_backend} (score: {scores[best_key]:.4f})")
    else:
        # Backend not found, fall back to best
        best_key = max(scores.keys(), key=lambda x: scores[x])
        canonical_label_source, canonical_backend = best_key
        print(f"\nForced backend '{FORCE_BACKEND}' not found, using best: {canonical_label_source}/{canonical_backend}")
else:
    # Automatic selection
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
hlca_scores = [scores[(ls, be)] for ls, be in scores if ls == 'hlca']
luca_scores = [scores[(ls, be)] for ls, be in scores if ls == 'luca']
hlca_avg = np.mean(hlca_scores) if hlca_scores else 0
luca_avg = np.mean(luca_scores) if luca_scores else 0

label_ablation['summary'] = {
    'hlca_mean_score': float(hlca_avg),
    'luca_mean_score': float(luca_avg),
    'better_label_source': 'hlca' if hlca_avg >= luca_avg else 'luca',
    'delta': float(abs(hlca_avg - luca_avg))
}

# =============================================================================
# Save outputs
# =============================================================================

# Save comparison
comparison = {
    'label_sources': label_sources,
    'backends': backends,
    'results': {ls: {be: m for be, m in bm.items()} for ls, bm in results.items()},
    'composite_scores': {f"{ls}/{be}": float(s) for (ls, be), s in scores.items()},
    'canonical': {
        'label_source': canonical_label_source,
        'backend': canonical_backend,
        'score': float(scores[best_key])
    },
    'metrics_config': {k: {"weight": v["weight"], "higher_better": v["higher_better"]}
                       for k, v in METRICS_CONFIG.items()},
    'timestamp': datetime.now().isoformat(),
}

with open(comparison_output, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"Saved comparison: {comparison_output}")

# Save canonical selection (used by downstream rules)
canonical = {
    'backend': canonical_backend,
    'label_source': canonical_label_source,
    'score': float(scores[best_key]),
    'forced': bool(FORCE_BACKEND),
    'reason': f"{'Forced by FORCE_BACKEND env var' if FORCE_BACKEND else 'Highest composite score'} ({scores[best_key]:.4f})",
    'alternatives': [
        {'label_source': ls, 'backend': be, 'score': float(s)}
        for (ls, be), s in sorted(scores.items(), key=lambda x: -x[1])
        if (ls, be) != best_key
    ][:3],
}
with open(canonical_output, 'w') as f:
    json.dump(canonical, f, indent=2)
print(f"Saved canonical: {canonical_output}")

# Save label ablation
with open(label_ablation_output, 'w') as f:
    json.dump(label_ablation, f, indent=2)
print(f"Saved label ablation: {label_ablation_output}")

# =============================================================================
# Create comparison figure
# =============================================================================
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
if ls_scores[0] > 0 or ls_scores[1] > 0:
    winner_idx = 0 if hlca_avg >= luca_avg else 1
    bars[winner_idx].set_edgecolor('green')
    bars[winner_idx].set_linewidth(3)

# 3. Key metrics for canonical backend
ax = axes[2]
canonical_metrics = results[canonical_label_source][canonical_backend]

# Show key metrics
display_metrics = ["types_per_spot_mean", "effective_coverage", "global_type_coverage",
                   "mean_entropy", "gini_coefficient_mean"]
available_metrics = [m for m in display_metrics if m in canonical_metrics]

if available_metrics:
    metric_vals = [canonical_metrics[m] for m in available_metrics]
    colors = ['#2ecc71' if METRICS_CONFIG.get(m, {}).get("higher_better", True) else '#e74c3c'
              for m in available_metrics]
    labels = [m.replace("_", "\n") for m in available_metrics]
    ax.barh(labels, metric_vals, color=colors)
    ax.set_xlabel('Value')
else:
    # Fallback to old metrics
    old_metrics = ["coverage", "mean_entropy", "sparsity"]
    metric_vals = [canonical_metrics.get(m, 0) for m in old_metrics]
    ax.barh(old_metrics, metric_vals, color='#2ecc71')
    ax.set_xlabel('Value')

forced_tag = " (FORCED)" if FORCE_BACKEND else ""
ax.set_title(f'Canonical: {canonical_label_source.upper()}/{canonical_backend}{forced_tag}')

plt.tight_layout()
plt.savefig(figure_output, dpi=150, bbox_inches='tight')
print(f"Saved figure: {figure_output}")

print()
print("=" * 60)
forced_msg = " (FORCED)" if FORCE_BACKEND else ""
print(f"Canonical backend selected: {canonical_label_source}/{canonical_backend}{forced_msg}")
print("=" * 60)
