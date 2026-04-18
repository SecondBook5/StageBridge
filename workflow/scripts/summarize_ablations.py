#!/usr/bin/env python3
"""Summarize ablation study results for publication.

Snakemake script - uses snakemake.input and snakemake.output.
"""

import json
import pandas as pd
from pathlib import Path

# Snakemake provides these
result_files = snakemake.input
summary_output = snakemake.output.summary
table_output = snakemake.output.table
latex_output = snakemake.output.latex

print("=" * 60)
print("Summarizing Ablation Studies")
print("=" * 60)

# Load all results
results = {}
for f in result_files:
    path = Path(f)
    ablation = path.parent.name
    with open(f) as fh:
        results[ablation] = json.load(fh)
    print(f"  Loaded {ablation}")

# Build summary table
rows = []
for ablation, data in results.items():
    row = {
        'ablation': ablation,
    }

    # Extract metrics - handle both old and new format
    if 'metrics' in data:
        # Old format
        metrics = data['metrics']
        row['mae'] = metrics.get('transition_mae', float('nan'))
        row['wasserstein'] = metrics.get('wasserstein', float('nan'))
    elif 'test_metrics' in data:
        # New format from run_v1_full.py
        metrics = data['test_metrics']
        row['mae'] = metrics.get('mae', float('nan'))
        row['mse'] = metrics.get('mse', float('nan'))
        row['wasserstein'] = metrics.get('wasserstein', float('nan'))
        row['loss'] = metrics.get('loss', float('nan'))
    else:
        # Fallback
        row['mae'] = float('nan')
        row['wasserstein'] = float('nan')

    # Get config info
    config = data.get('config', {})
    if isinstance(config, dict) and 'model' in config:
        model_config = config.get('model', {})
        row['niche_encoder'] = model_config.get('niche_encoder_type', 'unknown')
        row['use_set_encoder'] = model_config.get('use_set_encoder', False)
        row['no_niche'] = model_config.get('no_niche', False)
        row['deterministic'] = model_config.get('deterministic', False)
        row['use_prototypes'] = model_config.get('use_prototypes', False)

    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values('wasserstein', ascending=True)  # Best first

print()
print("Summary Table:")
print(df.to_string(index=False))

# Compute effect sizes vs full model (if present)
if 'full_model' in results and 'test_metrics' in results['full_model']:
    full_w = results['full_model']['test_metrics'].get('wasserstein', float('nan'))
    print(f"\nFull model Wasserstein: {full_w:.4f}")
    print("\nEffect sizes (delta from full model):")
    for ablation, data in results.items():
        if ablation == 'full_model':
            continue
        if 'test_metrics' in data:
            abl_w = data['test_metrics'].get('wasserstein', float('nan'))
            delta = abl_w - full_w
            pct = (delta / full_w) * 100 if full_w > 0 else float('nan')
            print(f"  {ablation}: {delta:+.4f} ({pct:+.1f}%)")

# Save summary JSON
summary = {
    'n_ablations': len(results),
    'ablations': list(results.keys()),
    'results': results,
}
with open(summary_output, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary: {summary_output}")

# Save CSV table
df.to_csv(table_output, index=False)
print(f"Saved table: {table_output}")

# Generate LaTeX table
# Select key columns for publication
pub_cols = ['ablation', 'wasserstein', 'mae', 'mse']
pub_cols = [c for c in pub_cols if c in df.columns]
df_pub = df[pub_cols].copy()

latex = df_pub.to_latex(
    index=False,
    float_format="%.4f",
    caption="Ablation study results. Lower Wasserstein distance and MAE indicate better transition modeling.",
    label="tab:ablations",
    column_format="l" + "r" * (len(df_pub.columns) - 1),
)

# Clean up column names for LaTeX
latex = latex.replace('wasserstein', 'Wasserstein')
latex = latex.replace('mae', 'MAE')
latex = latex.replace('mse', 'MSE')
latex = latex.replace('ablation', 'Ablation')
latex = latex.replace('_', r'\_')

with open(latex_output, 'w') as f:
    f.write(latex)
print(f"Saved LaTeX: {latex_output}")

print()
print("=" * 60)
print("Ablation Summary Complete")
print("=" * 60)
