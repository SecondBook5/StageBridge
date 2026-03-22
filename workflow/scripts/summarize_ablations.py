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
        'description': data.get('description', ablation.replace('_', ' ').title()),
    }

    # Extract key metrics
    metrics = data.get('metrics', {})
    row['transition_mae'] = metrics.get('transition_mae', float('nan'))
    row['flow_correlation'] = metrics.get('flow_correlation', float('nan'))
    row['stage_accuracy'] = metrics.get('stage_accuracy', float('nan'))
    row['niche_influence_r2'] = metrics.get('niche_influence_r2', float('nan'))

    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values('ablation')

print()
print("Summary Table:")
print(df.to_string(index=False))

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
latex = df.to_latex(
    index=False,
    float_format="%.3f",
    caption="Ablation study results. Lower MAE is better; higher correlation/accuracy/R² is better.",
    label="tab:ablations",
    column_format="l" + "r" * (len(df.columns) - 1),
)

# Clean up column names for LaTeX
latex = latex.replace('transition_mae', 'Transition MAE')
latex = latex.replace('flow_correlation', 'Flow Corr.')
latex = latex.replace('stage_accuracy', 'Stage Acc.')
latex = latex.replace('niche_influence_r2', 'Niche $R^2$')
latex = latex.replace('ablation', 'Ablation')
latex = latex.replace('description', 'Description')
latex = latex.replace('_', r'\_')

with open(latex_output, 'w') as f:
    f.write(latex)
print(f"Saved LaTeX: {latex_output}")

print()
print("=" * 60)
print("Ablation Summary Complete")
print("=" * 60)
