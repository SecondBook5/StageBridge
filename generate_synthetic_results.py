#!/usr/bin/env python3
"""
Generate Comprehensive Synthetic Results for Notebook

This script runs the synthetic pipeline and generates all analysis results
that can be embedded in the comprehensive notebook.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path('outputs/synthetic_test')
RESULTS_DIR = OUTPUT_DIR / 'results_summary'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING COMPREHENSIVE SYNTHETIC RESULTS")
print("="*80)

# Load data
print("\n[1/6] Loading data...")
cells_df = pd.read_parquet(OUTPUT_DIR / 'cells.parquet')
neighborhoods_df = pd.read_parquet(OUTPUT_DIR / 'neighborhoods.parquet')
stage_edges_df = pd.read_parquet(OUTPUT_DIR / 'stage_edges.parquet')
with open(OUTPUT_DIR / 'split_manifest.json') as f:
    splits = json.load(f)

print(f"  ✓ Loaded {len(cells_df)} cells")
print(f"  ✓ Loaded {len(neighborhoods_df)} neighborhoods")
print(f"  ✓ Loaded {len(stage_edges_df)} stage edges")
print(f"  ✓ Loaded {len(splits)} CV folds")

# Generate dataset statistics (Table 1 equivalent)
print("\n[2/6] Generating dataset statistics...")
table1 = pd.DataFrame([
    {
        'Metric': 'Total Cells',
        'Value': len(cells_df),
    },
    {
        'Metric': 'Donors',
        'Value': cells_df['donor_id'].nunique(),
    },
    {
        'Metric': 'Stages',
        'Value': cells_df['stage'].nunique(),
    },
    {
        'Metric': 'Latent Dimensions',
        'Value': 32,
    },
    {
        'Metric': 'Neighborhoods',
        'Value': len(neighborhoods_df),
    },
    {
        'Metric': 'Valid Transitions',
        'Value': len(stage_edges_df),
    },
])
table1.to_csv(RESULTS_DIR / 'table1_dataset_stats.csv', index=False)
print("  ✓ Saved Table 1: Dataset Statistics")
print(table1.to_string(index=False))

# Generate stage distribution figure
print("\n[3/6] Generating data overview figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Cells per stage
stage_counts = cells_df['stage'].value_counts().sort_index()
axes[0,0].bar(range(len(stage_counts)), stage_counts.values, color='steelblue')
axes[0,0].set_xticks(range(len(stage_counts)))
axes[0,0].set_xticklabels(stage_counts.index, rotation=45, ha='right')
axes[0,0].set_title("A. Cells per Stage", fontweight='bold', fontsize=12)
axes[0,0].set_ylabel("Cell Count")
axes[0,0].grid(axis='y', alpha=0.3)

# Panel B: Donors per stage
donor_stage = cells_df.groupby('stage')['donor_id'].nunique().sort_index()
axes[0,1].bar(range(len(donor_stage)), donor_stage.values, color='coral')
axes[0,1].set_xticks(range(len(donor_stage)))
axes[0,1].set_xticklabels(donor_stage.index, rotation=45, ha='right')
axes[0,1].set_title("B. Donors per Stage", fontweight='bold', fontsize=12)
axes[0,1].set_ylabel("Donor Count")
axes[0,1].grid(axis='y', alpha=0.3)

# Panel C: TMB distribution
for stage in cells_df['stage'].unique():
    stage_data = cells_df[cells_df['stage'] == stage]['tmb']
    axes[1,0].hist(stage_data, bins=20, alpha=0.5, label=stage)
axes[1,0].set_title("C. TMB Distribution by Stage", fontweight='bold', fontsize=12)
axes[1,0].set_xlabel("Tumor Mutational Burden")
axes[1,0].set_ylabel("Count")
axes[1,0].legend()
axes[1,0].grid(axis='y', alpha=0.3)

# Panel D: Latent space (first 2 dims)
for stage in cells_df['stage'].unique():
    stage_cells = cells_df[cells_df['stage'] == stage]
    z_values = np.stack(stage_cells['z_fused'].values)
    axes[1,1].scatter(z_values[:, 0], z_values[:, 1], alpha=0.6, label=stage, s=20)
axes[1,1].set_title("D. Latent Space (First 2D)", fontweight='bold', fontsize=12)
axes[1,1].set_xlabel("Latent Dimension 1")
axes[1,1].set_ylabel("Latent Dimension 2")
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.suptitle("Synthetic Dataset Overview", fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'figure2_data_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved Figure 2: Data Overview")

# Generate neighborhood analysis
print("\n[4/6] Analyzing neighborhood structure...")
# Compute average niche size
niche_sizes = []
for idx, row in neighborhoods_df.iterrows():
    tokens = row['tokens']
    if isinstance(tokens, (list, np.ndarray)):
        # Count cells in rings 1-4
        total_cells = sum(t.get('n_cells', 0) or 0 if isinstance(t, dict) else 0 for t in tokens[1:5])
        if total_cells > 0:
            niche_sizes.append(total_cells)

niche_analysis = pd.DataFrame([
    {'Metric': 'Mean Niche Size', 'Value': f"{np.mean(niche_sizes):.1f}"},
    {'Metric': 'Std Niche Size', 'Value': f"{np.std(niche_sizes):.1f}"},
    {'Metric': 'Min Niche Size', 'Value': int(np.min(niche_sizes))},
    {'Metric': 'Max Niche Size', 'Value': int(np.max(niche_sizes))},
])
niche_analysis.to_csv(RESULTS_DIR / 'niche_analysis.csv', index=False)
print("  ✓ Niche statistics:")
print(niche_analysis.to_string(index=False))

# Generate CV split analysis
print("\n[5/6] Analyzing CV splits...")
split_summary = []
for fold_name, cell_ids in splits.items():
    fold_cells = cells_df[cells_df['cell_id'].isin(cell_ids)]
    split_summary.append({
        'Fold': fold_name,
        'Cells': len(fold_cells),
        'Donors': fold_cells['donor_id'].nunique(),
        'Stages': fold_cells['stage'].nunique(),
    })
split_df = pd.DataFrame(split_summary)
split_df.to_csv(RESULTS_DIR / 'cv_splits.csv', index=False)
print("  ✓ CV split summary:")
print(split_df.to_string(index=False))

# Generate expected results placeholder
print("\n[6/6] Creating expected results template...")
expected_results = {
    "training": {
        "n_epochs": 3,
        "n_folds": len([k for k in splits.keys() if 'train' in k]),
        "expected_metrics": {
            "wasserstein": "~0.70-0.80 (lower is better)",
            "mse": "~0.30-0.40",
            "mae": "~0.25-0.35",
        },
        "notes": "Synthetic data has known ground truth, so metrics should be good"
    },
    "ablations": {
        "n_ablations": 8,
        "expected_rankings": [
            "1. full_model (best)",
            "2. no_wes (small degradation)",
            "3. flat_hierarchy (moderate degradation)",
            "4. pooled_niche (larger degradation)",
            "5-8. Others (varying degradation)"
        ]
    },
    "biology": {
        "expected_findings": [
            "Niche influence increases with stage progression",
            "TMB correlates with advanced stages",
            "Spatial proximity matters (rings 1-2 > rings 3-4)"
        ]
    }
}

with open(RESULTS_DIR / 'expected_results.json', 'w') as f:
    json.dump(expected_results, f, indent=2)

print("  ✓ Expected results template created")

# Summary
print("\n" + "="*80)
print("✓ RESULTS GENERATION COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {RESULTS_DIR}")
print("\nGenerated files:")
for f in sorted(RESULTS_DIR.glob("*")):
    print(f"  - {f.name}")

print("\n📊 Key Statistics:")
print(f"  Cells: {len(cells_df):,}")
print(f"  Stages: {cells_df['stage'].nunique()}")
print(f"  Donors: {cells_df['donor_id'].nunique()}")
print(f"  Mean niche size: {np.mean(niche_sizes):.1f} cells")
print(f"  CV folds: {len([k for k in splits.keys() if 'train' in k])}")

print("\n🎯 Next Steps:")
print("  1. Wait for training to complete")
print("  2. Load results.json from training directory")
print("  3. Generate attention visualizations")
print("  4. Add results to notebook")

print("\n✓ Ready to incorporate into comprehensive notebook!")
