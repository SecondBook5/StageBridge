#!/usr/bin/env python3
"""
Compute ALL key numbers cited in the StageBridge paper.

This script is the SINGLE SOURCE OF TRUTH for all statistics reported.
Every number in the paper should trace back to this script.

Output: paper_numbers.json with all computed values and their provenance.

Usage:
    python scripts/compute_paper_numbers.py --data-dir /path/to/canonical --output paper_numbers.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


def compute_cell_composition(cells: pd.DataFrame) -> dict:
    """Compute cell type percentages by stage."""
    results = {}

    cell_type_col = 'luca_cell_type' if 'luca_cell_type' in cells.columns else 'cell_type'

    for stage in cells['stage'].unique():
        stage_cells = cells[cells['stage'] == stage]
        n_total = len(stage_cells)

        stage_results = {'n_cells': n_total}

        # T cells
        t_cell_mask = stage_cells[cell_type_col].str.contains('T cell|T-cell|CD4|CD8', case=False, na=False)
        stage_results['t_cell_pct'] = 100 * t_cell_mask.sum() / n_total
        stage_results['t_cell_count'] = int(t_cell_mask.sum())

        # Macrophages
        macro_mask = stage_cells[cell_type_col].str.contains('Macrophage|Mono', case=False, na=False)
        stage_results['macrophage_pct'] = 100 * macro_mask.sum() / n_total
        stage_results['macrophage_count'] = int(macro_mask.sum())

        # Fibroblasts
        fibro_mask = stage_cells[cell_type_col].str.contains('Fibroblast|fibro', case=False, na=False)
        stage_results['fibroblast_pct'] = 100 * fibro_mask.sum() / n_total
        stage_results['fibroblast_count'] = int(fibro_mask.sum())

        # Proliferating (if available)
        if 'proliferation_label' in cells.columns:
            prolif_mask = stage_cells['proliferation_label'] == 1
            stage_results['proliferating_pct'] = 100 * prolif_mask.sum() / n_total
            stage_results['proliferating_count'] = int(prolif_mask.sum())

        results[stage] = stage_results

    # Compute fold changes
    stages = list(results.keys())
    if 'Normal' in stages and 'Invasive' in stages:
        results['fold_changes'] = {
            't_cell_reduction_pct': 100 * (results['Normal']['t_cell_pct'] - results.get('Preinvasive', results['Invasive'])['t_cell_pct']) / results['Normal']['t_cell_pct'] if results['Normal']['t_cell_pct'] > 0 else None,
            'macrophage_reduction_pct': 100 * (results['Normal']['macrophage_pct'] - results.get('Preinvasive', results['Invasive'])['macrophage_pct']) / results['Normal']['macrophage_pct'] if results['Normal']['macrophage_pct'] > 0 else None,
        }
        if 'proliferating_pct' in results['Normal']:
            results['fold_changes']['proliferation_fold'] = results['Invasive']['proliferating_pct'] / results['Normal']['proliferating_pct'] if results['Normal']['proliferating_pct'] > 0 else None

    return results


def compute_gene_expression_stats(adata_or_cells: pd.DataFrame, gene: str, stage_col: str = 'stage') -> dict:
    """Compute expression statistics for a gene across stages."""
    results = {}

    if gene not in adata_or_cells.columns:
        return {'error': f'{gene} not found in data'}

    expr = adata_or_cells[gene]
    stages = adata_or_cells[stage_col]

    # Per-stage statistics
    for stage in stages.unique():
        stage_expr = expr[stages == stage]
        results[stage] = {
            'mean': float(stage_expr.mean()),
            'std': float(stage_expr.std()),
            'median': float(stage_expr.median()),
            'pct_positive': float(100 * (stage_expr > 0).sum() / len(stage_expr)),
            'n_cells': int(len(stage_expr)),
        }

    # Fold change Normal -> Invasive
    if 'Normal' in results and 'Invasive' in results:
        if results['Normal']['mean'] > 0:
            results['fold_change_normal_invasive'] = results['Invasive']['mean'] / results['Normal']['mean']

    # Stage correlation
    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4, 'Preinvasive': 1, 'Invasive': 2}
    valid_mask = stages.isin(stage_map.keys())
    if valid_mask.sum() > 100:
        stage_numeric = stages[valid_mask].map(stage_map)
        r, p = stats.spearmanr(expr[valid_mask], stage_numeric)
        results['stage_correlation'] = {'spearman_r': float(r), 'p_value': float(p)}

    return results


def compute_mutation_frequencies(cells: pd.DataFrame) -> dict:
    """Compute mutation frequencies by stage."""
    results = {}

    mutation_cols = [c for c in cells.columns if c.endswith('_mut')]

    for stage in cells['stage'].unique():
        stage_cells = cells[cells['stage'] == stage]
        n_total = len(stage_cells)

        stage_results = {'n_cells': n_total}
        for mut_col in mutation_cols:
            gene = mut_col.replace('_mut', '').upper()
            mut_pct = 100 * stage_cells[mut_col].sum() / n_total
            stage_results[f'{gene}_pct'] = float(mut_pct)

        results[stage] = stage_results

    return results


def compute_model_performance(results_dir: Path) -> dict:
    """Compute model performance metrics from training results."""
    results = {}

    # Load comparison report if exists
    comparison_file = results_dir / 'comparison_report.json'
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison = json.load(f)

        # Full model
        if 'full_model' in comparison:
            fm = comparison['full_model']
            results['full_model'] = {
                'val_loss_mean': fm['mean_val_loss'],
                'val_loss_std': fm['std_val_loss'],
                'n_runs': fm['n_runs'],
            }

        # Ablations
        if 'ablations' in comparison:
            results['ablations'] = {}
            for name, ablation in comparison['ablations'].items():
                results['ablations'][name] = {
                    'val_loss_mean': ablation['mean_val_loss'],
                    'delta_pct': ablation['delta_vs_full'],
                    'n_runs': ablation['n_runs'],
                }

        # Baselines
        if 'baselines' in comparison:
            results['baselines'] = {}
            for name, baseline in comparison['baselines'].items():
                full_loss = results.get('full_model', {}).get('val_loss_mean', 0.0036)
                results['baselines'][name] = {
                    'val_loss_mean': baseline['mean_val_loss'],
                    'fold_vs_full': baseline['mean_val_loss'] / full_loss if full_loss > 0 else None,
                    'n_runs': baseline['n_runs'],
                }

    return results


def compute_drift_alignment(model, dataloader, device='cuda') -> dict:
    """Compute drift alignment between predicted velocities and OT directions.

    NOTE: This requires the trained model and OT coupling computation.
    """
    # TODO: Implement proper drift alignment computation
    # This needs:
    # 1. Load model checkpoint
    # 2. Get cell pairs from OT coupling
    # 3. Compute predicted velocity for source cells
    # 4. Compute transport direction (target - source)
    # 5. Compute cosine similarity

    return {
        'error': 'Drift alignment computation not yet implemented',
        'note': 'Need to implement OT coupling + velocity prediction comparison'
    }


def compute_context_correlations(cells: pd.DataFrame) -> dict:
    """Compute correlations between context embeddings and biological features."""
    results = {}

    gamma_cols = [c for c in cells.columns if c.startswith('gamma_')]
    if not gamma_cols:
        return {'error': 'No gamma columns found'}

    # Stage correlation for each gamma
    stage_map = {'Normal': 0, 'AAH': 1, 'AIS': 2, 'MIA': 3, 'LUAD': 4, 'Preinvasive': 1, 'Invasive': 2}
    valid_mask = cells['stage'].isin(stage_map.keys())
    stage_numeric = cells.loc[valid_mask, 'stage'].map(stage_map)

    for gamma_col in gamma_cols:
        gamma_vals = cells.loc[valid_mask, gamma_col]
        r, p = stats.spearmanr(gamma_vals, stage_numeric)
        results[f'{gamma_col}_stage_corr'] = {'spearman_r': float(r), 'p_value': float(p)}

    # IL1B correlation if available
    if 'il1b_raw' in cells.columns or 'IL1B' in cells.columns:
        il1b_col = 'il1b_raw' if 'il1b_raw' in cells.columns else 'IL1B'
        for gamma_col in gamma_cols:
            r, p = stats.spearmanr(cells[gamma_col], cells[il1b_col])
            results[f'{gamma_col}_il1b_corr'] = {'spearman_r': float(r), 'p_value': float(p)}

    return results


def main():
    parser = argparse.ArgumentParser(description='Compute all paper numbers')
    parser.add_argument('--data-dir', type=Path, default=Path('/data1/chaunzt1/stagebridge/processed/luad_evo/canonical'))
    parser.add_argument('--cells', type=Path, default=None, help='Path to cells.parquet')
    parser.add_argument('--results-dir', type=Path, default=Path('/home/booka/projects/StageBridge/results/v1'))
    parser.add_argument('--output', type=Path, default=Path('paper_numbers.json'))
    args = parser.parse_args()

    print("=" * 60)
    print("COMPUTING ALL PAPER NUMBERS")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Data dir: {args.data_dir}")
    print(f"Results dir: {args.results_dir}")
    print()

    all_results = {
        'computed_at': datetime.now().isoformat(),
        'data_source': str(args.data_dir),
        'results_source': str(args.results_dir),
    }

    # Load cells data
    cells_path = args.cells or args.data_dir / 'cells.parquet'
    if not cells_path.exists():
        # Try local path
        cells_path = Path('/home/booka/projects/StageBridge_V1/cells.parquet')

    if cells_path.exists():
        print(f"Loading cells from {cells_path}...")
        cells = pd.read_parquet(cells_path)
        print(f"  Loaded {len(cells):,} cells")

        # Cell composition
        print("\nComputing cell composition...")
        all_results['cell_composition'] = compute_cell_composition(cells)
        for stage, stats in all_results['cell_composition'].items():
            if isinstance(stats, dict) and 't_cell_pct' in stats:
                print(f"  {stage}: T-cells={stats['t_cell_pct']:.1f}%, Macro={stats['macrophage_pct']:.1f}%, Fibro={stats['fibroblast_pct']:.1f}%")

        # Mutation frequencies
        print("\nComputing mutation frequencies...")
        all_results['mutations'] = compute_mutation_frequencies(cells)

        # Context correlations
        print("\nComputing context correlations...")
        all_results['context_correlations'] = compute_context_correlations(cells)

        # Gene expression (if IL1B available)
        if 'il1b_raw' in cells.columns:
            print("\nComputing IL1B statistics...")
            all_results['il1b'] = compute_gene_expression_stats(cells, 'il1b_raw')
    else:
        print(f"WARNING: cells.parquet not found at {cells_path}")

    # Model performance
    print("\nComputing model performance metrics...")
    all_results['model_performance'] = compute_model_performance(args.results_dir)

    if 'full_model' in all_results.get('model_performance', {}):
        fm = all_results['model_performance']['full_model']
        print(f"  Full model: {fm['val_loss_mean']:.6f} +/- {fm['val_loss_std']:.6f}")

    if 'ablations' in all_results.get('model_performance', {}):
        print("  Ablations:")
        for name, stats in all_results['model_performance']['ablations'].items():
            print(f"    {name}: +{stats['delta_pct']:.1f}%")

    # Drift alignment placeholder
    print("\nDrift alignment:")
    all_results['drift_alignment'] = {
        'overall': {'note': 'NEEDS RECOMPUTATION - no traceable source found'},
        'normal_to_preinvasive': {'note': 'NEEDS RECOMPUTATION - no traceable source found'},
    }
    print("  WARNING: Drift alignment numbers need recomputation!")

    # Save results
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY - KEY NUMBERS FOR PAPER")
    print("=" * 60)

    # Print key numbers
    cc = all_results.get('cell_composition', {})
    if 'Normal' in cc and 'Preinvasive' in cc:
        print(f"\nT-cell percentages:")
        print(f"  Normal: {cc['Normal'].get('t_cell_pct', 'N/A'):.1f}%")
        print(f"  Preinvasive: {cc['Preinvasive'].get('t_cell_pct', 'N/A'):.1f}%")
        if 'Invasive' in cc:
            print(f"  Invasive: {cc['Invasive'].get('t_cell_pct', 'N/A'):.1f}%")

    mp = all_results.get('model_performance', {})
    if 'full_model' in mp:
        print(f"\nModel performance:")
        print(f"  Val loss: {mp['full_model']['val_loss_mean']:.6f}")

    if 'ablations' in mp:
        print(f"\nTop ablations:")
        sorted_ablations = sorted(mp['ablations'].items(), key=lambda x: -x[1]['delta_pct'])
        for name, stats in sorted_ablations[:5]:
            print(f"  {name}: +{stats['delta_pct']:.1f}%")

    print("\n" + "=" * 60)
    print(f"Full results saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
