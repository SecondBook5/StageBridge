#!/usr/bin/env python3
"""Aggregate training results into comparison_report.json for figure generation."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_full_results(results_dir: Path) -> dict:
    """Load full model training results."""
    losses = []

    for json_path in results_dir.glob("full/fold_*/seed_*/training_summary.json"):
        with open(json_path) as f:
            data = json.load(f)

        val_loss = data.get('transition', {}).get('best_val_loss')
        if val_loss is not None:
            losses.append(val_loss)

    if losses:
        return {
            'mean_val_loss': float(np.mean(losses)),
            'std_val_loss': float(np.std(losses)),
            'n_runs': len(losses),
            'all_losses': losses,
        }
    return {}


def load_ablation_results(results_dir: Path) -> dict:
    """Load all ablation study results."""
    ablations = defaultdict(list)

    for json_path in results_dir.glob("ablations/*/fold_*/seed_*/ablation_*.json"):
        with open(json_path) as f:
            data = json.load(f)

        ablation = data.get('ablation', 'unknown')
        val_loss = data.get('metrics', {}).get('transition', {}).get('best_val_loss')

        if val_loss is not None:
            ablations[ablation].append(val_loss)

    result = {}
    for ablation, losses in ablations.items():
        result[ablation] = {
            'mean_val_loss': float(np.mean(losses)),
            'std_val_loss': float(np.std(losses)),
            'n_runs': len(losses),
            'all_losses': losses,
        }

    return result


def load_baseline_results(results_dir: Path) -> dict:
    """Load all baseline results."""
    baselines = defaultdict(list)

    for json_path in results_dir.glob("baselines/*/fold_*/seed_*/baseline_*.json"):
        with open(json_path) as f:
            data = json.load(f)

        baseline = data.get('baseline', 'unknown')
        val_loss = data.get('metrics', {}).get('best_val_loss')

        if val_loss is not None:
            baselines[baseline].append(val_loss)

    result = {}
    for baseline, losses in baselines.items():
        result[baseline] = {
            'mean_val_loss': float(np.mean(losses)),
            'std_val_loss': float(np.std(losses)),
            'n_runs': len(losses),
            'all_losses': losses,
        }

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate results into comparison report")
    parser.add_argument("--results-dir", "-r", type=str, default="results/v1",
                        help="Results directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: results_dir/comparison_report.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "comparison_report.json"

    print("=" * 60)
    print("Aggregating Results")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Output: {output_path}")

    # Load all results
    print("\nLoading full model results...")
    full_results = load_full_results(results_dir)
    print(f"  Found {full_results.get('n_runs', 0)} runs")

    print("\nLoading ablation results...")
    ablation_results = load_ablation_results(results_dir)
    print(f"  Found {len(ablation_results)} ablation types")

    print("\nLoading baseline results...")
    baseline_results = load_baseline_results(results_dir)
    print(f"  Found {len(baseline_results)} baseline types")

    # Compute deltas vs full model
    full_mean = full_results.get('mean_val_loss', 0)

    for ablation, data in ablation_results.items():
        if full_mean > 0:
            data['delta_vs_full'] = (data['mean_val_loss'] - full_mean) / full_mean * 100
        else:
            data['delta_vs_full'] = 0

    for baseline, data in baseline_results.items():
        if full_mean > 0:
            data['delta_vs_full'] = (data['mean_val_loss'] - full_mean) / full_mean * 100
        else:
            data['delta_vs_full'] = 0

    # Build report
    report = {
        'full_model': full_results,
        'ablations': ablation_results,
        'baselines': baseline_results,
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if full_results:
        print(f"\nFull Model: {full_results['mean_val_loss']:.4f} +/- {full_results['std_val_loss']:.4f}")

    if ablation_results:
        print("\nAblations (sorted by impact):")
        sorted_abl = sorted(ablation_results.items(), key=lambda x: x[1]['delta_vs_full'], reverse=True)
        for name, data in sorted_abl:
            print(f"  {name:25s}: {data['mean_val_loss']:.4f} ({data['delta_vs_full']:+.1f}%)")

    if baseline_results:
        print("\nBaselines:")
        sorted_base = sorted(baseline_results.items(), key=lambda x: x[1]['mean_val_loss'])
        for name, data in sorted_base:
            print(f"  {name:25s}: {data['mean_val_loss']:.4f} ({data['delta_vs_full']:+.1f}%)")


if __name__ == "__main__":
    main()
