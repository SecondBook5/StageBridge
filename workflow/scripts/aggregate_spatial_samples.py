#!/usr/bin/env python3
"""Aggregate per-sample spatial deconvolution results into backend-level summaries.

Recomputes comprehensive metrics from cell_type_proportions.parquet files
to ensure all new metrics are available for backend comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_comprehensive_metrics(proportions: pd.DataFrame) -> dict:
    """
    Compute comprehensive metrics from cell type proportions.

    These are the metrics that matter for downstream niche modeling,
    aligned with the updated stagebridge.spatial_mapping.metrics module.
    """
    n_spots = len(proportions)
    n_celltypes = proportions.shape[1]

    if n_spots == 0 or n_celltypes == 0:
        return {"n_spots": 0, "n_celltypes": 0}

    # ==========================================================================
    # ENTROPY / DIVERSITY
    # ==========================================================================
    # Per-spot entropy (normalized to [0, 1])
    eps = 1e-10
    entropy_raw = -(proportions * np.log(proportions + eps)).sum(axis=1)
    max_entropy = np.log(n_celltypes)
    entropy = entropy_raw / max_entropy if max_entropy > 0 else entropy_raw

    # ==========================================================================
    # COVERAGE / RICHNESS
    # ==========================================================================
    # Types per spot: how many cell types detected (>1%) per spot
    types_per_spot = (proportions > 0.01).sum(axis=1)
    types_per_spot_5pct = (proportions > 0.05).sum(axis=1)

    # Effective coverage: types_per_spot / total_types
    effective_coverage = types_per_spot.mean() / n_celltypes if n_celltypes > 0 else 0

    # Global type coverage: fraction of types with >1% mean across all spots
    type_means = proportions.mean(axis=0)
    global_type_coverage = (type_means > 0.01).sum() / n_celltypes if n_celltypes > 0 else 0

    # Sparsity: fraction of zeros
    sparsity = (proportions == 0).sum().sum() / (n_spots * n_celltypes)

    # ==========================================================================
    # DOMINANCE / CONCENTRATION
    # ==========================================================================
    max_proportions = proportions.max(axis=1)

    # Old "coverage" metric (fraction where max > 0.5) - kept for compatibility
    coverage_old = (max_proportions > 0.5).mean()

    # Dominant types
    dominant_types = (proportions > 0.5).any(axis=0).sum()

    # Dominance ratio: max / 2nd-max
    sorted_props = np.sort(proportions.values, axis=1)[:, ::-1]
    second_max = sorted_props[:, 1] if n_celltypes > 1 else np.zeros(n_spots)
    dominance_ratio = max_proportions / (second_max + eps)

    # ==========================================================================
    # DISTRIBUTION SHAPE
    # ==========================================================================
    # Gini coefficient per spot
    def gini(arr):
        arr = np.array(arr).flatten()
        arr = arr[arr > 0]
        if len(arr) == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))

    gini_per_spot = [gini(proportions.iloc[i].values) for i in range(n_spots)]

    # ==========================================================================
    # PER-TYPE STATISTICS
    # ==========================================================================
    type_presence_rate = (proportions > 0.01).mean(axis=0)

    # ==========================================================================
    # COMPILE METRICS
    # ==========================================================================
    metrics = {
        # Basic counts
        "n_spots": int(n_spots),
        "n_celltypes": int(n_celltypes),

        # Entropy / Diversity
        "mean_entropy": float(entropy.mean()),
        "std_entropy": float(entropy.std()),
        "median_entropy": float(np.median(entropy)),

        # Coverage / Richness (THE IMPORTANT ONES)
        "types_per_spot_mean": float(types_per_spot.mean()),
        "types_per_spot_std": float(types_per_spot.std()),
        "types_per_spot_median": float(np.median(types_per_spot)),
        "types_per_spot_5pct_mean": float(types_per_spot_5pct.mean()),
        "effective_coverage": float(effective_coverage),
        "global_type_coverage": float(global_type_coverage),
        "sparsity": float(sparsity),
        "coverage": float(coverage_old),  # Legacy, for compatibility

        # Dominance
        "max_proportion_mean": float(max_proportions.mean()),
        "max_proportion_std": float(max_proportions.std()),
        "n_dominant_types": int(dominant_types),
        "dominance_ratio_mean": float(dominance_ratio.mean()),

        # Distribution shape
        "gini_coefficient_mean": float(np.mean(gini_per_spot)),
        "gini_coefficient_std": float(np.std(gini_per_spot)),

        # Per-type
        "type_presence_rate_mean": float(type_presence_rate.mean()),
        "type_presence_rate_std": float(type_presence_rate.std()),
        "n_types_never_present": int((type_presence_rate == 0).sum()),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, help="Backend name")
    parser.add_argument("--sample-dir", required=True, help="Directory with per-sample results")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregated results")
    parser.add_argument("--manifest", required=True, help="Sample manifest JSON")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    print(f"Aggregating {len(samples)} samples for {args.backend}")

    # Collect per-sample metrics and proportions
    all_metrics = []
    all_proportions = []

    for sample in samples:
        sample_path = sample_dir / sample

        # Load proportions and RECOMPUTE metrics
        props_file = sample_path / "cell_type_proportions.parquet"
        if props_file.exists():
            props = pd.read_parquet(props_file)

            # Recompute comprehensive metrics from proportions
            metrics = compute_comprehensive_metrics(props)
            metrics["sample"] = sample
            all_metrics.append(metrics)

            # Keep proportions for aggregation
            props["sample"] = sample
            all_proportions.append(props)
            print(f"  {sample}: {metrics['n_spots']} spots, {metrics['types_per_spot_mean']:.2f} types/spot")
        else:
            print(f"  {sample}: MISSING proportions file")

    if not all_metrics:
        raise RuntimeError(f"No proportions found for any samples in {sample_dir}")

    # Aggregate metrics (mean across samples)
    metrics_df = pd.DataFrame(all_metrics)
    numeric_cols = [c for c in metrics_df.columns if c != "sample" and metrics_df[c].dtype in [np.float64, np.int64, float, int]]

    aggregated_metrics = {
        "n_samples": len(all_metrics),
        "n_samples_total": len(samples),
        "backend": args.backend,
    }

    # Compute mean and std for each metric
    for col in numeric_cols:
        vals = metrics_df[col].dropna()
        if len(vals) > 0:
            aggregated_metrics[f"{col}_mean"] = float(vals.mean())
            aggregated_metrics[f"{col}_std"] = float(vals.std())

    # Add top-level metrics for comparison (means, for backward compatibility)
    key_metrics = [
        "mean_entropy", "coverage", "sparsity",  # Old
        "types_per_spot_mean", "effective_coverage", "global_type_coverage",  # New important
        "gini_coefficient_mean", "dominance_ratio_mean", "type_presence_rate_mean",  # New
    ]
    for m in key_metrics:
        if f"{m}_mean" in aggregated_metrics:
            aggregated_metrics[m] = aggregated_metrics[f"{m}_mean"]
        elif m in numeric_cols:
            aggregated_metrics[m] = float(metrics_df[m].mean())

    # Save aggregated metrics
    with open(output_dir / "upstream_metrics.json", "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

    # Concatenate all proportions
    if all_proportions:
        combined_props = pd.concat(all_proportions, ignore_index=False)
        combined_props.to_parquet(output_dir / "cell_type_proportions.parquet")
        print(f"Combined proportions: {combined_props.shape}")

    # Save per-sample metrics for reference
    metrics_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)

    print(f"\nAggregated results saved to {output_dir}")
    print(f"  Samples processed: {len(all_metrics)}/{len(samples)}")
    print(f"  Key metrics:")
    for m in ["types_per_spot_mean", "effective_coverage", "gini_coefficient_mean"]:
        if m in aggregated_metrics:
            print(f"    {m}: {aggregated_metrics[m]:.4f}")


if __name__ == "__main__":
    main()
