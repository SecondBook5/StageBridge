#!/usr/bin/env python3
"""Aggregate per-sample spatial deconvolution results into backend-level summaries."""

import argparse
import json
from pathlib import Path

import pandas as pd


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

        # Load metrics
        metrics_file = sample_path / "upstream_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                metrics["sample"] = sample
                all_metrics.append(metrics)

        # Load proportions
        props_file = sample_path / "cell_type_proportions.parquet"
        if props_file.exists():
            props = pd.read_parquet(props_file)
            props["sample"] = sample
            all_proportions.append(props)

    if not all_metrics:
        raise RuntimeError(f"No metrics found for any samples in {sample_dir}")

    # Aggregate metrics (mean across samples)
    metrics_df = pd.DataFrame(all_metrics)
    numeric_cols = metrics_df.select_dtypes(include="number").columns

    aggregated_metrics = {
        "n_samples": len(all_metrics),
        "n_samples_total": len(samples),
        "backend": args.backend,
    }

    for col in numeric_cols:
        aggregated_metrics[f"{col}_mean"] = float(metrics_df[col].mean())
        aggregated_metrics[f"{col}_std"] = float(metrics_df[col].std())

    # Add simplified top-level metrics for comparison
    if "mean_entropy" in numeric_cols:
        aggregated_metrics["mean_entropy"] = aggregated_metrics["mean_entropy_mean"]
    if "coverage" in numeric_cols:
        aggregated_metrics["coverage"] = aggregated_metrics["coverage_mean"]
    if "sparsity" in numeric_cols:
        aggregated_metrics["sparsity"] = aggregated_metrics["sparsity_mean"]

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

    print(f"Aggregated results saved to {output_dir}")
    print(f"  Samples processed: {len(all_metrics)}/{len(samples)}")


if __name__ == "__main__":
    main()
