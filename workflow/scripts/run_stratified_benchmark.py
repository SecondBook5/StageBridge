#!/usr/bin/env python3
"""
Stratified Spatial Backend Benchmark.

Runs all 4 backends on a stratified sample selection:
- 1 Normal (only available)
- 2 AAH (from different donors)
- 2 AIS (from different donors)
- 2 MIA (from different donors)
- 2 LUAD (from different donors)

Total: 9 samples x 4 backends = 36 runs

Supports label source ablation (HLCA vs LuCA):
- --label-source hlca: Use HLCA cell type labels (default)
- --label-source luca: Use LuCA cell type labels
- --label-source both: Run both and compare

Usage:
    python run_stratified_benchmark.py --output-dir /path/to/output [--label-source hlca|luca|both]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Stratified sample selection (maximizing donor diversity)
STRATIFIED_SAMPLES = {
    "Normal": [
        "GSM9226174_P4_Normal",  # Only Normal sample
    ],
    "AAH": [
        "GSM9226168_P1_AAH",     # Donor P1
        "GSM9226170_P2_AAH",     # Donor P2
    ],
    "AIS": [
        "GSM9226172_P3_AIS",     # Donor P3
        "GSM9226178_P5_AIS",     # Donor P5
    ],
    "MIA": [
        "GSM9226189_P10_MIA",    # Donor P10
        "GSM9226195_P13_MIA",    # Donor P13
    ],
    "LUAD": [
        "GSM9226169_P1_LUAD",    # Donor P1
        "GSM9226173_P3_LUAD",    # Donor P3
    ],
}

# Default data paths (HPC paths)
DEFAULT_SNRNA = "/scratch/chaunzt1/stagebridge/processed/luad_evo/snrna_with_celltypes.h5ad"
DEFAULT_SPATIAL = "/scratch/chaunzt1/stagebridge/processed/luad_evo/spatial_merged.h5ad"
DEFAULT_LABELS = "/scratch/chaunzt1/stagebridge/processed/luad_evo/reference_geometry/cell_types.parquet"

BACKENDS = ["tangram", "destvi", "tacco", "cell2location"]


def get_all_samples():
    """Get flat list of all stratified samples."""
    samples = []
    for stage, stage_samples in STRATIFIED_SAMPLES.items():
        for sample in stage_samples:
            samples.append((stage, sample))
    return samples


def run_benchmark_for_sample(
    sample_id: str,
    stage: str,
    snrna_path: Path,
    spatial_path: Path,
    output_dir: Path,
    backends: list[str],
    quick: bool = False,
    label_source: str = "hlca",
    labels_parquet: Path | None = None,
) -> dict:
    """Run benchmark for a single sample."""
    sample_output = output_dir / f"{stage}_{sample_id}"
    sample_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "stagebridge.pipelines.run_spatial_benchmark",
        "--snrna", str(snrna_path),
        "--spatial", str(spatial_path),
        "--output_dir", str(sample_output),
        "--sample", sample_id,
        "--sample-col", "sample_id",
        "--backends", *backends,
        "--label-source", label_source,
    ]

    if label_source == "luca" and labels_parquet:
        cmd.extend(["--labels-parquet", str(labels_parquet)])

    if quick:
        cmd.append("--quick")

    print(f"\n{'='*80}")
    print(f"Running benchmark for {sample_id} ({stage})")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")

    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()

    return {
        "sample_id": sample_id,
        "stage": stage,
        "returncode": result.returncode,
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "output_dir": str(sample_output),
    }


def aggregate_results(output_dir: Path) -> dict:
    """Aggregate results from all sample benchmarks."""
    results = {
        "samples": [],
        "backend_scores": {b: [] for b in BACKENDS},
        "backend_failures": {b: 0 for b in BACKENDS},
    }

    # Find all backend_comparison.json files
    for comparison_file in output_dir.glob("*/backend_comparison.json"):
        with open(comparison_file) as f:
            comparison = json.load(f)

        sample_dir = comparison_file.parent.name
        results["samples"].append(sample_dir)

        # Extract rankings
        if "rankings" in comparison:
            for ranking in comparison["rankings"]:
                backend = ranking["backend"]
                if backend in results["backend_scores"]:
                    results["backend_scores"][backend].append(ranking["composite_score"])

        # Count failures
        if "backends" in comparison:
            for backend, data in comparison["backends"].items():
                if data.get("status") == "failed":
                    results["backend_failures"][backend] += 1

    # Compute aggregate scores
    results["aggregate_scores"] = {}
    for backend, scores in results["backend_scores"].items():
        if scores:
            results["aggregate_scores"][backend] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "n_samples": len(scores),
                "n_failures": results["backend_failures"][backend],
            }

    # Rank backends
    if results["aggregate_scores"]:
        ranked = sorted(
            results["aggregate_scores"].items(),
            key=lambda x: (x[1]["n_failures"], -x[1]["mean"])  # Fewer failures, higher score
        )
        results["ranking"] = [b for b, _ in ranked]
        results["recommended_backend"] = ranked[0][0]

    return results


def main():
    parser = argparse.ArgumentParser(description="Stratified Spatial Backend Benchmark")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--snrna", type=str, default=DEFAULT_SNRNA,
        help="Path to snRNA h5ad"
    )
    parser.add_argument(
        "--spatial", type=str, default=DEFAULT_SPATIAL,
        help="Path to spatial h5ad"
    )
    parser.add_argument(
        "--labels-parquet", type=str, default=DEFAULT_LABELS,
        help="Path to cell_types.parquet (for LuCA labels)"
    )
    parser.add_argument(
        "--backends", type=str, nargs="+", default=BACKENDS,
        help="Backends to benchmark"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced epochs for faster testing"
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", default=None,
        help="Only run specific stages (e.g., --stages Normal AAH)"
    )
    parser.add_argument(
        "--label-source", type=str, default="hlca",
        choices=["hlca", "luca", "both"],
        help="Cell type label source: hlca (default), luca, or both for ablation"
    )
    args = parser.parse_args()

    # Determine label sources to run
    if args.label_source == "both":
        label_sources = ["hlca", "luca"]
    else:
        label_sources = [args.label_source]

    # Get samples to run
    all_samples = get_all_samples()
    if args.stages:
        all_samples = [(s, sid) for s, sid in all_samples if s in args.stages]

    print(f"Stratified Spatial Backend Benchmark")
    print(f"====================================")
    print(f"Samples: {len(all_samples)}")
    print(f"Backends: {args.backends}")
    print(f"Label sources: {label_sources}")
    print(f"Quick mode: {args.quick}")
    print()

    # Show sample selection
    print("Sample selection:")
    for stage, samples in STRATIFIED_SAMPLES.items():
        if args.stages is None or stage in args.stages:
            print(f"  {stage}: {samples}")
    print()

    # Run benchmarks for each label source
    all_results = {}
    for label_source in label_sources:
        output_dir = Path(args.output_dir) / label_source
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*80}")
        print(f"# RUNNING WITH {label_source.upper()} LABELS")
        print(f"# Output: {output_dir}")
        print(f"{'#'*80}")

        run_results = []
        for stage, sample_id in all_samples:
            result = run_benchmark_for_sample(
                sample_id=sample_id,
                stage=stage,
                snrna_path=Path(args.snrna),
                spatial_path=Path(args.spatial),
                output_dir=output_dir,
                backends=args.backends,
                quick=args.quick,
                label_source=label_source,
                labels_parquet=Path(args.labels_parquet) if args.labels_parquet else None,
            )
            run_results.append(result)

            # Save intermediate progress
            with open(output_dir / "run_progress.json", "w") as f:
                json.dump(run_results, f, indent=2)

        all_results[label_source] = run_results

    # Aggregate results for each label source
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")

    combined_results = {"label_sources": {}}

    for label_source in label_sources:
        output_dir = Path(args.output_dir) / label_source
        aggregate = aggregate_results(output_dir)
        aggregate["label_source"] = label_source
        combined_results["label_sources"][label_source] = aggregate

        # Save per-source aggregate
        with open(output_dir / "stratified_benchmark_results.json", "w") as f:
            json.dump(aggregate, f, indent=2)

        # Print summary
        print(f"\n{label_source.upper()} Labels Summary")
        print(f"-" * 40)
        print(f"Samples processed: {len(aggregate.get('samples', []))}")
        print()
        print("Backend Rankings:")
        for i, backend in enumerate(aggregate.get("ranking", []), 1):
            if backend in aggregate.get("aggregate_scores", {}):
                scores = aggregate["aggregate_scores"][backend]
                print(f"  {i}. {backend.upper()}: mean={scores['mean']:.3f}, "
                      f"failures={scores['n_failures']}/{scores['n_samples']}")

        if "recommended_backend" in aggregate:
            print(f"\nRecommended: {aggregate['recommended_backend'].upper()}")

    # Compare label sources if both were run
    if len(label_sources) > 1:
        print(f"\n{'='*80}")
        print("LABEL SOURCE COMPARISON")
        print(f"{'='*80}")

        for backend in BACKENDS:
            hlca_scores = combined_results["label_sources"].get("hlca", {}).get("aggregate_scores", {}).get(backend, {})
            luca_scores = combined_results["label_sources"].get("luca", {}).get("aggregate_scores", {}).get(backend, {})

            if hlca_scores and luca_scores:
                hlca_mean = hlca_scores.get("mean", 0)
                luca_mean = luca_scores.get("mean", 0)
                winner = "HLCA" if hlca_mean >= luca_mean else "LuCA"
                diff = abs(hlca_mean - luca_mean)
                print(f"{backend.upper()}: HLCA={hlca_mean:.3f}, LuCA={luca_mean:.3f} -> {winner} (+{diff:.3f})")

        # Overall recommendation
        hlca_rec = combined_results["label_sources"].get("hlca", {}).get("recommended_backend", "")
        luca_rec = combined_results["label_sources"].get("luca", {}).get("recommended_backend", "")
        print(f"\nHLCA recommends: {hlca_rec.upper() if hlca_rec else 'N/A'}")
        print(f"LuCA recommends: {luca_rec.upper() if luca_rec else 'N/A'}")

        # Save combined results
        combined_output = Path(args.output_dir) / "combined_ablation_results.json"
        with open(combined_output, "w") as f:
            json.dump(combined_results, f, indent=2)
        print(f"\nCombined results saved to: {combined_output}")

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
