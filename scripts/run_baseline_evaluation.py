#!/usr/bin/env python
"""
Run baseline comparison on semi-synthetic benchmark.

Usage:
    python scripts/run_baseline_evaluation.py \
        --benchmark_dir data/outputs/v1_demo/granular_medium \
        --output_dir results/baseline_comparison
"""

import argparse
from pathlib import Path

import torch

from stagebridge.baselines import run_baseline_comparison
from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baselines on semi-synthetic benchmark"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=Path,
        required=True,
        help="Path to exported benchmark directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Running baseline evaluation")
    log.info(f"  Benchmark: {args.benchmark_dir}")
    log.info(f"  Output: {args.output_dir}")
    log.info(f"  Device: {device}")

    # Run evaluation
    results = run_baseline_comparison(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        device=device,
    )

    log.info("\n" + "=" * 60)
    log.info("BASELINE COMPARISON RESULTS")
    log.info("=" * 60)
    print(results.to_string(index=False))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
