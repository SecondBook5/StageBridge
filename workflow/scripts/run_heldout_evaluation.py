#!/usr/bin/env python3
"""Run held-out test evaluation - Snakemake wrapper.

Core logic in stagebridge.evaluation.heldout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stagebridge.evaluation.heldout import run_heldout_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run held-out evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_heldout_evaluation(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        fold=args.fold,
        device=args.device,
        batch_size=args.batch_size,
    )

    results_path = output_dir / "heldout_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")

    if "transition_metrics" in results:
        print("\nKey metrics:")
        for key, value in results["transition_metrics"].items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
