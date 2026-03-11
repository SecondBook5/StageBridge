#!/usr/bin/env python
"""Post-hoc permutation test for rescue ablation results.

Reads saved test_predictions.parquet files from a completed ablation run,
permutes the true labels N times, recomputes macro-F1 for each permutation,
and reports a p-value + null distribution summary per study.

Usage:
    python scripts/run_permutation_test.py \
        --results-dir outputs/scratch/rescue_ablation_20250608/eamist_benchmark \
        --n-permutations 1000 \
        --seed 42 \
        --output-json reports/tables/permutation_test_results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def run_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute observed macro-F1 and null distribution via label permutation."""
    rng = np.random.RandomState(seed)
    observed_f1 = float(f1_score(y_true, y_pred, average="macro"))

    null_f1s = []
    for _ in range(n_permutations):
        perm_true = rng.permutation(y_true)
        null_f1 = float(f1_score(perm_true, y_pred, average="macro"))
        null_f1s.append(null_f1)

    null_f1s = np.array(null_f1s)
    p_value = float(np.mean(null_f1s >= observed_f1))

    return {
        "observed_f1": observed_f1,
        "null_mean": float(null_f1s.mean()),
        "null_std": float(null_f1s.std()),
        "null_p95": float(np.percentile(null_f1s, 95)),
        "null_p99": float(np.percentile(null_f1s, 99)),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "significant_p05": bool(p_value < 0.05),
        "significant_p01": bool(p_value < 0.01),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc permutation test")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/scratch/rescue_ablation_20250608/eamist_benchmark",
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/tables/permutation_test_results.json",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    # Discover all test prediction files
    pred_files = sorted(results_dir.glob("*/*/fold_*/seed_*/test_predictions.parquet"))
    if not pred_files:
        print(f"No prediction files found in {results_dir}")
        return

    # Aggregate by (mode, family, fold) — average across seeds first
    study_preds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for pf in pred_files:
        parts = pf.parts
        mode = parts[-5]
        family = parts[-4]
        fold = parts[-3]
        study_key = f"{family}_{mode}_{fold}"
        df = pd.read_parquet(pf)
        y_true = df["stage_index"].values
        y_pred = df["pred_stage_index"].values
        study_preds[study_key].append((y_true, y_pred))

    # Also aggregate by (mode, family) across all folds+seeds
    model_preds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for pf in pred_files:
        parts = pf.parts
        mode = parts[-5]
        family = parts[-4]
        model_key = f"{family}_{mode}"
        df = pd.read_parquet(pf)
        model_preds[model_key].append((df["stage_index"].values, df["pred_stage_index"].values))

    all_results = {}

    # Per-model aggregated test (all folds+seeds concatenated)
    print(f"Running permutation tests ({args.n_permutations} permutations)...")
    print(f"{'Study':<40} {'Obs F1':>8} {'Null mean':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 75)

    for key in sorted(model_preds.keys()):
        pairs = model_preds[key]
        y_true_all = np.concatenate([p[0] for p in pairs])
        y_pred_all = np.concatenate([p[1] for p in pairs])
        result = run_permutation_test(
            y_true_all, y_pred_all,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        result["n_samples"] = len(y_true_all)
        result["n_runs"] = len(pairs)
        all_results[key] = result
        sig = "***" if result["significant_p01"] else ("*" if result["significant_p05"] else "ns")
        print(f"{key:<40} {result['observed_f1']:8.3f} {result['null_mean']:10.3f} {result['p_value']:10.4f} {sig:>5}")

    # Write results
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    n_sig = sum(1 for r in all_results.values() if r["significant_p05"])
    print(f"\n{n_sig}/{len(all_results)} studies significant at p<0.05")


if __name__ == "__main__":
    main()
