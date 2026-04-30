"""CLI entrypoint for held-out test set evaluation.

Usage:
    python -m stagebridge.evaluation.evaluate \
        --predictions /path/to/predictions.parquet \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.evaluation.metrics import (
    compute_wasserstein,
    compute_mmd,
    compute_displacement,
    compute_stage_accuracy,
)
from stagebridge.loaders import create_dataloaders
from stagebridge.contracts import STAGE_TO_IDX


def evaluate_held_out(
    predictions_path: Path,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
) -> dict:
    """Evaluate model predictions on held-out test set.

    Args:
        predictions_path: Path to predictions.parquet from inference
        data_dir: Data directory with neighborhoods.parquet
        output_dir: Output directory for evaluation results
        fold_idx: Fold index (to get correct test set)

    Returns:
        Dict with all evaluation metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_path}")
    pred_df = pd.read_parquet(predictions_path)

    print(f"Loading test data from fold {fold_idx}")
    _, _, test_loader = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=1000)

    if test_loader is None:
        raise RuntimeError(f"No test data found for fold {fold_idx}")

    # Collect ground truth from test loader
    gt_receivers = []
    gt_stages = []
    gt_cell_ids = []

    for batch in test_loader:
        gt_receivers.append(batch.receiver.numpy())
        gt_stages.append(batch.stage_idx.numpy())
        gt_cell_ids.extend(batch.cell_ids)

    gt_receivers = np.concatenate(gt_receivers, axis=0)
    gt_stages = np.concatenate(gt_stages, axis=0)

    # Extract predictions - match by cell_id
    pred_df = pred_df.set_index("cell_id")
    pred_df = pred_df.loc[gt_cell_ids].reset_index()

    predicted = np.array(pred_df["predicted_z"].tolist())
    context = np.array(pred_df["context_z"].tolist())

    print(f"Evaluating {len(predicted)} predictions")
    print(f"  Predicted shape: {predicted.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Ground truth shape: {gt_receivers.shape}")

    metrics = {}

    # 1. Reconstruction quality (SSL objective) - compare predicted_z vs receiver
    print("Computing reconstruction metrics...")
    disp = compute_displacement(predicted, gt_receivers)
    metrics["reconstruction"] = disp

    # 2. Distribution metrics - compare in same dimensional space
    # Use predicted (reconstruction) vs ground truth, not context
    print("Computing distribution metrics...")
    metrics["wasserstein_reconstruction"] = compute_wasserstein(predicted, gt_receivers)
    metrics["mmd_reconstruction"] = compute_mmd(predicted, gt_receivers)

    # 3. Stage-stratified metrics
    print("Computing per-stage metrics...")
    stage_metrics = {}
    idx_to_stage = {v: k for k, v in STAGE_TO_IDX.items()}

    for stage_idx in np.unique(gt_stages):
        mask = gt_stages == stage_idx
        if mask.sum() < 10:
            continue

        stage_name = idx_to_stage.get(stage_idx, f"stage_{stage_idx}")
        stage_pred = predicted[mask]
        stage_gt = gt_receivers[mask]

        stage_metrics[stage_name] = {
            "n_samples": int(mask.sum()),
            "mean_displacement": float(np.linalg.norm(stage_pred - stage_gt, axis=1).mean()),
            "wasserstein": compute_wasserstein(stage_pred, stage_gt),
        }

    metrics["per_stage"] = stage_metrics

    # 4. Stage classification accuracy (k-NN in reconstruction space)
    print("Computing stage classification accuracy...")
    stage_acc = compute_stage_accuracy(
        predicted=predicted,
        reference_embeddings=gt_receivers,
        reference_stages=gt_stages,
        k=min(5, len(gt_receivers) - 1),  # k can't exceed n_samples - 1
    )
    metrics["stage_classification"] = stage_acc

    # 5. Summary statistics
    metrics["summary"] = {
        "n_test_samples": len(predicted),
        "n_stages": len(np.unique(gt_stages)),
        "overall_mean_displacement": disp["mean_displacement"],
        "overall_wasserstein": metrics["wasserstein_reconstruction"],
        "stage_accuracy": stage_acc["stage_accuracy"],
    }

    # Save results
    metrics["metadata"] = {
        "predictions_path": str(predictions_path),
        "data_dir": str(data_dir),
        "fold_idx": fold_idx,
        "evaluated_at": datetime.now().isoformat(),
    }

    output_path = output_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nEvaluation complete:")
    print(f"  Mean displacement: {disp['mean_displacement']:.4f}")
    print(f"  Wasserstein: {metrics['wasserstein_reconstruction']:.4f}")
    print(f"  Stage accuracy: {stage_acc['stage_accuracy']:.4f}")
    print(f"\nSaved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out test set")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Path to predictions.parquet from inference")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Data directory with neighborhoods.parquet")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for evaluation results")
    parser.add_argument("--fold-idx", type=int, default=0,
                        help="Fold index")
    args = parser.parse_args()

    evaluate_held_out(
        predictions_path=args.predictions,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
    )


if __name__ == "__main__":
    main()
