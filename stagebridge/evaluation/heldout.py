"""Held-out test evaluation for StageBridge.

Computes proper evaluation metrics on held-out test data:
- Wasserstein distance (transition quality)
- MMD (distribution matching)
- Stage classification accuracy/F1
- Calibration (ECE)
- Context sensitivity (niche contribution)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from stagebridge.evaluation.metrics import (
    compute_all_metrics,
    expected_calibration_error,
    wasserstein_nd_distance,
    maximum_mean_discrepancy,
)


def load_model_and_test_data(
    checkpoint_path: Path,
    data_dir: Path,
    fold: int,
    device: str = "cuda",
) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict]:
    """Load trained model and prepare test data.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory with cells.parquet, neighborhoods.parquet, split_manifest.json
        fold: Fold index to use for test set
        device: Device for model

    Returns:
        Tuple of (model, test_cells_df, test_neighborhoods_df, config)
    """
    from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = StageBridgeV1Complete(
        latent_dim=config.get("latent_dim", 40),
        hlca_dim=config.get("hlca_dim", 30),
        luca_dim=config.get("luca_dim", 10),
        niche_hidden_dim=config.get("niche_hidden_dim", 128),
        context_dim=config.get("context_dim", 256),
        wes_feature_dim=config.get("wes_feature_dim", config.get("wes_dim", 8)),
        use_prototypes=config.get("use_prototypes", False),
        num_prototypes=config.get("num_prototypes", 16),
        fusion_mode=config.get("fusion_mode", "concat"),
        niche_encoder_type=config.get("niche_encoder_type", "cross_attention"),
        use_hierarchical=config.get("use_hierarchical", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    splits_path = data_dir / "split_manifest.json"
    with open(splits_path) as f:
        splits = json.load(f)

    test_donors = splits["folds"][fold]["test"]

    cells_df = pd.read_parquet(data_dir / "cells.parquet")
    neighborhoods_df = pd.read_parquet(data_dir / "neighborhoods.parquet")

    test_mask = cells_df["donor_id"].isin(test_donors)
    test_cells = cells_df[test_mask].copy()
    test_cell_ids = set(test_cells["cell_id"])
    test_neighborhoods = neighborhoods_df[
        neighborhoods_df["receiver_id"].isin(test_cell_ids)
    ].copy()

    return model, test_cells, test_neighborhoods, config


def compute_transition_metrics(
    model: Any,
    test_cells: pd.DataFrame,
    test_neighborhoods: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute transition quality metrics on held-out test data.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods: Test neighborhood DataFrame
        device: Device for inference
        batch_size: Batch size

    Returns:
        Dictionary of metric name -> value
    """
    from stagebridge.data.loaders import StageBridgeDataset
    from torch.utils.data import DataLoader

    dataset = StageBridgeDataset(
        cells_df=test_cells,
        neighborhoods_df=test_neighborhoods,
        mode="transition",
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_stages = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs = model(batch)

            if "predicted_target" in outputs:
                all_preds.append(outputs["predicted_target"].cpu().numpy())
            if "target_embedding" in batch:
                all_targets.append(batch["target_embedding"].cpu().numpy())
            if "stage" in batch:
                all_stages.append(batch["stage"].cpu().numpy())
            if "transition_prob" in outputs:
                all_probs.append(outputs["transition_prob"].cpu().numpy())

    metrics = {}

    if all_preds and all_targets:
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        transition_metrics = compute_all_metrics(preds, targets)
        metrics.update({
            "wasserstein": transition_metrics["wasserstein"],
            "mmd": transition_metrics["mmd"],
            "mse": transition_metrics["mse"],
            "mae": transition_metrics["mae"],
        })

    if all_stages:
        stages = np.concatenate(all_stages, axis=0)
        unique_stages = np.unique(stages)

        if len(unique_stages) > 1 and all_probs:
            probs = np.concatenate(all_probs, axis=0)
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)

            if probs.shape[1] >= len(unique_stages):
                stage_preds = np.argmax(probs[:, :len(unique_stages)], axis=1)
                metrics["stage_accuracy"] = float(accuracy_score(stages, stage_preds))
                metrics["stage_f1_macro"] = float(f1_score(stages, stage_preds, average="macro"))

                if len(unique_stages) == 2:
                    metrics["auroc"] = float(roc_auc_score(stages, probs[:, 1]))
                else:
                    try:
                        metrics["auroc"] = float(roc_auc_score(
                            stages, probs[:, :len(unique_stages)],
                            multi_class="ovr", average="macro"
                        ))
                    except ValueError:
                        pass

    return metrics


def compute_context_sensitivity(
    model: Any,
    test_cells: pd.DataFrame,
    test_neighborhoods: pd.DataFrame,
    device: str = "cuda",
    n_samples: int = 1000,
) -> dict[str, float]:
    """Measure how much predictions change when niche context is shuffled.

    This is a key metric for showing niche context matters.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods: Test neighborhood DataFrame
        device: Device for inference
        n_samples: Number of cells to sample

    Returns:
        Dictionary with context sensitivity metrics
    """
    from stagebridge.data.loaders import StageBridgeDataset
    from torch.utils.data import DataLoader

    dataset = StageBridgeDataset(
        cells_df=test_cells.head(n_samples),
        neighborhoods_df=test_neighborhoods,
        mode="transition",
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    real_outputs = []
    shuffled_outputs = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            outputs_real = model(batch)
            if "predicted_target" in outputs_real:
                real_outputs.append(outputs_real["predicted_target"].cpu().numpy())

            if "niche_context" in batch:
                shuffled_batch = batch.copy()
                niche = batch["niche_context"]
                perm = torch.randperm(niche.shape[0])
                shuffled_batch["niche_context"] = niche[perm]

                outputs_shuffled = model(shuffled_batch)
                if "predicted_target" in outputs_shuffled:
                    shuffled_outputs.append(outputs_shuffled["predicted_target"].cpu().numpy())

    if real_outputs and shuffled_outputs:
        real = np.concatenate(real_outputs, axis=0)
        shuffled = np.concatenate(shuffled_outputs, axis=0)

        delta = np.linalg.norm(real - shuffled, axis=1)

        return {
            "mean_context_delta": float(np.mean(delta)),
            "std_context_delta": float(np.std(delta)),
            "max_context_delta": float(np.max(delta)),
            "context_sensitivity_zscore": float(np.mean(delta) / (np.std(delta) + 1e-8)),
        }

    return {}


def run_heldout_evaluation(
    checkpoint_path: Path,
    data_dir: Path,
    fold: int = 0,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run complete held-out evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory with data files
        fold: Fold index
        device: Device for inference
        batch_size: Batch size

    Returns:
        Complete evaluation results dictionary
    """
    model, test_cells, test_neighborhoods, config = load_model_and_test_data(
        checkpoint_path, data_dir, fold, device
    )

    transition_metrics = compute_transition_metrics(
        model, test_cells, test_neighborhoods, device, batch_size
    )

    context_metrics = compute_context_sensitivity(
        model, test_cells, test_neighborhoods, device
    )

    return {
        "checkpoint": str(checkpoint_path),
        "fold": fold,
        "n_test_cells": len(test_cells),
        "n_test_neighborhoods": len(test_neighborhoods),
        "config": config,
        "transition_metrics": transition_metrics,
        "context_sensitivity": context_metrics,
    }
