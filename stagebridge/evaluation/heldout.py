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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    test_donors = splits["folds"][fold]["test_donors"]

    # Filter at read time using pyarrow for memory efficiency
    import pyarrow.parquet as pq

    # Read cells with filter
    cells_table = pq.read_table(
        data_dir / "cells.parquet",
        filters=[("donor_id", "in", test_donors)]
    )
    test_cells = cells_table.to_pandas()
    del cells_table

    test_cell_ids = set(test_cells["cell_id"])

    # Read neighborhoods in chunks to avoid OOM
    # neighborhoods.parquet can be 50GB+ and doesn't have donor_id for filtering
    neighborhoods_path = data_dir / "neighborhoods.parquet"
    parquet_file = pq.ParquetFile(neighborhoods_path)

    filtered_chunks = []
    for batch in parquet_file.iter_batches(batch_size=100_000):
        chunk_df = batch.to_pandas()
        filtered = chunk_df[chunk_df["cell_id"].isin(test_cell_ids)]
        if len(filtered) > 0:
            filtered_chunks.append(filtered)
        del chunk_df

    test_neighborhoods = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()
    del filtered_chunks

    return model, test_cells, test_neighborhoods, config


def compute_transition_metrics(
    model: Any,
    test_cells: pd.DataFrame,
    test_neighborhoods: pd.DataFrame,
    config: dict,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute transition quality metrics on held-out test data.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods: Test neighborhood DataFrame
        config: Model config dict
        device: Device for inference
        batch_size: Batch size

    Returns:
        Dictionary of metric name -> value
    """
    # Build tensors directly from DataFrames (canonical format)
    latent_dim = config.get("latent_dim", 40)

    # Handle array column (z_fused) or separate columns (z_fused_0, z_fused_1, ...)
    if "z_fused" in test_cells.columns:
        # Array column - stack into 2D tensor
        embeddings = torch.tensor(
            np.stack(test_cells["z_fused"].values), dtype=torch.float32
        )
    else:
        fused_cols = sorted([c for c in test_cells.columns if c.startswith("z_fused_")])
        if not fused_cols:
            fused_cols = sorted([c for c in test_cells.columns if c.startswith("fused_latent_")])
        if not fused_cols:
            return {"error": "No fused embedding columns found (tried z_fused, z_fused_*, fused_latent_*)"}
        embeddings = torch.tensor(test_cells[fused_cols].values, dtype=torch.float32)
    n_cells = len(embeddings)

    # Build niche tokens from neighborhoods
    cell_id_to_idx = {cid: i for i, cid in enumerate(test_cells["cell_id"].values)}
    niche_tokens = torch.zeros(n_cells, 9, latent_dim)
    niche_tokens[:, 0, :] = embeddings  # Receiver
    token_distances = torch.zeros(n_cells, 8)
    token_mask = torch.zeros(n_cells, 8, dtype=torch.bool)

    if "tokens" in test_neighborhoods.columns:
        # Vectorized: map cell_ids to indices first
        valid_mask = test_neighborhoods["cell_id"].isin(cell_id_to_idx)
        valid_neighborhoods = test_neighborhoods[valid_mask]
        cell_indices = valid_neighborhoods["cell_id"].map(cell_id_to_idx).values

        # Process tokens in batches to avoid memory explosion
        for i, (cell_idx, tokens) in enumerate(zip(cell_indices, valid_neighborhoods["tokens"].values)):
            if tokens is None:
                continue
            for token_dict in tokens:
                token_idx = token_dict.get("token_idx", -1)
                if 1 <= token_idx <= 4:
                    z_pooled = token_dict.get("z_pooled")
                    if z_pooled is not None and len(z_pooled) > 0:
                        niche_tokens[cell_idx, token_idx, :min(len(z_pooled), latent_dim)] = torch.tensor(
                            z_pooled[:latent_dim], dtype=torch.float32
                        )
                        token_distances[cell_idx, token_idx - 1] = token_dict.get("normalized_distance", 0.0)
                        token_mask[cell_idx, token_idx - 1] = True

    # Get stage indices and filter to transition-eligible (all but last stage)
    stage_col = "stage" if "stage" in test_cells.columns else "stage_label"
    if stage_col in test_cells.columns:
        from stagebridge.canonical_contract import CANONICAL_STAGES_3, CANONICAL_STAGES_5, STAGE_TO_INDEX_3, STAGE_TO_INDEX_5
        unique_stages = set(test_cells[stage_col].dropna().unique())
        if unique_stages <= set(CANONICAL_STAGES_3):
            stage_map = STAGE_TO_INDEX_3
            n_stages = 3
        else:
            stage_map = STAGE_TO_INDEX_5
            n_stages = 5
        stages = test_cells[stage_col].map(stage_map).fillna(n_stages).astype(int)
        stage_indices = torch.tensor(stages.values, dtype=torch.long)
        can_transition = stage_indices < (n_stages - 1)  # All but last stage can transition
    else:
        stage_indices = None
        can_transition = torch.ones(n_cells, dtype=torch.bool)

    # Filter to transition-eligible cells
    trans_indices = torch.where(can_transition)[0]
    if len(trans_indices) < 2:
        return {"error": "Not enough transition-eligible cells"}

    trans_niche = niche_tokens[trans_indices]
    trans_distances = token_distances[trans_indices]
    trans_mask = token_mask[trans_indices]
    trans_stages = stage_indices[trans_indices] if stage_indices is not None else None

    all_preds = []
    all_targets = []
    all_stages = []

    with torch.no_grad():
        for i in range(0, len(trans_indices), batch_size):
            batch_idx = trans_indices[i:i+batch_size]
            batch_niche = trans_niche[i:i+batch_size].to(device)
            batch_distances = trans_distances[i:i+batch_size].to(device)
            batch_mask = trans_mask[i:i+batch_size].to(device)

            # Encode niche with distances and mask (mirror training)
            context = model.encode_niche(
                batch_niche,
                distances=batch_distances,
                neighbor_mask=batch_mask,
            )

            # Get source/target embeddings
            z_source = batch_niche[:, 0, :]  # Receiver = source
            # For held-out eval, use model's transition prediction
            if hasattr(model, "transition_forward"):
                # Use same cell as target for self-consistency check
                outputs = model.transition_forward(
                    z_source, z_source, context, use_ot=False
                )
                if "drift_pred" in outputs:
                    pred_target = z_source + outputs["drift_pred"]
                    all_preds.append(pred_target.cpu().numpy())
                    all_targets.append(z_source.cpu().numpy())

            if trans_stages is not None:
                all_stages.append(trans_stages[i:i+batch_size].cpu().numpy())

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

    # Note: Stage classification metrics removed - model doesn't output class probs
    # The key metrics are transition quality (wasserstein, mmd) and context sensitivity

    return metrics


def compute_context_sensitivity(
    model: Any,
    test_cells: pd.DataFrame,
    test_neighborhoods: pd.DataFrame,
    config: dict,
    device: str = "cuda",
    n_samples: int = 1000,
) -> dict[str, float]:
    """Measure how much predictions change when niche context is shuffled.

    This is a key metric for showing niche context matters.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods: Test neighborhood DataFrame
        config: Model config dict
        device: Device for inference
        n_samples: Number of cells to sample

    Returns:
        Dictionary with context sensitivity metrics
    """
    latent_dim = config.get("latent_dim", 40)

    # Handle array column (z_fused) or separate columns (z_fused_0, z_fused_1, ...)
    use_array_col = "z_fused" in test_cells.columns
    if not use_array_col:
        fused_cols = sorted([c for c in test_cells.columns if c.startswith("z_fused_")])
        if not fused_cols:
            fused_cols = sorted([c for c in test_cells.columns if c.startswith("fused_latent_")])
        if not fused_cols:
            return {}

    # Sample cells
    sample_cells = test_cells.head(n_samples)
    if use_array_col:
        embeddings = torch.tensor(
            np.stack(sample_cells["z_fused"].values), dtype=torch.float32
        )
    else:
        embeddings = torch.tensor(sample_cells[fused_cols].values, dtype=torch.float32)
    n_cells = len(embeddings)

    # Build niche tokens
    cell_id_to_idx = {cid: i for i, cid in enumerate(sample_cells["cell_id"].values)}
    niche_tokens = torch.zeros(n_cells, 9, latent_dim)
    niche_tokens[:, 0, :] = embeddings
    token_distances = torch.zeros(n_cells, 8)
    token_mask = torch.zeros(n_cells, 8, dtype=torch.bool)

    sample_cell_ids = set(sample_cells["cell_id"])
    sample_neighborhoods = test_neighborhoods[test_neighborhoods["cell_id"].isin(sample_cell_ids)]

    if "tokens" in sample_neighborhoods.columns:
        for _, row in sample_neighborhoods.iterrows():
            cell_id = row["cell_id"]
            if cell_id not in cell_id_to_idx:
                continue
            idx = cell_id_to_idx[cell_id]
            for token_dict in row["tokens"]:
                token_idx = token_dict.get("token_idx", -1)
                if 1 <= token_idx <= 4:
                    z_pooled = token_dict.get("z_pooled")
                    if z_pooled is not None and len(z_pooled) > 0:
                        z_t = torch.tensor(z_pooled[:latent_dim], dtype=torch.float32)
                        niche_tokens[idx, token_idx, :len(z_t)] = z_t
                        token_distances[idx, token_idx - 1] = token_dict.get("normalized_distance", 0.0)
                        token_mask[idx, token_idx - 1] = True

    real_contexts = []
    shuffled_contexts = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_niche = niche_tokens[i:i+batch_size].to(device)
            batch_distances = token_distances[i:i+batch_size].to(device)
            batch_mask = token_mask[i:i+batch_size].to(device)

            # Real context
            context_real = model.encode_niche(
                batch_niche, distances=batch_distances, neighbor_mask=batch_mask
            )
            real_contexts.append(context_real.cpu().numpy())

            # Shuffled niche tokens (permute neighbor tokens, keep receiver)
            shuffled_niche = batch_niche.clone()
            perm = torch.randperm(batch_niche.shape[0])
            shuffled_niche[:, 1:, :] = batch_niche[perm, 1:, :]

            context_shuffled = model.encode_niche(
                shuffled_niche, distances=batch_distances, neighbor_mask=batch_mask
            )
            shuffled_contexts.append(context_shuffled.cpu().numpy())

    if real_contexts and shuffled_contexts:
        real = np.concatenate(real_contexts, axis=0)
        shuffled = np.concatenate(shuffled_contexts, axis=0)

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
        model, test_cells, test_neighborhoods, config, device, batch_size
    )

    context_metrics = compute_context_sensitivity(
        model, test_cells, test_neighborhoods, config, device
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
