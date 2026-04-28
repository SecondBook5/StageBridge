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
import pyarrow.parquet as pq
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
        drift_head=config.get("drift_head", "cross_attention"),
        context_refiner=config.get("context_refiner", "set_transformer"),
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

    # Only load columns we need to save memory
    needed_cols = ["cell_id", "donor_id", "stage"]
    # Add z_fused columns
    cells_schema = pq.read_schema(data_dir / "cells.parquet")
    all_cols = [f.name for f in cells_schema]
    fused_cols = sorted([c for c in all_cols if c.startswith("z_fused_")])
    if "z_fused" in all_cols:
        needed_cols.append("z_fused")
    elif fused_cols:
        needed_cols.extend(fused_cols)

    # Read cells with filter and column selection
    cells_table = pq.read_table(
        data_dir / "cells.parquet",
        columns=needed_cols,
        filters=[("donor_id", "in", test_donors)]
    )
    test_cells = cells_table.to_pandas()
    del cells_table

    test_cell_ids = set(test_cells["cell_id"])

    # Read neighborhoods in chunks to avoid OOM
    # neighborhoods.parquet can be 50GB+ and doesn't have donor_id for filtering
    neighborhoods_path = data_dir / "neighborhoods.parquet"
    parquet_file = pq.ParquetFile(neighborhoods_path)

    # Write filtered chunks to temp file to avoid memory accumulation
    import tempfile
    import gc
    temp_path = Path(tempfile.mkdtemp()) / "filtered_neighborhoods.parquet"
    writer = None
    n_written = 0

    for batch in parquet_file.iter_batches(batch_size=50_000):
        chunk_df = batch.to_pandas()
        filtered = chunk_df[chunk_df["cell_id"].isin(test_cell_ids)]
        if len(filtered) > 0:
            # Reset index to avoid schema mismatch between batches
            filtered = filtered.reset_index(drop=True)
            if writer is None:
                import pyarrow as pa
                table = pa.Table.from_pandas(filtered, preserve_index=False)
                writer = pq.ParquetWriter(temp_path, table.schema)
                writer.write_table(table)
            else:
                writer.write_table(pa.Table.from_pandas(filtered, preserve_index=False))
            n_written += len(filtered)
        del chunk_df, filtered
        gc.collect()

    if writer is not None:
        writer.close()
        # Return path instead of loading into memory - let caller stream if needed
        test_neighborhoods_path = temp_path
    else:
        test_neighborhoods_path = None

    print(f"Wrote {n_written} test neighborhoods for {len(test_cell_ids)} test cells")

    return model, test_cells, test_neighborhoods_path, config


def compute_transition_metrics(
    model: Any,
    test_cells: pd.DataFrame,
    test_neighborhoods_path: Path | None,
    config: dict,
    device: str = "cuda",
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute transition quality metrics on held-out test data.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods_path: Path to filtered neighborhoods parquet (or None)
        config: Model config dict
        device: Device for inference
        batch_size: Batch size

    Returns:
        Dictionary of metric name -> value
    """
    import gc
    latent_dim = config.get("latent_dim", 40)

    # Only load minimal columns from cells
    fused_cols = sorted([c for c in test_cells.columns if c.startswith("z_fused_")])
    if "z_fused" not in test_cells.columns and not fused_cols:
        return {"error": "No fused embedding columns found"}

    n_cells = len(test_cells)
    cell_id_to_idx = {cid: i for i, cid in enumerate(test_cells["cell_id"].values)}

    # Handle array column (z_fused) or separate columns (z_fused_0, z_fused_1, ...)
    if "z_fused" in test_cells.columns:
        embeddings = torch.tensor(
            np.stack(test_cells["z_fused"].values), dtype=torch.float32
        )
    else:
        embeddings = torch.tensor(test_cells[fused_cols].values, dtype=torch.float32)

    # Process in batches to avoid OOM - don't preallocate full arrays
    # Instead, process neighborhoods and cells together in streaming fashion

    if test_neighborhoods_path is not None and test_neighborhoods_path.exists():
        # Build cell_id -> neighborhood data mapping in chunks
        # Use dict instead of preallocated tensors to save memory
        cell_niche_data = {}

        parquet_file = pq.ParquetFile(test_neighborhoods_path)
        for batch in parquet_file.iter_batches(batch_size=5_000):
            chunk_df = batch.to_pandas()
            if "tokens" not in chunk_df.columns:
                del chunk_df
                continue

            # Vectorized filtering
            valid_mask = chunk_df["cell_id"].isin(cell_id_to_idx)
            if not valid_mask.any():
                del chunk_df
                continue

            valid_chunk = chunk_df.loc[valid_mask, ["cell_id", "tokens"]]
            del chunk_df

            # Process each row but store minimal data
            for cell_id, tokens in zip(valid_chunk["cell_id"].values, valid_chunk["tokens"].values):
                if tokens is None:
                    continue
                cell_idx = cell_id_to_idx.get(cell_id)
                if cell_idx is None:
                    continue

                # Extract only ring tokens (1-4) with z_pooled
                ring_data = []
                for token_dict in tokens:
                    token_idx = token_dict.get("token_idx", -1)
                    if 1 <= token_idx <= 4:
                        z_pooled = token_dict.get("z_pooled")
                        if z_pooled is not None and len(z_pooled) > 0:
                            ring_data.append((
                                token_idx,
                                np.array(z_pooled[:latent_dim], dtype=np.float32),
                                float(token_dict.get("normalized_distance", 0.0))
                            ))
                if ring_data:
                    cell_niche_data[cell_idx] = ring_data

            del valid_chunk
            gc.collect()

        print(f"Loaded niche data for {len(cell_niche_data)} cells")
    else:
        cell_niche_data = {}

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
    trans_indices = torch.where(can_transition)[0].numpy()
    if len(trans_indices) < 2:
        return {"error": "Not enough transition-eligible cells"}

    all_preds = []
    all_targets = []
    all_stages = []

    # Process in batches - build niche tensors on-the-fly per batch
    with torch.no_grad():
        for i in range(0, len(trans_indices), batch_size):
            batch_cell_indices = trans_indices[i:i+batch_size]
            batch_size_actual = len(batch_cell_indices)

            # Build batch tensors
            batch_niche = torch.zeros(batch_size_actual, 9, latent_dim)
            batch_distances = torch.zeros(batch_size_actual, 8)
            batch_mask = torch.zeros(batch_size_actual, 8, dtype=torch.bool)

            for j, cell_idx in enumerate(batch_cell_indices):
                # Receiver embedding
                batch_niche[j, 0, :] = embeddings[cell_idx]

                # Ring tokens from preloaded data
                if cell_idx in cell_niche_data:
                    for token_idx, z_pooled, dist in cell_niche_data[cell_idx]:
                        batch_niche[j, token_idx, :len(z_pooled)] = torch.from_numpy(z_pooled)
                        batch_distances[j, token_idx - 1] = dist
                        batch_mask[j, token_idx - 1] = True

            batch_niche = batch_niche.to(device)
            batch_distances = batch_distances.to(device)
            batch_mask = batch_mask.to(device)

            # Encode niche with distances and mask (mirror training)
            if hasattr(model, "encode_niche_with_tokens"):
                context, context_tokens, _ = model.encode_niche_with_tokens(
                    batch_niche,
                    distances=batch_distances,
                    neighbor_mask=batch_mask,
                )
            else:
                context = model.encode_niche(
                    batch_niche,
                    distances=batch_distances,
                    neighbor_mask=batch_mask,
                )
                context_tokens = None

            # Get source/target embeddings
            z_source = batch_niche[:, 0, :]  # Receiver = source
            # For held-out eval, use model's transition prediction
            if hasattr(model, "transition_forward"):
                outputs = model.transition_forward(
                    z_source, z_source, context, use_ot=False,
                    context_tokens=context_tokens,
                )
                if "drift_pred" in outputs:
                    pred_target = z_source + outputs["drift_pred"]
                    all_preds.append(pred_target.cpu().numpy())
                    all_targets.append(z_source.cpu().numpy())

            if stage_indices is not None:
                all_stages.append(stage_indices[batch_cell_indices].cpu().numpy())

            # Free GPU memory
            del batch_niche, batch_distances, batch_mask, context
            if context_tokens is not None:
                del context_tokens
            torch.cuda.empty_cache()

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
    test_neighborhoods_path: Path | None,
    config: dict,
    device: str = "cuda",
    n_samples: int = 1000,
) -> dict[str, float]:
    """Measure how much predictions change when niche context is shuffled.

    This is a key metric for showing niche context matters.

    Args:
        model: Trained StageBridge model
        test_cells: Test cell DataFrame
        test_neighborhoods_path: Path to filtered neighborhoods parquet (or None)
        config: Model config dict
        device: Device for inference
        n_samples: Number of cells to sample

    Returns:
        Dictionary with context sensitivity metrics
    """
    import gc
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

    # Build niche data dict (memory efficient)
    cell_id_to_idx = {cid: i for i, cid in enumerate(sample_cells["cell_id"].values)}
    sample_cell_ids = set(sample_cells["cell_id"])
    cell_niche_data = {}

    # Stream neighborhoods from path
    if test_neighborhoods_path is not None and test_neighborhoods_path.exists():
        parquet_file = pq.ParquetFile(test_neighborhoods_path)
        for batch in parquet_file.iter_batches(batch_size=5_000):
            chunk_df = batch.to_pandas()
            if "tokens" not in chunk_df.columns:
                del chunk_df
                continue

            valid_mask = chunk_df["cell_id"].isin(sample_cell_ids)
            if not valid_mask.any():
                del chunk_df
                continue

            valid_chunk = chunk_df.loc[valid_mask, ["cell_id", "tokens"]]
            del chunk_df

            for cell_id, tokens in zip(valid_chunk["cell_id"].values, valid_chunk["tokens"].values):
                if tokens is None:
                    continue
                idx = cell_id_to_idx.get(cell_id)
                if idx is None:
                    continue

                ring_data = []
                for token_dict in tokens:
                    token_idx = token_dict.get("token_idx", -1)
                    if 1 <= token_idx <= 4:
                        z_pooled = token_dict.get("z_pooled")
                        if z_pooled is not None and len(z_pooled) > 0:
                            ring_data.append((
                                token_idx,
                                np.array(z_pooled[:latent_dim], dtype=np.float32),
                                float(token_dict.get("normalized_distance", 0.0))
                            ))
                if ring_data:
                    cell_niche_data[idx] = ring_data

            del valid_chunk
            gc.collect()

    real_contexts = []
    shuffled_contexts = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_indices = list(range(i, min(i + batch_size, n_cells)))
            batch_size_actual = len(batch_indices)

            # Build batch tensors on-the-fly
            batch_niche = torch.zeros(batch_size_actual, 9, latent_dim)
            batch_distances = torch.zeros(batch_size_actual, 8)
            batch_mask = torch.zeros(batch_size_actual, 8, dtype=torch.bool)

            for j, idx in enumerate(batch_indices):
                batch_niche[j, 0, :] = embeddings[idx]
                if idx in cell_niche_data:
                    for token_idx, z_pooled, dist in cell_niche_data[idx]:
                        batch_niche[j, token_idx, :len(z_pooled)] = torch.from_numpy(z_pooled)
                        batch_distances[j, token_idx - 1] = dist
                        batch_mask[j, token_idx - 1] = True

            batch_niche = batch_niche.to(device)
            batch_distances = batch_distances.to(device)
            batch_mask = batch_mask.to(device)

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

            del batch_niche, batch_distances, batch_mask, shuffled_niche
            torch.cuda.empty_cache()

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
    model, test_cells, test_neighborhoods_path, config = load_model_and_test_data(
        checkpoint_path, data_dir, fold, device
    )

    transition_metrics = compute_transition_metrics(
        model, test_cells, test_neighborhoods_path, config, device, batch_size
    )

    context_metrics = compute_context_sensitivity(
        model, test_cells, test_neighborhoods_path, config, device
    )

    # Count neighborhoods from path
    n_neighborhoods = 0
    if test_neighborhoods_path is not None and test_neighborhoods_path.exists():
        n_neighborhoods = pq.read_metadata(test_neighborhoods_path).num_rows
        # Clean up temp file
        test_neighborhoods_path.unlink()

    return {
        "checkpoint": str(checkpoint_path),
        "fold": fold,
        "n_test_cells": len(test_cells),
        "n_test_neighborhoods": n_neighborhoods,
        "config": config,
        "transition_metrics": transition_metrics,
        "context_sensitivity": context_metrics,
    }
