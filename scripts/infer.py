#!/usr/bin/env python
"""Run inference with trained StageBridge model.

Usage:
    python scripts/infer.py checkpoint=runs/exp1/checkpoints/best.pt
    python scripts/infer.py checkpoint=runs/exp1/checkpoints/best.pt data.stage_pairs=[[0,1],[1,2]]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from stagebridge.contracts import (
    InferenceOutputContract,
    assert_contract,
    validate_contract,
)
from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig

log = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str) -> tuple[StageBridge, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg = checkpoint.get("config", {}).get("model_config", {})
    config = StageBridgeConfig(**model_cfg)
    model = StageBridge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def run_inference(
    model: StageBridge,
    dataloader,
    stage_pairs: list[tuple[int, int]],
    device: str,
    num_steps: int = 8,
) -> pd.DataFrame:
    """Run inference on dataset.

    Args:
        model: Trained StageBridge model
        dataloader: DataLoader yielding NicheBatch
        stage_pairs: List of (source, target) stage pairs to predict
        device: Device to run on
        num_steps: Euler integration steps

    Returns:
        DataFrame with predictions
    """
    results = []

    for batch in tqdm(dataloader, desc="Inference"):
        batch = batch.to(device)

        # Encode niche context
        niche_output = model.encode_niche(
            receiver=batch.receiver,
            neighbors=batch.neighbors,
            distances=batch.distances,
            neighbor_mask=batch.neighbor_mask,
            token_type_ids=batch.token_type_ids,
        )

        context = niche_output.context
        context_tokens = niche_output.context_tokens
        attention_weights = niche_output.attention_weights

        # Predict for each stage pair
        for src_stage, tgt_stage in stage_pairs:
            stage_pair_id = model.encode_stage_pair_tensor(
                src_stage, tgt_stage, len(batch.receiver), device
            )

            # Integrate velocity field
            predicted = model.integrate_euler(
                x0=batch.receiver,
                context=context,
                stage_pair_id=stage_pair_id,
                num_steps=num_steps,
                context_tokens=context_tokens,
            )

            # Get gate values for interpretability
            # Forward one step to get gate
            t_mid = torch.full((len(batch.receiver),), 0.5, device=device)
            x_mid = 0.5 * batch.receiver + 0.5 * predicted

            if hasattr(model.drift_head, "last_context_gate_mean"):
                _ = model.forward_vector_field(
                    x_t=x_mid,
                    t=t_mid,
                    context=context,
                    stage_pair_id=stage_pair_id,
                    context_tokens=context_tokens,
                )
                gate_values = [model.drift_head.last_context_gate_mean] * len(batch.receiver)
            else:
                gate_values = [0.5] * len(batch.receiver)

            # Collect results
            for i in range(len(batch.receiver)):
                results.append({
                    "cell_id": batch.cell_ids[i],
                    "donor_id": batch.donor_ids[i],
                    "source_stage": src_stage,
                    "target_stage": tgt_stage,
                    "source_embedding": batch.receiver[i].cpu().numpy().tolist(),
                    "predicted_embedding": predicted[i].cpu().numpy().tolist(),
                    "context_gate": gate_values[i] if isinstance(gate_values, list) else float(gate_values),
                    "attention_weights": attention_weights[i].cpu().numpy().tolist() if attention_weights is not None else None,
                })

    return pd.DataFrame(results)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run inference."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate inputs
    checkpoint_path = Path(cfg.get("checkpoint", ""))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    validate_contract(cfg.paths.data_dir)

    # Device setup
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    log.info(f"Loading model from {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)

    # Create data loader (test set)
    log.info("Creating data loader...")
    _, _, test_loader = create_dataloaders(
        data_dir=cfg.paths.data_dir,
        fold_idx=cfg.experiment.fold_idx,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.hardware.num_workers,
    )

    if test_loader is None:
        log.warning("No test set, using validation")
        _, test_loader, _ = create_dataloaders(
            data_dir=cfg.paths.data_dir,
            fold_idx=cfg.experiment.fold_idx,
            batch_size=cfg.trainer.batch_size,
            num_workers=cfg.hardware.num_workers,
        )

    # Determine stage pairs
    stage_pairs = cfg.data.get("stage_pairs", None)
    if stage_pairs is None:
        # Default: all consecutive pairs
        n_stages = cfg.model.num_stages
        stage_pairs = [(i, i + 1) for i in range(n_stages - 1)]

    log.info(f"Stage pairs: {stage_pairs}")

    # Run inference
    log.info("Running inference...")
    predictions_df = run_inference(
        model=model,
        dataloader=test_loader,
        stage_pairs=stage_pairs,
        device=device,
        num_steps=8,
    )

    # Validate output contract
    errors = InferenceOutputContract.validate(predictions_df)
    assert_contract(errors, "inference")

    # Save predictions
    output_dir = Path(cfg.paths.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.parquet"
    predictions_df.to_parquet(output_path)

    log.info(f"Saved {len(predictions_df)} predictions to {output_path}")

    # Save metadata
    metadata = {
        "checkpoint": str(checkpoint_path),
        "fold_idx": cfg.experiment.fold_idx,
        "stage_pairs": stage_pairs,
        "n_predictions": len(predictions_df),
        "created_at": datetime.now().isoformat(),
    }
    with open(output_dir / "inference_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
