#!/usr/bin/env python
"""Train StageBridge model with Hydra configuration.

Usage:
    python scripts/train.py
    python scripts/train.py model=stagebridge_large trainer=fast
    python scripts/train.py experiment.fold_idx=0,1,2,3,4 --multirun
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.contracts import (
    TrainingOutputContract,
    validate_contract,
)
from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig

log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> StageBridge:
    """Build model from config."""
    model_cfg = StageBridgeConfig(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        max_neighbors=cfg.model.max_neighbors,
        num_stages=cfg.model.num_stages,
        time_dim=cfg.model.time_dim,
        stage_dim=cfg.model.stage_dim,
        dropout=cfg.model.dropout,
        use_cross_attn_drift=cfg.model.use_cross_attn_drift,
        use_evolution_branch=cfg.model.use_evolution_branch,
        evolution_dim=cfg.model.evolution_dim,
        evolution_mode=cfg.model.evolution_mode,
        # Gromov-Wasserstein fusion
        use_gw_fusion=cfg.model.get("use_gw_fusion", False),
        gw_output_dim=cfg.model.get("gw_output_dim", 64),
        gw_sinkhorn_iters=cfg.model.get("gw_sinkhorn_iters", 50),
        gw_sinkhorn_reg=cfg.model.get("gw_sinkhorn_reg", 0.1),
        gw_mode=cfg.model.get("gw_mode", "barycentric"),
    )
    return StageBridge(model_cfg)


def build_trainer_config(cfg: DictConfig) -> TrainerConfig:
    """Build trainer config from Hydra config."""
    return TrainerConfig(
        output_dir=Path(cfg.paths.run_dir),
        run_name="train",
        num_epochs=cfg.trainer.num_epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_epochs=cfg.trainer.warmup_epochs,
        min_lr=cfg.trainer.min_lr,
        ot_epsilon=cfg.trainer.ot_epsilon,
        sinkhorn_iters=cfg.trainer.sinkhorn_iters,
        num_ot_pairs=cfg.trainer.num_ot_pairs,
        use_ot=cfg.trainer.use_ot,
        sigma=cfg.trainer.sigma,
        flow_matching_weight=cfg.trainer.flow_matching_weight,
        entropy_weight=cfg.trainer.entropy_weight,
        checkpoint_every=cfg.trainer.checkpoint_every,
        keep_top_k=cfg.trainer.keep_top_k,
        eval_every=cfg.trainer.eval_every,
        mixed_precision=cfg.hardware.mixed_precision,
        gradient_clip=cfg.trainer.gradient_clip,
        accumulation_steps=cfg.trainer.accumulation_steps,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train StageBridge model."""
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate data directory contract
    log.info(f"Validating data contract at {cfg.paths.data_dir}")
    validate_contract(cfg.paths.data_dir)

    # Set seed
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Device setup
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Create data loaders
    log.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=cfg.paths.data_dir,
        fold_idx=cfg.experiment.fold_idx,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.hardware.num_workers,
    )

    if train_loader is None:
        raise RuntimeError("No training data found!")

    log.info(f"Train: {len(train_loader.dataset)} cells")
    if val_loader:
        log.info(f"Val: {len(val_loader.dataset)} cells")

    # Build model
    log.info("Building model...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}")

    # Build trainer
    trainer_config = build_trainer_config(cfg)
    trainer = StageBridgeTrainer(
        model=model,
        config=trainer_config,
        device=device,
    )

    # Save full config
    run_dir = Path(cfg.paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Train
    log.info("Starting training...")
    summary = trainer.train(train_loader, val_loader)

    # Save summary
    summary["completed_at"] = datetime.now().isoformat()
    summary["model_config"] = OmegaConf.to_container(cfg.model)
    summary["trainer_config"] = OmegaConf.to_container(cfg.trainer)

    summary_path = run_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"Training complete. Summary saved to {summary_path}")

    # Validate output contract
    errors = TrainingOutputContract.validate(run_dir)
    if errors:
        log.warning(f"Output contract violations: {errors}")
    else:
        log.info("Output contract validated successfully")

    # Return best val loss for Hydra sweeps
    return summary.get("best_val_loss", float("inf"))


if __name__ == "__main__":
    main()
