#!/usr/bin/env python
"""Train baseline models for comparison.

Usage:
    python scripts/train_baseline.py baseline=pooling_mlp
    python scripts/train_baseline.py baseline=deepsets,set_transformer,graphsage --multirun
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.baselines import get_baseline
from stagebridge.contracts import BaselineOutputContract, assert_contract, validate_contract
from stagebridge.loaders import create_dataloaders
from stagebridge.training import TrainerConfig

log = logging.getLogger(__name__)


class BaselineTrainer:
    """Simple trainer for baseline models."""

    def __init__(self, model, config: TrainerConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train(self, train_loader, val_loader=None) -> dict:
        """Train baseline model."""
        best_val_loss = float("inf")
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                # Baselines have same forward signature as StageBridge
                t = torch.rand(len(batch.receiver), device=self.device)
                stage_pair = torch.zeros(len(batch.receiver), dtype=torch.long, device=self.device)

                output = self.model(
                    receiver=batch.receiver,
                    neighbors=batch.neighbors,
                    distances=batch.distances,
                    x_t=batch.receiver,  # Self-reconstruction
                    t=t,
                    stage_pair_id=stage_pair,
                    neighbor_mask=batch.neighbor_mask,
                )

                # Simple MSE loss for baselines
                loss = output.pow(2).mean()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # Validation
            val_loss = float("inf")
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                log.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - train: {train_loss:.4f}, val: {val_loss:.4f}")

        training_time = time.time() - start_time

        return {
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss,
            "training_time_seconds": training_time,
            "total_epochs": self.config.num_epochs,
        }

    @torch.no_grad()
    def _validate(self, val_loader) -> float:
        """Validate baseline."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            t = torch.rand(len(batch.receiver), device=self.device)
            stage_pair = torch.zeros(len(batch.receiver), dtype=torch.long, device=self.device)

            output = self.model(
                receiver=batch.receiver,
                neighbors=batch.neighbors,
                distances=batch.distances,
                x_t=batch.receiver,
                t=t,
                stage_pair_id=stage_pair,
                neighbor_mask=batch.neighbor_mask,
            )

            loss = output.pow(2).mean()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train baseline model."""
    baseline_name = cfg.get("baseline", "pooling_mlp")
    log.info(f"Training baseline: {baseline_name}")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate data
    validate_contract(cfg.paths.data_dir)

    # Set seed
    torch.manual_seed(cfg.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment.seed)

    # Device
    device = cfg.hardware.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Create data loaders
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=cfg.paths.data_dir,
        fold_idx=cfg.experiment.fold_idx,
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.hardware.num_workers,
    )

    # Build baseline model
    model = get_baseline(
        baseline_name,
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Baseline {baseline_name}: {n_params:,} parameters")

    # Build trainer config
    trainer_config = TrainerConfig(
        output_dir=Path(cfg.paths.run_dir),
        run_name=f"baseline_{baseline_name}",
        num_epochs=cfg.trainer.num_epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
    )

    # Train
    trainer = BaselineTrainer(model, trainer_config, device)
    summary = trainer.train(train_loader, val_loader)

    # Assemble results
    results = {
        "baseline_name": baseline_name,
        "metrics": {
            "val_loss": summary["best_val_loss"],
            "train_loss": summary["final_train_loss"],
        },
        "n_parameters": n_params,
        "training_time_seconds": summary["training_time_seconds"],
        "config": OmegaConf.to_container(cfg),
        "completed_at": datetime.now().isoformat(),
    }

    # Validate contract
    errors = BaselineOutputContract.validate(results)
    if errors:
        log.warning(f"Contract warnings: {errors}")

    # Save results
    output_dir = Path(cfg.paths.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"baseline_{baseline_name}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "baseline_name": baseline_name,
        "n_parameters": n_params,
    }, checkpoint_dir / f"{baseline_name}_final.pt")

    log.info(f"Baseline {baseline_name} complete. Val loss: {summary['best_val_loss']:.4f}")

    return summary["best_val_loss"]


if __name__ == "__main__":
    main()
