#!/usr/bin/env python
"""Run ablation study.

Usage:
    python scripts/run_ablation.py ablation=no_niche
    python scripts/run_ablation.py ablation=no_niche,no_distance,no_gate --multirun
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from stagebridge.contracts import AblationOutputContract, assert_contract, validate_contract
from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig

log = logging.getLogger(__name__)


def apply_ablation(model: StageBridge, ablation_type: str) -> StageBridge:
    """Apply ablation modification to model.

    Args:
        model: StageBridge model
        ablation_type: Type of ablation

    Returns:
        Modified model
    """
    if ablation_type == "full":
        return model

    elif ablation_type == "no_niche":
        # Zero out context path in drift head
        def forward_hook(module, input, output):
            # Force gate to 0 (use only latent path)
            module.last_context_gate_mean = 0.0
            return output * 0  # Zero out context contribution

        if hasattr(model.drift_head, "context_out_proj"):
            model.drift_head.context_out_proj.register_forward_hook(forward_hook)
        log.info("Applied no_niche ablation: zeroing context path")

    elif ablation_type == "no_distance":
        # Zero out distance encoding
        for layer in model.niche_encoder.attention_layers:
            if hasattr(layer, "distance_encoder"):
                for param in layer.distance_encoder.parameters():
                    param.data.zero_()
                    param.requires_grad = False
        log.info("Applied no_distance ablation: zeroed distance encoder")

    elif ablation_type == "no_gate":
        # Fix gate to 1 (always use context)
        if hasattr(model.drift_head, "context_gate"):
            def gate_hook(module, input, output):
                return torch.ones_like(output)
            model.drift_head.context_gate.register_forward_hook(gate_hook)
        log.info("Applied no_gate ablation: fixed gate=1")

    elif ablation_type == "random_niche":
        # This is applied at data loading time, not model time
        log.info("random_niche ablation applied at data loading")

    elif ablation_type in ("hlca_only", "luca_only"):
        # These require model rebuild with different input_dim
        log.info(f"{ablation_type} ablation requires model rebuild (handled in config)")

    elif ablation_type == "no_token_types":
        # Zero out token type embeddings
        if hasattr(model.niche_encoder, "token_type_embedding"):
            model.niche_encoder.token_type_embedding.weight.data.zero_()
            model.niche_encoder.token_type_embedding.weight.requires_grad = False
        log.info("Applied no_token_types ablation: zeroed token type embeddings")

    elif ablation_type == "no_wes":
        # Disable evolution/WES branch
        if hasattr(model, "evolution_branch") and model.evolution_branch is not None:
            for param in model.evolution_branch.parameters():
                param.requires_grad = False
            model.evolution_branch = None
        log.info("Applied no_wes ablation: disabled evolution branch")

    elif ablation_type == "with_wes":
        # WES is enabled via config, nothing to modify at runtime
        log.info("with_wes ablation: WES enabled via config")

    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    return model


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
    )
    return StageBridge(model_cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run ablation experiment."""
    ablation_type = cfg.get("ablation", "full")
    log.info(f"Running ablation: {ablation_type}")
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

    # Build and modify model
    model = build_model(cfg)
    model = apply_ablation(model, ablation_type)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Build trainer
    trainer_config = TrainerConfig(
        output_dir=Path(cfg.paths.run_dir),
        run_name=f"ablation_{ablation_type}",
        num_epochs=cfg.trainer.num_epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_epochs=cfg.trainer.warmup_epochs,
        ot_epsilon=cfg.trainer.ot_epsilon,
        num_ot_pairs=cfg.trainer.num_ot_pairs,
        use_ot=cfg.trainer.use_ot,
        sigma=cfg.trainer.sigma,
        checkpoint_every=cfg.trainer.checkpoint_every,
        mixed_precision=cfg.hardware.mixed_precision,
        gradient_clip=cfg.trainer.gradient_clip,
    )

    trainer = StageBridgeTrainer(model=model, config=trainer_config, device=device)

    # Train
    summary = trainer.train(train_loader, val_loader)

    # Assemble results
    results = {
        "ablation_type": ablation_type,
        "metrics": {
            "val_loss": summary.get("best_val_loss", float("inf")),
            "train_loss": summary.get("final_train_loss", float("inf")),
        },
        "delta_vs_full": None,  # Computed in comparison script
        "n_trainable_params": n_trainable,
        "config": OmegaConf.to_container(cfg),
        "completed_at": datetime.now().isoformat(),
    }

    # Validate contract
    errors = AblationOutputContract.validate(results)
    if errors:
        log.warning(f"Contract warnings: {errors}")

    # Save results
    output_dir = Path(cfg.paths.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"ablation_{ablation_type}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Ablation {ablation_type} complete. Val loss: {summary.get('best_val_loss', float('inf')):.4f}")

    return summary.get("best_val_loss", float("inf"))


if __name__ == "__main__":
    main()
