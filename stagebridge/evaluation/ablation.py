"""CLI entrypoint for running ablation experiments.

Usage:
    python -m stagebridge.evaluation.ablation \
        --ablation no_niche \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig

ABLATION_CONFIGS = {
    # Full model: GW fusion enabled (the complete StageBridge architecture)
    "full": {"use_gw_fusion": True, "gw_mode": "barycentric"},
    "no_niche": {"max_neighbors": 0},
    "no_distance": {"refiner_use_spatial_rpe": False},
    "no_gate": {"use_cross_attn_drift": False},  # Falls back to MLP drift
    "random_niche": {},  # Handled at data level, not config
    "hlca_only": {"use_gw_fusion": False},  # Must disable GW (needs both refs)
    "luca_only": {"use_gw_fusion": False},  # Must disable GW (needs both refs)
    "no_token_types": {},  # Would need model change
    "frozen_encoder": {},  # Special handling: loads pretrained encoder, freezes it
    "no_ring_pooling": {"use_learned_ring_pooling": False},
    "no_context_refiner": {"use_context_refiner": False},
    # GW fusion ablations
    "no_gw_fusion": {"use_gw_fusion": False},  # Fall back to concat
    "gw_project_hlca": {"use_gw_fusion": True, "gw_mode": "project_to_hlca"},
    "gw_project_luca": {"use_gw_fusion": True, "gw_mode": "project_to_luca"},
    "gw_barycentric": {"use_gw_fusion": True, "gw_mode": "barycentric"},
}


def run_frozen_encoder_ablation(
    data_dir: Path,
    output_dir: Path,
    pretrained_checkpoint: Path,
    fold_idx: int,
    seed: int,
    transition_epochs: int,
    train_loader,
    val_loader,
    device: torch.device,
) -> dict:
    """Run frozen encoder ablation: load pretrained encoder, freeze it, train only transition head.

    This tests whether the SSL-pretrained encoder learns good representations that transfer
    to the transition task without fine-tuning. A strong frozen encoder result validates
    the SSL pretraining objective.

    Freezes: niche_tokenizer, context_refiner, hierarchical_aggregator, stats_conditioner
    Trains: drift_head, time_embedding, stage_embedding, sample_heads, pathway_head, proliferation_head
    """
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"Loading pretrained model from {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config_data = checkpoint.get("config", {})
    if isinstance(config_data, dict) and "model_config" in config_data:
        config_dict = config_data["model_config"]
    elif isinstance(config_data, dict):
        config_dict = config_data
    elif hasattr(config_data, "__dict__"):
        config_dict = vars(config_data)
    else:
        config_dict = {}

    config = StageBridgeConfig(**config_dict)
    model = StageBridge(config).to(device)

    # Load pretrained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded pretrained weights")

    # Freeze encoder components (niche encoding + context refinement)
    # Keep trainable: drift_head, time/stage embeddings, prediction heads
    encoder_modules = [
        "niche_tokenizer",
        "context_refiner",
        "hierarchical_aggregator",
        "stats_conditioner",
    ]

    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        # Freeze if parameter belongs to encoder modules
        should_freeze = any(name.startswith(enc_name) for enc_name in encoder_modules)
        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Use the standard trainer but skip SSL phase (ssl_epochs=0)
    # This ensures we use the exact same training loop as train_full
    trainer_config = TrainerConfig(
        output_dir=output_dir,
        run_name="ablation_frozen_encoder",
        ssl_epochs=0,  # Skip SSL - encoder is already trained and frozen
        transition_epochs=transition_epochs,
    )
    trainer = StageBridgeTrainer(model, trainer_config, device=device)
    metrics = trainer.train(train_loader, val_loader)

    # Save checkpoint
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / "best_checkpoint.pt",
    )

    # Save results
    result = {
        "ablation": "frozen_encoder",
        "ablation_config": {"frozen_params": frozen_params, "trainable_params": trainable_params},
        "fold_idx": fold_idx,
        "seed": seed,
        "n_parameters": frozen_params + trainable_params,
        "n_trainable_parameters": trainable_params,
        "metrics": metrics,
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "completed_at": datetime.now().isoformat(),
    }

    result_path = output_dir / "ablation_frozen_encoder.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    return result


def run_ablation(
    ablation: str,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
    seed: int = 42,
    ssl_epochs: int = 50,
    transition_epochs: int = 100,
    pretrained_checkpoint: Path | None = None,
) -> dict:
    """Run a single ablation experiment.

    Args:
        ablation: Name of ablation to run
        data_dir: Path to data directory
        output_dir: Path to output directory
        fold_idx: Cross-validation fold index
        seed: Random seed
        ssl_epochs: Number of SSL pretraining epochs
        transition_epochs: Number of transition training epochs
        pretrained_checkpoint: Path to pretrained checkpoint (required for frozen_encoder)
    """
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ablation '{ablation}' on {device}, fold {fold_idx}")

    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation: {ablation}. Available: {list(ABLATION_CONFIGS)}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, fold_idx=fold_idx, batch_size=64
    )

    # Detect evolution_dim from data
    sample_batch = next(iter(train_loader))
    evolution_dim = sample_batch.evolution_features.shape[-1] if sample_batch.evolution_features is not None else 0
    print(f"Detected evolution_dim={evolution_dim} from data")

    # Handle frozen_encoder ablation specially
    if ablation == "frozen_encoder":
        if pretrained_checkpoint is None:
            raise ValueError(
                "frozen_encoder ablation requires --pretrained-checkpoint pointing to "
                "the full model checkpoint from train_full"
            )
        return run_frozen_encoder_ablation(
            data_dir=data_dir,
            output_dir=output_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            fold_idx=fold_idx,
            seed=seed,
            transition_epochs=transition_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

    # Create model with ablation config
    ablation_kwargs = ABLATION_CONFIGS[ablation].copy()
    # Override evolution_dim with detected value if evolution branch is used
    if ablation_kwargs.get("use_evolution_branch", True) and evolution_dim > 0:
        ablation_kwargs["evolution_dim"] = evolution_dim
        ablation_kwargs["use_evolution_branch"] = True
    elif evolution_dim == 0:
        ablation_kwargs["use_evolution_branch"] = False
    config = StageBridgeConfig(**ablation_kwargs)
    model = StageBridge(config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Ablation config: {ablation_kwargs}")

    # Train
    trainer_config = TrainerConfig(
        output_dir=output_dir,
        run_name=f"ablation_{ablation}",
        ssl_epochs=ssl_epochs,
        transition_epochs=transition_epochs,
    )
    trainer = StageBridgeTrainer(model, trainer_config, device=device)
    metrics = trainer.train(train_loader, val_loader)

    # Save checkpoint (matches CheckpointManager naming)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        checkpoint_dir / "best_checkpoint.pt",
    )

    # Save results
    result = {
        "ablation": ablation,
        "ablation_config": ablation_kwargs,
        "fold_idx": fold_idx,
        "seed": seed,
        "n_parameters": n_params,
        "metrics": metrics,
        "completed_at": datetime.now().isoformat(),
    }

    result_path = output_dir / f"ablation_{ablation}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment")
    parser.add_argument("--ablation", required=True, help="Ablation name")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--transition-epochs", type=int, default=100)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Path to pretrained checkpoint (required for frozen_encoder ablation)",
    )
    args = parser.parse_args()

    run_ablation(
        ablation=args.ablation,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
        seed=args.seed,
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )


if __name__ == "__main__":
    main()
