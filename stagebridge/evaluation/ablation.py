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
    "frozen_encoder": {},  # Handled at training level
    "no_ring_pooling": {"use_learned_ring_pooling": False},
    "no_context_refiner": {"use_context_refiner": False},
    # GW fusion ablations
    "no_gw_fusion": {"use_gw_fusion": False},  # Fall back to concat
    "gw_project_hlca": {"use_gw_fusion": True, "gw_mode": "project_to_hlca"},
    "gw_project_luca": {"use_gw_fusion": True, "gw_mode": "project_to_luca"},
    "gw_barycentric": {"use_gw_fusion": True, "gw_mode": "barycentric"},
}


def run_ablation(
    ablation: str,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
    seed: int = 42,
    ssl_epochs: int = 50,
    transition_epochs: int = 100,
) -> dict:
    """Run a single ablation experiment."""
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

    # Create model with ablation config
    ablation_kwargs = ABLATION_CONFIGS[ablation]
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
    args = parser.parse_args()

    run_ablation(
        ablation=args.ablation,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
        seed=args.seed,
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
    )


if __name__ == "__main__":
    main()
