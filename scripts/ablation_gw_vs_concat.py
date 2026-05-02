#!/usr/bin/env python
"""Controlled ablation: GW fusion vs concatenation.

Uses best HPO parameters with only use_gw_fusion varied.
Both runs complete full training (no pruning) for fair comparison.

Usage:
    python scripts/ablation_gw_vs_concat.py --data-dir /path/to/data --output-dir runs/ablation_gw

    # On HPC with sbatch:
    sbatch scripts/hpc/ablation_gw_vs_concat.sbatch
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

import torch
from torch.optim import AdamW

from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig


# Best HPO parameters (Trial 29)
BEST_HPO_PARAMS = {
    "lr": 7.927e-05,
    "hidden_dim": 256,
    "num_heads": 8,
    "dropout": 0.078,
    "ssl_weight": 0.652,
    "pathway_weight": 0.048,
    "proliferation_weight": 0.166,
}


def run_ablation(
    data_dir: Path,
    output_dir: Path,
    use_gw_fusion: bool,
    fold_idx: int = 0,
    ssl_epochs: int = 50,
    transition_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cuda",
):
    """Run single ablation arm (GW or concat)."""

    arm_name = "gw_fusion" if use_gw_fusion else "concat"
    run_dir = output_dir / arm_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ABLATION: {'GW Fusion' if use_gw_fusion else 'Concatenation'}")
    print(f"{'='*60}")
    print(f"Output: {run_dir}")

    # Load data
    print(f"\nLoading data from {data_dir}, fold {fold_idx}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        fold_idx=fold_idx,
        batch_size=batch_size
    )

    # Detect evolution_dim from data
    sample_batch = next(iter(train_loader))
    evolution_dim = 0
    if sample_batch.evolution_features is not None:
        evolution_dim = sample_batch.evolution_features.shape[-1]
        print(f"Detected evolution_dim: {evolution_dim}")

    # Model config with best HPO params
    model_config = StageBridgeConfig(
        # Architecture (from HPO best)
        hidden_dim=BEST_HPO_PARAMS["hidden_dim"],
        num_heads=BEST_HPO_PARAMS["num_heads"],
        dropout=BEST_HPO_PARAMS["dropout"],

        # THE ABLATION VARIABLE
        use_gw_fusion=use_gw_fusion,
        gw_output_dim=64 if use_gw_fusion else 40,
        gw_sinkhorn_reg=0.1,

        # Fixed architectural choices
        use_learned_ring_pooling=True,
        use_context_refiner=True,
        use_cross_attn_drift=True,
        use_pathway_head=True,
        use_proliferation_head=True,
        use_evolution_branch=evolution_dim > 0,
        evolution_dim=evolution_dim,
    )

    # Trainer config
    trainer_config = TrainerConfig(
        output_dir=run_dir,
        run_name="train",

        # Training schedule
        ssl_epochs=ssl_epochs,
        transition_epochs=transition_epochs,
        freeze_encoder=False,

        # Learning rate (from HPO best)
        learning_rate=BEST_HPO_PARAMS["lr"],
        weight_decay=1e-4,
        warmup_epochs=5,

        # Loss weights (from HPO best)
        ssl_reconstruction_weight=BEST_HPO_PARAMS["ssl_weight"],
        pathway_weight=BEST_HPO_PARAMS["pathway_weight"],
        proliferation_weight=BEST_HPO_PARAMS["proliferation_weight"],

        # OT-CFM
        use_ot=True,
        ot_epsilon=0.05,
        sinkhorn_iters=80,

        # Checkpointing
        checkpoint_every=10,
        keep_top_k=3,

        # NO early stopping for fair comparison
        early_stopping_enabled=False,

        # Hardware
        mixed_precision=True,
        gradient_clip=1.0,
    )

    # Save configs
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_config": asdict(model_config),
            "trainer_config": {k: str(v) if isinstance(v, Path) else v
                              for k, v in asdict(trainer_config).items()},
            "hpo_params": BEST_HPO_PARAMS,
            "ablation_variable": "use_gw_fusion",
            "ablation_value": use_gw_fusion,
        }, f, indent=2)
    print(f"Saved config to {config_path}")

    # Create model
    model = StageBridge(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Train
    trainer = StageBridgeTrainer(
        model=model,
        config=trainer_config,
        device=device,
    )

    summary = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Save final results
    results = {
        "arm": arm_name,
        "use_gw_fusion": use_gw_fusion,
        "n_params": n_params,
        "ssl_final_loss": summary.get("ssl", {}).get("final_loss"),
        "transition_final_loss": summary.get("transition", {}).get("final_loss"),
        "best_val_loss": summary.get("transition", {}).get("best_val_loss"),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="GW vs Concat ablation")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablation_gw"))
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--transition-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--arm", type=str, choices=["gw", "concat", "both"], default="both",
                       help="Which arm(s) to run")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.arm in ["concat", "both"]:
        results["concat"] = run_ablation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_gw_fusion=False,
            fold_idx=args.fold_idx,
            ssl_epochs=args.ssl_epochs,
            transition_epochs=args.transition_epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.arm in ["gw", "both"]:
        results["gw_fusion"] = run_ablation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_gw_fusion=True,
            fold_idx=args.fold_idx,
            ssl_epochs=args.ssl_epochs,
            transition_epochs=args.transition_epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Compare results
    if len(results) == 2:
        print("\n" + "="*60)
        print("ABLATION COMPARISON")
        print("="*60)

        for arm, r in results.items():
            print(f"\n{arm.upper()}:")
            print(f"  Parameters: {r['n_params']:,}")
            print(f"  SSL final loss: {r['ssl_final_loss']:.6f}" if r['ssl_final_loss'] else "  SSL: N/A")
            print(f"  Transition final loss: {r['transition_final_loss']:.6f}" if r['transition_final_loss'] else "  Transition: N/A")
            print(f"  Best val loss: {r['best_val_loss']:.6f}" if r['best_val_loss'] else "  Best val: N/A")

        # Save comparison
        comparison_path = args.output_dir / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
