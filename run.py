#!/usr/bin/env python
"""StageBridge - Single Command Demo

This is the main entry point for the minimal StageBridge v1.

Usage:
    python run.py demo           # Run demo with synthetic data
    python run.py train          # Train on real data
    python run.py infer CKPT     # Run inference with checkpoint

Examples:
    python run.py demo --output results/
    python run.py train --data-dir /path/to/processed
    python run.py infer checkpoints/best.pt --data-dir /path/to/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def create_demo_data(
    output_dir: Path,
    n_cells: int = 500,
    n_donors: int = 4,
    max_cells_per_ring: int = 20,
) -> Path:
    """Create synthetic data for demo.

    Creates neighborhoods.parquet with individual cells per ring
    (the correct v1 format for learned ISAB+PMA pooling).
    """
    from stagebridge.contracts import LATENT_DIM, STAGE_TO_IDX

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    stages = list(STAGE_TO_IDX.keys())
    donors = [f"donor_{i:03d}" for i in range(n_donors)]

    neighborhoods_data = []

    for i in range(n_cells):
        cell_id = f"cell_{i:06d}"
        donor_id = donors[i % n_donors]
        stage = stages[i % len(stages)]

        receiver_z = np.random.randn(LATENT_DIM).astype(np.float32).tolist()
        hlca_z = np.random.randn(LATENT_DIM).astype(np.float32).tolist()
        luca_z = np.random.randn(LATENT_DIM).astype(np.float32).tolist()
        pathway_z = np.random.randn(LATENT_DIM).astype(np.float32).tolist()
        stats_z = np.random.randn(LATENT_DIM).astype(np.float32).tolist()

        row = {
            "cell_id": cell_id,
            "donor_id": donor_id,
            "stage": stage,
            "receiver_z": receiver_z,
            "hlca_z": hlca_z,
            "luca_z": luca_z,
            "pathway_z": pathway_z,
            "stats_z": stats_z,
        }

        # Individual cells per ring (model learns which matter via attention)
        for ring in range(1, 5):
            n_ring_cells = np.random.randint(5, max_cells_per_ring + 1)
            cells_list = [
                np.random.randn(LATENT_DIM).astype(np.float32).tolist()
                for _ in range(n_ring_cells)
            ]
            row[f"ring_{ring}_cells"] = cells_list

        neighborhoods_data.append(row)

    neighborhoods_df = pd.DataFrame(neighborhoods_data)
    neighborhoods_df.to_parquet(output_dir / "neighborhoods.parquet", index=False)

    split_manifest = {
        "folds": [
            {
                "fold": 0,
                "train_donors": donors[:2],
                "val_donors": [donors[2]],
                "test_donors": [donors[3]],
            },
            {
                "fold": 1,
                "train_donors": [donors[0], donors[2]],
                "val_donors": [donors[3]],
                "test_donors": [donors[1]],
            },
            {
                "fold": 2,
                "train_donors": [donors[0], donors[3]],
                "val_donors": [donors[1]],
                "test_donors": [donors[2]],
            },
        ],
    }

    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"Created demo data in {output_dir}/")
    print(f"  - {len(neighborhoods_df)} cells from {n_donors} donors")
    print(f"  - Stages: {stages}")
    print(f"  - Max cells per ring: {max_cells_per_ring}")

    return output_dir


def run_demo(args):
    """Run demo with synthetic data."""
    from stagebridge.loaders import create_dataloaders
    from stagebridge.models import StageBridge, StageBridgeConfig
    from stagebridge.training import StageBridgeTrainer, TrainerConfig

    output_dir = Path(args.output)
    data_dir = output_dir / "demo_data"

    print("\n=== StageBridge Demo ===\n")

    print("Step 1: Creating synthetic data...")
    create_demo_data(data_dir, n_cells=args.n_cells)

    print("\nStep 2: Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        fold_idx=0,
        batch_size=32,
        num_workers=0,
    )
    print(f"  Train: {len(train_loader.dataset)} cells")
    print(f"  Val: {len(val_loader.dataset)} cells")
    print(f"  Test: {len(test_loader.dataset)} cells")

    print("\nStep 3: Building model...")
    config = StageBridgeConfig(
        input_dim=40,
        hidden_dim=128,
        num_heads=4,
        num_encoder_layers=2,
        max_neighbors=8,
        num_stages=3,
        use_learned_ring_pooling=True,
        use_context_refiner=True,
    )
    model = StageBridge(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  NicheTokenizer: {model.niche_tokenizer is not None}")
    print(f"  HierarchicalSetTransformer: {model.context_refiner is not None}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("\nStep 4: Training (smoke test)...")
    ssl_epochs = max(1, args.epochs // 2)
    transition_epochs = args.epochs - ssl_epochs
    trainer_config = TrainerConfig(
        output_dir=output_dir / "runs",
        run_name="demo",
        ssl_epochs=ssl_epochs,
        transition_epochs=transition_epochs,
        learning_rate=1e-3,
        warmup_epochs=1,
        checkpoint_every=5,
        eval_every=1,
        strict_gradient_check=False,
        early_stopping_enabled=False,
    )
    trainer = StageBridgeTrainer(model=model, config=trainer_config, device=device)
    summary = trainer.train(train_loader, val_loader)

    ssl_best = summary.get('ssl', {}).get('best_val_loss', 'N/A')
    trans_best = summary.get('transition', {}).get('best_val_loss', 'N/A')
    print(f"\n  SSL best val loss: {ssl_best:.4f}" if isinstance(ssl_best, (int, float)) else f"\n  SSL best val loss: {ssl_best}")
    print(f"  Transition best val loss: {trans_best:.4f}" if isinstance(trans_best, (int, float)) else f"  Transition best val loss: {trans_best}")

    print("\nStep 5: Running inference...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader)).to(device)
        niche_output = model.encode_niche(
            receiver=batch.receiver,
            ring_cells=batch.ring_cells,
            ring_masks=batch.ring_masks,
            hlca=batch.hlca,
            luca=batch.luca,
            pathway=batch.pathway,
            stats=batch.stats,
        )
        stage_pair_id = model.encode_stage_pair_tensor(0, 1, len(batch.receiver), device)
        predicted = model.integrate_euler(
            x0=batch.receiver,
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            num_steps=8,
            context_tokens=niche_output.context_tokens,
        )

    displacement = (predicted - batch.receiver).norm(dim=-1).mean().item()
    print(f"  Mean displacement (Normal->Preinvasive): {displacement:.4f}")

    print(f"\nDemo complete. Results in {output_dir}/")
    print(f"  - Training summary: {output_dir}/runs/demo/")
    return 0


def run_train(args):
    """Run training with Hydra config."""
    import subprocess
    cmd = ["python", "scripts/train.py"]
    if args.data_dir:
        cmd.append(f"paths.data_dir={args.data_dir}")
    if args.output:
        cmd.append(f"paths.output_dir={args.output}")
    if args.epochs:
        cmd.append(f"trainer.num_epochs={args.epochs}")
    return subprocess.call(cmd)


def run_infer(args):
    """Run inference with checkpoint."""
    import subprocess
    cmd = ["python", "scripts/infer.py", f"checkpoint={args.checkpoint}"]
    if args.data_dir:
        cmd.append(f"paths.data_dir={args.data_dir}")
    if args.output:
        cmd.append(f"paths.output_dir={args.output}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="StageBridge - Receiver-centered niche modeling for cancer progression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    demo_parser = subparsers.add_parser("demo", help="Run quick demo with synthetic data")
    demo_parser.add_argument("--output", "-o", default="./demo_output", help="Output directory")
    demo_parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of epochs")
    demo_parser.add_argument("--n-cells", type=int, default=500, help="Number of cells to generate")

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data-dir", "-d", help="Data directory")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of epochs")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("checkpoint", help="Checkpoint path")
    infer_parser.add_argument("--data-dir", "-d", help="Data directory")
    infer_parser.add_argument("--output", "-o", help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "demo":
        return run_demo(args)
    elif args.command == "train":
        return run_train(args)
    elif args.command == "infer":
        return run_infer(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
