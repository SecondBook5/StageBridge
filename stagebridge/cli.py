"""Command-line interface for StageBridge.

Usage:
    stagebridge infer --checkpoint model.pt --data neighborhoods.parquet --output predictions.parquet
    stagebridge embed --checkpoint model.pt --data neighborhoods.parquet --output embeddings.parquet
    stagebridge train --data-dir /path/to/data --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_infer(args):
    """Run inference with a trained model."""
    import pandas as pd
    import stagebridge as sb

    print(f"Loading model from {args.checkpoint}")
    model = sb.StageBridge.from_pretrained(args.checkpoint, device=args.device)

    print(f"Loading data from {args.data}")
    neighborhoods = pd.read_parquet(args.data)

    print(f"Running inference: {args.source_stage} -> {args.target_stage}")
    predictions = model.predict(
        neighborhoods=neighborhoods,
        source_stage=args.source_stage,
        target_stage=args.target_stage,
    )

    output_path = Path(args.output)
    predictions.to_dataframe().to_parquet(output_path)
    print(f"Saved predictions to {output_path}")


def cmd_embed(args):
    """Get niche embeddings."""
    import pandas as pd
    import stagebridge as sb

    print(f"Loading model from {args.checkpoint}")
    model = sb.StageBridge.from_pretrained(args.checkpoint, device=args.device)

    print(f"Loading data from {args.data}")
    neighborhoods = pd.read_parquet(args.data)

    print("Computing niche embeddings...")
    embeddings = model.embed_niches(neighborhoods)

    output_path = Path(args.output)
    emb_df = pd.DataFrame(
        embeddings.embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.embeddings.shape[1])],
    )
    emb_df.to_parquet(output_path)
    print(f"Saved embeddings to {output_path}")


def cmd_train(args):
    """Train a StageBridge model."""
    from stagebridge.training import StageBridgeTrainer, TrainerConfig
    from stagebridge.models import StageBridge, StageBridgeConfig
    from stagebridge.loaders import create_dataloaders

    print(f"Loading data from {args.data_dir}")
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir,
        fold_idx=args.fold,
        batch_size=args.batch_size,
    )

    print("Creating model...")
    model_config = StageBridgeConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    model = StageBridge(model_config)

    trainer_config = TrainerConfig(
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
        learning_rate=args.lr,
    )

    trainer = StageBridgeTrainer(model, trainer_config, device=args.device)
    trainer.train(train_loader, val_loader)
    print(f"Training complete. Outputs in {args.output_dir}")


def cmd_version(args):
    """Print version."""
    import stagebridge
    print(f"stagebridge {stagebridge.__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="stagebridge",
        description="StageBridge: Niche-conditioned cell state transition modeling",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # infer
    p_infer = subparsers.add_parser("infer", help="Run inference")
    p_infer.add_argument("--checkpoint", required=True, help="Model checkpoint")
    p_infer.add_argument("--data", required=True, help="Neighborhoods parquet")
    p_infer.add_argument("--output", required=True, help="Output parquet")
    p_infer.add_argument("--source-stage", default="Normal")
    p_infer.add_argument("--target-stage", default="Invasive")
    p_infer.add_argument("--device", default="cuda")
    p_infer.set_defaults(func=cmd_infer)

    # embed
    p_embed = subparsers.add_parser("embed", help="Get embeddings")
    p_embed.add_argument("--checkpoint", required=True, help="Model checkpoint")
    p_embed.add_argument("--data", required=True, help="Neighborhoods parquet")
    p_embed.add_argument("--output", required=True, help="Output parquet")
    p_embed.add_argument("--device", default="cuda")
    p_embed.set_defaults(func=cmd_embed)

    # train
    p_train = subparsers.add_parser("train", help="Train model")
    p_train.add_argument("--data-dir", required=True, help="Data directory")
    p_train.add_argument("--output-dir", required=True, help="Output directory")
    p_train.add_argument("--run-name", default="")
    p_train.add_argument("--fold", type=int, default=0)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--hidden-dim", type=int, default=256)
    p_train.add_argument("--num-heads", type=int, default=8)
    p_train.add_argument("--ssl-epochs", type=int, default=50)
    p_train.add_argument("--transition-epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--device", default="cuda")
    p_train.set_defaults(func=cmd_train)

    # version
    p_version = subparsers.add_parser("version", help="Print version")
    p_version.set_defaults(func=cmd_version)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
