"""CLI entrypoint for model inference.

Usage:
    python -m stagebridge.pipelines.infer \
        --checkpoint /path/to/best.pt \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig


def run_inference(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
    save_embeddings: bool = False,
    save_attention: bool = False,
) -> None:
    """Run inference and save predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", StageBridgeConfig())
    model = StageBridge(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get test data
    _, _, test_loader = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=64)

    predictions = []
    embeddings = []
    attention_weights = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            out = model(batch, return_attention=save_attention)

            # Collect predictions
            pred_dict = {
                "cell_id": batch.cell_ids if hasattr(batch, "cell_ids") else list(range(len(batch.receiver))),
                "stage": batch.stage_idx.cpu().numpy(),
                "predicted_z": out["context"].cpu().numpy().tolist(),
            }
            predictions.append(pd.DataFrame(pred_dict))

            if save_embeddings:
                embeddings.append(out["context"].cpu().numpy())

            if save_attention and "attention" in out:
                attention_weights.append(out["attention"].cpu().numpy())

    # Save predictions
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_parquet(output_dir / "predictions.parquet")
    print(f"Saved predictions: {len(pred_df)} cells")

    if save_embeddings and embeddings:
        emb_arr = np.concatenate(embeddings, axis=0)
        emb_df = pd.DataFrame(emb_arr, columns=[f"emb_{i}" for i in range(emb_arr.shape[1])])
        emb_df.to_parquet(output_dir / "embeddings.parquet")
        print(f"Saved embeddings: {emb_arr.shape}")

    if save_attention and attention_weights:
        np.savez(output_dir / "attention_weights.npz", attention=np.concatenate(attention_weights))
        print("Saved attention weights")


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--save-attention", action="store_true")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
        save_embeddings=args.save_embeddings,
        save_attention=args.save_attention,
    )


if __name__ == "__main__":
    main()
