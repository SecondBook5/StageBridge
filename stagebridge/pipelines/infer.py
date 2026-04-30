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
    """Run inference on held-out test set and save predictions.

    For each cell in the test set, encodes the niche context and optionally
    generates flow predictions via the transition model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle config - could be dict, nested dict, or StageBridgeConfig
    config_data = checkpoint.get("config", {})
    if isinstance(config_data, StageBridgeConfig):
        config = config_data
    elif isinstance(config_data, dict):
        # Handle nested config from trainer (has model_config key)
        if "model_config" in config_data:
            config = StageBridgeConfig(**config_data["model_config"])
        else:
            config = StageBridgeConfig(**config_data)
    else:
        config = StageBridgeConfig()

    model = StageBridge(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loading test data from fold {fold_idx}...")
    _, _, test_loader = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=64)

    if test_loader is None:
        raise RuntimeError(f"No test data found for fold {fold_idx}")

    print(f"Running inference on {len(test_loader.dataset)} test samples...")

    predictions = []
    embeddings = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Encode niche to get context embeddings
            niche_output = model.encode_niche(
                receiver=batch.receiver,
                ring_cells=batch.ring_cells,
                ring_masks=batch.ring_masks,
                hlca=batch.hlca,
                luca=batch.luca,
                pathway=batch.pathway,
                stats=batch.stats,
                evolution_features=batch.evolution_features,
                return_reconstruction=True,
            )

            # Collect predictions
            pred_dict = {
                "cell_id": batch.cell_ids,
                "donor_id": batch.donor_ids,
                "stage_idx": batch.stage_idx.cpu().numpy(),
                "context_z": niche_output.context.cpu().numpy().tolist(),
            }

            # Add reconstruction if available (for SSL evaluation)
            if niche_output.receiver_reconstruction is not None:
                pred_dict["predicted_z"] = niche_output.receiver_reconstruction.cpu().numpy().tolist()
            else:
                pred_dict["predicted_z"] = niche_output.context.cpu().numpy().tolist()

            predictions.append(pd.DataFrame(pred_dict))

            if save_embeddings:
                embeddings.append(niche_output.context.cpu().numpy())

    # Save predictions
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_parquet(output_dir / "predictions.parquet")
    print(f"Saved predictions: {len(pred_df)} cells -> {output_dir / 'predictions.parquet'}")

    if save_embeddings and embeddings:
        emb_arr = np.concatenate(embeddings, axis=0)
        np.save(output_dir / "embeddings.npy", emb_arr)
        print(f"Saved embeddings: {emb_arr.shape}")


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
