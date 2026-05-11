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

from stagebridge.loaders import create_dataloaders, AMICIBatch
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

    # Extract config, inferring architecture settings from state_dict
    config = StageBridgeConfig.from_checkpoint(checkpoint)

    print(f"Model config: use_amici_attention={config.use_amici_attention}")

    # Load data - we'll peek at the first batch to detect settings, then reload
    print(f"Loading test data from fold {fold_idx}...")
    _, _, test_loader = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=512)

    if test_loader is None:
        raise RuntimeError(f"No test data found for fold {fold_idx}")

    # Peek at first batch to detect evolution_dim and batch type
    # We need to reload the dataloader after this to not skip the first batch
    sample_batch = next(iter(test_loader))
    is_amici = isinstance(sample_batch, AMICIBatch)

    if sample_batch.evolution_features is not None:
        data_evolution_dim = sample_batch.evolution_features.shape[-1]
        if config.use_evolution_branch and config.evolution_dim != data_evolution_dim:
            print(
                f"WARNING: Checkpoint has evolution_dim={config.evolution_dim} but data has "
                f"evolution_dim={data_evolution_dim}. This may cause dimension mismatches."
            )
            config = StageBridgeConfig(
                **{k: v for k, v in config.__dict__.items() if k != 'evolution_dim'},
                evolution_dim=data_evolution_dim,
            )

    # Reload dataloader so we don't skip the first batch
    _, _, test_loader = create_dataloaders(data_dir, fold_idx=fold_idx, batch_size=512)

    model = StageBridge(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Running inference on {len(test_loader.dataset)} test samples...")
    print(f"Batch type: {'AMICIBatch' if is_amici else 'NicheBatch'}")

    predictions = []
    embeddings = []
    attention_weights = []
    displacements = []

    batch_count = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_count += 1
            batch = batch.to(device)

            # Encode niche - use appropriate method based on architecture
            if is_amici:
                # AMICI batch has neighbors/distances instead of ring_cells/ring_masks
                niche_output = model.encode_niche_amici(
                    receiver=batch.receiver,
                    neighbors=batch.neighbors,
                    distances=batch.distances,
                    neighbor_mask=batch.neighbor_mask,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                    evolution_features=batch.evolution_features,
                    return_reconstruction=True,
                )
            else:
                # Standard NicheBatch with ring structure
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
                "gt_receiver": batch.receiver.cpu().numpy().tolist(),  # Ground truth for evaluation
            }

            # Add reconstruction (required for SSL evaluation - must be 40d to match gt_receivers)
            if niche_output.receiver_reconstruction is None:
                raise RuntimeError(
                    "receiver_reconstruction is None - model must be called with return_reconstruction=True"
                )
            pred_dict["predicted_z"] = niche_output.receiver_reconstruction.cpu().numpy().tolist()

            predictions.append(pd.DataFrame(pred_dict))

            if save_embeddings:
                embeddings.append(niche_output.context.cpu().numpy())

            if save_attention and niche_output.attention_weights is not None:
                attn = niche_output.attention_weights.cpu().numpy()
                attention_weights.append(attn)
                # Print attention stats for first batch
                if batch_count == 1:
                    print(f"  Attention shape: {attn.shape}, range: [{attn.min():.4f}, {attn.max():.4f}], mean: {attn.mean():.4f}")
                    # Also check empty attention if available
                    if niche_output.empty_attention is not None:
                        empty_attn = niche_output.empty_attention.cpu().numpy()
                        print(f"  Empty attention: range: [{empty_attn.min():.4f}, {empty_attn.max():.4f}], mean: {empty_attn.mean():.4f}")
                        if empty_attn.mean() > 0.9:
                            print("  WARNING: Model is attending mostly to empty token - neighbors may not be informative")

            # Get displacement/drift if model has sample heads
            if hasattr(model, 'sample_heads') and model.sample_heads is not None:
                try:
                    head_output = model.sample_heads(niche_output.context)
                    if head_output.get('displacement') is not None:
                        displacements.append(head_output['displacement'].cpu().numpy())
                        if batch_count == 1:
                            disp = head_output['displacement'].cpu().numpy()
                            print(f"  Displacement shape: {disp.shape}, range: [{disp.min():.4f}, {disp.max():.4f}]")
                except Exception as e:
                    if batch_count == 1:
                        print(f"  Note: Could not get displacement: {e}")

    # Save predictions
    pred_df = pd.concat(predictions, ignore_index=True)
    pred_df.to_parquet(output_dir / "predictions.parquet")
    print(f"Saved predictions: {len(pred_df)} cells -> {output_dir / 'predictions.parquet'}")

    if save_embeddings and embeddings:
        emb_arr = np.concatenate(embeddings, axis=0)
        # Save as parquet for consistency with Snakefile expectations
        emb_df = pd.DataFrame(emb_arr, columns=[f"emb_{i}" for i in range(emb_arr.shape[1])])
        emb_df.to_parquet(output_dir / "embeddings.parquet")
        print(f"Saved embeddings: {emb_arr.shape} -> {output_dir / 'embeddings.parquet'}")

    if save_attention:
        if attention_weights:
            attn_arr = np.concatenate(attention_weights, axis=0)
            np.savez(output_dir / "attention_weights.npz", attention=attn_arr)
            print(f"Saved attention: {attn_arr.shape}, range: [{attn_arr.min():.4f}, {attn_arr.max():.4f}] -> {output_dir / 'attention_weights.npz'}")
        else:
            # Create empty attention file to satisfy Snakefile output requirement
            np.savez(output_dir / "attention_weights.npz", attention=np.array([]))
            print("Warning: No attention weights available, saved empty file")

    if displacements:
        disp_arr = np.concatenate(displacements, axis=0)
        np.save(output_dir / "displacements.npy", disp_arr)
        print(f"Saved displacements: {disp_arr.shape}, range: [{disp_arr.min():.4f}, {disp_arr.max():.4f}] -> {output_dir / 'displacements.npy'}")


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
