#!/usr/bin/env python3
"""Debug where attention went - check empty token vs neighbors."""

import torch
import numpy as np
from pathlib import Path
import argparse

from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.loaders import create_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = StageBridgeConfig.from_checkpoint(ckpt)

    model = StageBridge(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, fold_idx=args.fold, batch_size=32)
    batch = next(iter(test_loader)).to(device)

    print(f"\nBatch: {batch.receiver.shape[0]} cells")
    print(f"Neighbors: {batch.neighbors.shape}")
    print(f"Distances: {batch.distances.shape}")

    # Forward pass - get raw attention from AMICI encoder
    with torch.no_grad():
        # Access the AMICI encoder directly
        encoder = model.amici_encoder

        receiver = batch.receiver
        neighbors = batch.neighbors
        distances = batch.distances
        neighbor_mask = batch.neighbor_mask

        # Project
        h_receiver = encoder.receiver_proj(receiver)
        h_neighbors = encoder.neighbor_proj(neighbors)

        # Add token type embeddings
        B, K, _ = neighbors.shape
        if encoder.use_token_type_embeddings and encoder.token_type_embedding is not None:
            receiver_type = torch.zeros(B, 1, dtype=torch.long, device=device)
            h_receiver = h_receiver + encoder.token_type_embedding(receiver_type).squeeze(1)

            token_type_ids = torch.arange(1, K + 1, dtype=torch.long, device=device)
            token_type_ids = token_type_ids.unsqueeze(0).expand(B, -1)
            token_type_ids = token_type_ids.clamp(max=encoder.NUM_TOKEN_TYPES - 1)
            h_neighbors = h_neighbors + encoder.token_type_embedding(token_type_ids)

        # Run first attention layer to see raw scores
        attn_layer = encoder.attention_layers[0]

        q = attn_layer.q_proj(h_receiver).unsqueeze(1)
        k = attn_layer.k_proj(h_neighbors)
        v = attn_layer.v_proj(h_neighbors)

        q = q.view(B, 1, attn_layer.num_heads, attn_layer.head_dim).transpose(1, 2)
        k = k.view(B, K, attn_layer.num_heads, attn_layer.head_dim).transpose(1, 2)

        # Raw phenotype score
        phenotype_score = torch.matmul(q, k.transpose(-2, -1)) * attn_layer.scale
        print(f"\nPhenotype score (q·k): shape={phenotype_score.shape}")
        print(f"  range: [{phenotype_score.min():.4f}, {phenotype_score.max():.4f}]")
        print(f"  mean: {phenotype_score.mean():.4f}")

        # Distance penalty
        if attn_layer.use_distance_modulation:
            distance_coef_raw = attn_layer.distance_coef_mlp(h_receiver) + attn_layer.distance_coef_offset
            distance_coef = torch.nn.functional.softplus(distance_coef_raw)
            print(f"\nDistance coefficient: shape={distance_coef.shape}")
            print(f"  range: [{distance_coef.min():.4f}, {distance_coef.max():.4f}]")
            print(f"  mean: {distance_coef.mean():.4f}")

            normalized_dist = distances / attn_layer.distance_scale
            print(f"\nNormalized distances: shape={normalized_dist.shape}")
            print(f"  range: [{normalized_dist.min():.4f}, {normalized_dist.max():.4f}]")
            print(f"  mean: {normalized_dist.mean():.4f}")

            distance_penalty = distance_coef.unsqueeze(-1) * normalized_dist.unsqueeze(1)
            distance_penalty = distance_penalty.unsqueeze(2)
            print(f"\nDistance penalty: shape={distance_penalty.shape}")
            print(f"  range: [{distance_penalty.min():.4f}, {distance_penalty.max():.4f}]")
            print(f"  mean: {distance_penalty.mean():.4f}")

            attn_logits = phenotype_score - distance_penalty
        else:
            attn_logits = phenotype_score

        print(f"\nAttention logits (before empty token): shape={attn_logits.shape}")
        print(f"  range: [{attn_logits.min():.4f}, {attn_logits.max():.4f}]")
        print(f"  mean: {attn_logits.mean():.4f}")

        # Mask
        if neighbor_mask is not None:
            mask = neighbor_mask.unsqueeze(1).unsqueeze(2)
            attn_logits_masked = attn_logits.masked_fill(~mask, float("-inf"))
            valid_logits = attn_logits[mask.expand_as(attn_logits)]
            print(f"\nAfter masking invalid neighbors:")
            print(f"  valid logits range: [{valid_logits.min():.4f}, {valid_logits.max():.4f}]")
            print(f"  valid logits mean: {valid_logits.mean():.4f}")
        else:
            attn_logits_masked = attn_logits

        # Empty token
        if attn_layer.use_empty_token:
            print(f"\nEmpty token score (fixed): {attn_layer.empty_token_score}")

            empty_score = torch.full(
                (B, attn_layer.num_heads, 1, 1),
                attn_layer.empty_token_score,
                device=device,
                dtype=attn_logits.dtype,
            )
            attn_logits_with_empty = torch.cat([attn_logits_masked, empty_score], dim=-1)

            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_logits_with_empty, dim=-1)

            # Attention to empty token
            empty_attn = attn_weights[:, :, :, -1]  # [B, heads, 1]
            neighbor_attn = attn_weights[:, :, :, :-1]  # [B, heads, 1, K]

            print(f"\nFinal attention (with empty token):")
            print(f"  Empty token attention: mean={empty_attn.mean():.4f}, range=[{empty_attn.min():.4f}, {empty_attn.max():.4f}]")
            print(f"  Neighbor attention sum: mean={neighbor_attn.sum(dim=-1).mean():.4f}")
            print(f"  Max neighbor attention: {neighbor_attn.max():.4f}")

            # Per-head breakdown
            print(f"\nPer-head empty attention mean:")
            for h in range(attn_layer.num_heads):
                print(f"  Head {h}: {empty_attn[:, h].mean():.4f}")

        # The problem
        print("\n" + "="*60)
        print("DIAGNOSIS:")
        if attn_layer.use_empty_token and empty_attn.mean() > 0.9:
            print("  -> Model puts ~100% attention on EMPTY TOKEN")
            print("  -> Neighbor attention is effectively ZERO")
            print(f"  -> Empty score ({attn_layer.empty_token_score}) >> neighbor logits ({attn_logits_masked.max():.2f})")
            print("\n  This means the model learned that neighbors are NOT informative")
            print("  for reconstructing the receiver. The niche signal is going")
            print("  through STATS token or REFERENCE tokens instead.")
        elif attn_layer.use_distance_modulation and distance_penalty.mean() > phenotype_score.mean():
            print("  -> Distance penalty is too large")
            print("  -> Model learned to heavily penalize all neighbors")
        else:
            print("  -> Attention is distributed across neighbors")
            print("  -> Check other layers or the context refiner")


if __name__ == '__main__':
    main()
