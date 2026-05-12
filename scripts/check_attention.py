#!/usr/bin/env python3
"""Quick check if attention is working (run on login node)."""
import torch
import sys
import json

sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

# Find checkpoint
import os
ckpt_path = '/data1/chaunzt1/stagebridge/outputs/v1.2/full/fold_0/seed_42/checkpoints/ssl_pretrained.pt'
hpo_path = '/data1/chaunzt1/stagebridge/outputs/v1.2/hpo/best_params.json'

if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found: {ckpt_path}")
    sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")

# Load HPO params
with open(hpo_path) as f:
    hpo = json.load(f)

# Create model
config = StageBridgeConfig(
    hidden_dim=hpo['hidden_dim'],
    num_heads=hpo['num_heads'],
    dropout=hpo['dropout'],
    amici_use_empty_token=True,
    amici_empty_token_score=-3.0,
)
model = StageBridge(config)

# Load weights
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# Test with random data
B, K = 4, 8
receiver = torch.randn(B, config.input_dim)
neighbors = torch.randn(B, K, config.input_dim)
distances = torch.rand(B, K) * 50  # 0-50 microns

with torch.no_grad():
    out = model.amici_encoder(receiver, neighbors, distances)

neighbor_attn = out.attention_weights.sum(dim=-1).mean().item()
empty_attn = out.empty_attention.mean().item()

print("\n" + "="*50)
print("ATTENTION CHECK")
print("="*50)
print(f"Neighbor attention total: {neighbor_attn:.4f}")
print(f"Empty attention:          {empty_attn:.4f}")
print("="*50)

if neighbor_attn > 0.7:
    print("PASS: Neighbors receiving majority of attention!")
elif neighbor_attn > 0.3:
    print("PARTIAL: Some attention to neighbors, but not great")
else:
    print("FAIL: Empty token still dominating attention")
