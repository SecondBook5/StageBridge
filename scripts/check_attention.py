#!/usr/bin/env python3
"""Quick check if attention is working (run on login node)."""
import torch
import sys
import json

sys.path.insert(0, '/home/booka/StageBridge')

from stagebridge.context.encoder import ReceiverCenteredNicheEncoder

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

# Create just the AMICI encoder (not full model)
encoder = ReceiverCenteredNicheEncoder(
    input_dim=40,  # z_fused dim
    hidden_dim=hpo['hidden_dim'],
    num_heads=hpo.get('amici_num_heads', 4),
    use_empty_token=True,
    empty_token_score=-3.0,
    distance_scale=hpo.get('amici_distance_scale', 50.0),
)

# Load just encoder weights from checkpoint
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['model_state_dict']

# Extract amici_encoder weights
encoder_state = {k.replace('amici_encoder.', ''): v for k, v in state_dict.items() if k.startswith('amici_encoder.')}
encoder.load_state_dict(encoder_state)
encoder.eval()
print("Encoder loaded successfully!")

# Test with random data
B, K = 4, 8
input_dim = 40
hidden_dim = hpo['hidden_dim']

# Input is raw features, encoder projects to hidden_dim internally
receiver = torch.randn(B, input_dim)
neighbors = torch.randn(B, K, input_dim)
distances = torch.rand(B, K) * 50  # 0-50 microns

with torch.no_grad():
    out = encoder(receiver, neighbors, distances)

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
