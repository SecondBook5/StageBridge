#!/usr/bin/env python3
"""Fuse HLCA and LuCA embeddings into a single representation.

Snakemake script - uses snakemake.input and snakemake.output.
"""

import numpy as np
import pandas as pd
import anndata
from pathlib import Path

# Snakemake provides these
hlca_path = snakemake.input.hlca
luca_path = snakemake.input.luca
fused_output = snakemake.output.fused
confidence_output = snakemake.output.confidence

print("=" * 60)
print("Fusing HLCA and LuCA Embeddings")
print("=" * 60)

# Load HLCA
print(f"Loading HLCA: {hlca_path}")
hlca = anndata.read_h5ad(hlca_path)
print(f"  Shape: {hlca.n_obs} cells, {hlca.n_vars} dims")

# Load LuCA
print(f"Loading LuCA: {luca_path}")
luca = anndata.read_h5ad(luca_path)
print(f"  Shape: {luca.n_obs} cells, {luca.n_vars} dims")

# Verify alignment
assert list(hlca.obs.index) == list(luca.obs.index), "Cell ID mismatch!"
print(f"✓ Cell IDs aligned: {hlca.n_obs} cells")

# L2 normalize
print("L2-normalizing embeddings...")
hlca_norm = hlca.X / (np.linalg.norm(hlca.X, axis=1, keepdims=True) + 1e-8)
luca_norm = luca.X / (np.linalg.norm(luca.X, axis=1, keepdims=True) + 1e-8)

# Concatenate
fused = np.concatenate([hlca_norm, luca_norm], axis=1).astype(np.float32)
print(f"  Fused shape: {fused.shape}")

# Build fused DataFrame
df = pd.DataFrame(index=hlca.obs.index)
df.index.name = 'cell_id'

# Copy metadata
for col in ['donor_id', 'stage', 'sample_id']:
    if col in hlca.obs.columns:
        df[col] = hlca.obs[col].values

# Add HLCA latent columns
for i in range(hlca.n_vars):
    df[f'hlca_latent_{i}'] = hlca_norm[:, i]

# Add LuCA latent columns
for i in range(luca.n_vars):
    df[f'luca_latent_{i}'] = luca_norm[:, i]

# Add fused columns
for i in range(fused.shape[1]):
    df[f'fused_latent_{i}'] = fused[:, i]

df['reference_mode_used'] = 'dual'

# Save fused embedding
print(f"Saving fused embedding: {fused_output}")
df.to_parquet(fused_output, index=True)

# Compute confidence scores
print("Computing confidence scores...")

# Simple confidence based on reconstruction (distance to centroid in latent space)
hlca_centroid = hlca_norm.mean(axis=0)
luca_centroid = luca_norm.mean(axis=0)

hlca_dist = np.linalg.norm(hlca_norm - hlca_centroid, axis=1)
luca_dist = np.linalg.norm(luca_norm - luca_centroid, axis=1)

# Convert to confidence (inverse distance, normalized)
hlca_conf = 1.0 / (1.0 + hlca_dist)
luca_conf = 1.0 / (1.0 + luca_dist)

# Percentile-rank calibration for comparability
hlca_conf_cal = 1.0 - (np.argsort(np.argsort(hlca_dist)) / len(hlca_dist))
luca_conf_cal = 1.0 - (np.argsort(np.argsort(luca_dist)) / len(luca_dist))

conf_df = pd.DataFrame({
    'cell_id': hlca.obs.index,
    'hlca_confidence': hlca_conf_cal,
    'luca_confidence': luca_conf_cal,
    'hlca_distance': hlca_dist,
    'luca_distance': luca_dist,
})
conf_df.set_index('cell_id', inplace=True)

print(f"Saving confidence scores: {confidence_output}")
conf_df.to_parquet(confidence_output, index=True)

print()
print("=" * 60)
print("Fusion Complete")
print(f"  Fused: {len(df)} cells x {fused.shape[1]} dims")
print(f"  HLCA confidence: mean={hlca_conf_cal.mean():.3f}")
print(f"  LuCA confidence: mean={luca_conf_cal.mean():.3f}")
print("=" * 60)
