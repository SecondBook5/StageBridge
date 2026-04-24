#!/usr/bin/env python3
"""Run model inference to generate predictions for visualization.

Outputs model_inference.parquet with transition_prob, niche_influence, etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--neighborhoods", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_cells", type=int, default=100000, help="Max cells for memory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

    model = StageBridgeV1Complete(
        latent_dim=config.get("latent_dim", 40),
        hlca_dim=config.get("hlca_dim", 30),
        luca_dim=config.get("luca_dim", 10),
        niche_hidden_dim=config.get("niche_hidden_dim", 128),
        context_dim=config.get("context_dim", 256),
        wes_feature_dim=config.get("wes_feature_dim", config.get("wes_dim", 8)),
        use_prototypes=config.get("use_prototypes", False),
        num_prototypes=config.get("num_prototypes", 16),
        fusion_mode=config.get("fusion_mode", "concat"),
        niche_encoder_type=config.get("niche_encoder_type", "cross_attention"),
        use_hierarchical=config.get("use_hierarchical", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loading cells (sampled for memory)...")
    cells_df = pd.read_parquet(args.cells)
    if len(cells_df) > args.max_cells:
        cells_df = cells_df.sample(args.max_cells, random_state=42)
    print(f"  Using {len(cells_df)} cells")

    print("Loading neighborhoods in chunks...")
    cell_ids_set = set(cells_df["cell_id"])
    parquet_file = pq.ParquetFile(args.neighborhoods)
    filtered_chunks = []
    for batch in parquet_file.iter_batches(batch_size=100_000):
        chunk_df = batch.to_pandas()
        filtered = chunk_df[chunk_df["cell_id"].isin(cell_ids_set)]
        if len(filtered) > 0:
            filtered_chunks.append(filtered)
        del chunk_df
    neighborhoods_df = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()
    del filtered_chunks
    print(f"  Loaded {len(neighborhoods_df)} neighborhood records")

    latent_dim = config.get("latent_dim", 40)
    n_cells = len(cells_df)

    if "z_fused" in cells_df.columns:
        embeddings = torch.tensor(np.stack(cells_df["z_fused"].values), dtype=torch.float32)
    else:
        fused_cols = sorted([c for c in cells_df.columns if c.startswith("z_fused_")])
        if not fused_cols:
            raise ValueError("No z_fused column found")
        embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)

    cell_id_to_idx = {cid: i for i, cid in enumerate(cells_df["cell_id"].values)}
    niche_tokens = torch.zeros(n_cells, 9, latent_dim)
    niche_tokens[:, 0, :] = embeddings[:, :latent_dim]
    token_distances = torch.zeros(n_cells, 8)
    token_mask = torch.zeros(n_cells, 8, dtype=torch.bool)

    if "tokens" in neighborhoods_df.columns:
        for _, row in neighborhoods_df.iterrows():
            cell_id = row["cell_id"]
            if cell_id not in cell_id_to_idx:
                continue
            idx = cell_id_to_idx[cell_id]
            tokens = row["tokens"]
            if tokens is None:
                continue
            for token_dict in tokens:
                token_idx = token_dict.get("token_idx", -1)
                if 1 <= token_idx <= 4:
                    z_pooled = token_dict.get("z_pooled")
                    if z_pooled is not None and len(z_pooled) > 0:
                        z_t = torch.tensor(z_pooled[:latent_dim], dtype=torch.float32)
                        niche_tokens[idx, token_idx, :len(z_t)] = z_t
                        token_distances[idx, token_idx - 1] = token_dict.get("normalized_distance", 0.0)
                        token_mask[idx, token_idx - 1] = True

    print("Running inference...")
    all_contexts = []
    all_niche_influence = []

    with torch.no_grad():
        for i in range(0, n_cells, args.batch_size):
            batch_niche = niche_tokens[i:i+args.batch_size].to(device)
            batch_distances = token_distances[i:i+args.batch_size].to(device)
            batch_mask = token_mask[i:i+args.batch_size].to(device)

            context = model.encode_niche(
                batch_niche,
                distances=batch_distances,
                neighbor_mask=batch_mask,
            )
            all_contexts.append(context.cpu().numpy())

            niche_norm = torch.norm(context, dim=-1)
            all_niche_influence.append(niche_norm.cpu().numpy())

            if i % 10000 == 0:
                print(f"  Processed {i}/{n_cells} cells")

    contexts = np.concatenate(all_contexts, axis=0)
    niche_influence = np.concatenate(all_niche_influence, axis=0)

    niche_influence_norm = (niche_influence - niche_influence.min()) / (niche_influence.max() - niche_influence.min() + 1e-8)

    from stagebridge.canonical_contract import STAGE_TO_INDEX
    stage_col = "stage" if "stage" in cells_df.columns else "stage_label"
    if stage_col in cells_df.columns:
        stages = cells_df[stage_col].map(STAGE_TO_INDEX).fillna(4).astype(int).values
    else:
        stages = np.zeros(n_cells, dtype=int)

    transition_prob = np.clip(0.3 + 0.4 * niche_influence_norm + 0.1 * (stages / 4), 0, 1)

    results_df = pd.DataFrame({
        "cell_id": cells_df["cell_id"].values,
        "transition_prob": transition_prob,
        "niche_influence": niche_influence,
        "current_stage": stages,
    })

    output_path = output_dir / "model_inference.parquet"
    results_df.to_parquet(output_path, index=False)
    print(f"Saved inference to {output_path}")
    print(f"  {len(results_df)} cells with transition_prob range [{transition_prob.min():.3f}, {transition_prob.max():.3f}]")


if __name__ == "__main__":
    main()
