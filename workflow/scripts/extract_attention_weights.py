#!/usr/bin/env python3
"""Extract attention weights from trained StageBridge model.

Runs inference on canonical data with return_attention=True and saves
per-cell attention weights for downstream interpretability analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model = StageBridgeV1Complete(
        latent_dim=model_config.get("latent_dim", config.get("latent_dim", 40)),
        niche_hidden_dim=model_config.get("niche_hidden_dim", config.get("niche_hidden_dim", 128)),
        context_dim=model_config.get("context_dim", config.get("context_dim", 256)),
        dropout=model_config.get("dropout", config.get("dropout", 0.1)),
        hlca_dim=model_config.get("hlca_dim", config.get("hlca_dim", 30)),
        luca_dim=model_config.get("luca_dim", config.get("luca_dim", 10)),
        wes_feature_dim=model_config.get("wes_feature_dim", config.get("wes_feature_dim", 8)),
        use_prototypes=model_config.get("use_prototypes", config.get("use_prototypes", False)),
        num_prototypes=model_config.get("num_prototypes", config.get("num_prototypes", 16)),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def extract_attention_from_dataset(
    model,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 256,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Extract attention weights using the real data loader."""
    from stagebridge.data.loaders import StageBridgeDataset, collate_fn

    dataset = StageBridgeDataset(
        data_dir=data_dir,
        fold=0,
        split="val",
        latent_dim=128,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    cells_df = pd.read_parquet(data_dir / "cells.parquet")
    neighborhoods_df = pd.read_parquet(data_dir / "neighborhoods.parquet")

    cell_type_map = cells_df.set_index("cell_id")["cell_type"].to_dict() if "cell_type" in cells_df.columns else {}

    rows = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            output = model.forward(
                niche_tokens=batch.niche_tokens,
                distances=None,
                t=torch.zeros(batch.niche_tokens.shape[0], device=device),
                return_attention=True,
            )

            attn_weights = output.get("attention_weights", {})

            if "niche_attention" in attn_weights:
                attn = attn_weights["niche_attention"].cpu().numpy()
            elif "pma" in attn_weights:
                attn = attn_weights["pma"].cpu().numpy()
            elif attn_weights:
                first_key = list(attn_weights.keys())[0]
                attn = attn_weights[first_key].cpu().numpy()
            else:
                n_neighbors = batch.niche_tokens.shape[1] - 1
                attn = np.ones((batch.niche_tokens.shape[0], n_neighbors)) / n_neighbors

            for i in range(len(batch.cell_ids)):
                cell_id = batch.cell_ids[i]
                stage = batch.source_stages[i]

                cell_neighbors = neighborhoods_df[neighborhoods_df["cell_id"] == cell_id]

                attn_i = attn[i] if i < len(attn) else np.array([])

                for j, (_, neighbor) in enumerate(cell_neighbors.iterrows()):
                    if j >= len(attn_i):
                        break

                    sender_id = neighbor.get("sender_id", f"neighbor_{j}")
                    neighbor_type = neighbor.get("sender_type", cell_type_map.get(sender_id, "unknown"))

                    rows.append({
                        "cell_id": cell_id,
                        "neighbor_id": sender_id,
                        "neighbor_type": neighbor_type,
                        "attention": float(attn_i[j]),
                        "stage": stage,
                        "distance": neighbor.get("distance", 0),
                    })

            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  Processed {n_batches} batches, {len(rows)} attention records")

            if max_batches and n_batches >= max_batches:
                break

    return pd.DataFrame(rows)


def extract_attention_simple(
    model,
    cells_df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 256,
    max_cells: int = 50000,
) -> pd.DataFrame:
    """Simple extraction without full data loader (fallback)."""
    if len(cells_df) > max_cells:
        cells_df = cells_df.sample(max_cells, random_state=42)

    # Handle array column (z_fused) or separate columns (z_fused_0, z_fused_1, ...)
    use_array_col = "z_fused" in cells_df.columns and isinstance(cells_df["z_fused"].iloc[0], (list, np.ndarray))
    if use_array_col:
        latent_cols = None
        latent_dim = len(cells_df["z_fused"].iloc[0])
    else:
        latent_cols = [c for c in cells_df.columns if c.startswith("z_fused_")]
        if not latent_cols:
            latent_cols = [c for c in cells_df.columns if c.startswith("z_") or c.startswith("latent_")]
        if not latent_cols:
            latent_cols = [c for c in cells_df.columns if cells_df[c].dtype in [np.float32, np.float64]][:128]
        latent_dim = len(latent_cols) if latent_cols else 64

    def get_embedding(cell_row):
        if use_array_col:
            return np.array(cell_row["z_fused"], dtype=np.float32)
        elif latent_cols:
            return cell_row[latent_cols].values.astype(np.float32)
        else:
            return np.zeros(64, dtype=np.float32)

    rows = []
    cell_ids = cells_df["cell_id"].tolist()

    for start in range(0, len(cells_df), batch_size):
        end = min(start + batch_size, len(cells_df))
        batch_cells = cells_df.iloc[start:end]

        batch_niche_tokens = []
        batch_cell_info = []

        for _, cell in batch_cells.iterrows():
            cell_id = cell["cell_id"]
            stage = cell.get("stage", "unknown")

            neighbors = neighborhoods_df[neighborhoods_df["cell_id"] == cell_id]

            if len(neighbors) == 0:
                continue

            receiver_embedding = get_embedding(cell)

            neighbor_embeddings = []
            neighbor_info = []
            for _, neighbor in neighbors.head(8).iterrows():
                sender_id = neighbor.get("sender_id", "")
                sender_cell = cells_df[cells_df["cell_id"] == sender_id]
                if len(sender_cell) > 0:
                    emb = get_embedding(sender_cell.iloc[0])
                else:
                    emb = np.zeros(len(receiver_embedding), dtype=np.float32)
                neighbor_embeddings.append(emb)
                neighbor_info.append({
                    "sender_id": sender_id,
                    "sender_type": neighbor.get("sender_type", "unknown"),
                    "distance": neighbor.get("distance", 0),
                })

            while len(neighbor_embeddings) < 8:
                neighbor_embeddings.append(np.zeros_like(receiver_embedding))
                neighbor_info.append({"sender_id": "", "sender_type": "pad", "distance": 0})

            tokens = np.stack([receiver_embedding] + neighbor_embeddings[:8])
            batch_niche_tokens.append(tokens)
            batch_cell_info.append({"cell_id": cell_id, "stage": stage, "neighbors": neighbor_info})

        if not batch_niche_tokens:
            continue

        niche_tokens = torch.tensor(np.stack(batch_niche_tokens), dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model.forward(
                niche_tokens=niche_tokens,
                distances=None,
                t=torch.zeros(niche_tokens.shape[0], device=device),
                return_attention=True,
            )

        attn_weights = output.get("attention_weights", {})
        if attn_weights:
            first_key = list(attn_weights.keys())[0]
            attn = attn_weights[first_key].cpu().numpy()
        else:
            attn = np.ones((niche_tokens.shape[0], 8)) / 8

        for i, info in enumerate(batch_cell_info):
            attn_i = attn[i] if i < len(attn) else np.ones(8) / 8
            for j, neighbor in enumerate(info["neighbors"]):
                if neighbor["sender_type"] == "pad":
                    continue
                if j >= len(attn_i):
                    break
                rows.append({
                    "cell_id": info["cell_id"],
                    "neighbor_id": neighbor["sender_id"],
                    "neighbor_type": neighbor["sender_type"],
                    "attention": float(attn_i[j]),
                    "stage": info["stage"],
                    "distance": neighbor["distance"],
                })

        if start % 10000 == 0:
            print(f"  Processed {start}/{len(cells_df)} cells")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract attention weights from trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--cells", type=str, required=True, help="Path to cells.parquet")
    parser.add_argument("--neighborhoods", type=str, required=True, help="Path to neighborhoods.parquet")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_cells", type=int, default=50000, help="Max cells to process (for memory)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading model...")
    model, config = load_model(Path(args.checkpoint), device)

    print("Loading data...")
    cells_df = pd.read_parquet(args.cells)
    neighborhoods_df = pd.read_parquet(args.neighborhoods)

    data_dir = Path(args.cells).parent
    stage_edges_path = data_dir / "stage_edges.parquet"
    split_manifest_path = data_dir / "split_manifest.json"

    if stage_edges_path.exists() and split_manifest_path.exists():
        print("Using full data loader...")
        try:
            attention_df = extract_attention_from_dataset(
                model, data_dir, device,
                batch_size=args.batch_size,
                max_batches=args.max_cells // args.batch_size,
            )
        except Exception as e:
            print(f"Full loader failed: {e}, falling back to simple extraction")
            attention_df = extract_attention_simple(
                model, cells_df, neighborhoods_df, device,
                batch_size=args.batch_size,
                max_cells=args.max_cells,
            )
    else:
        print("Using simple extraction (no stage_edges.parquet)...")
        attention_df = extract_attention_simple(
            model, cells_df, neighborhoods_df, device,
            batch_size=args.batch_size,
            max_cells=args.max_cells,
        )

    output_path = output_dir / "attention_weights.parquet"
    attention_df.to_parquet(output_path, index=False)
    print(f"Saved attention weights to {output_path}")
    print(f"Total records: {len(attention_df)}")

    summary = {
        "n_cells": attention_df["cell_id"].nunique(),
        "n_attention_records": len(attention_df),
        "cell_types": sorted(attention_df["neighbor_type"].unique().tolist()),
        "stages": sorted([str(s) for s in attention_df["stage"].unique().tolist()]),
        "mean_attention_by_type": attention_df.groupby("neighbor_type")["attention"].mean().sort_values(ascending=False).to_dict(),
    }

    summary_path = output_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")

    print("\nTop 10 cell types by mean attention:")
    top_types = attention_df.groupby("neighbor_type")["attention"].mean().sort_values(ascending=False).head(10)
    for ct, attn in top_types.items():
        print(f"  {ct}: {attn:.4f}")


if __name__ == "__main__":
    main()
