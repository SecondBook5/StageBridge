#!/usr/bin/env python3
"""Generate SHAP-style niche influence visualization.

Thin wrapper around stagebridge.viz.niche_influence for Snakemake integration.
Uses attention weights from the model to show which neighbor cell types
drive transition predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.viz.niche_influence import (
    plot_niche_beeswarm,
    plot_niche_importance_bar,
    plot_niche_stage_heatmap,
    plot_niche_influence_combined,
)


def load_attention_weights(inference_dir: Path) -> pd.DataFrame:
    """Load attention weights from model inference."""
    attn_path = inference_dir / "attention_weights.parquet"
    if attn_path.exists():
        return pd.read_parquet(attn_path)

    outputs_path = inference_dir / "model_inference.parquet"
    if outputs_path.exists():
        return pd.read_parquet(outputs_path)

    raise FileNotFoundError(f"No attention weights found in {inference_dir}")


def compute_niche_influence(
    cells_df: pd.DataFrame,
    neighborhoods_df: pd.DataFrame,
    attention_weights: np.ndarray,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute per-cell-type niche influence from attention weights."""
    rows = []

    for i, (_, receiver) in enumerate(cells_df.iterrows()):
        if i >= len(attention_weights):
            break

        receiver_id = receiver["cell_id"]
        receiver_stage = receiver.get("stage", "unknown")

        neighbors = neighborhoods_df[neighborhoods_df["receiver_id"] == receiver_id]

        if len(neighbors) == 0:
            continue

        attn = attention_weights[i]
        if isinstance(attn, (list, np.ndarray)) and len(attn) > 0:
            for j, (_, neighbor) in enumerate(neighbors.iterrows()):
                if j >= len(attn):
                    break
                neighbor_type = neighbor.get(cell_type_col, "unknown")
                rows.append({
                    "cell_id": receiver_id,
                    "neighbor_type": neighbor_type,
                    "attention": float(attn[j]) if hasattr(attn, '__getitem__') else float(attn),
                    "stage": receiver_stage,
                })

    return pd.DataFrame(rows)


def generate_synthetic_influence_data(
    cells_path: Path,
    neighborhoods_path: Path,
    n_samples: int = 5000,
) -> pd.DataFrame:
    """Generate synthetic influence data for visualization testing."""
    cells_df = pd.read_parquet(cells_path)
    neighborhoods_df = pd.read_parquet(neighborhoods_path)

    if len(cells_df) > n_samples:
        cells_df = cells_df.sample(n_samples, random_state=42)

    rows = []
    cell_types = cells_df["cell_type"].unique() if "cell_type" in cells_df.columns else [f"Type_{i}" for i in range(10)]
    stages = cells_df["stage"].unique() if "stage" in cells_df.columns else [0, 1, 2, 3]

    base_influence = {ct: np.random.exponential(0.1) for ct in cell_types}

    important_types = ["Macrophage", "CAF", "Fibroblast", "T cell", "Monocyte"]
    for ct in cell_types:
        if any(imp.lower() in ct.lower() for imp in important_types):
            base_influence[ct] *= 2.5

    for _, cell in cells_df.iterrows():
        stage = cell.get("stage", np.random.choice(list(stages)))

        n_neighbors = np.random.randint(5, 20)
        for _ in range(n_neighbors):
            neighbor_type = np.random.choice(list(cell_types))

            base = base_influence.get(neighbor_type, 0.1)
            stage_factor = 1 + 0.1 * (stage if isinstance(stage, (int, float)) else 0)
            noise = np.random.exponential(0.05)

            attention = base * stage_factor + noise

            rows.append({
                "cell_id": cell.get("cell_id", str(np.random.randint(int(1e6)))),
                "neighbor_type": neighbor_type,
                "attention": min(attention, 1.0),
                "stage": stage,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate niche influence figures")
    parser.add_argument("--inference_dir", type=str, help="Directory with model inference outputs")
    parser.add_argument("--cells", type=str, required=True, help="Path to cells.parquet")
    parser.add_argument("--neighborhoods", type=str, required=True, help="Path to neighborhoods.parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    if args.use_synthetic or args.inference_dir is None:
        print("Generating synthetic influence data...")
        influence_df = generate_synthetic_influence_data(
            Path(args.cells),
            Path(args.neighborhoods),
        )
    else:
        attn_df = load_attention_weights(Path(args.inference_dir))
        cells_df = pd.read_parquet(args.cells)
        neighborhoods_df = pd.read_parquet(args.neighborhoods)

        influence_df = compute_niche_influence(
            cells_df, neighborhoods_df,
            attn_df["attention_weights"].values if "attention_weights" in attn_df.columns else attn_df.values
        )

    print(f"Loaded {len(influence_df)} influence records")
    print(f"Cell types: {influence_df['neighbor_type'].nunique()}")
    print(f"Stages: {sorted(influence_df['stage'].unique())}")

    print("\nGenerating beeswarm plot...")
    plot_niche_beeswarm(influence_df, output_dir / "fig_niche_beeswarm.png")

    print("Generating summary bar plot...")
    plot_niche_importance_bar(influence_df, output_dir / "fig_niche_summary.png")

    print("Generating stage heatmap...")
    plot_niche_stage_heatmap(influence_df, output_dir / "fig_niche_heatmap.png")

    print("Generating combined figure...")
    plot_niche_influence_combined(influence_df, output_dir / "fig_niche_influence_combined.png")

    influence_df.to_parquet(output_dir / "niche_influence_data.parquet", index=False)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
