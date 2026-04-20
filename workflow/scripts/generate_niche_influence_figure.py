#!/usr/bin/env python3
"""Generate SHAP-style niche influence visualization.

Thin wrapper around stagebridge.viz.niche_influence for Snakemake integration.
Uses attention weights from the model to show which neighbor cell types
drive transition predictions.

REQUIRES: Real attention weights from extract_attention_weights.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stagebridge.viz.niche_influence import (
    plot_niche_beeswarm,
    plot_niche_importance_bar,
    plot_niche_stage_heatmap,
    plot_niche_influence_combined,
)


def load_attention_weights(inference_dir: Path) -> pd.DataFrame:
    """Load attention weights from model inference.

    Requires attention_weights.parquet from extract_attention_weights.py
    """
    attn_path = inference_dir / "attention_weights.parquet"
    if attn_path.exists():
        return pd.read_parquet(attn_path)

    raise FileNotFoundError(
        f"No attention_weights.parquet found in {inference_dir}. "
        "Run extract_attention_weights.py first to generate real attention data."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate niche influence figures")
    parser.add_argument("--inference_dir", type=str, required=True,
                       help="Directory with attention_weights.parquet from extract_attention_weights.py")
    parser.add_argument("--cells", type=str, required=True, help="Path to cells.parquet")
    parser.add_argument("--neighborhoods", type=str, required=True, help="Path to neighborhoods.parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading attention weights...")
    influence_df = load_attention_weights(Path(args.inference_dir))

    print(f"Loaded {len(influence_df)} influence records")
    print(f"Cell types: {influence_df['neighbor_type'].nunique()}")
    print(f"Stages: {sorted(influence_df['stage'].unique())}")

    print("\nTop 10 cell types by mean attention:")
    top_types = influence_df.groupby("neighbor_type")["attention"].mean().sort_values(ascending=False).head(10)
    for ct, attn in top_types.items():
        print(f"  {ct}: {attn:.4f}")

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
