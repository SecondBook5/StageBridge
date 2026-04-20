#!/usr/bin/env python3
"""Generate Sankey diagram showing cell state transitions.

Thin wrapper around stagebridge.viz.transition_flow for Snakemake integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stagebridge.viz.transition_flow import (
    compute_transition_flows,
    plot_sankey_flow,
    plot_alluvial,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Sankey flow figures")
    parser.add_argument("--cells", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--inference_dir", type=str, help="Directory with model inference for transition probs")
    parser.add_argument("--stage_col", type=str, default="stage")
    parser.add_argument("--cell_type_col", type=str, default="cell_type")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cells data...")
    cells_df = pd.read_parquet(args.cells)

    if args.inference_dir:
        inf_path = Path(args.inference_dir) / "model_inference.parquet"
        if inf_path.exists():
            inf_df = pd.read_parquet(inf_path)
            if "transition_prob" in inf_df.columns:
                cells_df = cells_df.merge(inf_df[["cell_id", "transition_prob"]], on="cell_id", how="left")
                print(f"Loaded transition probabilities for {cells_df['transition_prob'].notna().sum()} cells")

    print(f"Loaded {len(cells_df)} cells")
    print(f"Stages: {sorted(cells_df[args.stage_col].unique())}")
    print(f"Cell types: {cells_df[args.cell_type_col].nunique()}")

    print("\nComputing transition flows...")
    flows = compute_transition_flows(
        cells_df,
        stage_col=args.stage_col,
        cell_type_col=args.cell_type_col,
    )

    print("\nGenerating Sankey diagram...")
    plot_sankey_flow(
        cells_df,
        output_dir / "fig_sankey_flow.png",
        stage_col=args.stage_col,
        cell_type_col=args.cell_type_col,
    )

    print("Generating alluvial plot...")
    plot_alluvial(
        cells_df,
        output_dir / "fig_alluvial_composition.png",
        stage_col=args.stage_col,
        cell_type_col=args.cell_type_col,
    )

    with open(output_dir / "transition_flows.json", "w") as f:
        json.dump(flows, f, indent=2, default=str)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
