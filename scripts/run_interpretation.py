#!/usr/bin/env python
"""Run post-training interpretation analysis on a trained StageBridge model.

This script provides model interpretation using:
- Token ablation analysis (contribution of each token type)
- Attention pattern extraction (spatial attention decay, reference balance)
- Interaction network inference (cell-cell communication patterns)
- Trajectory visualization (flow fields, fate probabilities)

Usage:
    python scripts/run_interpretation.py \
        --checkpoint /path/to/best_checkpoint.pt \
        --data-dir /path/to/data \
        --output-dir /path/to/interpretation \
        --all

    # Or run specific analyses:
    python scripts/run_interpretation.py \
        --checkpoint /path/to/best_checkpoint.pt \
        --data-dir /path/to/data \
        --output-dir /path/to/interpretation \
        --ablation --attention
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig


def run_ablation_analysis(
    model: StageBridge,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run token ablation analysis."""
    from stagebridge.interpretation import AblationModule, plot_ablation_importance

    print("Running token ablation analysis...")
    ablation = AblationModule.compute(
        model=model,
        dataloader=dataloader,
        device=device,
        compute_per_stage=True,
        progress_bar=True,
    )

    # Save results
    results_dict = {
        name: {
            "baseline_loss": r.baseline_loss,
            "ablated_loss": r.ablated_loss,
            "delta_loss": r.delta_loss,
            "relative_importance": r.relative_importance,
            "p_value": r.p_value,
            "n_samples": r.n_samples,
        }
        for name, r in ablation.results.items()
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    if ablation.per_sample_losses is not None:
        ablation.per_sample_losses.to_parquet(output_dir / "ablation_per_sample.parquet")

    if ablation.stage_breakdown is not None:
        ablation.stage_breakdown.to_parquet(output_dir / "ablation_by_stage.parquet")

    # Generate figure
    fig_path = output_dir / "figures" / "ablation_importance.pdf"
    fig_path.parent.mkdir(exist_ok=True)
    plot_ablation_importance(ablation, save_path=fig_path)
    print(f"  Saved: {fig_path}")

    return results_dict


def run_attention_analysis(
    model: StageBridge,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run attention pattern analysis."""
    from stagebridge.interpretation import (
        AttentionModule,
        plot_ring_attention_decay,
        plot_reference_balance,
    )

    print("Running attention pattern analysis...")
    attention = AttentionModule.compute(
        model=model,
        dataloader=dataloader,
        device=device,
        progress_bar=True,
    )

    # Save results
    if attention.attention_df is not None:
        attention.attention_df.to_parquet(output_dir / "attention_patterns.parquet")

    if attention.empty_attention_df is not None:
        attention.empty_attention_df.to_parquet(output_dir / "empty_attention.parquet")

    with open(output_dir / "attention_summary.json", "w") as f:
        json.dump(attention.summary_stats, f, indent=2, default=str)

    # Generate figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_ring_attention_decay(attention, save_path=fig_dir / "ring_attention_decay.pdf")
    print(f"  Saved: {fig_dir / 'ring_attention_decay.pdf'}")

    plot_reference_balance(attention, save_path=fig_dir / "reference_balance.pdf")
    print(f"  Saved: {fig_dir / 'reference_balance.pdf'}")

    return attention.summary_stats


def run_network_analysis(
    model: StageBridge,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run interaction network inference."""
    from stagebridge.interpretation import (
        build_interaction_network,
        plot_interaction_network,
        plot_interaction_heatmap,
        plot_stage_network_comparison,
    )

    print("Running interaction network analysis...")
    network = build_interaction_network(
        model=model,
        dataloader=dataloader,
        device=device,
        progress_bar=True,
    )

    # Save network
    network.to_parquet(output_dir / "interaction_network.parquet")

    # Summary stats
    summary = {
        "n_edges": len(network.edges) if hasattr(network, "edges") else 0,
        "n_nodes": len(network.nodes) if hasattr(network, "nodes") else 0,
    }

    with open(output_dir / "network_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_interaction_network(network, save_path=fig_dir / "interaction_network.pdf")
    print(f"  Saved: {fig_dir / 'interaction_network.pdf'}")

    plot_interaction_heatmap(network, save_path=fig_dir / "interaction_heatmap.pdf")
    print(f"  Saved: {fig_dir / 'interaction_heatmap.pdf'}")

    plot_stage_network_comparison(network, save_path=fig_dir / "stage_network_comparison.pdf")
    print(f"  Saved: {fig_dir / 'stage_network_comparison.pdf'}")

    return summary


def run_trajectory_analysis(
    model: StageBridge,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run trajectory and dynamics analysis."""
    from stagebridge.interpretation import (
        TrajectoryAnalysis,
        plot_temporal_evolution,
        plot_fate_probability,
        plot_single_cell_trajectories,
    )

    print("Running trajectory analysis...")
    trajectory = TrajectoryAnalysis.compute(
        model=model,
        dataloader=dataloader,
        device=device,
        progress_bar=True,
    )

    # Save trajectory data
    if trajectory.embeddings is not None:
        trajectory.embeddings.to_parquet(output_dir / "trajectory_embeddings.parquet")

    if trajectory.fate_probabilities is not None:
        trajectory.fate_probabilities.to_parquet(output_dir / "fate_probabilities.parquet")

    summary = {
        "n_cells": trajectory.n_cells if hasattr(trajectory, "n_cells") else 0,
        "n_stages": trajectory.n_stages if hasattr(trajectory, "n_stages") else 0,
    }

    with open(output_dir / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_temporal_evolution(trajectory, save_path=fig_dir / "temporal_evolution.pdf")
    print(f"  Saved: {fig_dir / 'temporal_evolution.pdf'}")

    plot_fate_probability(trajectory, save_path=fig_dir / "fate_probability.pdf")
    print(f"  Saved: {fig_dir / 'fate_probability.pdf'}")

    plot_single_cell_trajectories(trajectory, save_path=fig_dir / "cell_trajectories.pdf")
    print(f"  Saved: {fig_dir / 'cell_trajectories.pdf'}")

    return summary


def run_manifold_analysis(
    model: StageBridge,
    dataloader: torch.utils.data.DataLoader,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Run manifold comparison and visualization."""
    from stagebridge.interpretation import (
        compute_manifold_comparison,
        plot_manifold_comparison,
        plot_phase_portrait_grid,
    )

    print("Running manifold analysis...")
    manifold = compute_manifold_comparison(
        model=model,
        dataloader=dataloader,
        device=device,
        progress_bar=True,
    )

    # Save comparison results
    summary = {
        "methods_compared": manifold.methods if hasattr(manifold, "methods") else [],
        "n_cells": manifold.n_cells if hasattr(manifold, "n_cells") else 0,
    }

    with open(output_dir / "manifold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_manifold_comparison(manifold, save_path=fig_dir / "manifold_comparison.pdf")
    print(f"  Saved: {fig_dir / 'manifold_comparison.pdf'}")

    plot_phase_portrait_grid(manifold, save_path=fig_dir / "phase_portrait_grid.pdf")
    print(f"  Saved: {fig_dir / 'phase_portrait_grid.pdf'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run post-training interpretation analysis")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--fold-idx", type=int, default=0, help="Fold index for data loading")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Analysis flags
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--ablation", action="store_true", help="Run token ablation analysis")
    parser.add_argument("--attention", action="store_true", help="Run attention pattern analysis")
    parser.add_argument("--network", action="store_true", help="Run interaction network analysis")
    parser.add_argument("--trajectory", action="store_true", help="Run trajectory analysis")
    parser.add_argument("--manifold", action="store_true", help="Run manifold comparison")

    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("StageBridge Post-Training Interpretation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = StageBridgeConfig.from_checkpoint(checkpoint)
    model = StageBridge(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        fold_idx=args.fold_idx,
        batch_size=args.batch_size,
        num_workers=4,
    )
    dataloader = val_loader if val_loader else train_loader
    print(f"  Using {'validation' if val_loader else 'training'} data: {len(dataloader)} batches")

    # Determine which analyses to run
    run_all = args.all or not any([args.ablation, args.attention, args.network, args.trajectory, args.manifold])

    results = {}

    # Run analyses
    if run_all or args.ablation:
        try:
            results["ablation"] = run_ablation_analysis(model, dataloader, args.output_dir, device)
        except Exception as e:
            print(f"  Ablation analysis failed: {e}")
            results["ablation"] = {"error": str(e)}

    if run_all or args.attention:
        try:
            results["attention"] = run_attention_analysis(model, dataloader, args.output_dir, device)
        except Exception as e:
            print(f"  Attention analysis failed: {e}")
            results["attention"] = {"error": str(e)}

    if run_all or args.network:
        try:
            results["network"] = run_network_analysis(model, dataloader, args.output_dir, device)
        except Exception as e:
            print(f"  Network analysis failed: {e}")
            results["network"] = {"error": str(e)}

    if run_all or args.trajectory:
        try:
            results["trajectory"] = run_trajectory_analysis(model, dataloader, args.output_dir, device)
        except Exception as e:
            print(f"  Trajectory analysis failed: {e}")
            results["trajectory"] = {"error": str(e)}

    if run_all or args.manifold:
        try:
            results["manifold"] = run_manifold_analysis(model, dataloader, args.output_dir, device)
        except Exception as e:
            print(f"  Manifold analysis failed: {e}")
            results["manifold"] = {"error": str(e)}

    # Save summary
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "fold_idx": args.fold_idx,
        "device": str(device),
        "completed_at": datetime.now().isoformat(),
        "analyses": results,
    }

    with open(args.output_dir / "interpretation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Interpretation Complete")
    print("=" * 60)
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
