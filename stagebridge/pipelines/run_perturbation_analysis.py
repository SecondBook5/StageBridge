#!/usr/bin/env python3
"""
Niche Perturbation Analysis Pipeline

In silico experiments to establish causal claims about niche effects on cell state.
This is the key analysis for publication - enables statements like:
"Removing IL1B+ macrophages from the niche reduces predicted IL1B-IL1R1 signaling in epithelial cells"

Usage:
    python -m stagebridge.pipelines.run_perturbation_analysis \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /path/to/canonical \
        --output_dir /path/to/perturbation_results

Key Experiments:
1. IL1B+ macrophage removal - Test Peng/Kadara hypothesis
2. CAF removal - Test stromal niche effects
3. T-cell removal - Test immune niche effects
4. Cell type swapping - What if epithelial cell was in immune-rich vs immune-poor niche?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Publication-ready plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


@dataclass
class PerturbationConfig:
    """Configuration for perturbation experiments."""
    checkpoint_path: str
    data_dir: str
    output_dir: str

    # Which perturbations to run
    run_il1b_removal: bool = True
    run_caf_removal: bool = True
    run_tcell_removal: bool = True
    run_niche_swap: bool = True

    # Analysis parameters
    n_samples: int = 10000  # Number of cells to analyze
    batch_size: int = 256
    device: str = "auto"
    seed: int = 42

    # Cell type patterns for perturbation
    il1b_macrophage_pattern: str = "Macrophage|Monocyte"
    caf_pattern: str = "Fibroblast|CAF"
    tcell_pattern: str = "T cell|CD4|CD8"
    epithelial_pattern: str = "Epithelial|AT2|AT1|Club|Basal"


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load trained StageBridge model."""
    from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})

    model = StageBridgeV1Complete(
        latent_dim=config.get("latent_dim", 32),
        niche_hidden_dim=config.get("niche_hidden_dim", 128),
        context_dim=config.get("context_dim", 256),
        dropout=config.get("dropout", 0.1),
    )

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def load_data(data_dir: Path, n_samples: int, seed: int) -> tuple[pd.DataFrame, torch.Tensor]:
    """Load canonical data for perturbation analysis."""
    cells_df = pd.read_parquet(data_dir / "cells.parquet")

    # Sample cells if needed
    if n_samples < len(cells_df):
        cells_df = cells_df.sample(n=n_samples, random_state=seed)

    # Extract embeddings
    fused_cols = [c for c in cells_df.columns if c.startswith("fused_")]
    embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)

    return cells_df, embeddings


def create_niche_tokens(
    embeddings: torch.Tensor,
    cells_df: pd.DataFrame,
    latent_dim: int = 32,
) -> torch.Tensor:
    """Create 9-token niche representations."""
    n_cells = len(embeddings)
    niche_tokens = embeddings.unsqueeze(1).expand(-1, 9, -1).clone()

    # Token 7 gets gamma if available
    gamma_cols = [c for c in cells_df.columns if c.startswith("gamma_")]
    if gamma_cols:
        gamma = torch.tensor(cells_df[sorted(gamma_cols)].values, dtype=torch.float32)
        n_gamma = gamma.shape[1]
        if n_gamma < latent_dim:
            gamma_padded = torch.zeros(n_cells, latent_dim)
            gamma_padded[:, :n_gamma] = gamma
            niche_tokens[:, 7, :] = gamma_padded

    return niche_tokens


def perturb_niche(
    niche_tokens: torch.Tensor,
    cells_df: pd.DataFrame,
    perturbation_type: str,
    cell_type_pattern: str,
) -> torch.Tensor:
    """Create perturbed niche by removing/modifying cell types.

    Args:
        niche_tokens: [N, 9, D] original niche tokens
        cells_df: DataFrame with cell metadata
        perturbation_type: "remove" or "zero"
        cell_type_pattern: Regex pattern for cell types to perturb

    Returns:
        [N, 9, D] perturbed niche tokens
    """
    import re

    perturbed = niche_tokens.clone()

    # Identify cells matching pattern
    if "cell_type" in cells_df.columns:
        match_mask = cells_df["cell_type"].str.contains(cell_type_pattern, regex=True, na=False).values
    elif "celltype" in cells_df.columns:
        match_mask = cells_df["celltype"].str.contains(cell_type_pattern, regex=True, na=False).values
    else:
        # Fallback: random 10% of cells
        match_mask = np.random.random(len(cells_df)) < 0.1

    match_indices = np.where(match_mask)[0]

    if perturbation_type == "remove":
        # Zero out matching cells in niche tokens (simulates removal)
        # Tokens 1-4 are spatial ring neighbors
        for i in match_indices:
            for token_idx in [1, 2, 3, 4]:  # Spatial ring tokens
                perturbed[i, token_idx, :] = 0.0
    elif perturbation_type == "zero":
        # Zero out all tokens for matching cells
        perturbed[match_indices, :, :] = 0.0

    return perturbed


def run_perturbation_experiment(
    model: nn.Module,
    original_niche: torch.Tensor,
    perturbed_niche: torch.Tensor,
    embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Run perturbation experiment and measure effects."""
    n_samples = len(embeddings)

    all_effect_magnitudes = []
    all_state_changes = []
    all_original_il1b = []
    all_perturbed_il1b = []

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Perturbation"):
            batch_end = min(i + batch_size, n_samples)

            receiver = embeddings[i:batch_end].to(device)
            orig_niche = original_niche[i:batch_end].to(device)
            pert_niche = perturbed_niche[i:batch_end].to(device)

            # Run counterfactual prediction
            cf_output = model.perturbation_forward(receiver, orig_niche, pert_niche)

            all_effect_magnitudes.append(cf_output["effect_magnitude"].cpu())
            all_state_changes.append(cf_output["state_change"].cpu())

            # Also get IL1B predictions
            orig_il1b = model.il1b_forward(orig_niche)["il1b_score"].cpu()
            pert_il1b = model.il1b_forward(pert_niche)["il1b_score"].cpu()
            all_original_il1b.append(orig_il1b)
            all_perturbed_il1b.append(pert_il1b)

    return {
        "effect_magnitude": torch.cat(all_effect_magnitudes, dim=0),
        "state_change": torch.cat(all_state_changes, dim=0),
        "original_il1b": torch.cat(all_original_il1b, dim=0),
        "perturbed_il1b": torch.cat(all_perturbed_il1b, dim=0),
    }


def compute_statistics(results: dict[str, torch.Tensor], cells_df: pd.DataFrame) -> dict[str, Any]:
    """Compute statistics for perturbation results."""
    effect_mag = results["effect_magnitude"].numpy().flatten()
    il1b_change = (results["perturbed_il1b"] - results["original_il1b"]).numpy().flatten()

    # Overall statistics
    stats = {
        "effect_magnitude": {
            "mean": float(np.mean(effect_mag)),
            "std": float(np.std(effect_mag)),
            "median": float(np.median(effect_mag)),
            "q25": float(np.percentile(effect_mag, 25)),
            "q75": float(np.percentile(effect_mag, 75)),
        },
        "il1b_change": {
            "mean": float(np.mean(il1b_change)),
            "std": float(np.std(il1b_change)),
            "median": float(np.median(il1b_change)),
            "pct_decreased": float((il1b_change < 0).mean()),
        },
    }

    # Stage-stratified statistics (key for publication)
    if "stage" in cells_df.columns:
        stage_stats = {}
        for stage in ["Normal", "AAH", "AIS", "MIA", "LUAD"]:
            mask = cells_df["stage"].values == stage
            if mask.sum() > 0:
                stage_stats[stage] = {
                    "n_cells": int(mask.sum()),
                    "effect_mean": float(np.mean(effect_mag[mask])),
                    "effect_std": float(np.std(effect_mag[mask])),
                    "il1b_change_mean": float(np.mean(il1b_change[mask])),
                    "il1b_change_std": float(np.std(il1b_change[mask])),
                }
        stats["by_stage"] = stage_stats

    # Donor-stratified statistics (robustness check)
    if "donor_id" in cells_df.columns:
        donor_effects = []
        for donor in cells_df["donor_id"].unique():
            mask = cells_df["donor_id"].values == donor
            if mask.sum() > 10:
                donor_effects.append(np.mean(effect_mag[mask]))

        stats["donor_consistency"] = {
            "n_donors": len(donor_effects),
            "effect_mean_across_donors": float(np.mean(donor_effects)),
            "effect_std_across_donors": float(np.std(donor_effects)),
        }

    return stats


def plot_perturbation_results(
    results: dict[str, torch.Tensor],
    cells_df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str,
):
    """Generate publication-quality plots for perturbation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    effect_mag = results["effect_magnitude"].numpy().flatten()
    il1b_change = (results["perturbed_il1b"] - results["original_il1b"]).numpy().flatten()

    # Figure 1: Effect magnitude distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.hist(effect_mag, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax1.axvline(np.mean(effect_mag), color='red', linestyle='--', label=f'Mean: {np.mean(effect_mag):.3f}')
    ax1.set_xlabel('Effect Magnitude (L2 norm)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{experiment_name}: Effect of Perturbation')
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(il1b_change, bins=50, alpha=0.7, color='coral', edgecolor='white')
    ax2.axvline(np.mean(il1b_change), color='red', linestyle='--', label=f'Mean: {np.mean(il1b_change):.3f}')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('IL1B Score Change')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{experiment_name}: IL1B Pathway Change')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'{experiment_name}_distribution.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{experiment_name}_distribution.png', bbox_inches='tight')
    plt.close()

    # Figure 2: Stage-stratified effects (key for publication)
    if "stage" in cells_df.columns:
        stages = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
        stage_effects = []
        stage_errors = []
        stage_il1b = []
        stage_il1b_errors = []

        for stage in stages:
            mask = cells_df["stage"].values == stage
            if mask.sum() > 0:
                stage_effects.append(np.mean(effect_mag[mask]))
                stage_errors.append(np.std(effect_mag[mask]) / np.sqrt(mask.sum()))
                stage_il1b.append(np.mean(il1b_change[mask]))
                stage_il1b_errors.append(np.std(il1b_change[mask]) / np.sqrt(mask.sum()))
            else:
                stage_effects.append(0)
                stage_errors.append(0)
                stage_il1b.append(0)
                stage_il1b_errors.append(0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        x = np.arange(len(stages))

        ax1 = axes[0]
        ax1.bar(x, stage_effects, yerr=stage_errors, capsize=5, color='steelblue', alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(stages)
        ax1.set_xlabel('Disease Stage')
        ax1.set_ylabel('Effect Magnitude')
        ax1.set_title(f'{experiment_name}: Effect by Stage')

        ax2 = axes[1]
        colors = ['green' if v < 0 else 'red' for v in stage_il1b]
        ax2.bar(x, stage_il1b, yerr=stage_il1b_errors, capsize=5, color=colors, alpha=0.8)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages)
        ax2.set_xlabel('Disease Stage')
        ax2.set_ylabel('IL1B Change')
        ax2.set_title(f'{experiment_name}: IL1B Change by Stage')

        plt.tight_layout()
        plt.savefig(output_dir / f'{experiment_name}_by_stage.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'{experiment_name}_by_stage.png', bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Niche Perturbation Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != 'auto'
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model and data
    print("Loading model...")
    model = load_model(Path(args.checkpoint), device)

    print("Loading data...")
    cells_df, embeddings = load_data(Path(args.data_dir), args.n_samples, args.seed)
    print(f"Loaded {len(cells_df)} cells")

    # Create original niche tokens
    niche_tokens = create_niche_tokens(embeddings, cells_df)

    # Define experiments
    experiments = {
        "il1b_macrophage_removal": {
            "pattern": "Macrophage|Monocyte",
            "perturbation": "remove",
            "hypothesis": "Removing IL1B+ macrophages reduces IL1B-IL1R1 signaling in epithelial cells",
        },
        "caf_removal": {
            "pattern": "Fibroblast|CAF",
            "perturbation": "remove",
            "hypothesis": "Removing CAFs affects epithelial cell state",
        },
        "tcell_removal": {
            "pattern": "T cell|CD4|CD8",
            "perturbation": "remove",
            "hypothesis": "Removing T cells affects epithelial cell state",
        },
    }

    all_results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"Hypothesis: {exp_config['hypothesis']}")
        print(f"{'='*60}")

        # Create perturbed niche
        perturbed_niche = perturb_niche(
            niche_tokens, cells_df,
            exp_config["perturbation"],
            exp_config["pattern"]
        )

        # Run perturbation
        results = run_perturbation_experiment(
            model, niche_tokens, perturbed_niche, embeddings,
            device, args.batch_size
        )

        # Compute statistics
        stats = compute_statistics(results, cells_df)
        all_results[exp_name] = {
            "config": exp_config,
            "statistics": stats,
        }

        # Generate plots
        plot_perturbation_results(results, cells_df, output_dir, exp_name)

        # Print summary
        print(f"\nResults for {exp_name}:")
        print(f"  Effect magnitude: {stats['effect_magnitude']['mean']:.4f} +/- {stats['effect_magnitude']['std']:.4f}")
        print(f"  IL1B change: {stats['il1b_change']['mean']:.4f} +/- {stats['il1b_change']['std']:.4f}")
        print(f"  % with decreased IL1B: {stats['il1b_change']['pct_decreased']*100:.1f}%")

        if "by_stage" in stats:
            print("\n  By stage:")
            for stage, stage_stat in stats["by_stage"].items():
                print(f"    {stage}: effect={stage_stat['effect_mean']:.4f}, IL1B_change={stage_stat['il1b_change_mean']:.4f}")

    # Save all results
    with open(output_dir / "perturbation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    # Generate summary figure
    generate_summary_figure(all_results, output_dir)


def generate_summary_figure(all_results: dict, output_dir: Path):
    """Generate summary comparison figure across all experiments."""
    experiments = list(all_results.keys())
    effects = [all_results[e]["statistics"]["effect_magnitude"]["mean"] for e in experiments]
    il1b_changes = [all_results[e]["statistics"]["il1b_change"]["mean"] for e in experiments]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(experiments))

    ax1 = axes[0]
    ax1.barh(x, effects, color='steelblue', alpha=0.8)
    ax1.set_yticks(x)
    ax1.set_yticklabels([e.replace('_', '\n') for e in experiments])
    ax1.set_xlabel('Effect Magnitude')
    ax1.set_title('Perturbation Effect Size')

    ax2 = axes[1]
    colors = ['green' if v < 0 else 'red' for v in il1b_changes]
    ax2.barh(x, il1b_changes, color=colors, alpha=0.8)
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels([e.replace('_', '\n') for e in experiments])
    ax2.set_xlabel('IL1B Score Change')
    ax2.set_title('IL1B Pathway Effect')

    plt.tight_layout()
    plt.savefig(output_dir / 'perturbation_summary.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'perturbation_summary.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
