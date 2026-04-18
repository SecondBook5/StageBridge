#!/usr/bin/env python3
"""
Ablation Study Orchestration for StageBridge V1

Runs all Tier 1 ablations across 5-fold cross-validation:
1. Full model (baseline)
2. No niche conditioning
3. No WES regularization
4. Pooled niche (mean instead of transformer)
5. HLCA only (no LuCA)
6. LuCA only (no HLCA)
7. Deterministic (no stochastic dynamics)
8. Flat hierarchy (no Set Transformer)

Generates:
- Table 3 (main results)
- Ablation heatmap (Figure 7)
- Statistical comparisons
"""

import argparse
from pathlib import Path
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stagebridge.config import FUSED_LATENT_DIM


ABLATION_CONFIGS = {
    # Only include args that run_v1_full.py actually supports:
    # --niche_encoder, --use_set_encoder, --use_ude, --use_wes, --wes_weight
    "full_model": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_ude": False,
        "use_wes": True,
        "wes_weight": 0.1,
    },
    "no_niche": {
        "niche_encoder": "mlp",
        "use_set_encoder": False,
        "use_ude": False,
        "use_wes": True,
        "wes_weight": 0.1,
        "no_niche": True,
        "note": "Receiver cell ONLY - no neighborhood context. KEY ABLATION for niche hypothesis.",
    },
    "no_wes": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_ude": False,
        "use_wes": False,
        "wes_weight": 0.0,
        "note": "No genomic/WES features",
    },
    "pooled_niche": {
        "niche_encoder": "mlp",
        "use_set_encoder": True,
        "use_ude": False,
        "use_wes": True,
        "wes_weight": 0.1,
        "note": "Mean pool niche instead of attention",
    },
    "flat_hierarchy": {
        "niche_encoder": "transformer",
        "use_set_encoder": False,
        "use_ude": False,
        "use_wes": True,
        "wes_weight": 0.1,
        "note": "No hierarchical Set Transformer",
    },
    # Single reference ablations
    "hlca_only": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_wes": True,
        "wes_weight": 0.1,
        "fusion_mode": "hlca_only",
        "note": "Use only HLCA (healthy) reference",
    },
    "luca_only": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_wes": True,
        "wes_weight": 0.1,
        "fusion_mode": "luca_only",
        "note": "Use only LuCA (cancer) reference",
    },
    # Stochastic vs deterministic
    "deterministic": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_wes": True,
        "wes_weight": 0.1,
        "no_stochastic": True,
        "note": "Deterministic dynamics only (no flow matching noise)",
    },
    # Interpretability
    "with_prototypes": {
        "niche_encoder": "transformer",
        "use_set_encoder": True,
        "use_wes": True,
        "wes_weight": 0.1,
        "use_prototypes": True,
        "num_prototypes": 16,
        "note": "Prototype bottleneck for interpretable niche clusters",
    },
}


def run_single_ablation(
    ablation_name: str,
    config: dict,
    data_dir: Path,
    fold: int,
    output_dir: Path,
    base_args: dict,
) -> dict:
    """Run single ablation experiment."""
    print(f"\n{'=' * 80}")
    print(f"Running: {ablation_name} (fold {fold})")
    print(f"{'=' * 80}")

    # Build command - use module execution to avoid path issues
    cmd = [
        "python",
        "-m",
        "stagebridge.pipelines.run_v1_full",
        "--data_dir",
        str(data_dir),
        "--fold",
        str(fold),
        "--output_dir",
        str(output_dir),
    ]

    # Add base args
    for key, val in base_args.items():
        cmd.extend([f"--{key}", str(val)])

    # Add ablation-specific args
    for key, val in config.items():
        if key == "note":
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(val)])

    # Debug: print full command
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {Path.cwd()}")
    print(f"  Output dir: {output_dir}")

    # Run
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Load results
        results_file = output_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            return {
                "success": True,
                "results": results,
                "stdout": result.stdout[-500:],  # Last 500 chars
            }
        else:
            return {
                "success": False,
                "error": "Results file not found",
            }

    except subprocess.CalledProcessError as e:
        print(f"   Failed: {e}")
        print(f"   Return code: {e.returncode}")
        # Print FULL stderr to catch tracebacks (they appear early, not at end)
        if e.stderr:
            print(f"   STDERR (FULL):\n{e.stderr}")
        if e.stdout:
            print(f"   STDOUT (last 2000): {e.stdout[-2000:]}")
        return {
            "success": False,
            "error": str(e),
            "stderr": e.stderr if e.stderr else "",
            "stdout": e.stdout[-2000:] if e.stdout else "",
        }


def run_all_ablations(
    data_dir: Path,
    output_base_dir: Path,
    n_folds: int = 5,
    base_args: dict = None,
    ablations: list[str] = None,
) -> pd.DataFrame:
    """Run all ablations across all folds."""
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    base_args = base_args or {
        "batch_size": 32,
        "n_epochs": 50,
        "lr": 1e-3,
        "latent_dim": FUSED_LATENT_DIM,
    }

    # Select ablations
    ablations_to_run = ablations or list(ABLATION_CONFIGS.keys())

    all_results = []

    # Run each ablation × fold
    for ablation_name in ablations_to_run:
        config = ABLATION_CONFIGS[ablation_name]

        for fold in range(n_folds):
            output_dir = output_base_dir / ablation_name / f"fold_{fold}"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_ablation(
                ablation_name=ablation_name,
                config=config,
                data_dir=data_dir,
                fold=fold,
                output_dir=output_dir,
                base_args=base_args,
            )

            if result["success"]:
                test_metrics = result["results"]["test_metrics"]

                all_results.append(
                    {
                        "ablation": ablation_name,
                        "fold": fold,
                        "success": True,
                        **test_metrics,
                    }
                )
            else:
                print(f"   Failed: {result.get('error', 'Unknown error')}")
                all_results.append(
                    {
                        "ablation": ablation_name,
                        "fold": fold,
                        "success": False,
                    }
                )

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_base_dir / "all_results.csv", index=False)

    return results_df


def generate_table3(results_df: pd.DataFrame, output_dir: Path):
    """Generate Table 3 (Main Results)."""
    print("\nGenerating Table 3 (Main Results)...")

    # Check if we have any successful results
    required_cols = ["wasserstein", "mse", "mae"]
    available_cols = [c for c in required_cols if c in results_df.columns]

    if not available_cols:
        print("  WARNING: No metric columns found - all ablations may have failed")
        print(f"  Available columns: {list(results_df.columns)}")
        # Create empty output files
        empty_df = pd.DataFrame({"Ablation": [], "Status": []})
        empty_df.to_csv(output_dir / "table3_main_results.csv", index=False)
        with open(output_dir / "table3_main_results.tex", "w") as f:
            f.write("% No results - ablations failed\n")
        return

    # Aggregate by ablation
    agg_dict = {col: ["mean", "std"] for col in available_cols}
    summary = (
        results_df.groupby("ablation")
        .agg(agg_dict)
        .round(4)
    )

    # Format for paper
    table = []
    for ablation in summary.index:
        row = {"Ablation": ablation.replace("_", " ").title()}
        if "wasserstein" in available_cols:
            row["W-dist"] = f"{summary.loc[ablation, ('wasserstein', 'mean')]:.4f} ± {summary.loc[ablation, ('wasserstein', 'std')]:.4f}"
        if "mse" in available_cols:
            row["MSE"] = f"{summary.loc[ablation, ('mse', 'mean')]:.4f} ± {summary.loc[ablation, ('mse', 'std')]:.4f}"
        if "mae" in available_cols:
            row["MAE"] = f"{summary.loc[ablation, ('mae', 'mean')]:.4f} ± {summary.loc[ablation, ('mae', 'std')]:.4f}"
        table.append(row)

    table_df = pd.DataFrame(table)
    table_df.to_csv(output_dir / "table3_main_results.csv", index=False)
    table_df.to_latex(output_dir / "table3_main_results.tex", index=False)

    print(f"  Saved: {output_dir / 'table3_main_results.csv'}")
    print("\nTable 3 Preview:")
    print(table_df.to_string(index=False))


def generate_figure7(results_df: pd.DataFrame, output_dir: Path):
    """Generate Figure 7 (Ablation Heatmap)."""
    print("\nGenerating Figure 7 (Ablation Heatmap)...")

    # Compute mean metrics per ablation
    metrics = ["wasserstein", "mse", "mae"]
    ablations = results_df["ablation"].unique()

    # Build matrix
    matrix = np.zeros((len(ablations), len(metrics)))
    for i, ablation in enumerate(ablations):
        ablation_data = results_df[results_df["ablation"] == ablation]
        for j, metric in enumerate(metrics):
            matrix[i, j] = ablation_data[metric].mean()

    # Normalize by full_model (row 0)
    if "full_model" in ablations:
        full_idx = list(ablations).index("full_model")
        baseline = matrix[full_idx]
        matrix_normalized = matrix / baseline
    else:
        matrix_normalized = matrix

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix_normalized,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        xticklabels=[m.upper() for m in metrics],
        yticklabels=[a.replace("_", " ").title() for a in ablations],
        ax=ax,
        cbar_kws={"label": "Normalized Metric (lower is better)"},
    )

    ax.set_title("Ablation Study: Impact on Transition Quality")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Ablation")

    plt.tight_layout()
    plt.savefig(output_dir / "figure7_ablation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure7_ablation_heatmap.pdf", bbox_inches="tight")

    print(f"  Saved: {output_dir / 'figure7_ablation_heatmap.png'}")


def generate_statistical_comparisons(results_df: pd.DataFrame, output_dir: Path):
    """Generate statistical comparisons (paired t-tests)."""
    print("\nGenerating statistical comparisons...")

    from scipy.stats import ttest_rel

    # Compare each ablation to full_model
    full_model_data = results_df[results_df["ablation"] == "full_model"]

    if len(full_model_data) == 0:
        print("  Warning: No full_model baseline found")
        return

    comparisons = []

    for ablation in results_df["ablation"].unique():
        if ablation == "full_model":
            continue

        ablation_data = results_df[results_df["ablation"] == ablation]

        for metric in ["wasserstein", "mse", "mae"]:
            # Paired t-test (same folds)
            full_vals = full_model_data[metric].values
            abl_vals = ablation_data[metric].values

            if len(full_vals) == len(abl_vals):
                t_stat, p_val = ttest_rel(full_vals, abl_vals)

                # Effect size (Cohen's d)
                diff = abl_vals.mean() - full_vals.mean()
                pooled_std = np.sqrt((full_vals.var() + abl_vals.var()) / 2)
                cohens_d = diff / pooled_std

                comparisons.append(
                    {
                        "ablation": ablation,
                        "metric": metric,
                        "full_model_mean": full_vals.mean(),
                        "ablation_mean": abl_vals.mean(),
                        "difference": diff,
                        "t_statistic": t_stat,
                        "p_value": p_val,
                        "cohens_d": cohens_d,
                        "significant": p_val < 0.05,
                    }
                )

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(output_dir / "statistical_comparisons.csv", index=False)

    print(f"  Saved: {output_dir / 'statistical_comparisons.csv'}")

    # Print significant results
    sig_df = comp_df[comp_df["significant"]]
    if len(sig_df) > 0:
        print("\nSignificant differences from full model (p < 0.05):")
        for _, row in sig_df.iterrows():
            print(
                f"  {row['ablation']} ({row['metric']}): d={row['cohens_d']:.3f}, p={row['p_value']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Suite")

    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ablations", type=str, nargs="+", help="Specific ablations to run")

    args = parser.parse_args()

    base_args = {
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "latent_dim": FUSED_LATENT_DIM,
    }

    # Run ablations
    results_df = run_all_ablations(
        data_dir=Path(args.data_dir),
        output_base_dir=Path(args.output_dir),
        n_folds=args.n_folds,
        base_args=base_args,
        ablations=args.ablations,
    )

    # Generate outputs
    output_dir = Path(args.output_dir)

    # Legacy outputs (Table 3, Figure 7)
    generate_table3(results_df, output_dir)
    generate_figure7(results_df, output_dir)
    generate_statistical_comparisons(results_df, output_dir)

    # New comprehensive visualizations
    print("\nGenerating comprehensive ablation visualizations...")
    try:
        from stagebridge.ablations import generate_ablation_visualizations

        # Convert results_df to format expected by visualization
        viz_df = (
            results_df.groupby("ablation")
            .agg(
                {
                    "wasserstein": "mean",
                    "mse": "mean",
                    "mae": "mean",
                }
            )
            .reset_index()
        )
        viz_df["label"] = viz_df["ablation"].str.replace("_", " ").str.title()

        # Add delta columns (relative to full_model baseline)
        if "full_model" in viz_df["ablation"].values:
            baseline = viz_df[viz_df["ablation"] == "full_model"].iloc[0]
            for col in ["wasserstein", "mse", "mae"]:
                viz_df[f"{col}_delta"] = viz_df[col] - baseline[col]

        # Generate visualizations
        figures = generate_ablation_visualizations(
            viz_df,
            output_dir / "advanced_figures",
        )
        print(f"  Generated {len(figures)} advanced figures:")
        for name, path in figures.items():
            print(f"    - {name}: {path}")

    except Exception as e:
        print(f"  Warning: Advanced visualizations failed: {e}")

    print("\n" + "=" * 80)
    print(" Ablation suite complete!")
    print(f"  Results: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
