#!/usr/bin/env python3
"""Hyperparameter Optimization wrapper for Snakemake.

This wraps the existing HPO from run_v1_complete.py for standalone execution.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge V1")
    parser.add_argument("--data_dir", type=str, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="HPO output directory")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--n_epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for trials")
    parser.add_argument("--latent_dim", type=int, default=40, help="Latent dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log.info("=" * 60)
    log.info("StageBridge V1 Hyperparameter Optimization")
    log.info("=" * 60)
    log.info(f"  Data dir: {args.data_dir}")
    log.info(f"  Output dir: {args.output_dir}")
    log.info(f"  Device: {device}")
    log.info(f"  Trials: {args.n_trials}")
    log.info(f"  Epochs per trial: {args.n_epochs}")
    log.info("=" * 60)

    # Import and run existing HPO
    from stagebridge.pipelines.run_v1_complete import run_hyperparameter_optimization

    study, best_params = run_hyperparameter_optimization(
        device=device,
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )

    # Save best_params.json (for Snakemake output)
    if best_params:
        with open(output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        log.info(f"Saved best_params.json: {best_params}")

    # Save optimization_history.json (for Snakemake output)
    history = {
        "best_params": best_params,
        "best_value": study.best_value if study else None,
        "n_trials": args.n_trials,
        "timestamp": datetime.now().isoformat(),
    }
    if study:
        history["all_trials"] = [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ]

    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate Optuna visualization plots
    if study and len([t for t in study.trials if t.value is not None]) > 1:
        log.info("Generating Optuna visualization plots...")
        try:
            import optuna.visualization as vis

            figures_dir = output_dir / "figures"

            # Optimization history (loss over trials)
            fig = vis.plot_optimization_history(study)
            fig.write_html(str(figures_dir / "optimization_history.html"))
            fig.write_image(str(figures_dir / "optimization_history.png"))
            log.info("  Saved optimization_history.html/png")

            # Parameter importances
            try:
                fig = vis.plot_param_importances(study)
                fig.write_html(str(figures_dir / "param_importances.html"))
                fig.write_image(str(figures_dir / "param_importances.png"))
                log.info("  Saved param_importances.html/png")
            except Exception as e:
                log.warning(f"  Could not generate param_importances: {e}")

            # Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(str(figures_dir / "parallel_coordinate.html"))
            fig.write_image(str(figures_dir / "parallel_coordinate.png"))
            log.info("  Saved parallel_coordinate.html/png")

            # Slice plot (parameter vs objective)
            fig = vis.plot_slice(study)
            fig.write_html(str(figures_dir / "slice_plot.html"))
            fig.write_image(str(figures_dir / "slice_plot.png"))
            log.info("  Saved slice_plot.html/png")

            # Contour plots for top parameter pairs
            try:
                params = list(study.best_params.keys())[:4]  # Top 4 params
                if len(params) >= 2:
                    fig = vis.plot_contour(study, params=params[:2])
                    fig.write_html(str(figures_dir / "contour_plot.html"))
                    fig.write_image(str(figures_dir / "contour_plot.png"))
                    log.info("  Saved contour_plot.html/png")
            except Exception as e:
                log.warning(f"  Could not generate contour_plot: {e}")

            log.info(f"  All plots saved to {figures_dir}")

        except ImportError:
            log.warning("Optuna visualization requires plotly: pip install plotly kaleido")
        except Exception as e:
            log.warning(f"Could not generate Optuna plots: {e}")

    log.info("=" * 60)
    log.info("HPO Complete")
    if study:
        log.info(f"  Best value: {study.best_value:.6f}")
        log.info(
            f"  Completed trials: {len([t for t in study.trials if t.value])}/{args.n_trials}"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
