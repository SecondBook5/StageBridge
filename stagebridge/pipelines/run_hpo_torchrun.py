#!/usr/bin/env python3
"""HPO with torchrun DDP - each trial runs via torchrun subprocess."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def run_single_trial(
    trial_params: dict,
    data_dir: Path,
    output_dir: Path,
    n_epochs: int,
    batch_size: int,
    world_size: int,
    gw_checkpoint_path: str | None,
) -> float:
    """Run a single trial using torchrun for DDP."""

    trial_dir = output_dir / f"trial_{trial_params['trial_id']}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write params to file for worker to read
    params_file = trial_dir / "params.json"
    with open(params_file, "w") as f:
        json.dump(trial_params, f)

    result_file = trial_dir / "result.json"

    # Run via torchrun
    cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--standalone",
        "-m", "stagebridge.pipelines._hpo_worker",
        "--params-file", str(params_file),
        "--result-file", str(result_file),
        "--data-dir", str(data_dir),
        "--n-epochs", str(n_epochs),
        "--batch-size", str(batch_size),
    ]
    if gw_checkpoint_path:
        cmd.extend(["--gw-checkpoint", gw_checkpoint_path])

    log.info(f"Running trial {trial_params['trial_id']} with torchrun...")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        log.error(f"Trial failed:\n{proc.stderr}")
        return float("inf")

    # Read result
    if result_file.exists():
        with open(result_file) as f:
            result = json.load(f)
        return result.get("val_loss", float("inf"))

    return float("inf")


def run_hpo(
    data_dir: Path,
    output_dir: Path,
    n_trials: int = 30,
    n_epochs: int = 15,
    batch_size: int = 64,
    seed: int = 42,
    world_size: int = 4,
    storage: str | None = None,
    study_name: str | None = None,
    gw_checkpoint_path: str | None = None,
):
    """Run HPO with torchrun DDP."""

    if not OPTUNA_AVAILABLE:
        log.error("Optuna not available")
        return None, {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    def objective(trial) -> float:
        fusion_options = ["concat", "learned_gw"]
        if gw_checkpoint_path and Path(gw_checkpoint_path).exists():
            fusion_options.append("precompute_gw")

        gw_fusion_type = trial.suggest_categorical("gw_fusion_type", fusion_options)
        use_gw = gw_fusion_type != "concat"

        trial_params = {
            "trial_id": trial.number,
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256]),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "gw_fusion_type": gw_fusion_type,
            "gw_output_dim": trial.suggest_categorical("gw_output_dim", [40, 64, 96]) if use_gw else 40,
            "amici_num_heads": trial.suggest_categorical("amici_num_heads", [2, 4, 8]),
            "amici_distance_scale": trial.suggest_float("amici_distance_scale", 50.0, 200.0),
            "dynamics_type": trial.suggest_categorical("dynamics_type", ["ot_cfm", "schrodinger_bridge"]),
            "sb_sigma": trial.suggest_float("sb_sigma", 0.05, 0.3),
            "ssl_weight": trial.suggest_float("ssl_weight", 0.3, 0.7),
            "pathway_weight": trial.suggest_float("pathway_weight", 0.01, 0.2, log=True),
            "proliferation_weight": trial.suggest_float("proliferation_weight", 0.01, 0.2, log=True),
        }

        log.info(f"Trial {trial.number}: {trial_params}")

        val_loss = run_single_trial(
            trial_params=trial_params,
            data_dir=data_dir,
            output_dir=output_dir,
            n_epochs=n_epochs,
            batch_size=batch_size,
            world_size=world_size,
            gw_checkpoint_path=gw_checkpoint_path,
        )

        log.info(f"Trial {trial.number} finished with val_loss: {val_loss:.6f}")
        return val_loss

    if study_name is None:
        study_name = f"stagebridge_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
        )
    else:
        study = optuna.create_study(direction="minimize")

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    log.info(f"Best params: {study.best_params}")
    log.info(f"Best value: {study.best_value:.6f}")

    return study, study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--gw-checkpoint", type=str, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("StageBridge HPO with torchrun DDP")
    log.info("=" * 60)

    study, best_params = run_hpo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        world_size=args.world_size,
        storage=args.storage,
        study_name=args.study_name,
        gw_checkpoint_path=args.gw_checkpoint,
    )

    if best_params:
        with open(args.output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

    log.info("HPO Complete")


if __name__ == "__main__":
    main()
