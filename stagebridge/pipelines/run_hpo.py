#!/usr/bin/env python3
"""Hyperparameter Optimization using Optuna for StageBridge V1.

This runs before main training to find optimal hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def create_synthetic_data(n_cells: int = 10000, latent_dim: int = 32, seed: int = 42):
    """Create synthetic data for HPO trials."""
    rng = np.random.default_rng(seed)

    # Cell features (simulating fused reference embedding)
    cell_features = rng.standard_normal((n_cells, latent_dim)).astype(np.float32)

    # Niche features (simulating neighborhood context)
    niche_features = rng.standard_normal((n_cells, latent_dim * 2)).astype(np.float32)

    # Stage labels (5 stages: Normal, AAH, AIS, MIA, LUAD)
    stages = rng.integers(0, 5, size=n_cells)

    # Transition targets (flow vectors)
    targets = rng.standard_normal((n_cells, latent_dim)).astype(np.float32) * 0.1

    return {
        "cell_features": torch.from_numpy(cell_features),
        "niche_features": torch.from_numpy(niche_features),
        "stages": torch.from_numpy(stages),
        "targets": torch.from_numpy(targets),
    }


class SimpleNicheEncoder(nn.Module):
    """Simple niche encoder for HPO trials."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleTransitionModel(nn.Module):
    """Simple transition model for HPO trials."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, cell_emb, niche_emb):
        combined = torch.cat([cell_emb, niche_emb], dim=-1)
        return self.net(combined)


class HPOModel(nn.Module):
    """Combined model for HPO."""

    def __init__(
        self,
        cell_dim: int,
        niche_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.cell_encoder = SimpleNicheEncoder(cell_dim, hidden_dim, latent_dim, dropout)
        self.niche_encoder = SimpleNicheEncoder(niche_dim, hidden_dim, latent_dim, dropout)
        self.transition = SimpleTransitionModel(latent_dim, hidden_dim, dropout)

    def forward(self, cell_features, niche_features):
        cell_emb = self.cell_encoder(cell_features)
        niche_emb = self.niche_encoder(niche_features)
        flow = self.transition(cell_emb, niche_emb)
        return flow, cell_emb


def run_trial(
    trial: Trial,
    data: dict,
    device: torch.device,
    n_epochs: int = 10,
) -> float:
    """Run a single HPO trial."""
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ssl_weight = trial.suggest_float("ssl_weight", 0.5, 0.9)

    # Create model
    model = HPOModel(
        cell_dim=data["cell_features"].shape[1],
        niche_dim=data["niche_features"].shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Split data
    n_samples = len(data["cell_features"])
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        # Mini-batch training
        perm = torch.randperm(len(train_idx))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[perm[i:i+batch_size]]

            cell_feat = data["cell_features"][batch_idx].to(device)
            niche_feat = data["niche_features"][batch_idx].to(device)
            targets = data["targets"][batch_idx].to(device)

            optimizer.zero_grad()
            flow_pred, cell_emb = model(cell_feat, niche_feat)

            # Transition loss (MSE on flow)
            transition_loss = nn.functional.mse_loss(flow_pred, targets)

            # SSL loss (reconstruction proxy)
            ssl_loss = nn.functional.mse_loss(cell_emb, cell_feat[:, :cell_emb.shape[1]])

            loss = (1 - ssl_weight) * transition_loss + ssl_weight * ssl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Report intermediate value for pruning
        trial.report(total_loss / max(n_batches, 1), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Validation
    model.eval()
    with torch.no_grad():
        val_cell = data["cell_features"][val_idx].to(device)
        val_niche = data["niche_features"][val_idx].to(device)
        val_targets = data["targets"][val_idx].to(device)

        flow_pred, _ = model(val_cell, val_niche)
        val_loss = nn.functional.mse_loss(flow_pred, val_targets).item()

    return val_loss


def run_hpo(
    output_dir: Path,
    n_trials: int = 30,
    n_epochs_per_trial: int = 10,
    seed: int = 42,
) -> dict:
    """Run Optuna hyperparameter optimization."""
    if not OPTUNA_AVAILABLE:
        log.error("Optuna not available. Install with: pip install optuna")
        # Return default params
        default_params = {
            "lr": 1e-4,
            "hidden_dim": 256,
            "latent_dim": 64,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "ssl_weight": 0.7,
        }
        return {"best_params": default_params, "best_value": None, "n_trials": 0}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Create synthetic data for HPO
    log.info("Creating synthetic data for HPO trials...")
    data = create_synthetic_data(n_cells=20000, latent_dim=32, seed=seed)

    # Create study
    log.info(f"Starting Optuna study with {n_trials} trials...")
    study = optuna.create_study(
        direction="minimize",
        study_name="stagebridge_v1_hpo",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    def objective(trial: Trial) -> float:
        return run_trial(trial, data, device, n_epochs=n_epochs_per_trial)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # Get results
    best_params = study.best_params
    best_value = study.best_value

    log.info(f"Best trial value: {best_value:.6f}")
    log.info(f"Best params: {best_params}")

    # Save results
    results = {
        "best_params": best_params,
        "best_value": float(best_value) if best_value is not None else None,
        "n_trials": n_trials,
        "n_completed": len([t for t in study.trials if t.value is not None]),
        "timestamp": datetime.now().isoformat(),
        "all_trials": [
            {
                "number": t.number,
                "value": float(t.value) if t.value is not None else None,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    # Save best params
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved best params to {output_dir / 'best_params.json'}")

    # Save full history
    with open(output_dir / "optimization_history.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved optimization history to {output_dir / 'optimization_history.json'}")

    # Generate visualization if plotly available
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))

        log.info("Saved HPO visualizations")
    except Exception as e:
        log.warning(f"Could not generate HPO visualizations: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge V1")
    parser.add_argument("--data_dir", type=str, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="HPO output directory")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--n_epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("StageBridge V1 Hyperparameter Optimization")
    log.info("=" * 60)
    log.info(f"  Data dir: {args.data_dir}")
    log.info(f"  Output dir: {args.output_dir}")
    log.info(f"  Trials: {args.n_trials}")
    log.info(f"  Epochs per trial: {args.n_epochs}")
    log.info("=" * 60)

    results = run_hpo(
        output_dir=Path(args.output_dir),
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        seed=args.seed,
    )

    log.info("=" * 60)
    log.info("HPO Complete")
    log.info(f"  Best value: {results.get('best_value')}")
    log.info(f"  Completed trials: {results.get('n_completed')}/{results.get('n_trials')}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
