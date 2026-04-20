#!/usr/bin/env python3
"""Hyperparameter Optimization for StageBridge V1.

Runs Optuna HPO on REAL data (not synthetic) using donor-held-out splits.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import optuna
    from optuna import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    log.warning("Optuna not available - install with: pip install optuna")


def load_real_data(data_dir: Path, fold: int = 0) -> tuple[TensorDataset, TensorDataset]:
    """Load real data with donor-held-out splits.

    Args:
        data_dir: Path to canonical data directory
        fold: Which fold to use for validation

    Returns:
        (train_dataset, val_dataset)
    """
    log.info(f"Loading real data from {data_dir}")

    cells_path = data_dir / "cells.parquet"
    neighborhoods_path = data_dir / "neighborhoods.parquet"
    splits_path = data_dir / "split_manifest.json"

    if not cells_path.exists():
        raise FileNotFoundError(f"cells.parquet not found at {cells_path}")

    cells_df = pd.read_parquet(cells_path)
    log.info(f"  Loaded {len(cells_df):,} cells")

    # Load neighborhoods
    if neighborhoods_path.exists():
        neighborhoods_df = pd.read_parquet(neighborhoods_path)
        log.info(f"  Loaded {len(neighborhoods_df):,} neighborhoods")
    else:
        neighborhoods_df = None
        log.warning("  No neighborhoods.parquet found")

    # Extract fused embeddings
    fused_cols = sorted([c for c in cells_df.columns if c.startswith("z_fused_")],
                        key=lambda x: int(x.split("_")[-1]))
    if not fused_cols:
        raise ValueError("No z_fused_* columns found")

    z_fused = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)
    latent_dim = z_fused.shape[1]
    log.info(f"  Fused embeddings: {z_fused.shape}")

    # Build niche tokens from neighborhoods
    n_cells = len(cells_df)
    n_tokens = 9

    if neighborhoods_df is not None and "receiver_id" in neighborhoods_df.columns:
        # Build lookup
        cell_id_col = "cell_id" if "cell_id" in cells_df.columns else cells_df.index.name
        if cell_id_col and cell_id_col in cells_df.columns:
            cell_ids = cells_df[cell_id_col].values
        else:
            cell_ids = cells_df.index.values

        cell_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        # Get ring columns
        ring_cols = [c for c in neighborhoods_df.columns if c.startswith("ring") and c.endswith("_mean")]

        niche_tokens = torch.zeros(n_cells, n_tokens, latent_dim, dtype=torch.float32)
        niche_tokens[:, 0, :] = z_fused  # Token 0 = receiver

        if ring_cols:
            # Use ring compositions
            for idx, row in neighborhoods_df.iterrows():
                receiver_id = row["receiver_id"]
                if receiver_id in cell_to_idx:
                    cell_idx = cell_to_idx[receiver_id]
                    for ring_i, col in enumerate(ring_cols[:4]):  # 4 rings
                        if col in row and row[col] is not None:
                            ring_data = row[col]
                            if isinstance(ring_data, (list, np.ndarray)) and len(ring_data) == latent_dim:
                                niche_tokens[cell_idx, ring_i + 1, :] = torch.tensor(ring_data, dtype=torch.float32)
                            else:
                                niche_tokens[cell_idx, ring_i + 1, :] = z_fused[cell_idx]
                        else:
                            niche_tokens[cell_idx, ring_i + 1, :] = z_fused[cell_idx]
        else:
            # Fallback: broadcast receiver to all tokens
            niche_tokens = z_fused.unsqueeze(1).expand(-1, n_tokens, -1).clone()

        log.info(f"  Built niche tokens: {niche_tokens.shape}")
    else:
        # No neighborhoods - broadcast receiver
        niche_tokens = z_fused.unsqueeze(1).expand(-1, n_tokens, -1).clone()
        log.warning("  No neighborhood data - using receiver for all tokens")

    # Stage indices
    if "stage_idx" in cells_df.columns:
        stage_indices = torch.tensor(cells_df["stage_idx"].values, dtype=torch.long)
    elif "stage" in cells_df.columns:
        stage_map = {"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4}
        stage_indices = torch.tensor(
            cells_df["stage"].map(stage_map).fillna(0).values, dtype=torch.long
        )
    else:
        stage_indices = torch.zeros(n_cells, dtype=torch.long)

    # Donor-held-out split
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)

        fold_spec = splits["folds"][fold]
        train_donors = set(fold_spec["train_donors"])
        val_donors = set(fold_spec["val_donors"])

        if "donor_id" in cells_df.columns:
            train_mask = cells_df["donor_id"].isin(train_donors).values
            val_mask = cells_df["donor_id"].isin(val_donors).values
            train_idx = torch.where(torch.tensor(train_mask))[0]
            val_idx = torch.where(torch.tensor(val_mask))[0]
            log.info(f"  Donor-held-out split: {len(train_idx):,} train, {len(val_idx):,} val")
        else:
            log.warning("  No donor_id column - using random split")
            n_train = int(0.9 * n_cells)
            perm = torch.randperm(n_cells)
            train_idx, val_idx = perm[:n_train], perm[n_train:]
    else:
        log.warning("  No split_manifest.json - using random split")
        n_train = int(0.9 * n_cells)
        perm = torch.randperm(n_cells)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Create datasets
    train_dataset = TensorDataset(
        niche_tokens[train_idx],
        z_fused[train_idx],  # source
        z_fused[train_idx],  # target (placeholder for HPO)
        stage_indices[train_idx],
    )
    val_dataset = TensorDataset(
        niche_tokens[val_idx],
        z_fused[val_idx],
        z_fused[val_idx],
        stage_indices[val_idx],
    )

    return train_dataset, val_dataset


def run_hpo(
    data_dir: Path,
    output_dir: Path,
    n_trials: int = 30,
    n_epochs_per_trial: int = 10,
    batch_size: int = 256,
    latent_dim: int = 40,
    seed: int = 42,
    device: torch.device = None,
) -> tuple:
    """Run HPO on real data.

    Args:
        data_dir: Path to canonical data
        output_dir: Output directory
        n_trials: Number of Optuna trials
        n_epochs_per_trial: Epochs per trial
        batch_size: Batch size
        latent_dim: Latent dimension
        seed: Random seed
        device: Torch device

    Returns:
        (study, best_params)
    """
    if not OPTUNA_AVAILABLE:
        log.error("Optuna not available")
        return None, {}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load real data once
    log.info("Loading real data for HPO...")
    train_dataset, val_dataset = load_real_data(data_dir, fold=0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    log.info(f"  Train batches: {len(train_loader)}")
    log.info(f"  Val batches: {len(val_loader)}")

    def objective(trial: Trial) -> float:
        from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

        # Hyperparameters to optimize
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        context_dim = trial.suggest_categorical("context_dim", [128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        ssl_weight = trial.suggest_float("ssl_weight", 0.5, 0.9)

        model = StageBridgeV1Complete(
            latent_dim=latent_dim,
            niche_hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training loop
        model.train()
        for epoch in range(n_epochs_per_trial):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                niche_tokens, z_source, z_target, stages = [b.to(device) for b in batch]

                optimizer.zero_grad()

                # SSL: masked token reconstruction
                context = model.encode_niche(niche_tokens)

                # Simple reconstruction loss
                if hasattr(model, 'niche_encoder') and hasattr(model.niche_encoder, 'reconstruction_head'):
                    recon = model.niche_encoder.reconstruction_head(context)
                    receiver = niche_tokens[:, 0, :]
                    ssl_loss = nn.functional.mse_loss(recon, receiver)
                else:
                    ssl_loss = torch.tensor(0.0, device=device)

                # Transition loss placeholder (context similarity)
                transition_loss = nn.functional.mse_loss(context[:, :latent_dim], z_source)

                loss = ssl_weight * ssl_loss + (1 - ssl_weight) * transition_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Report intermediate value for pruning
            trial.report(epoch_loss / n_batches, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                niche_tokens, z_source, z_target, stages = [b.to(device) for b in batch]
                context = model.encode_niche(niche_tokens)

                if hasattr(model, 'niche_encoder') and hasattr(model.niche_encoder, 'reconstruction_head'):
                    recon = model.niche_encoder.reconstruction_head(context)
                    receiver = niche_tokens[:, 0, :]
                    ssl_loss = nn.functional.mse_loss(recon, receiver)
                else:
                    ssl_loss = torch.tensor(0.0, device=device)

                transition_loss = nn.functional.mse_loss(context[:, :latent_dim], z_source)
                loss = ssl_weight * ssl_loss + (1 - ssl_weight) * transition_loss

                val_loss += loss.item()
                n_val += 1

        return val_loss / n_val

    # Run optimization
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    log.info(f"Best params: {best_params}")
    log.info(f"Best value: {study.best_value:.6f}")

    return study, best_params


def main():
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge V1 on REAL data")
    parser.add_argument("--data_dir", type=str, required=True, help="Canonical data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="HPO output directory")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--n_epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=40, help="Latent dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log.info("=" * 60)
    log.info("StageBridge V1 Hyperparameter Optimization (REAL DATA)")
    log.info("=" * 60)
    log.info(f"  Data dir: {args.data_dir}")
    log.info(f"  Output dir: {args.output_dir}")
    log.info(f"  Device: {device}")
    log.info(f"  Trials: {args.n_trials}")
    log.info(f"  Epochs per trial: {args.n_epochs}")
    log.info("=" * 60)

    study, best_params = run_hpo(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
        device=device,
    )

    # Save best_params.json
    if best_params:
        with open(output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        log.info(f"Saved best_params.json")

    # Save optimization_history.json
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

    # Generate plots
    if study and len([t for t in study.trials if t.value is not None]) > 1:
        log.info("Generating Optuna plots...")
        try:
            import optuna.visualization as vis
            figures_dir = output_dir / "figures"

            fig = vis.plot_optimization_history(study)
            fig.write_html(str(figures_dir / "optimization_history.html"))
            fig.write_image(str(figures_dir / "optimization_history.png"))

            try:
                fig = vis.plot_param_importances(study)
                fig.write_html(str(figures_dir / "param_importances.html"))
                fig.write_image(str(figures_dir / "param_importances.png"))
            except Exception as e:
                log.warning(f"Could not generate param_importances: {e}")

            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(str(figures_dir / "parallel_coordinate.html"))
            fig.write_image(str(figures_dir / "parallel_coordinate.png"))

        except ImportError:
            log.warning("Optuna visualization requires plotly: pip install plotly kaleido")
        except Exception as e:
            log.warning(f"Could not generate plots: {e}")

    log.info("=" * 60)
    log.info("HPO Complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
