#!/usr/bin/env python3
"""Hyperparameter Optimization for StageBridge.

Uses Optuna to search over model and training hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

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


def run_hpo(
    data_dir: Path,
    output_dir: Path,
    n_trials: int = 30,
    n_epochs_per_trial: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    device: torch.device | None = None,
    n_jobs: int = 1,
    storage: str | None = None,
    study_name: str | None = None,
    gw_checkpoint_path: str | None = None,
) -> tuple:
    """Run Optuna HPO for StageBridge.

    Args:
        data_dir: Path to data directory with neighborhoods.parquet
        output_dir: Output directory for results
        n_trials: Number of Optuna trials
        n_epochs_per_trial: Training epochs per trial
        batch_size: Batch size
        seed: Random seed
        device: Torch device
        n_jobs: Parallel trials (1 = sequential)
        storage: Optuna storage URL for distributed HPO
        study_name: Optuna study name (for resuming)
        gw_checkpoint_path: Path to precomputed GW alignment (enables pretrained fusion)

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

    from stagebridge.loaders import create_dataloaders

    log.info("Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, fold_idx=0, batch_size=batch_size, num_workers=4
    )

    if train_loader is None:
        raise ValueError("No training data found")

    log.info(f"  Train batches: {len(train_loader)}")
    log.info(f"  Val batches: {len(val_loader) if val_loader else 0}")

    # Detect data format and evolution_dim from first batch
    sample_batch = next(iter(train_loader))
    evolution_dim = sample_batch.evolution_features.shape[-1] if sample_batch.evolution_features is not None else 0
    is_amici_format = hasattr(sample_batch, "neighbors")  # AMICIBatch has neighbors, NicheBatch has ring_cells
    log.info(f"  Evolution dim: {evolution_dim}")
    data_format = "AMICI (continuous)" if is_amici_format else "Ring (discrete)"
    log.info(f"  Data format: {data_format}")

    # If data is AMICI format, model MUST use AMICI encoder
    force_amici = is_amici_format

    def objective(trial: Trial) -> float:
        from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

        # Hyperparameters to optimize
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256])
        num_heads = trial.suggest_categorical("num_heads", [4, 8])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)

        # GW fusion hyperparameters
        # Options: "concat" (baseline), "learned_gw" (recommended), "precompute_gw" (moscot-style)
        # "precompute_gw" requires gw_checkpoint_dir with precomputed coupling
        fusion_options = ["concat", "learned_gw"]
        if gw_checkpoint_path and Path(gw_checkpoint_path).exists():
            fusion_options.append("precompute_gw")

        gw_fusion_type = trial.suggest_categorical("gw_fusion_type", fusion_options)
        use_gw_fusion = gw_fusion_type != "concat"
        gw_checkpoint_dir = gw_checkpoint_path if gw_fusion_type == "precompute_gw" else None

        gw_output_dim = trial.suggest_categorical("gw_output_dim", [40, 64, 96]) if use_gw_fusion else 40

        # AMICI attention hyperparameters
        # If data is AMICI format, MUST use AMICI encoder; otherwise it's a hyperparameter
        if force_amici:
            use_amici = True
        else:
            use_amici = trial.suggest_categorical("use_amici_attention", [True, False])
        amici_num_heads = trial.suggest_categorical("amici_num_heads", [2, 4, 8]) if use_amici else 4
        amici_distance_scale = trial.suggest_float("amici_distance_scale", 50.0, 200.0) if use_amici else 100.0

        # Dynamics type: OT-CFM (deterministic) vs Schrödinger Bridge (stochastic)
        dynamics_type = trial.suggest_categorical("dynamics_type", ["ot_cfm", "schrodinger_bridge"])
        sb_sigma = trial.suggest_float("sb_sigma", 0.05, 0.3) if dynamics_type == "schrodinger_bridge" else 0.1

        # Training weights
        ssl_weight = trial.suggest_float("ssl_weight", 0.3, 0.7)
        pathway_weight = trial.suggest_float("pathway_weight", 0.01, 0.2, log=True)
        proliferation_weight = trial.suggest_float("proliferation_weight", 0.01, 0.2, log=True)

        config = StageBridgeConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_gw_fusion=use_gw_fusion,
            gw_fusion_type=gw_fusion_type,
            gw_checkpoint_dir=gw_checkpoint_dir,
            gw_output_dim=gw_output_dim,
            use_amici_attention=use_amici,
            amici_num_heads=amici_num_heads,
            amici_distance_scale=amici_distance_scale,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
            use_cross_attn_drift=True,
            use_pathway_head=True,
            use_proliferation_head=True,
            use_evolution_branch=evolution_dim > 0,
            evolution_dim=evolution_dim,
            # Required for SSL pretraining
            use_reconstruction_head=True,
        )

        model = StageBridge(config).to(device)

        # Create Schrödinger Bridge module if needed
        # SB uses StageBridge's CrossAttentionDrift for forward drift (same as OT-CFM)
        # SB only learns the score network for backward/reversibility
        sb_module = None
        if dynamics_type == "schrodinger_bridge":
            from stagebridge.transition.schrodinger_bridge import (
                SchrodingerBridge, SchrodingerBridgeConfig
            )
            sb_config = SchrodingerBridgeConfig(
                input_dim=config.input_dim,
                context_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                num_stages=config.num_stages,
                sigma=sb_sigma,
                use_external_drift=True,  # Use StageBridge's CrossAttentionDrift
            )
            sb_module = SchrodingerBridge(sb_config).to(device)

            # Set external drift to StageBridge's forward_vector_field
            def external_drift_fn(x_t, t, context, stage_pair_id):
                return model.forward_vector_field(
                    x_t=x_t,
                    t=t,
                    context=context,
                    stage_pair_id=stage_pair_id,
                    context_tokens=None,  # Will use cached tokens if available
                )
            sb_module.set_external_drift(external_drift_fn)

            # Include SB score network params in optimizer (drift comes from model)
            all_params = list(model.parameters()) + list(sb_module.parameters())
            optimizer = AdamW(all_params, lr=lr, weight_decay=1e-4)
        else:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training loop (simplified - SSL + transition combined)
        model.train()
        if sb_module is not None:
            sb_module.train()

        for epoch in range(n_epochs_per_trial):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # Encode niche (auto-detect format from batch type)
                if hasattr(batch, "neighbors"):
                    # AMICI format (continuous distance attention)
                    niche_output = model.encode_niche_amici(
                        receiver=batch.receiver,
                        neighbors=batch.neighbors,
                        distances=batch.distances,
                        neighbor_mask=batch.neighbor_mask,
                        hlca=batch.hlca,
                        luca=batch.luca,
                        pathway=batch.pathway,
                        stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                else:
                    # Ring format (discrete spatial bins)
                    niche_output = model.encode_niche(
                        receiver=batch.receiver,
                        ring_cells=batch.ring_cells,
                        ring_masks=batch.ring_masks,
                        hlca=batch.hlca,
                        luca=batch.luca,
                        pathway=batch.pathway,
                        stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )

                # SSL loss: receiver reconstruction
                ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)

                # Transition loss depends on dynamics type
                x0 = batch.receiver
                x1 = batch.receiver + 0.1 * torch.randn_like(batch.receiver)  # Simple target
                stage_pair_id = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

                # OT-CFM loss for drift (used by both OT-CFM and SB)
                # SB uses external drift from model, so we always train the drift this way
                t = torch.rand(x0.shape[0], device=device)
                x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
                u_t = x1 - x0

                v_t = model.forward_vector_field(
                    x_t=x_t,
                    t=t,
                    context=niche_output.context,
                    stage_pair_id=stage_pair_id,
                    context_tokens=niche_output.context_tokens,
                )
                drift_loss = F.mse_loss(v_t, u_t)

                if dynamics_type == "schrodinger_bridge" and sb_module is not None:
                    # Schrödinger Bridge: drift_loss + score matching loss
                    from stagebridge.transition.schrodinger_bridge import schrodinger_bridge_loss
                    sb_loss, _ = schrodinger_bridge_loss(
                        x_src=x0,
                        x_tgt=x1,
                        sb_module=sb_module,
                        context=niche_output.context,
                        stage_pair_id=stage_pair_id,
                        num_time_samples=4,
                    )
                    # Combined: drift for forward process, score for backward
                    transition_loss = drift_loss + sb_loss
                else:
                    # Pure OT-CFM: just drift loss
                    transition_loss = drift_loss

                # Auxiliary losses
                loss_pathway = torch.tensor(0.0, device=device)
                if model.pathway_head is not None and batch.pathway_targets is not None:
                    pathway_logits = model.pathway_head(niche_output.context)
                    loss_pathway = F.mse_loss(pathway_logits, batch.pathway_targets)

                loss_proliferation = torch.tensor(0.0, device=device)
                if model.proliferation_head is not None and batch.proliferation_target is not None:
                    prolif_logit = model.proliferation_head(niche_output.context)
                    loss_proliferation = F.binary_cross_entropy_with_logits(
                        prolif_logit.squeeze(-1), batch.proliferation_target
                    )

                loss = (
                    ssl_weight * ssl_loss
                    + (1 - ssl_weight) * transition_loss
                    + pathway_weight * loss_pathway
                    + proliferation_weight * loss_proliferation
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Report for pruning
            trial.report(epoch_loss / n_batches, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in (val_loader or []):
                batch = batch.to(device)

                # Encode niche (auto-detect format from batch type)
                if hasattr(batch, "neighbors"):
                    # AMICI format
                    niche_output = model.encode_niche_amici(
                        receiver=batch.receiver,
                        neighbors=batch.neighbors,
                        distances=batch.distances,
                        neighbor_mask=batch.neighbor_mask,
                        hlca=batch.hlca,
                        luca=batch.luca,
                        pathway=batch.pathway,
                        stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                else:
                    # Ring format
                    niche_output = model.encode_niche(
                        receiver=batch.receiver,
                        ring_cells=batch.ring_cells,
                        ring_masks=batch.ring_masks,
                        hlca=batch.hlca,
                        luca=batch.luca,
                        pathway=batch.pathway,
                        stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )

                ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)
                val_loss += ssl_loss.item()
                n_val += 1

        return val_loss / max(n_val, 1) if n_val > 0 else epoch_loss / n_batches

    # Create study
    if study_name is None:
        study_name = f"stagebridge_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    best_params = study.best_params
    log.info(f"Best params: {best_params}")
    log.info(f"Best value: {study.best_value:.6f}")

    return study, best_params


def main():
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default=None, help="Study name (for resuming)")
    parser.add_argument("--gw-checkpoint", type=str, default=None,
                        help="Path to precomputed GW alignment dir (enables pretrained fusion)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 60)
    log.info("StageBridge Hyperparameter Optimization")
    log.info("=" * 60)
    log.info(f"  Data: {args.data_dir}")
    log.info(f"  Output: {args.output_dir}")
    log.info(f"  Device: {device}")
    log.info(f"  Trials: {args.n_trials}")
    log.info("=" * 60)

    study, best_params = run_hpo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
        gw_checkpoint_path=args.gw_checkpoint,
    )

    # Save results
    if best_params:
        with open(args.output_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        history = {
            "best_params": best_params,
            "best_value": study.best_value if study else None,
            "n_trials": args.n_trials,
            "timestamp": datetime.now().isoformat(),
        }
        if study:
            history["all_trials"] = [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials if t.value is not None
            ]

        with open(args.output_dir / "optimization_history.json", "w") as f:
            json.dump(history, f, indent=2)

    log.info("HPO Complete")


if __name__ == "__main__":
    main()
