#!/usr/bin/env python3
"""Hyperparameter Optimization for StageBridge with DDP.

Single Optuna process, but each trial trains with DDP across multiple GPUs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
    log.warning("Optuna not available")


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_trial_ddp(
    rank: int,
    world_size: int,
    trial_params: dict,
    data_dir: Path,
    n_epochs: int,
    batch_size: int,
    gw_checkpoint_path: str | None,
    return_dict: dict,
):
    """Train a single trial with DDP."""
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    from stagebridge.loaders import create_dataloaders
    from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig

    # Load data with DDP sampler
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, fold_idx=0, batch_size=batch_size, num_workers=4, use_ddp=True,
    )

    if train_loader is None:
        if rank == 0:
            return_dict["val_loss"] = float("inf")
        cleanup_ddp()
        return

    # Get data info from first batch
    sample_batch = next(iter(train_loader))
    evolution_dim = sample_batch.evolution_features.shape[-1] if sample_batch.evolution_features is not None else 0
    is_amici = hasattr(sample_batch, "neighbors")

    # Unpack trial params
    lr = trial_params["lr"]
    hidden_dim = trial_params["hidden_dim"]
    num_heads = trial_params["num_heads"]
    dropout = trial_params["dropout"]
    gw_fusion_type = trial_params["gw_fusion_type"]
    gw_output_dim = trial_params.get("gw_output_dim", 40)
    amici_num_heads = trial_params.get("amici_num_heads", 4)
    amici_distance_scale = trial_params.get("amici_distance_scale", 100.0)
    dynamics_type = trial_params["dynamics_type"]
    sb_sigma = trial_params.get("sb_sigma", 0.1)
    ssl_weight = trial_params["ssl_weight"]
    pathway_weight = trial_params["pathway_weight"]
    proliferation_weight = trial_params["proliferation_weight"]

    use_gw_fusion = gw_fusion_type != "concat"
    gw_checkpoint_dir = gw_checkpoint_path if gw_fusion_type == "precompute_gw" else None

    config = StageBridgeConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        use_gw_fusion=use_gw_fusion,
        gw_fusion_type=gw_fusion_type,
        gw_checkpoint_dir=gw_checkpoint_dir,
        gw_output_dim=gw_output_dim,
        use_amici_attention=is_amici,
        amici_num_heads=amici_num_heads,
        amici_distance_scale=amici_distance_scale,
        use_learned_ring_pooling=True,
        use_context_refiner=True,
        use_cross_attn_drift=True,
        use_pathway_head=True,
        use_proliferation_head=True,
        use_evolution_branch=evolution_dim > 0,
        evolution_dim=evolution_dim,
    )

    model = StageBridge(config).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # SB module if needed
    sb_module = None
    if dynamics_type == "schrodinger_bridge":
        from stagebridge.transition.schrodinger_bridge import SchrodingerBridge, SchrodingerBridgeConfig
        sb_config = SchrodingerBridgeConfig(
            input_dim=config.input_dim,
            context_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_stages=config.num_stages,
            sigma=sb_sigma,
            use_external_drift=True,
        )
        sb_module = SchrodingerBridge(sb_config).to(device)
        sb_module = DDP(sb_module, device_ids=[rank], output_device=rank)

        def external_drift_fn(x_t, t, context, stage_pair_id):
            return model.module.forward_vector_field(
                x_t=x_t, t=t, context=context, stage_pair_id=stage_pair_id, context_tokens=None,
            )
        sb_module.module.set_external_drift(external_drift_fn)

        all_params = list(model.parameters()) + list(sb_module.parameters())
        optimizer = AdamW(all_params, lr=lr, weight_decay=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training
    model.train()
    if sb_module:
        sb_module.train()

    for epoch in range(n_epochs):
        # Set epoch for sampler (required for proper shuffling in DDP)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if hasattr(batch, "neighbors"):
                niche_output = model.module.encode_niche_amici(
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
                niche_output = model.module.encode_niche(
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

            x0 = batch.receiver
            x1 = batch.receiver + 0.1 * torch.randn_like(batch.receiver)
            stage_pair_id = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

            t = torch.rand(x0.shape[0], device=device)
            x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
            u_t = x1 - x0

            v_t = model.module.forward_vector_field(
                x_t=x_t, t=t, context=niche_output.context,
                stage_pair_id=stage_pair_id, context_tokens=niche_output.context_tokens,
            )
            drift_loss = F.mse_loss(v_t, u_t)

            if dynamics_type == "schrodinger_bridge" and sb_module is not None:
                from stagebridge.transition.schrodinger_bridge import schrodinger_bridge_loss
                sb_loss, _ = schrodinger_bridge_loss(
                    x_src=x0, x_tgt=x1, sb_module=sb_module.module,
                    context=niche_output.context, stage_pair_id=stage_pair_id, num_time_samples=4,
                )
                transition_loss = drift_loss + sb_loss
            else:
                transition_loss = drift_loss

            loss_pathway = torch.tensor(0.0, device=device)
            if model.module.pathway_head is not None and batch.pathway_targets is not None:
                pathway_logits = model.module.pathway_head(niche_output.context)
                loss_pathway = F.mse_loss(pathway_logits, batch.pathway_targets)

            loss_proliferation = torch.tensor(0.0, device=device)
            if model.module.proliferation_head is not None and batch.proliferation_target is not None:
                prolif_logit = model.module.proliferation_head(niche_output.context)
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

        if rank == 0:
            log.info(f"  Epoch {epoch+1}/{n_epochs}, loss: {epoch_loss / n_batches:.6f}")

    # Validation on rank 0 only
    if rank == 0:
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in (val_loader or []):
                batch = batch.to(device)
                if hasattr(batch, "neighbors"):
                    niche_output = model.module.encode_niche_amici(
                        receiver=batch.receiver, neighbors=batch.neighbors,
                        distances=batch.distances, neighbor_mask=batch.neighbor_mask,
                        hlca=batch.hlca, luca=batch.luca, pathway=batch.pathway,
                        stats=batch.stats, evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                else:
                    niche_output = model.module.encode_niche(
                        receiver=batch.receiver, ring_cells=batch.ring_cells,
                        ring_masks=batch.ring_masks, hlca=batch.hlca, luca=batch.luca,
                        pathway=batch.pathway, stats=batch.stats,
                        evolution_features=batch.evolution_features,
                        return_reconstruction=True,
                    )
                ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)
                val_loss += ssl_loss.item()
                n_val += 1

        final_loss = val_loss / max(n_val, 1) if n_val > 0 else epoch_loss / max(n_batches, 1)
        return_dict["val_loss"] = final_loss

    cleanup_ddp()


def run_hpo_ddp(
    data_dir: Path,
    output_dir: Path,
    n_trials: int = 30,
    n_epochs_per_trial: int = 15,
    batch_size: int = 64,
    seed: int = 42,
    world_size: int = 4,
    storage: str | None = None,
    study_name: str | None = None,
    gw_checkpoint_path: str | None = None,
):
    """Run HPO with DDP training for each trial."""

    if not OPTUNA_AVAILABLE:
        log.error("Optuna not available")
        return None, {}

    torch.manual_seed(seed)
    np.random.seed(seed)

    def objective(trial: Trial) -> float:
        # Sample hyperparameters
        fusion_options = ["concat", "learned_gw"]
        if gw_checkpoint_path and Path(gw_checkpoint_path).exists():
            fusion_options.append("precompute_gw")

        gw_fusion_type = trial.suggest_categorical("gw_fusion_type", fusion_options)
        use_gw = gw_fusion_type != "concat"

        trial_params = {
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

        # Shared dict for getting result from rank 0
        manager = mp.Manager()
        return_dict = manager.dict()

        # Spawn DDP processes
        mp.spawn(
            train_trial_ddp,
            args=(world_size, trial_params, data_dir, n_epochs_per_trial, batch_size, gw_checkpoint_path, return_dict),
            nprocs=world_size,
            join=True,
        )

        val_loss = return_dict.get("val_loss", float("inf"))
        log.info(f"Trial {trial.number} finished with val_loss: {val_loss:.6f}")
        return val_loss

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

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_params = study.best_params
    log.info(f"Best params: {best_params}")
    log.info(f"Best value: {study.best_value:.6f}")

    return study, best_params


def main():
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge with DDP")
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
    log.info("StageBridge HPO with DDP")
    log.info("=" * 60)
    log.info(f"  Data: {args.data_dir}")
    log.info(f"  World size: {args.world_size}")
    log.info(f"  Trials: {args.n_trials}")
    log.info("=" * 60)

    study, best_params = run_hpo_ddp(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
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
    mp.set_start_method("spawn", force=True)
    main()
