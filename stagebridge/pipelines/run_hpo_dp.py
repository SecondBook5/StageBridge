#!/usr/bin/env python3
"""HPO with DataParallel - single process, multi-GPU via DataParallel wrapper.

This is identical to run_hpo.py but uses DataParallel to utilize all GPUs
within a single process, ensuring TPE learns from all trials sequentially.
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


class HPOTrainingWrapperAMICI(nn.Module):
    """Wrapper that combines model + training step for DataParallel (AMICI format)."""

    def __init__(
        self,
        model: nn.Module,
        sb_module: nn.Module | None,
        dynamics_type: str,
        ssl_weight: float,
        pathway_weight: float,
        proliferation_weight: float,
    ):
        super().__init__()
        self.model = model
        self.sb_module = sb_module
        self.dynamics_type = dynamics_type
        self.ssl_weight = ssl_weight
        self.pathway_weight = pathway_weight
        self.proliferation_weight = proliferation_weight

    def forward(
        self,
        receiver: torch.Tensor,
        neighbors: torch.Tensor,
        distances: torch.Tensor,
        neighbor_mask: torch.Tensor,
        hlca: torch.Tensor,
        luca: torch.Tensor,
        pathway: torch.Tensor,
        stats: torch.Tensor,
        evolution_features: torch.Tensor | None,
        pathway_targets: torch.Tensor | None,
        proliferation_target: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass returns loss per sample (DataParallel will gather)."""

        device = receiver.device

        # Encode niche (AMICI format)
        niche_output = self.model.encode_niche_amici(
            receiver=receiver,
            neighbors=neighbors,
            distances=distances,
            neighbor_mask=neighbor_mask,
            hlca=hlca,
            luca=luca,
            pathway=pathway,
            stats=stats,
            evolution_features=evolution_features,
            return_reconstruction=True,
        )

        # SSL loss: receiver reconstruction
        ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, receiver, reduction='none').mean(dim=-1)

        # Transition loss
        x0 = receiver
        x1 = receiver + 0.1 * torch.randn_like(receiver)
        stage_pair_id = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

        t = torch.rand(x0.shape[0], device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        u_t = x1 - x0

        v_t = self.model.forward_vector_field(
            x_t=x_t,
            t=t,
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            context_tokens=niche_output.context_tokens,
        )
        drift_loss = F.mse_loss(v_t, u_t, reduction='none').mean(dim=-1)

        if self.dynamics_type == "schrodinger_bridge" and self.sb_module is not None:
            from stagebridge.transition.schrodinger_bridge import schrodinger_bridge_loss
            sb_loss, _ = schrodinger_bridge_loss(
                x_src=x0,
                x_tgt=x1,
                sb_module=self.sb_module,
                context=niche_output.context,
                stage_pair_id=stage_pair_id,
                num_time_samples=4,
            )
            transition_loss = drift_loss + sb_loss
        else:
            transition_loss = drift_loss

        # Auxiliary losses
        loss_pathway = torch.zeros_like(ssl_loss)
        if self.model.pathway_head is not None and pathway_targets is not None:
            pathway_logits = self.model.pathway_head(niche_output.context)
            loss_pathway = F.mse_loss(pathway_logits, pathway_targets, reduction='none').mean(dim=-1)

        loss_proliferation = torch.zeros_like(ssl_loss)
        if self.model.proliferation_head is not None and proliferation_target is not None:
            prolif_logit = self.model.proliferation_head(niche_output.context)
            loss_proliferation = F.binary_cross_entropy_with_logits(
                prolif_logit.squeeze(-1), proliferation_target, reduction='none'
            )

        # Total loss per sample
        loss = (
            self.ssl_weight * ssl_loss
            + (1 - self.ssl_weight) * transition_loss
            + self.pathway_weight * loss_pathway
            + self.proliferation_weight * loss_proliferation
        )

        return loss


class HPOTrainingWrapperRing(nn.Module):
    """Wrapper that combines model + training step for DataParallel (Ring format)."""

    def __init__(
        self,
        model: nn.Module,
        sb_module: nn.Module | None,
        dynamics_type: str,
        ssl_weight: float,
        pathway_weight: float,
        proliferation_weight: float,
    ):
        super().__init__()
        self.model = model
        self.sb_module = sb_module
        self.dynamics_type = dynamics_type
        self.ssl_weight = ssl_weight
        self.pathway_weight = pathway_weight
        self.proliferation_weight = proliferation_weight

    def forward(
        self,
        receiver: torch.Tensor,
        ring_cells: torch.Tensor,
        ring_masks: torch.Tensor,
        hlca: torch.Tensor,
        luca: torch.Tensor,
        pathway: torch.Tensor,
        stats: torch.Tensor,
        evolution_features: torch.Tensor | None,
        pathway_targets: torch.Tensor | None,
        proliferation_target: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass returns loss per sample (DataParallel will gather)."""

        device = receiver.device

        # Encode niche (Ring format)
        niche_output = self.model.encode_niche(
            receiver=receiver,
            ring_cells=ring_cells,
            ring_masks=ring_masks,
            hlca=hlca,
            luca=luca,
            pathway=pathway,
            stats=stats,
            evolution_features=evolution_features,
            return_reconstruction=True,
        )

        # SSL loss: receiver reconstruction
        ssl_loss = F.mse_loss(niche_output.receiver_reconstruction, receiver, reduction='none').mean(dim=-1)

        # Transition loss
        x0 = receiver
        x1 = receiver + 0.1 * torch.randn_like(receiver)
        stage_pair_id = torch.zeros(x0.shape[0], dtype=torch.long, device=device)

        t = torch.rand(x0.shape[0], device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        u_t = x1 - x0

        v_t = self.model.forward_vector_field(
            x_t=x_t,
            t=t,
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            context_tokens=niche_output.context_tokens,
        )
        drift_loss = F.mse_loss(v_t, u_t, reduction='none').mean(dim=-1)

        if self.dynamics_type == "schrodinger_bridge" and self.sb_module is not None:
            from stagebridge.transition.schrodinger_bridge import schrodinger_bridge_loss
            sb_loss, _ = schrodinger_bridge_loss(
                x_src=x0,
                x_tgt=x1,
                sb_module=self.sb_module,
                context=niche_output.context,
                stage_pair_id=stage_pair_id,
                num_time_samples=4,
            )
            transition_loss = drift_loss + sb_loss
        else:
            transition_loss = drift_loss

        # Auxiliary losses
        loss_pathway = torch.zeros_like(ssl_loss)
        if self.model.pathway_head is not None and pathway_targets is not None:
            pathway_logits = self.model.pathway_head(niche_output.context)
            loss_pathway = F.mse_loss(pathway_logits, pathway_targets, reduction='none').mean(dim=-1)

        loss_proliferation = torch.zeros_like(ssl_loss)
        if self.model.proliferation_head is not None and proliferation_target is not None:
            prolif_logit = self.model.proliferation_head(niche_output.context)
            loss_proliferation = F.binary_cross_entropy_with_logits(
                prolif_logit.squeeze(-1), proliferation_target, reduction='none'
            )

        # Total loss per sample
        loss = (
            self.ssl_weight * ssl_loss
            + (1 - self.ssl_weight) * transition_loss
            + self.pathway_weight * loss_pathway
            + self.proliferation_weight * loss_proliferation
        )

        return loss


def run_hpo(
    data_dir: Path,
    output_dir: Path,
    n_trials: int = 30,
    n_epochs_per_trial: int = 10,
    batch_size: int = 64,
    seed: int = 42,
    n_jobs: int = 1,
    storage: str | None = None,
    study_name: str | None = None,
    gw_checkpoint_path: str | None = None,
) -> tuple:
    """Run Optuna HPO for StageBridge with DataParallel.

    Args:
        data_dir: Path to data directory with neighborhoods.parquet
        output_dir: Output directory for results
        n_trials: Number of Optuna trials
        n_epochs_per_trial: Training epochs per trial
        batch_size: Batch size (per GPU, will be multiplied by n_gpus)
        seed: Random seed
        n_jobs: Parallel trials (1 = sequential, recommended for proper TPE)
        storage: Optuna storage URL for distributed HPO
        study_name: Optuna study name (for resuming)
        gw_checkpoint_path: Path to precomputed GW alignment (enables pretrained fusion)

    Returns:
        (study, best_params)
    """
    if not OPTUNA_AVAILABLE:
        log.error("Optuna not available")
        return None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    log.info(f"Using {n_gpus} GPUs with DataParallel")

    torch.manual_seed(seed)
    np.random.seed(seed)

    from stagebridge.loaders import create_dataloaders

    log.info("Loading data...")
    # Scale batch size by number of GPUs
    effective_batch_size = batch_size * max(n_gpus, 1)
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, fold_idx=0, batch_size=effective_batch_size, num_workers=4
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

        # Create wrapper and apply DataParallel
        if is_amici_format:
            wrapper = HPOTrainingWrapperAMICI(
                model=model,
                sb_module=sb_module,
                dynamics_type=dynamics_type,
                ssl_weight=ssl_weight,
                pathway_weight=pathway_weight,
                proliferation_weight=proliferation_weight,
            )
        else:
            wrapper = HPOTrainingWrapperRing(
                model=model,
                sb_module=sb_module,
                dynamics_type=dynamics_type,
                ssl_weight=ssl_weight,
                pathway_weight=pathway_weight,
                proliferation_weight=proliferation_weight,
            )

        if n_gpus > 1:
            wrapper = nn.DataParallel(wrapper)

        # Optimizer (include SB params if present)
        if sb_module is not None:
            all_params = list(model.parameters()) + list(sb_module.parameters())
        else:
            all_params = list(model.parameters())
        optimizer = AdamW(all_params, lr=lr, weight_decay=1e-4)

        # Training loop
        wrapper.train()

        for epoch in range(n_epochs_per_trial):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad()

                if is_amici_format:
                    # AMICI format
                    receiver = batch.receiver.to(device)
                    neighbors = batch.neighbors.to(device)
                    distances = batch.distances.to(device)
                    neighbor_mask = batch.neighbor_mask.to(device)
                    hlca = batch.hlca.to(device) if batch.hlca is not None else None
                    luca = batch.luca.to(device) if batch.luca is not None else None
                    pathway = batch.pathway.to(device) if batch.pathway is not None else None
                    stats = batch.stats.to(device) if batch.stats is not None else None
                    evolution_features = batch.evolution_features.to(device) if batch.evolution_features is not None else None
                    pathway_targets = batch.pathway_targets.to(device) if batch.pathway_targets is not None else None
                    proliferation_target = batch.proliferation_target.to(device) if batch.proliferation_target is not None else None

                    # Forward (DataParallel splits across GPUs)
                    loss = wrapper(
                        receiver, neighbors, distances, neighbor_mask,
                        hlca, luca, pathway, stats, evolution_features,
                        pathway_targets, proliferation_target,
                    )
                else:
                    # Ring format
                    receiver = batch.receiver.to(device)
                    ring_cells = batch.ring_cells.to(device)
                    ring_masks = batch.ring_masks.to(device)
                    hlca = batch.hlca.to(device) if batch.hlca is not None else None
                    luca = batch.luca.to(device) if batch.luca is not None else None
                    pathway = batch.pathway.to(device) if batch.pathway is not None else None
                    stats = batch.stats.to(device) if batch.stats is not None else None
                    evolution_features = batch.evolution_features.to(device) if batch.evolution_features is not None else None
                    pathway_targets = batch.pathway_targets.to(device) if batch.pathway_targets is not None else None
                    proliferation_target = batch.proliferation_target.to(device) if batch.proliferation_target is not None else None

                    # Forward (DataParallel splits across GPUs)
                    loss = wrapper(
                        receiver, ring_cells, ring_masks,
                        hlca, luca, pathway, stats, evolution_features,
                        pathway_targets, proliferation_target,
                    )

                # Mean across GPUs/samples
                loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Report for pruning
            trial.report(epoch_loss / n_batches, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            log.info(f"  Trial {trial.number} Epoch {epoch+1}/{n_epochs_per_trial}, loss: {epoch_loss / n_batches:.6f}")

        # Validation
        wrapper.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in (val_loader or []):
                if is_amici_format:
                    receiver = batch.receiver.to(device)
                    neighbors = batch.neighbors.to(device)
                    distances = batch.distances.to(device)
                    neighbor_mask = batch.neighbor_mask.to(device)
                    hlca = batch.hlca.to(device) if batch.hlca is not None else None
                    luca = batch.luca.to(device) if batch.luca is not None else None
                    pathway = batch.pathway.to(device) if batch.pathway is not None else None
                    stats = batch.stats.to(device) if batch.stats is not None else None
                    evolution_features = batch.evolution_features.to(device) if batch.evolution_features is not None else None
                    pathway_targets = batch.pathway_targets.to(device) if batch.pathway_targets is not None else None
                    proliferation_target = batch.proliferation_target.to(device) if batch.proliferation_target is not None else None

                    loss = wrapper(
                        receiver, neighbors, distances, neighbor_mask,
                        hlca, luca, pathway, stats, evolution_features,
                        pathway_targets, proliferation_target,
                    )
                else:
                    receiver = batch.receiver.to(device)
                    ring_cells = batch.ring_cells.to(device)
                    ring_masks = batch.ring_masks.to(device)
                    hlca = batch.hlca.to(device) if batch.hlca is not None else None
                    luca = batch.luca.to(device) if batch.luca is not None else None
                    pathway = batch.pathway.to(device) if batch.pathway is not None else None
                    stats = batch.stats.to(device) if batch.stats is not None else None
                    evolution_features = batch.evolution_features.to(device) if batch.evolution_features is not None else None
                    pathway_targets = batch.pathway_targets.to(device) if batch.pathway_targets is not None else None
                    proliferation_target = batch.proliferation_target.to(device) if batch.proliferation_target is not None else None

                    loss = wrapper(
                        receiver, ring_cells, ring_masks,
                        hlca, luca, pathway, stats, evolution_features,
                        pathway_targets, proliferation_target,
                    )

                val_loss += loss.mean().item()
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
    parser = argparse.ArgumentParser(description="Run HPO for StageBridge with DataParallel")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trials (1 recommended for proper TPE)")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default=None, help="Study name (for resuming)")
    parser.add_argument("--gw-checkpoint", type=str, default=None,
                        help="Path to precomputed GW alignment dir (enables pretrained fusion)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("StageBridge HPO with DataParallel")
    log.info("=" * 60)
    log.info(f"  Data: {args.data_dir}")
    log.info(f"  Output: {args.output_dir}")
    log.info(f"  GPUs: {torch.cuda.device_count()}")
    log.info(f"  Trials: {args.n_trials}")
    log.info(f"  Epochs per trial: {args.n_epochs}")
    log.info("=" * 60)

    study, best_params = run_hpo(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
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
