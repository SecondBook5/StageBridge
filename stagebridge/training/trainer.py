"""StageBridge trainer with OT-CFM flow matching.

Trains the receiver-centered niche model using Optimal Transport
Conditional Flow Matching for stage transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from stagebridge.loaders.dataset import NicheBatch
from stagebridge.models.stagebridge import StageBridge, StageBridgeConfig
from stagebridge.training.checkpoint import CheckpointManager
from stagebridge.training.metrics import MetricsLogger
from stagebridge.training.scheduler import create_lr_scheduler
from stagebridge.training.distributed import is_main_process


@dataclass
class TrainerConfig:
    """Configuration for StageBridge two-stage training.

    STAGE 1 (SSL): Masked receiver reconstruction from niche context
    STAGE 2 (Transition): OT-CFM flow matching for stage transitions

    The two-stage approach lets the encoder learn rich niche representations
    before fine-tuning for the downstream flow matching task.
    """

    # Directories
    output_dir: Path = field(default_factory=lambda: Path("runs/stagebridge"))
    run_name: str = ""  # Empty = save directly to output_dir (no subdirectory)

    # Two-stage training
    ssl_epochs: int = 50
    transition_epochs: int = 100
    freeze_encoder: bool = False

    # Learning rate
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    transition_lr_factor: float = 0.1

    # SSL loss weights
    ssl_reconstruction_weight: float = 1.0
    ssl_entropy_weight: float = 0.01
    ssl_value_l1_weight: float = 0.01

    # OT-CFM
    ot_epsilon: float = 0.05
    sinkhorn_iters: int = 80
    num_ot_pairs: int = 512
    use_ot: bool = True
    sigma: float = 0.0

    # Transition loss weights
    flow_matching_weight: float = 1.0
    entropy_weight: float = 0.01
    multihop_weight: float = 0.1
    pathway_weight: float = 0.1
    proliferation_weight: float = 0.1

    # Checkpointing
    checkpoint_every: int = 10
    keep_top_k: int = 3
    eval_every: int = 1

    # Hardware
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    accumulation_steps: int = 1

    # Validation
    strict_gradient_check: bool = True

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_enabled: bool = True

    # Stage transitions to train
    stage_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_epochs(self) -> int:
        """Total epochs (for backward compatibility)."""
        return self.ssl_epochs + self.transition_epochs

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

        if not self.stage_pairs:
            self.stage_pairs = [
                (0, 1),
                (1, 2),
                (0, 2),
            ]


class StageBridgeTrainer:
    """Two-stage trainer for StageBridge model.

    STAGE 1 (SSL): Masked receiver reconstruction from niche context
        - Trains the niche encoder to produce representations that can
          reconstruct the receiver from its neighborhood context
        - Uses SetTransformerRefiner to let context tokens interact

    STAGE 2 (Transition): OT-CFM flow matching for stage transitions
        - Trains the drift head to predict velocity fields
        - Uses Sinkhorn OT to pair cells across stages
        - Optional: freeze encoder to test SSL representation quality

    Handles:
    - Two-stage training (SSL → Transition)
    - Encoder freezing for ablation
    - OT coupling computation
    - Early stopping
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: StageBridge,
        config: TrainerConfig,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

        self.model.to(self.device)

        # If run_name is empty or ".", use output_dir directly
        if config.run_name and config.run_name != ".":
            run_dir = config.output_dir / config.run_name
        else:
            run_dir = config.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        self.checkpoint_manager = CheckpointManager(
            run_dir / "checkpoints",
            keep_top_k=config.keep_top_k,
        )
        self.metrics_logger = MetricsLogger(run_dir / "logs")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scaler = GradScaler("cuda") if config.mixed_precision else None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0
        self._gradient_flow_verified = False
        self._transition_gradient_verified = False
        self._current_phase = "ssl"

    def _print_training_info(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> None:
        """Print training configuration at startup."""
        print(f"\n{'=' * 60}")
        print("StageBridge Training")
        print(f"{'=' * 60}")

        # Device info
        print(f"\nDevice: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"  Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")

        # Data info
        print(f"\nData:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")

        # Model info
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel:")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")
        print(f"  Hidden dim: {self.model.config.hidden_dim}")
        print(f"  Num heads: {self.model.config.num_heads}")
        print(f"  GW fusion: {self.model.config.use_gw_fusion}")

        # Training config
        print(f"\nTraining:")
        print(f"  SSL epochs: {self.config.ssl_epochs}")
        print(f"  Transition epochs: {self.config.transition_epochs}")
        print(f"  Learning rate: {self.config.learning_rate:.2e}")
        print(f"  Weight decay: {self.config.weight_decay:.2e}")
        print(f"  Mixed precision: {self.config.mixed_precision}")
        print(f"  Gradient clip: {self.config.gradient_clip}")

        # Output
        print(f"\nOutput: {self.run_dir}")
        print(f"{'=' * 60}\n")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        resume_from: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run full two-stage training.

        Args:
            train_loader: Training DataLoader yielding NicheBatch
            val_loader: Optional validation DataLoader
            resume_from: Optional checkpoint path to resume from

        Returns:
            Final metrics summary with both SSL and transition metrics
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        # Print training configuration
        if is_main_process():
            self._print_training_info(train_loader, val_loader)

        summary = {"ssl": {}, "transition": {}}

        # STAGE 1: SSL Pretraining
        # Skip if resuming from transition phase (current_epoch >= ssl_epochs)
        ssl_already_complete = self.current_epoch >= self.config.ssl_epochs
        if self.config.ssl_epochs > 0 and not ssl_already_complete:
            if is_main_process():
                print(f"\n{'=' * 60}")
                print(f"STAGE 1: SSL Pretraining ({self.config.ssl_epochs} epochs)")
                print("Objective: Masked receiver reconstruction from niche context")
                print(f"{'=' * 60}\n")

            ssl_summary = self._train_ssl(train_loader, val_loader)
            summary["ssl"] = ssl_summary

            # Save SSL checkpoint
            self._save_ssl_checkpoint()
        elif ssl_already_complete and is_main_process():
            print(f"\nSkipping SSL (already complete, resuming from epoch {self.current_epoch})")

        # STAGE 2: Transition Model
        if self.config.transition_epochs > 0:
            if is_main_process():
                print(f"\n{'=' * 60}")
                print(f"STAGE 2: Transition Model ({self.config.transition_epochs} epochs)")
                print("Objective: Learn stage transition dynamics (OT-CFM flow)")
                if self.config.freeze_encoder:
                    print("Mode: FROZEN encoder (SSL representation transfer test)")
                else:
                    print("Mode: Fine-tuning (end-to-end optimization)")
                print(f"{'=' * 60}\n")

            # Optionally freeze encoder
            if self.config.freeze_encoder:
                self._freeze_encoder()

            # Reset optimizer for transition phase
            self._reset_optimizer_for_transition()

            transition_summary = self._train_transition(train_loader, val_loader)
            summary["transition"] = transition_summary

        # Save final checkpoint
        self.checkpoint_manager.save_final(
            model=self.model,
            config=self._get_config_dict(),
            metrics=self.metrics_logger.get_summary(),
        )
        self.metrics_logger.save()

        return summary

    def _train_ssl(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> dict[str, Any]:
        """Run SSL pretraining phase."""
        self._current_phase = "ssl"
        self._gradient_flow_verified = False

        # Support resuming: skip already-completed SSL epochs
        start_epoch = self.current_epoch if self.current_epoch < self.config.ssl_epochs else 0

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.config.ssl_epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
        )

        # Fast-forward scheduler to resume position
        if start_epoch > 0:
            for _ in range(start_epoch):
                self.scheduler.step()
            if is_main_process():
                print(f"Resuming SSL from epoch {start_epoch}")

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.config.ssl_epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch_ssl(train_loader, epoch)

            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self._validate_ssl(val_loader)

            epoch_metrics = {**train_metrics, **val_metrics, "phase": "ssl"}
            self.metrics_logger.end_epoch(epoch)

            is_best = False
            if "val_loss" in val_metrics:
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    is_best = True
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % self.config.checkpoint_every == 0 or is_best:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    config=self._get_config_dict(),
                    is_best=is_best,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            if is_main_process():
                self._log_epoch(epoch, epoch_metrics, phase="SSL")

            # Early stopping
            if (self.config.early_stopping_enabled and
                epochs_without_improvement >= self.config.early_stopping_patience):
                if is_main_process():
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": self.current_epoch,
        }

    def _train_transition(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> dict[str, Any]:
        """Run transition model training phase."""
        self._current_phase = "transition"
        self._transition_gradient_verified = False

        # Support resuming: skip already-completed transition epochs
        # current_epoch is global (includes SSL epochs), so subtract ssl_epochs to get transition epoch
        if self.current_epoch >= self.config.ssl_epochs:
            start_epoch = self.current_epoch - self.config.ssl_epochs
        else:
            start_epoch = 0

        self.scheduler = create_lr_scheduler(
            self.optimizer,
            self.config.transition_epochs,
            warmup_epochs=min(2, self.config.warmup_epochs),
            min_lr=self.config.min_lr,
        )

        # Fast-forward scheduler to resume position
        if start_epoch > 0:
            for _ in range(start_epoch):
                self.scheduler.step()
            if is_main_process():
                print(f"Resuming transition from epoch {start_epoch}")

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(start_epoch, self.config.transition_epochs):
            global_epoch = self.config.ssl_epochs + epoch
            self.current_epoch = global_epoch

            train_metrics = self._train_epoch_transition(train_loader, epoch)

            val_metrics = {}
            if val_loader is not None and (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self._validate_transition(val_loader)

            epoch_metrics = {**train_metrics, **val_metrics, "phase": "transition"}
            self.metrics_logger.end_epoch(global_epoch)

            is_best = False
            if "val_loss" in val_metrics:
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    is_best = True
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % self.config.checkpoint_every == 0 or is_best:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=global_epoch,
                    metrics=epoch_metrics,
                    config=self._get_config_dict(),
                    is_best=is_best,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            if is_main_process():
                self._log_epoch(epoch, epoch_metrics, phase="Trans")

            # Early stopping
            if (self.config.early_stopping_enabled and
                epochs_without_improvement >= self.config.early_stopping_patience):
                if is_main_process():
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": self.current_epoch,
        }

    def _freeze_encoder(self):
        """Freeze encoder components for ablation testing."""
        frozen_count = 0
        encoder_components = [
            "niche_encoder",
            "context_refiner",
            "niche_tokenizer",  # Learned ring pooling
        ]
        for name in encoder_components:
            if hasattr(self.model, name):
                component = getattr(self.model, name)
                if component is not None:
                    for param in component.parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()

        if is_main_process():
            print(f"Froze {frozen_count:,} encoder parameters")

    def _reset_optimizer_for_transition(self):
        """Reset optimizer for transition phase with lower learning rate."""
        lr = self.config.learning_rate * self.config.transition_lr_factor
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=self.config.weight_decay,
        )

        if is_main_process():
            n_trainable = sum(p.numel() for p in trainable_params)
            print(f"Transition phase: lr={lr:.2e}, trainable params={n_trainable:,}")

    def _save_ssl_checkpoint(self):
        """Save checkpoint after SSL phase completes."""
        if not is_main_process():
            return

        ssl_path = self.run_dir / "checkpoints" / "ssl_pretrained.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": self.config.ssl_epochs,
            "phase": "ssl_complete",
        }, ssl_path)

        if is_main_process():
            print(f"SSL pretraining complete. Checkpoint: {ssl_path}")

    def _train_epoch_ssl(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train SSL for one epoch (masked receiver reconstruction)."""
        self.model.train()

        epoch_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_entropy_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[SSL] E{epoch}",
            disable=not is_main_process(),
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            loss, metrics = self._ssl_step(batch)

            if self.config.accumulation_steps > 1:
                loss = loss / self.config.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if not self._gradient_flow_verified:
                if self.config.strict_gradient_check:
                    self._verify_gradient_flow(phase="ssl")
                self._gradient_flow_verified = True

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            epoch_loss += metrics["loss"]
            epoch_reconstruction_loss += metrics["loss_reconstruction"]
            epoch_entropy_loss += metrics.get("loss_entropy", 0.0)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            self.metrics_logger.log("train_loss", metrics["loss"])
            self.metrics_logger.log("train_loss_reconstruction", metrics["loss_reconstruction"])

        return {
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_loss_reconstruction": epoch_reconstruction_loss / max(n_batches, 1),
            "train_loss_entropy": epoch_entropy_loss / max(n_batches, 1),
        }

    def _ssl_step(self, batch: NicheBatch) -> tuple[Tensor, dict[str, float]]:
        """Single SSL training step: masked receiver reconstruction."""
        amp_context = torch.amp.autocast("cuda") if self.config.mixed_precision and self.device.type == "cuda" else nullcontext()
        with amp_context:
            niche_output = self.model.encode_niche(
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

            if niche_output.receiver_reconstruction is not None:
                loss_reconstruction = F.mse_loss(
                    niche_output.receiver_reconstruction,
                    batch.receiver,
                )
            else:
                raise RuntimeError(
                    "SSL requires reconstruction head. Set use_reconstruction_head=True "
                    "in NicheEncoderConfig or model config."
                )

            loss = self.config.ssl_reconstruction_weight * loss_reconstruction

            loss_entropy = torch.tensor(0.0, device=self.device)
            if niche_output.entropy_loss is not None and self.config.ssl_entropy_weight > 0:
                loss_entropy = niche_output.entropy_loss
                loss = loss + self.config.ssl_entropy_weight * loss_entropy

            loss_value_l1 = torch.tensor(0.0, device=self.device)
            if niche_output.value_l1_loss is not None and self.config.ssl_value_l1_weight > 0:
                loss_value_l1 = niche_output.value_l1_loss
                loss = loss + self.config.ssl_value_l1_weight * loss_value_l1

        metrics = {
            "loss": loss.item(),
            "loss_reconstruction": loss_reconstruction.item(),
            "loss_entropy": loss_entropy.item() if torch.is_tensor(loss_entropy) else loss_entropy,
            "loss_value_l1": loss_value_l1.item() if torch.is_tensor(loss_value_l1) else loss_value_l1,
        }

        return loss, metrics

    @torch.no_grad()
    def _validate_ssl(self, val_loader: DataLoader) -> dict[str, float]:
        """Run SSL validation loop."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            niche_output = self.model.encode_niche(
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

            loss = F.mse_loss(niche_output.receiver_reconstruction, batch.receiver)

            total_loss += loss.item()
            n_batches += 1

            self.metrics_logger.log("val_loss", loss.item())

        return {"val_loss": total_loss / max(n_batches, 1)}

    def _train_epoch_transition(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Train transition model for one epoch (OT-CFM flow matching)."""
        self.model.train()

        epoch_loss = 0.0
        epoch_fm_loss = 0.0
        epoch_entropy_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[Trans] E{epoch}",
            disable=not is_main_process(),
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            loss, metrics = self._transition_step(batch)

            if self.config.accumulation_steps > 1:
                loss = loss / self.config.accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if not self._transition_gradient_verified:
                if self.config.strict_gradient_check:
                    self._verify_gradient_flow(phase="transition")
                self._transition_gradient_verified = True

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.gradient_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.gradient_clip,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            epoch_loss += metrics["loss"]
            epoch_fm_loss += metrics["loss_fm"]
            epoch_entropy_loss += metrics.get("loss_entropy", 0.0)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            self.metrics_logger.log("train_loss", metrics["loss"])
            self.metrics_logger.log("train_loss_fm", metrics["loss_fm"])

        return {
            "train_loss": epoch_loss / max(n_batches, 1),
            "train_loss_fm": epoch_fm_loss / max(n_batches, 1),
            "train_loss_entropy": epoch_entropy_loss / max(n_batches, 1),
        }

    def _transition_step(self, batch: NicheBatch) -> tuple[Tensor, dict[str, float]]:
        """Single transition training step (OT-CFM flow matching)."""
        amp_context = torch.amp.autocast("cuda") if self.config.mixed_precision and self.device.type == "cuda" else nullcontext()
        with amp_context:
            stage_src, stage_tgt = self._sample_stage_pair(batch)

            niche_output = self.model.encode_niche(
                receiver=batch.receiver,
                ring_cells=batch.ring_cells,
                ring_masks=batch.ring_masks,
                hlca=batch.hlca,
                luca=batch.luca,
                pathway=batch.pathway,
                stats=batch.stats,
                evolution_features=batch.evolution_features,
            )

            context = niche_output.context
            context_tokens = niche_output.context_tokens

            x0 = batch.receiver
            x1 = self._sample_targets(batch, stage_tgt)

            loss_fm, coupling = self._flow_matching_loss(
                x0=x0,
                x1=x1,
                context=context,
                context_tokens=context_tokens,
                stage_src=stage_src,
                stage_tgt=stage_tgt,
            )

            loss = self.config.flow_matching_weight * loss_fm

            loss_entropy = torch.tensor(0.0, device=self.device)
            if niche_output.entropy_loss is not None and self.config.entropy_weight > 0:
                loss_entropy = niche_output.entropy_loss
                loss = loss + self.config.entropy_weight * loss_entropy

            # Auxiliary biological losses (if targets available)
            loss_pathway = torch.tensor(0.0, device=self.device)
            loss_proliferation = torch.tensor(0.0, device=self.device)

            if self.model.pathway_head is not None and self.config.pathway_weight > 0:
                pathway_logits = self.model.pathway_head(context)
                if hasattr(batch, "pathway_targets") and batch.pathway_targets is not None:
                    loss_pathway = F.mse_loss(pathway_logits, batch.pathway_targets)
                    loss = loss + self.config.pathway_weight * loss_pathway

            if self.model.proliferation_head is not None and self.config.proliferation_weight > 0:
                proliferation_logit = self.model.proliferation_head(context)
                if hasattr(batch, "proliferation_targets") and batch.proliferation_target is not None:
                    loss_proliferation = F.binary_cross_entropy_with_logits(
                        proliferation_logit.squeeze(-1),
                        batch.proliferation_target.float(),
                    )
                    loss = loss + self.config.proliferation_weight * loss_proliferation

        metrics = {
            "loss": loss.item(),
            "loss_fm": loss_fm.item(),
            "loss_entropy": loss_entropy.item() if torch.is_tensor(loss_entropy) else loss_entropy,
            "loss_pathway": loss_pathway.item() if torch.is_tensor(loss_pathway) else loss_pathway,
            "loss_proliferation": loss_proliferation.item() if torch.is_tensor(loss_proliferation) else loss_proliferation,
        }

        return loss, metrics

    @torch.no_grad()
    def _validate_transition(self, val_loader: DataLoader) -> dict[str, float]:
        """Run transition validation loop."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device)

            stage_src, stage_tgt = self._sample_stage_pair(batch)

            niche_output = self.model.encode_niche(
                receiver=batch.receiver,
                ring_cells=batch.ring_cells,
                ring_masks=batch.ring_masks,
                hlca=batch.hlca,
                luca=batch.luca,
                pathway=batch.pathway,
                stats=batch.stats,
                evolution_features=batch.evolution_features,
            )

            x0 = batch.receiver
            x1 = self._sample_targets(batch, stage_tgt)

            loss_fm, _ = self._flow_matching_loss(
                x0=x0,
                x1=x1,
                context=niche_output.context,
                context_tokens=niche_output.context_tokens,
                stage_src=stage_src,
                stage_tgt=stage_tgt,
            )

            total_loss += loss_fm.item()
            n_batches += 1

            self.metrics_logger.log("val_loss", loss_fm.item())

        return {"val_loss": total_loss / max(n_batches, 1)}

    def _flow_matching_loss(
        self,
        x0: Tensor,
        x1: Tensor,
        context: Tensor,
        context_tokens: Tensor | None,
        stage_src: int,
        stage_tgt: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute OT-CFM flow matching loss.

        Args:
            x0: [B, D] source states
            x1: [B, D] target states
            context: [B, C] niche context
            context_tokens: [B, K, C] context tokens for cross-attention
            stage_src: Source stage index
            stage_tgt: Target stage index

        Returns:
            (loss, coupling)
        """
        b = x0.shape[0]

        if self.config.use_ot:
            coupling = self._sinkhorn_coupling(x0, x1)
            src_idx, tgt_idx = self._sample_from_coupling(coupling, self.config.num_ot_pairs)
        else:
            coupling = torch.ones(b, b, device=self.device) / (b * b)
            src_idx = torch.randint(0, b, (self.config.num_ot_pairs,), device=self.device)
            tgt_idx = torch.randint(0, b, (self.config.num_ot_pairs,), device=self.device)

        x_i = x0[src_idx]
        y_j = x1[tgt_idx]
        ctx = context[src_idx]
        ctx_tokens = context_tokens[src_idx] if context_tokens is not None else None

        t = torch.rand(self.config.num_ot_pairs, device=self.device)

        x_t = (1 - t.unsqueeze(1)) * x_i + t.unsqueeze(1) * y_j
        if self.config.sigma > 0:
            noise_scale = self.config.sigma * (t * (1 - t)).sqrt().unsqueeze(1)
            x_t = x_t + noise_scale * torch.randn_like(x_t)

        u_t = y_j - x_i

        stage_pair_id = self.model.encode_stage_pair_tensor(
            stage_src, stage_tgt, self.config.num_ot_pairs, self.device
        )

        v_t = self.model.forward_vector_field(
            x_t=x_t,
            t=t,
            context=ctx,
            stage_pair_id=stage_pair_id,
            context_tokens=ctx_tokens,
        )

        loss = F.mse_loss(v_t, u_t)

        return loss, coupling

    def _sinkhorn_coupling(
        self,
        x_src: Tensor,
        x_tgt: Tensor,
    ) -> Tensor:
        """Compute Sinkhorn OT coupling."""
        n, m = x_src.shape[0], x_tgt.shape[0]
        dtype = x_src.dtype

        x_src_64 = x_src.double()
        x_tgt_64 = x_tgt.double()

        cost = torch.cdist(x_src_64, x_tgt_64, p=2).pow(2)

        log_a = torch.full((n,), -torch.log(torch.tensor(n, dtype=torch.float64)), device=self.device)
        log_b = torch.full((m,), -torch.log(torch.tensor(m, dtype=torch.float64)), device=self.device)

        log_K = -cost / self.config.ot_epsilon

        log_u = torch.zeros(n, dtype=torch.float64, device=self.device)
        log_v = torch.zeros(m, dtype=torch.float64, device=self.device)

        for _ in range(self.config.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)

        log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        pi = torch.exp(log_pi).to(dtype)

        return pi

    def _sample_from_coupling(
        self,
        coupling: Tensor,
        num_pairs: int,
    ) -> tuple[Tensor, Tensor]:
        """Sample index pairs from coupling matrix."""
        n, m = coupling.shape
        probs = coupling.reshape(-1)
        probs = probs.clamp_min(0)
        probs = probs / probs.sum().clamp_min(1e-12)

        sampled = torch.multinomial(probs, num_pairs, replacement=True)
        src_idx = sampled // m
        tgt_idx = sampled % m

        return src_idx, tgt_idx

    def _sample_stage_pair(self, batch: NicheBatch) -> tuple[int, int]:
        """Sample a stage transition pair for this batch."""
        if self.config.stage_pairs:
            idx = torch.randint(len(self.config.stage_pairs), (1,)).item()
            return self.config.stage_pairs[idx]

        stages = batch.stage_idx.unique().tolist()
        if len(stages) < 2:
            return (stages[0], stages[0])
        src_idx = torch.randint(len(stages) - 1, (1,)).item()
        return (stages[src_idx], stages[src_idx + 1])

    def _sample_targets(self, batch: NicheBatch, target_stage: int) -> Tensor:
        """Sample target states for flow matching from target stage population.

        Properly samples from cells at the target stage rather than adding noise
        to source cells. This is essential for learning meaningful transitions.
        """
        target_mask = batch.stage_idx == target_stage
        n_targets = target_mask.sum().item()

        if n_targets > 0:
            target_receivers = batch.receiver[target_mask]
            sample_idx = torch.randint(
                n_targets, (batch.receiver.shape[0],), device=batch.receiver.device
            )
            return target_receivers[sample_idx]
        else:
            return batch.receiver + 0.1 * torch.randn_like(batch.receiver)

    def _load_checkpoint(self, path: Path | str):
        """Load checkpoint and resume training."""
        checkpoint = CheckpointManager.load(path, self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # ssl_pretrained.pt only has model weights, no optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_epoch = checkpoint["epoch"] + 1
        else:
            # SSL checkpoint - start transition from epoch 0
            if is_main_process():
                print("No optimizer state in checkpoint (SSL pretrained) - starting transition fresh")
            self.current_epoch = self.config.ssl_epochs  # Skip SSL, start transition at epoch 0

        self.metrics_logger.load()

    def _get_config_dict(self) -> dict[str, Any]:
        """Get config as serializable dict."""
        return {
            "model_config": {
                "input_dim": self.model.config.input_dim,
                "hidden_dim": self.model.config.hidden_dim,
                "num_heads": self.model.config.num_heads,
                "num_encoder_layers": self.model.config.num_encoder_layers,
                "max_neighbors": self.model.config.max_neighbors,
                "num_stages": self.model.config.num_stages,
                # GW fusion config (critical for checkpoint loading)
                "use_gw_fusion": self.model.config.use_gw_fusion,
                "gw_output_dim": self.model.config.gw_output_dim,
                "gw_sinkhorn_iters": self.model.config.gw_sinkhorn_iters,
                "gw_sinkhorn_reg": self.model.config.gw_sinkhorn_reg,
                "gw_mode": self.model.config.gw_mode,
                # Other model config
                "use_learned_ring_pooling": self.model.config.use_learned_ring_pooling,
                "use_context_refiner": self.model.config.use_context_refiner,
                "use_cross_attn_drift": self.model.config.use_cross_attn_drift,
            },
            "trainer_config": {
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "ot_epsilon": self.config.ot_epsilon,
                "num_ot_pairs": self.config.num_ot_pairs,
                "use_ot": self.config.use_ot,
                "sigma": self.config.sigma,
            },
        }

    def _verify_gradient_flow(self, phase: str = "ssl"):
        """CONTRACT: Verify gradients flow through critical components.

        Simplified check - just verify SOME parameters in each critical module
        receive gradients, rather than checking every single parameter.

        Args:
            phase: Training phase - "ssl" or "transition".
        """
        # Define key modules to check (not all subparameters)
        if phase == "ssl":
            # SSL only trains reconstruction path
            modules_to_check = ["niche_tokenizer.token_proj"]
        else:  # transition
            # Transition should train drift head
            modules_to_check = ["drift_head"]

        # Check that at least one parameter per module has gradients
        for module_name in modules_to_check:
            found_grad = False
            for name, p in self.model.named_parameters():
                if module_name in name:
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        found_grad = True
                        break

            if not found_grad:
                raise AssertionError(
                    f"GRADIENT FLOW CONTRACT VIOLATED: No gradients flowing to {module_name} "
                    f"during {phase} phase. Check model architecture."
                )

    def _log_epoch(self, epoch: int, metrics: dict[str, float], phase: str = ""):
        """Log epoch summary."""
        prefix = f"[{phase}] " if phase else ""
        parts = [f"{prefix}Epoch {epoch}"]
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))


def train_stagebridge(
    data_dir: Path | str,
    output_dir: Path | str,
    model_config: StageBridgeConfig | None = None,
    trainer_config: TrainerConfig | None = None,
    fold_idx: int = 0,
    device: str = "cuda",
    resume_from: Path | str | None = None,
) -> dict[str, Any]:
    """Convenience function to train StageBridge.

    Args:
        data_dir: Directory with cells.parquet, neighborhoods.parquet
        output_dir: Output directory for checkpoints and logs
        model_config: Model configuration
        trainer_config: Trainer configuration
        fold_idx: Cross-validation fold index
        device: Device to train on
        resume_from: Optional checkpoint path to resume from

    Returns:
        Training summary
    """
    from stagebridge.loaders import create_dataloaders

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if model_config is None:
        model_config = StageBridgeConfig()

    if trainer_config is None:
        trainer_config = TrainerConfig(output_dir=output_dir)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        fold_idx=fold_idx,
        batch_size=64,
        num_workers=4,
    )

    # Detect evolution_dim from data (may differ from contracts.EVOLUTION_DIM)
    sample_batch = next(iter(train_loader))
    if sample_batch.evolution_features is not None and model_config.use_evolution_branch:
        detected_dim = sample_batch.evolution_features.shape[-1]
        if detected_dim != model_config.evolution_dim:
            print(f"Detected evolution_dim={detected_dim} from data (config had {model_config.evolution_dim})")
            model_config = StageBridgeConfig(
                **{k: v for k, v in model_config.__dict__.items() if k != 'evolution_dim'},
                evolution_dim=detected_dim,
            )

    model = StageBridge(model_config)

    trainer = StageBridgeTrainer(
        model=model,
        config=trainer_config,
        device=device,
    )

    return trainer.train(train_loader, val_loader, resume_from=resume_from)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train StageBridge model")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--transition-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    # GW fusion args
    parser.add_argument("--use-gw-fusion", action="store_true", default=True,
                        help="Enable Gromov-Wasserstein atlas fusion (default: True)")
    parser.add_argument("--no-gw-fusion", dest="use_gw_fusion", action="store_false",
                        help="Disable GW fusion, use naive concat")
    parser.add_argument("--gw-output-dim", type=int, default=64)
    parser.add_argument("--gw-mode", choices=["barycentric", "project_to_hlca", "project_to_luca"],
                        default="barycentric")
    # HPO params (overrides defaults with optimized values)
    parser.add_argument("--hpo-params", type=Path, default=None,
                        help="Path to best_params.json from HPO (overrides CLI args)")
    # Resume from checkpoint
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from (or 'auto' to find best)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load HPO params if provided (overrides CLI defaults)
    hpo = {}
    if args.hpo_params and args.hpo_params.exists():
        with open(args.hpo_params) as f:
            hpo = json.load(f)
        print(f"Loaded HPO params: {hpo}")

    # Detect evolution_dim from data before creating config
    from stagebridge.loaders import create_dataloaders as _create_dl
    _train_loader, _, _ = _create_dl(args.data_dir, fold_idx=args.fold_idx, batch_size=64, num_workers=0)
    _sample = next(iter(_train_loader))
    evolution_dim = _sample.evolution_features.shape[-1] if _sample.evolution_features is not None else 0
    print(f"Detected evolution_dim={evolution_dim} from data")
    del _train_loader, _sample, _create_dl

    # Model config from HPO or CLI
    model_config = StageBridgeConfig(
        hidden_dim=hpo.get("hidden_dim", 128),
        num_heads=hpo.get("num_heads", 4),
        dropout=hpo.get("dropout", 0.1),
        use_gw_fusion=hpo.get("use_gw_fusion", args.use_gw_fusion),
        gw_output_dim=hpo.get("gw_output_dim", args.gw_output_dim),
        gw_sinkhorn_reg=hpo.get("gw_sinkhorn_reg", 0.1),
        gw_mode=args.gw_mode,
        use_learned_ring_pooling=True,
        use_context_refiner=True,
        use_cross_attn_drift=True,
        use_evolution_branch=evolution_dim > 0,
        evolution_dim=evolution_dim,
    )

    trainer_config = TrainerConfig(
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
        learning_rate=hpo.get("lr", args.learning_rate),
        output_dir=args.output_dir,
    )

    # Auto-detect checkpoint to resume from
    resume_from = args.resume

    # Check if training already completed (final_checkpoint exists = training finished)
    final_ckpt = args.output_dir / "checkpoints" / "final_checkpoint.pt"
    if final_ckpt.exists():
        print(f"Training already complete: {final_ckpt}")
        # Load checkpoint and extract summary
        import torch
        ckpt = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        metrics = ckpt.get("metrics", {})
        result = {
            "ssl": {
                "best_val_loss": metrics.get("ssl_val_loss") or metrics.get("val_loss"),
                "final_epoch": ckpt.get("epoch"),
                "status": "completed"
            },
            "transition": {
                "best_val_loss": metrics.get("val_loss"),
                "final_epoch": ckpt.get("epoch"),
                "status": "completed"
            },
            "checkpoint_source": "final_checkpoint.pt",
        }
        with open(args.output_dir / "training_summary.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote training_summary.json from final_checkpoint.pt")
        exit(0)

    if resume_from is None:
        # Auto-resume: check for existing best checkpoint
        best_ckpt = args.output_dir / "checkpoints" / "best_checkpoint.pt"
        ssl_ckpt = args.output_dir / "checkpoints" / "ssl_pretrained.pt"
        if best_ckpt.exists():
            resume_from = best_ckpt
            print(f"Auto-resuming from {resume_from}")
        elif ssl_ckpt.exists():
            # SSL completed but transition crashed - resume from SSL checkpoint
            # Set current_epoch to ssl_epochs-1 so transition starts at epoch 0
            resume_from = ssl_ckpt
            print(f"Auto-resuming from SSL checkpoint: {resume_from}")

    result = train_stagebridge(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_config=model_config,
        trainer_config=trainer_config,
        fold_idx=args.fold_idx,
        resume_from=resume_from,
    )

    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(result, f, indent=2)
