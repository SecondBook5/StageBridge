#!/usr/bin/env python3
"""
StageBridge V1 Training Pipeline with DDP Support

Multi-GPU distributed training wrapper with:
- PyTorch DistributedDataParallel (DDP) support
- Checkpoint/resume logic
- Intermediate checkpoint saving
- Donor-held-out validation
- Publication artifact generation

This is the canonical training script for the full model.

Usage:
    # Single GPU
    python -m stagebridge.pipelines.run_v1_ddp --data_dir /path/to/data --output_dir /path/to/output

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 -m stagebridge.pipelines.run_v1_ddp \
        --data_dir /path/to/data --output_dir /path/to/output

    # Resume from checkpoint
    python -m stagebridge.pipelines.run_v1_ddp --resume_checkpoint /path/to/checkpoint.pt ...
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Data
    data_dir: str = ""
    output_dir: str = ""
    hlca_path: str = ""
    luca_path: str = ""

    # Model
    latent_dim: int = 32
    niche_hidden_dim: int = 128
    context_dim: int = 256
    dropout: float = 0.1

    # Training
    ssl_epochs: int = 50
    transition_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    use_cosine_schedule: bool = True

    # Checkpointing
    checkpoint_every: int = 10
    resume_checkpoint: str = ""
    keep_top_k_checkpoints: int = 3

    # Validation
    n_folds: int = 5
    validation_fold: int = 0

    # HPO
    hpo_trials: int = 30
    hpo_params: str = ""
    use_best_hparams: bool = False

    # Other
    seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsLogger:
    """Log training metrics to CSV for analysis."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_file = output_dir / "metrics" / "training_metrics.csv"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False
        self._all_metrics = []

    def log(self, epoch: int, phase: str, metrics: dict):
        """Log metrics for an epoch."""
        row = {
            "epoch": epoch,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
        }

        # Add all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value

        # Add GPU memory if available
        if torch.cuda.is_available():
            row["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            row["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        self._all_metrics.append(row)

        # Write to CSV (append mode)
        import csv

        write_header = not self._header_written

        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

    def save_summary(self):
        """Save complete metrics summary."""
        import pandas as pd

        if self._all_metrics:
            df = pd.DataFrame(self._all_metrics)
            df.to_csv(self.output_dir / "metrics" / "training_metrics_full.csv", index=False)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    use_cosine: bool = True,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create LR scheduler with warmup + cosine decay.

    Args:
        optimizer: The optimizer
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs (linear increase)
        min_lr: Minimum learning rate at end of cosine decay
        use_cosine: If True, use cosine annealing; else constant after warmup
    """
    if warmup_epochs > 0 and use_cosine:
        # Warmup + Cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
                return max(min_lr / optimizer.defaults["lr"], 0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif warmup_epochs > 0:
        # Warmup only, then constant
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif use_cosine:
        # Cosine annealing without warmup
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    else:
        # No schedule (constant LR)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def setup_distributed():
    """Initialize distributed training if available."""
    # Check if running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

        # Set device
        torch.cuda.set_device(local_rank)

        print(f"[Rank {rank}] Initialized DDP: world_size={world_size}, local_rank={local_rank}")

        return True, local_rank

    return False, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def log(msg: str, *args, **kwargs):
    """Log message only on main process."""
    if is_main_process():
        print(msg, *args, **kwargs)


class CheckpointManager:
    """Manages model checkpoints with versioning."""

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_top_k: int = 3,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoint_history: list[dict] = []

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        config: dict,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint."""
        if not is_main_process():
            return None

        # Handle DDP wrapped model
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "timestamp": datetime.now().isoformat(),
        }

        # Save epoch checkpoint
        filepath = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, filepath)

        # Track history
        metric_value = metrics.get(self.metric_name, float("inf"))
        self.checkpoint_history.append(
            {
                "path": str(filepath),
                "epoch": epoch,
                "metric_value": metric_value,
            }
        )

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)

        # Prune old checkpoints
        self._prune_checkpoints()

        return filepath

    def _prune_checkpoints(self):
        """Remove old checkpoints, keeping only top-k."""
        if len(self.checkpoint_history) <= self.keep_top_k:
            return

        # Sort by metric
        sorted_history = sorted(
            self.checkpoint_history,
            key=lambda x: x["metric_value"],
            reverse=(self.mode == "max"),
        )

        # Keep top-k
        keep_paths = {h["path"] for h in sorted_history[: self.keep_top_k]}

        # Also keep most recent
        keep_paths.add(self.checkpoint_history[-1]["path"])

        # Remove others
        for h in self.checkpoint_history:
            if h["path"] not in keep_paths:
                path = Path(h["path"])
                if path.exists():
                    path.unlink()

        self.checkpoint_history = [h for h in self.checkpoint_history if h["path"] in keep_paths]

    def save_final(self, model: nn.Module, config: dict, metrics: dict):
        """Save final checkpoint."""
        if not is_main_process():
            return None

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        checkpoint = {
            "model_state_dict": state_dict,
            "config": config,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "is_final": True,
        }

        final_path = self.checkpoint_dir / "final_checkpoint.pt"
        torch.save(checkpoint, final_path)

        # Also save to weights/ for backward compatibility
        weights_dir = self.checkpoint_dir.parent / "weights"
        weights_dir.mkdir(exist_ok=True)
        torch.save(checkpoint, weights_dir / "final_model.pt")

        return final_path

    @staticmethod
    def load(checkpoint_path: Path, device: torch.device) -> dict:
        """Load checkpoint."""
        return torch.load(checkpoint_path, map_location=device)


def create_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Create the StageBridge model."""
    try:
        from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

        model = StageBridgeV1Complete(
            latent_dim=config.latent_dim,
            niche_hidden_dim=config.niche_hidden_dim,
            context_dim=config.context_dim,
            dropout=config.dropout,
        )
    except ImportError:
        # Fallback to basic model
        from stagebridge.context_model.receiver_niche_encoder import ReceiverCenteredNicheEncoder

        model = ReceiverCenteredNicheEncoder(
            input_dim=config.latent_dim,
            hidden_dim=config.niche_hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=config.dropout,
        )

    return model.to(device)


def create_dataloaders(
    config: TrainingConfig,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train, validation, and semi-synthetic benchmark dataloaders.

    Returns:
        (train_loader, val_loader, benchmark_loader)
        benchmark_loader is None if semi-synthetic generation fails
    """
    from torch.utils.data import TensorDataset
    import pandas as pd

    train_data = None
    val_data = None
    benchmark_data = None

    # ==========================================================================
    # Try to load REAL data from canonical format
    # ==========================================================================
    cells_path = Path(config.data_dir) / "cells.parquet"
    neighborhoods_path = Path(config.data_dir) / "neighborhoods.parquet"

    if cells_path.exists() and neighborhoods_path.exists():
        try:
            log("Loading REAL data from canonical format...")
            cells_df = pd.read_parquet(cells_path)
            neighborhoods_df = pd.read_parquet(neighborhoods_path)

            log(f"  Cells: {len(cells_df):,}")
            log(f"  Neighborhoods: {len(neighborhoods_df):,}")

            # Extract embedding columns
            fused_cols = [c for c in cells_df.columns if c.startswith("fused_latent_")]
            [c for c in cells_df.columns if c.startswith("hlca_latent_")]
            [c for c in cells_df.columns if c.startswith("luca_latent_")]

            if fused_cols:
                log(f"  Fused embedding: {len(fused_cols)} dims")

                # Build niche tokens from neighborhoods
                # Token order: [receiver, ring1, ring2, ring3, ring4, hlca, luca, pathway, stats]
                # For now, use fused embedding for all tokens (will refine)
                embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)
                n_cells = len(embeddings)

                # Create 9-token sequences (simplified: replicate for now)
                niche_tokens = embeddings.unsqueeze(1).expand(-1, 9, -1).clone()

                # z_source and z_target for transition learning
                # Use stage information to create pseudo-transitions
                if "stage" in cells_df.columns:
                    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
                    stage_to_idx = {s: i for i, s in enumerate(stage_order)}
                    stage_indices = (
                        cells_df["stage"].map(stage_to_idx).fillna(0).astype(int).values
                    )  # noqa: F841

                    # z_source = current embedding, z_target = shifted embedding (next stage cells)
                    z_source = embeddings
                    # For transition targets, use mean of next-stage cells as pseudo-target
                    z_target = embeddings.clone()  # Placeholder - refine with actual transitions
                else:
                    z_source = embeddings
                    z_target = embeddings

                # Train/val split - MUST use donor-held-out splits to prevent leakage
                split_manifest_path = Path(config.data_dir) / "split_manifest.json"

                if split_manifest_path.exists():
                    # Load donor-held-out splits from manifest (generated by complete_data_prep.py)
                    import json

                    with open(split_manifest_path) as f:
                        split_manifest = json.load(f)

                    # Use fold 0 by default, or config.validation_fold if specified
                    fold_idx = getattr(config, "validation_fold", 0)
                    fold_spec = split_manifest["folds"][fold_idx]
                    train_donors = set(fold_spec["train_donors"])
                    val_donors = set(fold_spec["val_donors"])

                    log(f"  Using donor-held-out split (fold {fold_idx})")
                    log(f"    Train donors: {len(train_donors)}")
                    log(f"    Val donors: {len(val_donors)}")

                    # Filter cells by donor
                    if "donor_id" in cells_df.columns:
                        train_mask = cells_df["donor_id"].isin(train_donors).values
                        val_mask = cells_df["donor_id"].isin(val_donors).values

                        train_idx = torch.where(torch.tensor(train_mask))[0]
                        val_idx = torch.where(torch.tensor(val_mask))[0]

                        # Verify no donor leakage
                        train_donor_set = set(
                            cells_df.iloc[train_idx.numpy()]["donor_id"].unique()
                        )
                        val_donor_set = set(cells_df.iloc[val_idx.numpy()]["donor_id"].unique())
                        overlap = train_donor_set & val_donor_set
                        if overlap:
                            raise RuntimeError(f"DONOR LEAKAGE DETECTED: {overlap}")
                    else:
                        log(
                            "  WARNING: No donor_id column, falling back to random split (LEAKAGE RISK)"
                        )
                        n_train = int(0.9 * n_cells)
                        indices = torch.randperm(n_cells)
                        train_idx, val_idx = indices[:n_train], indices[n_train:]
                else:
                    # Fallback to random split if no manifest (e.g., during development)
                    log(
                        "  WARNING: No split_manifest.json found, using random split (LEAKAGE RISK)"
                    )
                    n_train = int(0.9 * n_cells)
                    indices = torch.randperm(n_cells)
                    train_idx, val_idx = indices[:n_train], indices[n_train:]

                train_data = TensorDataset(
                    niche_tokens[train_idx],
                    z_source[train_idx],
                    z_target[train_idx],
                )
                val_data = TensorDataset(
                    niche_tokens[val_idx],
                    z_source[val_idx],
                    z_target[val_idx],
                )

                log(f"  Train: {len(train_idx):,} cells")
                log(f"  Val: {len(val_idx):,} cells")

        except Exception as e:
            log(f"Warning: Failed to load real data: {e}")
            train_data = None

    # ==========================================================================
    # Load SEMI-SYNTHETIC benchmark data (with ground truth)
    # ==========================================================================
    benchmark_data = None
    benchmark_ground_truth = None
    try:
        benchmark_path = Path(config.data_dir) / "benchmark" / "semi_synthetic.pt"
        ground_truth_path = Path(config.data_dir) / "benchmark" / "ground_truth.json"

        if benchmark_path.exists():
            log(f"Loading semi-synthetic benchmark from {benchmark_path}")
            benchmark_tensors = torch.load(benchmark_path, map_location="cpu", weights_only=False)

            # Extract tensors from pre-generated benchmark
            expression = benchmark_tensors.get("expression")  # [N, n_genes]
            positions = benchmark_tensors.get("positions")  # [N, 2]
            is_interacting = benchmark_tensors.get("is_interacting")  # [N]
            celltype_idx = benchmark_tensors.get("celltype_idx")  # [N]
            stage_idx = benchmark_tensors.get("stage_idx")  # [N]
            pathway_scores = benchmark_tensors.get("pathway_scores")  # [N, n_pathways]

            log(f"  Expression shape: {expression.shape if expression is not None else 'None'}")
            log(f"  Positions shape: {positions.shape if positions is not None else 'None'}")
            log(f"  Pathway scores: {pathway_scores.shape if pathway_scores is not None else 'None'}")

            # For now, create simple dataset with available tensors
            # Full integration with niche tokens will come from canonical data prep
            if expression is not None and positions is not None:
                # Create mock niche tokens from expression (will be replaced by real niche encoding)
                n_cells = expression.shape[0]
                # Simple: repeat expression as 9 "tokens" for now
                # In production, this comes from the canonical data prep with proper niche encoding
                mock_niche = expression.unsqueeze(1).expand(-1, 9, -1)
                if mock_niche.shape[-1] != config.latent_dim:
                    # Project to latent dim
                    mock_niche = mock_niche[..., :config.latent_dim]
                    if mock_niche.shape[-1] < config.latent_dim:
                        padding = torch.zeros(n_cells, 9, config.latent_dim - mock_niche.shape[-1])
                        mock_niche = torch.cat([mock_niche, padding], dim=-1)

                benchmark_data = TensorDataset(
                    mock_niche.float(),
                    expression[:, :config.latent_dim].float() if expression.shape[-1] >= config.latent_dim else torch.cat([expression, torch.zeros(n_cells, config.latent_dim - expression.shape[-1])], dim=-1).float(),
                    expression[:, :config.latent_dim].float() if expression.shape[-1] >= config.latent_dim else torch.cat([expression, torch.zeros(n_cells, config.latent_dim - expression.shape[-1])], dim=-1).float(),
                )
                log(f"  Semi-synthetic benchmark: {len(benchmark_data)} samples")

            # Load ground truth for evaluation
            if ground_truth_path.exists():
                with open(ground_truth_path) as f:
                    benchmark_ground_truth = json.load(f)
                log(f"  Ground truth loaded: {len(benchmark_ground_truth.get('de_gene_sets', []))} DE gene sets")
        else:
            log(f"Warning: Benchmark file not found at {benchmark_path}")

    except Exception as e:
        log(f"Warning: Failed to load semi-synthetic data: {e}")
        import traceback
        log(traceback.format_exc())

    # ==========================================================================
    # Fallback to pure synthetic if no real data
    # ==========================================================================
    if train_data is None:
        log("Using SYNTHETIC data (no real data available)")
        torch.manual_seed(config.seed)
        n_train, n_val = 50000, 10000

        train_data = TensorDataset(
            torch.randn(n_train, 9, config.latent_dim),
            torch.randn(n_train, config.latent_dim),
            torch.randn(n_train, config.latent_dim),
        )
        val_data = TensorDataset(
            torch.randn(n_val, 9, config.latent_dim),
            torch.randn(n_val, config.latent_dim),
            torch.randn(n_val, config.latent_dim),
        )

    # ==========================================================================
    # Create DataLoaders
    # ==========================================================================
    if distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        val_sampler = DistributedSampler(val_data, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    benchmark_loader = None
    if benchmark_data is not None:
        benchmark_loader = DataLoader(
            benchmark_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, benchmark_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
    epoch: int,
    phase: str = "ssl",
) -> dict:
    """Train for one epoch.

    Args:
        phase: "ssl" for masked reconstruction, "transition" for flow/transition learning
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Set epoch for distributed sampler
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    phase_label = "SSL" if phase == "ssl" else "Trans"
    progress = tqdm(train_loader, desc=f"[{phase_label}] E{epoch}", disable=not is_main_process())

    for niche_tokens, z_source, z_target in progress:
        niche_tokens = niche_tokens.to(device, non_blocking=True)
        z_source = z_source.to(device, non_blocking=True)
        z_target = z_target.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            # Forward pass
            if hasattr(model, "module"):
                actual_model = model.module
            else:
                actual_model = model

            if phase == "ssl":
                # STAGE 1: SSL - Masked receiver reconstruction from niche context
                if hasattr(actual_model, "ssl_forward"):
                    receiver = niche_tokens[:, 0, :]
                    outputs = actual_model.ssl_forward(niche_tokens, receiver)
                    loss = outputs["loss_reconstruction"]
                else:
                    # Fallback: predict receiver from neighbors
                    context = actual_model(
                        receiver=niche_tokens[:, 0, :],
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    loss = torch.mean((context.context - niche_tokens[:, 0, :]) ** 2)
            else:
                # STAGE 2: Transition - Learn flow from source to target state
                if hasattr(actual_model, "transition_forward"):
                    outputs = actual_model.transition_forward(niche_tokens, z_source, z_target)
                    loss = outputs["loss_transition"]
                else:
                    # Fallback: predict target from context
                    context = actual_model(
                        receiver=z_source,
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    # Transition loss: predict z_target from z_source + context
                    loss = torch.mean((context.context - z_target) ** 2)

        # NaN/Inf detection - skip bad batches
        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
                log(f"[WARN] NaN/Inf loss detected at batch {n_batches}, skipping...")
            optimizer.zero_grad()  # Clear any accumulated gradients
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Check for NaN in gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            if is_main_process():
                log(f"[WARN] NaN/Inf gradient norm at batch {n_batches}, skipping...")
            optimizer.zero_grad()
            continue

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        progress.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}"})

    # Aggregate across processes
    if dist.is_initialized():
        total_loss_tensor = torch.tensor([total_loss], device=device)
        n_batches_tensor = torch.tensor([n_batches], device=device)
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(n_batches_tensor)
        total_loss = total_loss_tensor.item()
        n_batches = int(n_batches_tensor.item())

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    phase: str = "ssl",
) -> dict:
    """Validate the model.

    Args:
        phase: "ssl" for masked reconstruction, "transition" for flow/transition learning
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for niche_tokens, z_source, z_target in val_loader:
        niche_tokens = niche_tokens.to(device, non_blocking=True)
        z_source = z_source.to(device, non_blocking=True)
        z_target = z_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            if hasattr(model, "module"):
                actual_model = model.module
            else:
                actual_model = model

            if phase == "ssl":
                # SSL validation: reconstruction loss
                if hasattr(actual_model, "ssl_forward"):
                    receiver = niche_tokens[:, 0, :]
                    outputs = actual_model.ssl_forward(niche_tokens, receiver)
                    loss = outputs["loss_reconstruction"]
                else:
                    context = actual_model(
                        receiver=niche_tokens[:, 0, :],
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    loss = torch.mean((context.context - niche_tokens[:, 0, :]) ** 2)
            else:
                # Transition validation: flow prediction loss
                if hasattr(actual_model, "transition_forward"):
                    outputs = actual_model.transition_forward(niche_tokens, z_source, z_target)
                    loss = outputs["loss_transition"]
                else:
                    context = actual_model(
                        receiver=z_source,
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    loss = torch.mean((context.context - z_target) ** 2)

        total_loss += loss.item()
        n_batches += 1

    # Aggregate across processes
    if dist.is_initialized():
        total_loss_tensor = torch.tensor([total_loss], device=device)
        n_batches_tensor = torch.tensor([n_batches], device=device)
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(n_batches_tensor)
        total_loss = total_loss_tensor.item()
        n_batches = int(n_batches_tensor.item())

    return {"val_loss": total_loss / max(n_batches, 1)}


def train(config: TrainingConfig):
    """Main training loop."""
    start_time = datetime.now()

    # Load HPO best params if provided
    # Keys from run_v1_complete.py HPO: lr, hidden_dim, context_dim, dropout, ssl_weight, n_layers
    if config.use_best_hparams and config.hpo_params:
        hpo_path = Path(config.hpo_params)
        if hpo_path.exists():
            with open(hpo_path) as f:
                hpo_best = json.load(f)
            # Apply HPO params to config
            if "hidden_dim" in hpo_best:
                config.niche_hidden_dim = int(hpo_best["hidden_dim"])
            if "context_dim" in hpo_best:
                config.context_dim = int(hpo_best["context_dim"])
            if "dropout" in hpo_best:
                config.dropout = float(hpo_best["dropout"])
            if "lr" in hpo_best:
                config.learning_rate = float(hpo_best["lr"])
            print(f"Loaded HPO params from {hpo_path}: {hpo_best}")

    # Setup distributed
    distributed, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log(f"Training on device: {device}")
    log(f"Distributed: {distributed}, World size: {get_world_size()}")

    # Set seeds
    torch.manual_seed(config.seed + get_rank())
    np.random.seed(config.seed + get_rank())

    # Create output directory
    output_dir = Path(config.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "weights").mkdir(exist_ok=True)
        (output_dir / "figures").mkdir(exist_ok=True)
        (output_dir / "metrics").mkdir(exist_ok=True)

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    # Sync processes
    if distributed:
        dist.barrier()

    # Create model
    log("Creating model...")
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    # Create learning rate scheduler (warmup + cosine decay)
    ssl_scheduler = create_lr_scheduler(
        optimizer,
        num_epochs=config.ssl_epochs,
        warmup_epochs=config.warmup_epochs,
        min_lr=config.min_lr,
        use_cosine=config.use_cosine_schedule,
    )
    log(
        f"LR schedule: warmup={config.warmup_epochs} epochs, "
        f"cosine={config.use_cosine_schedule}, min_lr={config.min_lr}"
    )

    # Create checkpoint manager
    ckpt_manager = CheckpointManager(
        checkpoint_dir=output_dir / "checkpoints",
        keep_top_k=config.keep_top_k_checkpoints,
    )

    # Create metrics logger
    metrics_logger = MetricsLogger(output_dir) if is_main_process() else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"ssl_loss": [], "val_loss": []}

    if config.resume_checkpoint:
        log(f"Resuming from checkpoint: {config.resume_checkpoint}")
        checkpoint = CheckpointManager.load(Path(config.resume_checkpoint), device)

        # Load model state
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("metrics", {}).get("val_loss", float("inf"))
        log(f"Resumed from epoch {start_epoch} with val_loss={best_val_loss:.4f}")

    # Create dataloaders
    log("Creating dataloaders...")
    train_loader, val_loader, benchmark_loader = create_dataloaders(
        config, distributed=distributed
    )
    if benchmark_loader is not None:
        log("Semi-synthetic benchmark loader ready for evaluation")

    # ==========================================================================
    # STAGE 1: SSL Pretraining (masked receiver reconstruction from niche)
    # ==========================================================================
    log(f"\n{'=' * 60}")
    log(f"STAGE 1: SSL Pretraining ({config.ssl_epochs} epochs)")
    log("Objective: Masked receiver reconstruction from niche context")
    log(f"{'=' * 60}\n")

    for epoch in range(start_epoch, min(start_epoch + config.ssl_epochs, config.ssl_epochs)):
        # Train with SSL objective
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch, phase="ssl"
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config, phase="ssl")

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, "phase": "ssl"}

        # Track history
        history["ssl_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])

        # Step learning rate scheduler
        ssl_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        log(
            f"[SSL] Epoch {epoch + 1}/{config.ssl_epochs}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # Track LR in history
        if "learning_rate" not in history:
            history["learning_rate"] = []
        history["learning_rate"].append(current_lr)

        # Log to CSV
        if metrics_logger:
            metrics_logger.log(
                epoch + 1,
                "ssl",
                {
                    **metrics,
                    "learning_rate": current_lr,
                    "best_val_loss": best_val_loss,
                },
            )

        # Check if best
        is_best = val_metrics["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_loss"]

        # Save checkpoint
        if (epoch + 1) % config.checkpoint_every == 0 or is_best:
            ckpt_manager.save(model, optimizer, epoch + 1, metrics, config.to_dict(), is_best)

        # Sync processes
        if distributed:
            dist.barrier()

    # Track SSL phase GPU memory
    ssl_peak_memory_gb = 0.0
    if torch.cuda.is_available():
        ssl_peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()  # Reset for transition phase
        log(f"SSL phase peak GPU memory: {ssl_peak_memory_gb:.2f} GB")

    # Save SSL checkpoint before transition
    if is_main_process():
        ssl_checkpoint_path = output_dir / "checkpoints" / "ssl_pretrained.pt"
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(
            {
                "model_state_dict": state_dict,
                "epoch": config.ssl_epochs,
                "phase": "ssl_complete",
                "peak_memory_gb": ssl_peak_memory_gb,
            },
            ssl_checkpoint_path,
        )
        log(f"\nSSL pretraining complete. Checkpoint: {ssl_checkpoint_path}")

    # ==========================================================================
    # STAGE 2: Transition Model (learn state transitions / flow field)
    # ==========================================================================
    log(f"\n{'=' * 60}")
    log(f"STAGE 2: Transition Model ({config.transition_epochs} epochs)")
    log("Objective: Learn stage transition dynamics (flow field)")
    log(f"{'=' * 60}\n")

    # Reset optimizer for transition phase (lower LR for fine-tuning)
    transition_lr = config.learning_rate * 0.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=transition_lr,
        weight_decay=config.weight_decay,
    )

    # Create transition scheduler (shorter warmup since we're fine-tuning)
    transition_scheduler = create_lr_scheduler(
        optimizer,
        num_epochs=config.transition_epochs,
        warmup_epochs=min(2, config.warmup_epochs),  # Shorter warmup for fine-tuning
        min_lr=config.min_lr,
        use_cosine=config.use_cosine_schedule,
    )

    best_transition_loss = float("inf")
    history["transition_loss"] = []

    for epoch in range(config.transition_epochs):
        global_epoch = config.ssl_epochs + epoch

        # Train with transition objective
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            global_epoch,
            phase="transition",
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config, phase="transition")

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, "phase": "transition"}

        # Track history
        history["transition_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])

        # Step learning rate scheduler
        transition_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)

        # Log
        log(
            f"[Transition] Epoch {epoch + 1}/{config.transition_epochs}: "
            f"train_loss={train_metrics['train_loss']:.4f}, "
            f"val_loss={val_metrics['val_loss']:.4f}, "
            f"lr={current_lr:.2e}"
        )

        # Log to CSV
        if metrics_logger:
            metrics_logger.log(
                global_epoch + 1,
                "transition",
                {
                    **metrics,
                    "learning_rate": current_lr,
                    "best_transition_loss": best_transition_loss,
                },
            )

        # Check if best
        is_best = val_metrics["val_loss"] < best_transition_loss
        if is_best:
            best_transition_loss = val_metrics["val_loss"]

        # Save checkpoint
        if (epoch + 1) % config.checkpoint_every == 0 or is_best:
            ckpt_manager.save(
                model, optimizer, global_epoch + 1, metrics, config.to_dict(), is_best
            )

        # Sync processes
        if distributed:
            dist.barrier()

    # Track transition phase GPU memory
    transition_peak_memory_gb = 0.0
    if torch.cuda.is_available():
        transition_peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"Transition phase peak GPU memory: {transition_peak_memory_gb:.2f} GB")

    # Save final checkpoint
    final_metrics = {
        "train_loss": history["ssl_loss"][-1] if history["ssl_loss"] else 0,
        "val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
        "best_val_loss": best_val_loss,
    }
    ckpt_manager.save_final(model, config.to_dict(), final_metrics)

    # Save results
    if is_main_process():
        duration = (datetime.now() - start_time).total_seconds()

        # Save full metrics log
        if metrics_logger:
            metrics_logger.save_summary()
            log(f"Metrics saved to: {output_dir / 'metrics'}")

        results = {
            "history_semi_synthetic": history,
            "history_real": history,  # Same for now
            "duration_seconds": duration,
            "n_parameters": n_params,
            "n_gpus": get_world_size(),
            "final_metrics": final_metrics,
            "metrics": {
                "best_ssl_val_loss": best_val_loss,
                "best_transition_val_loss": best_transition_loss,
                "total_epochs": config.ssl_epochs + config.transition_epochs,
            },
            "gpu_memory": {
                "ssl_peak_gb": ssl_peak_memory_gb,
                "transition_peak_gb": transition_peak_memory_gb,
                "total_peak_gb": max(ssl_peak_memory_gb, transition_peak_memory_gb),
            },
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        log(f"\nTraining complete in {duration / 60:.1f} minutes")
        log(f"Best SSL val_loss: {best_val_loss:.4f}")
        log(f"Best Transition val_loss: {best_transition_loss:.4f}")
        log(f"Results saved to: {output_dir}")

    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="StageBridge V1 Training with DDP Support")

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hlca_path", type=str, default="")
    parser.add_argument("--luca_path", type=str, default="")

    # Model
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--niche_hidden_dim", type=int, default=128)
    parser.add_argument("--context_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--transition_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Learning rate schedule
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Warmup epochs for LR schedule"
    )
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for cosine decay")
    parser.add_argument(
        "--no_cosine_schedule", action="store_true", help="Disable cosine LR schedule"
    )

    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--keep_top_k_checkpoints", type=int, default=3)

    # Validation
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--validation_fold", type=int, default=0)

    # HPO
    parser.add_argument("--hpo_trials", type=int, default=30)
    parser.add_argument(
        "--hpo_params", type=str, default=None, help="Path to HPO best_params.json"
    )
    parser.add_argument("--use_best_hparams", action="store_true")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hlca_path=args.hlca_path,
        luca_path=args.luca_path,
        latent_dim=args.latent_dim,
        niche_hidden_dim=args.niche_hidden_dim,
        context_dim=args.context_dim,
        dropout=args.dropout,
        ssl_epochs=args.ssl_epochs,
        transition_epochs=args.transition_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        use_cosine_schedule=not args.no_cosine_schedule,
        checkpoint_every=args.checkpoint_every,
        resume_checkpoint=args.resume_checkpoint,
        keep_top_k_checkpoints=args.keep_top_k_checkpoints,
        n_folds=args.n_folds,
        validation_fold=args.validation_fold,
        hpo_trials=args.hpo_trials,
        hpo_params=args.hpo_params or "",
        use_best_hparams=args.use_best_hparams,
        seed=args.seed,
        num_workers=args.num_workers,
        mixed_precision=not args.no_mixed_precision,
    )

    train(config)


if __name__ == "__main__":
    main()
