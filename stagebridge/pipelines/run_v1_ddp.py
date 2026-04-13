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

from stagebridge.config import (
    FUSED_LATENT_DIM,
    HLCA_LATENT_DIM,
    LUCA_LATENT_DIM,
    N_NICHE_TOKENS,
)

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

    # Model - Dual Reference Geometry (dimensions from stagebridge.config)
    latent_dim: int = FUSED_LATENT_DIM  # 40: HLCA (30) + LuCA (10)
    hlca_dim: int = HLCA_LATENT_DIM     # 30: from HLCA scANVI model
    luca_dim: int = LUCA_LATENT_DIM     # 10: from LuCA scVI model
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

    # Ablation flags
    freeze_encoder: bool = False  # Freeze encoder during transition phase (for ablation)

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


# Pathway/proliferation auxiliary heads (paper-inspired: SpatialFusion, OSDR)
class PathwayHead(nn.Module):
    """Predict PROGENy pathway scores from context (SpatialFusion-inspired)."""
    def __init__(self, input_dim: int, n_pathways: int = 14):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_pathways),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ProliferationHead(nn.Module):
    """Predict Ki67 proliferation (OSDR-inspired)."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class IL1BHead(nn.Module):
    """Predict IL1B pathway activity (Peng/Kadara hypothesis test).

    Direct test of the biological hypothesis: IL1B+ macrophages in niche
    drive IL1B-IL1R1 signaling in epithelial cells. This is the most
    important biological claim to validate.

    IL1B is pathway index 4 in PROGENy (TNFa, NFkB, PI3K, JAK-STAT, IL1B...).
    We extract it explicitly for focused supervision and interpretability.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Single IL1B activity score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class KACHead(nn.Module):
    """Predict KAC (KRT8+ Alveolar Intermediate Cell) signature.

    From Nature 2024 Kadara lab: KACs are THE key intermediate state in
    AT2 -> LUAD progression. The trajectory is:
    Normal -> AT2 -> AIC -> KAC (KRT8+) -> LUAD tumor

    Key insight: KACs are found in tumor-adjacent normal tissue BEFORE
    tumors form. This is the critical progression window where intervention
    is possible. Our transition model should capture cells transitioning
    toward this state.

    KAC markers: KRT8, CLDN4, CDKN1A, CDKN2A, PLAUR (senescence + invasion)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # KAC signature score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def create_model(
    config: TrainingConfig,
    device: torch.device,
    hlca_dim: int = 30,
    luca_dim: int = 10,
) -> nn.Module:
    """Create the StageBridge model with dual-reference geometry.

    Args:
        config: Training configuration
        device: Target device
        hlca_dim: HLCA embedding dimension (default 30 from scANVI)
        luca_dim: LuCA embedding dimension (default 10 from scVI)
    """
    try:
        from stagebridge.pipelines.run_v1_complete import StageBridgeV1Complete

        model = StageBridgeV1Complete(
            latent_dim=config.latent_dim,
            niche_hidden_dim=config.niche_hidden_dim,
            context_dim=config.context_dim,
            dropout=config.dropout,
            hlca_dim=hlca_dim,  # Dual-reference geometry
            luca_dim=luca_dim,  # Dual-reference geometry
        )
        log(f"Created StageBridgeV1Complete with dual-reference encoder (HLCA={hlca_dim}d, LuCA={luca_dim}d)")
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
        log("WARNING: Using fallback ReceiverCenteredNicheEncoder (no dual-reference)")

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

            # Extract embedding columns (support both naming conventions)
            fused_cols = sorted([c for c in cells_df.columns if c.startswith("z_fused_") or c.startswith("fused_latent_")])
            hlca_cols = sorted([c for c in cells_df.columns if c.startswith("z_hlca_") or c.startswith("hlca_latent_")])
            luca_cols = sorted([c for c in cells_df.columns if c.startswith("z_luca_") or c.startswith("luca_latent_")])

            # Log embedding dimensions
            log(f"  Fused embedding: {len(fused_cols)} dims")
            log(f"  HLCA embedding: {len(hlca_cols)} dims (dual-reference)")
            log(f"  LuCA embedding: {len(luca_cols)} dims (dual-reference)")

            if fused_cols:
                # Build niche tokens from neighborhoods
                # Token order: [receiver, ring1, ring2, ring3, ring4, hlca, luca, pathway, stats]
                embeddings = torch.tensor(cells_df[fused_cols].values, dtype=torch.float32)
                n_cells = len(embeddings)

                # Load HLCA and LuCA embeddings separately for dual-reference encoder
                hlca_embeddings = None
                luca_embeddings = None
                if hlca_cols:
                    hlca_embeddings = torch.tensor(cells_df[hlca_cols].values, dtype=torch.float32)
                    log(f"  Loaded HLCA embeddings: {hlca_embeddings.shape}")
                if luca_cols:
                    luca_embeddings = torch.tensor(cells_df[luca_cols].values, dtype=torch.float32)
                    log(f"  Loaded LuCA embeddings: {luca_embeddings.shape}")

                # Create 9-token sequences
                # Token structure: [receiver, ring1, ring2, ring3, ring4, hlca, luca, pathway, stats]
                embed_dim = embeddings.shape[1]  # 40 for fused
                niche_tokens = torch.zeros(n_cells, 9, embed_dim)

                # Token 0: Receiver (fused embedding)
                niche_tokens[:, 0, :] = embeddings

                # ==========================================================================
                # Tokens 1-4: Ring embeddings from neighborhoods.parquet
                # ==========================================================================
                # Build cell_id to index mapping for fast lookup
                cell_id_to_idx = {cid: i for i, cid in enumerate(cells_df["cell_id"].values)}

                # Parse neighborhoods_df to extract ring tokens
                # neighborhoods_df has columns: cell_id, donor_id, stage, tokens
                # where 'tokens' is a list of 9 token dicts with z_pooled for rings
                n_rings_populated = 0
                n_stats_populated = 0

                if len(neighborhoods_df) > 0 and "tokens" in neighborhoods_df.columns:
                    log("  Parsing neighborhoods.parquet for REAL ring tokens...")
                    log(f"    Processing {len(neighborhoods_df):,} neighborhoods...")

                    # Pre-extract columns for faster iteration
                    cell_ids = neighborhoods_df["cell_id"].values
                    tokens_col = neighborhoods_df["tokens"].values

                    for row_idx in range(len(neighborhoods_df)):
                        cell_id = cell_ids[row_idx]
                        if cell_id not in cell_id_to_idx:
                            continue
                        cell_idx = cell_id_to_idx[cell_id]
                        tokens_list = tokens_col[row_idx]

                        # tokens_list is a list of 9 token dicts
                        for token_dict in tokens_list:
                            token_idx = token_dict.get("token_idx", -1)

                            # Tokens 1-4: Ring embeddings (z_pooled)
                            if 1 <= token_idx <= 4:
                                z_pooled = token_dict.get("z_pooled")
                                if z_pooled is not None and len(z_pooled) > 0:
                                    z_pooled_tensor = torch.tensor(z_pooled, dtype=torch.float32)
                                    # Ensure z_pooled fits embed_dim (pad or truncate)
                                    if len(z_pooled_tensor) < embed_dim:
                                        padded = torch.zeros(embed_dim)
                                        padded[:len(z_pooled_tensor)] = z_pooled_tensor
                                        z_pooled_tensor = padded
                                    elif len(z_pooled_tensor) > embed_dim:
                                        z_pooled_tensor = z_pooled_tensor[:embed_dim]
                                    niche_tokens[cell_idx, token_idx, :] = z_pooled_tensor
                                    n_rings_populated += 1

                            # Token 8: Stats (n_neighbors, mean_distance, diversity)
                            elif token_idx == 8:
                                n_neighbors = token_dict.get("n_neighbors", 0)
                                mean_distance = token_dict.get("mean_distance", 0.0)
                                diversity = token_dict.get("diversity", 0)
                                # Encode stats into first few dims of token 8
                                # Normalize: n_neighbors/20, mean_distance (already normalized), diversity/20
                                niche_tokens[cell_idx, 8, 0] = float(n_neighbors) / 20.0
                                niche_tokens[cell_idx, 8, 1] = float(mean_distance)
                                niche_tokens[cell_idx, 8, 2] = float(diversity) / 20.0
                                n_stats_populated += 1

                    log(f"  Ring tokens populated: {n_rings_populated} (from {len(neighborhoods_df)} neighborhoods)")
                    log(f"  Stats tokens populated: {n_stats_populated}")

                    # Verify ring tokens are different from receiver
                    if n_rings_populated > 0:
                        # Check a sample of cells to verify diversity
                        sample_idx = min(100, n_cells)
                        ring_diff = (niche_tokens[:sample_idx, 1:5, :] - niche_tokens[:sample_idx, 0:1, :]).abs().mean()
                        log(f"  Ring-receiver difference (sample): {ring_diff:.4f} (should be > 0)")
                        if ring_diff < 1e-6:
                            log("  WARNING: Ring tokens appear identical to receiver - check neighborhoods parsing")
                else:
                    # Fallback: use receiver embedding for rings (placeholder behavior)
                    log("  WARNING: No 'tokens' column in neighborhoods.parquet or empty DataFrame")
                    log("  FALLBACK: Using receiver embedding for ring tokens (degraded niche context)")
                    for ring_idx in range(1, 5):
                        niche_tokens[:, ring_idx, :] = embeddings

                # Fill any remaining cells without neighborhoods (snRNA cells) with receiver embedding
                # Spatial cells should have neighborhoods; snRNA cells (no spatial coords) won't
                spatial_mask = cells_df["cell_id"].str.startswith("spatial_")
                n_spatial = spatial_mask.sum()
                n_snrna = (~spatial_mask).sum()
                log(f"  Spatial cells: {n_spatial:,}, snRNA cells: {n_snrna:,}")

                # For cells without neighborhoods, use receiver as fallback
                # This is detected by all-zero ring tokens
                ring_sum = niche_tokens[:, 1:5, :].abs().sum(dim=(1, 2))
                cells_without_niche_mask = (ring_sum < 1e-6)
                cells_without_niche = cells_without_niche_mask.sum().item()
                if cells_without_niche > 0:
                    log(f"  Cells without niche context: {cells_without_niche:,} (using receiver as fallback)")
                    # Vectorized assignment: for cells without niche, copy receiver to all ring positions
                    for ring_idx in range(1, 5):
                        niche_tokens[cells_without_niche_mask, ring_idx, :] = embeddings[cells_without_niche_mask]

                # Token 5: HLCA embedding (pad to fused dim if needed)
                if hlca_embeddings is not None:
                    hlca_dim = hlca_embeddings.shape[1]
                    niche_tokens[:, 5, :hlca_dim] = hlca_embeddings
                    log(f"  Token 5 (HLCA): {hlca_dim} dims")
                else:
                    niche_tokens[:, 5, :] = embeddings
                    log("  Token 5 (HLCA): using fused (no separate HLCA)")

                # Token 6: LuCA embedding (pad to fused dim if needed)
                if luca_embeddings is not None:
                    luca_dim = luca_embeddings.shape[1]
                    niche_tokens[:, 6, :luca_dim] = luca_embeddings
                    log(f"  Token 6 (LuCA): {luca_dim} dims")
                else:
                    niche_tokens[:, 6, :] = embeddings
                    log("  Token 6 (LuCA): using fused (no separate LuCA)")

                # Token 7: Pathway (will be filled with gamma below if available)
                # Token 8: Stats (populated above from neighborhoods.parquet)
                log(f"  Token 7 (pathway): zeros (gamma will be added if available)")
                log(f"  Token 8 (stats): {n_stats_populated} cells have neighborhood stats")

                # z_source and z_target for transition learning
                # Create REAL cross-stage transition pairs using stage information
                stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
                stage_to_idx = {s: i for i, s in enumerate(stage_order)}

                if "stage" in cells_df.columns:
                    stage_indices = torch.tensor(
                        cells_df["stage"].map(stage_to_idx).fillna(0).astype(int).values,
                        dtype=torch.long
                    )
                    log(f"  Stage distribution: {dict(zip(stage_order, [(stage_indices == i).sum().item() for i in range(5)]))}")

                    # Create cross-stage transition pairs:
                    # For each cell in stage N, find target cells in stage N+1
                    # We'll compute z_target as mean of next-stage cells (per stage)
                    z_source = embeddings
                    z_target = torch.zeros_like(embeddings)

                    for stage_idx in range(4):  # 0-3 (Normal through MIA)
                        source_mask = (stage_indices == stage_idx)
                        target_mask = (stage_indices == stage_idx + 1)

                        if source_mask.any() and target_mask.any():
                            # Target is mean of next-stage cells (will be refined by OT in transition_forward)
                            target_mean = embeddings[target_mask].mean(dim=0)
                            z_target[source_mask] = target_mean
                        elif source_mask.any():
                            # No next stage available, use self (will be filtered in training)
                            z_target[source_mask] = embeddings[source_mask]

                    # For LUAD (stage 4), no progression target - use self
                    luad_mask = (stage_indices == 4)
                    z_target[luad_mask] = embeddings[luad_mask]

                    log("  Cross-stage transition pairs created (OT refinement in training)")
                else:
                    log("  WARNING: No stage column, using self-transitions (degenerate)")
                    stage_indices = torch.zeros(n_cells, dtype=torch.long)
                    z_source = embeddings
                    z_target = embeddings

                # Extract DestVI gamma values if available (intra-cell-type functional state)
                gamma_cols = [c for c in cells_df.columns if c.startswith("gamma_")]
                gamma_features = None
                if gamma_cols:
                    gamma_features = torch.tensor(
                        cells_df[sorted(gamma_cols)].values, dtype=torch.float32
                    )
                    n_gamma = gamma_features.shape[1]
                    log(f"  Gamma values: {n_gamma} dims (DestVI functional state)")

                    # Add gamma to token 7 (pathway token) - represents functional state
                    # Pad or truncate gamma to fit niche_tokens dimension (from fused embeddings)
                    embed_dim = niche_tokens.shape[2]  # Actual embedding dimension (40 for fused)
                    if n_gamma < embed_dim:
                        # Pad gamma with zeros to match embedding dim for token 7
                        gamma_padded = torch.zeros(n_cells, embed_dim)
                        gamma_padded[:, :n_gamma] = gamma_features
                        niche_tokens[:, 7, :] = gamma_padded  # Token 7 = functional state
                        log(f"  Token 7 (pathway) enriched with gamma ({n_gamma} dims, padded to {embed_dim})")

                # Extract pathway scores if available (pre-computed in complete_data_prep.py)
                pathway_cols = [c for c in cells_df.columns if c.startswith("pathway_")]
                pathway_targets = None
                if pathway_cols:
                    pathway_targets = torch.tensor(
                        cells_df[sorted(pathway_cols)].values, dtype=torch.float32
                    )
                    log(f"  Pathway targets: {pathway_targets.shape[1]} pathways")

                # Extract proliferation label if available
                prolif_targets = None
                if "proliferation_label" in cells_df.columns:
                    prolif_targets = torch.tensor(
                        cells_df["proliferation_label"].values, dtype=torch.float32
                    ).unsqueeze(1)
                    log(f"  Proliferation targets: {prolif_targets.shape}")

                # Extract WES features for evolutionary regularization
                # 8 features: tmb, kras_mut, egfr_mut, tp53_mut, stk11_mut, keap1_mut, smad4_mut, braf_mut
                wes_cols = ["tmb", "kras_mut", "egfr_mut", "tp53_mut", "stk11_mut", "keap1_mut", "smad4_mut", "braf_mut"]
                wes_features = None
                available_wes_cols = [c for c in wes_cols if c in cells_df.columns]
                if available_wes_cols:
                    wes_features = torch.tensor(
                        cells_df[available_wes_cols].values, dtype=torch.float32
                    )
                    log(f"  WES features: {wes_features.shape[1]} columns (evolutionary regularization)")
                    # Log mutation prevalence
                    for col in available_wes_cols:
                        if col != "tmb":
                            pct = (cells_df[col] > 0).mean() * 100
                            log(f"    {col}: {pct:.1f}% mutated")

                # Train/val split - MUST use donor-held-out splits to prevent leakage
                split_manifest_path = Path(config.data_dir) / "split_manifest.json"

                if split_manifest_path.exists():
                    # Load donor-held-out splits from manifest (generated by complete_data_prep.py)
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

                # Include pathway/proliferation targets in dataset (None-safe)
                train_pathway = pathway_targets[train_idx] if pathway_targets is not None else torch.zeros(len(train_idx), 14)
                val_pathway = pathway_targets[val_idx] if pathway_targets is not None else torch.zeros(len(val_idx), 14)
                train_prolif = prolif_targets[train_idx] if prolif_targets is not None else torch.zeros(len(train_idx), 1)
                val_prolif = prolif_targets[val_idx] if prolif_targets is not None else torch.zeros(len(val_idx), 1)

                # Include WES features for evolutionary regularization (None-safe)
                n_wes = 8  # tmb + 7 driver mutations
                train_wes = wes_features[train_idx] if wes_features is not None else torch.zeros(len(train_idx), n_wes)
                val_wes = wes_features[val_idx] if wes_features is not None else torch.zeros(len(val_idx), n_wes)

                # Include stage indices for cross-stage OT during transition training
                train_stages = stage_indices[train_idx]
                val_stages = stage_indices[val_idx]

                # Extract donor IDs for donor-consistency analysis
                if "donor_id" in cells_df.columns:
                    donor_to_idx = {d: i for i, d in enumerate(cells_df["donor_id"].unique())}
                    donor_indices = torch.tensor(
                        cells_df["donor_id"].map(donor_to_idx).values, dtype=torch.long
                    )
                    train_donors_idx = donor_indices[train_idx]
                    val_donors_idx = donor_indices[val_idx]
                else:
                    train_donors_idx = torch.zeros(len(train_idx), dtype=torch.long)
                    val_donors_idx = torch.zeros(len(val_idx), dtype=torch.long)

                # NOTE: HLCA/LuCA are already in the fused embedding (niche_tokens)
                # The Linear projection learns to weight them. See docs/architecture/dual_reference_encoder.md

                # Dataset: [niche_tokens, z_source, z_target, pathway, prolif, stages, donors, wes]
                train_data = TensorDataset(
                    niche_tokens[train_idx],
                    z_source[train_idx],
                    z_target[train_idx],
                    train_pathway,
                    train_prolif,
                    train_stages,
                    train_donors_idx,
                    train_wes,
                )
                val_data = TensorDataset(
                    niche_tokens[val_idx],
                    z_source[val_idx],
                    z_target[val_idx],
                    val_pathway,
                    val_prolif,
                    val_stages,
                    val_donors_idx,
                    val_wes,
                )

                log(f"  Train: {len(train_idx):,} cells")
                log(f"  Val: {len(val_idx):,} cells")
                log(f"  Embedding: {len(fused_cols)}d fused (HLCA {len(hlca_cols)}d + LuCA {len(luca_cols)}d)")

        except Exception as e:
            import traceback
            log(f"ERROR: Failed to load real data: {e}")
            log(f"Traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to fail loudly instead of silent fallback

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
    # FAIL if no real data - no silent fallback to synthetic
    # ==========================================================================
    if train_data is None:
        raise RuntimeError(
            f"Failed to load real data from {config.data_dir}. "
            f"Check that cells.parquet exists and has z_fused_* columns. "
            f"cells_path.exists()={cells_path.exists()}, "
            f"neighborhoods_path.exists()={neighborhoods_path.exists()}"
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
    pathway_head: nn.Module | None = None,
    prolif_head: nn.Module | None = None,
    il1b_head: nn.Module | None = None,
    kac_head: nn.Module | None = None,
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

    # Track auxiliary losses
    total_pathway_loss = 0.0
    total_prolif_loss = 0.0
    total_il1b_loss = 0.0  # Peng/Kadara hypothesis test
    total_kac_loss = 0.0  # KAC intermediate state (Nature 2024)
    total_wes_loss = 0.0  # WES evolutionary regularization
    n_aux_batches = 0

    # Stage-stratified metrics tracking
    stage_losses = {i: [] for i in range(5)}  # Normal=0, AAH=1, AIS=2, MIA=3, LUAD=4

    for batch in progress:
        # Unpack batch (8 tensors: niche_tokens, z_source, z_target, pathway, prolif, stage, donor, wes)
        niche_tokens = batch[0].to(device, non_blocking=True)
        z_source = batch[1].to(device, non_blocking=True)
        z_target = batch[2].to(device, non_blocking=True)
        pathway_targets = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None
        prolif_targets = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
        stage_indices = batch[5].to(device, non_blocking=True) if len(batch) > 5 else None
        donor_indices = batch[6].to(device, non_blocking=True) if len(batch) > 6 else None
        wes_features = batch[7].to(device, non_blocking=True) if len(batch) > 7 else None
        # hlca_embeddings = batch[7], luca_embeddings = batch[8]

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            # Forward pass
            if hasattr(model, "module"):
                actual_model = model.module
            else:
                actual_model = model

            if phase == "ssl":
                # STAGE 1: SSL - Masked receiver reconstruction from niche context
                # The fused embedding [HLCA | LuCA] already encodes dual-reference geometry
                if hasattr(actual_model, "ssl_forward"):
                    receiver = niche_tokens[:, 0, :]  # [B, 40] = fused
                    outputs = actual_model.ssl_forward(niche_tokens, receiver)
                    loss = outputs["loss_reconstruction"]

                    # Track stage-stratified SSL loss
                    if stage_indices is not None:
                        with torch.no_grad():
                            for s in range(5):
                                mask = (stage_indices == s)
                                if mask.any():
                                    stage_loss = torch.mean((outputs["receiver_pred"][mask] - receiver[mask]) ** 2)
                                    stage_losses[s].append(stage_loss.item())
                else:
                    # Fallback: predict receiver from neighbors
                    context = actual_model(
                        receiver=niche_tokens[:, 0, :],
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    loss = torch.mean((context.context - niche_tokens[:, 0, :]) ** 2)
            else:
                # STAGE 2: Transition - Learn flow with CROSS-STAGE OT pairing
                if hasattr(actual_model, "transition_forward"):
                    # Encode niche context (fused embedding contains dual-reference geometry)
                    context = actual_model.encode_niche(niche_tokens)

                    # Use stage-aware OT: only pair cells across adjacent stages
                    # Filter to cells that can transition (stages 0-3, not LUAD)
                    if stage_indices is not None:
                        can_transition = (stage_indices < 4)  # Not LUAD
                        if can_transition.sum() >= 4:  # Need enough cells for OT
                            # Create cross-stage batches for OT
                            trans_z_source = z_source[can_transition]
                            trans_z_target = z_target[can_transition]
                            trans_context = context[can_transition]
                            trans_stages = stage_indices[can_transition]

                            outputs = actual_model.transition_forward(
                                trans_z_source, trans_z_target, trans_context, use_ot=True
                            )
                            loss = outputs["loss_transition"]

                            # Track stage-stratified transition loss
                            with torch.no_grad():
                                for s in range(4):  # Only 0-3 can transition
                                    mask = (trans_stages == s)
                                    if mask.any():
                                        stage_losses[s].append(loss.item())  # Approximate
                        else:
                            # Fallback: use all cells
                            outputs = actual_model.transition_forward(z_source, z_target, context, use_ot=True)
                            loss = outputs["loss_transition"]
                    else:
                        outputs = actual_model.transition_forward(z_source, z_target, context, use_ot=True)
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

            # =================================================================
            # AUXILIARY LOSSES: Pathway + Proliferation (SpatialFusion/OSDR-inspired)
            # =================================================================
            # Get context representation for auxiliary heads
            if 'context' in dir() and hasattr(context, 'context'):
                aux_repr = context.context
            elif 'outputs' in dir() and isinstance(outputs, dict) and 'context' in outputs:
                aux_repr = outputs['context']
            else:
                # Use receiver embedding as fallback
                aux_repr = niche_tokens[:, 0, :]

            # Pathway auxiliary loss (weight: 0.05)
            pathway_loss = torch.tensor(0.0, device=device)
            if pathway_head is not None and pathway_targets is not None:
                # Only compute if targets have actual signal (not all zeros)
                if pathway_targets.abs().sum() > 0:
                    pathway_pred = pathway_head(aux_repr)
                    pathway_loss = torch.nn.functional.mse_loss(pathway_pred, pathway_targets)
                    total_pathway_loss += pathway_loss.item()
                    n_aux_batches += 1

            # Proliferation auxiliary loss (weight: 0.05)
            prolif_loss = torch.tensor(0.0, device=device)
            if prolif_head is not None and prolif_targets is not None:
                if prolif_targets.abs().sum() > 0:
                    prolif_pred = prolif_head(aux_repr)
                    prolif_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        prolif_pred, prolif_targets
                    )
                    total_prolif_loss += prolif_loss.item()

            # IL1B-specific auxiliary loss (weight: 0.05) - Peng/Kadara hypothesis test
            # IL1B is pathway index 4 in standard PROGENy ordering
            # This is THE key biological claim: IL1B+ macrophages → epithelial IL1B-IL1R1 signaling
            il1b_loss = torch.tensor(0.0, device=device)
            if il1b_head is not None and pathway_targets is not None:
                if pathway_targets.shape[1] > 4:  # Ensure IL1B index exists
                    il1b_targets = pathway_targets[:, 4:5]  # Extract IL1B (index 4)
                    if il1b_targets.abs().sum() > 0:
                        il1b_pred = il1b_head(aux_repr)
                        il1b_loss = torch.nn.functional.mse_loss(il1b_pred, il1b_targets)
                        total_il1b_loss += il1b_loss.item()

            # KAC signature loss (weight: 0.10) - Nature 2024 key intermediate state
            # KAC is computed from a subset of pathway genes (indices vary by dataset)
            kac_loss = torch.tensor(0.0, device=device)
            if kac_head is not None and pathway_targets is not None:
                # KAC score computed as mean z-score of KAC markers in pathway targets
                # For now, use p53 pathway (index 7) as proxy since CDKN1A/2A are in it
                if pathway_targets.shape[1] > 7:
                    kac_proxy = pathway_targets[:, 7:8]  # p53 pathway as proxy for senescence
                    if kac_proxy.abs().sum() > 0:
                        kac_pred = kac_head(aux_repr)
                        kac_loss = torch.nn.functional.mse_loss(kac_pred, kac_proxy)
                        total_kac_loss += kac_loss.item()

            # WES evolutionary regularization (weight: 0.05)
            # Penalizes transitions where cells with different driver mutations produce identical dynamics
            # Uses pairwise L1 distance on WES features (tmb + 7 driver mutations)
            wes_loss = torch.tensor(0.0, device=device)
            if wes_features is not None and phase == "transition":
                from stagebridge.transition_model.wes_regularizer import pairwise_wes_penalty
                # Only compute for cells that are transitioning
                if 'trans_z_source' in dir() and len(trans_z_source) > 1:
                    trans_wes = wes_features[can_transition]
                    # Compute pairwise penalty: cells with different WES profiles should have different dynamics
                    # We sample pairs to avoid O(N^2) computation
                    n_pairs = min(256, len(trans_wes))
                    idx1 = torch.randperm(len(trans_wes))[:n_pairs]
                    idx2 = torch.randperm(len(trans_wes))[:n_pairs]
                    wes_penalty = pairwise_wes_penalty(trans_wes[idx1], trans_wes[idx2], penalty_scale=0.1)
                    wes_loss = wes_penalty.mean()
                    total_wes_loss += wes_loss.item()

            # Add auxiliary losses (weighted at 0.05 each as per doctrine)
            # IL1B gets extra weight (0.10) as the key hypothesis test
            # KAC gets extra weight (0.10) as the key intermediate state
            # WES regularization: small weight (0.05) to encourage evolutionary-aware transitions
            loss = loss + 0.05 * pathway_loss + 0.05 * prolif_loss + 0.10 * il1b_loss + 0.10 * kac_loss + 0.05 * wes_loss

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
            scaler.update()  # Reset scaler state before continuing
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

    metrics = {"train_loss": total_loss / max(n_batches, 1)}
    if n_aux_batches > 0:
        metrics["train_pathway_loss"] = total_pathway_loss / n_aux_batches
        metrics["train_prolif_loss"] = total_prolif_loss / n_aux_batches
        metrics["train_il1b_loss"] = total_il1b_loss / n_aux_batches  # Peng/Kadara hypothesis
        metrics["train_kac_loss"] = total_kac_loss / n_aux_batches  # Nature 2024 intermediate

    # Add stage-stratified metrics (Task #5)
    stage_names = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    for s, name in enumerate(stage_names):
        if stage_losses[s]:
            metrics[f"train_loss_{name}"] = sum(stage_losses[s]) / len(stage_losses[s])

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    phase: str = "ssl",
    pathway_head: nn.Module | None = None,
    prolif_head: nn.Module | None = None,
    il1b_head: nn.Module | None = None,
    kac_head: nn.Module | None = None,
) -> dict:
    """Validate the model.

    Args:
        phase: "ssl" for masked reconstruction, "transition" for flow/transition learning
    """
    model.eval()
    total_loss = 0.0
    total_pathway_loss = 0.0
    total_prolif_loss = 0.0
    total_il1b_loss = 0.0  # Peng/Kadara hypothesis test
    total_kac_loss = 0.0  # KAC intermediate state (Nature 2024)
    total_wes_loss = 0.0  # WES evolutionary regularization
    n_batches = 0
    n_aux_batches = 0

    # Stage-stratified and donor-level metrics (Tasks #5, #6)
    stage_losses = {i: [] for i in range(5)}
    donor_losses = {}  # donor_idx -> list of losses

    for batch in val_loader:
        # Unpack batch (8 tensors: niche_tokens, z_source, z_target, pathway, prolif, stage, donor, wes)
        niche_tokens = batch[0].to(device, non_blocking=True)
        z_source = batch[1].to(device, non_blocking=True)
        z_target = batch[2].to(device, non_blocking=True)
        pathway_targets = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None
        prolif_targets = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
        stage_indices = batch[5].to(device, non_blocking=True) if len(batch) > 5 else None
        donor_indices = batch[6].to(device, non_blocking=True) if len(batch) > 6 else None
        wes_features = batch[7].to(device, non_blocking=True) if len(batch) > 7 else None

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

                    # Stage-stratified validation loss
                    if stage_indices is not None:
                        for s in range(5):
                            mask = (stage_indices == s)
                            if mask.any():
                                stage_loss = torch.mean((outputs["receiver_pred"][mask] - receiver[mask]) ** 2)
                                stage_losses[s].append(stage_loss.item())

                    # Donor-level validation loss (Task #6)
                    if donor_indices is not None:
                        unique_donors = donor_indices.unique()
                        for d in unique_donors:
                            d_idx = d.item()
                            mask = (donor_indices == d)
                            if mask.any():
                                donor_loss = torch.mean((outputs["receiver_pred"][mask] - receiver[mask]) ** 2)
                                if d_idx not in donor_losses:
                                    donor_losses[d_idx] = []
                                donor_losses[d_idx].append(donor_loss.item())
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
                    context = actual_model.encode_niche(niche_tokens)
                    outputs = actual_model.transition_forward(z_source, z_target, context, use_ot=True)
                    loss = outputs["loss_transition"]
                else:
                    context = actual_model(
                        receiver=z_source,
                        neighbors=niche_tokens[:, 1:, :],
                        distances=torch.ones(niche_tokens.shape[0], 8, device=device),
                    )
                    loss = torch.mean((context.context - z_target) ** 2)

            # Auxiliary losses for validation metrics
            if 'outputs' in dir() and isinstance(outputs, dict) and 'context' in outputs:
                aux_repr = outputs['context']
            else:
                aux_repr = niche_tokens[:, 0, :]

            if pathway_head is not None and pathway_targets is not None and pathway_targets.abs().sum() > 0:
                pathway_pred = pathway_head(aux_repr)
                total_pathway_loss += torch.nn.functional.mse_loss(pathway_pred, pathway_targets).item()
                n_aux_batches += 1

            if prolif_head is not None and prolif_targets is not None and prolif_targets.abs().sum() > 0:
                prolif_pred = prolif_head(aux_repr)
                total_prolif_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                    prolif_pred, prolif_targets
                ).item()

            # IL1B validation loss (Peng/Kadara hypothesis test)
            if il1b_head is not None and pathway_targets is not None:
                if pathway_targets.shape[1] > 4:
                    il1b_targets = pathway_targets[:, 4:5]
                    if il1b_targets.abs().sum() > 0:
                        il1b_pred = il1b_head(aux_repr)
                        total_il1b_loss += torch.nn.functional.mse_loss(il1b_pred, il1b_targets).item()

            # KAC validation loss (Nature 2024 intermediate state)
            if kac_head is not None and pathway_targets is not None:
                if pathway_targets.shape[1] > 7:
                    kac_proxy = pathway_targets[:, 7:8]  # p53 as proxy for senescence
                    if kac_proxy.abs().sum() > 0:
                        kac_pred = kac_head(aux_repr)
                        total_kac_loss += torch.nn.functional.mse_loss(kac_pred, kac_proxy).item()

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

    metrics = {"val_loss": total_loss / max(n_batches, 1)}
    if n_aux_batches > 0:
        metrics["val_pathway_loss"] = total_pathway_loss / n_aux_batches
        metrics["val_prolif_loss"] = total_prolif_loss / n_aux_batches
        metrics["val_il1b_loss"] = total_il1b_loss / n_aux_batches  # Peng/Kadara hypothesis
        metrics["val_kac_loss"] = total_kac_loss / n_aux_batches  # Nature 2024 intermediate

    # Add stage-stratified validation metrics (Task #5)
    stage_names = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    for s, name in enumerate(stage_names):
        if stage_losses[s]:
            metrics[f"val_loss_{name}"] = sum(stage_losses[s]) / len(stage_losses[s])

    # Add donor consistency metrics (Task #6)
    if donor_losses:
        donor_means = [sum(v) / len(v) for v in donor_losses.values() if v]
        if donor_means:
            metrics["val_donor_mean"] = sum(donor_means) / len(donor_means)
            metrics["val_donor_std"] = (sum((m - metrics["val_donor_mean"])**2 for m in donor_means) / len(donor_means)) ** 0.5
            metrics["val_donor_min"] = min(donor_means)
            metrics["val_donor_max"] = max(donor_means)

    return metrics


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

    # Create model with dual-reference geometry
    log("Creating model...")
    model = create_model(
        config,
        device,
        hlca_dim=config.hlca_dim,
        luca_dim=config.luca_dim,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: {n_params:,}")

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Create auxiliary heads for paper-inspired losses (SpatialFusion/OSDR)
    # Input dim matches context dim from the model
    aux_input_dim = config.context_dim
    pathway_head = PathwayHead(aux_input_dim, n_pathways=14).to(device)
    prolif_head = ProliferationHead(aux_input_dim).to(device)
    il1b_head = IL1BHead(aux_input_dim).to(device)  # Peng/Kadara IL1B hypothesis
    kac_head = KACHead(aux_input_dim).to(device)  # KAC intermediate state (Nature 2024)
    log("Auxiliary heads created: pathway (14), proliferation (Ki67), IL1B, KAC (Nature 2024)")

    # Wrap auxiliary heads with DDP if distributed
    if distributed:
        pathway_head = DDP(pathway_head, device_ids=[local_rank], find_unused_parameters=True)
        prolif_head = DDP(prolif_head, device_ids=[local_rank], find_unused_parameters=True)
        il1b_head = DDP(il1b_head, device_ids=[local_rank], find_unused_parameters=True)
        kac_head = DDP(kac_head, device_ids=[local_rank], find_unused_parameters=True)

    # Create optimizer and scaler (include auxiliary head parameters)
    all_params = (
        list(model.parameters())
        + list(pathway_head.parameters())
        + list(prolif_head.parameters())
        + list(il1b_head.parameters())
        + list(kac_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        all_params,
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
        # Train with SSL objective + auxiliary losses
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config, epoch, phase="ssl",
            pathway_head=pathway_head, prolif_head=prolif_head, il1b_head=il1b_head,
            kac_head=kac_head,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device, config, phase="ssl",
            pathway_head=pathway_head, prolif_head=prolif_head, il1b_head=il1b_head,
            kac_head=kac_head,
        )

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
    if config.freeze_encoder:
        log("Mode: FROZEN encoder (SSL representation transfer test)")
    else:
        log("Mode: Fine-tuning (end-to-end optimization)")
    log(f"{'=' * 60}\n")

    # Optionally freeze encoder for ablation (tests SSL representation quality)
    if config.freeze_encoder:
        actual_model = model.module if hasattr(model, "module") else model
        frozen_count = 0
        # Freeze encoder components (niche_encoder, context_encoder, context_projection, wes_proj)
        for name in ["niche_encoder", "context_encoder", "context_projection", "wes_proj", "token_type_embedding"]:
            if hasattr(actual_model, name):
                for param in getattr(actual_model, name).parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
        log(f"Froze {frozen_count:,} encoder parameters")

    # Reset optimizer for transition phase
    # Only include trainable parameters
    transition_lr = config.learning_rate * 0.1 if not config.freeze_encoder else config.learning_rate
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    all_params = (
        trainable_params
        + list(pathway_head.parameters())
        + list(prolif_head.parameters())
        + list(il1b_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        all_params,
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

        # Train with transition objective + auxiliary losses
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            global_epoch,
            phase="transition",
            pathway_head=pathway_head,
            prolif_head=prolif_head,
            il1b_head=il1b_head,
            kac_head=kac_head,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device, config, phase="transition",
            pathway_head=pathway_head, prolif_head=prolif_head, il1b_head=il1b_head,
            kac_head=kac_head,
        )

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
    parser.add_argument("--latent_dim", type=int, default=40)  # Must match fused embedding (HLCA 30 + LuCA 10)
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

    # Ablation flags
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder during transition phase (tests SSL representation transfer)",
    )

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
        freeze_encoder=args.freeze_encoder,
    )

    train(config)


if __name__ == "__main__":
    main()
