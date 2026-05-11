#!/usr/bin/env python3
"""Standalone DataParallel training script for transition phase.

Uses the same wrapper approach as HPO to properly distribute across GPUs.
Resumes from ssl_pretrained.pt and only trains transition phase.

Usage:
    python scripts/train_dp.py \
        --data-dir /data1/chaunzt1/stagebridge/processed/luad_evo/canonical \
        --output-dir /data1/chaunzt1/stagebridge/outputs/v1.1/full/fold_0/seed_44 \
        --checkpoint /data1/chaunzt1/stagebridge/outputs/v1.1/full/fold_0/seed_44/checkpoints/ssl_pretrained.pt \
        --hpo-params /data1/chaunzt1/stagebridge/outputs/v1.1/hpo/best_params.json \
        --epochs 100 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from stagebridge.loaders import create_dataloaders, AMICIBatch
from stagebridge.models import StageBridge, StageBridgeConfig


class TransitionWrapper(nn.Module):
    """Wrapper for DataParallel - puts encode + flow matching in forward().

    Takes pre-indexed/sampled tensors so DataParallel can properly scatter.
    All OT coupling and sampling happens BEFORE calling this.
    """

    def __init__(self, model: StageBridge):
        super().__init__()
        self.model = model

    def forward(
        self,
        # Pre-indexed receiver and neighbors for sampled pairs
        receiver: torch.Tensor,  # [N, D] sampled source receivers
        neighbors: torch.Tensor,  # [N, K, D] neighbors for sampled receivers
        distances: torch.Tensor,  # [N, K] distances for sampled receivers
        neighbor_mask: torch.Tensor,  # [N, K] mask for sampled receivers
        hlca: torch.Tensor,  # [N, D]
        luca: torch.Tensor,  # [N, D]
        pathway: torch.Tensor,  # [N, D]
        stats: torch.Tensor,  # [N, D]
        evolution_features: torch.Tensor | None,  # [N, E] or None
        # Pre-computed CFM targets
        x_t: torch.Tensor,  # [N, D] interpolated state
        u_t: torch.Tensor,  # [N, D] target velocity
        t: torch.Tensor,  # [N] time
        stage_pair_id: torch.Tensor,  # [N] stage pair indices
    ) -> torch.Tensor:
        """Returns per-sample flow matching loss."""
        # Encode niche for sampled receivers
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
            return_reconstruction=False,
        )

        # Predict velocity
        v_t = self.model.forward_vector_field(
            x_t=x_t,
            t=t,
            context=niche_output.context,
            stage_pair_id=stage_pair_id,
            context_tokens=niche_output.context_tokens,
        )

        # Per-sample MSE
        loss = F.mse_loss(v_t, u_t, reduction='none').mean(dim=-1)
        return loss


def sinkhorn_coupling(x_src: torch.Tensor, x_tgt: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """Compute Sinkhorn OT coupling between source and target."""
    n, m = x_src.shape[0], x_tgt.shape[0]

    # Cost matrix
    cost = torch.cdist(x_src.double(), x_tgt.double(), p=2).pow(2)

    # Log-domain Sinkhorn
    log_a = torch.full((n,), -torch.log(torch.tensor(n, dtype=torch.float64, device=x_src.device)), device=x_src.device)
    log_b = torch.full((m,), -torch.log(torch.tensor(m, dtype=torch.float64, device=x_src.device)), device=x_src.device)
    log_K = -cost / epsilon

    log_u = torch.zeros(n, dtype=torch.float64, device=x_src.device)
    log_v = torch.zeros(m, dtype=torch.float64, device=x_src.device)

    for _ in range(n_iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    coupling = torch.exp(log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0))
    return coupling.float()


def sample_from_coupling(coupling: torch.Tensor, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample source-target pairs from OT coupling."""
    flat_coupling = coupling.flatten()
    flat_coupling = flat_coupling / flat_coupling.sum()  # Normalize

    indices = torch.multinomial(flat_coupling, n_samples, replacement=True)
    src_idx = indices // coupling.shape[1]
    tgt_idx = indices % coupling.shape[1]

    return src_idx, tgt_idx


def sample_targets(batch: AMICIBatch, target_stage: int) -> torch.Tensor:
    """Sample target states from target stage population."""
    target_mask = batch.stage_idx == target_stage
    n_targets = target_mask.sum().item()

    if n_targets > 0:
        target_receivers = batch.receiver[target_mask]
        sample_idx = torch.randint(n_targets, (batch.receiver.shape[0],), device=batch.receiver.device)
        return target_receivers[sample_idx]
    else:
        return batch.receiver + 0.1 * torch.randn_like(batch.receiver)


def train_epoch(
    wrapper: nn.Module,
    train_loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    num_ot_pairs: int = 256,
    stage_pairs: list[tuple[int, int]] = None,
    sigma: float = 0.0,
) -> float:
    """Train one epoch with DataParallel and proper OT coupling."""
    if stage_pairs is None:
        stage_pairs = [(0, 1), (1, 2), (0, 2)]  # Default stage transitions

    wrapper.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"[Trans] E{epoch}")

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Sample stage pair
        pair_idx = torch.randint(len(stage_pairs), (1,)).item()
        stage_src, stage_tgt = stage_pairs[pair_idx]

        # Get target states from target stage population
        x1_full = sample_targets(batch, stage_tgt)

        # Compute OT coupling and sample pairs
        with torch.no_grad():
            coupling = sinkhorn_coupling(batch.receiver, x1_full)
            src_idx, tgt_idx = sample_from_coupling(coupling, num_ot_pairs)

        # Pre-index everything for the sampled pairs
        receiver_sampled = batch.receiver[src_idx]
        neighbors_sampled = batch.neighbors[src_idx]
        distances_sampled = batch.distances[src_idx]
        neighbor_mask_sampled = batch.neighbor_mask[src_idx]
        hlca_sampled = batch.hlca[src_idx]
        luca_sampled = batch.luca[src_idx]
        pathway_sampled = batch.pathway[src_idx]
        stats_sampled = batch.stats[src_idx]
        evolution_sampled = batch.evolution_features[src_idx] if batch.evolution_features is not None else None

        # Get paired targets
        x0 = receiver_sampled
        x1 = x1_full[tgt_idx]

        # Sample times and compute CFM interpolation
        t = torch.rand(num_ot_pairs, device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        if sigma > 0:
            noise_scale = sigma * (t * (1 - t)).sqrt().unsqueeze(1)
            x_t = x_t + noise_scale * torch.randn_like(x_t)
        u_t = x1 - x0

        stage_pair_id = torch.zeros(num_ot_pairs, dtype=torch.long, device=device)

        with autocast("cuda"):
            # Forward through DP wrapper with pre-indexed tensors
            loss_per_sample = wrapper(
                receiver=receiver_sampled,
                neighbors=neighbors_sampled,
                distances=distances_sampled,
                neighbor_mask=neighbor_mask_sampled,
                hlca=hlca_sampled,
                luca=luca_sampled,
                pathway=pathway_sampled,
                stats=stats_sampled,
                evolution_features=evolution_sampled,
                x_t=x_t,
                u_t=u_t,
                t=t,
                stage_pair_id=stage_pair_id,
            )
            loss = loss_per_sample.mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    wrapper: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_ot_pairs: int = 256,
    stage_pairs: list[tuple[int, int]] = None,
) -> float:
    """Validate with proper OT coupling."""
    if stage_pairs is None:
        stage_pairs = [(0, 1), (1, 2), (0, 2)]

    wrapper.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = batch.to(device)

        # Sample stage pair
        pair_idx = torch.randint(len(stage_pairs), (1,)).item()
        stage_src, stage_tgt = stage_pairs[pair_idx]

        # Get target states
        x1_full = sample_targets(batch, stage_tgt)

        # Compute OT coupling
        coupling = sinkhorn_coupling(batch.receiver, x1_full)
        src_idx, tgt_idx = sample_from_coupling(coupling, num_ot_pairs)

        # Pre-index everything
        receiver_sampled = batch.receiver[src_idx]
        neighbors_sampled = batch.neighbors[src_idx]
        distances_sampled = batch.distances[src_idx]
        neighbor_mask_sampled = batch.neighbor_mask[src_idx]
        hlca_sampled = batch.hlca[src_idx]
        luca_sampled = batch.luca[src_idx]
        pathway_sampled = batch.pathway[src_idx]
        stats_sampled = batch.stats[src_idx]
        evolution_sampled = batch.evolution_features[src_idx] if batch.evolution_features is not None else None

        x0 = receiver_sampled
        x1 = x1_full[tgt_idx]

        t = torch.rand(num_ot_pairs, device=device)
        x_t = (1 - t.unsqueeze(1)) * x0 + t.unsqueeze(1) * x1
        u_t = x1 - x0

        stage_pair_id = torch.zeros(num_ot_pairs, dtype=torch.long, device=device)

        with autocast("cuda"):
            loss_per_sample = wrapper(
                receiver=receiver_sampled,
                neighbors=neighbors_sampled,
                distances=distances_sampled,
                neighbor_mask=neighbor_mask_sampled,
                hlca=hlca_sampled,
                luca=luca_sampled,
                pathway=pathway_sampled,
                stats=stats_sampled,
                evolution_features=evolution_sampled,
                x_t=x_t,
                u_t=u_t,
                t=t,
                stage_pair_id=stage_pair_id,
            )
            loss = loss_per_sample.mean()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="DataParallel transition training")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ssl_pretrained.pt")
    parser.add_argument("--hpo-params", type=Path, required=True)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="Override HPO learning rate")
    args = parser.parse_args()

    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")

    # Load HPO params
    with open(args.hpo_params) as f:
        hpo_params = json.load(f)
    print(f"Loaded HPO params: {hpo_params}")

    lr = args.lr or hpo_params.get("lr", 6.7e-5)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build config from checkpoint
    config = StageBridgeConfig.from_checkpoint(checkpoint)

    # Load data to get evolution_dim
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir, fold_idx=args.fold_idx, batch_size=args.batch_size
    )
    sample_batch = next(iter(train_loader))
    if sample_batch.evolution_features is not None:
        evolution_dim = sample_batch.evolution_features.shape[-1]
        if config.evolution_dim != evolution_dim:
            config = StageBridgeConfig(
                **{k: v for k, v in config.__dict__.items() if k != 'evolution_dim'},
                evolution_dim=evolution_dim,
            )
        print(f"Detected evolution_dim={evolution_dim}")

    # Reload dataloader (consumed one batch)
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir, fold_idx=args.fold_idx, batch_size=args.batch_size
    )

    # Build model and load weights
    model = StageBridge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Wrap for DataParallel
    wrapper = TransitionWrapper(model)
    if n_gpus > 1:
        wrapper = nn.DataParallel(wrapper)
    wrapper.to(device)

    # Optimizer
    optimizer = AdamW(wrapper.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler("cuda")

    # Checkpointing
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\nStarting transition training for {args.epochs} epochs")
    print(f"Learning rate: {lr:.2e}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(args.epochs):
        train_loss = train_epoch(wrapper, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(wrapper, val_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Get raw model from wrapper
            raw_model = wrapper.module.model if hasattr(wrapper, 'module') else wrapper.model
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": {"val_loss": val_loss, "train_loss": train_loss},
                "config": {"model_config": config.__dict__},
            }, checkpoint_dir / "best_checkpoint.pt")
            print(f"  -> New best! Saved to best_checkpoint.pt")

    # Save final
    raw_model = wrapper.module.model if hasattr(wrapper, 'module') else wrapper.model
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "metrics": {"val_loss": val_loss, "train_loss": train_loss},
        "config": {"model_config": config.__dict__},
    }, checkpoint_dir / "final_checkpoint.pt")

    # Write summary
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_loss,
        "epochs": args.epochs,
        "n_gpus": n_gpus,
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
