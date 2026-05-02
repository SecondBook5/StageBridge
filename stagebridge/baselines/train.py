"""CLI entrypoint for training baseline models.

Usage:
    python -m stagebridge.baselines.train \
        --baseline deepsets \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --fold-idx 0 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from stagebridge.baselines import get_baseline
from stagebridge.loaders import create_dataloaders
from stagebridge.contracts import N_STAGES


def _convert_rings_to_neighbors(
    ring_cells: list[torch.Tensor],
    ring_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ring-based format to flat neighbors format for baselines.

    Baselines expect:
        neighbors: [B, K, D] - flat tensor of all neighbors
        neighbor_mask: [B, K] - boolean mask

    NicheBatch provides:
        ring_cells: List of 4 tensors, each [B, max_cells_per_ring, D]
        ring_masks: List of 4 tensors, each [B, max_cells_per_ring] boolean

    Args:
        ring_cells: List of 4 tensors of ring cell embeddings
        ring_masks: List of 4 tensors of validity masks

    Returns:
        (neighbors, neighbor_mask) in flat format
    """
    # Concatenate all rings along the sequence dimension
    neighbors = torch.cat(ring_cells, dim=1)  # [B, 4*max_cells, D]
    neighbor_mask = torch.cat(ring_masks, dim=1)  # [B, 4*max_cells]
    return neighbors, neighbor_mask


def train_baseline(
    baseline_name: str,
    data_dir: Path,
    output_dir: Path,
    fold_idx: int = 0,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 64,
) -> dict:
    """Train a baseline model and save results."""
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {baseline_name} on {device}, fold {fold_idx}")

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        data_dir, fold_idx=fold_idx, batch_size=batch_size
    )

    # Create baseline model
    model = get_baseline(baseline_name, input_dim=40, hidden_dim=128)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    # Stage pair for flow matching (use 0->1 as default)
    default_stage_pair = 0 * N_STAGES + 1  # Normal -> Preinvasive

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Convert ring format to flat neighbors format for baselines
            neighbors, neighbor_mask = _convert_rings_to_neighbors(
                batch.ring_cells, batch.ring_masks
            )

            # Get context from baseline model
            if hasattr(model, "encode_context"):
                context = model.encode_context(
                    receiver=batch.receiver,
                    neighbors=neighbors,
                    neighbor_mask=neighbor_mask,
                )
            else:
                context = batch.receiver

            # Simple flow matching loss on receiver reconstruction
            t = torch.rand(batch.receiver.shape[0], 1, device=device)
            noise = torch.randn_like(batch.receiver)
            x_t = (1 - t) * noise + t * batch.receiver

            # Create stage_pair_id tensor
            stage_pair_id = torch.full(
                (batch.receiver.shape[0],), default_stage_pair,
                dtype=torch.long, device=device
            )

            if hasattr(model, "forward_vector_field"):
                v_pred = model.forward_vector_field(
                    x_t, t.squeeze(-1), context, stage_pair_id
                )
            else:
                v_pred = model(x_t, t.squeeze(-1), context)

            v_target = batch.receiver - noise
            loss = ((v_pred - v_target) ** 2).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Convert ring format to flat neighbors format
                neighbors, neighbor_mask = _convert_rings_to_neighbors(
                    batch.ring_cells, batch.ring_masks
                )

                if hasattr(model, "encode_context"):
                    context = model.encode_context(
                        receiver=batch.receiver,
                        neighbors=neighbors,
                        neighbor_mask=neighbor_mask,
                    )
                else:
                    context = batch.receiver

                t = torch.rand(batch.receiver.shape[0], 1, device=device)
                noise = torch.randn_like(batch.receiver)
                x_t = (1 - t) * noise + t * batch.receiver

                # Create stage_pair_id tensor
                stage_pair_id = torch.full(
                    (batch.receiver.shape[0],), default_stage_pair,
                    dtype=torch.long, device=device
                )

                if hasattr(model, "forward_vector_field"):
                    v_pred = model.forward_vector_field(
                        x_t, t.squeeze(-1), context, stage_pair_id
                    )
                else:
                    v_pred = model(x_t, t.squeeze(-1), context)

                v_target = batch.receiver - noise
                loss = ((v_pred - v_target) ** 2).mean()
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                checkpoint_dir / f"{baseline_name}_final.pt",
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")

    # Save results
    result = {
        "baseline": baseline_name,
        "fold_idx": fold_idx,
        "seed": seed,
        "n_parameters": n_params,
        "epochs": epochs,
        "metrics": {
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1],
            "val_loss": avg_val_loss,
        },
        "history": history,
        "completed_at": datetime.now().isoformat(),
    }

    result_path = output_dir / f"baseline_{baseline_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--baseline", required=True, help="Baseline name")
    parser.add_argument("--data-dir", required=True, type=Path, help="Data directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--fold-idx", type=int, default=0, help="Fold index")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train_baseline(
        baseline_name=args.baseline,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fold_idx=args.fold_idx,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
