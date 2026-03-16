#!/usr/bin/env python3
"""
V1 Synthetic Data Pipeline

End-to-end test of StageBridge V1 architecture on synthetic data.

This script:
1. Generates synthetic dataset
2. Loads data with canonical loaders
3. Initializes all model layers (A-F)
4. Runs training loop
5. Evaluates with metrics
6. Produces visualizations

Purpose: Validate implementation before HPC deployment on real data.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

# StageBridge imports
from stagebridge.data.synthetic import generate_synthetic_dataset
from stagebridge.data.loaders_optimized import get_dataloader_optimized, StageBridgeBatch
from stagebridge.models.dual_reference import create_dual_reference_mapper
from stagebridge.context_model.local_niche_encoder import LocalNicheMLPEncoder
from stagebridge.context_model.set_encoder import SetTransformer


class SimpleWESRegularizer(nn.Module):
    """
    Simplified WES compatibility regularizer for V1 synthetic testing.

    Encourages matched donor transitions to have higher compatibility
    than mismatched donor transitions.
    """

    def __init__(self, wes_dim: int = 3, hidden_dim: int = 64, temperature: float = 0.1):
        super().__init__()

        self.temperature = temperature

        # Project WES features to compatibility scores
        self.compat_net = nn.Sequential(
            nn.Linear(wes_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
        wes_source: torch.Tensor,
        wes_target: torch.Tensor,
        has_wes_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute WES compatibility loss.

        Args:
            z_source: Source latent (B, latent_dim)
            z_target: Target latent (B, latent_dim)
            wes_source: Source WES features (B, wes_dim)
            wes_target: Target WES features (B, wes_dim)
            has_wes_mask: Boolean mask for valid WES (B,)

        Returns:
            Loss scalar
        """
        if not has_wes_mask.any():
            return torch.tensor(0.0, device=z_source.device)

        # Concatenate WES features
        wes_concat = torch.cat([wes_source, wes_target], dim=-1)

        # Compute compatibility score
        compat = self.compat_net(wes_concat).squeeze(-1)

        # Contrastive loss: maximize compatibility for matched pairs
        # For synthetic data, we assume all pairs are matched (same donor)
        # So we just minimize -log(sigmoid(compat))
        import torch.nn.functional as F
        loss = -torch.mean(F.logsigmoid(compat / self.temperature)[has_wes_mask])

        return loss


class SimpleFlowMatchingTransition(nn.Module):
    """
    Simplified flow matching transition model for V1 synthetic testing.

    Uses conditional flow matching with learned drift function.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        context_dim: int = 128,
        hidden_dims: list = None,
        time_embedding_dim: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.context_dim = context_dim
        hidden_dims = hidden_dims or [128, 128]

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
        )

        # Drift network: v_t(x_t, context)
        layers = []
        input_dim = latent_dim + context_dim + time_embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, latent_dim))
        self.drift_net = nn.Sequential(*layers)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        context: torch.Tensor,
        return_trajectory: bool = False,
    ):
        """
        Compute flow matching loss.

        Args:
            x0: Source latent (B, latent_dim)
            x1: Target latent (B, latent_dim)
            context: Context embedding (B, context_dim)
            return_trajectory: Return sampled trajectory

        Returns:
            Dictionary with loss and optionally trajectory
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Sample random time
        t = torch.rand(batch_size, 1, device=device)

        # Conditional flow: x_t = t * x1 + (1 - t) * x0
        x_t = t * x1 + (1 - t) * x0

        # Target velocity: dx/dt = x1 - x0
        v_target = x1 - x0

        # Predict velocity
        t_embed = self.time_embed(t)
        drift_input = torch.cat([x_t, context, t_embed], dim=-1)
        v_pred = self.drift_net(drift_input)

        # MSE loss
        loss = torch.mean((v_pred - v_target) ** 2)

        # Predict x1 from x0
        with torch.no_grad():
            x1_pred = self.sample(x0, context, n_steps=10)

        results = {
            "loss": loss,
            "x1_pred": x1_pred,
        }

        if return_trajectory:
            trajectory = self.sample_trajectory(x0, context, n_steps=20)
            results["trajectory"] = trajectory

        return results

    def sample(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Sample transition trajectory using ODE integration.

        Args:
            x0: Source latent (B, latent_dim)
            context: Context embedding (B, context_dim)
            n_steps: Number of integration steps

        Returns:
            x1: Predicted target latent (B, latent_dim)
        """
        dt = 1.0 / n_steps
        x_t = x0

        for step in range(n_steps):
            t = torch.full((x0.shape[0], 1), step * dt, device=x0.device)
            t_embed = self.time_embed(t)
            drift_input = torch.cat([x_t, context, t_embed], dim=-1)
            v_t = self.drift_net(drift_input)
            x_t = x_t + v_t * dt

        return x_t

    def sample_trajectory(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Sample full trajectory."""
        trajectory = [x0]
        dt = 1.0 / n_steps
        x_t = x0

        for step in range(n_steps):
            t = torch.full((x0.shape[0], 1), step * dt, device=x0.device)
            t_embed = self.time_embed(t)
            drift_input = torch.cat([x_t, context, t_embed], dim=-1)
            v_t = self.drift_net(drift_input)
            x_t = x_t + v_t * dt
            trajectory.append(x_t)

        return torch.stack(trajectory, dim=1)  # (B, n_steps+1, latent_dim)


class StageBridgeV1Model(nn.Module):
    """
    Full StageBridge V1 model integrating all layers.

    Architecture:
    - Layer A: Dual-Reference Latent (precomputed for synthetic)
    - Layer B: Local Niche Encoder (9-token transformer)
    - Layer C: Hierarchical Set Transformer (ISAB/SAB/PMA)
    - Layer D: Stochastic Transition Model (Flow Matching)
    - Layer F: Evolutionary Compatibility (WES regularizer)
    """

    def __init__(
        self,
        latent_dim: int = 2,
        niche_hidden_dim: int = 64,
        niche_heads: int = 4,
        set_hidden_dim: int = 128,
        set_heads: int = 4,
        n_inducing: int = 16,
        wes_dim: int = 3,
        use_wes: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_wes = use_wes

        # Layer A: Dual-Reference (precomputed for synthetic)
        self.dual_reference = create_dual_reference_mapper(
            mode="precomputed",
            latent_dim=latent_dim,
        )

        # Layer B: Local Niche Encoder (9 tokens → flattened)
        # For V1 synthetic: use simple MLP encoder
        niche_token_dim = latent_dim + 4  # latent + extra features
        self.niche_encoder = LocalNicheMLPEncoder(
            input_dim=9 * niche_token_dim,  # 9 tokens flattened
            hidden_dim=niche_hidden_dim,
            dropout=0.1,
        )

        # Layer C: Set Transformer (hierarchical aggregation)
        self.set_transformer = SetTransformer(
            dim_input=niche_hidden_dim,
            dim_hidden=set_hidden_dim,
            dim_output=set_hidden_dim,
            num_heads=set_heads,
            num_inds=n_inducing,
            ln=True,
        )

        # Layer D: Flow Matching Transition Model
        # Use niche_hidden_dim since we're not using Set Transformer in V1 synthetic
        self.transition_model = SimpleFlowMatchingTransition(
            latent_dim=latent_dim,
            context_dim=niche_hidden_dim,  # Changed from set_hidden_dim
            hidden_dims=[128, 128],
            time_embedding_dim=32,
        )

        # Layer F: WES Compatibility Regularizer
        if use_wes:
            self.wes_regularizer = SimpleWESRegularizer(
                wes_dim=wes_dim,
                hidden_dim=64,
                temperature=0.1,
            )

    def forward(
        self,
        batch: StageBridgeBatch,
        return_trajectory: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all layers.

        Args:
            batch: Input batch
            return_trajectory: Return full ODE trajectory

        Returns:
            Dictionary with:
            - z_pred: Predicted target latent
            - loss_transition: Transition loss
            - loss_wes: WES compatibility loss (if enabled)
            - trajectory: Full trajectory (if requested)
        """
        # Layer A: Already computed (z_source, z_target in batch)
        z_source = batch.z_source  # (B, latent_dim)
        z_target = batch.z_target  # (B, latent_dim)

        # Layer B: Encode 9-token neighborhoods
        niche_tokens = batch.niche_tokens  # (B, 9, token_dim)
        niche_mask = batch.niche_mask  # (B, 9)

        # Flatten tokens for MLP encoder
        batch_size = niche_tokens.shape[0]
        niche_flat = niche_tokens.reshape(batch_size, -1)  # (B, 9 * token_dim)

        # Encode each cell's niche
        niche_output = self.niche_encoder(niche_flat)
        niche_encoded = niche_output.token_embeddings  # (B, 1, hidden_dim)

        # Layer C: Hierarchical set aggregation
        # For V1: use niche embedding directly (already pooled by MLP)
        niche_context = niche_encoded.squeeze(1)  # (B, hidden_dim)

        # Layer D: Flow matching transition
        outputs = self.transition_model(
            x0=z_source,
            x1=z_target,
            context=niche_context,
            return_trajectory=return_trajectory,
        )

        loss_transition = outputs["loss"]
        z_pred = outputs["x1_pred"]

        results = {
            "z_pred": z_pred,
            "loss_transition": loss_transition,
        }

        if return_trajectory:
            results["trajectory"] = outputs["trajectory"]

        # Layer F: WES compatibility regularizer
        if self.use_wes and batch.wes_features is not None:
            wes_loss = self.wes_regularizer(
                z_source=z_source,
                z_target=z_pred,
                wes_source=batch.wes_features,
                wes_target=batch.wes_features,  # Same donor for synthetic
                has_wes_mask=batch.has_wes,
            )
            results["loss_wes"] = wes_loss
        else:
            results["loss_wes"] = torch.tensor(0.0, device=z_source.device)

        return results

    def sample_transition(
        self,
        z_source: torch.Tensor,
        niche_tokens: torch.Tensor,
        niche_mask: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Sample stochastic transition trajectory.

        Args:
            z_source: Source latent (B, latent_dim)
            niche_tokens: Niche tokens (B, 9, token_dim)
            niche_mask: Token mask (B, 9)
            n_steps: Number of ODE steps

        Returns:
            z_target: Predicted target latent (B, latent_dim)
        """
        # Flatten and encode niche
        batch_size = niche_tokens.shape[0]
        niche_flat = niche_tokens.reshape(batch_size, -1)
        niche_output = self.niche_encoder(niche_flat)
        niche_context = niche_output.token_embeddings.squeeze(1)

        # Sample transition
        z_target = self.transition_model.sample(
            x0=z_source,
            context=niche_context,
            n_steps=n_steps,
        )

        return z_target


def train_epoch(
    model: StageBridgeV1Model,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    wes_weight: float = 0.1,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_transition = 0.0
    total_wes = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Combined loss
        loss = outputs["loss_transition"] + wes_weight * outputs["loss_wes"]

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_transition += outputs["loss_transition"].item()
        total_wes += outputs["loss_wes"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": total_loss / n_batches,
            "transition": total_transition / n_batches,
            "wes": total_wes / n_batches,
        })

    return {
        "loss": total_loss / n_batches,
        "loss_transition": total_transition / n_batches,
        "loss_wes": total_wes / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: StageBridgeV1Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    z_preds = []
    z_targets = []
    n_batches = 0

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)

        outputs = model(batch)

        total_loss += outputs["loss_transition"].item()
        z_preds.append(outputs["z_pred"].cpu())
        z_targets.append(batch.z_target.cpu())
        n_batches += 1

    z_preds = torch.cat(z_preds, dim=0)
    z_targets = torch.cat(z_targets, dim=0)

    # Compute MSE
    mse = torch.mean((z_preds - z_targets) ** 2).item()

    # Compute Wasserstein-1 (approximation)
    distances = torch.norm(z_preds - z_targets, dim=1)
    wasserstein = torch.mean(distances).item()

    return {
        "loss": total_loss / n_batches,
        "mse": mse,
        "wasserstein": wasserstein,
    }


def visualize_transitions(
    model: StageBridgeV1Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Path,
):
    """Visualize predicted transitions in 2D latent space."""
    model.eval()

    z_sources = []
    z_targets = []
    z_preds = []
    stages = []

    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting for viz"):
            batch = batch.to(device)
            outputs = model(batch)

            z_sources.append(batch.z_source.cpu().numpy())
            z_targets.append(batch.z_target.cpu().numpy())
            z_preds.append(outputs["z_pred"].cpu().numpy())
            stages.extend(batch.source_stages)

            # Limit for visualization
            if len(z_sources) > 10:
                break

    z_sources = np.concatenate(z_sources, axis=0)
    z_targets = np.concatenate(z_targets, axis=0)
    z_preds = np.concatenate(z_preds, axis=0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    ax = axes[0]
    ax.scatter(z_sources[:, 0], z_sources[:, 1], c="blue", alpha=0.5, label="Source")
    ax.scatter(z_targets[:, 0], z_targets[:, 1], c="red", alpha=0.5, label="Target (GT)")
    for i in range(min(50, len(z_sources))):
        ax.arrow(
            z_sources[i, 0], z_sources[i, 1],
            z_targets[i, 0] - z_sources[i, 0],
            z_targets[i, 1] - z_sources[i, 1],
            alpha=0.3, head_width=0.05, color="gray",
        )
    ax.set_title("Ground Truth Transitions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Predicted
    ax = axes[1]
    ax.scatter(z_sources[:, 0], z_sources[:, 1], c="blue", alpha=0.5, label="Source")
    ax.scatter(z_preds[:, 0], z_preds[:, 1], c="green", alpha=0.5, label="Target (Pred)")
    for i in range(min(50, len(z_sources))):
        ax.arrow(
            z_sources[i, 0], z_sources[i, 1],
            z_preds[i, 0] - z_sources[i, 0],
            z_preds[i, 1] - z_sources[i, 1],
            alpha=0.3, head_width=0.05, color="gray",
        )
    ax.set_title("Predicted Transitions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="StageBridge V1 Synthetic Pipeline")
    parser.add_argument("--output_dir", type=str, default="outputs/synthetic_v1")
    parser.add_argument("--n_cells", type=int, default=1000)
    parser.add_argument("--n_donors", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wes_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("=" * 80)
    print("StageBridge V1 Synthetic Data Pipeline")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n[1/6] Generating synthetic dataset...")
    data_dir = generate_synthetic_dataset(
        output_dir="data/processed/synthetic",
        n_cells=args.n_cells,
        n_donors=args.n_donors,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )

    # Step 2: Create dataloaders
    print("\n[2/6] Creating dataloaders...")
    train_loader = get_dataloader_optimized(
        data_dir=data_dir,
        fold=0,
        split="train",
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        shuffle=True,
    )

    val_loader = get_dataloader_optimized(
        data_dir=data_dir,
        fold=0,
        split="val",
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        shuffle=False,
    )

    test_loader = get_dataloader_optimized(
        data_dir=data_dir,
        fold=0,
        split="test",
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        shuffle=False,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Step 3: Initialize model
    print("\n[3/6] Initializing model...")
    model = StageBridgeV1Model(
        latent_dim=args.latent_dim,
        niche_hidden_dim=64,
        set_hidden_dim=128,
        use_wes=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Step 4: Training loop
    print(f"\n[4/6] Training for {args.n_epochs} epochs...")
    history = {"train": [], "val": []}

    for epoch in range(args.n_epochs):
        print(f"\nEpoch {epoch + 1}/{args.n_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, wes_weight=args.wes_weight
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        history["val"].append(val_metrics)

        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val MSE: {val_metrics['mse']:.4f} | Val W-dist: {val_metrics['wasserstein']:.4f}")

        scheduler.step()

    # Step 5: Test evaluation
    print("\n[5/6] Testing...")
    test_metrics = evaluate(model, test_loader, device)

    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test MSE: {test_metrics['mse']:.4f}")
    print(f"  Test W-dist: {test_metrics['wasserstein']:.4f}")

    # Step 6: Visualizations
    print("\n[6/6] Generating visualizations...")
    visualize_transitions(
        model, test_loader, device,
        save_path=output_dir / "transitions_visualization.png"
    )

    # Save results
    results = {
        "args": vars(args),
        "history": history,
        "test_metrics": test_metrics,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    print("\n" + "=" * 80)
    print(" Pipeline complete!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
