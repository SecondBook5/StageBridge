"""Dynamic Driver Index computation inspired by GeoBridge.

Reference: GeoBridge (bioRxiv 2026) - "Generating and navigating single cell
dynamics via a geodesic bridge between nonlinear transcriptional and linear
latent manifolds"

The Dynamic Driver Index (DI) identifies genes that drive cellular state
transitions by computing the dot product between trajectory velocity and
gene expression gradients:

    DI(g) = dz/dx_g * dz/dt

Where:
- dz/dt is the velocity in latent space along the trajectory
- dz/dx_g is the gradient of the latent embedding with respect to gene g

A positive DI indicates the gene promotes the transition.
A negative DI indicates the gene inhibits the transition.

This provides interpretable, gene-level attribution of what drives
stage-to-stage progression in LUAD.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DriverIndexResult:
    """Results from dynamic driver index computation."""

    driver_index: Tensor  # [n_genes] or [B, n_genes] - per-gene driver scores
    top_drivers: list[tuple[str, float]]  # Top promoting genes
    top_inhibitors: list[tuple[str, float]]  # Top inhibiting genes
    velocity_norm: float  # Magnitude of transition velocity
    gene_names: list[str]  # Gene names for interpretation


def compute_dynamic_driver_index(
    model: nn.Module,
    source_expression: Tensor,
    target_expression: Tensor,
    gene_names: list[str],
    latent_encoder: nn.Module | None = None,
    n_interpolation_steps: int = 10,
    top_k: int = 50,
) -> DriverIndexResult:
    """Compute GeoBridge-style dynamic driver index.

    Identifies genes that drive the transition from source to target state.

    Args:
        model: Model with encode() method that maps expression to latent
        source_expression: [B, n_genes] source state expression
        target_expression: [B, n_genes] target state expression
        gene_names: List of gene names corresponding to expression columns
        latent_encoder: Optional separate encoder (if model doesn't have encode())
        n_interpolation_steps: Number of steps for trajectory interpolation
        top_k: Number of top drivers/inhibitors to return

    Returns:
        DriverIndexResult with gene-level driver scores
    """
    device = source_expression.device
    batch_size, n_genes = source_expression.shape

    # Get encoder
    if latent_encoder is not None:
        encoder = latent_encoder
    elif hasattr(model, "encode"):
        encoder = model.encode
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Model must have encode() method or provide latent_encoder")

    # Compute latent representations
    with torch.no_grad():
        z_source = encoder(source_expression)
        z_target = encoder(target_expression)

    # Trajectory velocity in latent space: dz/dt = z_target - z_source
    velocity = z_target - z_source  # [B, latent_dim]
    velocity_norm = velocity.norm(dim=-1).mean().item()

    # Compute gene-level gradients via finite differences
    # For each gene, perturb and measure change in latent
    driver_index = torch.zeros(batch_size, n_genes, device=device)
    epsilon = 0.1  # Perturbation magnitude

    # Use midpoint of trajectory for gradient computation
    midpoint_expr = (source_expression + target_expression) / 2

    for g in range(n_genes):
        # Perturb gene g
        perturbed_plus = midpoint_expr.clone()
        perturbed_plus[:, g] += epsilon

        perturbed_minus = midpoint_expr.clone()
        perturbed_minus[:, g] -= epsilon

        # Compute latent gradient: dz/dx_g
        with torch.no_grad():
            z_plus = encoder(perturbed_plus)
            z_minus = encoder(perturbed_minus)

        dz_dx_g = (z_plus - z_minus) / (2 * epsilon)  # [B, latent_dim]

        # Dynamic driver index: DI(g) = dz/dx_g dot dz/dt
        # Dot product gives alignment between gene gradient and trajectory
        di_g = (dz_dx_g * velocity).sum(dim=-1)  # [B]
        driver_index[:, g] = di_g

    # Average across batch
    mean_di = driver_index.mean(dim=0)  # [n_genes]

    # Get top drivers (positive DI) and inhibitors (negative DI)
    sorted_idx = mean_di.argsort(descending=True)
    top_drivers = [
        (gene_names[idx.item()], mean_di[idx].item())
        for idx in sorted_idx[:top_k]
        if mean_di[idx] > 0
    ]

    sorted_idx_asc = mean_di.argsort()
    top_inhibitors = [
        (gene_names[idx.item()], mean_di[idx].item())
        for idx in sorted_idx_asc[:top_k]
        if mean_di[idx] < 0
    ]

    return DriverIndexResult(
        driver_index=mean_di,
        top_drivers=top_drivers,
        top_inhibitors=top_inhibitors,
        velocity_norm=velocity_norm,
        gene_names=gene_names,
    )


def compute_driver_index_along_trajectory(
    model: nn.Module,
    trajectory_expressions: list[Tensor],
    gene_names: list[str],
    stage_names: list[str] | None = None,
    latent_encoder: nn.Module | None = None,
    top_k: int = 50,
) -> dict[str, DriverIndexResult]:
    """Compute driver index for each transition along a trajectory.

    Args:
        model: Model with encode() method
        trajectory_expressions: List of [B, n_genes] expression at each stage
        gene_names: List of gene names
        stage_names: Optional names for each stage (e.g., ["Normal", "AAH", "AIS", "MIA", "ADC"])
        latent_encoder: Optional separate encoder
        top_k: Number of top drivers to return

    Returns:
        Dict mapping transition name to DriverIndexResult
    """
    n_stages = len(trajectory_expressions)
    if stage_names is None:
        stage_names = [f"Stage_{i}" for i in range(n_stages)]

    results = {}
    for i in range(n_stages - 1):
        transition_name = f"{stage_names[i]}_to_{stage_names[i + 1]}"
        results[transition_name] = compute_dynamic_driver_index(
            model=model,
            source_expression=trajectory_expressions[i],
            target_expression=trajectory_expressions[i + 1],
            gene_names=gene_names,
            latent_encoder=latent_encoder,
            top_k=top_k,
        )

    return results


def compute_driver_index_efficient(
    encoder: nn.Module,
    source_latent: Tensor,
    target_latent: Tensor,
    expression: Tensor,
    gene_names: list[str],
    top_k: int = 50,
) -> DriverIndexResult:
    """Compute driver index using pre-computed latents and Jacobian.

    More efficient version that avoids repeated forward passes by using
    autograd to compute the Jacobian directly.

    Args:
        encoder: Encoder module (expression -> latent)
        source_latent: [B, latent_dim] pre-computed source latents
        target_latent: [B, latent_dim] pre-computed target latents
        expression: [B, n_genes] expression at evaluation point
        gene_names: List of gene names
        top_k: Number of top drivers to return

    Returns:
        DriverIndexResult with gene-level driver scores
    """
    device = expression.device
    batch_size, n_genes = expression.shape

    # Trajectory velocity
    velocity = target_latent - source_latent  # [B, latent_dim]
    velocity_norm = velocity.norm(dim=-1).mean().item()

    # Enable gradient computation for expression
    expression_grad = expression.clone().requires_grad_(True)

    # Forward pass
    z = encoder(expression_grad)  # [B, latent_dim]

    # Compute Jacobian via backprop for each latent dimension
    latent_dim = z.shape[-1]
    jacobian = torch.zeros(batch_size, latent_dim, n_genes, device=device)

    for d in range(latent_dim):
        # Select d-th latent dimension
        grad_outputs = torch.zeros_like(z)
        grad_outputs[:, d] = 1.0

        # Backprop to get dz_d/dx for all genes
        if expression_grad.grad is not None:
            expression_grad.grad.zero_()

        z_d_grad = torch.autograd.grad(
            outputs=z,
            inputs=expression_grad,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
        )[0]  # [B, n_genes]

        jacobian[:, d, :] = z_d_grad

    # Dynamic driver index: DI(g) = sum_d (dz_d/dx_g * velocity_d)
    # = jacobian.T @ velocity
    driver_index = torch.einsum("bdg,bd->bg", jacobian, velocity)  # [B, n_genes]

    # Average across batch
    mean_di = driver_index.mean(dim=0)  # [n_genes]

    # Get top drivers and inhibitors
    sorted_idx = mean_di.argsort(descending=True)
    top_drivers = [
        (gene_names[idx.item()], mean_di[idx].item())
        for idx in sorted_idx[:top_k]
        if mean_di[idx] > 0
    ]

    sorted_idx_asc = mean_di.argsort()
    top_inhibitors = [
        (gene_names[idx.item()], mean_di[idx].item())
        for idx in sorted_idx_asc[:top_k]
        if mean_di[idx] < 0
    ]

    return DriverIndexResult(
        driver_index=mean_di,
        top_drivers=top_drivers,
        top_inhibitors=top_inhibitors,
        velocity_norm=velocity_norm,
        gene_names=gene_names,
    )


# Convenience function for LUAD stages
LUAD_STAGES = ["Normal", "AAH", "AIS", "MIA", "ADC"]


def analyze_luad_progression_drivers(
    model: nn.Module,
    stage_expressions: dict[str, Tensor],
    gene_names: list[str],
    latent_encoder: nn.Module | None = None,
    top_k: int = 50,
) -> dict[str, DriverIndexResult]:
    """Analyze driver genes for each LUAD progression transition.

    Args:
        model: Trained StageBridge model
        stage_expressions: Dict mapping stage name to [B, n_genes] expression
        gene_names: List of gene names
        latent_encoder: Optional encoder
        top_k: Number of top drivers

    Returns:
        Dict with driver analysis for each transition:
        - Normal_to_AAH
        - AAH_to_AIS
        - AIS_to_MIA
        - MIA_to_ADC
    """
    # Order stages
    ordered_expressions = []
    ordered_names = []
    for stage in LUAD_STAGES:
        if stage in stage_expressions:
            ordered_expressions.append(stage_expressions[stage])
            ordered_names.append(stage)

    return compute_driver_index_along_trajectory(
        model=model,
        trajectory_expressions=ordered_expressions,
        gene_names=gene_names,
        stage_names=ordered_names,
        latent_encoder=latent_encoder,
        top_k=top_k,
    )
