"""Schrödinger bridge formulation for cross-dataset OT flow.

The Schrödinger bridge problem finds the most likely stochastic process
that transforms a source distribution π₀ into a target π₁ while
minimising KL divergence from a reference process (Brownian motion).

For StageBridge, this means:
  - π₀ = cell distribution at stage s (from any dataset)
  - π₁ = cell distribution at stage s+1 (from any dataset, possibly different)
  - Reference = Brownian motion in HLCA latent space

The key advantage over standard OT-CFM for cross-dataset flow:
  1. **Entropic regularization** naturally handles the uncertainty from
     merging different patient populations
  2. **Stochastic interpolant** accounts for biological noise in cell
     state transitions
  3. **Time-reversal symmetry** enables backward inference (metastasis
     back to primary, for counterfactual analysis)

Implementation follows the Iterative Proportional Fitting (IPF) / Sinkhorn
approach to SB, adapted for neural drift estimation.

References
----------
De Bortoli et al. "Diffusion Schrödinger Bridge" NeurIPS 2021
Shi et al. "Diffusion Schrödinger Bridge Matching" ICLR 2024
Tong et al. "Conditional Flow Matching" ICML 2023
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from stagebridge.transition_model.couplings import build_ot_coupling
from stagebridge.transition_model.gaussian_init import (
    GaussianBridgeMoments,
    build_gaussian_bridge,
    interpolate_bridge_moments,
)
from stagebridge.transition_model.losses import (
    build_sinkhorn_coupling,
    pairwise_squared_euclidean,
    sample_coupling_pairs,
)


def schrodinger_bridge_interpolant(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    sigma: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute Schrödinger bridge interpolant and target velocity.

    The SB interpolant is:
        x_t = (1-t)x₀ + t·x₁ + σ·√(t(1-t))·ε,  ε ~ N(0,I)

    The conditional velocity target (drift) is:
        u_t = (x₁ - x₀) + σ²·(1-2t)/(2·t(1-t))·(x_t - (1-t)x₀ - t·x₁)

    For σ→0 this reduces to the deterministic OT-CFM target u_t = x₁ - x₀.

    Parameters
    ----------
    x0 : (B, D) source points
    x1 : (B, D) target points
    t : (B,) or (B, 1) time values in [0, 1]
    sigma : float, noise scale

    Returns
    -------
    x_t : (B, D) interpolated points
    u_t : (B, D) target drift velocity
    noise : (B, D) the noise realization (for score matching if needed)
    """
    if t.ndim == 1:
        t = t.unsqueeze(1)  # (B, 1)

    # Interpolant mean
    mu_t = (1.0 - t) * x0 + t * x1

    # Noise scale: σ·√(t(1-t))
    t_clamped = t.clamp(1e-5, 1.0 - 1e-5)  # avoid division by zero
    noise_scale = sigma * (t_clamped * (1.0 - t_clamped)).sqrt()
    noise = torch.randn_like(x0)
    x_t = mu_t + noise_scale * noise

    # Conditional velocity (drift target)
    if sigma > 1e-8:
        # Score correction term for SB
        score_coeff = sigma ** 2 * (1.0 - 2.0 * t_clamped) / (2.0 * t_clamped * (1.0 - t_clamped))
        u_t = (x1 - x0) + score_coeff * (noise_scale * noise)
    else:
        u_t = x1 - x0

    return x_t, u_t, noise


def schrodinger_bridge_loss(
    batch: Any,
    model: Any,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
    num_ot_pairs: int = 512,
    sigma: float = 0.1,
    score_weight: float = 0.1,
    coupling: Tensor | None = None,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Schrödinger bridge loss with entropic OT coupling.

    This combines:
      1. Entropic OT coupling (Sinkhorn) for optimal pairing
      2. Stochastic interpolant with σ > 0 for biological noise
      3. Drift matching: MSE between predicted and SB-target velocity
      4. Optional score matching: learn the noise score for backward SDE

    Parameters
    ----------
    batch : StageBatch
    model : StageBridgeModel or compatible
    epsilon : Sinkhorn regularization
    sinkhorn_iters : number of Sinkhorn iterations
    num_ot_pairs : pairs to sample from coupling
    sigma : SB noise level (σ=0 reduces to OT-CFM)
    score_weight : weight for the score matching term
    coupling : pre-computed coupling matrix, or None

    Returns
    -------
    (loss, diagnostics, coupling)
    """
    x_src = batch.x_src
    x_tgt = batch.x_tgt
    x_set = batch.x_set if batch.x_set is not None else x_src
    device = x_src.device

    # Step 1: Entropic OT coupling
    if coupling is None:
        coupling = build_sinkhorn_coupling(
            x_src=x_src,
            x_tgt=x_tgt,
            epsilon=epsilon,
            n_iters=sinkhorn_iters,
        )
    src_idx, tgt_idx = sample_coupling_pairs(coupling=coupling, num_pairs=num_ot_pairs)

    x_i = x_src[src_idx]
    y_j = x_tgt[tgt_idx]

    # Step 2: Schrödinger bridge interpolant
    t = torch.rand((num_ot_pairs,), device=device, dtype=x_src.dtype)
    x_t, u_t, noise = schrodinger_bridge_interpolant(x_i, y_j, t, sigma=sigma)

    # Step 3: Get model prediction
    # Context from set transformer
    kwargs: dict[str, Any] = {}
    if batch.niche_coords is not None:
        kwargs["niche_coords"] = batch.niche_coords
    if batch.spatial_niche is not None:
        kwargs["spatial_niche"] = batch.spatial_niche
    c_s = model.forward_set_context(x_set, mask=batch.context_mask, **kwargs)

    if c_s.ndim == 2 and c_s.shape[0] == 1:
        c_rep = c_s.expand(num_ot_pairs, -1)
    else:
        c_rep = c_s

    stage_pair_id = torch.zeros((num_ot_pairs,), dtype=torch.long, device=device)
    if hasattr(model, "encode_stage_pair_tensor"):
        stage_pair_id = model.encode_stage_pair_tensor(
            stage_src=batch.stage_src,
            stage_tgt=batch.stage_tgt,
            n=num_ot_pairs,
            device=device,
        )

    pred_velocity = model.forward_vector_field(
        x_t=x_t, t=t, c_s=c_rep,
        stage_pair_id=stage_pair_id,
        wes_features=batch.wes_features,
        lr_features=batch.lr_features,
    )

    # Step 4: Drift matching loss
    loss_drift = F.mse_loss(pred_velocity, u_t)

    # Step 5: Score matching (optional, for backward SDE capability)
    loss_score = torch.tensor(0.0, device=device, dtype=x_src.dtype)
    if score_weight > 0.0 and sigma > 1e-8:
        # The score of the SB marginal at time t is proportional to -noise/noise_scale
        t_col = t.unsqueeze(1).clamp(1e-5, 1.0 - 1e-5)
        noise_scale = sigma * (t_col * (1.0 - t_col)).sqrt()
        target_score = -noise / noise_scale.clamp_min(1e-8)
        # Predict score as residual from velocity prediction
        pred_score = (pred_velocity - (y_j - x_i)) / (sigma ** 2 + 1e-8)
        loss_score = F.mse_loss(pred_score, target_score)

    total = loss_drift + score_weight * loss_score

    # Diagnostics
    with torch.no_grad():
        cost = pairwise_squared_euclidean(x_src, x_tgt)
        ot_cost = float((coupling * cost).sum().item())
        # Entropy of the coupling (higher = more diffuse matching)
        pi_flat = coupling.reshape(-1)
        pi_flat = pi_flat[pi_flat > 1e-12]
        entropy = float(-(pi_flat * pi_flat.log()).sum().item())

    diagnostics = {
        "loss_total": float(total.detach().item()),
        "loss_drift": float(loss_drift.detach().item()),
        "loss_score": float(loss_score.detach().item()),
        "ot_cost": ot_cost,
        "coupling_entropy": entropy,
        "sigma": sigma,
        "num_pairs": float(num_ot_pairs),
    }
    return total, diagnostics, coupling


def ipf_update_coupling(
    model: Any,
    x_src: Tensor,
    x_tgt: Tensor,
    epsilon: float = 0.05,
    sigma: float = 0.1,
    num_ipf_steps: int = 5,
    num_euler_steps: int = 8,
) -> Tensor:
    """Iterative Proportional Fitting update for the SB coupling.

    After training the drift network for a while, update the coupling
    by running the learned forward process and recomputing Sinkhorn on
    the transported distribution.  This is one step of the IPF algorithm
    for Schrödinger bridges.

    Parameters
    ----------
    model : trained StageBridgeModel
    x_src : (N, D) source distribution
    x_tgt : (M, D) target distribution
    epsilon : Sinkhorn regularization
    sigma : noise level
    num_ipf_steps : number of IPF iterations (typically 1-5)
    num_euler_steps : Euler steps for forward simulation

    Returns
    -------
    (N, M) updated coupling matrix
    """
    coupling = build_sinkhorn_coupling(x_src, x_tgt, epsilon=epsilon)

    for _ in range(num_ipf_steps):
        # Forward: transport source using current model
        with torch.no_grad():
            c_s = model.forward_set_context(x_src)
            if c_s.shape[0] == 1:
                c_s = c_s.expand(x_src.shape[0], -1)
            pair_id = torch.zeros(x_src.shape[0], dtype=torch.long, device=x_src.device)
            x_transported = model.integrate_euler(
                x0=x_src, c_s=c_s, stage_pair_id=pair_id,
                num_steps=num_euler_steps,
            )
            if sigma > 0:
                x_transported = x_transported + sigma * torch.randn_like(x_transported)

        # Recompute coupling between transported source and target
        coupling = build_sinkhorn_coupling(x_transported, x_tgt, epsilon=epsilon)

    return coupling


def edgewise_schrodinger_bridge_loss(
    model: Any,
    *,
    x_src: Tensor,
    x_tgt: Tensor,
    context: Tensor,
    edge_ids: Tensor,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
    num_ot_pairs: int = 128,
    sigma: float = 0.1,
    diffusion_weight: float = 0.1,
    extra_cost: Tensor | None = None,
    coupling: Tensor | None = None,
    bridge: GaussianBridgeMoments | None = None,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Bridge loss for explicit drift-diffusion models with edge-wise conditioning."""
    device = x_src.device
    if coupling is None:
        coupling = build_ot_coupling(
            x_src=x_src,
            x_tgt=x_tgt,
            epsilon=epsilon,
            n_iters=sinkhorn_iters,
            extra_cost=extra_cost,
        )
    src_idx, tgt_idx = sample_coupling_pairs(coupling=coupling, num_pairs=num_ot_pairs)
    x_i = x_src[src_idx]
    y_j = x_tgt[tgt_idx]
    sampled_edge_ids = edge_ids[src_idx] if edge_ids.shape[0] == x_src.shape[0] else edge_ids

    t = torch.rand((num_ot_pairs,), device=device, dtype=x_src.dtype)
    x_t, target_drift, _ = schrodinger_bridge_interpolant(x_i, y_j, t, sigma=sigma)

    pred_drift = model.forward_drift(x_t=x_t, t=t, context=context, edge_ids=sampled_edge_ids)
    pred_diffusion = model.forward_diffusion(x_t=x_t, t=t, context=context, edge_ids=sampled_edge_ids)

    if bridge is None:
        bridge = build_gaussian_bridge(x_src, x_tgt, sigma=sigma)
    bridge_moments = interpolate_bridge_moments(bridge, t)
    target_diffusion = bridge_moments.variance.sqrt()
    if target_diffusion.ndim == 1:
        target_diffusion = target_diffusion.unsqueeze(0).expand_as(pred_diffusion)

    loss_drift = F.mse_loss(pred_drift, target_drift)
    loss_diffusion = F.mse_loss(pred_diffusion, target_diffusion)
    total = loss_drift + float(diffusion_weight) * loss_diffusion

    with torch.no_grad():
        cost = pairwise_squared_euclidean(x_src, x_tgt)
        if extra_cost is not None:
            cost = cost + extra_cost.to(cost.dtype)
        ot_cost = float((coupling * cost).sum().item())

    diagnostics = {
        "loss_total": float(total.detach().item()),
        "loss_drift": float(loss_drift.detach().item()),
        "loss_diffusion": float(loss_diffusion.detach().item()),
        "ot_cost": ot_cost,
        "sigma": float(sigma),
        "num_pairs": float(num_ot_pairs),
    }
    return total, diagnostics, coupling
