"""Schrödinger Bridge for stochastic stage transitions.

Implements entropy-regularized OT on path space, learning both forward and
backward drift networks. Unlike OT-CFM (deterministic mean flow), SB models
the *distribution* of possible transitions, naturally capturing branching
(e.g., partial EMT vs full EMT) and reversibility (MET).

Key equations:
    Forward SDE:  dx = f(x,t)dt + σdW
    Backward SDE: dx = [f(x,t) - σ²∇log p_t(x)]dt + σd\bar{W}

The forward/backward drifts are jointly optimized via the IPF (iterative
proportional fitting) algorithm or score matching.

References:
    - De Bortoli et al. (2021): Diffusion Schrödinger Bridge
    - Tong et al. (2023): Conditional Flow Matching (comparison)
    - Chen et al. (2021): Likelihood Training of SB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from stagebridge.context.layers import SinusoidalTimeEmbedding


@dataclass
class SchrodingerBridgeConfig:
    """Configuration for Schrödinger Bridge dynamics.

    Attributes:
        input_dim: State dimension (40d fused embedding)
        context_dim: Niche context dimension
        hidden_dim: MLP hidden dimension
        time_dim: Time embedding dimension
        stage_dim: Stage embedding dimension
        num_stages: Number of disease stages
        sigma: Diffusion coefficient (noise level)
        num_ipf_iters: IPF iterations per training step (1 = single step DSB)
        use_score_matching: Use score matching loss (vs IPF)
        langevin_steps: Langevin corrector steps during sampling
        langevin_snr: Signal-to-noise ratio for Langevin
    """
    input_dim: int = 40
    context_dim: int = 256
    hidden_dim: int = 256
    time_dim: int = 64
    stage_dim: int = 32
    num_stages: int = 4
    sigma: float = 0.1
    num_ipf_iters: int = 1
    use_score_matching: bool = True
    langevin_steps: int = 0
    langevin_snr: float = 0.1


class DriftNetwork(nn.Module):
    """Drift network for forward or backward process.

    Predicts drift velocity conditioned on state, time, context, and stage.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_dim: int,
        stage_dim: int,
        num_stages: int,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.stage_embedding = nn.Embedding(num_stages * num_stages, stage_dim)

        total_input = input_dim + context_dim + time_dim + stage_dim

        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Predict drift.

        Args:
            x_t: [B, D] state
            t: [B] time in [0, 1]
            context: [B, C] niche context
            stage_pair_id: [B] stage pair indices

        Returns:
            [B, D] drift velocity
        """
        time_emb = self.time_embedding(t)
        stage_emb = self.stage_embedding(stage_pair_id.long())

        inp = torch.cat([x_t, context, time_emb, stage_emb], dim=-1)
        return self.net(inp)


class ScoreNetwork(nn.Module):
    """Score network for estimating ∇log p_t(x).

    Used for backward drift computation and score matching training.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dim: int,
        time_dim: int,
        stage_dim: int,
        num_stages: int,
    ):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.stage_embedding = nn.Embedding(num_stages * num_stages, stage_dim)

        total_input = input_dim + context_dim + time_dim + stage_dim

        self.net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Predict score ∇log p_t(x).

        Args:
            x_t: [B, D] state
            t: [B] time
            context: [B, C] niche context
            stage_pair_id: [B] stage pair indices

        Returns:
            [B, D] score estimate
        """
        time_emb = self.time_embedding(t)
        stage_emb = self.stage_embedding(stage_pair_id.long())

        inp = torch.cat([x_t, context, time_emb, stage_emb], dim=-1)
        return self.net(inp)


class SchrodingerBridge(nn.Module):
    """Schrödinger Bridge dynamics module.

    Learns stochastic transitions between stage distributions via entropy-
    regularized optimal transport on path space. Supports both forward
    (source→target) and backward (target→source) sampling.

    Key advantages over OT-CFM:
    - Models distribution of paths, not just mean
    - Natural handling of branching (partial EMT fates)
    - Built-in reversibility (forward ↔ backward)
    - Probability estimates for transition outcomes
    """

    def __init__(self, config: SchrodingerBridgeConfig):
        super().__init__()
        self.config = config

        # Forward drift f(x, t)
        self.forward_drift = DriftNetwork(
            input_dim=config.input_dim,
            context_dim=config.context_dim,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            stage_dim=config.stage_dim,
            num_stages=config.num_stages,
        )

        # Score network for ∇log p_t
        self.score_net = ScoreNetwork(
            input_dim=config.input_dim,
            context_dim=config.context_dim,
            hidden_dim=config.hidden_dim,
            time_dim=config.time_dim,
            stage_dim=config.stage_dim,
            num_stages=config.num_stages,
        )

        # Stage embedding for external use
        self.stage_embedding = nn.Embedding(
            config.num_stages * config.num_stages,
            config.stage_dim,
        )
        self.time_embedding = SinusoidalTimeEmbedding(config.time_dim)

    def forward_velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Forward drift velocity f(x, t).

        For SDE: dx = f(x,t)dt + σdW
        """
        return self.forward_drift(x_t, t, context, stage_pair_id)

    def backward_velocity(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Backward drift velocity.

        backward_drift = forward_drift - σ² * score

        For reverse SDE: dx = [f - σ²∇log p_t]dt + σd\bar{W}
        """
        f = self.forward_drift(x_t, t, context, stage_pair_id)
        score = self.score_net(x_t, t, context, stage_pair_id)
        sigma_sq = self.config.sigma ** 2
        return f - sigma_sq * score

    def score(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Score estimate ∇log p_t(x)."""
        return self.score_net(x_t, t, context, stage_pair_id)

    def sample_forward(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample forward SDE from t=0 to t=1.

        Args:
            x0: [B, D] initial state (source distribution)
            context: [B, C] niche context
            stage_pair_id: [B] stage pair indices
            num_steps: Number of integration steps
            return_trajectory: Return full trajectory

        Returns:
            Final state [B, D] or (final, trajectory [B, T+1, D])
        """
        x = x0
        dt = 1.0 / num_steps
        sqrt_dt = dt ** 0.5
        sigma = self.config.sigma

        trajectory = [x] if return_trajectory else None

        for k in range(num_steps):
            t = torch.full((x.shape[0],), k * dt, device=x.device, dtype=x.dtype)

            # Drift
            v = self.forward_velocity(x, t, context, stage_pair_id)

            # Euler-Maruyama step
            x = x + v * dt + sigma * sqrt_dt * torch.randn_like(x)

            # Optional Langevin corrector
            if self.config.langevin_steps > 0 and k < num_steps - 1:
                x = self._langevin_corrector(x, t + dt, context, stage_pair_id)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        return x

    def sample_backward(
        self,
        x1: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 50,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample backward SDE from t=1 to t=0.

        This computes the reverse transition (target→source), useful for:
        - Checking reversibility (MET from EMT)
        - Ancestral sampling
        - Probability flow analysis

        Args:
            x1: [B, D] terminal state (target distribution)
            context: [B, C] niche context
            stage_pair_id: [B] stage pair indices
            num_steps: Number of integration steps
            return_trajectory: Return full trajectory

        Returns:
            Initial state [B, D] or (initial, trajectory [B, T+1, D])
        """
        x = x1
        dt = 1.0 / num_steps
        sqrt_dt = dt ** 0.5
        sigma = self.config.sigma

        trajectory = [x] if return_trajectory else None

        for k in range(num_steps):
            t = torch.full((x.shape[0],), 1.0 - k * dt, device=x.device, dtype=x.dtype)

            # Backward drift
            v = self.backward_velocity(x, t, context, stage_pair_id)

            # Euler-Maruyama step (time goes backward)
            x = x - v * dt + sigma * sqrt_dt * torch.randn_like(x)

            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        return x

    def _langevin_corrector(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
    ) -> Tensor:
        """Langevin MCMC corrector step.

        Improves sample quality by running a few Langevin steps at each time.
        """
        snr = self.config.langevin_snr

        for _ in range(self.config.langevin_steps):
            score = self.score(x, t, context, stage_pair_id)
            noise = torch.randn_like(x)

            # Step size from score norm
            grad_norm = score.norm(dim=-1, keepdim=True).mean()
            noise_norm = noise.norm(dim=-1, keepdim=True).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2

            x = x + step_size * score + (2 * step_size) ** 0.5 * noise

        return x

    def sample_multiple(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_samples: int = 10,
        num_steps: int = 50,
    ) -> Tensor:
        """Sample multiple stochastic trajectories from same initial state.

        Returns [B, num_samples, D] - useful for estimating transition
        probability distributions.
        """
        samples = []
        for _ in range(num_samples):
            x1 = self.sample_forward(x0, context, stage_pair_id, num_steps)
            samples.append(x1)
        return torch.stack(samples, dim=1)


def schrodinger_bridge_loss(
    x_src: Tensor,
    x_tgt: Tensor,
    sb_module: SchrodingerBridge,
    context: Tensor,
    stage_pair_id: Tensor,
    num_time_samples: int = 4,
) -> tuple[Tensor, dict[str, float]]:
    """Compute Schrödinger Bridge training loss.

    Uses denoising score matching: train score network to predict noise
    added to linear interpolant between source and target.

    Args:
        x_src: [B, D] source states
        x_tgt: [B, D] target states
        sb_module: Schrödinger Bridge module
        context: [B, C] niche context
        stage_pair_id: [B] stage pair indices
        num_time_samples: Number of time points to sample per pair

    Returns:
        (loss, diagnostics)
    """
    device = x_src.device
    B, D = x_src.shape
    sigma = sb_module.config.sigma

    # Sample time uniformly
    t = torch.rand(B * num_time_samples, device=device)

    # Repeat source/target for multiple time samples
    x0 = x_src.repeat_interleave(num_time_samples, dim=0)
    x1 = x_tgt.repeat_interleave(num_time_samples, dim=0)
    ctx = context.repeat_interleave(num_time_samples, dim=0)
    stage = stage_pair_id.repeat_interleave(num_time_samples, dim=0)

    # Linear interpolant with noise (Brownian bridge)
    t_expand = t.unsqueeze(1)
    mean = (1 - t_expand) * x0 + t_expand * x1

    # Variance of Brownian bridge: σ² * t * (1-t)
    std = sigma * (t * (1 - t)).clamp_min(1e-6).sqrt().unsqueeze(1)
    noise = torch.randn_like(mean)
    x_t = mean + std * noise

    # Score matching target: -noise / std
    target_score = -noise / std.clamp_min(1e-6)

    # Predicted score
    pred_score = sb_module.score(x_t, t, ctx, stage)

    # Score matching loss (weighted by std to handle boundary)
    weight = std.clamp_min(1e-3)
    loss_score = ((pred_score - target_score) ** 2 * weight).mean()

    # Forward drift loss: should match velocity of OT path
    # Target velocity for linear interpolant: x1 - x0
    target_vel = x1 - x0
    pred_vel = sb_module.forward_velocity(x_t, t, ctx, stage)
    loss_drift = F.mse_loss(pred_vel, target_vel)

    # Total loss
    loss = loss_score + 0.5 * loss_drift

    diagnostics = {
        "loss_total": loss.item(),
        "loss_score": loss_score.item(),
        "loss_drift": loss_drift.item(),
        "mean_score_norm": pred_score.norm(dim=-1).mean().item(),
        "mean_drift_norm": pred_vel.norm(dim=-1).mean().item(),
    }

    return loss, diagnostics


def sb_ot_coupled_loss(
    x_src: Tensor,
    x_tgt: Tensor,
    sb_module: SchrodingerBridge,
    context: Tensor,
    stage_pair_id: Tensor,
    coupling: Tensor | None = None,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    num_pairs: int = 256,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """SB loss with OT-coupled pairs (like OT-CFM but stochastic).

    First computes OT coupling between source and target, then trains
    SB on the coupled pairs. This gives the optimal SB matching the
    marginal distributions.

    Args:
        x_src: [N, D] source distribution
        x_tgt: [M, D] target distribution
        sb_module: Schrödinger Bridge module
        context: [B, C] niche context (will be indexed)
        stage_pair_id: [B] stage pair indices
        coupling: Precomputed coupling [N, M] or None
        epsilon: Sinkhorn regularization
        sinkhorn_iters: Sinkhorn iterations
        num_pairs: Number of pairs to sample

    Returns:
        (loss, diagnostics, coupling)
    """
    from stagebridge.transition.losses import build_sinkhorn_coupling, sample_coupling_pairs

    device = x_src.device

    # Compute or use provided coupling
    if coupling is None:
        coupling = build_sinkhorn_coupling(
            x_src, x_tgt,
            epsilon=epsilon,
            n_iters=sinkhorn_iters,
        )

    # Sample pairs from coupling
    src_idx, tgt_idx = sample_coupling_pairs(coupling, num_pairs)

    x0 = x_src[src_idx]
    x1 = x_tgt[tgt_idx]

    # Expand context if needed
    if context.shape[0] == 1:
        ctx = context.expand(num_pairs, -1)
    elif context.shape[0] == x_src.shape[0]:
        ctx = context[src_idx]
    else:
        ctx = context[:num_pairs]

    if stage_pair_id.shape[0] == 1:
        stage = stage_pair_id.expand(num_pairs)
    elif stage_pair_id.shape[0] == x_src.shape[0]:
        stage = stage_pair_id[src_idx]
    else:
        stage = stage_pair_id[:num_pairs]

    # Compute SB loss on coupled pairs
    loss, diag = schrodinger_bridge_loss(
        x0, x1, sb_module, ctx, stage,
        num_time_samples=4,
    )

    # Add OT cost diagnostic
    with torch.no_grad():
        cost = ((x_src.unsqueeze(1) - x_tgt.unsqueeze(0)) ** 2).sum(-1)
        ot_cost = (coupling * cost).sum().item()
    diag["ot_cost"] = ot_cost

    return loss, diag, coupling


class SchrodingerBridgeWrapper(nn.Module):
    """Wrapper to make SB compatible with StageBridge interface.

    Provides forward_vector_field, integrate_euler etc. methods that
    match the OT-CFM interface, allowing drop-in replacement.
    """

    def __init__(self, sb: SchrodingerBridge):
        super().__init__()
        self.sb = sb

    def forward_vector_field(
        self,
        x_t: Tensor,
        t: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        **kwargs,
    ) -> Tensor:
        """Returns forward drift (for compatibility with OT-CFM interface)."""
        return self.sb.forward_velocity(x_t, t, context, stage_pair_id)

    def integrate_euler(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 50,
        **kwargs,
    ) -> Tensor:
        """Deterministic integration (ignores stochasticity)."""
        x = x0
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = torch.full((x.shape[0],), k * dt, device=x.device, dtype=x.dtype)
            v = self.sb.forward_velocity(x, t, context, stage_pair_id)
            x = x + v * dt
        return x

    def integrate_sde(
        self,
        x0: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 50,
    ) -> Tensor:
        """Stochastic integration (uses full SB)."""
        return self.sb.sample_forward(x0, context, stage_pair_id, num_steps)

    def sample_backward(
        self,
        x1: Tensor,
        context: Tensor,
        stage_pair_id: Tensor,
        num_steps: int = 50,
    ) -> Tensor:
        """Reverse SDE integration."""
        return self.sb.sample_backward(x1, context, stage_pair_id, num_steps)


def get_dynamics_module(
    dynamics_type: Literal["ot_cfm", "schrodinger_bridge"],
    config: dict,
) -> nn.Module:
    """Factory function for dynamics modules.

    Args:
        dynamics_type: "ot_cfm" or "schrodinger_bridge"
        config: Configuration dict

    Returns:
        Dynamics module
    """
    if dynamics_type == "schrodinger_bridge":
        sb_config = SchrodingerBridgeConfig(
            input_dim=config.get("input_dim", 40),
            context_dim=config.get("context_dim", 256),
            hidden_dim=config.get("hidden_dim", 256),
            time_dim=config.get("time_dim", 64),
            stage_dim=config.get("stage_dim", 32),
            num_stages=config.get("num_stages", 4),
            sigma=config.get("sigma", 0.1),
        )
        sb = SchrodingerBridge(sb_config)
        return SchrodingerBridgeWrapper(sb)
    else:
        raise ValueError(f"Use StageBridge directly for ot_cfm, not this factory")
