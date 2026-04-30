"""OT coupling and flow-matching losses for StageBridge training.

This module provides:
- Sinkhorn-based optimal transport coupling
- OT-CFM (Conditional Flow Matching) loss with optional Brownian bridge noise
- Multi-hop trajectory consistency loss for skip-stage transitions
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


class TransitionModelProtocol(Protocol):
    """Protocol for models compatible with flow matching losses."""

    def forward_set_context(
        self,
        x_set: Tensor,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor: ...

    def forward_vector_field(
        self,
        x_t: Tensor,
        t: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        wes_features: Tensor | None = None,
        lr_features: Tensor | None = None,
    ) -> Tensor: ...

    def encode_stage_pair_tensor(
        self,
        stage_src: int,
        stage_tgt: int,
        n: int,
        device: torch.device,
    ) -> Tensor: ...

    def integrate_euler(
        self,
        x0: Tensor,
        c_s: Tensor,
        stage_pair_id: Tensor,
        num_steps: int,
        wes_features: Tensor | None = None,
    ) -> Tensor: ...


def pairwise_squared_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distance matrix.

    Args:
        x: [N, D] source points
        y: [M, D] target points

    Returns:
        [N, M] squared distance matrix
    """
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return (x2 + y2 - 2.0 * (x @ y.T)).clamp_min(0.0)


def build_sinkhorn_coupling(
    x_src: Tensor,
    x_tgt: Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> Tensor:
    """Build entropically regularized OT coupling via Sinkhorn iterations.

    Uses log-domain updates for numerical stability.

    Args:
        x_src: [N, D] source points
        x_tgt: [M, D] target points
        epsilon: Entropic regularization strength
        n_iters: Number of Sinkhorn iterations

    Returns:
        [N, M] coupling matrix with marginals summing to uniform
    """
    n = x_src.shape[0]
    m = x_tgt.shape[0]
    device = x_src.device
    dtype = x_src.dtype
    work_dtype = torch.float64

    x_src_work = x_src.to(work_dtype)
    x_tgt_work = x_tgt.to(work_dtype)

    a = torch.full((n,), 1.0 / n, device=device, dtype=work_dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=work_dtype)
    log_a = torch.log(a + 1e-12)
    log_b = torch.log(b + 1e-12)

    cost = pairwise_squared_euclidean(x_src_work, x_tgt_work)
    log_k = -cost / max(epsilon, 1e-8)

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)

    log_pi = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    pi = torch.exp(log_pi)

    for _ in range(20):
        pi = pi * (a / pi.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        pi = pi * (b / pi.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)

    return pi.to(dtype)


def sinkhorn_distance(
    x_src: Tensor,
    x_tgt: Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> Tensor:
    """Compute Sinkhorn-approximated OT distance.

    Args:
        x_src: [N, D] source points
        x_tgt: [M, D] target points
        epsilon: Entropic regularization
        n_iters: Sinkhorn iterations

    Returns:
        Scalar OT distance
    """
    pi = build_sinkhorn_coupling(x_src=x_src, x_tgt=x_tgt, epsilon=epsilon, n_iters=n_iters)
    cost = pairwise_squared_euclidean(x_src, x_tgt)
    return (pi * cost).sum()


def sample_coupling_pairs(
    coupling: Tensor,
    num_pairs: int,
) -> tuple[Tensor, Tensor]:
    """Sample source/target index pairs from coupling matrix.

    Args:
        coupling: [N, M] coupling matrix
        num_pairs: Number of pairs to sample

    Returns:
        (src_idx, tgt_idx) each with shape [num_pairs]
    """
    n, m = coupling.shape
    probs = coupling.reshape(-1)

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(0.0)

    prob_sum = probs.sum()
    if prob_sum < 1e-12:
        src_idx = torch.randint(0, n, (num_pairs,), device=coupling.device)
        tgt_idx = torch.randint(0, m, (num_pairs,), device=coupling.device)
        return src_idx, tgt_idx

    probs = probs / prob_sum
    sampled = torch.multinomial(probs, num_samples=num_pairs, replacement=True)
    src_idx = sampled // m
    tgt_idx = sampled % m

    src_idx = src_idx.clamp(0, n - 1)
    tgt_idx = tgt_idx.clamp(0, m - 1)

    return src_idx, tgt_idx


def random_pair_indices(
    n_src: int,
    n_tgt: int,
    num_pairs: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample random index pairs (for no-OT ablations).

    Args:
        n_src: Number of source points
        n_tgt: Number of target points
        num_pairs: Number of pairs
        device: Torch device

    Returns:
        (src_idx, tgt_idx) each with shape [num_pairs]
    """
    src_idx = torch.randint(0, n_src, (num_pairs,), device=device)
    tgt_idx = torch.randint(0, n_tgt, (num_pairs,), device=device)
    return src_idx, tgt_idx


def flow_matching_loss(
    x_src: Tensor,
    x_tgt: Tensor,
    model: TransitionModelProtocol,
    stage_src: int,
    stage_tgt: int,
    x_set: Tensor | None = None,
    context_mask: Tensor | None = None,
    niche_coords: Tensor | None = None,
    spatial_niche: Tensor | None = None,
    wes_features: Tensor | None = None,
    lr_features: Tensor | None = None,
    coupling: Tensor | None = None,
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
    num_ot_pairs: int = 512,
    context_consistency_weight: float = 0.1,
    use_ot: bool = True,
    sigma: float = 0.0,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Compute OT-CFM flow matching loss with optional Brownian bridge noise.

    When sigma > 0, uses Brownian bridge interpolant:
        x_t = (1-t)*x_i + t*y_j + sigma * sqrt(t*(1-t)) * z
    The velocity target remains u_t = y_j - x_i (conditional mean).
    Setting sigma = 0 recovers deterministic OT-CFM.

    Args:
        x_src: [N, D] source cell embeddings
        x_tgt: [M, D] target cell embeddings
        model: Transition model with context and vector field methods
        stage_src: Source stage index
        stage_tgt: Target stage index
        x_set: [K, D] context set (defaults to x_src)
        context_mask: [1, K] context validity mask
        niche_coords: Optional spatial coordinates for context
        spatial_niche: Optional per-cell spatial niche features
        wes_features: Optional WES conditioning features
        lr_features: Optional LR conditioning features
        coupling: Precomputed coupling (computed if None)
        ot_epsilon: Sinkhorn regularization
        sinkhorn_iters: Sinkhorn iterations
        num_ot_pairs: Number of OT pairs to sample
        context_consistency_weight: Weight for context consistency loss
        use_ot: Whether to use OT coupling (False = random pairs)
        sigma: Brownian bridge noise level

    Returns:
        (total_loss, diagnostics, coupling)
    """
    if x_set is None:
        x_set = x_src
    device = x_src.device

    if use_ot:
        if coupling is None:
            coupling = build_sinkhorn_coupling(
                x_src=x_src,
                x_tgt=x_tgt,
                epsilon=ot_epsilon,
                n_iters=sinkhorn_iters,
            )
        src_idx, tgt_idx = sample_coupling_pairs(coupling=coupling, num_pairs=num_ot_pairs)
    else:
        if coupling is None:
            coupling = torch.full(
                (x_src.shape[0], x_tgt.shape[0]),
                1.0 / (x_src.shape[0] * x_tgt.shape[0]),
                dtype=x_src.dtype,
                device=device,
            )
        src_idx, tgt_idx = random_pair_indices(
            n_src=x_src.shape[0],
            n_tgt=x_tgt.shape[0],
            num_pairs=num_ot_pairs,
            device=device,
        )

    x_i = x_src[src_idx]
    y_j = x_tgt[tgt_idx]

    t = torch.rand((num_ot_pairs,), device=device, dtype=x_src.dtype)
    x_t = (1.0 - t.unsqueeze(1)) * x_i + t.unsqueeze(1) * y_j
    if sigma > 0.0:
        noise_scale = sigma * (t * (1.0 - t)).clamp_min(0.0).sqrt().unsqueeze(1)
        x_t = x_t + noise_scale * torch.randn_like(x_i)
    u_t = y_j - x_i

    kwargs: dict[str, Any] = {}
    if niche_coords is not None:
        kwargs["niche_coords"] = niche_coords
    if spatial_niche is not None:
        kwargs["spatial_niche"] = spatial_niche

    c_s = model.forward_set_context(x_set, mask=context_mask, **kwargs)
    if c_s.ndim == 2 and c_s.shape[0] == 1:
        c_rep = c_s.expand(num_ot_pairs, -1)
    else:
        c_rep = c_s

    stage_pair_id = model.encode_stage_pair_tensor(
        stage_src=stage_src,
        stage_tgt=stage_tgt,
        n=num_ot_pairs,
        device=device,
    )

    pred = model.forward_vector_field(
        x_t=x_t,
        t=t,
        c_s=c_rep,
        stage_pair_id=stage_pair_id,
        wes_features=wes_features,
        lr_features=lr_features,
    )

    loss_fm = F.mse_loss(pred, u_t)

    loss_ctx = torch.tensor(0.0, device=device, dtype=x_src.dtype)
    if context_consistency_weight > 0.0 and x_set.shape[0] >= 4:
        subset_n = max(2, x_set.shape[0] // 2)
        subset_idx = torch.randperm(x_set.shape[0], device=device)[:subset_n]
        mask_sub = None
        if context_mask is not None:
            if context_mask.ndim == 2 and context_mask.shape[0] == 1:
                mask_sub = context_mask[:, subset_idx]
            elif context_mask.ndim == 1:
                mask_sub = context_mask[subset_idx].unsqueeze(0)
        c_sub = model.forward_set_context(x_set[subset_idx], mask=mask_sub)
        c_full = model.forward_set_context(x_set, mask=context_mask, **kwargs)
        loss_ctx = F.mse_loss(c_full, c_sub)

    total = loss_fm + context_consistency_weight * loss_ctx

    with torch.no_grad():
        if use_ot:
            cost = pairwise_squared_euclidean(x_src, x_tgt)
            ot_cost = float((coupling * cost).sum().item())
        else:
            ot_cost = float("nan")

    diagnostics = {
        "loss_total": float(total.detach().item()),
        "loss_fm": float(loss_fm.detach().item()),
        "loss_ctx": float(loss_ctx.detach().item()),
        "ot_cost": ot_cost,
        "num_pairs": float(num_ot_pairs),
    }
    return total, diagnostics, coupling


def _compose_trajectory(
    model: TransitionModelProtocol,
    x_src: Tensor,
    stage_sequence: list[int],
    num_steps: int = 8,
    wes_features: Tensor | None = None,
) -> Tensor:
    """Compose sequential transitions through intermediate stages.

    Args:
        model: Transition model
        x_src: [B, D] source cells
        stage_sequence: [s0, s1, s2, ...] ordered stage indices
        num_steps: Euler steps per hop
        wes_features: Optional WES conditioning

    Returns:
        [B, D] endpoint after composing all hops
    """
    x = x_src
    for i in range(len(stage_sequence) - 1):
        s_src = stage_sequence[i]
        s_tgt = stage_sequence[i + 1]
        c_s = model.forward_set_context(x)
        pair_id = model.encode_stage_pair_tensor(
            stage_src=s_src,
            stage_tgt=s_tgt,
            n=x.shape[0],
            device=x.device,
        )
        x = model.integrate_euler(
            x0=x,
            c_s=c_s,
            stage_pair_id=pair_id,
            num_steps=num_steps,
            wes_features=wes_features,
        )
    return x


def multihop_consistency_loss(
    model: TransitionModelProtocol,
    x_src: Tensor,
    stage_src: int,
    stage_tgt: int,
    num_stages: int = 5,
    num_steps: int = 8,
    wes_features: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute trajectory composition consistency loss for skip transitions.

    For transitions with gap >= 2 (e.g., Normal->AIS), computes:
    - Direct:  x_T^direct = integrate(x_src, src->tgt)
    - Chained: x_T^chain  = compose(x_src, src->mid1->...->tgt)
    - Loss:    MSE(x_T^direct, sg(x_T^chain))

    Stop-gradient on chained prevents collapse. Returns zero for gap=1.

    Args:
        model: Transition model
        x_src: [B, D] source cells
        stage_src: Source stage index
        stage_tgt: Target stage index
        num_stages: Total number of stages
        num_steps: Euler steps per hop
        wes_features: Optional WES conditioning

    Returns:
        (loss, diagnostics)
    """
    gap = stage_tgt - stage_src
    if gap <= 1:
        zero = torch.tensor(0.0, device=x_src.device, dtype=x_src.dtype)
        return zero, {"multihop_loss": 0.0, "gap": float(gap)}

    stage_sequence = list(range(stage_src, stage_tgt + 1))

    c_s_direct = model.forward_set_context(x_src)
    pair_direct = model.encode_stage_pair_tensor(
        stage_src=stage_src,
        stage_tgt=stage_tgt,
        n=x_src.shape[0],
        device=x_src.device,
    )
    x_direct = model.integrate_euler(
        x0=x_src,
        c_s=c_s_direct,
        stage_pair_id=pair_direct,
        num_steps=num_steps,
        wes_features=wes_features,
    )

    x_chained = _compose_trajectory(
        model=model,
        x_src=x_src,
        stage_sequence=stage_sequence,
        num_steps=num_steps,
        wes_features=wes_features,
    )

    loss = F.mse_loss(x_direct, x_chained.detach())

    diagnostics = {
        "multihop_loss": float(loss.detach().item()),
        "gap": float(gap),
        "direct_norm": float(x_direct.norm(dim=-1).mean().item()),
        "chained_norm": float(x_chained.norm(dim=-1).mean().item()),
    }
    return loss, diagnostics
