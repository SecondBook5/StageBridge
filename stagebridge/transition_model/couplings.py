"""Coupling helpers for the transition layer."""

from __future__ import annotations

import torch
from torch import Tensor

from stagebridge.transition_model.losses import pairwise_squared_euclidean


def build_cost_matrix(
    x_src: Tensor,
    x_tgt: Tensor,
    *,
    extra_cost: Tensor | None = None,
) -> Tensor:
    """Build the transport cost matrix, optionally adding regularization costs."""
    cost = pairwise_squared_euclidean(x_src, x_tgt)
    if extra_cost is not None:
        if extra_cost.shape != cost.shape:
            raise ValueError(
                f"extra_cost shape {tuple(extra_cost.shape)} does not match cost shape {tuple(cost.shape)}."
            )
        cost = cost + extra_cost.to(cost.dtype)
    return cost


def build_sinkhorn_coupling_from_cost(
    cost: Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> Tensor:
    """Build an entropically regularized OT coupling from a precomputed cost matrix."""
    n, m = cost.shape
    work_dtype = torch.float64
    device = cost.device
    cost_work = cost.to(work_dtype)
    a = torch.full((n,), 1.0 / n, device=device, dtype=work_dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=work_dtype)
    log_a = torch.log(a + 1e-12)
    log_b = torch.log(b + 1e-12)
    log_k = -cost_work / max(float(epsilon), 1e-8)

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)

    log_pi = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    coupling = torch.exp(log_pi)
    for _ in range(20):
        coupling = coupling * (a / coupling.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        coupling = coupling * (b / coupling.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
    return coupling.to(cost.dtype)


def build_ot_coupling(
    x_src: Tensor,
    x_tgt: Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 80,
    extra_cost: Tensor | None = None,
) -> Tensor:
    """Build a transport coupling with optional regularization-aware costs."""
    cost = build_cost_matrix(x_src, x_tgt, extra_cost=extra_cost)
    return build_sinkhorn_coupling_from_cost(cost, epsilon=epsilon, n_iters=n_iters)
