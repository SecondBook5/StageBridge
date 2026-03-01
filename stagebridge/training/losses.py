"""OT coupling and flow-matching losses for StageBridge training."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from stagebridge.utils.types import StageBatch


def pairwise_squared_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Compute pairwise squared Euclidean distance matrix."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    return (x2 + y2 - 2.0 * (x @ y.T)).clamp_min(0.0)


def build_sinkhorn_coupling(
    x_src: Tensor,
    x_tgt: Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> Tensor:
    """Build an entropically regularized OT coupling with log-domain updates."""
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

    for _ in range(max(n_iters, 400)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)

    log_pi = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    pi = torch.exp(log_pi)
    # Final balancing pass improves marginal accuracy after finite-iteration updates.
    for _ in range(5):
        pi = pi * (a / pi.sum(dim=1).clamp_min(1e-12)).unsqueeze(1)
        pi = pi * (b / pi.sum(dim=0).clamp_min(1e-12)).unsqueeze(0)
    return pi.to(dtype)


def sinkhorn_distance(
    x_src: Tensor,
    x_tgt: Tensor,
    epsilon: float = 0.05,
    n_iters: int = 80,
) -> Tensor:
    """Compute Sinkhorn-approximated OT distance from learned coupling."""
    pi = build_sinkhorn_coupling(x_src=x_src, x_tgt=x_tgt, epsilon=epsilon, n_iters=n_iters)
    cost = pairwise_squared_euclidean(x_src, x_tgt)
    return (pi * cost).sum()


def sample_coupling_pairs(
    coupling: Tensor,
    num_pairs: int,
) -> tuple[Tensor, Tensor]:
    """Sample source/target indices from a coupling matrix."""
    n, m = coupling.shape
    probs = coupling.reshape(-1)
    probs = probs / probs.sum().clamp_min(1e-12)
    sampled = torch.multinomial(probs, num_samples=num_pairs, replacement=True)
    src_idx = sampled // m
    tgt_idx = sampled % m
    return src_idx, tgt_idx


def random_pair_indices(n_src: int, n_tgt: int, num_pairs: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Sample random indices for no-OT ablations."""
    src_idx = torch.randint(0, n_src, (num_pairs,), device=device)
    tgt_idx = torch.randint(0, n_tgt, (num_pairs,), device=device)
    return src_idx, tgt_idx


def _stage_pair_tensor(model: Any, batch: StageBatch, n: int, device: torch.device) -> Tensor:
    if hasattr(model, "encode_stage_pair_tensor"):
        return model.encode_stage_pair_tensor(
            stage_src=batch.stage_src,
            stage_tgt=batch.stage_tgt,
            n=n,
            device=device,
        )
    return torch.zeros((n,), dtype=torch.long, device=device)


def flow_matching_loss(
    batch: StageBatch,
    model: Any,
    coupling: Tensor | None = None,
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
    num_ot_pairs: int = 512,
    context_consistency_weight: float = 0.1,
    use_ot: bool = True,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Compute OT-informed flow matching loss and diagnostics."""
    x_src = batch.x_src
    x_tgt = batch.x_tgt
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
    u_t = y_j - x_i

    c_s = model.forward_set_context(x_src)
    if c_s.ndim == 2 and c_s.shape[0] == 1:
        c_rep = c_s.expand(num_ot_pairs, -1)
    else:
        c_rep = c_s

    stage_pair_id = _stage_pair_tensor(model=model, batch=batch, n=num_ot_pairs, device=device)
    pred = model.forward_vector_field(
        x_t=x_t,
        t=t,
        c_s=c_rep,
        stage_pair_id=stage_pair_id,
    )

    loss_fm = F.mse_loss(pred, u_t)

    loss_ctx = torch.tensor(0.0, device=device, dtype=x_src.dtype)
    if context_consistency_weight > 0.0 and x_src.shape[0] >= 4:
        subset_n = max(2, x_src.shape[0] // 2)
        subset_idx = torch.randperm(x_src.shape[0], device=device)[:subset_n]
        c_sub = model.forward_set_context(x_src[subset_idx])
        c_full = model.forward_set_context(x_src)
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
