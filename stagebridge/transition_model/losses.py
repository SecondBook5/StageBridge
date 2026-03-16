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

    for _ in range(max(int(n_iters), 1)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)

    log_pi = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    pi = torch.exp(log_pi)
    # Final balancing pass improves marginal accuracy after finite-iteration updates.
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


def random_pair_indices(
    n_src: int, n_tgt: int, num_pairs: int, device: torch.device
) -> tuple[Tensor, Tensor]:
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


def _forward_set_context(
    model: Any,
    x_set: Tensor,
    mask: Tensor | None = None,
    niche_coords: Tensor | None = None,
    spatial_niche: Tensor | None = None,
) -> Tensor:
    """Call ``forward_set_context`` passing optional conditioning kwargs.

    All model classes (StageBridgeModel, DeepSetsFlowModel, NoContextFlowModel)
    accept ``**kwargs`` in forward_set_context, so unknown kwargs are safely
    absorbed by baselines that don't use them.
    """
    kwargs: dict[str, Any] = {}
    if niche_coords is not None:
        kwargs["niche_coords"] = niche_coords
    if spatial_niche is not None:
        kwargs["spatial_niche"] = spatial_niche
    return model.forward_set_context(x_set, mask=mask, **kwargs)


def flow_matching_loss(
    batch: StageBatch,
    model: Any,
    coupling: Tensor | None = None,
    ot_epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
    num_ot_pairs: int = 512,
    context_consistency_weight: float = 0.1,
    use_ot: bool = True,
    sigma: float = 0.0,
) -> tuple[Tensor, dict[str, float], Tensor]:
    """Compute OT-informed flow matching loss and diagnostics.

    When ``sigma > 0`` the Brownian bridge interpolant is used:
    ``x_t = (1-t)*x_i + t*y_j + sigma * sqrt(t*(1-t)) * z``, z ~ N(0, I).
    The velocity target remains ``u_t = y_j - x_i`` (conditional mean).
    Setting ``sigma = 0`` recovers the deterministic OT-CFM path.
    """
    x_src = batch.x_src
    x_tgt = batch.x_tgt
    x_set = batch.x_set if batch.x_set is not None else x_src
    context_mask = batch.context_mask
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

    niche_coords = batch.niche_coords
    spatial_niche = batch.spatial_niche
    c_s = _forward_set_context(
        model=model,
        x_set=x_set,
        mask=context_mask,
        niche_coords=niche_coords,
        spatial_niche=spatial_niche,
    )
    if c_s.ndim == 2 and c_s.shape[0] == 1:
        c_rep = c_s.expand(num_ot_pairs, -1)
    else:
        c_rep = c_s

    stage_pair_id = _stage_pair_tensor(model=model, batch=batch, n=num_ot_pairs, device=device)
    wes_features = batch.wes_features
    lr_features = batch.lr_features

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
        c_sub = _forward_set_context(model=model, x_set=x_set[subset_idx], mask=mask_sub)
        c_full = _forward_set_context(
            model=model,
            x_set=x_set,
            mask=context_mask,
            niche_coords=niche_coords,
            spatial_niche=spatial_niche,
        )
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


# ---------------------------------------------------------------------------
# Multi-hop skip-stage trajectory consistency loss
# ---------------------------------------------------------------------------


def _compose_trajectory(
    model: Any,
    x_src: Tensor,
    stage_sequence: list[int],
    num_steps: int = 8,
    wes_features: Tensor | None = None,
) -> Tensor:
    """Compose sequential transitions through intermediate stages.

    Parameters
    ----------
    model : StageBridgeModel
        Model with ``forward_set_context`` and ``integrate_euler``.
    x_src : (B, D)
        Source cells.
    stage_sequence : list[int]
        Ordered stage indices [s0, s1, s2, ...] where the chain is
        s0→s1→s2→...→sN.
    num_steps : int
        Euler integration steps per hop.
    wes_features : Tensor or None
        Optional WES conditioning.

    Returns
    -------
    Tensor : (B, D) — endpoint after composing all hops.
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
    model: Any,
    x_src: Tensor,
    stage_src: int,
    stage_tgt: int,
    num_stages: int = 5,
    num_steps: int = 8,
    wes_features: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute trajectory composition consistency loss for skip transitions.

    For a transition with gap ≥ 2 (e.g., Normal→AIS, gap=2), computes:

    * **Direct**:  x_T^direct = integrate(x_src, src→tgt)
    * **Chained**: x_T^chain  = compose(x_src, src→mid₁→mid₂→...→tgt)
    * **Loss**:    MSE(x_T^direct, sg(x_T^chain))

    Stop-gradient on the chained result prevents collapse to trivial solutions.
    For adjacent transitions (gap=1), returns zero loss.

    Parameters
    ----------
    model : StageBridgeModel
    x_src : (B, D) — source cells
    stage_src, stage_tgt : int — stage indices
    num_stages : int — total number of stages
    num_steps : int — Euler steps per hop
    wes_features : Tensor or None

    Returns
    -------
    (loss, diagnostics) — scalar loss and info dict
    """
    gap = stage_tgt - stage_src
    if gap <= 1:
        zero = torch.tensor(0.0, device=x_src.device, dtype=x_src.dtype)
        return zero, {"multihop_loss": 0.0, "gap": float(gap)}

    # Build intermediate stage sequence
    stage_sequence = list(range(stage_src, stage_tgt + 1))

    # Direct transition: src → tgt in one shot
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

    # Chained transition: src → mid₁ → ... → tgt
    x_chained = _compose_trajectory(
        model=model,
        x_src=x_src,
        stage_sequence=stage_sequence,
        num_steps=num_steps,
        wes_features=wes_features,
    )

    # Stop gradient on chained to prevent collapse
    loss = F.mse_loss(x_direct, x_chained.detach())

    diagnostics = {
        "multihop_loss": float(loss.detach().item()),
        "gap": float(gap),
        "direct_norm": float(x_direct.norm(dim=-1).mean().item()),
        "chained_norm": float(x_chained.norm(dim=-1).mean().item()),
    }
    return loss, diagnostics
