"""Held-out metrics for edge-wise transition evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from stagebridge.transition_model.infer import classifier_two_sample_auc, mmd_rbf
from stagebridge.transition_model.losses import sinkhorn_distance


def rollout_edge_transition(
    model: Any,
    x_src: Tensor,
    *,
    context: Tensor,
    context_tokens: Tensor | None = None,
    edge_id: int,
    num_steps: int = 8,
    stochastic: bool = False,
) -> Tensor:
    edge_ids = torch.full((x_src.shape[0],), int(edge_id), dtype=torch.long, device=x_src.device)
    x_pred, _ = model.rollout(
        x_src,
        context=context,
        context_tokens=context_tokens,
        edge_ids=edge_ids,
        num_steps=int(num_steps),
        stochastic=bool(stochastic),
    )
    return x_pred


def heldout_transition_metrics(
    model: Any,
    x_src: Tensor,
    x_tgt: Tensor,
    *,
    context: Tensor,
    context_tokens: Tensor | None = None,
    edge_id: int,
    num_steps: int = 8,
    stochastic: bool = False,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 80,
) -> dict[str, float]:
    """Compute honest held-out distribution-matching metrics for one edge."""
    x_pred = rollout_edge_transition(
        model,
        x_src,
        context=context,
        context_tokens=context_tokens,
        edge_id=edge_id,
        num_steps=num_steps,
        stochastic=stochastic,
    )
    n = min(x_pred.shape[0], x_tgt.shape[0], x_src.shape[0])
    x_pred = x_pred[:n]
    x_tgt = x_tgt[:n]
    x_src = x_src[:n]

    sink_model = float(
        sinkhorn_distance(
            x_src=x_pred,
            x_tgt=x_tgt,
            epsilon=float(epsilon),
            n_iters=int(sinkhorn_iters),
        ).item()
    )
    sink_identity = float(
        sinkhorn_distance(
            x_src=x_src,
            x_tgt=x_tgt,
            epsilon=float(epsilon),
            n_iters=int(sinkhorn_iters),
        ).item()
    )
    mmd = float(mmd_rbf(x_pred, x_tgt).item())
    auc = float(classifier_two_sample_auc(x_pred, x_tgt))
    src_mean = x_src.mean(dim=0)
    tgt_mean = x_tgt.mean(dim=0)
    pred_mean = x_pred.mean(dim=0)
    true_dir = tgt_mean - src_mean
    pred_dir = pred_mean - src_mean
    denom = float(true_dir.norm().item() * pred_dir.norm().item())
    direction_cosine = float(torch.dot(true_dir, pred_dir).item() / denom) if denom > 1e-8 else float("nan")

    return {
        "sinkhorn": sink_model,
        "sinkhorn_delta": sink_identity - sink_model,
        "mmd_rbf": mmd,
        "classifier_auc": auc,
        "direction_cosine": direction_cosine,
    }


def summarize_shift_magnitudes(x_src: Tensor, x_pred: Tensor, x_tgt: Tensor) -> dict[str, float]:
    """Summarize movement magnitudes without overclaiming trajectory structure."""
    pred_shift = x_pred.mean(dim=0) - x_src.mean(dim=0)
    true_shift = x_tgt.mean(dim=0) - x_src.mean(dim=0)
    return {
        "pred_shift_norm": float(pred_shift.norm().item()),
        "true_shift_norm": float(true_shift.norm().item()),
        "shift_norm_ratio": float(pred_shift.norm().item() / max(true_shift.norm().item(), 1e-8)),
    }
