"""Low-claim progression-structure summaries."""

from __future__ import annotations

import torch
from torch import Tensor


def summarize_pseudotime_structure(
    x_src: Tensor, x_pred: Tensor, x_tgt: Tensor
) -> dict[str, float]:
    n = min(x_src.shape[0], x_pred.shape[0], x_tgt.shape[0])
    pred_progress = (x_pred[:n] - x_src[:n]).norm(dim=1)
    true_progress = (x_tgt[:n] - x_src[:n]).norm(dim=1)
    if pred_progress.numel() < 2:
        corr = float("nan")
    else:
        corr = float(torch.corrcoef(torch.stack([pred_progress, true_progress]))[0, 1].item())
    return {
        "progress_norm_correlation": corr,
        "mean_pred_progress_norm": float(pred_progress.mean().item()),
        "mean_true_progress_norm": float(true_progress.mean().item()),
    }
