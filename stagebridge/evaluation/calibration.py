"""Calibration diagnostics for edge-wise transition predictions."""
from __future__ import annotations

import torch
from torch import Tensor


def summarize_transition_calibration(x_src: Tensor, x_pred: Tensor, x_tgt: Tensor) -> dict[str, float]:
    """Compare predicted vs observed edge-level displacement magnitudes."""
    n = min(x_src.shape[0], x_pred.shape[0], x_tgt.shape[0])
    pred_shift = x_pred[:n] - x_src[:n]
    true_shift = x_tgt[:n] - x_src[:n]
    pred_norm = pred_shift.norm(dim=1)
    true_norm = true_shift.norm(dim=1)
    error = (pred_norm - true_norm).abs()
    return {
        "mean_abs_shift_error": float(error.mean().item()),
        "mean_pred_shift_norm": float(pred_norm.mean().item()),
        "mean_true_shift_norm": float(true_norm.mean().item()),
    }
