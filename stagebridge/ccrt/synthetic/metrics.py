"""Mechanism-recovery metrics (PyTorch only).

Compares predicted context effects against teacher context effects. Includes
RMSE / relative RMSE, row-cosine, Pearson, sign agreement, effect norm, and a
tie-aware rank-order correlation implemented without SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "root_mean_squared_error",
    "relative_root_mean_squared_error",
    "mean_cosine_recovery",
    "pearson_recovery",
    "sign_agreement",
    "mean_effect_norm",
    "rank_order_recovery",
    "CounterfactualRecoveryMetrics",
]


def _check_same_shape(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("inputs must be finite")


def root_mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_same_shape(prediction, target)
    return torch.sqrt(((prediction - target) ** 2).mean())


def relative_root_mean_squared_error(
    prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    _check_same_shape(prediction, target)
    rmse = torch.sqrt(((prediction - target) ** 2).mean())
    rms_target = torch.sqrt((target ** 2).mean())
    return rmse / (rms_target + eps)


def mean_cosine_recovery(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
    exclude_zero_target: bool = True,
) -> torch.Tensor:
    _check_same_shape(prediction, target)
    if prediction.dim() == 1:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    cos = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=eps)
    if exclude_zero_target:
        active = target.norm(dim=-1) > eps
        active_f = active.to(cos.dtype)
        denom = active_f.sum()
        if float(denom) <= 0:
            return torch.zeros((), dtype=cos.dtype, device=cos.device)
        return (cos * active_f).sum() / denom
    return cos.mean()


def pearson_recovery(
    prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    _check_same_shape(prediction, target)
    p = prediction.reshape(-1)
    t = target.reshape(-1)
    p = p - p.mean()
    t = t - t.mean()
    p_std = torch.sqrt((p ** 2).sum())
    t_std = torch.sqrt((t ** 2).sum())
    if float(p_std) <= eps or float(t_std) <= eps:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return (p * t).sum() / (p_std * t_std)


def sign_agreement(
    prediction: torch.Tensor, target: torch.Tensor, *, zero_tolerance: float = 1e-8
) -> torch.Tensor:
    _check_same_shape(prediction, target)
    p = prediction.reshape(-1)
    t = target.reshape(-1)
    valid = t.abs() > zero_tolerance
    if not bool(valid.any()):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    agree = (torch.sign(p[valid]) == torch.sign(t[valid])).to(prediction.dtype)
    return agree.mean()


def mean_effect_norm(effect: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(effect).all()):
        raise ValueError("effect must be finite")
    if effect.dim() == 1:
        return effect.abs().mean()
    return effect.norm(dim=-1).mean()


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Average ranks (1..n) with deterministic tie handling."""
    n = values.shape[0]
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float64)
    # resolve ties by averaging equal values' ranks
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and bool(sorted_vals[j + 1] == sorted_vals[i]):
            j += 1
        if j > i:
            avg = (torch.arange(i + 1, j + 2, dtype=torch.float64)).mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def rank_order_recovery(
    predicted_effect_magnitudes: torch.Tensor,
    true_effect_magnitudes: torch.Tensor,
) -> torch.Tensor:
    """Spearman-style rank correlation (Pearson on average ranks)."""
    _check_same_shape(predicted_effect_magnitudes, true_effect_magnitudes)
    if predicted_effect_magnitudes.dim() != 1:
        raise ValueError("rank_order_recovery expects 1-D magnitude vectors")
    if predicted_effect_magnitudes.shape[0] < 2:
        return torch.zeros((), dtype=torch.float64)
    pr = _average_ranks(predicted_effect_magnitudes.to(torch.float64))
    tr = _average_ranks(true_effect_magnitudes.to(torch.float64))
    return pearson_recovery(pr, tr)


@dataclass(frozen=True)
class CounterfactualRecoveryMetrics:
    """Recovery metrics for a factual-minus-counterfactual context effect."""

    drift_rmse: torch.Tensor
    drift_relative_rmse: torch.Tensor
    drift_cosine: torch.Tensor
    drift_pearson: torch.Tensor
    growth_rmse: torch.Tensor
    growth_pearson: torch.Tensor
    predicted_drift_effect_norm: torch.Tensor
    true_drift_effect_norm: torch.Tensor
    predicted_growth_effect_norm: torch.Tensor
    true_growth_effect_norm: torch.Tensor
