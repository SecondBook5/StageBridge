"""Gaussian bridge initialization for edge-wise stochastic transitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(slots=True, frozen=True)
class GaussianBridgeInit:
    """Configuration for a diagonal Gaussian bridge prior."""

    sigma: float = 0.1
    min_variance: float = 1e-4


@dataclass(slots=True, frozen=True)
class DiagonalGaussianMoments:
    """Diagonal Gaussian moments in latent space."""

    mean: Tensor
    variance: Tensor


@dataclass(slots=True, frozen=True)
class GaussianBridgeMoments:
    """Endpoint moments and configuration for a diagonal Gaussian bridge."""

    source: DiagonalGaussianMoments
    target: DiagonalGaussianMoments
    config: GaussianBridgeInit


def _as_2d_tensor(x: Any) -> Tensor:
    tensor = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(tensor.shape)}.")
    return tensor


def estimate_diagonal_gaussian(
    x: Any,
    *,
    min_variance: float = 1e-4,
) -> DiagonalGaussianMoments:
    """Estimate diagonal Gaussian moments from latent samples."""
    x_tensor = _as_2d_tensor(x)
    if x_tensor.shape[0] < 2:
        raise ValueError("x must contain at least two samples.")

    mean = x_tensor.mean(dim=0)
    variance = x_tensor.var(dim=0, unbiased=False).clamp_min(float(min_variance))
    return DiagonalGaussianMoments(mean=mean, variance=variance)


def build_gaussian_bridge(
    x_src: Any,
    x_tgt: Any,
    *,
    sigma: float = 0.1,
    min_variance: float = 1e-4,
) -> GaussianBridgeMoments:
    """Estimate endpoint moments and package a diagonal Gaussian bridge prior."""
    config = GaussianBridgeInit(sigma=float(sigma), min_variance=float(min_variance))
    return GaussianBridgeMoments(
        source=estimate_diagonal_gaussian(x_src, min_variance=config.min_variance),
        target=estimate_diagonal_gaussian(x_tgt, min_variance=config.min_variance),
        config=config,
    )


def interpolate_bridge_moments(
    bridge: GaussianBridgeMoments,
    t: float | Tensor,
) -> DiagonalGaussianMoments:
    """Interpolate diagonal Gaussian bridge moments at time ``t`` in ``[0, 1]``."""
    t_tensor = torch.as_tensor(
        t,
        dtype=bridge.source.mean.dtype,
        device=bridge.source.mean.device,
    ).clamp(0.0, 1.0)

    if t_tensor.ndim == 0:
        t_work = t_tensor
        source_mean = bridge.source.mean
        target_mean = bridge.target.mean
        source_var = bridge.source.variance
        target_var = bridge.target.variance
    else:
        t_work = t_tensor.reshape(-1, 1)
        source_mean = bridge.source.mean.unsqueeze(0)
        target_mean = bridge.target.mean.unsqueeze(0)
        source_var = bridge.source.variance.unsqueeze(0)
        target_var = bridge.target.variance.unsqueeze(0)

    mean = (1.0 - t_work) * source_mean + t_work * target_mean
    variance = (
        ((1.0 - t_work) ** 2) * source_var
        + (t_work**2) * target_var
        + (bridge.config.sigma**2) * t_work * (1.0 - t_work)
    ).clamp_min(bridge.config.min_variance)
    return DiagonalGaussianMoments(mean=mean, variance=variance)


def sample_bridge_state(
    bridge: GaussianBridgeMoments,
    *,
    t: float | Tensor,
    n_samples: int,
) -> Tensor:
    """Sample latent states from the interpolated diagonal Gaussian bridge."""
    if int(n_samples) <= 0:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

    moments = interpolate_bridge_moments(bridge, t)
    if moments.mean.ndim == 1:
        noise = torch.randn(
            int(n_samples),
            moments.mean.shape[0],
            device=moments.mean.device,
            dtype=moments.mean.dtype,
        )
        return moments.mean.unsqueeze(0) + noise * moments.variance.sqrt().unsqueeze(0)

    if int(n_samples) != moments.mean.shape[0]:
        raise ValueError(
            "When t is batched, n_samples must match the number of interpolated bridge states. "
            f"Got n_samples={n_samples}, expected {moments.mean.shape[0]}."
        )
    noise = torch.randn_like(moments.mean)
    return moments.mean + noise * moments.variance.sqrt()
