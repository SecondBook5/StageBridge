"""Native differentiable Sinkhorn — the transparent CCRT reference backend.

Implements entropic optimal transport in the log domain with the canonical CCRT
conventions:

* cost is the FULL squared Euclidean distance (no 1/2 factor);
* epsilon is in cost units;
* the entropic objective is

      OT_eps(a, b) = <pi, C> + eps * KL(pi || a (x) b)

  with the full generalized KL

      KL(pi || ref) = sum[ pi * (log pi - log ref) - pi + ref ],  ref_ij = a_i b_j.

Couplings, transport cost, KL, and the regularized objective all keep gradients;
only convergence/early-stopping decisions use detached scalars. The Sinkhorn
divergence debiases the cross term with the two self terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..representations.semantic import SemanticGeometryConfig, prepare_semantic_features
from .backends import NATIVE_BACKEND
from .costs import build_transport_cost

__all__ = [
    "SinkhornConfig",
    "SinkhornOutput",
    "SinkhornDivergenceOutput",
    "normalize_measure_weights",
    "sinkhorn_coupling_native",
    "sinkhorn_divergence_native",
]


@dataclass(frozen=True)
class SinkhornConfig:
    """Configuration for the native Sinkhorn solver."""

    epsilon: float = 0.05
    max_iterations: int = 200
    tolerance: float = 1e-6
    check_interval: int = 10
    early_stopping: bool = False

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be > 0")
        if self.check_interval <= 0:
            raise ValueError("check_interval must be > 0")
        if self.check_interval > self.max_iterations:
            raise ValueError("check_interval must be <= max_iterations")


@dataclass(frozen=True)
class SinkhornOutput:
    """The native entropic-OT solution and its diagnostics."""

    backend: str
    coupling: torch.Tensor
    transport_cost: torch.Tensor
    kl_regularization: torch.Tensor
    regularized_objective: torch.Tensor
    source_weights: torch.Tensor
    target_weights: torch.Tensor
    source_marginal: torch.Tensor
    target_marginal: torch.Tensor
    source_marginal_error: torch.Tensor
    target_marginal_error: torch.Tensor
    marginal_error: torch.Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class SinkhornDivergenceOutput:
    """A debiased native Sinkhorn divergence and its three OT terms."""

    backend: str
    divergence: torch.Tensor
    cross: SinkhornOutput
    source_self: SinkhornOutput
    target_self: SinkhornOutput


def normalize_measure_weights(
    weights: torch.Tensor | None,
    *,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return normalized, strictly-positive measure weights (uniform if None)."""
    if weights is None:
        return torch.full((size,), 1.0 / size, dtype=dtype, device=device)
    if not isinstance(weights, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor or None")
    if not torch.is_floating_point(weights):
        raise ValueError(f"{name} must be a floating-point tensor")
    if weights.dim() != 1:
        raise ValueError(f"{name} must be rank 1, got shape {tuple(weights.shape)}")
    if weights.shape[0] != size:
        raise ValueError(f"{name} length {weights.shape[0]} != expected {size}")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool((weights <= 0).any()):
        raise ValueError(f"{name} must be strictly positive")
    return weights / weights.sum()


def _tiny(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).tiny


def sinkhorn_coupling_native(
    *,
    cost_matrix: torch.Tensor,
    config: SinkhornConfig,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> SinkhornOutput:
    """Solve entropic OT in the log domain for a fixed cost matrix."""
    if not isinstance(cost_matrix, torch.Tensor):
        raise ValueError("cost_matrix must be a torch.Tensor")
    if not torch.is_floating_point(cost_matrix):
        raise ValueError("cost_matrix must be floating point")
    if cost_matrix.dim() != 2:
        raise ValueError(f"cost_matrix must be rank 2 [N, M], got {tuple(cost_matrix.shape)}")
    n, m = cost_matrix.shape
    if n <= 0 or m <= 0:
        raise ValueError("cost_matrix must have N > 0 and M > 0")
    if not bool(torch.isfinite(cost_matrix).all()):
        raise ValueError("cost_matrix contains non-finite values")
    # non-negative within a small numerical tolerance
    if bool((cost_matrix < -1e-6).any()):
        raise ValueError("cost_matrix must be non-negative")

    dtype, device = cost_matrix.dtype, cost_matrix.device
    a = normalize_measure_weights(
        source_weights, size=n, dtype=dtype, device=device, name="source_weights"
    )
    b = normalize_measure_weights(
        target_weights, size=m, dtype=dtype, device=device, name="target_weights"
    )

    log_a = torch.log(a)
    log_b = torch.log(b)
    log_K = -cost_matrix / config.epsilon

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    iterations = 0
    converged = False
    for it in range(1, config.max_iterations + 1):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K.transpose(0, 1) + log_u.unsqueeze(0), dim=1)
        iterations = it

        if config.early_stopping and (it % config.check_interval == 0):
            with torch.no_grad():
                log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
                pi_det = torch.exp(log_pi)
                src_err = (pi_det.sum(dim=1) - a).abs().max()
                tgt_err = (pi_det.sum(dim=0) - b).abs().max()
                err = torch.maximum(src_err, tgt_err)
            if float(err) <= config.tolerance:
                converged = True
                break

    log_pi = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    pi = torch.exp(log_pi)

    source_marginal = pi.sum(dim=1)
    target_marginal = pi.sum(dim=0)
    source_marginal_error = (source_marginal - a).abs().max()
    target_marginal_error = (target_marginal - b).abs().max()
    marginal_error = torch.maximum(source_marginal_error, target_marginal_error)

    transport_cost = (pi * cost_matrix).sum()

    tiny = _tiny(dtype)
    reference = a.unsqueeze(1) * b.unsqueeze(0)
    kl = (
        pi * (torch.log(pi + tiny) - torch.log(reference + tiny)) - pi + reference
    ).sum()
    regularized_objective = transport_cost + config.epsilon * kl

    if not config.early_stopping:
        with torch.no_grad():
            converged = bool(marginal_error <= config.tolerance)

    return SinkhornOutput(
        backend=NATIVE_BACKEND,
        coupling=pi,
        transport_cost=transport_cost,
        kl_regularization=kl,
        regularized_objective=regularized_objective,
        source_weights=a,
        target_weights=b,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        source_marginal_error=source_marginal_error,
        target_marginal_error=target_marginal_error,
        marginal_error=marginal_error,
        iterations=iterations,
        converged=converged,
    )


def sinkhorn_divergence_native(
    *,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    geometry: SemanticGeometryConfig,
    config: SinkhornConfig,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> SinkhornDivergenceOutput:
    """Debiased Sinkhorn divergence: cross - 0.5*source_self - 0.5*target_self."""
    prepared_source = prepare_semantic_features(source_features, geometry)
    prepared_target = prepare_semantic_features(target_features, geometry)

    cross_cost = build_transport_cost(
        source_features=prepared_source,
        target_features=prepared_target,
        geometry=geometry,
    ).cost_matrix
    source_cost = build_transport_cost(
        source_features=prepared_source,
        target_features=prepared_source,
        geometry=geometry,
    ).cost_matrix
    target_cost = build_transport_cost(
        source_features=prepared_target,
        target_features=prepared_target,
        geometry=geometry,
    ).cost_matrix

    cross = sinkhorn_coupling_native(
        cost_matrix=cross_cost,
        config=config,
        source_weights=source_weights,
        target_weights=target_weights,
    )
    source_self = sinkhorn_coupling_native(
        cost_matrix=source_cost,
        config=config,
        source_weights=source_weights,
        target_weights=source_weights,
    )
    target_self = sinkhorn_coupling_native(
        cost_matrix=target_cost,
        config=config,
        source_weights=target_weights,
        target_weights=target_weights,
    )

    divergence = (
        cross.regularized_objective
        - 0.5 * source_self.regularized_objective
        - 0.5 * target_self.regularized_objective
    )

    # Clamp only tiny numerical negatives; never hide a materially negative value.
    with torch.no_grad():
        is_tiny_negative = bool(divergence >= -1e-6) and bool(divergence < 0)
    if is_tiny_negative:
        divergence = divergence.clamp_min(0.0)

    return SinkhornDivergenceOutput(
        backend=NATIVE_BACKEND,
        divergence=divergence,
        cross=cross,
        source_self=source_self,
        target_self=target_self,
    )
