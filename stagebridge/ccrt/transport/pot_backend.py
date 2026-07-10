"""Optional POT (Python Optimal Transport) coupling backend.

A reference/validation backend for explicit couplings via ``ot.sinkhorn`` on
POT's PyTorch array backend. Optional: ``ot`` is imported lazily inside the
operation, never at module import, and there is no silent fallback to native.
Inputs/outputs stay as torch tensors — no NumPy conversion. POT is not the
default differentiable training path in Milestone 5, and gradient equivalence is
not required of it here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch

from .backends import POT_BACKEND, TransportBackendError, require_optional_backend
from .native_sinkhorn import normalize_measure_weights

__all__ = ["POTSinkhornConfig", "POTSinkhornOutput", "sinkhorn_coupling_pot"]


@dataclass(frozen=True)
class POTSinkhornConfig:
    """Configuration for the optional POT Sinkhorn coupling."""

    epsilon: float = 0.05
    max_iterations: int = 200
    tolerance: float = 1e-9
    method: str = "sinkhorn_log"

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be > 0")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")


@dataclass(frozen=True)
class POTSinkhornOutput:
    """Optional POT coupling output."""

    backend: str
    coupling: torch.Tensor
    transport_cost: torch.Tensor
    source_marginal: torch.Tensor
    target_marginal: torch.Tensor
    marginal_error: torch.Tensor


def sinkhorn_coupling_pot(
    *,
    cost_matrix: torch.Tensor,
    config: POTSinkhornConfig,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> POTSinkhornOutput:
    """Compute an explicit Sinkhorn coupling via POT (optional; no fallback)."""
    require_optional_backend(POT_BACKEND)

    if not isinstance(cost_matrix, torch.Tensor):
        raise ValueError("cost_matrix must be a torch.Tensor")
    if not torch.is_floating_point(cost_matrix):
        raise ValueError("cost_matrix must be floating point")
    if cost_matrix.dim() != 2:
        raise ValueError(
            f"cost_matrix must be rank 2 [N, M], got {tuple(cost_matrix.shape)}"
        )
    n, m = cost_matrix.shape
    if n <= 0 or m <= 0:
        raise ValueError("cost_matrix must have N > 0 and M > 0")
    if not bool(torch.isfinite(cost_matrix).all()):
        raise ValueError("cost_matrix contains non-finite values")

    ot = importlib.import_module("ot")

    dtype, device = cost_matrix.dtype, cost_matrix.device
    a = normalize_measure_weights(
        source_weights, size=n, dtype=dtype, device=device, name="source_weights"
    )
    b = normalize_measure_weights(
        target_weights, size=m, dtype=dtype, device=device, name="target_weights"
    )

    try:
        coupling = ot.sinkhorn(
            a,
            b,
            cost_matrix,
            reg=config.epsilon,
            method=config.method,
            numItermax=config.max_iterations,
            stopThr=config.tolerance,
        )
    except (ValueError, NotImplementedError, KeyError) as exc:
        raise TransportBackendError(
            f"POT sinkhorn failed for method '{config.method}': {exc}"
        ) from exc

    if not isinstance(coupling, torch.Tensor):
        raise TransportBackendError(
            "POT did not return a torch.Tensor coupling; ensure inputs use "
            "POT's PyTorch backend"
        )
    if tuple(coupling.shape) != (n, m):
        raise TransportBackendError(
            f"POT coupling shape {tuple(coupling.shape)} != expected {(n, m)}"
        )
    if not torch.is_floating_point(coupling):
        raise TransportBackendError("POT coupling must be floating point")
    if not bool(torch.isfinite(coupling).all()):
        raise TransportBackendError("POT coupling contains non-finite values")
    if bool((coupling < -1e-9).any()):
        raise TransportBackendError("POT coupling must be non-negative")

    source_marginal = coupling.sum(dim=1)
    target_marginal = coupling.sum(dim=0)
    marginal_error = torch.maximum(
        (source_marginal - a).abs().max(), (target_marginal - b).abs().max()
    )
    transport_cost = (coupling * cost_matrix).sum()

    return POTSinkhornOutput(
        backend=POT_BACKEND,
        coupling=coupling,
        transport_cost=transport_cost,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        marginal_error=marginal_error,
    )
