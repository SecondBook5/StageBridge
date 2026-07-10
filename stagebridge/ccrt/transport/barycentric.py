"""Barycentric transport targets.

Given an explicit coupling ``pi`` [N, M] and target features ``z_target`` [M, D],
the barycentric projection maps each source row to the coupling-weighted mean of
the targets it transports to:

    z_barycentric_i = (sum_j pi_ij z_target_j) / (sum_j pi_ij)

The target displacement is ``z_barycentric - z_source``. These are the transport
supervision signals for the semantic loss. Gradient-preserving; no NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..contracts.errors import CCRTShapeError, CCRTValidationError

__all__ = [
    "BarycentricTransportOutput",
    "barycentric_projection",
    "build_barycentric_transport_target",
]


@dataclass(frozen=True)
class BarycentricTransportOutput:
    """Barycentric target, displacement, transported mass, and backend label."""

    barycentric_target: torch.Tensor
    target_displacement: torch.Tensor
    transported_source_mass: torch.Tensor
    coupling_backend: str


def _validate_matrix(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise CCRTValidationError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(t):
        raise CCRTValidationError(f"{name} must be floating point")
    if t.dim() != 2:
        raise CCRTShapeError(f"{name} must be rank 2, got shape {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise CCRTValidationError(f"{name} contains non-finite values")


def barycentric_projection(
    *,
    coupling: torch.Tensor,
    target_features: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (barycentric_target [N, D], row_mass [N]) from a coupling."""
    _validate_matrix(coupling, "coupling")
    _validate_matrix(target_features, "target_features")
    if bool((coupling < 0).any()):
        raise CCRTValidationError("coupling must be non-negative")
    if eps <= 0.0:
        raise CCRTValidationError("eps must be > 0")

    n, m = coupling.shape
    if target_features.shape[0] != m:
        raise CCRTShapeError(
            f"coupling M {m} != target_features rows {target_features.shape[0]}"
        )

    row_mass = coupling.sum(dim=1)  # [N]
    if bool((row_mass <= eps).any()):
        raise CCRTValidationError(
            "every coupling row must carry mass > eps for barycentric projection"
        )

    weighted = coupling @ target_features  # [N, D]
    barycentric_target = weighted / row_mass.unsqueeze(1)
    return barycentric_target, row_mass


def build_barycentric_transport_target(
    *,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    coupling: torch.Tensor,
    coupling_backend: str,
    eps: float = 1e-8,
) -> BarycentricTransportOutput:
    """Build the barycentric target and displacement for source rows."""
    _validate_matrix(source_features, "source_features")
    barycentric_target, row_mass = barycentric_projection(
        coupling=coupling, target_features=target_features, eps=eps
    )
    if source_features.shape[0] != coupling.shape[0]:
        raise CCRTShapeError(
            f"source_features rows {source_features.shape[0]} != coupling N "
            f"{coupling.shape[0]}"
        )
    if source_features.shape[1] != target_features.shape[1]:
        raise CCRTShapeError(
            f"source dim {source_features.shape[1]} != target dim "
            f"{target_features.shape[1]}"
        )

    target_displacement = barycentric_target - source_features
    return BarycentricTransportOutput(
        barycentric_target=barycentric_target,
        target_displacement=target_displacement,
        transported_source_mass=row_mass,
        coupling_backend=coupling_backend,
    )
