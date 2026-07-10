"""Optional GeomLoss Sinkhorn-divergence backend.

Scalable differentiable Sinkhorn divergence via the stable ``geomloss.SamplesLoss``
interface. This adapter is *optional*: ``geomloss`` is imported lazily inside the
operation, never at module import, and there is no silent fallback to native.

CCRT uses the FULL squared Euclidean cost, so we pass an explicit squared-distance
cost rather than relying on GeomLoss's default half-squared convention. With
``p = 2`` the entropic blur relates to CCRT's epsilon by ``blur = sqrt(epsilon)``.
Only ``squared_euclidean`` geometry is supported here in Milestone 5.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from dataclasses import dataclass

import torch

from ..representations.semantic import SemanticGeometryConfig, prepare_semantic_features
from .backends import (
    GEOMLOSS_BACKEND,
    OptionalTransportBackendUnavailable,
    TransportBackendError,
    require_optional_backend,
)
from .native_sinkhorn import normalize_measure_weights

__all__ = [
    "GeomLossDivergenceConfig",
    "GeomLossDivergenceOutput",
    "sinkhorn_divergence_geomloss",
]

_ALLOWED_GEOMLOSS_BACKENDS = frozenset({"auto", "tensorized", "online", "multiscale"})


@dataclass(frozen=True)
class GeomLossDivergenceConfig:
    """Configuration for the optional GeomLoss divergence adapter."""

    epsilon: float = 0.05
    backend: str = "tensorized"
    scaling: float = 0.8
    debias: bool = True

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.backend not in _ALLOWED_GEOMLOSS_BACKENDS:
            raise ValueError(
                f"geomloss backend '{self.backend}' invalid; allowed: "
                f"{sorted(_ALLOWED_GEOMLOSS_BACKENDS)}"
            )
        if not (0.0 < self.scaling < 1.0):
            raise ValueError("scaling must be in (0, 1)")
        if self.debias is not True:
            raise ValueError(
                "debias must be True for CCRT Sinkhorn-divergence parity"
            )

    @property
    def blur(self) -> float:
        # p = 2 => blur = sqrt(epsilon)
        return math.sqrt(self.epsilon)


@dataclass(frozen=True)
class GeomLossDivergenceOutput:
    """Optional GeomLoss divergence output."""

    backend: str
    divergence: torch.Tensor
    geomloss_backend: str
    epsilon: float
    blur: float


def _full_squared_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Full squared Euclidean cost for GeomLoss ``tensorized`` mode.

    GeomLoss calls the cost with batched inputs ``[B, N, D]`` / ``[B, M, D]`` and
    expects ``[B, N, M]``. ``torch.cdist`` handles the batch dimension.
    """
    dist = torch.cdist(x, y, p=2)
    return dist * dist


def sinkhorn_divergence_geomloss(
    *,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    geometry: SemanticGeometryConfig,
    config: GeomLossDivergenceConfig,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
) -> GeomLossDivergenceOutput:
    """Compute a Sinkhorn divergence via GeomLoss (optional; no fallback)."""
    require_optional_backend(GEOMLOSS_BACKEND)

    if geometry.metric != "squared_euclidean":
        raise TransportBackendError(
            f"GeomLoss adapter supports only 'squared_euclidean' geometry in "
            f"Milestone 5, got '{geometry.metric}'"
        )

    # online/multiscale need PyKeOps; a custom torch cost callable only works in
    # tensorized mode. Require KeOps availability for the KeOps-backed modes.
    if config.backend in ("online", "multiscale"):
        if importlib.util.find_spec("pykeops") is None:
            raise OptionalTransportBackendUnavailable(
                f"geomloss backend '{config.backend}' requires PyKeOps, which is "
                "not installed"
            )

    geomloss = importlib.import_module("geomloss")

    prepared_source = prepare_semantic_features(source_features, geometry)
    prepared_target = prepare_semantic_features(target_features, geometry)

    n = prepared_source.shape[0]
    m = prepared_target.shape[0]
    dtype, device = prepared_source.dtype, prepared_source.device

    if config.backend in ("tensorized", "auto"):
        # Use an explicit custom cost so CCRT's full-squared convention holds.
        loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            cost=_full_squared_cost,
            blur=config.blur,
            scaling=config.scaling,
            debias=config.debias,
            backend="tensorized",
        )
    else:
        # online / multiscale: KeOps expression for full squared distance.
        loss = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            cost="SqDist(X,Y)",
            blur=config.blur,
            scaling=config.scaling,
            debias=config.debias,
            backend=config.backend,
        )

    if source_weights is None and target_weights is None:
        divergence = loss(prepared_source, prepared_target)
    else:
        a = normalize_measure_weights(
            source_weights, size=n, dtype=dtype, device=device, name="source_weights"
        )
        b = normalize_measure_weights(
            target_weights, size=m, dtype=dtype, device=device, name="target_weights"
        )
        divergence = loss(a, prepared_source, b, prepared_target)

    return GeomLossDivergenceOutput(
        backend=GEOMLOSS_BACKEND,
        divergence=divergence,
        geomloss_backend=config.backend,
        epsilon=config.epsilon,
        blur=config.blur,
    )
