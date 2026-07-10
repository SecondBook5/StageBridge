"""Transport backend policy: native reference + explicit optional backends.

The native PyTorch backend is mandatory and default. GeomLoss (scalable
differentiable Sinkhorn divergence) and POT (coupling reference/validation) are
optional: they are never imported at module-import time, never silently
substituted, and must be selected explicitly. Requesting an unavailable optional
backend raises a clear error rather than falling back to native.

Every transport output records the backend that produced it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..contracts.errors import CCRTValidationError
from ..representations.semantic import SemanticGeometryConfig

if TYPE_CHECKING:  # avoid import cycles at runtime
    from .geomloss_backend import GeomLossDivergenceConfig
    from .native_sinkhorn import SinkhornConfig
    from .pot_backend import POTSinkhornConfig

__all__ = [
    "NATIVE_BACKEND",
    "GEOMLOSS_BACKEND",
    "POT_BACKEND",
    "EXPLICIT_COUPLING_BACKENDS",
    "DIVERGENCE_BACKENDS",
    "TransportBackendError",
    "OptionalTransportBackendUnavailable",
    "TransportBackendMetadata",
    "optional_backend_available",
    "require_optional_backend",
    "get_backend_metadata",
    "compute_explicit_coupling",
    "compute_sinkhorn_divergence",
]

NATIVE_BACKEND = "native"
GEOMLOSS_BACKEND = "geomloss"
POT_BACKEND = "pot"

EXPLICIT_COUPLING_BACKENDS = frozenset({"native", "pot"})
DIVERGENCE_BACKENDS = frozenset({"native", "geomloss"})

_ALL_BACKENDS = frozenset({NATIVE_BACKEND, GEOMLOSS_BACKEND, POT_BACKEND})
#: Import module name for each optional backend.
_OPTIONAL_MODULES = {GEOMLOSS_BACKEND: "geomloss", POT_BACKEND: "ot"}


class TransportBackendError(CCRTValidationError):
    """An invalid or unsupported transport backend was requested."""


class OptionalTransportBackendUnavailable(CCRTValidationError):
    """A requested optional transport backend is not installed."""


def optional_backend_available(backend: str) -> bool:
    """True if ``backend`` can be used now (native is always available)."""
    if backend == NATIVE_BACKEND:
        return True
    module = _OPTIONAL_MODULES.get(backend)
    if module is None:
        raise TransportBackendError(
            f"unknown backend '{backend}'; known: {sorted(_ALL_BACKENDS)}"
        )
    return importlib.util.find_spec(module) is not None


def require_optional_backend(backend: str) -> None:
    """Raise if ``backend`` is unknown or an unavailable optional backend."""
    if backend not in _ALL_BACKENDS:
        raise TransportBackendError(
            f"unknown backend '{backend}'; known: {sorted(_ALL_BACKENDS)}"
        )
    if backend == NATIVE_BACKEND:
        return
    if not optional_backend_available(backend):
        module = _OPTIONAL_MODULES[backend]
        raise OptionalTransportBackendUnavailable(
            f"optional transport backend '{backend}' requires the '{module}' "
            "package, which is not installed. Install it explicitly; CCRT will "
            "not silently fall back to the native backend."
        )


@dataclass(frozen=True)
class TransportBackendMetadata:
    """Static + lazily-resolved metadata for a transport backend."""

    backend: str
    implementation: str
    version: str | None
    differentiable: bool
    explicit_coupling: bool


def _lazy_version(module_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - defensive
        return None
    return getattr(module, "__version__", None)


def get_backend_metadata(backend: str) -> TransportBackendMetadata:
    """Return metadata for a backend without requiring it to be installed."""
    if backend == NATIVE_BACKEND:
        return TransportBackendMetadata(
            backend=NATIVE_BACKEND,
            implementation="stagebridge.ccrt.transport.native_sinkhorn",
            version=torch.__version__,
            differentiable=True,
            explicit_coupling=True,
        )
    if backend == GEOMLOSS_BACKEND:
        return TransportBackendMetadata(
            backend=GEOMLOSS_BACKEND,
            implementation="geomloss.SamplesLoss",
            version=_lazy_version("geomloss"),
            differentiable=True,
            explicit_coupling=False,
        )
    if backend == POT_BACKEND:
        return TransportBackendMetadata(
            backend=POT_BACKEND,
            implementation="ot.sinkhorn",
            version=_lazy_version("ot"),
            differentiable=False,
            explicit_coupling=True,
        )
    raise TransportBackendError(
        f"unknown backend '{backend}'; known: {sorted(_ALL_BACKENDS)}"
    )


def compute_explicit_coupling(
    *,
    backend: str,
    cost_matrix: torch.Tensor,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    native_config: "SinkhornConfig | None" = None,
    pot_config: "POTSinkhornConfig | None" = None,
) -> Any:
    """Dispatch an explicit-coupling solve to native or POT (no fallback)."""
    if backend == NATIVE_BACKEND:
        from .native_sinkhorn import SinkhornConfig, sinkhorn_coupling_native

        return sinkhorn_coupling_native(
            cost_matrix=cost_matrix,
            config=native_config or SinkhornConfig(),
            source_weights=source_weights,
            target_weights=target_weights,
        )
    if backend == POT_BACKEND:
        require_optional_backend(POT_BACKEND)
        from .pot_backend import POTSinkhornConfig, sinkhorn_coupling_pot

        return sinkhorn_coupling_pot(
            cost_matrix=cost_matrix,
            config=pot_config or POTSinkhornConfig(),
            source_weights=source_weights,
            target_weights=target_weights,
        )
    raise TransportBackendError(
        f"backend '{backend}' is not a valid explicit-coupling backend; "
        f"allowed: {sorted(EXPLICIT_COUPLING_BACKENDS)}"
    )


def compute_sinkhorn_divergence(
    *,
    backend: str,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    geometry: SemanticGeometryConfig,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    native_config: "SinkhornConfig | None" = None,
    geomloss_config: "GeomLossDivergenceConfig | None" = None,
) -> Any:
    """Dispatch a Sinkhorn divergence to native or GeomLoss (no fallback)."""
    if backend == NATIVE_BACKEND:
        from .native_sinkhorn import SinkhornConfig, sinkhorn_divergence_native

        return sinkhorn_divergence_native(
            source_features=source_features,
            target_features=target_features,
            geometry=geometry,
            config=native_config or SinkhornConfig(),
            source_weights=source_weights,
            target_weights=target_weights,
        )
    if backend == GEOMLOSS_BACKEND:
        require_optional_backend(GEOMLOSS_BACKEND)
        from .geomloss_backend import (
            GeomLossDivergenceConfig,
            sinkhorn_divergence_geomloss,
        )

        return sinkhorn_divergence_geomloss(
            source_features=source_features,
            target_features=target_features,
            geometry=geometry,
            config=geomloss_config or GeomLossDivergenceConfig(),
            source_weights=source_weights,
            target_weights=target_weights,
        )
    raise TransportBackendError(
        f"backend '{backend}' is not a valid divergence backend; "
        f"allowed: {sorted(DIVERGENCE_BACKENDS)}"
    )
