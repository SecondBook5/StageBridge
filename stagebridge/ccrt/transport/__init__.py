"""CCRT transport — registered semantic OT geometry and objectives.

Provides the native differentiable Sinkhorn reference backend, optional GeomLoss
and POT adapters (never imported at package import time), barycentric transport
targets, the semantic transport loss, and geometry/stability diagnostics.

Importing this package must succeed without ``geomloss`` or ``ot`` installed;
those optional dependencies are imported lazily only inside their adapter
operations.
"""

from __future__ import annotations

from .backends import (
    DIVERGENCE_BACKENDS,
    EXPLICIT_COUPLING_BACKENDS,
    GEOMLOSS_BACKEND,
    NATIVE_BACKEND,
    POT_BACKEND,
    OptionalTransportBackendUnavailable,
    TransportBackendError,
    TransportBackendMetadata,
    compute_explicit_coupling,
    compute_sinkhorn_divergence,
    get_backend_metadata,
    optional_backend_available,
    require_optional_backend,
)
from .barycentric import (
    BarycentricTransportOutput,
    barycentric_projection,
    build_barycentric_transport_target,
)
from .costs import TransportCostOutput, build_transport_cost
from .diagnostics import (
    coupling_entropy,
    coupling_marginal_error,
    effective_rank,
    mean_drift_alignment,
)
from .geomloss_backend import (
    GeomLossDivergenceConfig,
    GeomLossDivergenceOutput,
    sinkhorn_divergence_geomloss,
)
from .native_sinkhorn import (
    SinkhornConfig,
    SinkhornDivergenceOutput,
    SinkhornOutput,
    normalize_measure_weights,
    sinkhorn_coupling_native,
    sinkhorn_divergence_native,
)
from .pot_backend import (
    POTSinkhornConfig,
    POTSinkhornOutput,
    sinkhorn_coupling_pot,
)
from .semantic_loss import (
    SemanticTransportLoss,
    SemanticTransportLossConfig,
    SemanticTransportLossOutput,
)
from .stability import (
    coupling_frobenius_distance,
    displacement_cosine_stability,
    feature_geometry_alignment,
)

__all__ = [
    # backend policy
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
    # costs
    "TransportCostOutput",
    "build_transport_cost",
    # native sinkhorn
    "SinkhornConfig",
    "SinkhornOutput",
    "SinkhornDivergenceOutput",
    "normalize_measure_weights",
    "sinkhorn_coupling_native",
    "sinkhorn_divergence_native",
    # optional geomloss
    "GeomLossDivergenceConfig",
    "GeomLossDivergenceOutput",
    "sinkhorn_divergence_geomloss",
    # optional pot
    "POTSinkhornConfig",
    "POTSinkhornOutput",
    "sinkhorn_coupling_pot",
    # barycentric
    "BarycentricTransportOutput",
    "barycentric_projection",
    "build_barycentric_transport_target",
    # semantic loss
    "SemanticTransportLossConfig",
    "SemanticTransportLossOutput",
    "SemanticTransportLoss",
    # diagnostics
    "effective_rank",
    "coupling_entropy",
    "coupling_marginal_error",
    "mean_drift_alignment",
    # stability
    "coupling_frobenius_distance",
    "displacement_cosine_stability",
    "feature_geometry_alignment",
]
