"""CCRT operators — the context-residual transport decomposition.

Composes typed sender-context attention (sender_context/) with a regulatory
bottleneck and context-residual drift/growth heads into the full CCRT operator.
Every behavior decomposes as ``full = self + (regulatory + residual)``. Depends
only on torch, the standard library, and the system-agnostic sender_context/
layer — never on adapters, transport, training, or disease vocabulary.
"""

from __future__ import annotations

from .context_residual import (
    ContextResidualComponents,
    compose_context_residual,
)
from .drift import DriftHead, DriftHeadConfig, DriftOutput
from .edge_conditioning import EdgeLinear, EdgeLinearConfig
from .growth import GrowthHead, GrowthHeadConfig, GrowthOutput
from .model import (
    ContextResidualTransportConfig,
    ContextResidualTransportOperator,
    ContextResidualTransportOutput,
)
from .regulatory_bottleneck import (
    RegulatoryBottleneck,
    RegulatoryBottleneckConfig,
    RegulatoryBottleneckOutput,
)

__all__ = [
    # edge conditioning
    "EdgeLinearConfig",
    "EdgeLinear",
    # context-residual arithmetic
    "ContextResidualComponents",
    "compose_context_residual",
    # regulatory bottleneck
    "RegulatoryBottleneckConfig",
    "RegulatoryBottleneck",
    "RegulatoryBottleneckOutput",
    # drift head
    "DriftHeadConfig",
    "DriftHead",
    "DriftOutput",
    # growth head
    "GrowthHeadConfig",
    "GrowthHead",
    "GrowthOutput",
    # full operator
    "ContextResidualTransportConfig",
    "ContextResidualTransportOperator",
    "ContextResidualTransportOutput",
]
