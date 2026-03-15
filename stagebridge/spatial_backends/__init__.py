"""
Spatial transcriptomics mapping backend wrappers.

Provides unified interface for multiple spatial mapping methods:
- Tangram: Marker-based mapping with gradient-based optimization
- DestVI: VAE-based probabilistic mapping with amortized inference
- TACCO: Compositional transfer with optimal transport

All backends output standardized format for downstream StageBridge modules.
"""

from .base import SpatialBackend, SpatialMappingResult
from .tangram_wrapper import TangramBackend
from .destvi_wrapper import DestVIBackend
from .tacco_wrapper import TACCOBackend

__all__ = [
    "SpatialBackend",
    "SpatialMappingResult",
    "TangramBackend",
    "DestVIBackend",
    "TACCOBackend",
]


def get_backend(name: str) -> type[SpatialBackend]:
    """
    Get spatial mapping backend by name.

    Args:
        name: Backend name ('tangram', 'destvi', or 'tacco')

    Returns:
        Backend class
    """
    backends = {
        "tangram": TangramBackend,
        "destvi": DestVIBackend,
        "tacco": TACCOBackend,
    }

    if name.lower() not in backends:
        raise ValueError(
            f"Unknown backend: {name}. "
            f"Available: {list(backends.keys())}"
        )

    return backends[name.lower()]
