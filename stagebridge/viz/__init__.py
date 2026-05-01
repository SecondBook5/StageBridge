"""Visualization module for StageBridge publication figures."""

from .figures import (
    load_data,
    compute_embedding,
    compute_ot_flow_field,
    compute_flux_decomposition,
)

__all__ = [
    "load_data",
    "compute_embedding", 
    "compute_ot_flow_field",
    "compute_flux_decomposition",
]
