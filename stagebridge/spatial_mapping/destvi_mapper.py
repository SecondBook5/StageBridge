"""DestVI interface placeholder."""
from __future__ import annotations

from .base import SpatialMappingResult


def run_destvi(*args: object, **kwargs: object) -> SpatialMappingResult:
    return SpatialMappingResult(method="destvi", status="not_implemented")
