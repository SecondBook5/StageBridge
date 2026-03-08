"""TACCO interface placeholder."""
from __future__ import annotations

from .base import SpatialMappingResult


def run_tacco(*args: object, **kwargs: object) -> SpatialMappingResult:
    return SpatialMappingResult(method="tacco", status="not_implemented")
