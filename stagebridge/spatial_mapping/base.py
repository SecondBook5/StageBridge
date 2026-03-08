"""Base interfaces for spatial mapping methods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True, frozen=True)
class SpatialMappingResult:
    """Standardized spatial-mapping result contract."""

    method: str
    status: str


class SpatialMapper(Protocol):
    """Protocol implemented by spatial-mapping wrappers."""

    def run(self) -> SpatialMappingResult:
        ...
