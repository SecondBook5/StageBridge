"""Shared output objects for spatial-mapping methods."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class SpatialMappingArtifacts:
    """Filesystem outputs produced by one spatial-mapping run."""

    method: str
    output_path: Path
