"""Base interfaces for spatial mapping methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(slots=True, frozen=True)
class SpatialMappingResult:
    """Standardized spatial-mapping result contract."""

    method: str
    status: str
    provider_version: str | None = None
    execution_mode: str | None = None
    compositions: np.ndarray | None = None
    coords: np.ndarray | None = None
    obs: pd.DataFrame | None = None
    feature_names: tuple[str, ...] = ()
    source_path: Path | None = None
    qc: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    notes: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "provider_version": self.provider_version,
            "execution_mode": self.execution_mode,
            "n_spots": 0 if self.compositions is None else int(self.compositions.shape[0]),
            "n_features": 0 if self.compositions is None else int(self.compositions.shape[1]),
            "feature_names": list(self.feature_names),
            "source_path": None if self.source_path is None else str(self.source_path),
            "qc": self.qc or {},
            "provenance": self.provenance or {},
            "notes": self.notes,
        }


class SpatialMapper(Protocol):
    """Protocol implemented by spatial-mapping wrappers."""

    def run(self) -> SpatialMappingResult: ...
