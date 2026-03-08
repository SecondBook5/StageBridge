"""Minimal schema objects for the StageBridge data layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Named dataset contract used by data-layer readers."""

    name: str
    modality: str
    stage_column: str = "stage"
    donor_column: str = "patient_id"
