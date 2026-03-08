"""Metadata helpers for the LUAD evolution cohort."""
from __future__ import annotations

from dataclasses import dataclass

from .stages import CANONICAL_STAGE_ORDER


@dataclass(slots=True, frozen=True)
class LuadEvoDataset:
    """High-level dataset contract for the active v1 cohort."""

    name: str = "luad_evo"
    stages: tuple[str, ...] = CANONICAL_STAGE_ORDER
    modalities: tuple[str, ...] = ("snRNA-seq", "Visium", "WES")
