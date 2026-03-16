"""Metadata helpers for the secondary brain metastasis cohort."""

from __future__ import annotations

from dataclasses import dataclass

from stagebridge.data.brainmets._raw import load_brainmets_metadata


@dataclass(slots=True, frozen=True)
class BrainMetsDataset:
    """Secondary utility dataset retained outside the flagship v1 path."""

    name: str = "brainmets"
    stages: tuple[str, ...] = ("LUAD", "BrainMet", "ChestWallMet")
    modalities: tuple[str, ...] = ("snRNA-seq", "spatial", "lpWGS")


__all__ = ["BrainMetsDataset", "load_brainmets_metadata"]
