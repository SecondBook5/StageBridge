"""Diffusion-network placeholder for Mission 1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DiffusionConfig:
    """Placeholder diffusion configuration."""

    state_dependent: bool = True
