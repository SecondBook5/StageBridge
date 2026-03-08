"""Gaussian bridge initialization placeholders."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class GaussianBridgeInit:
    """Configuration for Gaussian bridge initialization."""

    sigma: float = 0.1
