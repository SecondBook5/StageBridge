"""Graph-encoder placeholders for Mission 1 structural rebuild."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class GraphEncodingSummary:
    """Minimal graph-encoder summary."""

    status: str = "structural_stub"
