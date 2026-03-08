"""Typed token schema used by the context model."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TokenType:
    """One typed biological token category."""

    name: str
    prefix: str
