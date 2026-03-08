"""Coupling helpers for the transition layer."""
from __future__ import annotations

from typing import Any


def build_ot_coupling(*args: Any, **kwargs: Any) -> Any:
    """Dispatch to the existing loss-layer coupling helper when needed."""
    from stagebridge.transition_model.losses import build_sinkhorn_coupling

    return build_sinkhorn_coupling(*args, **kwargs)
