"""Niche-regime summaries from typed context tokens."""

from __future__ import annotations

import numpy as np


def summarize_niche_regimes(
    tokens: np.ndarray, feature_names: tuple[str, ...] | list[str]
) -> dict[str, float]:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got shape {tokens.shape}.")
    return {str(name): float(tokens[:, idx].mean()) for idx, name in enumerate(feature_names)}
