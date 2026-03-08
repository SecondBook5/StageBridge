"""Reference-layer diagnostic helpers."""
from __future__ import annotations

from typing import Any

import numpy as np


def summarize_latent(embedding: Any) -> dict[str, object]:
    """Return a small numeric summary for a latent matrix-like object."""
    arr = np.asarray(embedding, dtype=np.float32)
    return {
        "shape": tuple(arr.shape),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }
