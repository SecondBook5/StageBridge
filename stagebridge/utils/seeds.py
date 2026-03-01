"""Reproducibility: global random seed management."""
from __future__ import annotations

import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python random, NumPy.

    For deep learning frameworks (PyTorch, JAX), call their own seed APIs.
    """
    random.seed(seed)
    np.random.seed(seed)
