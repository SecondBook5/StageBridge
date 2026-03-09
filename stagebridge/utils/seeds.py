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


def seed_everything(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds when PyTorch is available."""
    set_global_seed(seed)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
