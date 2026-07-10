"""Reproducible seeding for CCRT training.

Seeds Python's ``random`` and PyTorch (CPU + all CUDA devices when present) and
enables deterministic algorithms. No NumPy dependency; no global environment
mutation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

__all__ = ["ReproducibilityState", "set_reproducible_seed"]


@dataclass(frozen=True)
class ReproducibilityState:
    """Records the seeding that was applied."""

    seed: int
    deterministic_algorithms: bool
    cuda_seeded: bool


def set_reproducible_seed(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
) -> ReproducibilityState:
    """Seed Python + torch reproducibly and (optionally) force determinism."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    random.seed(seed)
    torch.manual_seed(seed)

    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
        # deterministic cuDNN behavior when available
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(deterministic_algorithms)

    return ReproducibilityState(
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        cuda_seeded=cuda_seeded,
    )
