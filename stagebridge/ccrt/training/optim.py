"""Optimizer and scheduler factories.

Minimal, explicit factories: AdamW and an optional cosine schedule. No warmup,
no plateau scheduling, no hidden defaults beyond the documented ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "build_optimizer",
    "build_scheduler",
]


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the optimizer factory."""

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"unsupported optimizer '{self.name}'; allowed: adamw")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if not (0.0 < self.beta1 < 1.0):
            raise ValueError("beta1 must be in (0, 1)")
        if not (0.0 < self.beta2 < 1.0):
            raise ValueError("beta2 must be in (0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the scheduler factory."""

    name: str = "none"
    t_max: int = 10
    eta_min: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in ("none", "cosine"):
            raise ValueError(f"unsupported scheduler '{self.name}'; allowed: none, cosine")
        if self.t_max <= 0:
            raise ValueError("t_max must be > 0")
        if self.eta_min < 0:
            raise ValueError("eta_min must be >= 0")


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer over the trainable parameters."""
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters (requires_grad) to optimize")
    return torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: SchedulerConfig):
    """Build a scheduler (None or CosineAnnealingLR)."""
    if config.name == "none":
        return None
    if config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max, eta_min=config.eta_min
        )
    raise ValueError(f"unsupported scheduler '{config.name}'")  # pragma: no cover
