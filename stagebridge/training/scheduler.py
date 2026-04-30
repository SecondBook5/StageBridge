"""Learning rate scheduler factory."""

from __future__ import annotations

import numpy as np
import torch


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int,
    min_lr: float,
    use_cosine: bool = True,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create LR scheduler with warmup + cosine decay.

    Args:
        optimizer: The optimizer
        num_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs (linear increase)
        min_lr: Minimum learning rate at end of cosine decay
        use_cosine: If True, use cosine annealing; else constant after warmup

    Returns:
        Configured learning rate scheduler
    """
    if warmup_epochs > 0 and use_cosine:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return max(min_lr / optimizer.defaults["lr"], 0.5 * (1 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif use_cosine:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )

    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
