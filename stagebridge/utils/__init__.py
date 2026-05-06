"""General-purpose utilities for StageBridge."""

from .types import DatasetAuditReport, RunManifest, StageBatch, StageBridgeConfig
from .multigpu import (
    setup_distributed_env,
    get_gpu_info,
    get_device,
    get_accelerator_config,
    get_lightning_trainer_kwargs,
    get_scvi_train_kwargs,
    print_gpu_status,
)

__all__ = [
    # Types
    "StageBridgeConfig",
    "StageBatch",
    "DatasetAuditReport",
    "RunManifest",
    # Multi-GPU utilities
    "setup_distributed_env",
    "get_gpu_info",
    "get_device",
    "get_accelerator_config",
    "get_lightning_trainer_kwargs",
    "get_scvi_train_kwargs",
    "print_gpu_status",
]
