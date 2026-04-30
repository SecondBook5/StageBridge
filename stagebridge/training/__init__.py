"""Training infrastructure for StageBridge."""

from stagebridge.training.config import TrainingConfig
from stagebridge.training.checkpoint import CheckpointManager
from stagebridge.training.metrics import MetricsLogger
from stagebridge.training.scheduler import create_lr_scheduler
from stagebridge.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
)
from stagebridge.training.trainer import (
    StageBridgeTrainer,
    TrainerConfig,
    train_stagebridge,
)

__all__ = [
    # Trainer
    "StageBridgeTrainer",
    "TrainerConfig",
    "train_stagebridge",
    # Config
    "TrainingConfig",
    # Infrastructure
    "CheckpointManager",
    "MetricsLogger",
    "create_lr_scheduler",
    # Distributed
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_world_size",
    "get_rank",
]
