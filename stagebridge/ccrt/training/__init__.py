"""CCRT training — objective, trainer, optimizer factories, and checkpointing.

Composes the operator (operators/) and semantic transport loss (transport/) into
a reproducible training loop over pre-ordered ``CCRTTrainingBatch`` inputs. Fully
system-agnostic: it knows grammar-typed categorical indices and generic tensors,
never disease vocabulary. Imports the core model/transport/data packages; nothing
upstream imports training.
"""

from __future__ import annotations

from .batch import CCRTTrainingBatch, build_training_batch
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointMetadata,
    build_checkpoint_state,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from .objective import (
    CompositeCCRTObjective,
    CompositeCCRTObjectiveConfig,
    CompositeCCRTObjectiveOutput,
)
from .optim import (
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)
from .reproducibility import ReproducibilityState, set_reproducible_seed
from .trainer import CCRTTrainer, EpochMetrics, TrainerConfig, TrainingStepMetrics

__all__ = [
    # reproducibility
    "ReproducibilityState",
    "set_reproducible_seed",
    # batch
    "CCRTTrainingBatch",
    "build_training_batch",
    # objective
    "CompositeCCRTObjectiveConfig",
    "CompositeCCRTObjectiveOutput",
    "CompositeCCRTObjective",
    # optim
    "OptimizerConfig",
    "SchedulerConfig",
    "build_optimizer",
    "build_scheduler",
    # trainer
    "TrainerConfig",
    "TrainingStepMetrics",
    "EpochMetrics",
    "CCRTTrainer",
    # checkpointing
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "build_checkpoint_state",
    "save_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
]
