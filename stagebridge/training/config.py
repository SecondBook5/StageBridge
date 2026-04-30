"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stagebridge.contracts import LATENT_DIM, N_STAGES, N_PROGENY_PATHWAYS


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults.

    All hyperparameters in one place. Import from contracts for dimensions.
    """

    # Data
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("runs"))

    # Model dimensions (from contracts)
    latent_dim: int = LATENT_DIM
    n_stages: int = N_STAGES
    n_pathways: int = N_PROGENY_PATHWAYS

    # Architecture
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss weights
    transition_weight: float = 1.0
    pathway_weight: float = 0.1
    proliferation_weight: float = 0.1

    # Cross-validation
    n_folds: int = 5
    n_seeds: int = 3

    # Checkpointing
    checkpoint_every: int = 10
    keep_top_k: int = 3

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
