"""
Dataset readers, loaders, and synthetic data generation for StageBridge.

This module provides:
- Synthetic data generation for testing and validation
- Data loaders optimized for training (loaders_optimized.py)
- QC and normalization pipelines (qc.py, normalize.py)
- Data ingestion from various sources (ingest.py)

Synthetic Data:
    Two generators are available:
    - SyntheticDataGenerator: Simple generator for quick tests
    - SyntheticDataGeneratorV2: Full ground truth with Suites A-D

Example:
    >>> from stagebridge.data import generate_synthetic_v2, SyntheticConfig
    >>> config = SyntheticConfig(n_cells=1000, n_donors=5)
    >>> data = generate_synthetic_v2(config)
"""

from stagebridge.data.synthetic import (
    SyntheticDataGenerator,
    SyntheticDataGeneratorV2,
    SyntheticConfig,
    generate_synthetic_dataset,
    generate_synthetic_v2,
)

__all__ = [
    "SyntheticDataGenerator",
    "SyntheticDataGeneratorV2",
    "SyntheticConfig",
    "generate_synthetic_dataset",
    "generate_synthetic_v2",
]
