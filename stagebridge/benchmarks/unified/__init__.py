"""
Unified benchmark system for StageBridge evaluation.

Consolidates synthetic_v2 and semi-synthetic benchmarks into a single
flexible framework that supports:
- Fully synthetic mode (fast tests, known ground truth)
- Semi-synthetic mode (real expression profiles)
- Hybrid mode (real profiles with causal niche dynamics)
"""

from stagebridge.benchmarks.unified.config import (
    UnifiedBenchmarkConfig,
    SmokeTestConfig,
    FullBenchmarkConfig,
    NicheInfluenceSpec,
    InteractionRule,
    CellGroupSpec,
    DynamicsConfig,
)
from stagebridge.benchmarks.unified.generator import (
    UnifiedBenchmarkGenerator,
    generate_benchmark,
)
from stagebridge.benchmarks.unified.ground_truth import (
    GroundTruth,
    GroundTruthRecovery,
)

__all__ = [
    # Config
    "UnifiedBenchmarkConfig",
    "SmokeTestConfig",
    "FullBenchmarkConfig",
    "NicheInfluenceSpec",
    "InteractionRule",
    "CellGroupSpec",
    "DynamicsConfig",
    # Generator
    "UnifiedBenchmarkGenerator",
    "generate_benchmark",
    # Ground truth
    "GroundTruth",
    "GroundTruthRecovery",
]
