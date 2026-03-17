"""StageBridge benchmark infrastructure.

This module provides benchmark generation for StageBridge evaluation:

- unified: Consolidated benchmark system (recommended)
- semi_synthetic: Legacy semi-synthetic benchmarks (uses unified internally)
"""

from __future__ import annotations

from stagebridge.benchmarks.unified import (
    UnifiedBenchmarkConfig,
    SmokeTestConfig,
    FullBenchmarkConfig,
    UnifiedBenchmarkGenerator,
    generate_benchmark,
    GroundTruth,
    GroundTruthRecovery,
)

__all__ = [
    "UnifiedBenchmarkConfig",
    "SmokeTestConfig",
    "FullBenchmarkConfig",
    "UnifiedBenchmarkGenerator",
    "generate_benchmark",
    "GroundTruth",
    "GroundTruthRecovery",
]
