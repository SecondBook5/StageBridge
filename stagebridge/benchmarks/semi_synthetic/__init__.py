"""
Semi-synthetic benchmark for StageBridge representation learning and niche modeling.

This benchmark uses real expression profiles from HLCA, LuCA, and progression snRNA
data, combined with synthetic spatial structure and explicit sender->receiver interaction
rules to test whether StageBridge can recover:

- Receiver-centered niche effects
- Distance-dependent interactions
- Stage-aware progression structure
- Dual-reference usefulness

The benchmark is NOT a generic synthetic toy - it is a StageBridge-specific evaluation
pipeline that uses real cells as the expression substrate.
"""

from __future__ import annotations

from stagebridge.benchmarks.semi_synthetic.configs import (
    BenchmarkConfig,
    InteractionRule,
    SmokeConfig,
)
from stagebridge.benchmarks.semi_synthetic.benchmark_generator import (
    SemiSyntheticBenchmarkGenerator,
    generate_benchmark,
)
from stagebridge.benchmarks.semi_synthetic.metrics import (
    evaluate_receiver_state_recovery,
    evaluate_sender_attribution,
    evaluate_distance_sensitivity,
    compute_benchmark_metrics,
)

__all__ = [
    "BenchmarkConfig",
    "InteractionRule",
    "SmokeConfig",
    "SemiSyntheticBenchmarkGenerator",
    "generate_benchmark",
    "evaluate_receiver_state_recovery",
    "evaluate_sender_attribution",
    "evaluate_distance_sensitivity",
    "compute_benchmark_metrics",
]
