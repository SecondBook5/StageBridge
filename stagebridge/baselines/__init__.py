"""
Baseline models for comparison with StageBridge.

These baselines test the core novelty claim:
"Cross-sectional progression becomes more identifiable when cell representations
are conditioned on receiver-centered local niche context and anchored to both
healthy and disease references."

Baselines:
1. PoolingMLP - Simple bag-of-cells pooling + MLP (no structure)
2. DeepSets - Permutation-invariant set function
3. SetTransformer - Flat attention without spatial structure
4. GraphSAGE - Graph neural network with spatial edges
5. StageBridge - Receiver-centered niche + dual references (full model)

Usage:
    from stagebridge.baselines import run_baseline_comparison

    results = run_baseline_comparison(
        benchmark_dir=Path("data/benchmarks/granular_medium"),
        output_dir=Path("results/baselines"),
    )
"""

from stagebridge.baselines.evaluate import (
    run_baseline_comparison,
    load_benchmark_world,
    load_benchmark_split,
    BenchmarkWorld,
    EvaluationMetrics,
)
from stagebridge.baselines.graph_sage import GraphSAGEBaseline

# Import existing baselines from other modules
from stagebridge.context_model.baselines_lesion import (
    PooledLesionBaseline,
    DeepSetsLesionBaseline,
    LesionSetTransformerBaseline,
)
from stagebridge.transition_model.baselines import (
    DeepSetsFlowModel,
    NoContextFlowModel,
    LinearTransitionBaseline,
)

__all__ = [
    # Evaluation
    "run_baseline_comparison",
    "load_benchmark_world",
    "load_benchmark_split",
    "BenchmarkWorld",
    "EvaluationMetrics",
    # Baselines
    "PooledLesionBaseline",
    "DeepSetsLesionBaseline",
    "LesionSetTransformerBaseline",
    "GraphSAGEBaseline",
    "DeepSetsFlowModel",
    "NoContextFlowModel",
    "LinearTransitionBaseline",
]
