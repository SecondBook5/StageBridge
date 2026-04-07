"""
Unified benchmark system for StageBridge evaluation.

Consolidates synthetic_v2 and semi-synthetic benchmarks into a single
flexible framework that supports:
- Fully synthetic mode (fast tests, known ground truth)
- Semi-synthetic mode (real expression profiles)
- Hybrid mode (real profiles with causal niche dynamics)
- Expression-aware mode (proper semi-synthetic with DE gene ground truth)
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
from stagebridge.benchmarks.unified.expression_semisynthetic import (
    ExpressionSemisyntheticConfig,
    ExpressionSemisyntheticGenerator,
    InteractionSpec,
    DEGeneSet,
    generate_expression_benchmark,
    create_default_config as create_expression_config,
)
from stagebridge.benchmarks.unified.interaction_metrics import (
    InteractionEvaluationReport,
    DEGenePredictionMetrics,
    SenderPredictionMetrics,
    ReceiverPredictionMetrics,
    PathwayPredictionMetrics,
    StageEffectMetrics,
    evaluate_de_gene_prediction,
    evaluate_sender_prediction,
    evaluate_receiver_prediction,
    evaluate_interaction_benchmark,
    evaluate_length_scale_recovery,
    evaluate_pathway_prediction,
    evaluate_stage_effect_recovery,
    evaluate_pathway_benchmark,
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
    # Expression-aware semi-synthetic
    "ExpressionSemisyntheticConfig",
    "ExpressionSemisyntheticGenerator",
    "InteractionSpec",
    "DEGeneSet",
    "generate_expression_benchmark",
    "create_expression_config",
    # Interaction metrics
    "InteractionEvaluationReport",
    "DEGenePredictionMetrics",
    "SenderPredictionMetrics",
    "ReceiverPredictionMetrics",
    "PathwayPredictionMetrics",
    "StageEffectMetrics",
    "evaluate_de_gene_prediction",
    "evaluate_sender_prediction",
    "evaluate_receiver_prediction",
    "evaluate_interaction_benchmark",
    "evaluate_length_scale_recovery",
    "evaluate_pathway_prediction",
    "evaluate_stage_effect_recovery",
    "evaluate_pathway_benchmark",
]
