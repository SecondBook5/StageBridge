"""CCRT synthetic — known-truth mechanism-recovery benchmark.

An independent mathematical teacher generates controlled context mechanisms; the
CCRT student trains only on observable synthetic data and is evaluated, after
training, on how well it recovers the hidden mechanism via factual/null
counterfactuals and scenario-specific diagnostics. The teacher is
mathematically independent from the student model.
"""

from __future__ import annotations

from .benchmark import (
    SyntheticBenchmarkComponents,
    SyntheticBenchmarkOutput,
    SyntheticPrediction,
    SyntheticScenarioResult,
    build_synthetic_benchmark_components,
    build_synthetic_training_sequence,
    evaluate_context_counterfactual,
    predict_synthetic_batch,
    run_synthetic_benchmark_matrix,
    run_synthetic_scenario_benchmark,
)
from .config import SyntheticBenchmarkConfig, SyntheticSystemConfig
from .counterfactuals import (
    attach_teacher_targets,
    remove_all_sender_context,
    remove_sender_context_type,
    replace_transition_edge,
    set_sender_distances,
)
from .generator import (
    SyntheticDatasetBundle,
    SyntheticExample,
    build_synthetic_biological_system_spec,
    generate_synthetic_dataset,
)
from .ground_truth import (
    SyntheticGroundTruth,
    SyntheticTeacher,
    SyntheticTeacherParameters,
)
from .mechanisms import (
    DISTANCE_DEPENDENT,
    DRIFT_ONLY,
    GROWTH_ONLY,
    MIXED_DRIFT_GROWTH,
    NULL_CONTEXT,
    REGULATORY_MEDIATED,
    SENDER_TYPE_SPECIFIC,
    SYNTHETIC_SCENARIO_IDS,
    TRANSITION_EDGE_SPECIFIC,
    WRONG_CONTEXT_NEGATIVE_CONTROL,
    SyntheticMechanismSpec,
    build_synthetic_mechanism_spec,
)
from .metrics import (
    CounterfactualRecoveryMetrics,
    mean_cosine_recovery,
    mean_effect_norm,
    pearson_recovery,
    rank_order_recovery,
    relative_root_mean_squared_error,
    root_mean_squared_error,
    sign_agreement,
)

__all__ = [
    # config
    "SyntheticSystemConfig",
    "SyntheticBenchmarkConfig",
    # scenario constants
    "NULL_CONTEXT",
    "DRIFT_ONLY",
    "GROWTH_ONLY",
    "MIXED_DRIFT_GROWTH",
    "REGULATORY_MEDIATED",
    "DISTANCE_DEPENDENT",
    "SENDER_TYPE_SPECIFIC",
    "TRANSITION_EDGE_SPECIFIC",
    "WRONG_CONTEXT_NEGATIVE_CONTROL",
    "SYNTHETIC_SCENARIO_IDS",
    # mechanisms
    "SyntheticMechanismSpec",
    "build_synthetic_mechanism_spec",
    # teacher
    "SyntheticTeacherParameters",
    "SyntheticGroundTruth",
    "SyntheticTeacher",
    # generator
    "SyntheticExample",
    "SyntheticDatasetBundle",
    "build_synthetic_biological_system_spec",
    "generate_synthetic_dataset",
    # counterfactuals
    "remove_all_sender_context",
    "remove_sender_context_type",
    "set_sender_distances",
    "replace_transition_edge",
    "attach_teacher_targets",
    # metrics
    "root_mean_squared_error",
    "relative_root_mean_squared_error",
    "mean_cosine_recovery",
    "pearson_recovery",
    "sign_agreement",
    "mean_effect_norm",
    "rank_order_recovery",
    "CounterfactualRecoveryMetrics",
    # benchmark
    "SyntheticPrediction",
    "SyntheticBenchmarkComponents",
    "SyntheticScenarioResult",
    "SyntheticBenchmarkOutput",
    "predict_synthetic_batch",
    "evaluate_context_counterfactual",
    "build_synthetic_benchmark_components",
    "build_synthetic_training_sequence",
    "run_synthetic_scenario_benchmark",
    "run_synthetic_benchmark_matrix",
]
