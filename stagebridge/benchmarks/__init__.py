"""Benchmark modules for StageBridge evaluation.

This package provides tools for creating benchmarks with ground-truth
interaction rules.

Two approaches:
1. **GroundTruthLabeler**: Apply ground truth labels to REAL data (h5ad, parquet)
2. **SemiSyntheticGenerator**: Generate fully synthetic data with ground truth
"""

from stagebridge.benchmarks.ground_truth_labeler import (
    GroundTruthLabeler,
    GroundTruthLabels,
    InteractionRule,
    DEFAULT_LUAD_RULES,
)

from stagebridge.benchmarks.semisynthetic import (
    SemiSyntheticGenerator,
    SemiSyntheticConfig,
    SemiSyntheticGroundTruth,
    InteractionRule as SemiSyntheticInteractionRule,
    create_demo_semisynthetic,
)

__all__ = [
    # Real data labeling (primary)
    "GroundTruthLabeler",
    "GroundTruthLabels",
    "InteractionRule",
    "DEFAULT_LUAD_RULES",
    # Semi-synthetic (fallback/testing)
    "SemiSyntheticGenerator",
    "SemiSyntheticConfig",
    "SemiSyntheticGroundTruth",
    "create_demo_semisynthetic",
]
