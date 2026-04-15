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
    from stagebridge.baselines import load_benchmark_tensors, train_baseline

    tensors = load_benchmark_tensors(Path("data/canonical/benchmark"))
    # Train and evaluate baselines on semi_synthetic.pt
"""

# Note: Legacy per-cell evaluation functions removed (were deprecated)
# Use set_baselines for H2 validation instead

# Proper set-based baselines for H2 validation
from stagebridge.baselines.set_baselines import (
    BaselineOutput,
    PoolingMLPBaseline,
    DeepSetsBaseline,
    SetTransformerBaseline,
    GraphSAGEBaseline,
    ReceiverCenteredBaseline,
    BASELINE_REGISTRY,
    create_baseline,
)
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
    # Set-based baselines (CORRECT - for H2 validation)
    "BaselineOutput",
    "PoolingMLPBaseline",
    "DeepSetsBaseline",
    "SetTransformerBaseline",
    "GraphSAGEBaseline",
    "ReceiverCenteredBaseline",
    "BASELINE_REGISTRY",
    "create_baseline",
    # Lesion-level baselines
    "PooledLesionBaseline",
    "DeepSetsLesionBaseline",
    "LesionSetTransformerBaseline",
    # Transition baselines
    "DeepSetsFlowModel",
    "NoContextFlowModel",
    "LinearTransitionBaseline",
]
