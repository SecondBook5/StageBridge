"""Baseline models for StageBridge.

The baseline ladder tests the core scientific claim:
"Cross-sectional progression becomes more identifiable when conditioned
on receiver-centered local niche context"

Ladder (simplest to most complex):
1. PoolingMLP - No structure (bag-of-cells)
2. DeepSets - Permutation invariance only
3. SetTransformer - Flat attention (no spatial structure)
4. GraphSAGE - Spatial graph structure (symmetric aggregation)
5. StageBridge - Receiver-centered niche + dual references (full model)

Each baseline uses the same input format as StageBridge:
    forward(receiver, neighbors, distances, x_t, t, stage_pair_id, neighbor_mask)

This allows fair comparison with identical data loading and training loops.
"""

from stagebridge.baselines.pooling import (
    PoolingMLP,
    MaxPoolMLP,
    PoolingBaselineConfig,
)
from stagebridge.baselines.deepsets import DeepSets, DeepSetsConfig
from stagebridge.baselines.set_transformer import SetTransformer, SetTransformerConfig
from stagebridge.baselines.graph_sage import GraphSAGE, GraphSAGEConfig

__all__ = [
    # Pooling baselines
    "PoolingMLP",
    "MaxPoolMLP",
    "PoolingBaselineConfig",
    # DeepSets
    "DeepSets",
    "DeepSetsConfig",
    # Set Transformer
    "SetTransformer",
    "SetTransformerConfig",
    # GraphSAGE
    "GraphSAGE",
    "GraphSAGEConfig",
]


def get_baseline(name: str, **kwargs):
    """Get a baseline model by name.

    Args:
        name: One of "pooling", "maxpool", "deepsets", "set_transformer", "graphsage"
        **kwargs: Config overrides

    Returns:
        Instantiated baseline model
    """
    baselines = {
        "pooling": (PoolingMLP, PoolingBaselineConfig),
        "maxpool": (MaxPoolMLP, PoolingBaselineConfig),
        "deepsets": (DeepSets, DeepSetsConfig),
        "set_transformer": (SetTransformer, SetTransformerConfig),
        "graphsage": (GraphSAGE, GraphSAGEConfig),
    }

    if name not in baselines:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(baselines.keys())}")

    model_cls, config_cls = baselines[name]
    config = config_cls(**kwargs)
    return model_cls(config)
