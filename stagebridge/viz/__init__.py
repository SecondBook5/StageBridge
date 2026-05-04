"""Visualization module for StageBridge publication figures.

Provides publication-quality plotting functions following scanpy conventions.

Example usage:
    import stagebridge as sb

    # Plot embedding colored by stage
    sb.pl.embedding(embeddings, stages=adata.obs["stage"])

    # Plot velocity field
    sb.pl.flow_field(embeddings, velocities)

    # Plot attention heatmap
    sb.pl.niche_attention(attention_weights)

    # LIANA cell-cell communication with Ricci curvature
    from stagebridge.viz import liana
    df = liana.load_liana_data("interactions.parquet")
    liana.plot_ricci_network(df)
"""

from .figures import (
    load_data,
    compute_embedding,
    compute_ot_flow_field,
    compute_flux_decomposition,
)

# Import plotting API
from .plotting import (
    embedding,
    flow_field,
    niche_attention,
    trajectory,
    stage_centroids,
    STAGE_COLORS,
    PlottingNamespace,
)

# LIANA visualization submodule
from . import liana

# Scanpy-style namespace (sb.pl.embedding, etc.)
pl = PlottingNamespace()

__all__ = [
    # Figures module
    "load_data",
    "compute_embedding",
    "compute_ot_flow_field",
    "compute_flux_decomposition",
    # Plotting API
    "embedding",
    "flow_field",
    "niche_attention",
    "trajectory",
    "stage_centroids",
    "STAGE_COLORS",
    "pl",
    # LIANA submodule
    "liana",
]
