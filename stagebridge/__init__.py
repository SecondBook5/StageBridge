"""StageBridge: Receiver-centered niche modeling for cancer progression.

Publication API
---------------
This package provides a clean interface for using StageBridge, designed
to be intuitive for researchers familiar with scanpy/scvi-tools.

Quick Start:
    import stagebridge as sb

    # Load pretrained model
    model = sb.StageBridge.from_pretrained("checkpoint.pt")

    # Prepare data from AnnData
    sb.prepare_neighborhoods(adata, ring_radii=[50, 100, 150, 200])

    # Run inference
    predictions = model.predict(adata, source_stage="Normal", target_stage="Invasive")

    # Get niche embeddings
    embeddings = model.embed_niches(neighborhoods)

    # Visualize
    sb.pl.embedding(predictions.context_embeddings, stages=adata.obs["stage"])
    sb.pl.flow_field(predictions.source_embeddings, predictions.predicted_embeddings - predictions.source_embeddings)

See documentation for more details.
"""

from stagebridge.contracts import (
    STAGES_3,
    STAGES_4,
    STAGES_5,
    STAGES,
    N_STAGES,
    LATENT_DIM,
    HLCA_DIM,
    LUCA_DIM,
    N_TOKENS,
    TOKEN_NAMES,
    WES_COLS,
    WES_DIM,
)
from stagebridge.models import StageBridgeConfig, StageBridgeOutput

# High-level API (recommended for most users)
from stagebridge.api import (
    StageBridgeAPI,
    PredictionOutput,
    NicheEmbeddingOutput,
    TransitionOutput,
)

# Data loading
from stagebridge.data import (
    prepare_neighborhoods,
    prepare_neighborhoods_from_graph,
    StageBridgeDataset,
    create_data_loaders,
)

# Visualization (scanpy-style namespace: sb.pl.*)
from stagebridge.viz import pl

__version__ = "1.0.0"


class StageBridge(StageBridgeAPI):
    """StageBridge model with high-level API.

    This is the recommended interface for using StageBridge.
    Provides methods like from_pretrained(), predict(), embed_niches().

    For low-level access to the PyTorch model, use stagebridge.models.StageBridge.

    Example:
        import stagebridge as sb

        # Load pretrained model
        model = sb.StageBridge.from_pretrained("checkpoint.pt")

        # Prepare neighborhoods
        sb.prepare_neighborhoods(adata)

        # Run inference
        predictions = model.predict(adata, source_stage="Normal", target_stage="Invasive")

        # Get niche embeddings
        embeddings = model.embed_niches(neighborhoods)

        # Plot results
        sb.pl.embedding(embeddings.embeddings, stages=adata.obs["stage"])
    """

    pass  # Inherits everything from StageBridgeAPI


__all__ = [
    # High-level API (recommended)
    "StageBridge",
    "StageBridgeConfig",
    "StageBridgeOutput",
    "PredictionOutput",
    "NicheEmbeddingOutput",
    "TransitionOutput",
    # Data loading
    "prepare_neighborhoods",
    "prepare_neighborhoods_from_graph",
    "StageBridgeDataset",
    "create_data_loaders",
    # Visualization namespace
    "pl",
    # Constants
    "STAGES_3",
    "STAGES_4",
    "STAGES_5",
    "STAGES",
    "N_STAGES",
    "LATENT_DIM",
    "HLCA_DIM",
    "LUCA_DIM",
    "N_TOKENS",
    "TOKEN_NAMES",
    "WES_COLS",
    "WES_DIM",
]
