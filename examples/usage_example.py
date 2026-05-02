#!/usr/bin/env python
"""Example script demonstrating StageBridge API usage.

This script shows the typical workflow for:
1. Loading a pretrained model
2. Preparing data from AnnData
3. Running inference
4. Visualizing results

Usage:
    python examples/usage_example.py --checkpoint runs/exp1/checkpoints/best.pt --data spatial.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Note: Install with: pip install scanpy anndata


def create_synthetic_data():
    """Create synthetic spatial AnnData for demonstration."""
    import numpy as np
    import pandas as pd
    import anndata as ad

    np.random.seed(42)
    n_cells = 1000

    # Generate synthetic embeddings
    X_scvi = np.random.randn(n_cells, 40).astype(np.float32)

    # Add HLCA and LuCA reference embeddings
    X_hlca = np.random.randn(n_cells, 30).astype(np.float32)
    X_luca = np.random.randn(n_cells, 10).astype(np.float32)

    # Spatial coordinates (Visium-like grid)
    coords = np.random.rand(n_cells, 2) * 1000  # 1mm x 1mm tissue

    # Stage labels (Normal -> Preinvasive -> Invasive gradient)
    x_normalized = coords[:, 0] / 1000
    stage_probs = np.column_stack([
        1 - x_normalized,  # Normal on left
        np.abs(x_normalized - 0.5),  # Preinvasive in middle
        x_normalized,  # Invasive on right
    ])
    stage_idx = np.argmax(stage_probs + np.random.randn(n_cells, 3) * 0.3, axis=1)
    stages = np.array(["Normal", "Preinvasive", "Invasive"])[stage_idx]

    # Create AnnData
    adata = ad.AnnData(
        X=np.random.randn(n_cells, 2000).astype(np.float32),  # Fake gene expression
        obs=pd.DataFrame({
            "stage": pd.Categorical(stages),
            "donor_id": np.random.choice(["D1", "D2", "D3"], n_cells),
            "cell_type": np.random.choice(["Epithelial", "Immune", "Stromal"], n_cells),
        }),
        obsm={
            "X_scvi": X_scvi,
            "X_scANVI_hlca": X_hlca,
            "X_scVI_luca": X_luca,
            "spatial": coords,
        },
    )

    print(f"Created synthetic AnnData: {adata.n_obs} cells")
    print(f"  Stages: {adata.obs['stage'].value_counts().to_dict()}")

    return adata


def example_prepare_neighborhoods(adata):
    """Example: Prepare neighborhoods from spatial data."""
    import stagebridge as sb

    print("\n--- Preparing Neighborhoods ---")

    # Compute receiver-centered neighborhoods
    sb.prepare_neighborhoods(
        adata,
        ring_radii=[50, 100, 150, 200],  # Distances in microns
        embedding_key="X_scvi",          # Cell embeddings
        hlca_key="X_scANVI_hlca",         # HLCA reference
        luca_key="X_scVI_luca",           # LuCA reference
        spatial_key="spatial",            # Spatial coordinates
    )

    # Check results
    print(f"Neighborhoods stored in adata.uns['X_neighborhoods']")
    print(f"  Shape: {adata.uns['X_neighborhoods'].shape}")

    return adata


def example_load_model(checkpoint_path: str | None = None):
    """Example: Load pretrained model."""
    import stagebridge as sb
    import torch

    print("\n--- Loading Model ---")

    if checkpoint_path and Path(checkpoint_path).exists():
        # Load from checkpoint
        model = sb.StageBridge.from_pretrained(checkpoint_path)
    else:
        # Create a model with default config for demonstration
        from stagebridge.models import StageBridge as StageBridgeModel, StageBridgeConfig

        config = StageBridgeConfig(
            hidden_dim=128,
            num_heads=4,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
            use_cross_attn_drift=True,
        )
        model = StageBridgeModel(config)

        # Wrap in API
        model = sb.StageBridge(model, config, device="cpu")
        print("Created model with default config (no pretrained weights)")

    print(f"Model device: {model.device}")
    print(f"Hidden dim: {model.config.hidden_dim}")

    return model


def example_get_embeddings(model, neighborhoods):
    """Example: Get niche embeddings."""
    print("\n--- Computing Niche Embeddings ---")

    # Get niche context embeddings
    embeddings = model.embed_niches(
        neighborhoods,
        batch_size=256,
        return_tokens=False,  # Set True to get individual token embeddings
    )

    print(f"Embeddings shape: {embeddings.embeddings.shape}")

    if embeddings.attention_weights is not None:
        print(f"Attention weights shape: {embeddings.attention_weights.shape}")

    return embeddings


def example_predict_transitions(model, neighborhoods):
    """Example: Predict cell state transitions."""
    print("\n--- Predicting Transitions ---")

    # Predict Normal -> Invasive transitions
    predictions = model.predict(
        neighborhoods=neighborhoods,
        source_stage="Normal",
        target_stage="Invasive",
        num_integration_steps=8,
        return_trajectories=False,  # Set True for full trajectories
    )

    print(f"Source embeddings: {predictions.source_embeddings.shape}")
    print(f"Predicted embeddings: {predictions.predicted_embeddings.shape}")
    print(f"Context embeddings: {predictions.context_embeddings.shape}")

    # Can also get as DataFrame
    df = predictions.to_dataframe()
    print(f"Predictions DataFrame: {df.shape}")

    return predictions


def example_compute_transitions(model, embeddings, context):
    """Example: Compute full transition trajectories."""
    print("\n--- Computing Full Trajectories ---")

    transitions = model.compute_transitions(
        embeddings=embeddings,
        context=context,
        source_stage="Preinvasive",
        target_stage="Invasive",
        num_steps=20,
    )

    print(f"Trajectories shape: {transitions.trajectories.shape}")  # [N, T+1, D]
    print(f"Velocities shape: {transitions.velocities.shape}")      # [N, T, D]
    print(f"Time points: {len(transitions.transition_times)}")

    return transitions


def example_visualizations(embeddings, predictions, stages):
    """Example: Create visualizations."""
    import stagebridge as sb

    print("\n--- Creating Visualizations ---")

    # 1. Plot embedding colored by stage
    print("Plotting embedding...")
    sb.pl.embedding(
        embeddings.embeddings,
        stages=stages,
        method="pca",  # Use PCA for speed (try "umap" or "phate" for better results)
        title="Niche Embeddings by Stage",
        show=False,
        save_path="figures/embedding_by_stage.png",
    )
    print("  Saved: figures/embedding_by_stage.png")

    # 2. Plot flow field
    print("Plotting flow field...")
    velocities = predictions.predicted_embeddings - predictions.source_embeddings
    sb.pl.flow_field(
        predictions.source_embeddings,
        velocities,
        stages=stages[:len(predictions.source_embeddings)],
        method="pca",
        title="Transition Velocity Field",
        show=False,
        save_path="figures/flow_field.png",
    )
    print("  Saved: figures/flow_field.png")

    # 3. Plot attention heatmap
    if embeddings.attention_weights is not None:
        print("Plotting attention...")
        sb.pl.niche_attention(
            embeddings.attention_weights[:50],  # First 50 cells
            title="Niche Token Attention",
            show=False,
            save_path="figures/attention_heatmap.png",
        )
        print("  Saved: figures/attention_heatmap.png")

    # 4. Plot stage centroids
    print("Plotting stage centroids...")
    sb.pl.stage_centroids(
        embeddings.embeddings,
        stages=stages,
        method="pca",
        stage_order=["Normal", "Preinvasive", "Invasive"],
        title="Stage Progression",
        show=False,
        save_path="figures/stage_centroids.png",
    )
    print("  Saved: figures/stage_centroids.png")


def example_create_dataset(adata):
    """Example: Create PyTorch dataset for training."""
    import stagebridge as sb

    print("\n--- Creating Dataset ---")

    # From AnnData
    dataset = sb.StageBridgeDataset.from_anndata(adata)
    print(f"Dataset size: {len(dataset)}")
    print(f"Stage distribution: {dataset.get_stage_distribution()}")
    print(f"Donor distribution: {dataset.get_donor_distribution()}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = sb.create_data_loaders(
        adata,
        batch_size=64,
        train_frac=0.8,
        val_frac=0.1,
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="StageBridge usage example")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to AnnData file (h5ad)")
    parser.add_argument("--create-figures", action="store_true",
                       help="Create and save figures")
    args = parser.parse_args()

    # Create figures directory
    Path("figures").mkdir(exist_ok=True)

    # Load or create data
    if args.data and Path(args.data).exists():
        import scanpy as sc
        print(f"Loading data from {args.data}")
        adata = sc.read_h5ad(args.data)
    else:
        print("Creating synthetic data for demonstration...")
        adata = create_synthetic_data()

    # Prepare neighborhoods
    adata = example_prepare_neighborhoods(adata)

    # Load model
    model = example_load_model(args.checkpoint)

    # Get neighborhoods DataFrame
    neighborhoods = adata.uns["X_neighborhoods"]
    stages = adata.obs["stage"].values

    # Get embeddings
    embeddings = example_get_embeddings(model, neighborhoods)

    # Predict transitions
    predictions = example_predict_transitions(model, neighborhoods)

    # Compute full trajectories (subset for speed)
    n_subset = min(100, len(embeddings.embeddings))
    transitions = example_compute_transitions(
        model,
        embeddings.embeddings[:n_subset],
        embeddings.embeddings[:n_subset],  # Use embeddings as context
    )

    # Create visualizations
    if args.create_figures:
        example_visualizations(embeddings, predictions, stages)

    # Create dataset for training
    train_loader, val_loader, test_loader = example_create_dataset(adata)

    print("\n--- Complete! ---")
    print("See the StageBridge documentation for more examples.")


if __name__ == "__main__":
    main()
