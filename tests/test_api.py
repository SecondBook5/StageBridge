"""Tests for StageBridge publication API."""

import numpy as np
import pandas as pd
import pytest
import torch

import stagebridge as sb
from stagebridge.contracts import LATENT_DIM, HLCA_DIM, LUCA_DIM


class TestAPIImports:
    """Test that all API components are importable."""

    def test_version(self):
        assert sb.__version__ == "1.0.0"

    def test_stagebridge_class(self):
        assert hasattr(sb, "StageBridge")
        assert hasattr(sb.StageBridge, "from_pretrained")
        assert hasattr(sb.StageBridge, "predict")
        assert hasattr(sb.StageBridge, "embed_niches")
        assert hasattr(sb.StageBridge, "compute_transitions")

    def test_output_types(self):
        assert hasattr(sb, "PredictionOutput")
        assert hasattr(sb, "NicheEmbeddingOutput")
        assert hasattr(sb, "TransitionOutput")

    def test_data_functions(self):
        assert hasattr(sb, "prepare_neighborhoods")
        assert hasattr(sb, "prepare_neighborhoods_from_graph")
        assert hasattr(sb, "StageBridgeDataset")
        assert hasattr(sb, "create_data_loaders")

    def test_plotting_namespace(self):
        assert hasattr(sb, "pl")
        assert hasattr(sb.pl, "embedding")
        assert hasattr(sb.pl, "flow_field")
        assert hasattr(sb.pl, "niche_attention")
        assert hasattr(sb.pl, "trajectory")
        assert hasattr(sb.pl, "stage_centroids")

    def test_constants(self):
        assert sb.STAGES == ("Normal", "Preinvasive", "Invasive")
        assert sb.LATENT_DIM == 40
        assert sb.HLCA_DIM == 30
        assert sb.LUCA_DIM == 10


class TestStageBridgeAPI:
    """Test StageBridge API functionality."""

    @pytest.fixture
    def model_and_config(self):
        """Create a model for testing."""
        from stagebridge.models import StageBridge, StageBridgeConfig

        config = StageBridgeConfig(
            hidden_dim=64,
            num_heads=2,
            use_learned_ring_pooling=True,
            use_context_refiner=True,
            use_cross_attn_drift=True,
        )
        model = StageBridge(config)
        return model, config

    @pytest.fixture
    def neighborhoods_df(self):
        """Create test neighborhoods DataFrame."""
        n = 50
        return pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(n)],
            "donor_id": ["D1"] * (n // 2) + ["D2"] * (n // 2),
            "stage": ["Normal"] * (n // 3) + ["Preinvasive"] * (n // 3) + ["Invasive"] * (n - 2 * (n // 3)),
            "receiver_z": [np.random.randn(LATENT_DIM).tolist() for _ in range(n)],
            "ring_1_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(3)] for _ in range(n)],
            "ring_2_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(2)] for _ in range(n)],
            "ring_3_cells": [[np.random.randn(LATENT_DIM).tolist() for _ in range(1)] for _ in range(n)],
            "ring_4_cells": [[] for _ in range(n)],
            "hlca_z": [np.random.randn(HLCA_DIM).tolist() for _ in range(n)],
            "luca_z": [np.random.randn(LUCA_DIM).tolist() for _ in range(n)],
        })

    def test_api_wrapper_creation(self, model_and_config):
        """Test creating API wrapper."""
        model, config = model_and_config
        api = sb.StageBridge(model, config, device="cpu")

        assert api.config == config
        assert api.device == torch.device("cpu")
        assert api.num_stages == config.num_stages
        assert api.latent_dim == LATENT_DIM

    def test_embed_niches(self, model_and_config, neighborhoods_df):
        """Test niche embedding."""
        model, config = model_and_config
        api = sb.StageBridge(model, config, device="cpu")

        output = api.embed_niches(neighborhoods_df, batch_size=16)

        assert isinstance(output, sb.NicheEmbeddingOutput)
        assert output.embeddings.shape == (len(neighborhoods_df), config.hidden_dim)

    def test_predict(self, model_and_config, neighborhoods_df):
        """Test prediction."""
        model, config = model_and_config
        api = sb.StageBridge(model, config, device="cpu")

        output = api.predict(
            neighborhoods=neighborhoods_df,
            source_stage="Normal",
            target_stage="Invasive",
            batch_size=16,
        )

        assert isinstance(output, sb.PredictionOutput)
        assert output.predicted_embeddings.shape[0] == len(neighborhoods_df)
        assert output.source_embeddings.shape[0] == len(neighborhoods_df)
        assert output.context_embeddings.shape[0] == len(neighborhoods_df)
        assert output.source_stage == "Normal"
        assert output.target_stage == "Invasive"

    def test_predict_to_dataframe(self, model_and_config, neighborhoods_df):
        """Test prediction DataFrame conversion."""
        model, config = model_and_config
        api = sb.StageBridge(model, config, device="cpu")

        output = api.predict(
            neighborhoods=neighborhoods_df,
            source_stage="Normal",
            target_stage="Invasive",
            batch_size=16,
        )

        df = output.to_dataframe()
        assert len(df) == len(neighborhoods_df)
        assert "cell_id" in df.columns
        assert "predicted_embedding" in df.columns
        assert "source_stage" in df.columns

    def test_compute_transitions(self, model_and_config):
        """Test transition computation."""
        model, config = model_and_config
        api = sb.StageBridge(model, config, device="cpu")

        n = 20
        embeddings = np.random.randn(n, LATENT_DIM).astype(np.float32)
        context = np.random.randn(n, config.hidden_dim).astype(np.float32)

        output = api.compute_transitions(
            embeddings=embeddings,
            context=context,
            source_stage="Normal",
            target_stage="Invasive",
            num_steps=10,
        )

        assert isinstance(output, sb.TransitionOutput)
        assert output.trajectories.shape == (n, 11, LATENT_DIM)  # T+1 steps
        assert output.velocities.shape == (n, 10, LATENT_DIM)  # T steps
        assert len(output.transition_times) == 11


class TestDataAPI:
    """Test data loading API."""

    @pytest.fixture
    def synthetic_adata(self):
        """Create synthetic AnnData for testing."""
        import anndata as ad

        n = 100
        np.random.seed(42)

        adata = ad.AnnData(
            X=np.random.randn(n, 100).astype(np.float32),
            obs=pd.DataFrame({
                "stage": pd.Categorical(["Normal"] * 40 + ["Preinvasive"] * 30 + ["Invasive"] * 30),
                "donor_id": ["D1"] * 50 + ["D2"] * 50,
            }),
            obsm={
                "X_scvi": np.random.randn(n, LATENT_DIM).astype(np.float32),
                "X_scANVI_hlca": np.random.randn(n, HLCA_DIM).astype(np.float32),
                "X_scVI_luca": np.random.randn(n, LUCA_DIM).astype(np.float32),
                "spatial": np.random.rand(n, 2) * 1000,
            },
        )
        return adata

    def test_prepare_neighborhoods(self, synthetic_adata):
        """Test neighborhood preparation."""
        sb.prepare_neighborhoods(
            synthetic_adata,
            ring_radii=[100, 200, 300, 400],
        )

        assert "X_neighborhoods" in synthetic_adata.uns
        neighborhoods = synthetic_adata.uns["X_neighborhoods"]
        assert len(neighborhoods) == synthetic_adata.n_obs
        assert "receiver_z" in neighborhoods.columns
        assert "ring_1_cells" in neighborhoods.columns

    def test_dataset_from_anndata(self, synthetic_adata):
        """Test creating dataset from AnnData."""
        sb.prepare_neighborhoods(synthetic_adata)

        dataset = sb.StageBridgeDataset.from_anndata(synthetic_adata)
        assert len(dataset) == synthetic_adata.n_obs

        # Test __getitem__
        item = dataset[0]
        assert "receiver" in item
        assert "ring_cells" in item
        assert "hlca" in item

    def test_dataset_stage_filter(self, synthetic_adata):
        """Test dataset stage filtering."""
        sb.prepare_neighborhoods(synthetic_adata)

        dataset = sb.StageBridgeDataset.from_anndata(
            synthetic_adata,
            stages=["Normal", "Preinvasive"],
        )

        # Should exclude Invasive cells
        assert len(dataset) < synthetic_adata.n_obs
        assert "Invasive" not in dataset.get_stage_distribution()


class TestPlottingAPI:
    """Test plotting API (just that functions are callable, not output)."""

    def test_embedding_callable(self):
        """Test embedding plot is callable."""
        embeddings = np.random.randn(100, 10)
        stages = ["Normal"] * 50 + ["Invasive"] * 50

        # Should not raise (but don't show)
        fig = sb.pl.embedding(
            embeddings,
            stages=stages,
            method="pca",
            show=False,
        )
        assert fig is not None

    def test_niche_attention_callable(self):
        """Test niche attention plot is callable."""
        attention = np.random.rand(50, 8)

        fig = sb.pl.niche_attention(attention, show=False)
        assert fig is not None
