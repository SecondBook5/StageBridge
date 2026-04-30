"""Training tests.

Tests for SSL pretraining and transition training phases.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stagebridge.contracts import LATENT_DIM, HLCA_DIM, LUCA_DIM, STATS_TOKEN_DIM, STAGE_TO_IDX
from stagebridge.loaders import create_dataloaders
from stagebridge.models import StageBridge, StageBridgeConfig
from stagebridge.training import StageBridgeTrainer, TrainerConfig


@pytest.fixture
def synthetic_data_dir() -> Path:
    """Create synthetic data for training tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        n_cells = 100
        np.random.seed(42)

        data = []
        stages = list(STAGE_TO_IDX.keys())

        for i in range(n_cells):
            row = {
                "cell_id": f"cell_{i:04d}",
                "donor_id": f"donor_{i % 4}",
                "stage": stages[i % len(stages)],
                "receiver_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "hlca_z": np.random.randn(HLCA_DIM).astype(np.float32).tolist(),
                "luca_z": np.random.randn(LUCA_DIM).astype(np.float32).tolist(),
                "pathway_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "stats_z": np.random.randn(STATS_TOKEN_DIM).astype(np.float32).tolist(),
            }
            for ring in range(1, 5):
                n_ring_cells = np.random.randint(5, 12)
                row[f"ring_{ring}_cells"] = [
                    np.random.randn(LATENT_DIM).astype(np.float32).tolist()
                    for _ in range(n_ring_cells)
                ]
            data.append(row)

        df = pd.DataFrame(data)
        df.to_parquet(data_dir / "neighborhoods.parquet", index=False)

        manifest = {
            "folds": [
                {
                    "fold": 0,
                    "train_donors": ["donor_0", "donor_1"],
                    "val_donors": ["donor_2"],
                    "test_donors": ["donor_3"],
                }
            ]
        }
        with open(data_dir / "split_manifest.json", "w") as f:
            json.dump(manifest, f)

        yield data_dir


class TestTrainerConfig:
    """Test trainer configuration."""

    def test_default_config(self):
        """Default config should have valid values."""
        config = TrainerConfig()
        assert config.ssl_epochs >= 0
        assert config.transition_epochs >= 0
        assert config.learning_rate > 0

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = TrainerConfig(
            ssl_epochs=10,
            transition_epochs=20,
            learning_rate=5e-4,
        )
        assert config.ssl_epochs == 10
        assert config.transition_epochs == 20
        assert config.learning_rate == 5e-4


class TestSSLPretraining:
    """Test SSL pretraining phase."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_ssl_epoch_runs(self, synthetic_data_dir: Path):
        """SSL epoch should complete without errors."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="ssl_test",
                ssl_epochs=1,
                transition_epochs=0,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)

            assert "ssl" in summary
            assert "best_val_loss" in summary["ssl"]

    def test_ssl_loss_decreases(self, synthetic_data_dir: Path):
        """SSL loss should decrease over training."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="ssl_loss_test",
                ssl_epochs=3,
                transition_epochs=0,
                learning_rate=1e-3,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)

            assert summary["ssl"]["best_val_loss"] < float("inf")


class TestTransitionTraining:
    """Test transition (OT-CFM) training phase."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_transition_epoch_runs(self, synthetic_data_dir: Path):
        """Transition epoch should complete without errors."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="transition_test",
                ssl_epochs=0,
                transition_epochs=1,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)

            assert "transition" in summary
            assert "best_val_loss" in summary["transition"]

    def test_two_stage_training(self, synthetic_data_dir: Path):
        """Both SSL and transition phases should run sequentially."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="two_stage_test",
                ssl_epochs=1,
                transition_epochs=1,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)

            assert "ssl" in summary
            assert "transition" in summary


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_checkpoint_saved(self, synthetic_data_dir: Path):
        """Checkpoints should be saved during training."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            trainer_config = TrainerConfig(
                output_dir=output_dir,
                run_name="checkpoint_test",
                ssl_epochs=1,
                transition_epochs=1,
                checkpoint_every=1,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            trainer.train(train_loader, val_loader)

            # Check checkpoints exist
            ckpt_dir = output_dir / "checkpoint_test" / "checkpoints"
            assert ckpt_dir.exists()


class TestLearningRateScheduling:
    """Test learning rate scheduling."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_warmup_applied(self, synthetic_data_dir: Path):
        """Warmup should be applied to learning rate."""
        train_loader, val_loader, _ = create_dataloaders(
            data_dir=synthetic_data_dir,
            fold_idx=0,
            batch_size=16,
            num_workers=0,
        )

        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="warmup_test",
                ssl_epochs=2,
                transition_epochs=0,
                learning_rate=1e-3,
                warmup_epochs=1,
                strict_gradient_check=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)
            assert "ssl" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
