"""Integration tests.

End-to-end tests for the full StageBridge pipeline.
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
    """Create synthetic data for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        n_cells = 200
        n_donors = 4
        np.random.seed(42)

        data = []
        stages = list(STAGE_TO_IDX.keys())
        donors = [f"donor_{i}" for i in range(n_donors)]

        for i in range(n_cells):
            row = {
                "cell_id": f"cell_{i:04d}",
                "donor_id": donors[i % n_donors],
                "stage": stages[i % len(stages)],
                "receiver_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "hlca_z": np.random.randn(HLCA_DIM).astype(np.float32).tolist(),
                "luca_z": np.random.randn(LUCA_DIM).astype(np.float32).tolist(),
                "pathway_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "stats_z": np.random.randn(STATS_TOKEN_DIM).astype(np.float32).tolist(),
            }
            for ring in range(1, 5):
                n_ring_cells = np.random.randint(5, 15)
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


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_full_training_loop(self, synthetic_data_dir: Path):
        """Full training loop should complete without errors."""
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
            use_learned_ring_pooling=True,
            use_context_refiner=True,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="integration_test",
                ssl_epochs=1,
                transition_epochs=1,
                learning_rate=1e-3,
                checkpoint_every=10,
                eval_every=1,
                strict_gradient_check=False,
                early_stopping_enabled=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            summary = trainer.train(train_loader, val_loader)

            assert "ssl" in summary
            assert "transition" in summary

    def test_inference_after_training(self, synthetic_data_dir: Path):
        """Model should produce valid outputs after training."""
        train_loader, val_loader, test_loader = create_dataloaders(
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
            use_learned_ring_pooling=True,
            use_context_refiner=True,
        )
        model = StageBridge(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_config = TrainerConfig(
                output_dir=Path(tmpdir),
                run_name="inference_test",
                ssl_epochs=1,
                transition_epochs=1,
                learning_rate=1e-3,
                strict_gradient_check=False,
                early_stopping_enabled=False,
            )
            trainer = StageBridgeTrainer(model=model, config=trainer_config, device="cpu")
            trainer.train(train_loader, val_loader)

            # Run inference
            model.eval()
            with torch.no_grad():
                batch = next(iter(test_loader))
                niche_output = model.encode_niche(
                    receiver=batch.receiver,
                    ring_cells=batch.ring_cells,
                    ring_masks=batch.ring_masks,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                )

                stage_pair_id = model.encode_stage_pair_tensor(0, 1, len(batch.receiver), "cpu")
                predicted = model.integrate_euler(
                    x0=batch.receiver,
                    context=niche_output.context,
                    stage_pair_id=stage_pair_id,
                    num_steps=5,
                    context_tokens=niche_output.context_tokens,
                )

            assert predicted.shape == batch.receiver.shape
            assert not torch.isnan(predicted).any()

    def test_checkpoint_save_load(self, synthetic_data_dir: Path):
        """Model should be saveable and loadable."""
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
            # Save
            ckpt_path = Path(tmpdir) / "model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
            }, ckpt_path)

            # Load
            loaded = torch.load(ckpt_path, weights_only=False)
            new_model = StageBridge(loaded["config"])
            new_model.load_state_dict(loaded["model_state_dict"])

            # Verify same outputs
            model.eval()
            new_model.eval()
            batch = next(iter(train_loader))

            with torch.no_grad():
                out1 = model.encode_niche(
                    receiver=batch.receiver,
                    ring_cells=batch.ring_cells,
                    ring_masks=batch.ring_masks,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                )
                out2 = new_model.encode_niche(
                    receiver=batch.receiver,
                    ring_cells=batch.ring_cells,
                    ring_masks=batch.ring_masks,
                    hlca=batch.hlca,
                    luca=batch.luca,
                    pathway=batch.pathway,
                    stats=batch.stats,
                )

            assert torch.allclose(out1.context, out2.context, atol=1e-5)


class TestBaselineComparison:
    """Test baseline models work with same data."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_all_baselines_forward(self):
        """All baselines should produce valid outputs."""
        from stagebridge.baselines import get_baseline

        batch_size = 4
        batch = {
            "receiver": torch.randn(batch_size, LATENT_DIM),
            "neighbors": torch.randn(batch_size, 8, LATENT_DIM),
            "distances": torch.rand(batch_size, 8) * 50,
            "x_t": torch.randn(batch_size, LATENT_DIM),
            "t": torch.rand(batch_size),
            "stage_pair_id": torch.zeros(batch_size, dtype=torch.long),
            "neighbor_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        }

        for name in ["pooling", "deepsets", "set_transformer", "graphsage"]:
            model = get_baseline(name, input_dim=LATENT_DIM, hidden_dim=64)
            output = model(**batch)
            assert output.shape == (batch_size, LATENT_DIM)
            assert not torch.isnan(output).any(), f"{name} produced NaN"


class TestDeviceCompatibility:
    """Test model works on different devices."""

    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_cpu_forward(self):
        """Model should work on CPU."""
        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config).to("cpu")

        batch_size = 4
        output = model.encode_niche(
            receiver=torch.randn(batch_size, LATENT_DIM),
            ring_cells=[torch.randn(batch_size, 10, LATENT_DIM) for _ in range(4)],
            ring_masks=[torch.ones(batch_size, 10, dtype=torch.bool) for _ in range(4)],
            hlca=torch.randn(batch_size, HLCA_DIM),
            luca=torch.randn(batch_size, LUCA_DIM),
            pathway=torch.randn(batch_size, LATENT_DIM),
            stats=torch.randn(batch_size, STATS_TOKEN_DIM),
        )
        assert output.context.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Model should work on CUDA."""
        config = StageBridgeConfig(
            input_dim=LATENT_DIM,
            hidden_dim=64,
            num_heads=2,
            num_encoder_layers=1,
        )
        model = StageBridge(config).to("cuda")

        batch_size = 4
        output = model.encode_niche(
            receiver=torch.randn(batch_size, LATENT_DIM, device="cuda"),
            ring_cells=[torch.randn(batch_size, 10, LATENT_DIM, device="cuda") for _ in range(4)],
            ring_masks=[torch.ones(batch_size, 10, dtype=torch.bool, device="cuda") for _ in range(4)],
            hlca=torch.randn(batch_size, HLCA_DIM, device="cuda"),
            luca=torch.randn(batch_size, LUCA_DIM, device="cuda"),
            pathway=torch.randn(batch_size, LATENT_DIM, device="cuda"),
            stats=torch.randn(batch_size, STATS_TOKEN_DIM, device="cuda"),
        )
        assert output.context.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
