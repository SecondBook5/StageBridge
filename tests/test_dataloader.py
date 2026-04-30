"""Data loading tests.

Tests for dataset, dataloader, and batch creation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stagebridge.contracts import LATENT_DIM, STAGE_TO_IDX
from stagebridge.loaders import create_dataloaders
from stagebridge.loaders.dataset import NicheBatch


class TestNicheBatch:
    """Test NicheBatch dataclass."""

    @pytest.fixture
    def sample_batch(self) -> NicheBatch:
        batch_size = 4
        max_cells = 10
        return NicheBatch(
            receiver=torch.randn(batch_size, LATENT_DIM),
            ring_cells=[torch.randn(batch_size, max_cells, LATENT_DIM) for _ in range(4)],
            ring_masks=[torch.ones(batch_size, max_cells, dtype=torch.bool) for _ in range(4)],
            hlca=torch.randn(batch_size, LATENT_DIM),
            luca=torch.randn(batch_size, LATENT_DIM),
            pathway=torch.randn(batch_size, LATENT_DIM),
            stats=torch.randn(batch_size, LATENT_DIM),
            stage_idx=torch.zeros(batch_size, dtype=torch.long),
            cell_ids=["cell_0", "cell_1", "cell_2", "cell_3"],
            donor_ids=["donor_0", "donor_0", "donor_1", "donor_1"],
        )

    def test_batch_to_device(self, sample_batch: NicheBatch):
        """Batch should move to device correctly."""
        batch = sample_batch.to("cpu")
        assert batch.receiver.device.type == "cpu"
        for ring in batch.ring_cells:
            assert ring.device.type == "cpu"

    def test_batch_length(self, sample_batch: NicheBatch):
        """Batch length should match receiver size."""
        assert len(sample_batch) == 4

    def test_batch_has_all_fields(self, sample_batch: NicheBatch):
        """Batch should have all required fields."""
        assert sample_batch.receiver is not None
        assert len(sample_batch.ring_cells) == 4
        assert len(sample_batch.ring_masks) == 4
        assert sample_batch.hlca is not None
        assert sample_batch.luca is not None


class TestCreateDataloaders:
    """Test dataloader creation."""

    @pytest.fixture
    def temp_data_dir(self) -> Path:
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            n_cells = 100
            np.random.seed(42)

            data = []
            for i in range(n_cells):
                row = {
                    "cell_id": f"cell_{i:04d}",
                    "donor_id": f"donor_{i % 4}",
                    "stage": list(STAGE_TO_IDX.keys())[i % 3],
                    "receiver_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                    "hlca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                    "luca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                    "pathway_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                    "stats_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                }
                for ring in range(1, 5):
                    n_ring_cells = np.random.randint(3, 15)
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

    def test_create_dataloaders(self, temp_data_dir: Path):
        """create_dataloaders should return train/val/test loaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=temp_data_dir,
            fold_idx=0,
            batch_size=8,
            num_workers=0,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_dataloader_yields_batches(self, temp_data_dir: Path):
        """Dataloaders should yield NicheBatch objects."""
        train_loader, _, _ = create_dataloaders(
            data_dir=temp_data_dir,
            fold_idx=0,
            batch_size=8,
            num_workers=0,
        )

        batch = next(iter(train_loader))
        assert isinstance(batch, NicheBatch)
        assert batch.receiver.shape[1] == LATENT_DIM

    def test_batch_collation(self, temp_data_dir: Path):
        """Batches should be properly collated with padding."""
        train_loader, _, _ = create_dataloaders(
            data_dir=temp_data_dir,
            fold_idx=0,
            batch_size=8,
            num_workers=0,
        )

        batch = next(iter(train_loader))

        # All ring_cells should have same max_cells dimension within batch
        for i, ring_cells in enumerate(batch.ring_cells):
            assert ring_cells.ndim == 3  # (batch, max_cells, latent_dim)
            assert ring_cells.shape[2] == LATENT_DIM

        # Masks should match ring_cells shape
        for ring_cells, ring_mask in zip(batch.ring_cells, batch.ring_masks):
            assert ring_mask.shape == ring_cells.shape[:2]


class TestDataIntegrity:
    """Test data integrity and edge cases."""

    @pytest.fixture
    def temp_data_dir(self) -> Path:
        """Create data with edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            np.random.seed(42)
            data = []

            # Normal cell
            data.append({
                "cell_id": "normal",
                "donor_id": "donor_0",
                "stage": "Normal",
                "receiver_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "hlca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "luca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "pathway_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "stats_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "ring_1_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist() for _ in range(5)],
                "ring_2_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist() for _ in range(8)],
                "ring_3_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist() for _ in range(3)],
                "ring_4_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist() for _ in range(10)],
            })

            # Cell with single neighbor per ring
            data.append({
                "cell_id": "sparse",
                "donor_id": "donor_0",
                "stage": "Preinvasive",
                "receiver_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "hlca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "luca_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "pathway_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "stats_z": np.random.randn(LATENT_DIM).astype(np.float32).tolist(),
                "ring_1_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist()],
                "ring_2_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist()],
                "ring_3_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist()],
                "ring_4_cells": [np.random.randn(LATENT_DIM).astype(np.float32).tolist()],
            })

            df = pd.DataFrame(data)
            df.to_parquet(data_dir / "neighborhoods.parquet", index=False)

            manifest = {
                "folds": [{"fold": 0, "train_donors": ["donor_0"], "val_donors": ["donor_0"], "test_donors": ["donor_0"]}]
            }
            with open(data_dir / "split_manifest.json", "w") as f:
                json.dump(manifest, f)

            yield data_dir

    def test_sparse_neighborhoods(self, temp_data_dir: Path):
        """Dataset should handle sparse neighborhoods (few neighbors)."""
        train_loader, _, _ = create_dataloaders(
            data_dir=temp_data_dir,
            fold_idx=0,
            batch_size=2,
            num_workers=0,
        )
        batch = next(iter(train_loader))
        # Should not crash with sparse data
        assert batch.receiver.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
