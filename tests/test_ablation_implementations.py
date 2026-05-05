"""Test that ablation implementations actually modify behavior."""

import numpy as np
import pytest
import torch

from stagebridge.loaders.dataset import StageBridgeDataset, collate_niche_batch
from stagebridge.evaluation.ablation import ABLATION_CONFIGS


class TestRandomNicheAblation:
    """Test that random_niche ablation shuffles cells across rings."""

    @pytest.fixture
    def mock_neighborhoods_df(self, tmp_path):
        """Create a minimal neighborhoods.parquet for testing."""
        import pandas as pd

        # Create predictable ring data where each ring has distinct values
        # Ring 1: values [1, 2], Ring 2: values [3, 4], etc.
        rows = []
        for i in range(10):
            row = {
                "cell_id": f"cell_{i}",
                "donor_id": "donor_0",
                "stage": "Normal",
                "receiver_z": np.ones(40) * i,
                "hlca_z": np.ones(30) * i,
                "luca_z": np.ones(10) * i,
                "ring_1_cells": [np.ones(40) * 1.0, np.ones(40) * 2.0],
                "ring_2_cells": [np.ones(40) * 3.0, np.ones(40) * 4.0],
                "ring_3_cells": [np.ones(40) * 5.0, np.ones(40) * 6.0],
                "ring_4_cells": [np.ones(40) * 7.0, np.ones(40) * 8.0],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = tmp_path / "neighborhoods.parquet"
        df.to_parquet(parquet_path)
        return tmp_path

    def test_shuffle_rings_changes_ring_assignment(self, mock_neighborhoods_df):
        """Verify that shuffle_rings=True changes which cells are in which ring."""
        # Without shuffle
        ds_normal = StageBridgeDataset(
            mock_neighborhoods_df,
            shuffle_rings=False,
        )

        # With shuffle
        ds_shuffled = StageBridgeDataset(
            mock_neighborhoods_df,
            shuffle_rings=True,
        )

        # Get same sample from both
        np.random.seed(42)
        sample_normal = ds_normal[0]

        np.random.seed(42)
        sample_shuffled = ds_shuffled[0]

        # In normal mode, ring 1 should have cells with values 1.0 and 2.0
        ring1_normal = sample_normal["ring_cells"][0]
        ring1_mask = sample_normal["ring_masks"][0]
        ring1_values_normal = ring1_normal[ring1_mask][:, 0]  # First dim of each cell

        # Check ring 1 has expected values (1.0 and 2.0)
        assert set(ring1_values_normal.tolist()) == {1.0, 2.0}, \
            f"Normal ring 1 should have values 1.0, 2.0 but got {ring1_values_normal}"

        # In shuffled mode, the same ring should have different values
        ring1_shuffled = sample_shuffled["ring_cells"][0]
        ring1_mask_shuffled = sample_shuffled["ring_masks"][0]
        ring1_values_shuffled = ring1_shuffled[ring1_mask_shuffled][:, 0]

        # The shuffled version should NOT have exactly {1.0, 2.0}
        # (it will have a mix from all rings)
        all_original_values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
        shuffled_values = set(ring1_values_shuffled.tolist())

        # Shuffled ring should contain values from the full pool
        assert shuffled_values.issubset(all_original_values), \
            f"Shuffled values should be from original pool"

        # And should NOT be exactly the same as original (with high probability)
        # Run multiple times to ensure shuffle is working
        different_count = 0
        for seed in range(10):
            np.random.seed(seed)
            sample = ds_shuffled[0]
            ring1 = sample["ring_cells"][0]
            mask = sample["ring_masks"][0]
            vals = set(ring1[mask][:, 0].tolist())
            if vals != {1.0, 2.0}:
                different_count += 1

        assert different_count > 0, "Shuffle should change ring contents at least sometimes"

    def test_ablation_config_has_shuffle_flag(self):
        """Verify random_niche ablation config has the shuffle flag."""
        config = ABLATION_CONFIGS["random_niche"]
        assert "_shuffle_rings" in config, "random_niche should have _shuffle_rings flag"
        assert config["_shuffle_rings"] is True, "_shuffle_rings should be True"


class TestNoTokenTypesAblationRemoved:
    """Verify no_token_types ablation was intentionally removed."""

    def test_no_token_types_not_in_configs(self):
        """no_token_types was removed - architecture uses structural separation instead."""
        assert "no_token_types" not in ABLATION_CONFIGS, \
            "no_token_types should be removed - current architecture uses structural separation"


class TestAblationConfigsValid:
    """Test that all ablation configs are properly defined."""

    def test_no_empty_configs_without_special_handling(self):
        """Verify no ablation has empty config without special handling."""
        special_ablations = {"random_niche", "frozen_encoder"}  # Have special handling

        for name, config in ABLATION_CONFIGS.items():
            if name in special_ablations:
                continue

            # Remove internal keys starting with _
            model_config = {k: v for k, v in config.items() if not k.startswith("_")}

            assert len(model_config) > 0 or name in special_ablations, \
                f"Ablation '{name}' has empty config - this means it runs as full model (no ablation)"

    def test_all_ablations_have_valid_keys(self):
        """Verify ablation config keys are valid StageBridgeConfig parameters."""
        from stagebridge.models import StageBridgeConfig
        import inspect

        valid_params = set(inspect.signature(StageBridgeConfig).parameters.keys())

        for name, config in ABLATION_CONFIGS.items():
            for key in config:
                if key.startswith("_"):  # Internal keys like _shuffle_rings
                    continue
                assert key in valid_params, \
                    f"Ablation '{name}' has invalid config key '{key}'. Valid keys: {valid_params}"
