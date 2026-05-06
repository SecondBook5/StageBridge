"""Test that ablation implementations actually modify behavior."""

import numpy as np
import pytest
import torch

from stagebridge.loaders.dataset import StageBridgeDataset, collate_niche_batch
from stagebridge.evaluation.ablation import ABLATION_CONFIGS, is_ablation_redundant


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


class TestAblationRedundancyDetection:
    """Test that redundant ablations are correctly detected.

    This prevents wasting HPC compute on ablations that would be identical
    to the main model given HPO choices.
    """

    # =========================================================================
    # GW Fusion Redundancy
    # =========================================================================

    def test_no_gw_fusion_redundant_when_hpo_chose_concat(self):
        """no_gw_fusion ablation is redundant if HPO already chose concat."""
        hpo_params = {"use_gw_fusion": False}
        is_redundant, reason = is_ablation_redundant("no_gw_fusion", hpo_params)
        assert is_redundant, "no_gw_fusion should be redundant when HPO chose use_gw_fusion=False"
        assert "use_gw_fusion=False" in reason

    def test_no_gw_fusion_not_redundant_when_hpo_chose_pretrained(self):
        """no_gw_fusion ablation is valid if HPO chose pretrained GW."""
        hpo_params = {"use_gw_fusion": True, "gw_fusion_type": "pretrained"}
        is_redundant, _ = is_ablation_redundant("no_gw_fusion", hpo_params)
        assert not is_redundant, "no_gw_fusion should NOT be redundant when HPO chose pretrained"

    def test_no_gw_fusion_not_redundant_when_hpo_chose_learned(self):
        """no_gw_fusion ablation is valid if HPO chose learned projection."""
        hpo_params = {"use_gw_fusion": True, "gw_fusion_type": "learned_projection"}
        is_redundant, _ = is_ablation_redundant("no_gw_fusion", hpo_params)
        assert not is_redundant, "no_gw_fusion should NOT be redundant when HPO chose learned_projection"

    def test_gw_learned_redundant_when_hpo_chose_learned(self):
        """gw_learned ablation is redundant if HPO already chose learned_projection."""
        hpo_params = {"use_gw_fusion": True, "gw_fusion_type": "learned_projection"}
        is_redundant, reason = is_ablation_redundant("gw_learned", hpo_params)
        assert is_redundant, "gw_learned should be redundant when HPO chose learned_projection"
        assert "learned_projection" in reason

    def test_gw_learned_not_redundant_when_hpo_chose_pretrained(self):
        """gw_learned ablation is valid if HPO chose pretrained GW."""
        hpo_params = {"use_gw_fusion": True, "gw_fusion_type": "pretrained"}
        is_redundant, _ = is_ablation_redundant("gw_learned", hpo_params)
        assert not is_redundant, "gw_learned should NOT be redundant when HPO chose pretrained"

    def test_gw_learned_not_redundant_when_hpo_chose_concat(self):
        """gw_learned ablation is valid if HPO chose no GW (tests if GW helps)."""
        hpo_params = {"use_gw_fusion": False}
        is_redundant, _ = is_ablation_redundant("gw_learned", hpo_params)
        assert not is_redundant, "gw_learned should NOT be redundant when HPO chose concat"

    # =========================================================================
    # Evolution Branch Redundancy
    # =========================================================================

    def test_no_evolution_redundant_when_no_evolution_data(self):
        """no_evolution ablation is redundant if data has no evolution features."""
        hpo_params = {}
        is_redundant, reason = is_ablation_redundant("no_evolution", hpo_params, evolution_dim=0)
        assert is_redundant, "no_evolution should be redundant when evolution_dim=0"
        assert "No evolution features" in reason

    def test_no_evolution_redundant_when_hpo_disabled(self):
        """no_evolution ablation is redundant if HPO already disabled evolution."""
        hpo_params = {"use_evolution_branch": False}
        is_redundant, reason = is_ablation_redundant("no_evolution", hpo_params, evolution_dim=28)
        assert is_redundant, "no_evolution should be redundant when HPO chose use_evolution_branch=False"
        assert "use_evolution_branch=False" in reason

    def test_no_evolution_not_redundant_when_hpo_enabled(self):
        """no_evolution ablation is valid if HPO enabled evolution."""
        hpo_params = {"use_evolution_branch": True}
        is_redundant, _ = is_ablation_redundant("no_evolution", hpo_params, evolution_dim=28)
        assert not is_redundant, "no_evolution should NOT be redundant when HPO enabled evolution"

    def test_no_evolution_not_redundant_when_hpo_default(self):
        """no_evolution ablation is valid if HPO didn't specify (uses default)."""
        hpo_params = {}  # No explicit choice, will use default
        is_redundant, _ = is_ablation_redundant("no_evolution", hpo_params, evolution_dim=28)
        assert not is_redundant, "no_evolution should NOT be redundant when HPO used default"

    # =========================================================================
    # Architecture Ablations Redundancy
    # =========================================================================

    def test_no_ring_pooling_redundant_when_hpo_disabled(self):
        """no_ring_pooling ablation is redundant if HPO already disabled it."""
        hpo_params = {"use_learned_ring_pooling": False}
        is_redundant, reason = is_ablation_redundant("no_ring_pooling", hpo_params)
        assert is_redundant, "no_ring_pooling should be redundant when HPO disabled it"
        assert "use_learned_ring_pooling=False" in reason

    def test_no_ring_pooling_not_redundant_when_hpo_enabled(self):
        """no_ring_pooling ablation is valid if HPO enabled ring pooling."""
        hpo_params = {"use_learned_ring_pooling": True}
        is_redundant, _ = is_ablation_redundant("no_ring_pooling", hpo_params)
        assert not is_redundant, "no_ring_pooling should NOT be redundant when HPO enabled it"

    def test_no_context_refiner_redundant_when_hpo_disabled(self):
        """no_context_refiner ablation is redundant if HPO already disabled it."""
        hpo_params = {"use_context_refiner": False}
        is_redundant, reason = is_ablation_redundant("no_context_refiner", hpo_params)
        assert is_redundant, "no_context_refiner should be redundant when HPO disabled it"
        assert "use_context_refiner=False" in reason

    def test_no_gate_redundant_when_hpo_disabled(self):
        """no_gate ablation is redundant if HPO already disabled cross-attn drift."""
        hpo_params = {"use_cross_attn_drift": False}
        is_redundant, reason = is_ablation_redundant("no_gate", hpo_params)
        assert is_redundant, "no_gate should be redundant when HPO disabled cross-attn drift"
        assert "use_cross_attn_drift=False" in reason

    # =========================================================================
    # AMICI Attention Redundancy
    # =========================================================================

    def test_no_distance_redundant_when_amici_disabled(self):
        """no_distance ablation is redundant if AMICI attention is disabled."""
        hpo_params = {"use_amici_attention": False}
        is_redundant, reason = is_ablation_redundant("no_distance", hpo_params)
        assert is_redundant, "no_distance should be redundant when AMICI disabled"
        assert "use_amici_attention=False" in reason

    def test_no_distance_not_redundant_when_amici_enabled(self):
        """no_distance ablation is valid if AMICI attention is enabled."""
        hpo_params = {"use_amici_attention": True}
        is_redundant, _ = is_ablation_redundant("no_distance", hpo_params)
        assert not is_redundant, "no_distance should NOT be redundant when AMICI enabled"

    # =========================================================================
    # Ablations That Should Never Be Redundant
    # =========================================================================

    def test_no_niche_never_redundant(self):
        """no_niche ablation tests core novelty - should never be skipped."""
        for hpo_params in [{}, {"use_gw_fusion": True}, {"use_gw_fusion": False}]:
            is_redundant, _ = is_ablation_redundant("no_niche", hpo_params)
            assert not is_redundant, f"no_niche should never be redundant: {hpo_params}"

    def test_random_niche_never_redundant(self):
        """random_niche ablation tests spatial structure - should never be skipped."""
        for hpo_params in [{}, {"use_gw_fusion": True}, {"use_gw_fusion": False}]:
            is_redundant, _ = is_ablation_redundant("random_niche", hpo_params)
            assert not is_redundant, f"random_niche should never be redundant: {hpo_params}"

    def test_hlca_only_never_redundant(self):
        """hlca_only ablation tests dual-reference value - should never be skipped."""
        for hpo_params in [{}, {"use_gw_fusion": True}, {"use_gw_fusion": False}]:
            is_redundant, _ = is_ablation_redundant("hlca_only", hpo_params)
            assert not is_redundant, f"hlca_only should never be redundant: {hpo_params}"

    def test_luca_only_never_redundant(self):
        """luca_only ablation tests dual-reference value - should never be skipped."""
        for hpo_params in [{}, {"use_gw_fusion": True}, {"use_gw_fusion": False}]:
            is_redundant, _ = is_ablation_redundant("luca_only", hpo_params)
            assert not is_redundant, f"luca_only should never be redundant: {hpo_params}"

    def test_frozen_encoder_never_redundant(self):
        """frozen_encoder ablation tests SSL quality - should never be skipped."""
        for hpo_params in [{}, {"use_gw_fusion": True}, {"use_gw_fusion": False}]:
            is_redundant, _ = is_ablation_redundant("frozen_encoder", hpo_params)
            assert not is_redundant, f"frozen_encoder should never be redundant: {hpo_params}"

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_unknown_ablation_not_redundant(self):
        """Unknown ablation names should not be marked redundant."""
        is_redundant, _ = is_ablation_redundant("nonexistent_ablation", {})
        assert not is_redundant, "Unknown ablations should not be marked redundant"

    def test_empty_hpo_params_uses_defaults(self):
        """Empty HPO params should use defaults (most ablations valid)."""
        for ablation in ABLATION_CONFIGS:
            if ablation in {"no_distance"}:  # This depends on AMICI which defaults False
                continue
            is_redundant, _ = is_ablation_redundant(ablation, {}, evolution_dim=28)
            # Most should not be redundant with empty HPO
            if ablation not in {"no_distance"}:
                assert not is_redundant, f"{ablation} should not be redundant with empty HPO params"
