"""Tests for contracts.py - the single source of truth for constants and schemas."""

import pytest

from stagebridge.contracts import (
    # Stage definitions
    STAGES_3,
    STAGES_5,
    STAGE_5_TO_3,
    STAGE_5_TO_4,
    get_stage_system,
    convert_stage,
    stage_to_idx,
    idx_to_stage,
    # Latent dimensions
    HLCA_DIM,
    LUCA_DIM,
    LATENT_DIM,
    # GW Fusion
    GW_FUSION_TYPES,
    FUSION_METHODS,
    get_fused_dim,
    # Evolution features
    WES_COLS,
    WES_DIM,
    CLONAL_COLS,
    CLONAL_DIM,
    EVOLUTION_COLS,
    EVOLUTION_DIM,
    # Token structure
    N_TOKENS,
    TOKEN_NAMES,
    TOKEN_TYPE_IDS,
    # Ablations
    AblationOutputContract,
)


class TestStageDefinitions:
    """Test stage system definitions and conversions."""

    def test_stage_3_has_three_stages(self):
        assert len(STAGES_3) == 3
        assert STAGES_3 == ("Normal", "Preinvasive", "Invasive")

    def test_stage_5_has_five_stages(self):
        assert len(STAGES_5) == 5
        assert STAGES_5 == ("Normal", "AAH", "AIS", "MIA", "LUAD")

    def test_stage_5_to_3_mapping_complete(self):
        """Every 5-stage must map to a 3-stage."""
        for stage in STAGES_5:
            assert stage in STAGE_5_TO_3
            assert STAGE_5_TO_3[stage] in STAGES_3

    def test_stage_5_to_4_mapping_complete(self):
        """Every 5-stage must map to a 4-stage."""
        for stage in STAGES_5:
            assert stage in STAGE_5_TO_4

    def test_get_stage_system_3(self):
        stages, s2i, i2s = get_stage_system("3")
        assert stages == STAGES_3
        assert s2i["Normal"] == 0
        assert s2i["Invasive"] == 2
        assert i2s[0] == "Normal"

    def test_get_stage_system_5(self):
        stages, s2i, i2s = get_stage_system("5")
        assert stages == STAGES_5
        assert s2i["AAH"] == 1
        assert i2s[4] == "LUAD"

    def test_convert_stage_5_to_3(self):
        assert convert_stage("Normal", "3") == "Normal"
        assert convert_stage("AAH", "3") == "Preinvasive"
        assert convert_stage("AIS", "3") == "Preinvasive"
        assert convert_stage("MIA", "3") == "Invasive"
        assert convert_stage("LUAD", "3") == "Invasive"

    def test_stage_to_idx_and_back(self):
        for i, stage in enumerate(STAGES_3):
            assert stage_to_idx(stage, "3") == i
            assert idx_to_stage(i, "3") == stage

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            stage_to_idx("InvalidStage", "3")
        with pytest.raises(ValueError):
            idx_to_stage(99, "3")


class TestLatentDimensions:
    """Test latent dimension constants."""

    def test_hlca_dim(self):
        assert HLCA_DIM == 30

    def test_luca_dim(self):
        assert LUCA_DIM == 10

    def test_latent_dim_is_sum(self):
        assert LATENT_DIM == HLCA_DIM + LUCA_DIM
        assert LATENT_DIM == 40


class TestGWFusionTypes:
    """Test GW fusion type definitions."""

    def test_gw_fusion_types_defined(self):
        assert "concat" in GW_FUSION_TYPES
        assert "learned_projection" in GW_FUSION_TYPES
        assert "pretrained" in GW_FUSION_TYPES

    def test_broken_modes_not_in_gw_fusion_types(self):
        """Broken per-batch modes should not be in valid types."""
        assert "per_batch" not in GW_FUSION_TYPES
        assert "barycentric" not in GW_FUSION_TYPES
        assert "project_to_hlca" not in GW_FUSION_TYPES
        assert "project_to_luca" not in GW_FUSION_TYPES

    def test_legacy_fusion_methods_still_exist(self):
        """Legacy methods kept for backward compatibility."""
        assert "concat" in FUSION_METHODS
        assert "weighted" in FUSION_METHODS
        assert "gated" in FUSION_METHODS
        assert "film" in FUSION_METHODS

    def test_get_fused_dim_concat(self):
        assert get_fused_dim("concat") == 40

    def test_get_fused_dim_pretrained(self):
        assert get_fused_dim("pretrained") == 40

    def test_get_fused_dim_learned_projection(self):
        assert get_fused_dim("learned_projection") == 40

    def test_get_fused_dim_legacy_methods(self):
        assert get_fused_dim("weighted") == 30
        assert get_fused_dim("gated") == 30
        assert get_fused_dim("film") == 30

    def test_get_fused_dim_invalid_raises(self):
        with pytest.raises(ValueError):
            get_fused_dim("invalid_method")


class TestEvolutionFeatures:
    """Test WES and clonal feature definitions."""

    def test_wes_dim_matches_cols(self):
        assert WES_DIM == len(WES_COLS)

    def test_clonal_dim_matches_cols(self):
        assert CLONAL_DIM == len(CLONAL_COLS)

    def test_evolution_dim_is_sum(self):
        assert EVOLUTION_DIM == WES_DIM + CLONAL_DIM

    def test_evolution_cols_is_concatenation(self):
        assert EVOLUTION_COLS == WES_COLS + CLONAL_COLS

    def test_wes_cols_has_expected_features(self):
        assert "tmb" in WES_COLS
        assert "kras_mut" in WES_COLS
        assert "egfr_mut" in WES_COLS
        assert "tp53_mut" in WES_COLS

    def test_clonal_cols_has_expected_features(self):
        assert "cnv_score" in CLONAL_COLS
        assert "clone_size" in CLONAL_COLS
        assert "n_clones" in CLONAL_COLS
        assert "clonal_entropy" in CLONAL_COLS


class TestTokenStructure:
    """Test 9-token niche structure definitions."""

    def test_n_tokens(self):
        assert N_TOKENS == 9

    def test_token_names_count(self):
        assert len(TOKEN_NAMES) == N_TOKENS

    def test_token_names_order(self):
        assert TOKEN_NAMES[0] == "receiver"
        assert TOKEN_NAMES[1] == "ring1"
        assert TOKEN_NAMES[5] == "hlca"
        assert TOKEN_NAMES[6] == "luca"
        assert TOKEN_NAMES[8] == "stats"

    def test_token_type_ids_defined(self):
        assert TOKEN_TYPE_IDS["receiver"] == 0
        assert TOKEN_TYPE_IDS["spatial"] == 1
        assert TOKEN_TYPE_IDS["hlca"] == 2
        assert TOKEN_TYPE_IDS["luca"] == 3


class TestAblationOutputContract:
    """Test ablation output contract."""

    def test_ablation_types_includes_valid_ablations(self):
        types = AblationOutputContract.ABLATION_TYPES
        assert "no_niche" in types
        assert "no_distance" in types
        assert "no_gate" in types
        assert "random_niche" in types
        assert "hlca_only" in types
        assert "luca_only" in types
        assert "frozen_encoder" in types
        assert "no_ring_pooling" in types
        assert "no_context_refiner" in types
        assert "no_gw_fusion" in types
        assert "gw_learned" in types
        assert "no_evolution" in types

    def test_ablation_types_excludes_broken_gw_modes(self):
        types = AblationOutputContract.ABLATION_TYPES
        assert "gw_barycentric" not in types
        assert "gw_project_hlca" not in types
        assert "gw_project_luca" not in types

    def test_validate_missing_keys(self):
        errors = AblationOutputContract.validate({})
        assert any("ablation_type" in e for e in errors)
        assert any("metrics" in e for e in errors)
        assert any("delta_vs_full" in e for e in errors)

    def test_validate_unknown_ablation(self):
        result = {
            "ablation_type": "nonexistent_ablation",
            "metrics": {},
            "delta_vs_full": 0.0,
        }
        errors = AblationOutputContract.validate(result)
        assert any("Unknown ablation" in e for e in errors)

    def test_validate_valid_result(self):
        result = {
            "ablation_type": "no_niche",
            "metrics": {"val_loss": 0.1},
            "delta_vs_full": 0.05,
        }
        errors = AblationOutputContract.validate(result)
        assert len(errors) == 0
