"""Tests for LUAD adapter configuration."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.luad import (
    REFERENCE_LUAD_DATASET_NAME,
    LUADColumnMap,
    LUADContextBackendConfig,
    LUADFeatureBlockConfig,
    LUADModalityColumnMap,
    LUADNeighborhoodConfig,
    LUADSplitConfig,
    build_reference_luad_adapter_config,
)


def test_reference_config_builds():
    cfg = build_reference_luad_adapter_config("/tmp/luad")
    assert cfg.biological_system_id == "luad_premalignant_progression"
    assert cfg.primary_platform == "visium"
    assert cfg.spatial_platform == "visium"
    assert cfg.snrna_platform == "snrna"
    assert len(cfg.transition_edges) == 4
    assert cfg.semantic_feature_block.role == "semantic"
    assert cfg.semantic_feature_block.metric == "squared_euclidean"
    # regulatory space genuinely available for LUAD
    assert cfg.regulatory_feature_block is not None
    assert cfg.regulatory_feature_block.role == "regulatory"
    assert len(cfg.regulatory_feature_block.feature_ids) == 34
    assert len(cfg.semantic_feature_block.feature_ids) == 8
    assert cfg.context_backend.backend_id == "tangram"
    assert len(cfg.context_backend.feature_ids) == 9


def test_dataset_name_matches_source():
    assert REFERENCE_LUAD_DATASET_NAME == "luad_premalignant_progression"


def test_edges_are_adjacent_progression():
    cfg = build_reference_luad_adapter_config("/tmp/luad")
    ids = [e[0] for e in cfg.transition_edges]
    assert ids == [
        "normal__to__aah",
        "aah__to__ais",
        "ais__to__mia",
        "mia__to__invasive_luad",
    ]


def test_progression_adjacent_evo_can_be_excluded():
    strict = build_reference_luad_adapter_config(
        "/tmp/luad", include_progression_adjacent_evo=False
    )
    fids = strict.regulatory_feature_block.feature_ids
    assert "evo_progression_risk_score" not in fids
    assert "evo_evidence_of_progression_link" not in fids
    assert len(fids) == 32


def test_column_map_requires_coordinates_for_visium():
    with pytest.raises(ValueError):
        LUADColumnMap(
            snrna_columns=LUADModalityColumnMap(observation_id="cell_id", sample_id="s"),
            visium_columns=LUADModalityColumnMap(observation_id="spot_id", sample_id="s"),
        )


def test_modality_column_map_requires_core_columns():
    with pytest.raises(ValueError):
        LUADModalityColumnMap(observation_id="", sample_id="s")


def test_context_backend_requires_features():
    with pytest.raises(ValueError):
        LUADContextBackendConfig(
            backend_id="tangram", source_path="c.csv", spot_id_key="spot_id",
            sender_context_type_key="t", abundance_key="a", feature_ids=(),
        )


def test_context_backend_optional_uncertainty():
    cb = LUADContextBackendConfig(
        backend_id="tangram", source_path="c.csv", spot_id_key="spot_id",
        sender_context_type_key="t", abundance_key="a", feature_ids=("f1",),
    )
    assert cb.uncertainty_key is None


def test_feature_block_semantic_requires_metric():
    with pytest.raises(ValueError):
        LUADFeatureBlockConfig(
            feature_space_id="z", role="semantic", source_path="p.csv",
            observation_id_key="niche_id", feature_ids=("a", "b"),
        )


def test_neighborhood_validation():
    with pytest.raises(ValueError):
        LUADNeighborhoodConfig(max_neighbors=0, max_distance=10.0, distance_units="microns", coordinate_scale_to_microns=1.0)
    with pytest.raises(ValueError):
        LUADNeighborhoodConfig(max_neighbors=5, max_distance=-1.0, distance_units="microns", coordinate_scale_to_microns=1.0)
    with pytest.raises(ValueError):
        LUADNeighborhoodConfig(max_neighbors=5, max_distance=None, distance_units="microns", coordinate_scale_to_microns=0.0)


def test_split_config_no_silent_sample_fallback():
    with pytest.raises(ValueError):
        LUADSplitConfig(grouping_level="sample", allow_sample_level_grouping=False)
    LUADSplitConfig(grouping_level="sample", allow_sample_level_grouping=True)


def test_split_config_folds_minimum():
    with pytest.raises(ValueError):
        LUADSplitConfig(num_folds=1)
