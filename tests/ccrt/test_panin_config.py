"""Tests for PanIN adapter configuration."""

from __future__ import annotations

import pytest

from stagebridge.ccrt.adapters.panin import (
    REFERENCE_PANIN_REPOSITORY_NAME,
    PanINColumnMap,
    PanINFeatureBlockConfig,
    PanINNeighborhoodConfig,
    PanINSplitConfig,
    build_reference_panin_adapter_config,
)


def test_reference_config_builds():
    cfg = build_reference_panin_adapter_config("/tmp/panin")
    assert cfg.biological_system_id == "panin_progression"
    assert cfg.primary_platform == "xenium"
    assert cfg.regulatory_feature_block is None  # unavailable in source
    assert len(cfg.transition_edges) == 2
    assert cfg.semantic_feature_block.role == "semantic"
    assert cfg.semantic_feature_block.metric == "squared_euclidean"


def test_repository_name_matches_source():
    assert REFERENCE_PANIN_REPOSITORY_NAME == "PanIN_carcinogeneisis_spatial_analysis"


def test_column_map_requires_core_columns():
    with pytest.raises(ValueError):
        PanINColumnMap(
            observation_id="", donor_id="d", sample_id="s", section_id="sec",
            platform="p", stage="st", cell_type="ct", receiver_state="rs",
            x_coordinate="x", y_coordinate="y",
        )


def test_column_map_optional_none_ok():
    cm = PanINColumnMap(
        observation_id="cell_id", donor_id=None, sample_id="s", section_id="sec",
        platform="p", stage="st", cell_type="ct", receiver_state="rs",
        x_coordinate="x", y_coordinate="y",
    )
    assert cm.donor_id is None


def test_feature_block_semantic_requires_metric():
    with pytest.raises(ValueError):
        PanINFeatureBlockConfig(
            feature_space_id="z", role="semantic", source_path="p.csv",
            observation_id_key="cell_id", feature_ids=("a", "b"),
        )


def test_feature_block_unique_features():
    with pytest.raises(ValueError):
        PanINFeatureBlockConfig(
            feature_space_id="z", role="reconstruction", source_path="p.csv",
            observation_id_key="cell_id", feature_ids=("a", "a"),
        )


def test_neighborhood_validation():
    with pytest.raises(ValueError):
        PanINNeighborhoodConfig(max_neighbors=0, max_distance=10.0, distance_units="microns", coordinate_scale_to_microns=1.0)
    with pytest.raises(ValueError):
        PanINNeighborhoodConfig(max_neighbors=5, max_distance=-1.0, distance_units="microns", coordinate_scale_to_microns=1.0)
    with pytest.raises(ValueError):
        PanINNeighborhoodConfig(max_neighbors=5, max_distance=None, distance_units="microns", coordinate_scale_to_microns=0.0)


def test_split_config_no_silent_sample_fallback():
    with pytest.raises(ValueError):
        PanINSplitConfig(grouping_level="sample", allow_sample_level_grouping=False)
    # explicit opt-in works
    PanINSplitConfig(grouping_level="sample", allow_sample_level_grouping=True)


def test_split_config_folds_minimum():
    with pytest.raises(ValueError):
        PanINSplitConfig(num_folds=1)
