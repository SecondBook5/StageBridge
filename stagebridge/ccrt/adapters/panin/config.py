"""PanIN adapter configuration.

Frozen, explicit configuration for translating the verified PanIN reference
source (see ``docs/ccrt/panin/SOURCE_AUDIT.md``) into CCRT inputs. No fuzzy
runtime discovery: column names, ontology maps, edges, and feature blocks are
declared explicitly and validated strictly.

The reference layout constants encode the actual GEO-hosted PanIN project layout
(the source repo is code-only; data live under ``data/xenium/PanIN*/`` etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = [
    "REFERENCE_PANIN_REPOSITORY_NAME",
    "REFERENCE_PANIN_SOURCE_LAYOUT_VERSION",
    "PanINColumnMap",
    "PanINFeatureBlockConfig",
    "PanINNeighborhoodConfig",
    "PanINSplitConfig",
    "PanINAdapterConfig",
    "build_reference_panin_adapter_config",
]

REFERENCE_PANIN_REPOSITORY_NAME = "PanIN_carcinogeneisis_spatial_analysis"
REFERENCE_PANIN_SOURCE_LAYOUT_VERSION = "panin-source-2024-xenium-cogaps-v1"

_ALLOWED_ROLES = frozenset({"semantic", "reconstruction", "regulatory"})
_ALLOWED_METRICS = frozenset({"squared_euclidean", "cosine"})
_ALLOWED_NORMALIZATIONS = frozenset({"none", "l2"})


@dataclass(frozen=True)
class PanINColumnMap:
    """Source-schema column names (from the audit). None = genuinely absent."""

    observation_id: str
    donor_id: str | None
    sample_id: str
    section_id: str
    platform: str
    stage: str
    cell_type: str
    receiver_state: str
    x_coordinate: str
    y_coordinate: str
    z_coordinate: str | None = None
    uncertainty: str | None = None

    def __post_init__(self) -> None:
        required = (
            "observation_id", "sample_id", "section_id", "platform", "stage",
            "cell_type", "receiver_state", "x_coordinate", "y_coordinate",
        )
        for name in required:
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"PanINColumnMap.{name} must be a non-empty string")
        for name in ("donor_id", "z_coordinate", "uncertainty"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(f"PanINColumnMap.{name} must be None or non-empty")


@dataclass(frozen=True)
class PanINFeatureBlockConfig:
    """Configuration for one feature block (receiver / sender / semantic / reg)."""

    feature_space_id: str
    role: str
    source_path: str
    observation_id_key: str
    feature_ids: tuple[str, ...]
    matrix_key: str | None = None
    normalization: str = "none"
    metric: str | None = None
    version: str = "1"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_space_id.strip():
            raise ValueError("feature_space_id must be non-empty")
        if self.role not in _ALLOWED_ROLES:
            raise ValueError(f"role '{self.role}' invalid; allowed: {sorted(_ALLOWED_ROLES)}")
        if not self.source_path.strip():
            raise ValueError("source_path must be non-empty")
        if not self.observation_id_key.strip():
            raise ValueError("observation_id_key must be non-empty")
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        if not self.feature_ids:
            raise ValueError("feature_ids must be non-empty and ordered")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("feature_ids must be unique")
        if self.normalization not in _ALLOWED_NORMALIZATIONS:
            raise ValueError(f"normalization '{self.normalization}' invalid")
        if self.role == "semantic":
            if self.metric is None or self.metric not in _ALLOWED_METRICS:
                raise ValueError("semantic role requires a supported metric")
        elif self.metric is not None and self.metric not in _ALLOWED_METRICS:
            raise ValueError(f"metric '{self.metric}' invalid")


@dataclass(frozen=True)
class PanINNeighborhoodConfig:
    """Continuous spatial-neighborhood configuration."""

    max_neighbors: int
    max_distance: float | None
    distance_units: str
    coordinate_scale_to_microns: float
    include_receiver_lineage_as_sender: bool = False

    def __post_init__(self) -> None:
        if self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be > 0")
        if self.max_distance is not None and self.max_distance <= 0:
            raise ValueError("max_distance must be None or > 0")
        if not self.distance_units.strip():
            raise ValueError("distance_units must be non-empty")
        if self.coordinate_scale_to_microns <= 0:
            raise ValueError("coordinate_scale_to_microns must be > 0")


@dataclass(frozen=True)
class PanINSplitConfig:
    """Donor-grouped split configuration."""

    grouping_level: str = "donor"
    num_folds: int = 5
    seed: int = 0
    allow_sample_level_grouping: bool = False

    def __post_init__(self) -> None:
        if self.grouping_level not in ("donor", "sample"):
            raise ValueError("grouping_level must be 'donor' or 'sample'")
        if self.num_folds < 2:
            raise ValueError("num_folds must be >= 2")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.grouping_level == "sample" and not self.allow_sample_level_grouping:
            raise ValueError(
                "sample grouping requires allow_sample_level_grouping=True "
                "(no silent fallback from donor to sample)"
            )


@dataclass(frozen=True)
class PanINAdapterConfig:
    """Full explicit PanIN adapter configuration."""

    source_root: str
    source_layout_version: str
    columns: PanINColumnMap
    receiver_annotation_map: Mapping[str, str]
    sender_context_annotation_map: Mapping[str, str]
    stage_map: Mapping[str, str]
    transition_edges: tuple[tuple[str, str, str], ...]  # (edge_id, source, target)
    receiver_feature_block: PanINFeatureBlockConfig
    sender_feature_block: PanINFeatureBlockConfig
    semantic_feature_block: PanINFeatureBlockConfig
    neighborhood: PanINNeighborhoodConfig
    splits: PanINSplitConfig
    primary_platform: str
    allowed_platforms: tuple[str, ...]
    excluded_annotations: tuple[str, ...] = ()
    regulatory_feature_block: PanINFeatureBlockConfig | None = None
    minimum_receivers_per_edge: int = 1
    minimum_targets_per_edge: int = 1
    strict_unknown_annotations: bool = True
    strict_provenance: bool = True
    biological_system_id: str = "panin_progression"

    def __post_init__(self) -> None:
        if not self.source_root.strip():
            raise ValueError("source_root must be non-empty")
        if not self.source_layout_version.strip():
            raise ValueError("source_layout_version must be non-empty")
        object.__setattr__(self, "receiver_annotation_map", dict(self.receiver_annotation_map))
        object.__setattr__(self, "sender_context_annotation_map", dict(self.sender_context_annotation_map))
        object.__setattr__(self, "stage_map", dict(self.stage_map))
        object.__setattr__(self, "transition_edges", tuple(tuple(e) for e in self.transition_edges))
        object.__setattr__(self, "allowed_platforms", tuple(self.allowed_platforms))
        object.__setattr__(self, "excluded_annotations", tuple(self.excluded_annotations))
        if not self.receiver_annotation_map:
            raise ValueError("receiver_annotation_map must be non-empty")
        if not self.sender_context_annotation_map:
            raise ValueError("sender_context_annotation_map must be non-empty")
        if not self.transition_edges:
            raise ValueError("transition_edges must be non-empty")
        for edge in self.transition_edges:
            if len(edge) != 3:
                raise ValueError("each transition edge must be (edge_id, source, target)")
        if self.primary_platform not in self.allowed_platforms:
            raise ValueError("primary_platform must be in allowed_platforms")
        if self.minimum_receivers_per_edge < 1 or self.minimum_targets_per_edge < 1:
            raise ValueError("minimum receivers/targets per edge must be >= 1")
        if self.semantic_feature_block.role != "semantic":
            raise ValueError("semantic_feature_block must have role 'semantic'")


def build_reference_panin_adapter_config(source_root: str | Path) -> PanINAdapterConfig:
    """Encode the verified reference PanIN (Xenium cell-resolved) layout.

    Uses the audited column names, ontology maps, edges, and the 8 projected
    CoGAPS patterns as the semantic space. Fails clearly if the layout differs at
    audit/validation time; this function itself performs no fuzzy discovery.
    """
    columns = PanINColumnMap(
        observation_id="cell_id",
        donor_id="donor_id",
        sample_id="sample_id",
        section_id="section_id",
        platform="platform",
        stage="cell_type_confirmed",
        cell_type="cell_type",
        receiver_state="cell_type_confirmed",
        x_coordinate="x_centroid",
        y_coordinate="y_centroid",
    )

    receiver_annotation_map = {
        "normal epithelium": "normal_duct",
        "normal_duct": "normal_duct",
        "low_grade_PanIN": "low_grade_panin",
        "high_grade_PanIN": "high_grade_panin",
    }
    sender_context_annotation_map = {
        "panCAF": "caf",
        "CAF": "caf",
        "apCAF": "apcaf",
        "iCAF": "icaf",
        "myCAF": "mycaf",
        "immune": "immune",
        "CD45": "immune",
    }
    stage_map = {
        "normal epithelium": "normal_duct",
        "normal_duct": "normal_duct",
        "low_grade_PanIN": "low_grade_panin",
        "high_grade_PanIN": "high_grade_panin",
    }
    transition_edges = (
        ("normal_duct__to__low_grade_panin", "normal_duct", "low_grade_panin"),
        ("low_grade_panin__to__high_grade_panin", "low_grade_panin", "high_grade_panin"),
    )

    # SCT-scaled expression panel (illustrative Xenium panel genes verified in
    # source scripts). The real adapter validates these against the source matrix.
    panel = ("TSPAN8", "TFF1", "COL1A1", "CD8A", "FAP", "LUM", "DCN", "PTPRC")
    receiver_block = PanINFeatureBlockConfig(
        feature_space_id="panin_receiver_xenium_sct",
        role="reconstruction",
        source_path="processed_data/01_Load_Xenium_Data/Xenium_combined_SCT_processed.rds",
        observation_id_key="cell_id",
        feature_ids=panel,
        matrix_key="SCT.scale.data",
        normalization="none",
        version="1",
        description="Xenium SCT-scaled expression (receiver epithelial cells)",
    )
    sender_block = PanINFeatureBlockConfig(
        feature_space_id="panin_sender_xenium_sct",
        role="reconstruction",
        source_path="processed_data/01_Load_Xenium_Data/Xenium_combined_SCT_processed.rds",
        observation_id_key="cell_id",
        feature_ids=panel,
        matrix_key="SCT.scale.data",
        normalization="none",
        version="1",
        description="Xenium SCT-scaled expression (candidate sender cells)",
    )
    semantic_block = PanINFeatureBlockConfig(
        feature_space_id="panin_semantic_cogaps_n8",
        role="semantic",
        source_path="processed_data/02_Pattern_Projection/Xenium_projectedPatterns.rds",
        observation_id_key="cell_id",
        feature_ids=tuple(f"Pattern_{i}" for i in range(1, 9)),
        matrix_key="projectedPatterns",
        normalization="none",
        metric="squared_euclidean",
        version="1",
        description="8 projected CoGAPS patterns from the PDAC atlas (z_sem)",
    )

    neighborhood = PanINNeighborhoodConfig(
        max_neighbors=20,
        max_distance=50.0,
        distance_units="microns",
        coordinate_scale_to_microns=1.0,  # Xenium centroids already microns
    )
    splits = PanINSplitConfig(grouping_level="donor", num_folds=5, seed=0)

    return PanINAdapterConfig(
        source_root=str(source_root),
        source_layout_version=REFERENCE_PANIN_SOURCE_LAYOUT_VERSION,
        columns=columns,
        receiver_annotation_map=receiver_annotation_map,
        sender_context_annotation_map=sender_context_annotation_map,
        stage_map=stage_map,
        transition_edges=transition_edges,
        receiver_feature_block=receiver_block,
        sender_feature_block=sender_block,
        semantic_feature_block=semantic_block,
        regulatory_feature_block=None,  # unavailable in source (not fabricated)
        neighborhood=neighborhood,
        splits=splits,
        primary_platform="xenium",
        allowed_platforms=("xenium",),
        excluded_annotations=("fat",),
        minimum_receivers_per_edge=1,
        minimum_targets_per_edge=1,
        strict_unknown_annotations=True,
        strict_provenance=True,
    )
