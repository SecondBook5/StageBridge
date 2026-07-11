"""LUAD multimodal adapter configuration.

Frozen, explicit configuration for translating the verified LUAD premalignant
progression source (see ``docs/ccrt/luad/SOURCE_AUDIT.md``) into CCRT inputs. No
fuzzy runtime discovery: column names, ontology maps, edges, feature blocks, and
the deconvolution context backend are declared explicitly and validated strictly.

The reference layout constants encode the actual local StageBridge LUAD data tree
(``processed/{features,tangram}/*.parquet``). The dataset accuracy is NOT
verified (see the audit) — this configuration locks *structure*, never biology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = [
    "REFERENCE_LUAD_DATASET_NAME",
    "REFERENCE_LUAD_SOURCE_LAYOUT_VERSION",
    "PROGRESSION_ADJACENT_EVO_FEATURES",
    "LUADModalityColumnMap",
    "LUADColumnMap",
    "LUADFeatureBlockConfig",
    "LUADContextBackendConfig",
    "LUADNeighborhoodConfig",
    "LUADSplitConfig",
    "LUADAdapterConfig",
    "build_reference_luad_adapter_config",
]

REFERENCE_LUAD_DATASET_NAME = "luad_premalignant_progression"
REFERENCE_LUAD_SOURCE_LAYOUT_VERSION = "luad-source-2025-snrna-visium-tangram-hlca-v1"

_ALLOWED_ROLES = frozenset({"semantic", "reconstruction", "regulatory"})
_ALLOWED_METRICS = frozenset({"squared_euclidean", "cosine"})
_ALLOWED_NORMALIZATIONS = frozenset({"none", "l2"})

#: Progression-adjacent evolutionary features. They are source-provided
#: regulatory-mediator features (documented provenance) and are NEVER used as
#: CCRT targets or receiver-state derivations. A strict-hygiene config may drop
#: them from the regulatory block.
PROGRESSION_ADJACENT_EVO_FEATURES = (
    "evo_progression_risk_score",
    "evo_evidence_of_progression_link",
)


@dataclass(frozen=True)
class LUADModalityColumnMap:
    """Source-schema column names for one modality. None = genuinely absent."""

    observation_id: str
    sample_id: str
    donor_id: str | None = None
    patient_id: str | None = None
    section_id: str | None = None
    stage: str | None = None
    cell_type: str | None = None
    niche_id: str | None = None
    lesion_id: str | None = None
    x_coordinate: str | None = None
    y_coordinate: str | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "sample_id"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"LUADModalityColumnMap.{name} must be a non-empty string")
        for name in (
            "donor_id", "patient_id", "section_id", "stage", "cell_type",
            "niche_id", "lesion_id", "x_coordinate", "y_coordinate",
        ):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v.strip()):
                raise ValueError(f"LUADModalityColumnMap.{name} must be None or non-empty")


@dataclass(frozen=True)
class LUADColumnMap:
    """The two-modality column map (snRNA reference + Visium spatial)."""

    snrna_columns: LUADModalityColumnMap
    visium_columns: LUADModalityColumnMap

    def __post_init__(self) -> None:
        for name in ("snrna_columns", "visium_columns"):
            v = getattr(self, name)
            if not isinstance(v, LUADModalityColumnMap):
                raise ValueError(f"LUADColumnMap.{name} must be a LUADModalityColumnMap")
        # Visium receivers require coordinates (spatial neighborhoods).
        vc = self.visium_columns
        if vc.x_coordinate is None or vc.y_coordinate is None:
            raise ValueError(
                "LUADColumnMap.visium_columns must declare x/y coordinate columns "
                "(observed spots carry spatial coordinates)"
            )


@dataclass(frozen=True)
class LUADFeatureBlockConfig:
    """Configuration for one source-derived feature block (semantic / reconstruction / regulatory)."""

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
class LUADContextBackendConfig:
    """Deconvolution context backend schema (one backend, preserved by identity).

    Describes the (melted, long-form) typed-context table produced by a single
    deconvolution backend: one row per (spot, cell-type) component. Each row
    carries the spot id, the source cell-type label, its abundance/score, an
    optional uncertainty column, and the ordered feature-vector columns for the
    component. The backend is never selected/averaged/combined with others.
    """

    backend_id: str
    source_path: str
    spot_id_key: str
    sender_context_type_key: str
    abundance_key: str
    feature_ids: tuple[str, ...]
    uncertainty_key: str | None = None
    feature_space_id: str = "luad_context_tangram_profile"
    version: str = "1"
    description: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "backend_id", "source_path", "spot_id_key",
            "sender_context_type_key", "abundance_key",
        ):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"LUADContextBackendConfig.{name} must be a non-empty string")
        if self.uncertainty_key is not None and (
            not isinstance(self.uncertainty_key, str) or not self.uncertainty_key.strip()
        ):
            raise ValueError("uncertainty_key must be None or a non-empty string")
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        if not self.feature_ids:
            raise ValueError("context backend feature_ids must be non-empty and ordered")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("context backend feature_ids must be unique")


@dataclass(frozen=True)
class LUADNeighborhoodConfig:
    """Continuous spatial-neighborhood configuration for typed context components."""

    max_neighbors: int
    max_distance: float | None
    distance_units: str
    coordinate_scale_to_microns: float

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
class LUADSplitConfig:
    """Donor-grouped split configuration (donor = patient)."""

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
class LUADAdapterConfig:
    """Full explicit LUAD multimodal adapter configuration."""

    source_root: str
    source_layout_version: str
    columns: LUADColumnMap
    receiver_annotation_map: Mapping[str, str]
    sender_context_annotation_map: Mapping[str, str]
    stage_map: Mapping[str, str]
    transition_edges: tuple[tuple[str, str, str], ...]  # (edge_id, source, target)
    receiver_feature_block: LUADFeatureBlockConfig
    semantic_feature_block: LUADFeatureBlockConfig
    context_backend: LUADContextBackendConfig
    neighborhood: LUADNeighborhoodConfig
    splits: LUADSplitConfig
    primary_platform: str
    spatial_platform: str
    snrna_platform: str
    allowed_platforms: tuple[str, ...]
    regulatory_feature_block: LUADFeatureBlockConfig | None = None
    excluded_annotations: tuple[str, ...] = ()
    minimum_receivers_per_edge: int = 1
    minimum_targets_per_edge: int = 1
    strict_unknown_annotations: bool = True
    strict_provenance: bool = True
    biological_system_id: str = "luad_premalignant_progression"

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
        if self.spatial_platform not in self.allowed_platforms:
            raise ValueError("spatial_platform must be in allowed_platforms")
        if self.snrna_platform not in self.allowed_platforms:
            raise ValueError("snrna_platform must be in allowed_platforms")
        if self.minimum_receivers_per_edge < 1 or self.minimum_targets_per_edge < 1:
            raise ValueError("minimum receivers/targets per edge must be >= 1")
        if self.receiver_feature_block.role != "reconstruction":
            raise ValueError("receiver_feature_block must have role 'reconstruction'")
        if self.semantic_feature_block.role != "semantic":
            raise ValueError("semantic_feature_block must have role 'semantic'")
        if (
            self.regulatory_feature_block is not None
            and self.regulatory_feature_block.role != "regulatory"
        ):
            raise ValueError("regulatory_feature_block must have role 'regulatory'")


def _reference_evo_feature_ids(*, include_progression_adjacent: bool) -> tuple[str, ...]:
    """The 34 verified ``evo_*`` regulatory features (WES / evolutionary block)."""
    core = (
        "evo_tmb",
        "evo_driver_burden",
        "evo_kras_mutation",
        "evo_egfr_mutation",
        "evo_tp53_mutation",
        "evo_stk11_mutation",
        "evo_keap1_mutation",
        "evo_smad4_mutation",
        "evo_braf_mutation",
        "evo_purity",
        "evo_ploidy",
        "evo_cna_burden",
        "evo_clonal_fraction",
        "evo_subclonal_fraction",
        "evo_num_clones",
        "evo_clonal_diversity",
        "evo_wgd_status",
        "evo_loh_fraction",
        "evo_num_drivers",
        "evo_num_snvs",
        "evo_num_indels",
        "evo_num_cna_segments",
        "evo_frac_genome_altered",
        "evo_mutation_rate",
        "evo_neoantigen_burden",
        "evo_apobec_signature",
        "evo_smoking_signature",
        "evo_clock_signature",
        "evo_dnds_ratio",
        "evo_selection_intensity",
        "evo_shannon_index",
        "evo_simpson_index",
    )
    if include_progression_adjacent:
        return core + PROGRESSION_ADJACENT_EVO_FEATURES
    return core


def build_reference_luad_adapter_config(
    source_root: str | Path,
    *,
    include_progression_adjacent_evo: bool = True,
) -> LUADAdapterConfig:
    """Encode the verified LUAD multimodal layout.

    Uses the audited stage labels, HLCA niche semantic space, the Tangram
    deconvolution context backend, and the lesion evolutionary regulatory block.
    Performs no fuzzy discovery itself; the audit/validation step fails clearly if
    the source layout differs.
    """
    snrna_columns = LUADModalityColumnMap(
        observation_id="cell_id",
        sample_id="sample_id",
        donor_id="donor_id",
        patient_id="patient_id",
        section_id=None,
        stage="stage",
        cell_type="cell_type",
    )
    visium_columns = LUADModalityColumnMap(
        observation_id="spot_id",
        sample_id="sample_id",
        donor_id="donor_id",
        patient_id="patient_id",
        section_id="section_id",
        stage="stage",
        niche_id="niche_id",
        lesion_id="lesion_id",
        x_coordinate="x_spatial_microns",
        y_coordinate="y_spatial_microns",
    )
    columns = LUADColumnMap(snrna_columns=snrna_columns, visium_columns=visium_columns)

    receiver_annotation_map = {
        "Normal": "normal",
        "AAH": "aah",
        "AIS": "ais",
        "MIA": "mia",
        "LUAD": "invasive_luad",
    }
    stage_map = dict(receiver_annotation_map)
    sender_context_annotation_map = {
        "AT2": "at2",
        "Basal": "basal",
        "Capillary": "capillary",
        "Ciliated": "ciliated",
        "Fibroblast lineage": "fibroblast",
        "Macrophages": "macrophage",
        "Mast cells": "mast_cell",
        "Secretory": "secretory",
        "T cell lineage": "t_cell",
    }
    transition_edges = (
        ("normal__to__aah", "normal", "aah"),
        ("aah__to__ais", "aah", "ais"),
        ("ais__to__mia", "ais", "mia"),
        ("mia__to__invasive_luad", "mia", "invasive_luad"),
    )

    # Receiver reconstruction space: HLCA-smoothed niche composition tokens.
    composition_ids = (
        "tok_smooth_at2",
        "tok_smooth_basal",
        "tok_smooth_capillary",
        "tok_smooth_ciliated",
        "tok_smooth_fibroblast",
        "tok_smooth_macrophage",
        "tok_smooth_mast_cell",
        "tok_smooth_secretory",
        "tok_smooth_t_cell",
    )
    receiver_block = LUADFeatureBlockConfig(
        feature_space_id="luad_receiver_niche_composition",
        role="reconstruction",
        source_path="processed/features/niche_tokens_full.parquet",
        observation_id_key="niche_id",
        feature_ids=composition_ids,
        matrix_key=None,
        normalization="none",
        version="1",
        description="HLCA-smoothed niche composition tokens (receiver reconstruction)",
    )

    # Semantic z_sem: HLCA-projected epithelial-state similarity/deviation scores.
    hlca_ids = (
        "hlca_normal_likeness_score",
        "hlca_deviation_from_normal_score",
        "hlca_lineage_fidelity_score",
        "hlca_max_state_similarity",
        "hlca_topk_entropy",
        "hlca_epithelial_like_similarity",
        "hlca_immune_like_similarity",
        "hlca_stromal_endothelial_like_similarity",
    )
    semantic_block = LUADFeatureBlockConfig(
        feature_space_id="luad_semantic_hlca_niche",
        role="semantic",
        source_path="processed/features/niche_hlca_features.parquet",
        observation_id_key="niche_id",
        feature_ids=hlca_ids,
        matrix_key=None,
        normalization="none",
        metric="squared_euclidean",
        version="1",
        description="HLCA-projected epithelial semantic-state features (z_sem)",
    )

    # Regulatory r: lesion WES / evolutionary features.
    evo_ids = _reference_evo_feature_ids(
        include_progression_adjacent=include_progression_adjacent_evo
    )
    regulatory_block = LUADFeatureBlockConfig(
        feature_space_id="luad_regulatory_lesion_evo",
        role="regulatory",
        source_path="processed/features/lesion_evo_features.parquet",
        observation_id_key="lesion_id",
        feature_ids=evo_ids,
        matrix_key=None,
        normalization="none",
        version="1",
        description="Lesion WES / evolutionary regulatory-mediator features (r)",
    )

    # Deconvolution context backend: Tangram (the ONLY backend present).
    context_feature_ids = (
        "tangram_at2",
        "tangram_basal",
        "tangram_capillary",
        "tangram_ciliated",
        "tangram_fibroblast",
        "tangram_macrophage",
        "tangram_mast_cell",
        "tangram_secretory",
        "tangram_t_cell",
    )
    context_backend = LUADContextBackendConfig(
        backend_id="tangram",
        source_path="processed/tangram/spatial_tangram_celltype_scores.parquet",
        spot_id_key="spot_id",
        sender_context_type_key="sender_context_type",
        abundance_key="abundance",
        feature_ids=context_feature_ids,
        uncertainty_key=None,
        feature_space_id="luad_context_tangram_profile",
        version="1",
        description="Tangram deconvolved per-(spot, cell-type) context components",
    )

    neighborhood = LUADNeighborhoodConfig(
        max_neighbors=16,
        max_distance=200.0,
        distance_units="microns",
        coordinate_scale_to_microns=1.0,  # loader yields microns (Space Ranger scale applied)
    )
    splits = LUADSplitConfig(grouping_level="donor", num_folds=5, seed=0)

    return LUADAdapterConfig(
        source_root=str(source_root),
        source_layout_version=REFERENCE_LUAD_SOURCE_LAYOUT_VERSION,
        columns=columns,
        receiver_annotation_map=receiver_annotation_map,
        sender_context_annotation_map=sender_context_annotation_map,
        stage_map=stage_map,
        transition_edges=transition_edges,
        receiver_feature_block=receiver_block,
        semantic_feature_block=semantic_block,
        regulatory_feature_block=regulatory_block,
        context_backend=context_backend,
        neighborhood=neighborhood,
        splits=splits,
        primary_platform="visium",
        spatial_platform="visium",
        snrna_platform="snrna",
        allowed_platforms=("visium", "snrna"),
        excluded_annotations=(),
        minimum_receivers_per_edge=1,
        minimum_targets_per_edge=1,
        strict_unknown_annotations=True,
        strict_provenance=True,
    )
