"""LUAD multimodal adapter — verified LUAD source to CCRT inputs.

Source-locked, provenance-preserving translation of the LUAD premalignant
progression project (snRNA reference + Visium spatial + Tangram deconvolved
context) into CCRT grammar, records, and model-ready edge partitions. Knows LUAD
biology; imports only the system-agnostic core (contracts/grammar/io/data/
representations) — never model or training layers. Validates STRUCTURE only; the
underlying dataset accuracy is not verified (see docs/ccrt/luad/SOURCE_AUDIT.md).
"""

from __future__ import annotations

from .adapter import (
    LUADAdapterOutput,
    LUADEdgePartition,
    LUADSpatialLoader,
    adapt_reference_luad,
)
from .config import (
    PROGRESSION_ADJACENT_EVO_FEATURES,
    REFERENCE_LUAD_DATASET_NAME,
    REFERENCE_LUAD_SOURCE_LAYOUT_VERSION,
    LUADAdapterConfig,
    LUADColumnMap,
    LUADContextBackendConfig,
    LUADFeatureBlockConfig,
    LUADModalityColumnMap,
    LUADNeighborhoodConfig,
    LUADSplitConfig,
    build_reference_luad_adapter_config,
)
from .context_components import (
    LUADContextComponent,
    load_luad_context_components,
    validate_luad_context_components,
)
from .features import (
    FORBIDDEN_FEATURE_ID_TOKENS,
    LUADFeatureBlock,
    load_luad_feature_block,
    register_luad_feature_spaces,
)
from .manifests import (
    ALLOWED_MODALITY_RELATIONSHIP_TYPES,
    LUADModalityRecord,
    LUADModalityRelationship,
    build_luad_modality_manifest,
    validate_luad_modality_manifest,
)
from .neighborhoods import (
    LUADReceiverContextRecord,
    LUADSpatialSpot,
    build_luad_context_neighborhoods,
)
from .ontology import LUADOntology, LUADOntologyEntry, build_luad_ontology
from .records import LUADRecordBundle, build_luad_record_bundle
from .source_audit import (
    LUADSourceAudit,
    LUADSourceFile,
    audit_luad_source,
    validate_reference_source_audit,
)
from .splits import (
    LUADFoldAssignment,
    LUADGroupRecord,
    build_grouped_luad_folds,
    validate_no_luad_group_leakage,
)
from .validation import LUADValidationReport, validate_luad_adapter_output

__all__ = [
    # config
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
    # source audit
    "LUADSourceFile",
    "LUADSourceAudit",
    "audit_luad_source",
    "validate_reference_source_audit",
    # ontology
    "LUADOntologyEntry",
    "LUADOntology",
    "build_luad_ontology",
    # manifests
    "ALLOWED_MODALITY_RELATIONSHIP_TYPES",
    "LUADModalityRecord",
    "LUADModalityRelationship",
    "build_luad_modality_manifest",
    "validate_luad_modality_manifest",
    # features
    "FORBIDDEN_FEATURE_ID_TOKENS",
    "LUADFeatureBlock",
    "load_luad_feature_block",
    "register_luad_feature_spaces",
    # context components
    "LUADContextComponent",
    "load_luad_context_components",
    "validate_luad_context_components",
    # neighborhoods
    "LUADSpatialSpot",
    "LUADReceiverContextRecord",
    "build_luad_context_neighborhoods",
    # splits
    "LUADGroupRecord",
    "LUADFoldAssignment",
    "build_grouped_luad_folds",
    "validate_no_luad_group_leakage",
    # records
    "LUADRecordBundle",
    "build_luad_record_bundle",
    # adapter
    "LUADEdgePartition",
    "LUADAdapterOutput",
    "LUADSpatialLoader",
    "adapt_reference_luad",
    # validation
    "LUADValidationReport",
    "validate_luad_adapter_output",
]
