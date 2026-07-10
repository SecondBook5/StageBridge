"""PanIN adapter — verified PanIN source to CCRT inputs.

Source-locked, provenance-preserving translation of the PanIN reference project
(Xenium cell-resolved primary path) into CCRT grammar, records, and model-ready
edge partitions. Knows PanIN biology; imports only the system-agnostic core
(contracts/grammar/io/data/representations) — never model or training layers.
"""

from __future__ import annotations

from .adapter import (
    PanINAdapterOutput,
    PanINEdgePartition,
    PanINSpatialLoader,
    adapt_reference_panin,
)
from .config import (
    REFERENCE_PANIN_REPOSITORY_NAME,
    REFERENCE_PANIN_SOURCE_LAYOUT_VERSION,
    PanINAdapterConfig,
    PanINColumnMap,
    PanINFeatureBlockConfig,
    PanINNeighborhoodConfig,
    PanINSplitConfig,
    build_reference_panin_adapter_config,
)
from .features import (
    PanINFeatureBlock,
    load_panin_feature_block,
    register_panin_feature_spaces,
)
from .neighborhoods import (
    PanINNeighborRecord,
    PanINSpatialObservation,
    build_continuous_sender_neighborhoods,
)
from .ontology import PanINOntology, PanINOntologyEntry, build_panin_ontology
from .records import PanINRecordBundle, build_panin_record_bundle
from .source_audit import (
    PanINSourceAudit,
    PanINSourceFile,
    audit_panin_source,
    validate_reference_source_audit,
)
from .splits import (
    PanINFoldAssignment,
    PanINGroupRecord,
    build_grouped_panin_folds,
    validate_no_panin_group_leakage,
)
from .validation import PanINValidationReport, validate_panin_adapter_output

__all__ = [
    # config
    "REFERENCE_PANIN_REPOSITORY_NAME",
    "REFERENCE_PANIN_SOURCE_LAYOUT_VERSION",
    "PanINColumnMap",
    "PanINFeatureBlockConfig",
    "PanINNeighborhoodConfig",
    "PanINSplitConfig",
    "PanINAdapterConfig",
    "build_reference_panin_adapter_config",
    # source audit
    "PanINSourceFile",
    "PanINSourceAudit",
    "audit_panin_source",
    "validate_reference_source_audit",
    # ontology
    "PanINOntologyEntry",
    "PanINOntology",
    "build_panin_ontology",
    # features
    "PanINFeatureBlock",
    "load_panin_feature_block",
    "register_panin_feature_spaces",
    # neighborhoods
    "PanINSpatialObservation",
    "PanINNeighborRecord",
    "build_continuous_sender_neighborhoods",
    # splits
    "PanINGroupRecord",
    "PanINFoldAssignment",
    "build_grouped_panin_folds",
    "validate_no_panin_group_leakage",
    # records
    "PanINRecordBundle",
    "build_panin_record_bundle",
    # adapter
    "PanINEdgePartition",
    "PanINAdapterOutput",
    "PanINSpatialLoader",
    "adapt_reference_panin",
    # validation
    "PanINValidationReport",
    "validate_panin_adapter_output",
]
