"""Reference mapping and latent-alignment utilities.

This package provides the dual-reference geometry layer for StageBridge,
enabling query cells to be mapped to both HLCA (healthy reference) and
LuCa (disease-aware reference) coordinate systems.

Key modules:
- loaders: Reference loading and validation
- prepare: Reference preparation and harmonization
- map_query: Query-to-reference mapping
- fuse: Dual-reference fusion
- confidence: Confidence scoring and quality metrics
- schema: Standardized output schemas
- visualize: Reference visualizations
- pipeline: Main pipeline integration
"""

from __future__ import annotations

# Core functionality from existing modules
from stagebridge.reference.hlca_mapper import (
    HLCAReference,
    HLCAMappingResult,
    load_hlca_reference as load_hlca_reference_legacy,
    run_active_reference_latent,
)
from stagebridge.reference.latent_store import LatentStore
from stagebridge.reference.diagnostics import (
    summarize_latent,
    stage_preservation_diagnostics,
    donor_leakage_diagnostics,
    gene_overlap_diagnostics,
    reference_alignment_gate,
)

# New dual-reference geometry modules
from stagebridge.reference.loaders import (
    LoadedReference,
    ReferenceInfo,
    FeatureOverlapReport,
    load_hlca_reference,
    load_luca_reference,
    validate_reference,
    compute_feature_overlap,
)
from stagebridge.reference.map_query import (
    MappingResult,
    ReferenceNeighborhood,
    map_to_hlca,
    map_to_luca,
)
from stagebridge.reference.fuse import (
    FusedEmbeddingResult,
    fuse_dual_reference,
    fuse_single_reference,
)
from stagebridge.reference.confidence import (
    ConfidenceScores,
    compute_hlca_confidence,
    compute_luca_confidence,
    compute_dual_confidence,
    detect_mapping_collapse,
    detect_nan_embeddings,
)
from stagebridge.reference.schema import (
    ReferenceEmbeddingSchema,
    ReferenceManifest,
    SCHEMA,
    export_reference_outputs,
    load_reference_outputs,
    validate_output_integrity,
    create_manifest,
)
from stagebridge.reference.pipeline import (
    ReferenceGeometryConfig,
    ReferenceGeometryResult,
    run_reference_pipeline,
    run_smoke_test,
)

__all__ = [
    # Legacy exports (hlca_mapper)
    "HLCAReference",
    "HLCAMappingResult",
    "load_hlca_reference_legacy",
    "run_active_reference_latent",
    "LatentStore",
    "summarize_latent",
    "stage_preservation_diagnostics",
    "donor_leakage_diagnostics",
    "gene_overlap_diagnostics",
    "reference_alignment_gate",
    # Loaders
    "LoadedReference",
    "ReferenceInfo",
    "FeatureOverlapReport",
    "load_hlca_reference",
    "load_luca_reference",
    "validate_reference",
    "compute_feature_overlap",
    # Mapping
    "MappingResult",
    "ReferenceNeighborhood",
    "map_to_hlca",
    "map_to_luca",
    # Fusion
    "FusedEmbeddingResult",
    "fuse_dual_reference",
    "fuse_single_reference",
    # Confidence
    "ConfidenceScores",
    "compute_hlca_confidence",
    "compute_luca_confidence",
    "compute_dual_confidence",
    "detect_mapping_collapse",
    "detect_nan_embeddings",
    # Schema
    "ReferenceEmbeddingSchema",
    "ReferenceManifest",
    "SCHEMA",
    "export_reference_outputs",
    "load_reference_outputs",
    "validate_output_integrity",
    "create_manifest",
    # Pipeline
    "ReferenceGeometryConfig",
    "ReferenceGeometryResult",
    "run_reference_pipeline",
    "run_smoke_test",
]
