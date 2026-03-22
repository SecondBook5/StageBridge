"""Reference mapping and latent-alignment utilities.

This package provides the dual-reference geometry layer for StageBridge,
enabling query cells to be mapped to both HLCA (healthy reference) and
LuCa (disease-aware reference) coordinate systems.

Key modules:
- loaders: Reference loading and validation
- map_query: Query-to-reference mapping (in-memory)
- map_query_chunked: Memory-efficient chunked mapping (HPC/large refs)
- fuse: Dual-reference fusion
- confidence: Confidence scoring and quality metrics
- schema: Standardized output schemas
- diagnose_reference: Latent integrity checking tool
- pipeline: Main pipeline integration

Reference Sources:
- HLCA: CZI cellxgene via scvi-tools HF Hub, latent_key='X_scanvi_emb' (30 dims)
- LuCA: Zenodo, latent_key='X_scVI' (10 dims) - USE CORE VERSION (not Extended)

See stagebridge/reference/README.md for full documentation.
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
# k-NN fallback mapping (use hlca_mapper/luca_mapper for model-based)
from stagebridge.reference.map_query import (
    map_to_hlca,
    map_to_luca,
)
from stagebridge.reference.map_query_chunked import (
    map_query_chunked,
    map_to_dual_reference_chunked,
)
from stagebridge.reference.diagnose_reference import (
    diagnose_latent_integrity,
    clean_reference_latent,
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
    MappingResult,
    ReferenceNeighborhood,
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
    # Chunked mapping (HPC)
    "map_query_chunked",
    "map_to_dual_reference_chunked",
    # Diagnostics
    "diagnose_latent_integrity",
    "clean_reference_latent",
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
