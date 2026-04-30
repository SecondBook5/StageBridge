"""Translational genomic interpretation layer for StageBridge.

This module integrates WES-derived germline/somatic variants with StageBridge
transition fields for post-hoc translational interpretation.

IMPORTANT SCIENTIFIC FRAMING:
- WES defines the high-confidence variant truth set
- ACMG/AMP-aligned interpretation for germline pathogenicity
- OncoKB/CIViC-style interpretation for somatic actionability
- Spatial/snRNA data localizes expressed variant evidence only
- StageBridge tests whether high-transition niches are enriched
  for clinically interpretable genomic features

This is a POST-HOC interpretation layer. Genomic annotations are NOT used
as model supervision unless explicitly configured.
"""

from stagebridge.genomics.schemas import (
    VariantRecord,
    GermlineAnnotation,
    SomaticActionability,
    ClonalityEstimate,
    SpatialVariantEvidence,
    TransitionGenomicEnrichment,
)

__all__ = [
    # Schemas
    "VariantRecord",
    "GermlineAnnotation",
    "SomaticActionability",
    "ClonalityEstimate",
    "SpatialVariantEvidence",
    "TransitionGenomicEnrichment",
]
