"""Data schemas for genomic interpretation layer.

These dataclasses define the structure for variant records, annotations,
and enrichment results used throughout the genomics module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Type aliases for classification categories
ACMGClassification = Literal[
    "pathogenic",
    "likely_pathogenic",
    "vus",
    "likely_benign",
    "benign",
    "unknown",
]

OncogenicityLevel = Literal[
    "oncogenic",
    "likely_oncogenic",
    "predicted_oncogenic",
    "likely_neutral",
    "inconclusive",
    "unknown",
]

ActionabilityLevel = Literal[
    "level_1",  # FDA-approved, standard of care
    "level_2",  # Standard of care, compelling evidence
    "level_3A",  # Clinical evidence, investigational
    "level_3B",  # Clinical evidence, another tumor type
    "level_4",  # Biological evidence
    "level_R1",  # Resistance, standard of care
    "level_R2",  # Resistance, investigational
    "gene_level_cancer_relevance",  # Gene is cancer-relevant but no variant-level evidence
    "not_actionable",
    "unknown",
]

ClonalityLabel = Literal[
    "clonal",
    "clonal_like",
    "intermediate",
    "subclonal",
    "subclonal_like",
    "low_confidence",
    "unknown",
]

EvidenceLabel = Literal[
    "alt_supported",
    "weak_alt_evidence",
    "ref_only_observed",
    "no_coverage",
    "low_coverage",
]


@dataclass
class VariantRecord:
    """A variant record from WES or annotated variant table.

    This represents the WES truth set. Variants are identified by WES,
    not discovered from spatial/snRNA data.
    """
    sample_id: str
    donor_id: str
    chromosome: str
    position: int
    reference_allele: str
    alternate_allele: str
    gene: str | None = None
    transcript: str | None = None
    consequence: str | None = None
    protein_change: str | None = None
    variant_type: str = "SNV"
    source: str = "WES"
    is_germline: bool | None = None
    is_somatic: bool | None = None
    tumor_vaf: float | None = None
    normal_vaf: float | None = None
    tumor_depth: int | None = None
    normal_depth: int | None = None
    quality: float | None = None
    filter_status: str | None = None

    @property
    def variant_id(self) -> str:
        """Normalized variant identifier."""
        return f"{self.chromosome}:{self.position}:{self.reference_allele}:{self.alternate_allele}"

    def __post_init__(self):
        if not self.chromosome:
            raise ValueError("chromosome cannot be empty")
        if self.position <= 0:
            raise ValueError("position must be positive")
        if not self.reference_allele:
            raise ValueError("reference_allele cannot be empty")
        if not self.alternate_allele:
            raise ValueError("alternate_allele cannot be empty")


@dataclass
class GermlineAnnotation:
    """ACMG/AMP-aligned germline pathogenicity annotation.

    IMPORTANT: These are ACMG-aligned interpretations, not clinical-grade
    ACMG adjudications. Automated output requires clinical genetics review.
    """
    variant_id: str
    gene: str
    clinvar_significance: str | None = None
    acmg_aligned_classification: ACMGClassification = "unknown"
    acmg_evidence_codes: list[str] = field(default_factory=list)
    inheritance: str | None = None
    cancer_predisposition_gene: bool = False
    notes: str = ""

    def __post_init__(self):
        if not self.notes:
            self.notes = (
                "ACMG/AMP-aligned interpretation. Not a clinical-grade ACMG "
                "classification. Requires review by clinical genetics."
            )


@dataclass
class SomaticActionability:
    """OncoKB/CIViC-style somatic actionability annotation.

    Uses cancer-specific actionability frameworks, NOT ACMG.
    """
    variant_id: str
    gene: str
    oncogenicity: OncogenicityLevel = "unknown"
    actionability_level: ActionabilityLevel = "unknown"
    therapeutic_implication: str | None = None
    diagnostic_implication: str | None = None
    prognostic_implication: str | None = None
    resistance_implication: str | None = None
    knowledgebase_source: str = "none"
    notes: str = ""


@dataclass
class ClonalityEstimate:
    """Clonality estimate from WES variant allele fractions.

    When purity/CNV/ploidy are available, computes cancer cell fraction.
    Otherwise uses naive VAF thresholds with explicit uncertainty labeling.
    """
    variant_id: str
    sample_id: str
    tumor_vaf: float
    local_copy_number: float | None = None
    purity: float | None = None
    ploidy: float | None = None
    cancer_cell_fraction: float | None = None
    clonality_label: ClonalityLabel = "unknown"
    confidence: str = "low"
    method: str = "naive_vaf"
    notes: str = ""

    def __post_init__(self):
        if self.method == "naive_vaf" and not self.notes:
            self.notes = (
                "Clonality estimated from naive VAF thresholds without "
                "purity/CNV correction. Approximate only."
            )


@dataclass
class SpatialVariantEvidence:
    """Expressed variant evidence from spatial/snRNA data.

    CRITICAL: This does NOT call new variants. It only counts evidence
    for WES-confirmed variants in RNA-based data. Absence of alternate
    reads does NOT mean the mutation is absent - RNA coverage is sparse.
    """
    variant_id: str
    sample_id: str
    barcode: str
    x: float | None = None
    y: float | None = None
    ref_count: int = 0
    alt_count: int = 0
    total_count: int = 0
    expressed_alt_fraction: float | None = None
    evidence_label: EvidenceLabel = "no_coverage"
    caution: str = ""

    def __post_init__(self):
        self.total_count = self.ref_count + self.alt_count
        if self.total_count > 0:
            self.expressed_alt_fraction = self.alt_count / self.total_count
        if not self.caution:
            self.caution = (
                "RNA coverage absence is NOT mutation absence. "
                "Expression-based evidence only."
            )


@dataclass
class TransitionGenomicEnrichment:
    """Enrichment test result for genomic features in transition zones.

    Tests whether high-transition niches are enriched for clinically
    interpretable genomic risk/actionability features.
    """
    sample_id: str
    donor_id: str
    comparison: str  # e.g., "high_transition_vs_low"
    feature_name: str  # e.g., "TP53_mutated", "actionable_variant"
    high_transition_mean: float
    low_transition_mean: float
    effect_size: float
    p_value: float
    q_value: float = 1.0
    n_high: int = 0
    n_low: int = 0
    test_method: str = "mann_whitney_u"
    notes: str = ""
