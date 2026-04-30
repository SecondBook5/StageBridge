"""Biological validation and discovery layer for StageBridge.

This module evaluates whether the model:
1. Recovers known biology (IL1B-IL1R1, KAC progenitors, CAF interactions)
2. Discovers novel biology via attention patterns
3. Clearly separates supported findings from speculative hypotheses
4. Identifies clinically actionable mutations (ACMG/OncoKB)

Key modules:
- lr_scoring: Attention-weighted ligand-receptor interaction analysis
- known_biology: Validation against established mechanisms (Peng et al.)
- novel_discovery: Hypothesis generation from unexplained attention patterns
- validation_report: Publication-ready validation summaries
- clinical_actionability: ACMG/OncoKB variant annotation and therapy prioritization
"""

from stagebridge.biology.lr_scoring import (
    LRPair,
    LRScoreResult,
    compute_attention_weighted_lr_scores,
    aggregate_lr_scores_by_stage,
    identify_stage_specific_interactions,
)
from stagebridge.biology.known_biology import (
    KnownMechanism,
    ValidationResult,
    validate_known_mechanisms,
    compute_mechanism_recovery_score,
    KNOWN_MECHANISMS,
)
from stagebridge.biology.novel_discovery import (
    NovelHypothesis,
    DiscoveryResult,
    generate_novel_hypotheses,
    rank_hypotheses_by_confidence,
    filter_spurious_associations,
)
from stagebridge.biology.validation_report import (
    BiologyValidationReport,
    generate_validation_report,
    export_for_publication,
)
from stagebridge.biology.clinical_actionability import (
    ACMGClassification,
    OncoKBLevel,
    MutationType,
    ClinicalVariant,
    ActionabilityReport,
    annotate_variants,
    generate_actionability_report,
    prioritize_therapies,
    check_resistance_mutations,
    export_actionability_table,
    LUNG_CANCER_ONCOKB_DATABASE,
    GERMLINE_ACMG_DATABASE,
)
from stagebridge.biology.germline_niche import (
    GermlineCausalDiscovery,
    discover_trajectory_divergence,
    discover_attention_gradient_features,
    discover_velocity_modifiers,
    discover_intervention_targets,
    run_germline_causal_discovery,
)

__all__ = [
    # L-R scoring
    "LRPair",
    "LRScoreResult",
    "compute_attention_weighted_lr_scores",
    "aggregate_lr_scores_by_stage",
    "identify_stage_specific_interactions",
    # Known biology validation
    "KnownMechanism",
    "ValidationResult",
    "validate_known_mechanisms",
    "compute_mechanism_recovery_score",
    "KNOWN_MECHANISMS",
    # Novel discovery
    "NovelHypothesis",
    "DiscoveryResult",
    "generate_novel_hypotheses",
    "rank_hypotheses_by_confidence",
    "filter_spurious_associations",
    # Validation report
    "BiologyValidationReport",
    "generate_validation_report",
    "export_for_publication",
    # Clinical actionability (ACMG/OncoKB)
    "ACMGClassification",
    "OncoKBLevel",
    "MutationType",
    "ClinicalVariant",
    "ActionabilityReport",
    "annotate_variants",
    "generate_actionability_report",
    "prioritize_therapies",
    "check_resistance_mutations",
    "export_actionability_table",
    "LUNG_CANCER_ONCOKB_DATABASE",
    "GERMLINE_ACMG_DATABASE",
    # Germline-niche causal discovery
    "GermlineCausalDiscovery",
    "discover_trajectory_divergence",
    "discover_attention_gradient_features",
    "discover_velocity_modifiers",
    "discover_intervention_targets",
    "run_germline_causal_discovery",
]
