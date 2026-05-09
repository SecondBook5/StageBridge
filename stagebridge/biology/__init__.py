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
- niche_discovery: NMF-based niche archetype discovery from deconvolution
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
from stagebridge.biology.progression import (
    compute_cytotrace,
    compute_diffusion_pseudotime,
    compute_progression_scores,
    load_progression_scores,
)
from stagebridge.biology.features import (
    EMT_MESENCHYMAL,
    EMT_EPITHELIAL,
    SENESCENCE_CORE,
    SASP_GENES,
    AP1_TF_GENES,
    AP1_TARGET_GENES,
    compute_emt_score,
    compute_senescence_score,
    compute_sasp_score,
    compute_ap1_score,
    compute_ap1_target_score,
    compute_all_signatures,
    run_liana,
    extract_il1b_interactions,
    run_liana_pathway_enrichment,
    compute_cell_lr_scores,
    compute_biological_features,
)
from stagebridge.biology.tf_pathway_activity import (
    compute_tf_activity,
    compute_pathway_activity,
    compute_hallmark_activity,
    rank_by_progression,
    rank_by_group,
    compute_pseudobulk_activity,
    compute_spatial_activity,
)
from stagebridge.biology.gsea import (
    run_gsea_prerank,
    run_gsea_from_de,
    run_gsea_all_stages,
    run_gsea,
    HALLMARK_PATHWAYS,
)
from stagebridge.biology.trajectories import (
    compute_diffusion_map,
    compute_paga,
    run_trajectories,
)
from stagebridge.biology.niche_discovery import (
    NicheArchetype,
    NicheDiscoveryResult,
    discover_niches,
    select_n_archetypes,
    plot_niche_discovery,
    run_niche_discovery,
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
    # Progression scoring
    "compute_cytotrace",
    "compute_diffusion_pseudotime",
    "compute_progression_scores",
    "load_progression_scores",
    # Biological features (EMT, senescence, AP-1, LIANA)
    "EMT_MESENCHYMAL",
    "EMT_EPITHELIAL",
    "SENESCENCE_CORE",
    "SASP_GENES",
    "AP1_TF_GENES",
    "AP1_TARGET_GENES",
    "compute_emt_score",
    "compute_senescence_score",
    "compute_sasp_score",
    "compute_ap1_score",
    "compute_ap1_target_score",
    "compute_all_signatures",
    "run_liana",
    "extract_il1b_interactions",
    "run_liana_pathway_enrichment",
    "compute_cell_lr_scores",
    "compute_biological_features",
    # TF/Pathway activity (decoupleR)
    "compute_tf_activity",
    "compute_pathway_activity",
    "compute_hallmark_activity",
    "rank_by_progression",
    "rank_by_group",
    "compute_pseudobulk_activity",
    "compute_spatial_activity",
    # GSEA
    "run_gsea_prerank",
    "run_gsea_from_de",
    "run_gsea_all_stages",
    "run_gsea",
    "HALLMARK_PATHWAYS",
    # Trajectories
    "compute_diffusion_map",
    "compute_paga",
    "run_trajectories",
    # Niche discovery (NMF on deconvolution)
    "NicheArchetype",
    "NicheDiscoveryResult",
    "discover_niches",
    "select_n_archetypes",
    "plot_niche_discovery",
    "run_niche_discovery",
]
