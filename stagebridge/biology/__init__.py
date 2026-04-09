"""
StageBridge Biology Module.

Provides biological interpretation and clinical relevance tools:
- Gene signatures: EMT, CAF, immune, proliferation, lung-specific
- Pathway scoring: Score cells against curated gene programs
- Niche-biology associations: Link model outputs to biological processes
- Clinical relevance: Connect findings to clinical outcomes
"""

from .signatures import (
    GENE_SIGNATURES,
    EMT_SIGNATURES,
    CAF_SIGNATURES,
    IMMUNE_SIGNATURES,
    LUNG_CANCER_SIGNATURES,
    PROLIFERATION_SIGNATURE,
    score_signature,
    score_all_signatures,
    get_signature_genes,
)

from .pathway_analysis import (
    compute_pathway_activity,
    run_enrichment_analysis,
    compare_pathway_activity_by_stage,
    identify_stage_specific_pathways,
)

from .niche_biology import (
    correlate_niche_influence_with_biology,
    identify_biological_drivers,
    compute_niche_pathway_associations,
    generate_biological_hypotheses,
)

from .plots import (
    plot_signature_scores_by_stage,
    plot_niche_biology_heatmap,
    plot_pathway_activity_ridge,
    plot_emt_caf_immune_triangle,
    plot_stage_pathway_radar,
    plot_biological_summary_panel,
)

from .clinical import (
    compute_risk_scores,
    stratify_by_niche_phenotype,
    generate_clinical_summary,
)

from .attention_lr_scoring import (
    LR_PRIORS,
    LRInteractionScore,
    NicheEcosystemSummary,
    compute_attention_weighted_lr_scores,
    aggregate_lr_scores_by_stage,
    identify_stage_specific_interactions,
    compute_il1b_axis_score,
    generate_niche_ecosystem_summary,
    create_lr_interaction_report,
    export_lr_scores_for_visualization,
)

from .intervention_targets import (
    DRUGGABILITY_DATABASE,
    InterventionTarget,
    NicheRiskScore,
    InterventionPlan,
    prioritize_intervention_targets,
    compute_niche_level_risk,
    aggregate_niche_risks_by_region,
    generate_intervention_plan,
    export_intervention_report,
)

from .dynamic_driver_index import (
    DriverIndexResult,
    compute_dynamic_driver_index,
    compute_driver_index_along_trajectory,
    compute_driver_index_efficient,
    analyze_luad_progression_drivers,
    LUAD_STAGES,
)

__all__ = [
    # Signatures
    "GENE_SIGNATURES",
    "EMT_SIGNATURES",
    "CAF_SIGNATURES",
    "IMMUNE_SIGNATURES",
    "LUNG_CANCER_SIGNATURES",
    "PROLIFERATION_SIGNATURE",
    "score_signature",
    "score_all_signatures",
    "get_signature_genes",
    # Pathway analysis
    "compute_pathway_activity",
    "run_enrichment_analysis",
    "compare_pathway_activity_by_stage",
    "identify_stage_specific_pathways",
    # Niche-biology
    "correlate_niche_influence_with_biology",
    "identify_biological_drivers",
    "compute_niche_pathway_associations",
    "generate_biological_hypotheses",
    # Plots
    "plot_signature_scores_by_stage",
    "plot_niche_biology_heatmap",
    "plot_pathway_activity_ridge",
    "plot_emt_caf_immune_triangle",
    "plot_stage_pathway_radar",
    "plot_biological_summary_panel",
    # Clinical
    "compute_risk_scores",
    "stratify_by_niche_phenotype",
    "generate_clinical_summary",
    # Attention-weighted L-R scoring (novel)
    "LR_PRIORS",
    "LRInteractionScore",
    "NicheEcosystemSummary",
    "compute_attention_weighted_lr_scores",
    "aggregate_lr_scores_by_stage",
    "identify_stage_specific_interactions",
    "compute_il1b_axis_score",
    "generate_niche_ecosystem_summary",
    "create_lr_interaction_report",
    "export_lr_scores_for_visualization",
    # Intervention targets (novel)
    "DRUGGABILITY_DATABASE",
    "InterventionTarget",
    "NicheRiskScore",
    "InterventionPlan",
    "prioritize_intervention_targets",
    "compute_niche_level_risk",
    "aggregate_niche_risks_by_region",
    "generate_intervention_plan",
    "export_intervention_report",
    # Dynamic driver index (GeoBridge-inspired)
    "DriverIndexResult",
    "compute_dynamic_driver_index",
    "compute_driver_index_along_trajectory",
    "compute_driver_index_efficient",
    "analyze_luad_progression_drivers",
    "LUAD_STAGES",
]
