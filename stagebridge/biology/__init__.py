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
]
