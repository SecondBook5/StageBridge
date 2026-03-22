"""
Biological interpretation and analysis tools for StageBridge.

Modules:
- explainability: SHAP analysis, sender influence, attention visualization
- biological_interpretation: Niche influence extraction and pathway signatures
- transformer_analysis: Attention pattern analysis
- biology_paper_outputs: Full analysis pipeline for biology paper
- biology_paper_figures: Publication-quality figures for biology paper

For comprehensive biological analysis, use the stagebridge.biology module which provides:
- Gene signatures (EMT, CAF, immune, lung cancer-specific)
- Pathway activity analysis
- Niche-biology associations
- Clinical relevance tools
"""

from .explainability import (
    BaselineSHAPAnalyzer,
    SenderInfluenceAnalyzer,
    AttentionAnalyzer,
    SHAPResult,
    SenderInfluenceResult,
    plot_sender_influence_heatmap,
    compare_baseline_importance,
)

from .biological_interpretation import (
    InfluenceTensorExtractor,
    visualize_niche_influence,
    extract_pathway_signatures,
    generate_biological_summary,
)

from .biology_paper_outputs import (
    compute_cell_progression_risk,
    compute_niche_risk_scores,
    identify_proinflammatory_niches,
    score_kac_alveolar_progenitor_state,
    perturbation_analysis,
    compute_stage_ecosystem_summary,
    compute_donor_consistency,
    run_donor_consistency_tests,
    generate_biology_paper_report,
    run_biology_paper_analysis,
    CellProgressionRisk,
    NicheRiskAssessment,
    StageEcosystemSummary,
    KAC_MARKERS,
    PROINFLAMMATORY_MACROPHAGE_MARKERS,
    IL1B_PATHWAY_GENES,
    CAF_MARKERS,
    EMT_MARKERS,
)

from .biology_paper_figures import (
    generate_all_biology_figures,
    plot_progression_risk_by_stage,
    plot_niche_ecosystem_comparison,
    plot_kac_vs_niche_risk,
    plot_fold_change_heatmap,
    plot_proinflammatory_enrichment_trajectory,
    plot_perturbation_effects,
)

__all__ = [
    # Explainability
    "BaselineSHAPAnalyzer",
    "SenderInfluenceAnalyzer",
    "AttentionAnalyzer",
    "SHAPResult",
    "SenderInfluenceResult",
    "plot_sender_influence_heatmap",
    "compare_baseline_importance",
    # Biological interpretation
    "InfluenceTensorExtractor",
    "visualize_niche_influence",
    "extract_pathway_signatures",
    "generate_biological_summary",
    # Biology paper outputs
    "compute_cell_progression_risk",
    "compute_niche_risk_scores",
    "identify_proinflammatory_niches",
    "score_kac_alveolar_progenitor_state",
    "perturbation_analysis",
    "compute_stage_ecosystem_summary",
    "compute_donor_consistency",
    "run_donor_consistency_tests",
    "generate_biology_paper_report",
    "run_biology_paper_analysis",
    "CellProgressionRisk",
    "NicheRiskAssessment",
    "StageEcosystemSummary",
    "KAC_MARKERS",
    "PROINFLAMMATORY_MACROPHAGE_MARKERS",
    "IL1B_PATHWAY_GENES",
    "CAF_MARKERS",
    "EMT_MARKERS",
    # Biology paper figures
    "generate_all_biology_figures",
    "plot_progression_risk_by_stage",
    "plot_niche_ecosystem_comparison",
    "plot_kac_vs_niche_risk",
    "plot_fold_change_heatmap",
    "plot_proinflammatory_enrichment_trajectory",
    "plot_perturbation_effects",
]
