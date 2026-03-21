"""
Biological interpretation and analysis tools for StageBridge.

Modules:
- explainability: SHAP analysis, sender influence, attention visualization
- biological_interpretation: Niche influence extraction and pathway signatures
- transformer_analysis: Attention pattern analysis

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
]
