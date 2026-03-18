"""
Biological interpretation and analysis tools for StageBridge.

Modules:
- explainability: SHAP analysis, sender influence, attention visualization
- biological_interpretation: Gene programs, pathway analysis
- transformer_analysis: Attention pattern analysis
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

__all__ = [
    "BaselineSHAPAnalyzer",
    "SenderInfluenceAnalyzer",
    "AttentionAnalyzer",
    "SHAPResult",
    "SenderInfluenceResult",
    "plot_sender_influence_heatmap",
    "compare_baseline_importance",
]
