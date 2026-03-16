"""Publication-quality visualization components for StageBridge."""

from .figure_generation import (
    generate_figure1_architecture,
    generate_figure2_dimensionality_reduction,
    generate_figure3_niche_influence_biology,
    generate_figure4_model_performance,
    generate_figure5_attention_patterns,
)
from .individual_plots import (
    plot_confusion_matrix,
    plot_loss_curve,
    plot_pca_with_variance,
    plot_tsne,
    plot_umap,
)
from .plot_cache import clear_cache, get_cache
from .professional_figures import (
    generate_figure2_dimensionality_reduction as generate_fig2_pro,
    generate_figure4_model_performance as generate_fig4_pro,
)

__all__ = [
    # Figure generation
    "generate_figure1_architecture",
    "generate_figure2_dimensionality_reduction",
    "generate_figure3_niche_influence_biology",
    "generate_figure4_model_performance",
    "generate_figure5_attention_patterns",
    # Individual plots
    "plot_confusion_matrix",
    "plot_loss_curve",
    "plot_pca_with_variance",
    "plot_tsne",
    "plot_umap",
    # Cache
    "clear_cache",
    "get_cache",
    # Professional figures
    "generate_fig2_pro",
    "generate_fig4_pro",
]
