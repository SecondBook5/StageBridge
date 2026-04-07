"""
Publication-grade visualization for StageBridge.

Generates figures meeting Nature Methods standards:
- Vector formats (PDF/SVG) + high-res PNG
- Consistent typography and color palette
- Proper dimensions and spacing
- Statistical annotations
"""

from .style import (
    NatureStyle,
    apply_nature_style,
    get_color_palette,
    save_figure,
    SINGLE_COL_WIDTH,
    DOUBLE_COL_WIDTH,
)
from .benchmark_figures import (
    SpatialBenchmarkFigures,
    generate_all_benchmark_figures,
)

__all__ = [
    "NatureStyle",
    "apply_nature_style",
    "get_color_palette",
    "save_figure",
    "SINGLE_COL_WIDTH",
    "DOUBLE_COL_WIDTH",
    "SpatialBenchmarkFigures",
    "generate_all_benchmark_figures",
]
