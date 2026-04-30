"""Publication-quality theme for StageBridge figures.

Provides consistent styling for all plots:
- Pure white backgrounds for publication
- 300 DPI output
- Colorblind-friendly stage palette (LungPCA paper colors)
- Top/right spines removed
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Stage colors - deep, rich hues
STAGE_COLORS = {
    "Normal": "#228B22",      # Forest green
    "AAH": "#50C878",         # Emerald
    "AIS": "#DAA520",         # Gold
    "MIA": "#CD5C5C",         # Indian red
    "LUAD": "#8B0000",        # Dark red / brick
    "Preinvasive": "#4169E1", # Royal blue (combined)
    "Invasive": "#8B0000",    # Dark red / brick
    "Unknown": "#696969",     # Dim gray
}

# Cell type colors - deep, rich palette
CELLTYPE_COLORS = {
    # HLCA coarse types
    "AT2": "#4B0082",         # Indigo
    "AT1": "#191970",         # Midnight blue
    "Basal": "#000080",       # Navy
    "Club": "#6B8E23",        # Olive drab
    "Ciliated": "#DAA520",    # Gold
    "Macrophages": "#228B22", # Forest green
    "Fibroblast lineage": "#8B4513",  # Saddle brown
    "T cell lineage": "#4169E1",      # Royal blue
    "Capillary": "#800000",   # Maroon
    "Mast cells": "#9932CC",  # Dark orchid
    "Secretory": "#9400D3",   # Dark violet
    # LuCA fine-grained
    "pulmonary alveolar type 2 cell": "#4B0082",  # Indigo
    "pulmonary alveolar type 1 cell": "#191970",  # Midnight blue
    "capillary endothelial cell": "#800000",      # Maroon
    "fibroblast of lung": "#8B4513",              # Saddle brown
    "malignant cell": "#8B0000",                  # Dark red
    "alveolar macrophage": "#228B22",             # Forest green
    "CD8-positive, alpha-beta T cell": "#4169E1", # Royal blue
    "CD4-positive, alpha-beta T cell": "#6495ED", # Cornflower blue
    "natural killer cell": "#483D8B",             # Dark slate blue
    "plasma cell": "#9932CC",                     # Dark orchid
    "mast cell": "#9400D3",                       # Dark violet
    "smooth muscle cell": "#2F4F4F",              # Dark slate gray
    "vein endothelial cell": "#DC143C",           # Crimson
    "club cell": "#6B8E23",                       # Olive drab
    "epithelial cell of lung": "#DDA0DD",         # Plum / lilac
}


def configure_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams.update({
        # Background (pure white)
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        # DPI
        "savefig.dpi": 300,
        "figure.dpi": 150,
        # Fonts
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Font weights
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        # Spines (remove top/right)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        # Ticks
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid (off by default)
        "axes.grid": False,
        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.5,
    })


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: list[str] = None,
    dpi: int = 300,
) -> list[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_dir: Output directory
        name: Base filename (without extension)
        formats: List of formats (default: ['png', 'pdf'])
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="#FFFFFF")
        paths.append(path)

    return paths
