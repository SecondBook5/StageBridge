"""Publication-quality theme and styling utilities for StageBridge figures.

This module provides a centralized configuration system for generating
publication-ready figures with consistent styling across all plot types.

Key Features:
- Pure white backgrounds (#FFFFFF) for publication
- 300 DPI for saved figures
- Colorblind-friendly stage palette
- Top/right spines removed
- Proper font sizes (10-14pt)
- Multi-format export (PNG, PDF, SVG)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)

# Publication-grade color palette
# Now uses LungPCA colors for consistency with original paper
# Import the canonical colors from lungpca_style
from .lungpca_style import STAGE_COLORS as _LUNGPCA_STAGE_COLORS

PUBLICATION_PALETTE = {
    # Stage colors (from LungPCA paper - Peng et al.)
    "Normal": _LUNGPCA_STAGE_COLORS["Normal"],      # #33a02c green
    "AAH": _LUNGPCA_STAGE_COLORS["AAH"],            # #b2df8a light green
    "AIS": _LUNGPCA_STAGE_COLORS["AIS"],            # #fdbf6f light orange
    "MIA": _LUNGPCA_STAGE_COLORS["MIA"],            # #fb9a99 pink
    "LUAD": _LUNGPCA_STAGE_COLORS["LUAD"],          # #ff7f00 orange
    "Unknown": _LUNGPCA_STAGE_COLORS.get("Unknown", "#d9d9d9"),  # gray
    # Utility colors
    "ink": "#000000",  # Pure black for text
    "grid": "#CCCCCC",  # Light gray for grid
    "background": "#FFFFFF",  # Pure white background
}


def configure_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets rcParams for:
    - Pure white backgrounds (#FFFFFF)
    - 300 DPI output
    - Readable font sizes (10-14pt)
    - Bold axis labels
    - Top/right spines removed
    - Consistent legend styling
    """
    mpl.rcParams.update(
        {
            # Background colors (pure white for publication)
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            # DPI settings
            "savefig.dpi": 300,
            "figure.dpi": 150,
            # Fonts (readable at column width)
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            # Font weights
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            # Colors
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "text.color": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            # Spines (remove top and right for cleaner look)
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 1.5,
            # Ticks
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Grid
            "grid.color": "#CCCCCC",
            "grid.alpha": 0.3,
            "grid.linewidth": 0.8,
            "grid.linestyle": ":",
            "axes.grid": False,  # Off by default, enable per plot
            # Legend
            "legend.framealpha": 1.0,
            "legend.edgecolor": "#666666",
            "legend.fancybox": False,
            "legend.shadow": False,
            # Saving
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.transparent": False,
            "savefig.format": "png",
            # PDF-specific settings
            "pdf.fonttype": 42,  # TrueType fonts for editability
            "ps.fonttype": 42,
            # SVG-specific settings
            "svg.fonttype": "none",  # Embed fonts as paths
        }
    )
    log.info("Publication style configured (300 DPI, white background, 10-14pt fonts)")


def save_publication_figure(
    fig: plt.Figure,
    output_path: Path | str,
    formats: list[str] | None = None,
    dpi: int = 300,
    transparent: bool = False,
) -> dict[str, Path]:
    """Save figure in multiple publication-ready formats.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure object to save
    output_path : Path or str
        Base output path (without extension)
    formats : list of str, optional
        List of formats to save (default: ["png", "pdf", "svg"])
    dpi : int
        Resolution for raster formats (default: 300)
    transparent : bool
        Whether to use transparent background (default: False)

    Returns
    -------
    saved_paths : dict
        Dictionary mapping format to saved file path

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> paths = save_publication_figure(fig, "output/figure1")
    >>> # Saves: figure1.png, figure1.pdf, figure1.svg
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension if provided
    if output_path.suffix:
        output_path = output_path.with_suffix("")

    saved_paths = {}

    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        try:
            fig.savefig(
                save_path,
                dpi=dpi if fmt in ["png", "jpg", "jpeg"] else None,
                bbox_inches="tight",
                facecolor="white" if not transparent else "none",
                transparent=transparent,
                format=fmt,
            )
            saved_paths[fmt] = save_path
            log.info(f"Saved {fmt.upper()}: {save_path}")
        except Exception as e:
            log.warning(f"Failed to save {fmt.upper()} format: {e}")

    return saved_paths


def get_stage_color(stage: str) -> str:
    """Get colorblind-friendly color for a lung cancer stage.

    Parameters
    ----------
    stage : str
        Stage name (Normal, AAH, AIS, MIA, LUAD, or Unknown)

    Returns
    -------
    color : str
        Hex color code
    """
    return PUBLICATION_PALETTE.get(str(stage), PUBLICATION_PALETTE["Unknown"])


def apply_clean_spines(ax: plt.Axes) -> None:
    """Remove top and right spines for cleaner publication figures.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to modify
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def add_clean_legend(
    ax: plt.Axes,
    title: str | None = None,
    loc: str = "best",
    framealpha: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Add publication-quality legend with clean styling.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    title : str, optional
        Legend title
    loc : str
        Legend location (default: "best")
    framealpha : float
        Legend background opacity (default: 1.0)
    **kwargs
        Additional arguments passed to ax.legend()

    Returns
    -------
    legend : Legend
        Matplotlib legend object
    """
    legend = ax.legend(
        title=title,
        loc=loc,
        framealpha=framealpha,
        edgecolor="#666666",
        fancybox=False,
        shadow=False,
        **kwargs,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(1.5)
    return legend


def create_figure(
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    layout: str = "tight",
) -> tuple[plt.Figure, plt.Axes]:
    """Create publication-ready figure with clean styling.

    Parameters
    ----------
    figsize : tuple of float
        Figure size in inches (width, height)
    dpi : int
        Display DPI (default: 150, saved at 300)
    layout : str
        Layout engine ("tight", "constrained", or None)

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    ax : Axes
        Matplotlib axes object

    Examples
    --------
    >>> fig, ax = create_figure(figsize=(10, 8))
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_publication_figure(fig, "output/figure1")
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, layout=layout)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    apply_clean_spines(ax)
    return fig, ax


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    layout: str = "tight",
    **kwargs: Any,
) -> tuple[plt.Figure, Any]:
    """Create publication-ready figure with multiple subplots.

    Parameters
    ----------
    nrows : int
        Number of subplot rows
    ncols : int
        Number of subplot columns
    figsize : tuple of float, optional
        Figure size in inches (width, height). If None, auto-calculated.
    dpi : int
        Display DPI (default: 150, saved at 300)
    layout : str
        Layout engine ("tight", "constrained", or None)
    **kwargs
        Additional arguments passed to plt.subplots()

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    axes : Axes or array of Axes
        Matplotlib axes object(s)
    """
    if figsize is None:
        # Auto-calculate reasonable size
        figsize = (4 * ncols + 1, 4 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, layout=layout, **kwargs
    )
    fig.patch.set_facecolor("white")

    # Apply clean styling to all axes
    if isinstance(axes, plt.Axes):
        axes.set_facecolor("white")
        apply_clean_spines(axes)
    else:
        for ax in axes.flat:
            ax.set_facecolor("white")
            apply_clean_spines(ax)

    return fig, axes


# Convenience function for quick setup
def setup_publication_plotting() -> None:
    """One-line setup for publication-quality plotting.

    Call this at the start of a script or notebook to configure
    all matplotlib settings for publication figures.
    """
    configure_publication_style()
    log.info("Publication plotting ready (use create_figure() or create_subplots())")
