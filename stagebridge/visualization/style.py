"""
publication publication style configuration.

Standards:
- Single column: 88mm (3.46 in)
- Double column: 180mm (7.09 in)
- Font: Arial/Helvetica, 6-8pt
- Line width: 0.5-1pt
- Colors: Colorblind-friendly palette
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# publication dimensions (in inches)
SINGLE_COL_WIDTH = 3.46  # 88mm
DOUBLE_COL_WIDTH = 7.09  # 180mm
MAX_HEIGHT = 9.45  # 240mm

# DPI for raster outputs
PRINT_DPI = 300
SCREEN_DPI = 150


@dataclass
class NatureStyle:
    """
    Publication style configuration.

    Attributes:
        width: Figure width in inches
        font_size: Base font size in points
        font_family: Font family (sans-serif for Nature)
        linewidth: Default line width
        palette: Color palette name or list of colors
    """

    width: float = SINGLE_COL_WIDTH
    font_size: float = 7
    font_family: str = "Arial"
    linewidth: float = 0.75
    palette: str = "colorblind"

    # Derived sizes
    title_size: float = field(init=False)
    label_size: float = field(init=False)
    tick_size: float = field(init=False)
    legend_size: float = field(init=False)

    def __post_init__(self):
        self.title_size = self.font_size + 1
        self.label_size = self.font_size
        self.tick_size = self.font_size - 1
        self.legend_size = self.font_size - 1


# Colorblind-friendly palettes
PALETTES = {
    # Paul Tol's colorblind-friendly palette
    "colorblind": [
        "#4477AA",  # Blue
        "#EE6677",  # Red/Pink
        "#228833",  # Green
        "#CCBB44",  # Yellow
        "#66CCEE",  # Cyan
        "#AA3377",  # Purple
        "#BBBBBB",  # Grey
    ],
    # IBM Design colorblind safe
    "ibm": [
        "#648FFF",  # Blue
        "#DC267F",  # Magenta
        "#FFB000",  # Gold
        "#FE6100",  # Orange
        "#785EF0",  # Purple
    ],
    # Qualitative palette for many categories
    "categorical": [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Grey
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ],
    # Sequential for heatmaps
    "sequential": "viridis",
    # Diverging for comparisons
    "diverging": "RdBu_r",
}


def get_color_palette(
    name: str = "colorblind",
    n_colors: int | None = None,
) -> list[str]:
    """
    Get a colorblind-friendly color palette.

    Args:
        name: Palette name ('colorblind', 'ibm', 'categorical')
        n_colors: Number of colors needed (cycles if exceeds palette)

    Returns:
        List of hex color strings
    """
    if name not in PALETTES:
        raise ValueError(f"Unknown palette: {name}. Choose from {list(PALETTES.keys())}")

    palette = PALETTES[name]

    if isinstance(palette, str):
        # It's a matplotlib colormap name
        cmap = plt.get_cmap(palette)
        if n_colors is None:
            n_colors = 7
        return [mpl.colors.rgb2hex(cmap(i / (n_colors - 1))) for i in range(n_colors)]

    if n_colors is None:
        return palette

    # Cycle colors if needed
    return [palette[i % len(palette)] for i in range(n_colors)]


def apply_nature_style(style: NatureStyle | None = None) -> NatureStyle:
    """
    Apply publication style to matplotlib.

    Args:
        style: Style configuration (uses defaults if None)

    Returns:
        The applied style configuration
    """
    if style is None:
        style = NatureStyle()

    # Reset to defaults first
    plt.rcdefaults()

    # Typography
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [style.font_family, "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": style.font_size,
        "axes.titlesize": style.title_size,
        "axes.labelsize": style.label_size,
        "xtick.labelsize": style.tick_size,
        "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.legend_size,
        "figure.titlesize": style.title_size + 1,
    })

    # Line widths
    plt.rcParams.update({
        "axes.linewidth": style.linewidth,
        "xtick.major.width": style.linewidth,
        "ytick.major.width": style.linewidth,
        "xtick.minor.width": style.linewidth * 0.5,
        "ytick.minor.width": style.linewidth * 0.5,
        "lines.linewidth": style.linewidth * 1.5,
        "patch.linewidth": style.linewidth,
    })

    # Ticks
    plt.rcParams.update({
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    # Axes
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
    })

    # Legend
    plt.rcParams.update({
        "legend.frameon": False,
        "legend.borderpad": 0,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
    })

    # Figure
    plt.rcParams.update({
        "figure.dpi": SCREEN_DPI,
        "savefig.dpi": PRINT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Don't use constrained_layout globally - causes issues with colorbars/polar
        "figure.constrained_layout.use": False,
    })

    # Color cycle
    colors = get_color_palette(style.palette)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    return style


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    formats: list[str] = None,
    dpi: int = PRINT_DPI,
    transparent: bool = False,
):
    """
    Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        formats: List of formats to save (default: png, pdf, svg)
        dpi: DPI for raster formats
        transparent: Whether background is transparent
    """
    if formats is None:
        formats = ["png", "pdf", "svg"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = path.with_suffix(f".{fmt}")
        fig.savefig(
            output_path,
            format=fmt,
            dpi=dpi if fmt == "png" else None,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.02,
        )


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.15,
    y: float = 1.05,
    fontsize: float | None = None,
    fontweight: str = "bold",
):
    """
    Add panel label (a, b, c, etc.) to axes.

    Args:
        ax: Matplotlib axes
        label: Panel label (e.g., "a", "b", "c")
        x, y: Position in axes coordinates
        fontsize: Font size (default: axes title size + 2)
        fontweight: Font weight
    """
    if fontsize is None:
        fontsize = plt.rcParams["axes.titlesize"] + 2

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom",
        ha="right",
    )


def create_figure(
    n_panels: int = 1,
    n_cols: int | None = None,
    width: float | Literal["single", "double"] = "single",
    aspect_ratio: float = 0.8,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with consistent sizing.

    Args:
        n_panels: Number of panels
        n_cols: Number of columns (default: auto)
        width: Figure width ('single', 'double', or inches)
        aspect_ratio: Height/width ratio per panel
        **kwargs: Additional arguments to plt.subplots

    Returns:
        Figure and axes array
    """
    if width == "single":
        width = SINGLE_COL_WIDTH
    elif width == "double":
        width = DOUBLE_COL_WIDTH

    if n_cols is None:
        n_cols = min(n_panels, 3)

    n_rows = int(np.ceil(n_panels / n_cols))

    # Calculate height
    panel_width = width / n_cols
    panel_height = panel_width * aspect_ratio
    height = min(panel_height * n_rows, MAX_HEIGHT)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height),
                             constrained_layout=True, **kwargs)

    # Ensure axes is always a flat array
    if n_panels == 1:
        axes = np.array([axes])
    else:
        axes = np.atleast_1d(axes).flatten()

    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:n_panels]


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.0001:
        return "p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.4f}"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


def add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float,
    height: float = 0.02,
):
    """
    Add significance bar between two positions.

    Args:
        ax: Matplotlib axes
        x1, x2: X positions of the two groups
        y: Y position for the bar
        p: P-value
        height: Height of the bar caps
    """
    # Get y range for scaling
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    cap_height = y_range * height

    # Draw bar
    ax.plot([x1, x1, x2, x2], [y, y + cap_height, y + cap_height, y],
            color="black", linewidth=0.75)

    # Add significance annotation
    if p < 0.001:
        text = "***"
    elif p < 0.01:
        text = "**"
    elif p < 0.05:
        text = "*"
    else:
        text = "ns"

    ax.text((x1 + x2) / 2, y + cap_height, text,
            ha="center", va="bottom", fontsize=plt.rcParams["font.size"])
