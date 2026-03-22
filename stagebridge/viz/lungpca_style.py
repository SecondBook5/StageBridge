"""LungPCA Publication Style - Exact color palettes and figure conventions.

This module provides the exact styling from the Peng et al. LungPCA paper
for consistent, publication-quality figures in StageBridge.

Reference: Peng et al. Nature (2024) - Lung precancer atlas

Color palettes extracted from:
- Figure 1.R - Sankey diagrams, stage colors
- Figure 2C.py - Spatial hexbin plots
- Figure 3.R - Cell type colors, MP colors, alluvial plots
- Figure 4.R - Correlation matrices, CytoSignal
- Figure 5.R - Neighborhood composition, violin plots
- Figure 6.R - Mouse validation figures
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from typing import Any, Literal

# =============================================================================
# COLOR PALETTES (exact from LungPCA R/Python code)
# =============================================================================

# Stage/Histology colors (Figure 1B, used throughout)
STAGE_COLORS = {
    "Normal": "#33a02c",        # green
    "AAH": "#b2df8a",           # light green
    "AIS": "#fdbf6f",           # light orange
    "MIA": "#fb9a99",           # pink
    "LUAD": "#ff7f00",          # orange
    "Tumor necrosis": "#e31a1c", # red
    "Unknown": "#d9d9d9",       # gray
}

# Alternative stage colors (Figure 3H, 5G style)
STAGE_COLORS_ALT = {
    "Normal": "#11A579",
    "AAH": "#66C5CC",
    "AIS": "#F6CF71",
    "MIA": "#f97b72",
    "LUAD": "#ed5887",
    "No_lesion": "#B0C4DE",
    "LUAD_lepidic": "#4363D8",
    "LUAD_acinar": "#F58231",
    "LUAD_papillary": "#911EB4",
    "LUAD_solid": "#46F0F0",
    "Unassigned": "#d9d9d9",
}

# Epithelial cell type colors (Figure 3A)
EPITHELIAL_COLORS = {
    "Basal": "#1f77b4",
    "AT1": "#ff7f0e",           # or #aec7e8 in some figs
    "AT2": "#e377c2",
    "KAC": "#2ca02c",           # or #d62728 or #17becf
    "Ciliated": "#ffbb78",
    "Club": "#ff9896",
    "Club.secretory": "#9467bd",
    "Club and Secretory": "#ff9896",
    "AIC": "#d62728",
    "Tumor": "#f7b6d2",
    "Precursor": "#bcbd22",
    "Invasive": "#17becf",
    "Neuroendocrine": "#9467bd",
    "Proliferating": "#ff7f0e",
    "Tuft": "#1f77b4",
}

# Alternative epithelial (Figure 6B mouse)
EPITHELIAL_COLORS_MOUSE = {
    "KAC": "#17becf",
    "AT1": "#aec7e8",
    "AT2": "#e377c2",
    "Basal": "#98df8a",
    "Ciliated": "#ffbb78",
    "Club and Secretory": "#ff9896",
    "Neuroendocrine": "#9467bd",
    "Proliferating": "#ff7f0e",
    "Tuft": "#1f77b4",
    "Tumor": "#f7b6d2",
}

# Stromal/TME cell type colors (Figure 1B)
STROMAL_COLORS = {
    "Lymphoid": "#ffff99",
    "Myeloid": "#6a3d9a",
    "Fibroblast": "#cab2d6",
    "Vessel": "#b15928",
    "Smooth muscle cells": "#a6761d",
    "Other": "#d9d9d9",
    "Epi_spots": "#1f78b4",
    "nonEpi_spots": "#a6cee3",
}

# Major cell type colors (Figure 5A, 5H - full palette)
MAJOR_CELLTYPE_COLORS = {
    "Epithelial": "#FF0000",
    "T cells": "#FF00FF",
    "B cells": "#808000",
    "Plasma": "#800000",
    "Mast": "#54278f",
    "Myeloid": "#FF9900",
    "DC": "#33FFFF",
    "Fibroblast": "#E6BEFF",
    "Endothelial": "#66FF33",
    "Smooth muscle": "#A9A9A9",
    "Pericyte": "#FABEBE",
    "Lymphatic": "#008080",
    "Mesothelial": "#b3de69",
    "Neuronal": "#FFFF00",
    "Cycling": "#8dd3c7",
    "Unknown": "#D2691E",
    "Adipocyte": "#4363D8",
    "Other": "#deebf7",
}

# Meta-program (MP) colors (Figure 3F-L)
MP_COLORS = {
    "MP1": "#F6CF71",
    "MP2": "#3969AC",
    "MP3": "#80BA5A",
    "MP4": "#F2B701",
    "MP5": "#11A579",
    "MP6": "#CF1C90",
    "MP7": "#66C5CC",
    "MP8": "#f97b72",
    "MP9": "#ed5887",
}

# Clone colors (Figure 2C, 3M)
CLONE_COLORS = {
    "Shared": "#11A579",
    "Invasive-specific": "#66C5CC",
    "Non-invasive-specific": "#ed5887",
    "Ref": "#1f77b4",
}

# Lineage colors (Figure 5B-C)
LINEAGE_COLORS = {
    "Alveolar": "#0570b0",
    "Airway": "#8dd3c7",
    "Basal": "#006837",
    "Club": "#feb24c",
    "Tumor": "#bd0026",
    "Other_epithelial": "#4eb3d3",
    "Stromal": "#b3de69",
    "Immune": "#fcc5c0",
    "Endothelial": "#54278f",
    "Other": "#fddbc7",
    "Unknown": "#fccde5",
}

# Treatment group colors (Figure 6J-K)
TREATMENT_COLORS = {
    "Control": "#17becf",
    "IL-1β": "#ff7f0e",
    "IM": "#e377c2",
}

# =============================================================================
# COLORMAPS
# =============================================================================

def get_magma_white() -> mcolors.LinearSegmentedColormap:
    """Custom magma colormap with white start (Figure 3E, 3G style)."""
    from matplotlib import cm
    magma = cm.get_cmap("magma", 323)
    white_portion = plt.cm.colors.LinearSegmentedColormap.from_list(
        "white_start", ["white", magma(0.15)], N=10
    )
    # Combine white start with reversed magma
    colors_white = [white_portion(i/9) for i in range(10)]
    colors_magma = [magma(1 - i/322 * 0.82) for i in range(323)]
    all_colors = colors_white + colors_magma
    return mcolors.LinearSegmentedColormap.from_list("magma_white", all_colors)


def get_rdbu_diverging() -> mcolors.LinearSegmentedColormap:
    """RdBu diverging colormap for heatmaps (Figure 1D style)."""
    return plt.cm.get_cmap("RdBu_r")


def get_turbo_truncated() -> mcolors.LinearSegmentedColormap:
    """Turbo colormap for spatial expression (Figure 1E style)."""
    return plt.cm.get_cmap("turbo")


def get_correlation_cmap() -> mcolors.LinearSegmentedColormap:
    """Correlation colormap (Figure 4B style) - blue-white-red with asymmetry."""
    colors = (
        list(plt.cm.colors.LinearSegmentedColormap.from_list(
            "blue_white", ["#2171b5", "#ffffbf"], N=709
        )(np.linspace(0, 1, 709))) +
        list(plt.cm.colors.LinearSegmentedColormap.from_list(
            "white_red", ["#ffffbf", "#d73027"], N=300
        )(np.linspace(0, 1, 300))) +
        list(plt.cm.colors.LinearSegmentedColormap.from_list(
            "red_dark", ["#d73027", "#bd0026"], N=221
        )(np.linspace(0, 1, 221)))
    )
    return mcolors.LinearSegmentedColormap.from_list("correlation", colors)


# Feature expression colormap (Figure 1C style)
EXPRESSION_CMAP_COLORS = [
    "white", "#ffffe5", "#ffffcc", "#ffffb2",
    "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"
]


# =============================================================================
# FIGURE CONFIGURATION
# =============================================================================

# Standard figure settings from LungPCA
FIGURE_SETTINGS = {
    "dpi": 300,
    "font_size": 6,           # Base font size
    "title_size": 12,
    "label_size": 6,
    "tick_size": 6,
    "legend_size": 6,
    "linewidth": 0.2,         # For boxplots, violins
    "boxplot_width": 0.5,
    "jitter_size": 0.5,
    "pt_size": 0.1,           # Scatter point size
}


def configure_lungpca_style() -> None:
    """Configure matplotlib to match LungPCA publication style."""
    mpl.rcParams.update({
        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        # DPI
        "savefig.dpi": 300,
        "figure.dpi": 150,

        # Fonts (LungPCA uses small fonts)
        "font.family": "sans-serif",
        "font.size": 6,
        "axes.titlesize": 8,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,

        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,

        # Colors
        "axes.edgecolor": "black",
        "text.color": "black",

        # Grid (off by default)
        "axes.grid": False,

        # Legend
        "legend.frameon": False,

        # Saving
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def get_stage_color(stage: str, style: Literal["default", "alt"] = "default") -> str:
    """Get stage color matching LungPCA palette."""
    palette = STAGE_COLORS if style == "default" else STAGE_COLORS_ALT
    return palette.get(stage, "#d9d9d9")


def get_celltype_color(
    celltype: str,
    category: Literal["epithelial", "stromal", "major", "mouse"] = "epithelial"
) -> str:
    """Get cell type color matching LungPCA palette."""
    palettes = {
        "epithelial": EPITHELIAL_COLORS,
        "stromal": STROMAL_COLORS,
        "major": MAJOR_CELLTYPE_COLORS,
        "mouse": EPITHELIAL_COLORS_MOUSE,
    }
    return palettes[category].get(celltype, "#d9d9d9")


def get_mp_color(mp: str) -> str:
    """Get meta-program color."""
    return MP_COLORS.get(mp, "#d9d9d9")


def create_lungpca_figure(
    width_inches: float = 5,
    height_inches: float = 5,
    dpi: int = 300,
) -> tuple[plt.Figure, plt.Axes]:
    """Create figure with LungPCA dimensions and style."""
    configure_lungpca_style()
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def save_lungpca_figure(
    fig: plt.Figure,
    output_path: Path | str,
    width_inches: float | None = None,
    height_inches: float | None = None,
) -> None:
    """Save figure with LungPCA specifications (300 DPI, PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if width_inches and height_inches:
        fig.set_size_inches(width_inches, height_inches)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format=output_path.suffix[1:] if output_path.suffix else "png",
    )

    # Also save PDF for vector graphics
    if output_path.suffix.lower() == ".png":
        fig.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
        )


# =============================================================================
# SPECIFIC PLOT TYPES (matching LungPCA figures)
# =============================================================================

def plot_violin_boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    labels: list[str] | None = None,
    width: float = 0.8,
    show_points: bool = False,
) -> None:
    """Violin + boxplot combination (Figure 5E, 5K style)."""
    parts = ax.violinplot(
        data, positions=positions, widths=width,
        showmeans=False, showmedians=False, showextrema=False
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.8)

    # Add boxplot overlay
    bp = ax.boxplot(
        data, positions=positions, widths=width * 0.3,
        patch_artist=True, showfliers=False
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.5)
    for cap in bp["caps"]:
        cap.set_linewidth(0.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1)

    if labels:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)


def plot_stacked_bar(
    ax: plt.Axes,
    data: dict[str, list[float]],  # {category: [values per group]}
    groups: list[str],
    colors: dict[str, str],
    normalize: bool = True,
) -> None:
    """Stacked bar chart (Figure 5D, 5J style)."""
    x = np.arange(len(groups))
    width = 0.8

    # Convert to arrays
    categories = list(data.keys())
    values = np.array([data[cat] for cat in categories])

    if normalize:
        totals = values.sum(axis=0)
        values = values / totals[np.newaxis, :]

    # Stack bars
    bottom = np.zeros(len(groups))
    for i, cat in enumerate(categories):
        ax.bar(
            x, values[i], width, bottom=bottom,
            label=cat, color=colors.get(cat, "#d9d9d9"),
            edgecolor="none"
        )
        bottom += values[i]

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Proportion" if normalize else "Count", fontsize=6)


def plot_pie_chart(
    ax: plt.Axes,
    values: list[float],
    labels: list[str],
    colors: list[str],
    show_labels: bool = False,
) -> None:
    """Pie chart (Figure 4C, 5G style)."""
    ax.pie(
        values,
        labels=labels if show_labels else None,
        colors=colors,
        autopct=None,
        startangle=90,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax.axis("equal")


def plot_boxplot_jitter(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    labels: list[str] | None = None,
    width: float = 0.5,
    jitter_size: float = 1.0,
) -> None:
    """Boxplot with jittered points (Figure 1B, 3H style)."""
    bp = ax.boxplot(
        data, positions=positions, widths=width,
        patch_artist=True, showfliers=False
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.2)
        patch.set_alpha(0.8)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.2)
    for cap in bp["caps"]:
        cap.set_linewidth(0.2)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(0.5)

    # Add jittered points
    for i, (d, pos) in enumerate(zip(data, positions)):
        jitter = np.random.uniform(-0.2, 0.2, len(d))
        ax.scatter(
            pos + jitter, d,
            c=colors[i % len(colors)],
            s=jitter_size, alpha=0.6,
            edgecolors="none"
        )

    if labels:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=6)


def plot_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str | mcolors.Colormap = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    show_values: bool = False,
    cbar: bool = True,
) -> Any:
    """Publication heatmap (Figure 1D, 4B style)."""
    im = ax.imshow(
        data, cmap=cmap, aspect="auto",
        vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    if row_labels:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)

    if col_labels:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6)

    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=4,
                    color="white" if abs(data[i, j]) > (vmax or data.max()) * 0.5 else "black"
                )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return im


def plot_spatial_hexbin(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    gridsize: int = 50,
    cmap: str = "magma",
    background_img: np.ndarray | None = None,
    background_alpha: float = 0.3,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Spatial hexbin plot on tissue (Figure 2C style)."""
    if background_img is not None:
        ax.imshow(background_img, alpha=background_alpha)

    hb = ax.hexbin(
        x, y, C=values,
        gridsize=gridsize,
        cmap=cmap,
        edgecolors="face",
        linewidths=0,
        vmin=vmin, vmax=vmax,
        alpha=1.0
    )

    ax.axis("off")
    ax.set_frame_on(False)

    return hb


def plot_spatial_categorical(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
    color_map: dict[str, str],
    gridsize: int = 50,
    background_img: np.ndarray | None = None,
    background_alpha: float = 1.0,
) -> None:
    """Categorical spatial plot (Figure 2C Histology style)."""
    if background_img is not None:
        ax.imshow(background_img, alpha=background_alpha)

    unique_cats = [c for c in color_map.keys() if c in np.unique(categories)]
    colors = [color_map[c] for c in unique_cats]
    cmap = mcolors.ListedColormap(colors)

    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
    color_values = np.array([cat_to_idx.get(c, 0) for c in categories])

    ax.hexbin(
        x, y, C=color_values,
        gridsize=gridsize,
        cmap=cmap,
        edgecolors="face",
        linewidths=-0.15,
        alpha=1.0
    )

    ax.axis("off")
    ax.set_frame_on(False)


# =============================================================================
# SANKEY DIAGRAM (Figure 1B style)
# =============================================================================

def plot_sankey_diagram(
    source_labels: list[str],
    target_labels: list[str],
    flow_matrix: np.ndarray,
    source_colors: dict[str, str] | None = None,
    target_colors: dict[str, str] | None = None,
    output_path: Path | str | None = None,
    title: str = "",
    width: int = 800,
    height: int = 500,
) -> Any:
    """Sankey diagram matching LungPCA Figure 1B style.

    Uses plotly for interactive Sankey, falls back to matplotlib.
    """
    try:
        import plotly.graph_objects as go

        n_src = len(source_labels)
        all_labels = source_labels + target_labels

        # Build links
        sources, targets, values = [], [], []
        for i in range(flow_matrix.shape[0]):
            for j in range(flow_matrix.shape[1]):
                if flow_matrix[i, j] > 0:
                    sources.append(i)
                    targets.append(n_src + j)
                    values.append(float(flow_matrix[i, j]))

        # Colors
        if source_colors is None:
            source_colors = STAGE_COLORS
        if target_colors is None:
            target_colors = STAGE_COLORS

        node_colors = (
            [source_colors.get(l, "#d9d9d9") for l in source_labels] +
            [target_colors.get(l, "#d9d9d9") for l in target_labels]
        )

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(150, 150, 150, 0.4)",
            ),
        )])

        fig.update_layout(
            title_text=title,
            font_size=10,
            width=width,
            height=height,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_image(str(output_path), width=width, height=height, scale=2)

        return fig

    except ImportError:
        # Matplotlib fallback - show as heatmap
        fig, ax = create_lungpca_figure(width_inches=8, height_inches=6)
        im = plot_heatmap(
            ax, flow_matrix,
            row_labels=source_labels,
            col_labels=target_labels,
            cmap="Blues",
        )
        ax.set_title(f"{title} (Flow Matrix)", fontsize=8)
        ax.set_xlabel("Target", fontsize=6)
        ax.set_ylabel("Source", fontsize=6)

        if output_path:
            save_lungpca_figure(fig, output_path)

        return fig


# =============================================================================
# ALLUVIAL PLOT (Figure 3L style)
# =============================================================================

def plot_alluvial(
    ax: plt.Axes,
    data: dict[str, dict[str, float]],  # {stage: {category: value}}
    stage_order: list[str],
    colors: dict[str, str],
    alpha: float = 0.6,
) -> None:
    """Alluvial/stream plot for stage transitions (Figure 3L style).

    Simplified matplotlib version - for full alluvial use R's ggalluvial.
    """
    categories = list(colors.keys())
    x = np.arange(len(stage_order))

    # Normalize each stage
    normalized = {}
    for stage in stage_order:
        total = sum(data.get(stage, {}).values())
        normalized[stage] = {
            cat: data.get(stage, {}).get(cat, 0) / max(total, 1)
            for cat in categories
        }

    # Stack areas
    bottom = np.zeros(len(stage_order))
    for cat in categories:
        values = [normalized[stage].get(cat, 0) for stage in stage_order]
        ax.fill_between(
            x, bottom, bottom + values,
            label=cat, color=colors.get(cat, "#d9d9d9"),
            alpha=alpha, edgecolor="none"
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(stage_order, fontsize=6)
    ax.set_ylabel("Fraction", fontsize=6)
    ax.set_xlim(-0.5, len(stage_order) - 0.5)
    ax.set_ylim(0, 1)


# =============================================================================
# CONVENIENCE: STAGE COLOR LIST/ORDER
# =============================================================================

STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

def get_stage_colors_list(stages: list[str] | None = None) -> list[str]:
    """Get list of colors for stages in order."""
    if stages is None:
        stages = STAGE_ORDER
    return [get_stage_color(s) for s in stages]


def get_stage_cmap() -> mcolors.ListedColormap:
    """Get colormap for stages."""
    return mcolors.ListedColormap(get_stage_colors_list())
