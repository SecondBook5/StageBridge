"""UMAP and trajectory embedding visualizations for StageBridge.

Poster-quality figures showing:
  - UMAP scatter colored by lung cancer stage (Normal→AAH→AIS→MIA→LUAD)
  - Predicted cell-state trajectories overlaid as quiver arrows on UMAP
  - Context vector c_s UMAP (dimensionality-reduced context embeddings)

Enhanced with:
  - Density contours and convex hulls
  - Statistical annotations
  - Color-blind friendly palettes
  - Publication-quality styling
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

from stagebridge.logging_utils import get_logger
from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Stage color palette — ordered Normal→AAH→AIS→MIA→LUAD (color-blind friendly)
_STAGE_COLORS: dict[str, str] = {
    "Normal": "#00BA38",  # green (healthy) - colorblind safe
    "AAH": "#F8766D",  # coral (early precursor)
    "AIS": "#619CFF",  # blue (intermediate precursor)
    "MIA": "#E58700",  # orange (late precursor)
    "LUAD": "#A3A500",  # olive (invasive)
    "Unknown": "#999999",  # gray
}


def _get_umap_coords(adata: Any) -> np.ndarray:
    """Extract 2-D embedding coordinates from adata.obsm.

    Tries X_umap first, then X_pca as fallback.
    """
    for key in ("X_umap", "X_pca"):
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key], dtype=np.float32)
            if arr.shape[1] >= 2:
                return arr[:, :2]
    raise KeyError(
        "No 2-D embedding found. "
        "Run sc.tl.umap(adata) first, or ensure adata.obsm['X_umap'] exists."
    )


def _confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 2.0,
    facecolor: str = "none",
    edgecolor: str = "black",
    alpha: float = 0.5,
    linewidth: float = 2,
) -> Ellipse:
    """Draw confidence ellipse for a 2D point cloud.

    Parameters
    ----------
    x, y : array-like
        Coordinates
    ax : Axes
        Target axes
    n_std : float
        Number of standard deviations (default 2.0 = ~95% confidence)
    """
    if len(x) < 3:
        return None

    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle="--",
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _draw_convex_hull(
    coords: np.ndarray, ax: plt.Axes, color: str, alpha: float = 0.15, linewidth: float = 2
) -> None:
    """Draw convex hull around point cloud."""
    if len(coords) < 3:
        return

    try:
        hull = ConvexHull(coords)
        for simplex in hull.simplices:
            ax.plot(
                coords[simplex, 0],
                coords[simplex, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha * 3,
                linestyle="-",
            )

        # Fill the hull
        hull_points = coords[hull.vertices]
        ax.fill(hull_points[:, 0], hull_points[:, 1], color=color, alpha=alpha)
    except Exception as e:
        log.debug(f"Could not draw convex hull: {e}")


def _stage_scatter(
    ax: plt.Axes,
    coords: np.ndarray,
    stages: np.ndarray,
    s: float,
    alpha: float,
    show_hulls: bool = False,
    show_ellipses: bool = False,
) -> None:
    """Draw per-stage scatter ensuring canonical order in legend.

    Parameters
    ----------
    show_hulls : bool
        Whether to draw convex hulls around each stage cluster
    show_ellipses : bool
        Whether to draw confidence ellipses (95% confidence interval)
    """
    ordered = list(CANONICAL_STAGE_ORDER) + [
        st for st in np.unique(stages) if st not in CANONICAL_STAGE_ORDER
    ]
    for stage in ordered:
        mask = stages == stage
        if not mask.any():
            continue

        color = _STAGE_COLORS.get(stage, "#999999")
        stage_coords = coords[mask]

        # Draw convex hull first (background)
        if show_hulls and len(stage_coords) >= 3:
            _draw_convex_hull(stage_coords, ax, color, alpha=0.1, linewidth=1.5)

        # Draw confidence ellipse
        if show_ellipses and len(stage_coords) >= 3:
            _confidence_ellipse(
                stage_coords[:, 0],
                stage_coords[:, 1],
                ax,
                n_std=2.0,
                edgecolor=color,
                alpha=0.4,
                linewidth=2,
            )

        # Draw scatter points on top
        ax.scatter(
            stage_coords[:, 0],
            stage_coords[:, 1],
            c=color,
            s=s,
            alpha=alpha,
            label=stage,
            rasterized=True,
            edgecolors="white",
            linewidths=0.3,
        )


def plot_umap_by_stage(
    adata: Any,
    output_path: Path,
    title: str = "UMAP by Stage",
    stage_col: str = "stage",
    point_size: float = 3.0,
    alpha: float = 0.6,
    show_density: bool = True,
    show_hulls: bool = True,
    show_ellipses: bool = False,
) -> None:
    """Scatter UMAP colored by lung cancer stage with advanced visualization options.

    Parameters
    ----------
    adata : AnnData
        Must have obsm['X_umap'] and obs[stage_col].
    output_path : Path
        Figure output path (.png).  A .pdf is also saved alongside.
    title : str
        Figure title.
    stage_col : str
        Column in adata.obs containing stage labels.
    point_size : float
        Size of scatter points
    alpha : float
        Transparency of points
    show_density : bool
        Whether to show density contours for major clusters
    show_hulls : bool
        Whether to draw convex hulls around stage clusters
    show_ellipses : bool
        Whether to draw 95% confidence ellipses
    """
    output_path = Path(output_path)
    coords = _get_umap_coords(adata)
    stages = (
        adata.obs[stage_col].astype(str).values
        if stage_col in adata.obs.columns
        else np.array(["Unknown"] * adata.n_obs)
    )

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Draw density contours for overall distribution
    if show_density and len(coords) > 100:
        try:
            kde = gaussian_kde(coords.T)
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min

            xx, yy = np.mgrid[
                x_min - 0.1 * x_range : x_max + 0.1 * x_range : 100j,
                y_min - 0.1 * y_range : y_max + 0.1 * y_range : 100j,
            ]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            ax.contour(
                xx,
                yy,
                density,
                levels=5,
                colors="gray",
                alpha=0.2,
                linewidths=0.5,
                linestyles="dashed",
            )
        except Exception as e:
            log.debug(f"Could not draw density contours: {e}")

    # Draw scatter with optional hulls and ellipses
    _stage_scatter(
        ax,
        coords,
        stages,
        s=point_size,
        alpha=alpha,
        show_hulls=show_hulls,
        show_ellipses=show_ellipses,
    )

    # Enhanced styling
    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Improved legend
    legend = ax.legend(
        markerscale=3, framealpha=0.95, fontsize=11, loc="best", title="Stage", title_fontsize=12
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Enhanced UMAP by stage written: %s", output_path)


def plot_umap_with_trajectories(
    adata: Any,
    uv0: np.ndarray,
    uv1_pred: np.ndarray,
    output_path: Path,
    title: str = "Predicted Cell-State Trajectories",
    n_arrows: int = 200,
    stage_col: str = "stage",
    arrow_color: str = "#1D4ED8",
    arrow_alpha: float = 0.7,
    arrow_width: float = 0.004,
    show_density: bool = True,
) -> None:
    """Overlay Euler-integrated trajectory arrows on UMAP with enhanced styling.

    Parameters
    ----------
    adata : AnnData
        Background scatter data; must have obsm['X_umap'].
    uv0 : ndarray, shape (n, 2)
        Source cell 2-D UMAP positions.
    uv1_pred : ndarray, shape (n, 2)
        Predicted target cell 2-D UMAP positions (already projected).
    output_path : Path
    n_arrows : int
        Randomly subsample this many arrows for visual clarity.
    stage_col : str
        obs column for background scatter stage colors.
    arrow_color : str
        Color for trajectory arrows
    arrow_alpha : float
        Arrow transparency
    arrow_width : float
        Width of arrows
    show_density : bool
        Whether to show density contours
    """
    output_path = Path(output_path)
    bg_coords = _get_umap_coords(adata)
    stages = (
        adata.obs[stage_col].astype(str).values
        if stage_col in adata.obs.columns
        else np.array(["Unknown"] * adata.n_obs)
    )

    uv0 = np.asarray(uv0, dtype=np.float32)
    uv1_pred = np.asarray(uv1_pred, dtype=np.float32)

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Draw density contours
    if show_density and len(bg_coords) > 100:
        try:
            kde = gaussian_kde(bg_coords.T)
            x_min, x_max = bg_coords[:, 0].min(), bg_coords[:, 0].max()
            y_min, y_max = bg_coords[:, 1].min(), bg_coords[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min

            xx, yy = np.mgrid[
                x_min - 0.1 * x_range : x_max + 0.1 * x_range : 100j,
                y_min - 0.1 * y_range : y_max + 0.1 * y_range : 100j,
            ]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            ax.contour(
                xx,
                yy,
                density,
                levels=5,
                colors="gray",
                alpha=0.15,
                linewidths=0.5,
                linestyles="dashed",
            )
        except Exception as e:
            log.debug(f"Could not draw density contours: {e}")

    # Background scatter with semi-transparent points
    _stage_scatter(ax, bg_coords, stages, s=2.0, alpha=0.25, show_hulls=False, show_ellipses=False)

    # Subsample arrows for clarity
    rng = np.random.default_rng(42)
    idx = rng.choice(len(uv0), size=min(n_arrows, len(uv0)), replace=False)
    dx = uv1_pred[idx, 0] - uv0[idx, 0]
    dy = uv1_pred[idx, 1] - uv0[idx, 1]

    # Draw arrows with gradient effect (thicker at base)
    quiver = ax.quiver(
        uv0[idx, 0],
        uv0[idx, 1],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=arrow_color,
        alpha=arrow_alpha,
        width=arrow_width,
        headwidth=4,
        headlength=5,
        headaxislength=4.5,
        edgecolors="white",
        linewidths=0.3,
    )

    # Enhanced styling
    ax.set_xlabel("UMAP 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Legend for background stages
    legend1 = ax.legend(
        markerscale=3,
        framealpha=0.95,
        fontsize=10,
        loc="upper right",
        title="Background Stage",
        title_fontsize=11,
    )
    legend1.get_frame().set_facecolor("white")
    legend1.get_frame().set_edgecolor("gray")
    legend1.get_frame().set_linewidth(1.5)

    # Add arrow legend manually
    arrow_patch = mpatches.FancyArrow(
        0, 0, 0.1, 0.1, width=0.05, color=arrow_color, alpha=arrow_alpha
    )
    from matplotlib.lines import Line2D

    arrow_legend = Line2D(
        [0],
        [0],
        marker=">",
        markersize=10,
        color=arrow_color,
        alpha=arrow_alpha,
        linestyle="none",
        label="Predicted trajectory",
    )
    legend2 = ax.legend(handles=[arrow_legend], loc="lower right", framealpha=0.95, fontsize=11)
    legend2.get_frame().set_facecolor("white")
    legend2.get_frame().set_edgecolor("gray")
    legend2.get_frame().set_linewidth(1.5)
    ax.add_artist(legend1)  # Keep both legends

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Enhanced UMAP with trajectories written: %s", output_path)


def plot_context_vector_umap(
    adata: Any,
    context_vectors: np.ndarray,
    output_path: Path,
    title: str = "Context Vector UMAP (c_s)",
    stage_col: str = "stage",
    point_size: float = 6.0,
    alpha: float = 0.65,
    show_hulls: bool = True,
    show_ellipses: bool = True,
) -> None:
    """2-D UMAP of per-cell context vectors c_s, colored by stage with enhanced visualization.

    The context vectors come from ``stagebridge.evaluation.gene_attribution.extract_context_vectors``
    and live in ``adata.obsm["X_context"]``.  This function reduces them further to 2-D for
    visualization, testing whether the Set Transformer encodes stage-relevant information.

    Parameters
    ----------
    adata : AnnData
        Used only for stage labels (obs[stage_col]).
    context_vectors : ndarray, shape (n_cells, context_dim)
        Per-cell context embeddings from the PMA seed vector output.
    output_path : Path
    title : str
        Figure title
    stage_col : str
        Column name for stage labels
    point_size : float
        Size of scatter points
    alpha : float
        Point transparency
    show_hulls : bool
        Whether to draw convex hulls
    show_ellipses : bool
        Whether to draw confidence ellipses
    """
    output_path = Path(output_path)
    cv = np.asarray(context_vectors, dtype=np.float32)

    # Reduce to 2-D: try umap-learn, fall back to PCA
    try:
        import umap as umap_lib

        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
        coords = np.asarray(reducer.fit_transform(cv), dtype=np.float32)
        embed_label = "UMAP"
        log.info("Using UMAP for context vector embedding")
    except ImportError:
        log.warning("umap-learn not available; using PCA for context vector embedding.")
        from sklearn.decomposition import PCA

        coords = np.asarray(PCA(n_components=2).fit_transform(cv), dtype=np.float32)
        embed_label = "PC"

    stages = (
        adata.obs[stage_col].astype(str).values
        if stage_col in adata.obs.columns
        else np.array(["Unknown"] * len(cv))
    )

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Draw density contours
    if len(coords) > 100:
        try:
            kde = gaussian_kde(coords.T)
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min

            xx, yy = np.mgrid[
                x_min - 0.1 * x_range : x_max + 0.1 * x_range : 100j,
                y_min - 0.1 * y_range : y_max + 0.1 * y_range : 100j,
            ]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = np.reshape(kde(positions).T, xx.shape)

            # Use filled contours for better visual effect
            contourf = ax.contourf(xx, yy, density, levels=8, cmap="Greys", alpha=0.3)
            ax.contour(
                xx,
                yy,
                density,
                levels=8,
                colors="gray",
                alpha=0.2,
                linewidths=0.5,
                linestyles="solid",
            )
        except Exception as e:
            log.debug(f"Could not draw density contours: {e}")

    # Draw scatter with hulls and ellipses
    _stage_scatter(
        ax,
        coords,
        stages,
        s=point_size,
        alpha=alpha,
        show_hulls=show_hulls,
        show_ellipses=show_ellipses,
    )

    # Enhanced styling
    ax.set_xlabel(f"Context {embed_label} 1", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Context {embed_label} 2", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Statistical annotation - count per stage
    stage_counts = {stage: np.sum(stages == stage) for stage in np.unique(stages)}
    count_text = "Stage counts:\n" + "\n".join([f"{s}: n={c}" for s, c in stage_counts.items()])
    ax.text(
        0.02,
        0.98,
        count_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Improved legend
    legend = ax.legend(
        markerscale=2.5, framealpha=0.95, fontsize=11, loc="best", title="Stage", title_fontsize=12
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2, linestyle=":", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Enhanced context vector UMAP written: %s", output_path)
