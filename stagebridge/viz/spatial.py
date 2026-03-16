"""Poster-ready panel plotting and spatial Visium visualizations for StageBridge.

Enhanced with:
  - Better heatmaps with hierarchical clustering
  - Improved color scales and color-blind friendly palettes
  - Statistical annotations
  - Publication-quality styling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


def plot_method_schematic(output_path: Path) -> None:
    """Create a clean method schematic for poster panel A."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")

    boxes = [
        (0.05, 0.55, 0.23, 0.28, "Cross-sectional\ncell sets\n(snRNA + spatial)"),
        (0.34, 0.55, 0.23, 0.28, "Set Transformer\ncontext encoder\n(ISAB/SAB/PMA)"),
        (0.63, 0.55, 0.23, 0.28, "OT coupling\n(Sinkhorn)\nfor pseudo-pairs"),
        (0.34, 0.15, 0.23, 0.28, "Flow matching\nconditional dynamics\n$v_\\phi(x,t,c_s)$"),
        (0.63, 0.15, 0.23, 0.28, "Generated next-stage\ncell distribution\n+ eval metrics"),
    ]

    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.6,
            edgecolor="#0F172A",
            facecolor="#E2E8F0",
            alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.28, 0.69), (0.34, 0.69)),
        ((0.57, 0.69), (0.63, 0.69)),
        ((0.45, 0.55), (0.45, 0.43)),
        ((0.57, 0.29), (0.63, 0.29)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=2.0))

    ax.set_title("Panel A: Transformer-First StageBridge Pipeline", fontsize=14, pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_transition_trajectory(eval_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-transition trajectory metrics for panel C."""
    if eval_df.empty:
        raise ValueError("eval_df is empty")

    transitions = eval_df.apply(lambda r: f"{r['stage_src']}→{r['stage_tgt']}", axis=1).tolist()
    x = np.arange(len(transitions))

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()

    ax1.plot(
        x, eval_df["sinkhorn"].astype(float).values, marker="o", color="#0EA5E9", label="Sinkhorn"
    )
    ax1.plot(x, eval_df["mmd_rbf"].astype(float).values, marker="s", color="#0284C7", label="MMD")
    ax2.plot(
        x,
        eval_df["classifier_auc"].astype(float).values,
        marker="^",
        color="#F97316",
        label="Classifier AUC",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(transitions, rotation=20, ha="right")
    ax1.set_ylabel("Distance Metrics")
    ax2.set_ylabel("AUC")
    ax1.set_title("Panel C: Stage-to-Stage Trajectory Fidelity")
    ax1.grid(alpha=0.2)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Path,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    show_values: bool = True,
    figsize: tuple[float, float] = (11, 7),
) -> None:
    """Plot model-vs-metric heatmap with hierarchical clustering and enhanced styling.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with model metrics
    output_path : Path
        Output file path
    cluster_rows : bool
        Whether to cluster rows (models) by similarity
    cluster_cols : bool
        Whether to cluster columns (metrics) by similarity
    show_values : bool
        Whether to annotate cells with values
    figsize : tuple
        Figure size in inches
    """
    metric_cols = [
        c
        for c in ["sinkhorn_mean", "mmd_rbf_mean", "classifier_auc_mean", "jsd_composition_mean"]
        if c in metrics_df.columns
    ]
    if metrics_df.empty or not metric_cols:
        raise ValueError("metrics_df lacks required aggregate metric columns")

    mat = metrics_df[metric_cols].astype(float).values

    # z-score by metric column for comparability across scales.
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True) + 1e-8
    z = (mat - mu) / sd

    # Hierarchical clustering
    row_labels = metrics_df["label"].astype(str).tolist()
    col_labels = [c.replace("_mean", "").replace("_", " ").title() for c in metric_cols]

    row_order = np.arange(len(row_labels))
    col_order = np.arange(len(col_labels))

    if cluster_rows and len(row_labels) > 2:
        try:
            row_linkage = linkage(pdist(z, metric="euclidean"), method="average")
            row_dendrogram = dendrogram(row_linkage, no_plot=True)
            row_order = row_dendrogram["leaves"]
        except Exception as e:
            log.debug(f"Could not cluster rows: {e}")

    if cluster_cols and len(col_labels) > 2:
        try:
            col_linkage = linkage(pdist(z.T, metric="euclidean"), method="average")
            col_dendrogram = dendrogram(col_linkage, no_plot=True)
            col_order = col_dendrogram["leaves"]
        except Exception as e:
            log.debug(f"Could not cluster columns: {e}")

    # Reorder data
    z_ordered = z[row_order][:, col_order]
    row_labels_ordered = [row_labels[i] for i in row_order]
    col_labels_ordered = [col_labels[i] for i in col_order]

    # Set up publication-quality figure with dendrogram space
    fig = plt.figure(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")

    # Create grid for heatmap and dendrograms
    if cluster_rows:
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(1, 2, width_ratios=[0.15, 0.85], wspace=0.02)
        ax_dendro = fig.add_subplot(gs[0])
        ax_heatmap = fig.add_subplot(gs[1])

        # Draw row dendrogram
        if len(row_labels) > 2:
            try:
                row_linkage = linkage(pdist(z, metric="euclidean"), method="average")
                dendrogram(
                    row_linkage,
                    ax=ax_dendro,
                    orientation="left",
                    color_threshold=0,
                    above_threshold_color="gray",
                )
                ax_dendro.set_xticks([])
                ax_dendro.set_yticks([])
                ax_dendro.spines[:].set_visible(False)
            except Exception:
                ax_dendro.axis("off")
    else:
        ax_heatmap = fig.add_subplot(111)

    # Draw heatmap with improved colormap
    im = ax_heatmap.imshow(
        z_ordered, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2, interpolation="nearest"
    )

    # Add grid lines
    for i in range(len(row_labels_ordered) + 1):
        ax_heatmap.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(col_labels_ordered) + 1):
        ax_heatmap.axvline(j - 0.5, color="white", linewidth=1.5)

    # Set ticks and labels
    ax_heatmap.set_xticks(np.arange(len(col_labels_ordered)))
    ax_heatmap.set_xticklabels(col_labels_ordered, rotation=35, ha="right", fontsize=11)
    ax_heatmap.set_yticks(np.arange(len(row_labels_ordered)))
    ax_heatmap.set_yticklabels(row_labels_ordered, fontsize=11)
    ax_heatmap.set_title(
        "Model Performance Heatmap (Z-scored Metrics)", fontsize=15, fontweight="bold", pad=15
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
    cbar.set_label("Z-score", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Annotate cells with values
    if show_values:
        for i in range(z_ordered.shape[0]):
            for j in range(z_ordered.shape[1]):
                val = z_ordered[i, j]
                # Use white text for extreme values, black for moderate
                text_color = "white" if abs(val) > 1.5 else "black"
                ax_heatmap.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Enhanced metric heatmap written: %s", output_path)


# ---------------------------------------------------------------------------
# Visium / spatial spot plots
# ---------------------------------------------------------------------------

# Stage colors — color-blind friendly palette
_STAGE_COLORS: dict[str, str] = {
    "Normal": "#00BA38",  # green (healthy) - colorblind safe
    "AAH": "#F8766D",  # coral (early precursor)
    "AIS": "#619CFF",  # blue (intermediate precursor)
    "MIA": "#E58700",  # orange (late precursor)
    "LUAD": "#A3A500",  # olive (invasive)
    "Unknown": "#999999",  # gray
}


def _get_spot_coords(adata: Any) -> np.ndarray:
    """Return (n_spots, 2) pixel coordinates from obsm['spatial']."""
    if "spatial" not in adata.obsm:
        raise KeyError(
            "adata.obsm['spatial'] not found. "
            "Load Visium data with squidpy.read.visium() or the manual fallback."
        )
    return np.asarray(adata.obsm["spatial"], dtype=float)


def plot_spatial_stage_map(
    adata_spatial: Any,
    output_path: Path,
    sample_id: str | None = None,
    stage_col: str = "stage",
    spot_size: float = 10.0,
    alpha: float = 0.8,
    show_scale_bar: bool = True,
) -> None:
    """Visium spot plot colored by lung cancer stage with enhanced visualization.

    Each spot is colored according to its stage annotation in obs[stage_col].
    Biologically, AAH/AIS spots show the proinflammatory niche signature (KAC + IL1B+
    macrophages) while LUAD spots show the tumor-dominant phenotype (Peng et al. 2026).

    Parameters
    ----------
    adata_spatial : AnnData
        Spatial AnnData with obsm['spatial'] pixel coordinates.
    output_path : Path
        Figure output path (.png).  PDF also saved alongside.
    sample_id : str or None
        Sample identifier for the figure title.
    stage_col : str
        obs column containing stage labels.
    spot_size : float
        Size of scatter points for spots
    alpha : float
        Transparency of spots
    show_scale_bar : bool
        Whether to add a scale bar annotation
    """
    from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER

    output_path = Path(output_path)
    coords = _get_spot_coords(adata_spatial)
    stages = (
        adata_spatial.obs[stage_col].astype(str).values
        if stage_col in adata_spatial.obs.columns
        else np.array(["Unknown"] * adata_spatial.n_obs)
    )

    # Visium convention: col 0 = y-axis (row on tissue), col 1 = x-axis (col on tissue)
    px_y, px_x = coords[:, 0], coords[:, 1]

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(9, 8.5), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    ordered = list(CANONICAL_STAGE_ORDER) + [
        s for s in np.unique(stages) if s not in CANONICAL_STAGE_ORDER
    ]
    for stage in ordered:
        mask = stages == stage
        if not mask.any():
            continue
        ax.scatter(
            px_x[mask],
            -px_y[mask],
            c=_STAGE_COLORS.get(stage, "#999999"),
            s=spot_size,
            alpha=alpha,
            label=stage,
            rasterized=True,
            edgecolors="white",
            linewidths=0.2,
        )

    # Enhanced title and labels
    title = f"Spatial Stage Map — {sample_id}" if sample_id else "Spatial Stage Map"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Spatial X (μm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spatial Y (μm, inverted)", fontsize=12, fontweight="bold")

    # Improved legend with stage counts
    stage_counts = {
        stage: np.sum(stages == stage) for stage in ordered if np.sum(stages == stage) > 0
    }
    legend_labels = [
        f"{stage} (n={stage_counts[stage]})" for stage in ordered if stage in stage_counts
    ]
    handles = [
        plt.scatter(
            [],
            [],
            s=50,
            c=_STAGE_COLORS.get(stage, "#999999"),
            edgecolors="white",
            linewidths=0.5,
            alpha=alpha,
        )
        for stage in ordered
        if stage in stage_counts
    ]
    legend = ax.legend(
        handles,
        legend_labels,
        markerscale=2,
        framealpha=0.95,
        fontsize=11,
        loc="best",
        title="Cancer Stage",
        title_fontsize=12,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.5)

    # Add scale bar if requested
    if show_scale_bar:
        x_range = px_x.max() - px_x.min()
        scale_length = x_range * 0.15  # 15% of width
        scale_x = px_x.min() + x_range * 0.75
        scale_y = -px_y.max() + (px_y.max() - px_y.min()) * 0.08
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], "k-", linewidth=3)
        ax.text(
            scale_x + scale_length / 2,
            scale_y - (px_y.max() - px_y.min()) * 0.03,
            f"{int(scale_length)} μm",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.5)
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
    log.info("Enhanced spatial stage map written: %s", output_path)


def plot_spatial_context_score(
    adata_spatial: Any,
    context_scores: np.ndarray,
    output_path: Path,
    sample_id: str | None = None,
    cmap: str = "viridis",
    spot_size: float = 12.0,
    alpha: float = 0.85,
    show_scale_bar: bool = True,
    add_contours: bool = True,
) -> None:
    """Visium spot plot colored by context vector score with enhanced visualization.

    Biologically, spots with high context scores are expected to localize at
    AAH/AIS lesion edges where KAC-IL1B+ macrophage niches concentrate (Peng 2026).
    Low scores correspond to normal parenchyma or LUAD (depleted inflammatory niche).

    Parameters
    ----------
    adata_spatial : AnnData
        Spatial AnnData with obsm['spatial'].
    context_scores : ndarray, shape (n_spots,)
        Scalar context score per spot — typically ||c_s|| projected onto spatial
        spots via Tangram mapping.
    output_path : Path
    sample_id : str or None
        Sample identifier for the figure title.
    cmap : str
        Colormap name (try 'magma', 'viridis', 'plasma', 'inferno')
    spot_size : float
        Size of scatter points
    alpha : float
        Transparency
    show_scale_bar : bool
        Whether to add scale bar
    add_contours : bool
        Whether to add density contours over the heatmap
    """
    output_path = Path(output_path)
    coords = _get_spot_coords(adata_spatial)
    scores = np.asarray(context_scores, dtype=float).ravel()

    if len(scores) != adata_spatial.n_obs:
        raise ValueError(f"context_scores length {len(scores)} != n_obs {adata_spatial.n_obs}")

    px_y, px_x = coords[:, 0], coords[:, 1]

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(9, 8.5), dpi=150)
    ax.set_facecolor("#F8F8F8")
    fig.patch.set_facecolor("white")

    # Robust percentile-based color scaling
    vmin = np.percentile(scores, 2)
    vmax = np.percentile(scores, 98)

    # Main scatter plot
    sc = ax.scatter(
        px_x,
        -px_y,
        c=scores,
        cmap=cmap,
        s=spot_size,
        alpha=alpha,
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.2,
    )

    # Add contour lines if requested
    if add_contours and len(scores) > 20:
        try:
            from scipy.interpolate import griddata

            # Create grid for interpolation
            grid_x = np.linspace(px_x.min(), px_x.max(), 100)
            grid_y = np.linspace(-px_y.max(), -px_y.min(), 100)
            grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

            # Interpolate scores to grid
            grid_Z = griddata((px_x, -px_y), scores, (grid_X, grid_Y), method="cubic")

            # Draw contours
            contours = ax.contour(
                grid_X, grid_Y, grid_Z, levels=6, colors="white", alpha=0.4, linewidths=1
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")
        except Exception as e:
            log.debug(f"Could not draw contours: {e}")

    # Enhanced colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Context Score ‖c_s‖", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # Title and labels
    title = f"Spatial Context Score — {sample_id}" if sample_id else "Spatial Context Score"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Spatial X (μm)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spatial Y (μm, inverted)", fontsize=12, fontweight="bold")

    # Summary statistics annotation
    stats_text = (
        f"Mean: {scores.mean():.3f}\n"
        f"Median: {np.median(scores):.3f}\n"
        f"Range: [{scores.min():.3f}, {scores.max():.3f}]"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
    )

    # Add scale bar if requested
    if show_scale_bar:
        x_range = px_x.max() - px_x.min()
        scale_length = x_range * 0.15
        scale_x = px_x.min() + x_range * 0.75
        scale_y = -px_y.max() + (px_y.max() - px_y.min()) * 0.08
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], "k-", linewidth=3)
        ax.text(
            scale_x + scale_length / 2,
            scale_y - (px_y.max() - px_y.min()) * 0.03,
            f"{int(scale_length)} μm",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15, linestyle=":", linewidth=0.5)
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
    log.info("Enhanced spatial context score plot written: %s", output_path)


# ---------------------------------------------------------------------------
# Tangram spatial composition plots
# ---------------------------------------------------------------------------


def _extract_tangram_components(
    adata_tangram: Any,
) -> tuple[np.ndarray, list[str], np.ndarray, pd.Series]:
    """Return normalized Tangram scores, labels, coordinates, and sample ids."""
    if "X_tangram_ct" not in adata_tangram.obsm:
        raise KeyError("Expected adata.obsm['X_tangram_ct'] in Tangram spatial output.")
    if "spatial" not in adata_tangram.obsm:
        raise KeyError("Expected adata.obsm['spatial'] coordinates in Tangram spatial output.")

    ct_raw = np.asarray(adata_tangram.obsm["X_tangram_ct"], dtype=np.float32)
    ct_cols = list(adata_tangram.uns.get("tangram_ct_columns", []))
    if len(ct_cols) != ct_raw.shape[1]:
        ct_cols = [f"celltype_{i}" for i in range(ct_raw.shape[1])]

    # Normalize per-spot so each row is a composition over cell types.
    row_sum = ct_raw.sum(axis=1, keepdims=True)
    ct_prop = np.divide(
        ct_raw,
        row_sum,
        out=np.zeros_like(ct_raw),
        where=row_sum > 0,
    )
    coords = np.asarray(adata_tangram.obsm["spatial"], dtype=np.float32)

    if "sample_id" in adata_tangram.obs.columns:
        sample_series = adata_tangram.obs["sample_id"].astype(str)
    else:
        sample_series = pd.Series("all", index=adata_tangram.obs_names)

    return ct_prop, ct_cols, coords, sample_series


def _resolve_sample_mask(
    sample_series: pd.Series,
    sample_id: str | None = None,
) -> tuple[np.ndarray, str]:
    """Return boolean mask and effective sample id."""
    if sample_id is not None:
        sample_id = str(sample_id)
        mask = sample_series.to_numpy() == sample_id
        if not mask.any():
            raise ValueError(f"sample_id={sample_id!r} not found in Tangram spatial output.")
        return mask, sample_id

    chosen = str(sample_series.iloc[0])
    mask = sample_series.to_numpy() == chosen
    return mask, chosen


def plot_tangram_celltype_maps(
    adata_tangram: Any,
    output_path: Path,
    sample_id: str | None = None,
    point_size: float = 2.0,
    cmap: str = "viridis",
) -> str:
    """Plot per-celltype spatial composition maps from Tangram output."""
    output_path = Path(output_path)
    ct_prop, ct_cols, coords, sample_series = _extract_tangram_components(adata_tangram)
    mask, sample_used = _resolve_sample_mask(sample_series, sample_id=sample_id)

    coords_sub = coords[mask]
    ct_sub = ct_prop[mask]
    n_ct = len(ct_cols)
    n_cols = 3
    n_rows = int(np.ceil(n_ct / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.8 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(ct_cols):
        ax = axes[i]
        vals = ct_sub[:, i]
        sc = ax.scatter(
            coords_sub[:, 0],
            coords_sub[:, 1],
            c=vals,
            s=point_size,
            cmap=cmap,
            rasterized=True,
        )
        ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)

    for j in range(n_ct, len(axes)):
        axes[j].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Tangram per-celltype maps written: %s", output_path)
    return sample_used


def plot_tangram_winner_map(
    adata_tangram: Any,
    output_path: Path,
    sample_id: str | None = None,
    point_size: float = 2.0,
    cmap: str = "tab10",
) -> tuple[str, pd.Series]:
    """Plot argmax winner label per spot from normalized Tangram scores."""
    output_path = Path(output_path)
    ct_prop, ct_cols, coords, sample_series = _extract_tangram_components(adata_tangram)
    mask, sample_used = _resolve_sample_mask(sample_series, sample_id=sample_id)

    coords_sub = coords[mask]
    ct_sub = ct_prop[mask]
    winner_idx = ct_sub.argmax(axis=1)
    winner_labels = np.asarray(ct_cols, dtype=object)[winner_idx]
    winner_counts = pd.Series(winner_labels).value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sc = ax.scatter(
        coords_sub[:, 0],
        coords_sub[:, 1],
        c=winner_idx,
        s=point_size,
        cmap=cmap,
        rasterized=True,
    )
    ax.set_title(f"Tangram winner label per spot — {sample_used}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_ticks(np.arange(len(ct_cols)))
    cbar.set_ticklabels(ct_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Tangram winner map written: %s", output_path)
    return sample_used, winner_counts
