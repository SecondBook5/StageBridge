"""Reference visualizations for embedding analysis and quality assessment.

This module provides visualization functions for reference geometry outputs,
supporting both exploratory analysis and publication-quality figures.

All visualizations follow consistent styling for notebook integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)

# Lazy imports for matplotlib to avoid import errors in headless environments
_plt = None
_sns = None


def _get_plt():
    """Lazy import matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt

        _plt = plt
    return _plt


def _get_sns():
    """Lazy import seaborn."""
    global _sns
    if _sns is None:
        import seaborn as sns

        _sns = sns
    return _sns


def plot_reference_structure(
    reference: Any,
    *,
    latent_key: str = "X_scanvi_emb",
    color_by: str | None = None,
    method: Literal["umap", "pca", "tsne"] = "umap",
    title: str = "Reference Structure",
    figsize: tuple[float, float] = (8, 6),
    save_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Plot reference atlas structure using dimensionality reduction.

    Parameters
    ----------
    reference : AnnData or LoadedReference
        Reference atlas data
    latent_key : str
        Key in obsm containing latent embeddings
    color_by : str, optional
        Column in obs to color by
    method : str
        Dimensionality reduction method
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    **kwargs
        Additional arguments passed to scatter plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    # Handle LoadedReference wrapper
    if hasattr(reference, "adata"):
        reference = reference.adata

    if latent_key not in reference.obsm:
        raise KeyError(
            f"Reference missing latent key '{latent_key}'. "
            f"Available: {list(reference.obsm.keys())}"
        )

    latent = np.asarray(reference.obsm[latent_key], dtype=np.float32)

    # Compute 2D embedding
    coords_2d = _compute_2d_embedding(latent, method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors
    if color_by and color_by in reference.obs.columns:
        colors = reference.obs[color_by].astype(str)
        unique_colors = colors.unique()
        n_colors = len(unique_colors)

        if n_colors <= 20:
            # Categorical coloring
            cmap = plt.cm.get_cmap("tab20", n_colors)
            color_map = {c: cmap(i) for i, c in enumerate(unique_colors)}
            c = [color_map[cc] for cc in colors]
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                c=c,
                s=kwargs.get("s", 1),
                alpha=kwargs.get("alpha", 0.5),
                rasterized=True,
            )
            # Add legend
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color_map[c],
                    label=c,
                    markersize=6,
                )
                for c in unique_colors[:20]
            ]
            ax.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
            )
        else:
            # Too many categories, use default coloring
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                c=pd.Categorical(colors).codes,
                cmap="tab20",
                s=kwargs.get("s", 1),
                alpha=kwargs.get("alpha", 0.5),
                rasterized=True,
            )
    else:
        ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=kwargs.get("c", "steelblue"),
            s=kwargs.get("s", 1),
            alpha=kwargs.get("alpha", 0.5),
            rasterized=True,
        )

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        # Save at publication quality (300 DPI) in both PNG and PDF
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        if save_path.suffix.lower() != ".pdf":
            fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        log.info("Saved reference structure plot to %s (PNG + PDF)", save_path)

    return fig


def plot_hlca_structure(
    hlca_reference: Any,
    *,
    color_by: str = "ann_level_2",
    save_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Plot HLCA reference structure.

    Parameters
    ----------
    hlca_reference : AnnData or LoadedReference
        HLCA reference atlas
    color_by : str
        Column to color by (default: ann_level_2 for cell types)
    save_path : str or Path, optional
        Path to save figure
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    return plot_reference_structure(
        hlca_reference,
        latent_key=kwargs.pop("latent_key", "X_scanvi_emb"),
        color_by=color_by,
        title="HLCA Reference Structure",
        save_path=save_path,
        **kwargs,
    )


def plot_luca_structure(
    luca_reference: Any,
    *,
    color_by: str = "cell_type",
    save_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Plot LuCa reference structure.

    Parameters
    ----------
    luca_reference : AnnData or LoadedReference
        LuCa reference atlas
    color_by : str
        Column to color by
    save_path : str or Path, optional
        Path to save figure
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    return plot_reference_structure(
        luca_reference,
        latent_key=kwargs.pop("latent_key", "X_scVI"),
        color_by=color_by,
        title="LuCa Reference Structure",
        save_path=save_path,
        **kwargs,
    )


def plot_query_projection(
    query_embeddings: np.ndarray,
    reference: Any,
    *,
    latent_key: str = "X_scanvi_emb",
    query_labels: np.ndarray | None = None,
    title: str = "Query Projection onto Reference",
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Plot query cells projected onto reference embedding.

    Parameters
    ----------
    query_embeddings : np.ndarray
        Query cell embeddings (n_cells, latent_dim)
    reference : AnnData or LoadedReference
        Reference atlas
    latent_key : str
        Key in reference.obsm
    query_labels : np.ndarray, optional
        Labels for query cells (for coloring)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    **kwargs
        Additional arguments

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    # Handle LoadedReference wrapper
    if hasattr(reference, "adata"):
        reference = reference.adata

    ref_latent = np.asarray(reference.obsm[latent_key], dtype=np.float32)

    # Combine for joint embedding
    combined = np.vstack([ref_latent, query_embeddings])
    coords_2d = _compute_2d_embedding(combined, method="umap")

    n_ref = ref_latent.shape[0]
    ref_coords = coords_2d[:n_ref]
    query_coords = coords_2d[n_ref:]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot reference (gray background)
    ax.scatter(
        ref_coords[:, 0],
        ref_coords[:, 1],
        c="lightgray",
        s=1,
        alpha=0.3,
        label="Reference",
        rasterized=True,
    )

    # Plot query cells
    if query_labels is not None:
        unique_labels = np.unique(query_labels)
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = query_labels == label
            ax.scatter(
                query_coords[mask, 0],
                query_coords[mask, 1],
                c=[cmap(i)],
                s=kwargs.get("s", 5),
                alpha=kwargs.get("alpha", 0.7),
                label=str(label),
                rasterized=True,
            )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.scatter(
            query_coords[:, 0],
            query_coords[:, 1],
            c=kwargs.get("c", "crimson"),
            s=kwargs.get("s", 5),
            alpha=kwargs.get("alpha", 0.7),
            label="Query",
            rasterized=True,
        )
        ax.legend()

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved query projection plot to %s", save_path)

    return fig


def plot_confidence_histogram(
    confidence_scores: Any,
    *,
    figsize: tuple[float, float] = (10, 4),
    save_path: str | Path | None = None,
) -> Any:
    """Plot confidence score histograms for HLCA and LuCa.

    Parameters
    ----------
    confidence_scores : ConfidenceScores or dict
        Confidence scores object or dict with hlca_confidence and luca_confidence
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    # Handle ConfidenceScores object
    if hasattr(confidence_scores, "hlca_confidence"):
        hlca_conf = confidence_scores.hlca_confidence
        luca_conf = confidence_scores.luca_confidence
    else:
        hlca_conf = confidence_scores.get("hlca_confidence")
        luca_conf = confidence_scores.get("luca_confidence")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # HLCA histogram
    axes[0].hist(hlca_conf, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(
        np.median(hlca_conf),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(hlca_conf):.2f}",
    )
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("HLCA Mapping Confidence")
    axes[0].legend()

    # LuCa histogram
    axes[1].hist(luca_conf, bins=50, color="coral", alpha=0.7, edgecolor="white")
    axes[1].axvline(
        np.median(luca_conf),
        color="red",
        linestyle="--",
        label=f"Median: {np.median(luca_conf):.2f}",
    )
    axes[1].set_xlabel("Confidence Score")
    axes[1].set_ylabel("Count")
    axes[1].set_title("LuCa Mapping Confidence")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved confidence histogram to %s", save_path)

    return fig


def plot_donor_colored(
    embeddings: np.ndarray,
    donor_ids: np.ndarray,
    *,
    method: Literal["umap", "pca"] = "umap",
    title: str = "Embeddings by Donor",
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Plot embeddings colored by donor.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings (n_cells, latent_dim)
    donor_ids : np.ndarray
        Donor IDs for each cell
    method : str
        Dimensionality reduction method
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    coords_2d = _compute_2d_embedding(embeddings, method=method)

    unique_donors = np.unique(donor_ids)
    n_donors = len(unique_donors)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap("tab20", min(n_donors, 20))
    for i, donor in enumerate(unique_donors):
        mask = donor_ids == donor
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[cmap(i % 20)],
            s=3,
            alpha=0.5,
            label=donor if n_donors <= 20 else None,
            rasterized=True,
        )

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"{title} (n={n_donors} donors)")

    if n_donors <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved donor-colored plot to %s", save_path)

    return fig


def plot_stage_colored(
    embeddings: np.ndarray,
    stage_ids: np.ndarray,
    *,
    method: Literal["umap", "pca"] = "umap",
    title: str = "Embeddings by Stage",
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
) -> Any:
    """Plot embeddings colored by disease stage.

    Parameters
    ----------
    embeddings : np.ndarray
        Cell embeddings (n_cells, latent_dim)
    stage_ids : np.ndarray
        Stage IDs for each cell
    method : str
        Dimensionality reduction method
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    coords_2d = _compute_2d_embedding(embeddings, method=method)

    # Define stage order for consistent coloring
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD", "Unknown"]
    unique_stages = sorted(
        np.unique(stage_ids),
        key=lambda x: stage_order.index(x) if x in stage_order else len(stage_order),
    )
    n_stages = len(unique_stages)

    fig, ax = plt.subplots(figsize=figsize)

    # Use a diverging colormap for progression
    cmap = plt.cm.get_cmap("RdYlBu_r", n_stages)
    for i, stage in enumerate(unique_stages):
        mask = stage_ids == stage
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[cmap(i)],
            s=5,
            alpha=0.6,
            label=stage,
            rasterized=True,
        )

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(f"{title} (n={n_stages} stages)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved stage-colored plot to %s", save_path)

    return fig


def plot_fused_overview(
    fused_result: Any,
    *,
    method: Literal["umap", "pca"] = "umap",
    figsize: tuple[float, float] = (16, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Create overview plot of fused embeddings.

    Creates a 3-panel figure showing:
    1. Fused embeddings colored by donor
    2. Fused embeddings colored by stage
    3. Fused embeddings colored by reference mode

    Parameters
    ----------
    fused_result : FusedEmbeddingResult
        Fused embedding result
    method : str
        Dimensionality reduction method
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    plt = _get_plt()

    embeddings = fused_result.fused_embeddings
    coords_2d = _compute_2d_embedding(embeddings, method=method)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Donor
    unique_donors = np.unique(fused_result.donor_ids)
    cmap_donor = plt.cm.get_cmap("tab20", min(len(unique_donors), 20))
    for i, donor in enumerate(unique_donors):
        mask = fused_result.donor_ids == donor
        axes[0].scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[cmap_donor(i % 20)],
            s=1,
            alpha=0.5,
            rasterized=True,
        )
    axes[0].set_title("By Donor")
    axes[0].set_xlabel(f"{method.upper()} 1")
    axes[0].set_ylabel(f"{method.upper()} 2")

    # Panel 2: Stage
    stage_order = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
    unique_stages = sorted(
        np.unique(fused_result.stage_ids),
        key=lambda x: stage_order.index(x) if x in stage_order else len(stage_order),
    )
    cmap_stage = plt.cm.get_cmap("RdYlBu_r", len(unique_stages))
    for i, stage in enumerate(unique_stages):
        mask = fused_result.stage_ids == stage
        axes[1].scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[cmap_stage(i)],
            s=1,
            alpha=0.5,
            label=stage,
            rasterized=True,
        )
    axes[1].set_title("By Stage")
    axes[1].set_xlabel(f"{method.upper()} 1")
    axes[1].legend(loc="upper right", fontsize=8)

    # Panel 3: Reference mode
    if fused_result.reference_mode_used is not None:
        modes = fused_result.reference_mode_used
        mode_colors = {"hlca": "steelblue", "luca": "coral", "both": "green"}
        for mode, color in mode_colors.items():
            mask = modes == mode
            if mask.any():
                axes[2].scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=color,
                    s=1,
                    alpha=0.5,
                    label=f"{mode} ({mask.sum()})",
                    rasterized=True,
                )
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("By Reference Mode")
    axes[2].set_xlabel(f"{method.upper()} 1")

    plt.suptitle("Fused Embedding Overview", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved fused overview plot to %s", save_path)

    return fig


def _compute_2d_embedding(
    latent: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D embedding for visualization.

    Parameters
    ----------
    latent : np.ndarray
        High-dimensional embeddings (n_cells, latent_dim)
    method : str
        Method: "umap", "pca", or "tsne"
    random_state : int
        Random seed

    Returns
    -------
    np.ndarray
        2D coordinates (n_cells, 2)
    """
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(latent)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=random_state).fit_transform(latent)
    elif method == "umap":
        try:
            from umap import UMAP

            return UMAP(n_components=2, random_state=random_state).fit_transform(latent)
        except ImportError:
            log.warning("UMAP not available, falling back to PCA")
            from sklearn.decomposition import PCA

            return PCA(n_components=2, random_state=random_state).fit_transform(latent)
    else:
        raise ValueError(f"Unknown method: {method}")
