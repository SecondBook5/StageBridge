"""UMAP and trajectory embedding visualizations for StageBridge.

Poster-quality figures showing:
  - UMAP scatter colored by lung cancer stage (Normal→AAH→AIS→MIA→LUAD)
  - Predicted cell-state trajectories overlaid as quiver arrows on UMAP
  - Context vector c_s UMAP (dimensionality-reduced context embeddings)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from stagebridge.logging_utils import get_logger
from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Stage color palette — ordered Normal→AAH→AIS→MIA→LUAD
_STAGE_COLORS: dict[str, str] = {
    "Normal": "#4ADE80",   # green (healthy)
    "AAH":    "#FACC15",   # yellow (early precursor)
    "AIS":    "#FB923C",   # orange (intermediate precursor)
    "MIA":    "#F87171",   # light red (late precursor)
    "LUAD":   "#7F1D1D",   # dark red (invasive)
    "Unknown": "#94A3B8",  # slate
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


def _stage_scatter(ax: plt.Axes, coords: np.ndarray, stages: np.ndarray, s: float, alpha: float) -> None:
    """Draw per-stage scatter ensuring canonical order in legend."""
    ordered = list(CANONICAL_STAGE_ORDER) + [
        st for st in np.unique(stages) if st not in CANONICAL_STAGE_ORDER
    ]
    for stage in ordered:
        mask = stages == stage
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=_STAGE_COLORS.get(stage, "#94A3B8"),
            s=s, alpha=alpha, label=stage, rasterized=True,
        )


def plot_umap_by_stage(
    adata: Any,
    output_path: Path,
    title: str = "UMAP by Stage",
    stage_col: str = "stage",
    point_size: float = 3.0,
    alpha: float = 0.6,
) -> None:
    """Scatter UMAP colored by lung cancer stage.

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
    """
    output_path = Path(output_path)
    coords = _get_umap_coords(adata)
    stages = (
        adata.obs[stage_col].astype(str).values
        if stage_col in adata.obs.columns
        else np.array(["Unknown"] * adata.n_obs)
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _stage_scatter(ax, coords, stages, s=point_size, alpha=alpha)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("UMAP by stage written: %s", output_path)


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
) -> None:
    """Overlay Euler-integrated trajectory arrows on UMAP.

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

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _stage_scatter(ax, bg_coords, stages, s=2.0, alpha=0.3)

    # Subsample arrows for clarity
    rng = np.random.default_rng(42)
    idx = rng.choice(len(uv0), size=min(n_arrows, len(uv0)), replace=False)
    dx = uv1_pred[idx, 0] - uv0[idx, 0]
    dy = uv1_pred[idx, 1] - uv0[idx, 1]
    ax.quiver(
        uv0[idx, 0], uv0[idx, 1], dx, dy,
        angles="xy", scale_units="xy", scale=1,
        color=arrow_color, alpha=arrow_alpha, width=0.004,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("UMAP with trajectories written: %s", output_path)


def plot_context_vector_umap(
    adata: Any,
    context_vectors: np.ndarray,
    output_path: Path,
    title: str = "Context Vector UMAP (c_s)",
    stage_col: str = "stage",
    point_size: float = 6.0,
    alpha: float = 0.65,
) -> None:
    """2-D UMAP of per-cell context vectors c_s, colored by stage.

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
    """
    output_path = Path(output_path)
    cv = np.asarray(context_vectors, dtype=np.float32)

    # Reduce to 2-D: try umap-learn, fall back to PCA
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
        coords = np.asarray(reducer.fit_transform(cv), dtype=np.float32)
        embed_label = "UMAP"
    except ImportError:
        log.warning("umap-learn not available; using PCA for context vector embedding.")
        from sklearn.decomposition import PCA
        coords = np.asarray(
            PCA(n_components=2).fit_transform(cv), dtype=np.float32
        )
        embed_label = "PC"

    stages = (
        adata.obs[stage_col].astype(str).values
        if stage_col in adata.obs.columns
        else np.array(["Unknown"] * len(cv))
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    _stage_scatter(ax, coords, stages, s=point_size, alpha=alpha)
    ax.set_xlabel(f"Context {embed_label} 1")
    ax.set_ylabel(f"Context {embed_label} 2")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Context vector UMAP written: %s", output_path)
