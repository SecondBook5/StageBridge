"""Poster-ready panel plotting and spatial Visium visualizations for StageBridge."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

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

    ax1.plot(x, eval_df["sinkhorn"].astype(float).values, marker="o", color="#0EA5E9", label="Sinkhorn")
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
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_heatmap(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Plot model-vs-metric heatmap for panel D interpretability summary."""
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

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(z, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(metrics_df.shape[0]))
    ax.set_yticklabels(metrics_df["label"].astype(str).tolist())
    ax.set_title("Panel D: Comparative Metric Signature (Interpretability Summary)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("z-score")

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visium / spatial spot plots
# ---------------------------------------------------------------------------

# Stage colors — same palette as embeddings.py
_STAGE_COLORS: dict[str, str] = {
    "Normal": "#4ADE80",
    "AAH":    "#FACC15",
    "AIS":    "#FB923C",
    "MIA":    "#F87171",
    "LUAD":   "#7F1D1D",
    "Unknown": "#94A3B8",
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
    alpha: float = 0.7,
) -> None:
    """Visium spot plot colored by lung cancer stage.

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
    """
    from stagebridge.preprocessing.stage_ontology import CANONICAL_STAGE_ORDER

    output_path = Path(output_path)
    coords = _get_spot_coords(adata_spatial)
    stages = (
        adata_spatial.obs[stage_col].astype(str).values
        if stage_col in adata_spatial.obs.columns
        else np.array(["Unknown"] * adata_spatial.n_obs)
    )

    # Visium convention: col 0 = y-axis (row on tissue), col 1 = x-axis (col on tissue)
    px_y, px_x = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ordered = list(CANONICAL_STAGE_ORDER) + [s for s in np.unique(stages) if s not in CANONICAL_STAGE_ORDER]
    for stage in ordered:
        mask = stages == stage
        if not mask.any():
            continue
        ax.scatter(
            px_x[mask], -px_y[mask],
            c=_STAGE_COLORS.get(stage, "#94A3B8"),
            s=spot_size, alpha=alpha, label=stage, rasterized=True,
        )

    title = f"Spatial Stage Map — {sample_id}" if sample_id else "Spatial Stage Map"
    ax.set_title(title)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row (inverted)")
    ax.legend(markerscale=2, framealpha=0.85, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Spatial stage map written: %s", output_path)


def plot_spatial_context_score(
    adata_spatial: Any,
    context_scores: np.ndarray,
    output_path: Path,
    sample_id: str | None = None,
    cmap: str = "magma",
    spot_size: float = 12.0,
    alpha: float = 0.8,
) -> None:
    """Visium spot plot colored by context vector score (L2 norm of c_s).

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
    """
    output_path = Path(output_path)
    coords = _get_spot_coords(adata_spatial)
    scores = np.asarray(context_scores, dtype=float).ravel()

    if len(scores) != adata_spatial.n_obs:
        raise ValueError(
            f"context_scores length {len(scores)} != n_obs {adata_spatial.n_obs}"
        )

    px_y, px_x = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    sc = ax.scatter(
        px_x, -px_y,
        c=scores, cmap=cmap, s=spot_size, alpha=alpha, rasterized=True,
        vmin=np.percentile(scores, 2), vmax=np.percentile(scores, 98),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Context score (‖c_s‖)")
    title = f"Spatial Context Score — {sample_id}" if sample_id else "Spatial Context Score"
    ax.set_title(title)
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row (inverted)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    log.info("Spatial context score plot written: %s", output_path)
