"""Poster figure assembly for StageBridge HCA presentation.

Generates publication-quality multi-panel figures that tell the StageBridge story:

  Panel A — Model architecture schematic (Set Transformer + OT flow matching)
  Panel B — Donor-held-out benchmark (StageBridge vs. baselines, all transitions)
  Panel C — Context sensitivity by transition (peaks at AAH/AIS per Peng 2026 biology)
  Panel D — Context vector UMAP + top gene-context correlation heatmap

Design principles:
  - Uses a dark-on-light poster palette with a distinct teal accent for StageBridge
  - All panels export at 300 DPI as both PNG and PDF
  - Text scaled for A0 poster at 40% reduction (~28pt display)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Poster color system
# ---------------------------------------------------------------------------
PALETTE = {
    "stagebridge": "#0E7490",   # teal — StageBridge primary
    "baseline":    "#334155",   # slate — baselines
    "ablation":    "#64748B",   # light slate — ablations
    "accent":      "#F59E0B",   # amber — highlight / context sensitivity
    "bg":          "#F8FAFC",   # near-white background
    "text":        "#0F172A",   # near-black text
    "grid":        "#CBD5E1",   # subtle grid
    # Stage progression colors (Normal→LUAD)
    "Normal":      "#4ADE80",
    "AAH":         "#FACC15",
    "AIS":         "#FB923C",
    "MIA":         "#F87171",
    "LUAD":        "#7F1D1D",
}

FONT_TITLE   = {"fontsize": 15, "fontweight": "bold", "color": PALETTE["text"]}
FONT_LABEL   = {"fontsize": 12, "color": PALETTE["text"]}
FONT_TICK    = {"labelsize": 10}
FONT_ANNOT   = {"fontsize": 9, "color": PALETTE["text"]}

# Canonical transitions for x-axis labels
TRANSITIONS = ["Normal→AAH", "AAH→AIS", "AIS→MIA", "MIA→LUAD"]


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=PALETTE["bg"])
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    log.info("Poster figure written: %s", output_path)


# ---------------------------------------------------------------------------
# Panel A: Architecture schematic
# ---------------------------------------------------------------------------

def make_panel_a_model_diagram(output_path: Path) -> None:
    """Architecture schematic: cross-sectional set → ISAB/PMA → c_s → OT + FM → predicted cells.

    Shows the full StageBridge pipeline as a horizontal flow diagram with
    annotated architectural components and the key innovation (population context).
    """
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)

    def _box(cx: float, cy: float, w: float, h: float, label: str, sublabel: str = "",
             color: str = "#E2E8F0", textcolor: str = PALETTE["text"]) -> None:
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.8, edgecolor=PALETTE["text"],
            facecolor=color, alpha=0.95, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(cx, cy + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=textcolor, zorder=3)
        if sublabel:
            ax.text(cx, cy - 0.32, sublabel, ha="center", va="center",
                    fontsize=9.5, color=textcolor, style="italic", zorder=3)

    def _arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate("", xy=(x1, y0), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=PALETTE["text"]),
                    zorder=4)
        if label:
            mx = (x0 + x1) / 2
            ax.text(mx, y0 + 0.28, label, ha="center", va="bottom",
                    fontsize=8.5, color=PALETTE["text"], style="italic")

    cy = 2.75

    # Block 1: Input sets
    _box(1.4, cy, 2.2, 2.2,
         "Source cells", "cross-sectional\n(snRNA-seq)",
         color="#DBEAFE")

    # Block 2: Set Transformer encoder
    _box(4.3, cy, 2.4, 2.2,
         "Set Transformer", "ISAB × 2 + SAB\n+ PMA → c_s",
         color=PALETTE["stagebridge"], textcolor="white")

    # Block 3: OT coupling
    _box(7.1, cy, 2.0, 2.2,
         "OT coupling", "Sinkhorn\npseudo-pairs",
         color="#E0F2FE")

    # Block 4: Flow matching
    _box(9.9, cy, 2.2, 2.2,
         "Flow matching", "v_φ(x,t,c_s,s)\nFiLM conditioned",
         color="#FEF3C7")

    # Block 5: Output
    _box(12.6, cy, 2.2, 2.2,
         "Predicted cells", "target stage\ndistribution",
         color="#DCFCE7")

    # Arrows
    _arrow(2.5, cy, 3.1, cy)
    _arrow(5.5, cy, 6.1, cy)
    _arrow(8.1, cy, 9.0, cy)
    _arrow(11.0, cy, 11.5, cy)

    # Context vector annotation
    ax.annotate(
        "context\nvector c_s",
        xy=(4.3, cy - 1.1), xytext=(4.3, 0.6),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PALETTE["accent"],
                        connectionstyle="arc3,rad=0.0"),
        fontsize=9, color=PALETTE["accent"], ha="center", fontweight="bold",
        zorder=5,
    )
    # c_s feeds into flow matching
    ax.annotate(
        "",
        xy=(9.9, cy - 1.1), xytext=(4.3, cy - 1.1),
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PALETTE["accent"],
                        linestyle="dashed", connectionstyle="arc3,rad=-0.15"),
        zorder=5,
    )
    ax.text(7.1, 0.55, "population context conditions trajectory", ha="center",
            fontsize=9, color=PALETTE["accent"], fontstyle="italic")

    ax.set_title(
        "StageBridge: Population-Context-Conditioned OT Flow Matching",
        **FONT_TITLE, pad=14,
    )
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Panel B: Benchmark comparison
# ---------------------------------------------------------------------------

def make_panel_b_benchmark(
    results: dict | pd.DataFrame,
    output_path: Path,
    primary_metric: str = "sinkhorn_mean",
    std_metric: str = "sinkhorn_std",
    title: str = "Panel B: Donor-Held-Out Benchmark",
) -> None:
    """Grouped bar chart comparing StageBridge against baselines/ablations.

    StageBridge bars are highlighted in teal; baselines in slate; ablations in light slate.
    Error bars show cross-fold standard deviation.

    Parameters
    ----------
    results : dict or DataFrame
        Either a metrics JSON payload dict (with key 'results') or a pre-built DataFrame
        with columns including 'label', primary_metric, and optionally std_metric.
    output_path : Path
    primary_metric : str
        Column name for the primary metric (lower = better for Sinkhorn/MMD).
    """
    from stagebridge.viz.curves import build_metrics_dataframe

    if isinstance(results, dict):
        df = build_metrics_dataframe(results)
    else:
        df = results.copy()

    if df.empty or primary_metric not in df.columns:
        raise ValueError(f"Cannot plot benchmark: missing '{primary_metric}' in df")

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    x = np.arange(len(df))
    y = df[primary_metric].astype(float).values
    yerr = df[std_metric].astype(float).values if std_metric in df.columns else None

    # Color by model type
    colors = []
    for lbl in df["label"].astype(str):
        if "stagebridge" in lbl.lower():
            colors.append(PALETTE["stagebridge"])
        elif "ablat" in lbl.lower() or lbl.startswith("no_"):
            colors.append(PALETTE["ablation"])
        else:
            colors.append(PALETTE["baseline"])

    bars = ax.bar(x, y, yerr=yerr, color=colors, alpha=0.92, capsize=4,
                  edgecolor=PALETTE["text"], linewidth=0.8, width=0.65)

    # Annotate StageBridge bar value
    sb_idx = [i for i, lbl in enumerate(df["label"].astype(str)) if "stagebridge" in lbl.lower()]
    for i in sb_idx:
        ax.text(x[i], y[i] + (yerr[i] if yerr is not None else 0) + 0.002,
                f"{y[i]:.3f}", ha="center", va="bottom", fontsize=9,
                color=PALETTE["stagebridge"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].astype(str).tolist(), rotation=28, ha="right",
                       **FONT_ANNOT)
    ax.tick_params(**FONT_TICK)
    ax.set_ylabel(primary_metric.replace("_", " ").title(), **FONT_LABEL)
    ax.set_title(title, **FONT_TITLE)
    ax.grid(axis="y", alpha=0.3, color=PALETTE["grid"])
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=PALETTE["stagebridge"], label="StageBridge (ours)"),
        mpatches.Patch(color=PALETTE["baseline"], label="Baselines"),
        mpatches.Patch(color=PALETTE["ablation"], label="Ablations"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Panel C: Context sensitivity by transition
# ---------------------------------------------------------------------------

def make_panel_c_context_sensitivity(
    sensitivity_dict: dict[str, float],
    output_path: Path,
    title: str = "Panel C: Context Sensitivity by Transition",
) -> None:
    """Bar chart showing context sensitivity scores per transition.

    Context sensitivity = Δ Sinkhorn(real context) − Sinkhorn(shuffled context).
    A larger gap means population context matters more for that transition.

    Biologically, we expect the highest sensitivity at Normal→AAH and AAH→AIS
    transitions (where the IL1B-driven proinflammatory niche is most compositionally
    distinct from shuffled; Peng et al. 2026 Cancer Cell).

    Parameters
    ----------
    sensitivity_dict : dict
        {transition_label: sensitivity_score}, e.g. {"Normal→AAH": 0.12, ...}
    output_path : Path
    """
    transitions = list(sensitivity_dict.keys())
    scores = np.array([sensitivity_dict[t] for t in transitions], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    x = np.arange(len(transitions))
    # Color bars by expected biological significance
    bar_colors = [PALETTE["accent"] if score == scores.max() else PALETTE["stagebridge"]
                  for score in scores]

    bars = ax.bar(x, scores, color=bar_colors, alpha=0.92,
                  edgecolor=PALETTE["text"], linewidth=0.8, width=0.6)

    # Annotate values above bars
    for i, (xi, s) in enumerate(zip(x, scores)):
        ax.text(xi, s + scores.max() * 0.015, f"{s:.3f}", ha="center", va="bottom",
                fontsize=9, color=PALETTE["text"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(transitions, rotation=20, ha="right", **FONT_ANNOT)
    ax.tick_params(**FONT_TICK)
    ax.set_ylabel("Δ Sinkhorn (real − shuffled context)", **FONT_LABEL)
    ax.set_title(title, **FONT_TITLE)
    ax.grid(axis="y", alpha=0.3, color=PALETTE["grid"])
    ax.spines[["top", "right"]].set_visible(False)

    # Biological annotation on peak bar
    if len(scores) > 0:
        peak_idx = int(np.argmax(scores))
        ax.annotate(
            "peak: IL1B-IL1R1\nniche stage",
            xy=(x[peak_idx], scores[peak_idx]),
            xytext=(x[peak_idx] + 0.7, scores[peak_idx] + scores.max() * 0.08),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=PALETTE["accent"]),
            fontsize=8.5, color=PALETTE["accent"], ha="left",
        )

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Panel D: Gene-context correlation heatmap
# ---------------------------------------------------------------------------

def make_panel_d_gene_context_heatmap(
    gene_corr_df: pd.DataFrame,
    output_path: Path,
    title: str = "Panel D: Top Gene–Context Correlations",
    n_genes: int = 20,
    n_dims: int = 8,
) -> None:
    """Heatmap of top gene–context dimension Pearson correlations.

    Rows = genes (top n_genes by max absolute correlation).
    Columns = context dimensions (top n_dims).
    Expected pattern: KAC markers (KRT8, CEACAM5), macrophage signals (IL1B, CCL2),
    NF-κB genes (RELA, NFKB1), and AT2 depletion markers (SFTPC) should appear
    in the top rows, reflecting the IL1B-driven proinflammatory niche.

    Parameters
    ----------
    gene_corr_df : DataFrame
        Shape (n_genes_total, n_context_dims) — Pearson r values.
        Index = gene names, columns = context dimension labels.
    output_path : Path
    n_genes : int
        Show top n genes by max |r| across all dimensions.
    n_dims : int
        Show top n_dims context dimensions by variance.
    """
    # Select top context dims by variance
    col_var = gene_corr_df.var(axis=0)
    top_dims = col_var.nlargest(min(n_dims, len(col_var))).index.tolist()
    sub = gene_corr_df[top_dims]

    # Select top genes by max |r|
    max_abs = sub.abs().max(axis=1)
    top_genes = max_abs.nlargest(min(n_genes, len(max_abs))).index.tolist()
    mat = sub.loc[top_genes].values.astype(float)

    fig_h = max(5.5, len(top_genes) * 0.32 + 1.5)
    fig_w = max(8.0, len(top_dims) * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(top_dims)))
    ax.set_xticklabels(top_dims, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=9)
    ax.tick_params(**FONT_TICK)

    # Annotate cells with r values (only if matrix is small enough)
    if mat.shape[0] * mat.shape[1] <= 200:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(mat[i, j]) > 0.5 else PALETTE["text"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label("Pearson r", fontsize=10)

    ax.set_title(title, **FONT_TITLE, pad=12)
    ax.set_xlabel("Context dimension", **FONT_LABEL)
    ax.set_ylabel("Gene", **FONT_LABEL)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Full 4-panel poster figure assembly
# ---------------------------------------------------------------------------

def make_full_poster(
    panel_paths: dict[str, Path],
    output_path: Path,
) -> None:
    """Assemble pre-rendered panel images into a single 4-panel poster figure.

    Parameters
    ----------
    panel_paths : dict
        Keys: "A", "B", "C", "D"  — paths to PNG panel images.
    output_path : Path
        Output path for the assembled poster figure.
    """
    from PIL import Image

    fig = plt.figure(figsize=(22, 14), facecolor=PALETTE["bg"])
    fig.patch.set_facecolor(PALETTE["bg"])

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.06,
                           left=0.02, right=0.98, top=0.94, bottom=0.02)

    panel_labels = [("A", 0, 0), ("B", 0, 1), ("C", 1, 0), ("D", 1, 1)]
    for key, row, col in panel_labels:
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PALETTE["bg"])
        if key in panel_paths and panel_paths[key].exists():
            img = np.asarray(Image.open(panel_paths[key]))
            ax.imshow(img, aspect="auto")
        else:
            ax.text(0.5, 0.5, f"Panel {key}\n(not yet generated)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="#94A3B8")
        ax.axis("off")
        ax.text(0.01, 0.99, key, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left",
                color=PALETTE["text"])

    fig.suptitle(
        "StageBridge: Population-Context-Conditioned Cell-State Transition Modeling\n"
        "Reveals Microenvironmental Drivers of Lung Pre-Cancer Progression",
        fontsize=16, fontweight="bold", color=PALETTE["text"], y=0.98,
    )
    _save(fig, output_path)
    log.info("Full 4-panel poster written: %s", output_path)
