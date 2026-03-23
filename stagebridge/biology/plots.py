"""
Biology-facing visualizations for StageBridge.

Publication-quality plots for biological interpretation.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# Stage colors - LungPCA canonical palette (Peng et al. Nature 2024)
# Import from authoritative source for consistency
from stagebridge.viz.lungpca_style import STAGE_COLORS, STAGE_ORDER


def _setup_style():
    """Set up publication-quality plot style."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
    })


def plot_signature_scores_by_stage(
    adata: Any,
    signatures: list[str],
    stage_col: str = "stage",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Plot signature scores across progression stages.

    Creates violin plots showing score distribution per stage.

    Parameters
    ----------
    adata : AnnData
        Data with signature scores in obs
    signatures : list
        Signature names (must exist as sig_* columns in obs)
    stage_col : str
        Stage column
    save_path : Path, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()

    # Get signature columns
    sig_cols = [f"sig_{s}" for s in signatures if f"sig_{s}" in adata.obs.columns]
    if not sig_cols:
        raise ValueError("No signature scores found in adata.obs")

    n_sigs = len(sig_cols)
    n_cols = min(3, n_sigs)
    n_rows = (n_sigs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_sigs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]
    palette = {s: STAGE_COLORS.get(s, "#999999") for s in stages}

    for i, col in enumerate(sig_cols):
        ax = axes[i]
        sig_name = col.replace("sig_", "")

        df = pd.DataFrame({
            "score": adata.obs[col].values,
            "stage": adata.obs[stage_col].values,
        })
        df = df[df["stage"].isin(stages)]

        sns.violinplot(
            data=df,
            x="stage",
            y="score",
            order=stages,
            palette=palette,
            ax=ax,
            inner="box",
            scale="width",
        )

        ax.set_title(sig_name.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Score (z-score)")
        ax.tick_params(axis="x", rotation=45)

    # Hide empty axes
    for i in range(len(sig_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved signature scores plot: {save_path}")
    plt.close()


def plot_niche_biology_heatmap(
    niche_biology_df: pd.DataFrame,
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    top_n: int = 20,
) -> None:
    """
    Plot heatmap of niche-biology correlations.

    Parameters
    ----------
    niche_biology_df : DataFrame
        Output from correlate_niche_influence_with_biology
    save_path : Path, optional
        Path to save figure
    top_n : int
        Number of top pathways to show
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_style()

    # Get top pathways by significance
    df = niche_biology_df.head(top_n).copy()
    df = df.set_index("pathway")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Correlation values
    corr_values = df["spearman_rho"].values.reshape(-1, 1)

    # Significance annotations
    annot = []
    for i, (_, row) in enumerate(df.iterrows()):
        pval = row["spearman_pval"]
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        annot.append([f"{row['spearman_rho']:.2f}{stars}"])

    sns.heatmap(
        corr_values,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        yticklabels=df.index,
        xticklabels=["Niche Influence"],
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )

    ax.set_title("Niche Influence - Pathway Correlations\n(* p<0.05, ** p<0.01, *** p<0.001)")
    ax.set_ylabel("Pathway")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved niche-biology heatmap: {save_path}")
    plt.close()


def plot_pathway_activity_ridge(
    adata: Any,
    pathways: list[str],
    stage_col: str = "stage",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 12),
) -> None:
    """
    Ridge plot of pathway activity distributions by stage.

    Parameters
    ----------
    adata : AnnData
        Data with signature scores
    pathways : list
        Pathways to plot
    stage_col : str
        Stage column
    save_path : Path, optional
        Output path
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    _setup_style()

    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]
    n_pathways = len(pathways)

    fig, axes = plt.subplots(n_pathways, 1, figsize=figsize, sharex=True)
    if n_pathways == 1:
        axes = [axes]

    for i, pathway in enumerate(pathways):
        ax = axes[i]
        col = f"sig_{pathway}"

        if col not in adata.obs.columns:
            ax.set_visible(False)
            continue

        # KDE for each stage
        for j, stage in enumerate(stages):
            mask = adata.obs[stage_col] == stage
            values = adata.obs.loc[mask, col].values

            if len(values) < 10:
                continue

            # Compute KDE
            kde = stats.gaussian_kde(values)
            x = np.linspace(values.min() - 1, values.max() + 1, 200)
            y = kde(x)

            # Offset and fill
            offset = j * 0.3
            color = STAGE_COLORS.get(stage, "#999999")
            ax.fill_between(x, offset, y + offset, alpha=0.6, color=color, label=stage)
            ax.plot(x, y + offset, color=color, linewidth=1)

        ax.set_ylabel(pathway.replace("_", "\n"), fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Score (z-score)")
    plt.suptitle("Pathway Activity Distributions by Stage", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved pathway ridge plot: {save_path}")
    plt.close()


def plot_emt_caf_immune_triangle(
    adata: Any,
    stage_col: str = "stage",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Ternary plot showing EMT/CAF/Immune balance by stage.

    Parameters
    ----------
    adata : AnnData
        Data with signature scores
    stage_col : str
        Stage column
    save_path : Path, optional
        Output path
    """
    import matplotlib.pyplot as plt

    _setup_style()

    # Check required signatures
    required = ["sig_emt_hallmark", "sig_caf_general", "sig_macrophage_m2"]
    missing = [r for r in required if r not in adata.obs.columns]
    if missing:
        log.warning(f"Missing signatures for triangle plot: {missing}")
        return

    # Normalize to positive values (shift by min)
    emt = adata.obs["sig_emt_hallmark"].values
    caf = adata.obs["sig_caf_general"].values
    immune = adata.obs["sig_macrophage_m2"].values

    emt = emt - emt.min() + 0.1
    caf = caf - caf.min() + 0.1
    immune = immune - immune.min() + 0.1

    # Normalize to sum to 1
    total = emt + caf + immune
    emt / total
    caf_norm = caf / total
    immune_norm = immune / total

    # Convert to cartesian coordinates for ternary
    x = 0.5 * (2 * caf_norm + immune_norm)
    y = (np.sqrt(3) / 2) * immune_norm

    fig, ax = plt.subplots(figsize=figsize)

    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]

    for stage in stages:
        mask = adata.obs[stage_col] == stage
        color = STAGE_COLORS.get(stage, "#999999")

        ax.scatter(
            x[mask], y[mask],
            c=color,
            label=stage,
            alpha=0.5,
            s=10,
        )

        # Stage centroid
        cx, cy = x[mask].mean(), y[mask].mean()
        ax.scatter(cx, cy, c=color, s=200, marker="*", edgecolor="black", linewidth=1)

    # Draw triangle
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
        fill=False, edgecolor="black", linewidth=2,
    )
    ax.add_patch(triangle)

    # Labels
    ax.text(0, -0.05, "EMT", ha="center", fontsize=12, fontweight="bold")
    ax.text(1, -0.05, "CAF", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, np.sqrt(3)/2 + 0.05, "Immune", ha="center", fontsize=12, fontweight="bold")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("EMT / CAF / Immune Balance by Stage", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved triangle plot: {save_path}")
    plt.close()


def plot_stage_pathway_radar(
    pathway_stage_df: pd.DataFrame,
    pathways: list[str],
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> None:
    """
    Radar chart comparing pathway activity across stages.

    Parameters
    ----------
    pathway_stage_df : DataFrame
        Output from compare_pathway_activity_by_stage
    pathways : list
        Pathways to include
    save_path : Path, optional
        Output path
    """
    import matplotlib.pyplot as plt
    from math import pi

    _setup_style()

    # Filter to selected pathways
    df = pathway_stage_df[pathway_stage_df["pathway"].isin(pathways)].copy()

    if len(df) == 0:
        log.warning("No pathways found for radar plot")
        return

    # Get stage columns
    stage_cols = [c for c in df.columns if c.startswith("mean_")]
    stages = [c.replace("mean_", "") for c in stage_cols]

    # Prepare data
    categories = df["pathway"].tolist()
    N = len(categories)

    # Angles for radar
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for stage in stages:
        if stage not in STAGE_ORDER:
            continue

        col = f"mean_{stage}"
        if col not in df.columns:
            continue

        values = df[col].tolist()
        values += values[:1]

        color = STAGE_COLORS.get(stage, "#999999")
        ax.plot(angles, values, "o-", linewidth=2, label=stage, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.set_title("Pathway Activity by Stage", size=14, pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved radar plot: {save_path}")
    plt.close()


def plot_biological_summary_panel(
    adata: Any,
    influence_df: pd.DataFrame,
    niche_biology_df: pd.DataFrame,
    stage_col: str = "stage",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> None:
    """
    Multi-panel biological summary figure.

    Combines:
    A) Signature scores by stage
    B) Niche-biology heatmap
    C) Influence distribution
    D) Key findings summary

    Parameters
    ----------
    adata : AnnData
        Expression data with signatures
    influence_df : DataFrame
        Niche influence scores
    niche_biology_df : DataFrame
        Niche-biology correlations
    stage_col : str
        Stage column
    save_path : Path, optional
        Output path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    _setup_style()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.8])

    stages = [s for s in STAGE_ORDER if s in adata.obs[stage_col].unique()]
    palette = {s: STAGE_COLORS.get(s, "#999999") for s in stages}

    # Panel A: EMT score by stage
    ax_a = fig.add_subplot(gs[0, 0])
    if "sig_emt_hallmark" in adata.obs.columns:
        df = pd.DataFrame({
            "score": adata.obs["sig_emt_hallmark"].values,
            "stage": adata.obs[stage_col].values,
        })
        df = df[df["stage"].isin(stages)]
        sns.violinplot(data=df, x="stage", y="score", order=stages, palette=palette, ax=ax_a)
        ax_a.set_title("A) EMT Score by Stage")
        ax_a.set_xlabel("")
        ax_a.set_ylabel("EMT Score")

    # Panel B: Immune score by stage
    ax_b = fig.add_subplot(gs[0, 1])
    if "sig_macrophage_m2" in adata.obs.columns:
        df = pd.DataFrame({
            "score": adata.obs["sig_macrophage_m2"].values,
            "stage": adata.obs[stage_col].values,
        })
        df = df[df["stage"].isin(stages)]
        sns.violinplot(data=df, x="stage", y="score", order=stages, palette=palette, ax=ax_b)
        ax_b.set_title("B) M2 Macrophage Score by Stage")
        ax_b.set_xlabel("")
        ax_b.set_ylabel("Score")

    # Panel C: Top niche-biology correlations
    ax_c = fig.add_subplot(gs[0, 2])
    top_df = niche_biology_df.head(10)
    colors = ["#d73027" if r > 0 else "#4575b4" for r in top_df["spearman_rho"]]
    ax_c.barh(range(len(top_df)), top_df["spearman_rho"].values, color=colors)
    ax_c.set_yticks(range(len(top_df)))
    ax_c.set_yticklabels(top_df["pathway"].values, fontsize=8)
    ax_c.set_xlabel("Spearman rho")
    ax_c.set_title("C) Niche-Biology\nCorrelations")
    ax_c.axvline(0, color="black", linewidth=0.5)
    ax_c.invert_yaxis()

    # Panel D: Influence by stage
    ax_d = fig.add_subplot(gs[1, 0])
    if "stage" in influence_df.columns and "ring_influence" in influence_df.columns:
        inf_df = influence_df[influence_df["stage"].isin(stages)]
        sns.boxplot(data=inf_df, x="stage", y="ring_influence", order=stages, palette=palette, ax=ax_d)
        ax_d.set_title("D) Niche Influence by Stage")
        ax_d.set_xlabel("")
        ax_d.set_ylabel("Influence Score")

    # Panel E: CAF score by stage
    ax_e = fig.add_subplot(gs[1, 1])
    if "sig_caf_general" in adata.obs.columns:
        df = pd.DataFrame({
            "score": adata.obs["sig_caf_general"].values,
            "stage": adata.obs[stage_col].values,
        })
        df = df[df["stage"].isin(stages)]
        sns.violinplot(data=df, x="stage", y="score", order=stages, palette=palette, ax=ax_e)
        ax_e.set_title("E) CAF Score by Stage")
        ax_e.set_xlabel("")
        ax_e.set_ylabel("Score")

    # Panel F: Key findings text
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")

    # Generate key findings
    findings = ["Key Biological Findings:", ""]

    # Top positive correlation
    pos = niche_biology_df[niche_biology_df["spearman_rho"] > 0].head(1)
    if len(pos) > 0:
        row = pos.iloc[0]
        findings.append(f"+ {row['pathway']}")
        findings.append(f"  (rho={row['spearman_rho']:.2f})")
        findings.append("")

    # Top negative correlation
    neg = niche_biology_df[niche_biology_df["spearman_rho"] < 0].head(1)
    if len(neg) > 0:
        row = neg.iloc[0]
        findings.append(f"- {row['pathway']}")
        findings.append(f"  (rho={row['spearman_rho']:.2f})")

    ax_f.text(0.1, 0.9, "\n".join(findings), transform=ax_f.transAxes,
              fontsize=10, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("StageBridge Biological Interpretation Summary", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info(f"Saved biological summary panel: {save_path}")
    plt.close()
