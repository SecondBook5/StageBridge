"""
Biology Paper Figure Generation for StageBridge

Publication-quality figures for the biology/clinical paper:
1. Stage progression risk distributions
2. Niche ecosystem comparison across stages
3. Proinflammatory niche spatial visualization
4. KAC state vs niche risk scatter
5. Fold change heatmaps
6. Perturbation effect plots
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns

# Publication style settings
STAGE_COLORS = {
    "Normal": "#00BA38",
    "AAH": "#F8766D",
    "AIS": "#619CFF",
    "MIA": "#E58700",
    "LUAD": "#A3A500",
    "Unknown": "#999999",
}

STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]

NICHE_COLORS = {
    "Normal-like": "#66c2a5",
    "Immune-infiltrated": "#fc8d62",
    "CAF-enriched": "#8da0cb",
    "Proinflammatory": "#e78ac3",
    "Proinflammatory-CAF": "#a6d854",
}


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_progression_risk_by_stage(
    cell_risks: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot progression risk score distributions by stage.

    Shows violin + box + swarm plot for each stage.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Filter to canonical stages
    df = cell_risks[cell_risks["stage"].isin(STAGE_ORDER)].copy()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)

    # Violin plot
    parts = ax.violinplot(
        [df[df["stage"] == s]["progression_risk_score"].values for s in STAGE_ORDER if s in df["stage"].values],
        positions=range(len([s for s in STAGE_ORDER if s in df["stage"].values])),
        showmeans=False,
        showmedians=True,
        widths=0.7,
    )

    # Color violins by stage
    present_stages = [s for s in STAGE_ORDER if s in df["stage"].values]
    for i, (pc, stage) in enumerate(zip(parts['bodies'], present_stages)):
        pc.set_facecolor(STAGE_COLORS[stage])
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Add box plots inside
    bp = ax.boxplot(
        [df[df["stage"] == s]["progression_risk_score"].values for s in present_stages],
        positions=range(len(present_stages)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
    )

    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_alpha(0.9)

    # Labels
    ax.set_xticks(range(len(present_stages)))
    ax.set_xticklabels(present_stages)
    ax.set_xlabel("Disease Stage")
    ax.set_ylabel("Progression Risk Score")
    ax.set_title("Cell-Level Progression Risk by Stage")
    ax.set_ylim(0, 1)

    # Add sample sizes
    for i, stage in enumerate(present_stages):
        n = len(df[df["stage"] == stage])
        ax.text(i, -0.08, f"n={n:,}", ha='center', va='top', fontsize=8, transform=ax.get_xaxis_transform())

    # Statistical annotation (if significant)
    # Add brackets for significant comparisons
    ax.text(0.02, 0.98, "Higher = More Progression-Prone",
            transform=ax.transAxes, fontsize=8, va='top', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_niche_ecosystem_comparison(
    niche_risks: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot niche ecosystem comparison across stages.

    Stacked bar showing niche category proportions per stage.
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Filter and prepare data
    df = niche_risks[niche_risks["stage"].isin(STAGE_ORDER)].copy()

    # Panel A: Stacked bar of niche categories
    ax = axes[0]
    if "niche_category" in df.columns:
        pivot = df.groupby(["stage", "niche_category"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(STAGE_ORDER).dropna(how='all')
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        categories = list(NICHE_COLORS.keys())
        categories = [c for c in categories if c in pivot_pct.columns]

        bottom = np.zeros(len(pivot_pct))
        for cat in categories:
            if cat in pivot_pct.columns:
                values = pivot_pct[cat].values
                ax.bar(range(len(pivot_pct)), values, bottom=bottom,
                       label=cat, color=NICHE_COLORS.get(cat, '#999999'), width=0.7)
                bottom += values

        ax.set_xticks(range(len(pivot_pct)))
        ax.set_xticklabels(pivot_pct.index)
        ax.set_ylabel("Percentage of Cells")
        ax.set_xlabel("Disease Stage")
        ax.set_title("Niche Category Distribution")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, "No niche category data", ha='center', va='center')

    # Panel B: Mean risk scores by stage
    ax = axes[1]
    metrics = ["niche_risk_score", "proinflammatory_score", "caf_enrichment"]
    metric_labels = ["Niche Risk", "Proinflammatory", "CAF Enrichment"]

    x = np.arange(len(STAGE_ORDER))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric in df.columns:
            means = [df[df["stage"] == s][metric].mean() if s in df["stage"].values else 0 for s in STAGE_ORDER]
            stds = [df[df["stage"] == s][metric].std() if s in df["stage"].values else 0 for s in STAGE_ORDER]
            ax.bar(x + i * width, means, width, label=label, yerr=stds, capsize=2, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel("Score")
    ax.set_xlabel("Disease Stage")
    ax.set_title("Niche Risk Metrics by Stage")
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_kac_vs_niche_risk(
    cell_risks: pd.DataFrame,
    niche_risks: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (8, 6),
) -> None:
    """
    Scatter plot of KAC state vs niche risk, colored by stage.

    Tests hypothesis: KAC-like cells in high-risk niches are most progression-prone.
    """
    setup_publication_style()

    # Merge data
    df = cell_risks.merge(niche_risks[["cell_id", "niche_risk_score"]], on="cell_id")
    df = df[df["stage"].isin(STAGE_ORDER)]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each stage
    for stage in STAGE_ORDER:
        stage_data = df[df["stage"] == stage]
        if len(stage_data) > 0:
            # Subsample for visibility if too many points
            if len(stage_data) > 5000:
                stage_data = stage_data.sample(5000, random_state=42)

            ax.scatter(
                stage_data["kac_state_score"],
                stage_data["niche_risk_score"],
                c=STAGE_COLORS[stage],
                label=f"{stage} (n={len(df[df['stage']==stage]):,})",
                alpha=0.3,
                s=10,
                edgecolors='none',
            )

    ax.set_xlabel("KAC State Score (Alveolar Progenitor)")
    ax.set_ylabel("Niche Risk Score (Proinflammatory)")
    ax.set_title("KAC State vs Niche Risk by Stage")
    ax.legend(loc='upper right', fontsize=8, markerscale=2)

    # Add quadrant lines
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Quadrant labels
    ax.text(0.75, 0.85, "High Risk\nQuadrant", ha='center', va='center',
            transform=ax.transAxes, fontsize=9, style='italic', alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_fold_change_heatmap(
    stage_summaries: Dict[str, Any],
    output_path: Path,
    figsize: tuple = (8, 5),
) -> None:
    """
    Heatmap of fold changes vs Normal for each stage.
    """
    setup_publication_style()

    # Extract fold change data
    metrics = ["progression_risk_fc", "niche_risk_fc", "proinflammatory_fc", "caf_fc"]
    metric_labels = ["Progression\nRisk", "Niche\nRisk", "Proinflammatory", "CAF\nEnrichment"]

    stages = [s for s in STAGE_ORDER[1:] if s in stage_summaries]  # Skip Normal

    data = np.zeros((len(stages), len(metrics)))
    for i, stage in enumerate(stages):
        if hasattr(stage_summaries[stage], 'comparison_to_normal'):
            comp = stage_summaries[stage].comparison_to_normal
        else:
            comp = stage_summaries[stage].get("comparison_to_normal", {})

        for j, metric in enumerate(metrics):
            data[i, j] = comp.get(metric, 1.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Log2 transform for better visualization
    data_log2 = np.log2(data + 0.01)

    im = ax.imshow(data_log2, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Disease Stage")
    ax.set_title("Fold Change vs Normal (log2)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log2(Fold Change)")

    # Add text annotations
    for i in range(len(stages)):
        for j in range(len(metrics)):
            fc = data[i, j]
            text_color = 'white' if abs(data_log2[i, j]) > 1 else 'black'
            ax.text(j, i, f"{fc:.1f}x", ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_proinflammatory_enrichment_trajectory(
    niche_risks: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (7, 5),
) -> None:
    """
    Line plot showing proinflammatory niche enrichment across disease progression.

    Key visualization for Peng/Kadara hypothesis: proinflammatory niches should
    be more common in precursors (AAH, AIS) than in LUAD.
    """
    setup_publication_style()

    df = niche_risks[niche_risks["stage"].isin(STAGE_ORDER)].copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate metrics per stage
    stages = []
    proinflam_fracs = []
    proinflam_errs = []
    caf_fracs = []
    combined_fracs = []

    for stage in STAGE_ORDER:
        stage_data = df[df["stage"] == stage]
        if len(stage_data) > 0:
            stages.append(stage)

            # Proinflammatory fraction
            if "is_proinflammatory_niche" in stage_data.columns:
                pf = stage_data["is_proinflammatory_niche"].mean()
            else:
                pf = (stage_data["proinflammatory_score"] >= 0.3).mean()
            proinflam_fracs.append(pf)

            # Bootstrap CI
            n_boot = 100
            boot_fracs = []
            for _ in range(n_boot):
                sample = stage_data.sample(len(stage_data), replace=True)
                if "is_proinflammatory_niche" in sample.columns:
                    boot_fracs.append(sample["is_proinflammatory_niche"].mean())
                else:
                    boot_fracs.append((sample["proinflammatory_score"] >= 0.3).mean())
            proinflam_errs.append(np.std(boot_fracs))

            # CAF fraction
            cf = (stage_data["caf_enrichment"] >= 0.2).mean()
            caf_fracs.append(cf)

            # Combined (proinflam + CAF)
            if "niche_category" in stage_data.columns:
                combined = (stage_data["niche_category"] == "Proinflammatory-CAF").mean()
            else:
                combined = ((stage_data["proinflammatory_score"] >= 0.3) &
                           (stage_data["caf_enrichment"] >= 0.2)).mean()
            combined_fracs.append(combined)

    x = np.arange(len(stages))

    # Plot lines
    ax.errorbar(x, proinflam_fracs, yerr=proinflam_errs, marker='o', markersize=8,
                label='IL1B-high Mac Niche', color='#e78ac3', linewidth=2, capsize=3)
    ax.plot(x, caf_fracs, marker='s', markersize=8,
            label='CAF-enriched Niche', color='#8da0cb', linewidth=2)
    ax.plot(x, combined_fracs, marker='^', markersize=8,
            label='Proinflammatory + CAF', color='#a6d854', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlabel("Disease Stage")
    ax.set_ylabel("Fraction of Cells")
    ax.set_title("Niche Enrichment Across LUAD Progression")
    ax.legend(loc='upper right', fontsize=9)

    # Highlight precursor window
    ax.axvspan(-0.5, 3.5, alpha=0.1, color='orange', label='Precursor Window')
    ax.text(1.5, ax.get_ylim()[1] * 0.95, "Interception\nWindow",
            ha='center', va='top', fontsize=9, style='italic', alpha=0.7)

    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_perturbation_effects(
    perturbation_results: List[Any],
    output_path: Path,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot effects of removing a cell type from neighborhoods.
    """
    setup_publication_style()

    if not perturbation_results:
        print("No perturbation results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    deltas = [r.prediction_delta for r in perturbation_results]
    risk_deltas = [r.progression_risk_delta for r in perturbation_results]
    cell_type = perturbation_results[0].removed_cell_type

    # Panel A: Distribution of prediction changes
    ax = axes[0]
    ax.hist(deltas, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(np.mean(deltas), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(deltas):.3f}')
    ax.axvline(0.1, color='gray', linestyle=':', linewidth=1,
               label='Significance threshold')
    ax.set_xlabel("Prediction Change (L2 norm)")
    ax.set_ylabel("Number of Cells")
    ax.set_title(f"Effect of Removing {cell_type}")
    ax.legend(fontsize=8)

    # Panel B: Progression risk change
    ax = axes[1]
    ax.hist(risk_deltas, bins=50, alpha=0.7, color='coral', edgecolor='white')
    ax.axvline(np.mean(risk_deltas), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(risk_deltas):.3f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel("Progression Risk Change")
    ax.set_ylabel("Number of Cells")
    ax.set_title(f"Risk Change After {cell_type} Ablation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_biology_figures(
    cell_risks: pd.DataFrame,
    niche_risks: pd.DataFrame,
    stage_summaries: Dict[str, Any],
    output_dir: Path,
    perturbation_results: Optional[List[Any]] = None,
) -> None:
    """
    Generate all figures for biology paper.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating biology paper figures...")

    # Figure 1: Progression risk by stage
    plot_progression_risk_by_stage(
        cell_risks,
        output_dir / "fig1_progression_risk_by_stage.png"
    )

    # Figure 2: Niche ecosystem comparison
    plot_niche_ecosystem_comparison(
        niche_risks,
        output_dir / "fig2_niche_ecosystem_comparison.png"
    )

    # Figure 3: KAC vs niche risk scatter
    plot_kac_vs_niche_risk(
        cell_risks, niche_risks,
        output_dir / "fig3_kac_vs_niche_risk.png"
    )

    # Figure 4: Fold change heatmap
    plot_fold_change_heatmap(
        stage_summaries,
        output_dir / "fig4_fold_change_heatmap.png"
    )

    # Figure 5: Proinflammatory trajectory
    plot_proinflammatory_enrichment_trajectory(
        niche_risks,
        output_dir / "fig5_proinflammatory_trajectory.png"
    )

    # Figure 6: Perturbation effects (if available)
    if perturbation_results:
        plot_perturbation_effects(
            perturbation_results,
            output_dir / "fig6_perturbation_effects.png"
        )

    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    print("Biology paper figures module loaded.")
    print("\nUsage:")
    print("  from stagebridge.analysis.biology_paper_figures import generate_all_biology_figures")
    print("  generate_all_biology_figures(cell_risks, niche_risks, stage_summaries, output_dir)")
