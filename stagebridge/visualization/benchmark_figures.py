"""
Publication-grade figures for spatial backend benchmarking.

Generates Nature Methods quality figures for:
- Backend performance comparison
- Spatial cell type visualization
- Metric distributions and variability
- Statistical comparisons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

from .style import (
    NatureStyle,
    apply_nature_style,
    get_color_palette,
    save_figure,
    add_panel_label,
    create_figure,
    add_significance_bar,
    format_pvalue,
    SINGLE_COL_WIDTH,
    DOUBLE_COL_WIDTH,
)

log = logging.getLogger(__name__)


@dataclass
class SpatialBenchmarkFigures:
    """
    Generator for spatial benchmark publication figures.

    Attributes:
        metrics_df: DataFrame with per-sample metrics
        output_dir: Directory for saving figures
        style: Nature style configuration
    """

    metrics_df: pd.DataFrame
    output_dir: Path
    style: NatureStyle = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.style is None:
            self.style = NatureStyle()
        apply_nature_style(self.style)

        self.colors = get_color_palette("colorblind")
        self.backends = sorted(self.metrics_df["backend"].unique())
        self.backend_colors = {b: self.colors[i % len(self.colors)]
                               for i, b in enumerate(self.backends)}

    def generate_all(self):
        """Generate all publication figures."""
        log.info("Generating publication figures...")

        self.figure_1_overview()
        self.figure_2_metric_comparison()
        self.figure_3_distributions()
        self.figure_4_spatial_patterns()
        self.figure_5_statistical_comparison()
        self.figure_supplementary_all_metrics()

        log.info("All figures saved to %s", self.output_dir)

    # =========================================================================
    # FIGURE 1: Overview bar chart with key metrics
    # =========================================================================
    def figure_1_overview(self):
        """
        Main comparison figure showing key metrics per backend.

        Panel layout:
        (a) Types per spot (bar + error)
        (b) Effective coverage (bar + error)
        (c) Entropy quality (bar + error)
        """
        fig, axes = create_figure(n_panels=3, n_cols=3, width="double", aspect_ratio=1.0)

        metrics = [
            ("types_per_spot_mean", "Cell types per spot", True),
            ("effective_coverage", "Effective coverage", True),
            ("mean_entropy", "Mean entropy", None),
        ]

        for ax, (metric, label, higher_better) in zip(axes, metrics):
            self._plot_metric_bars(ax, metric, label, higher_better)

        # Add panel labels
        for i, ax in enumerate(axes):
            add_panel_label(ax, chr(ord('a') + i))

        fig.suptitle("Spatial Deconvolution Backend Comparison", fontsize=self.style.title_size + 2)

        save_figure(fig, self.output_dir / "figure1_overview")
        plt.close(fig)
        log.info("Generated Figure 1: Overview")

    def _plot_metric_bars(
        self,
        ax: plt.Axes,
        metric: str,
        label: str,
        higher_better: bool | None,
    ):
        """Plot bar chart with error bars for a metric."""
        if metric not in self.metrics_df.columns:
            ax.text(0.5, 0.5, f"Metric '{metric}'\nnot available",
                    ha="center", va="center", transform=ax.transAxes)
            return

        # Compute statistics
        grouped = self.metrics_df.groupby("backend")[metric]
        means = grouped.mean()
        sems = grouped.sem()

        # Sort by mean
        order = means.sort_values(ascending=False if higher_better else True).index

        x = np.arange(len(order))
        colors = [self.backend_colors[b] for b in order]

        bars = ax.bar(x, [means[b] for b in order], yerr=[sems[b] for b in order],
                      capsize=3, color=colors, edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 0.75})

        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in order], rotation=45, ha="right")
        ax.set_ylabel(label)

        # Add value labels on bars
        for bar, b in zip(bars, order):
            height = bar.get_height()
            ax.annotate(f"{means[b]:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=self.style.tick_size)

    # =========================================================================
    # FIGURE 2: Multi-metric comparison (radar + heatmap)
    # =========================================================================
    def figure_2_metric_comparison(self):
        """
        Comprehensive metric comparison.

        Panel layout:
        (a) Radar chart of normalized metrics
        (b) Heatmap of all metrics
        """
        # Use manual layout to avoid tight_layout/constrained_layout conflicts
        fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5),
                         constrained_layout=False)

        # Radar chart
        ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8], projection="polar")
        self._plot_radar(ax1)
        add_panel_label(ax1, "a", x=-0.1, y=1.1)

        # Heatmap
        ax2 = fig.add_axes([0.55, 0.15, 0.4, 0.7])
        self._plot_metric_heatmap(ax2)
        add_panel_label(ax2, "b", x=-0.15, y=1.05)

        save_figure(fig, self.output_dir / "figure2_comparison")
        plt.close(fig)
        log.info("Generated Figure 2: Metric comparison")

    def _plot_radar(self, ax: plt.Axes):
        """Plot radar chart of key metrics."""
        metrics = [
            "types_per_spot_mean",
            "effective_coverage",
            "global_type_coverage",
            "mean_entropy",
            "gini_coefficient_mean",
        ]

        # Filter to available metrics
        available = [m for m in metrics if m in self.metrics_df.columns]
        if len(available) < 3:
            ax.text(0.5, 0.5, "Insufficient metrics", ha="center", va="center")
            return

        # Compute means and normalize to [0, 1]
        data = {}
        for backend in self.backends:
            mask = self.metrics_df["backend"] == backend
            values = []
            for m in available:
                v = self.metrics_df.loc[mask, m].mean()
                values.append(v)
            data[backend] = values

        # Normalize each metric
        df = pd.DataFrame(data, index=available)
        df_norm = (df - df.min(axis=1).values[:, np.newaxis]) / \
                  (df.max(axis=1).values[:, np.newaxis] - df.min(axis=1).values[:, np.newaxis] + 1e-10)

        # For Gini, lower is better, so invert
        if "gini_coefficient_mean" in df_norm.index:
            df_norm.loc["gini_coefficient_mean"] = 1 - df_norm.loc["gini_coefficient_mean"]

        # Plot
        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]

        for backend in self.backends:
            values = df_norm[backend].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=1.5, label=backend.upper(),
                    color=self.backend_colors[backend], markersize=4)
            ax.fill(angles, values, alpha=0.15, color=self.backend_colors[backend])

        ax.set_xticks(angles[:-1])
        labels = [m.replace("_", "\n").replace("mean", "").strip() for m in available]
        ax.set_xticklabels(labels, fontsize=self.style.tick_size)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=self.style.legend_size)
        ax.set_title("Normalized Performance", pad=20)

    def _plot_metric_heatmap(self, ax: plt.Axes):
        """Plot heatmap of all metrics by backend."""
        # Select key metrics
        metrics = [
            "types_per_spot_mean",
            "effective_coverage",
            "global_type_coverage",
            "mean_entropy",
            "gini_coefficient_mean",
            "dominance_ratio_mean",
            "type_presence_rate_mean",
        ]

        available = [m for m in metrics if m in self.metrics_df.columns]

        # Compute means
        data = self.metrics_df.groupby("backend")[available].mean()

        # Z-score normalize for display
        data_z = (data - data.mean()) / (data.std() + 1e-10)

        # Plot
        sns.heatmap(
            data_z.T,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            annot=data.T.round(2),
            fmt=".2f",
            annot_kws={"fontsize": self.style.tick_size - 1},
            cbar_kws={"label": "Z-score", "shrink": 0.8},
            linewidths=0.5,
        )

        ax.set_xticklabels([b.upper() for b in data.index], rotation=45, ha="right")
        labels = [m.replace("_", " ").replace("mean", "").title().strip() for m in available]
        ax.set_yticklabels(labels, rotation=0)
        ax.set_title("Metric Comparison (Z-scored)")

    # =========================================================================
    # FIGURE 3: Distribution plots
    # =========================================================================
    def figure_3_distributions(self):
        """
        Distribution comparison with violin plots.

        Panel layout:
        (a) Types per spot distribution
        (b) Entropy distribution
        (c) Max proportion distribution
        """
        fig, axes = create_figure(n_panels=3, n_cols=3, width="double", aspect_ratio=0.9)

        metrics = [
            ("types_per_spot_mean", "Cell types per spot"),
            ("mean_entropy", "Entropy"),
            ("max_proportion_mean", "Max proportion"),
        ]

        for ax, (metric, label) in zip(axes, metrics):
            self._plot_violin(ax, metric, label)

        for i, ax in enumerate(axes):
            add_panel_label(ax, chr(ord('a') + i))

        save_figure(fig, self.output_dir / "figure3_distributions")
        plt.close(fig)
        log.info("Generated Figure 3: Distributions")

    def _plot_violin(self, ax: plt.Axes, metric: str, label: str):
        """Plot violin plot for a metric."""
        if metric not in self.metrics_df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            return

        # Order by median
        medians = self.metrics_df.groupby("backend")[metric].median().sort_values(ascending=False)
        order = medians.index.tolist()

        palette = [self.backend_colors[b] for b in order]

        sns.violinplot(
            data=self.metrics_df,
            x="backend",
            y=metric,
            hue="backend",
            order=order,
            hue_order=order,
            palette=dict(zip(order, palette)),
            ax=ax,
            inner="box",
            linewidth=0.75,
            cut=0,
            legend=False,
        )

        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([b.upper() for b in order], rotation=45, ha="right")

        # Add sample size annotations
        for i, backend in enumerate(order):
            n = (self.metrics_df["backend"] == backend).sum()
            ax.annotate(f"n={n}", xy=(i, ax.get_ylim()[0]),
                        xytext=(0, -15), textcoords="offset points",
                        ha="center", fontsize=self.style.tick_size - 1)

    # =========================================================================
    # FIGURE 4: Spatial pattern visualization (placeholder for real data)
    # =========================================================================
    def figure_4_spatial_patterns(self):
        """
        Spatial composition patterns.

        This requires actual spatial coordinates and proportions.
        Creates a placeholder layout for when data is available.
        """
        fig, axes = create_figure(n_panels=4, n_cols=2, width="double", aspect_ratio=1.0)

        for i, (ax, backend) in enumerate(zip(axes, self.backends[:4])):
            ax.text(0.5, 0.5, f"{backend.upper()}\n(spatial plot)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.style.font_size)
            ax.set_title(backend.upper())
            ax.axis("off")
            add_panel_label(ax, chr(ord('a') + i))

        fig.suptitle("Spatial Cell Type Composition", fontsize=self.style.title_size + 2)

        save_figure(fig, self.output_dir / "figure4_spatial_placeholder")
        plt.close(fig)
        log.info("Generated Figure 4: Spatial patterns (placeholder)")

    # =========================================================================
    # FIGURE 5: Statistical comparison
    # =========================================================================
    def figure_5_statistical_comparison(self):
        """
        Statistical comparison between backends.

        Panel layout:
        (a) Pairwise comparison matrix (effect sizes)
        (b) Ranking summary
        """
        fig, axes = create_figure(n_panels=2, n_cols=2, width="double", aspect_ratio=0.9)

        self._plot_pairwise_comparison(axes[0])
        self._plot_ranking_summary(axes[1])

        for i, ax in enumerate(axes):
            add_panel_label(ax, chr(ord('a') + i))

        save_figure(fig, self.output_dir / "figure5_statistics")
        plt.close(fig)
        log.info("Generated Figure 5: Statistical comparison")

    def _plot_pairwise_comparison(self, ax: plt.Axes):
        """Plot pairwise statistical comparison matrix."""
        metric = "types_per_spot_mean"
        if metric not in self.metrics_df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            return

        n = len(self.backends)
        effect_sizes = np.zeros((n, n))
        pvalues = np.ones((n, n))

        for i, b1 in enumerate(self.backends):
            for j, b2 in enumerate(self.backends):
                if i == j:
                    continue

                g1 = self.metrics_df.loc[self.metrics_df["backend"] == b1, metric].dropna()
                g2 = self.metrics_df.loc[self.metrics_df["backend"] == b2, metric].dropna()

                if len(g1) < 2 or len(g2) < 2:
                    continue

                # Mann-Whitney U test
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                pvalues[i, j] = p

                # Cohen's d effect size
                pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
                if pooled_std > 0:
                    effect_sizes[i, j] = (g1.mean() - g2.mean()) / pooled_std

        # Plot heatmap
        mask = np.eye(n, dtype=bool)

        sns.heatmap(
            effect_sizes,
            ax=ax,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            vmin=-2,
            vmax=2,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": self.style.tick_size},
            cbar_kws={"label": "Cohen's d", "shrink": 0.8},
            linewidths=0.5,
            xticklabels=[b.upper() for b in self.backends],
            yticklabels=[b.upper() for b in self.backends],
        )

        # Add significance markers
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if pvalues[i, j] < 0.001:
                    sig = "***"
                elif pvalues[i, j] < 0.01:
                    sig = "**"
                elif pvalues[i, j] < 0.05:
                    sig = "*"
                else:
                    continue
                ax.text(j + 0.5, i + 0.75, sig, ha="center", va="center",
                        fontsize=self.style.tick_size - 1, fontweight="bold")

        ax.set_title("Pairwise Effect Sizes\n(Types per Spot)")
        ax.tick_params(axis="x", rotation=45)

    def _plot_ranking_summary(self, ax: plt.Axes):
        """Plot overall ranking summary."""
        # Compute composite scores
        ranking_metrics = [
            "types_per_spot_mean",
            "effective_coverage",
            "global_type_coverage",
        ]

        available = [m for m in ranking_metrics if m in self.metrics_df.columns]
        if not available:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            return

        # Normalize and average
        scores = {}
        for backend in self.backends:
            mask = self.metrics_df["backend"] == backend
            backend_scores = []
            for m in available:
                v = self.metrics_df.loc[mask, m].mean()
                # Normalize by global min/max
                vmin = self.metrics_df[m].min()
                vmax = self.metrics_df[m].max()
                if vmax > vmin:
                    v_norm = (v - vmin) / (vmax - vmin)
                else:
                    v_norm = 0.5
                backend_scores.append(v_norm)
            scores[backend] = np.mean(backend_scores)

        # Sort and plot
        sorted_backends = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        y = np.arange(len(sorted_backends))
        colors = [self.backend_colors[b] for b in sorted_backends]
        values = [scores[b] for b in sorted_backends]

        bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels([b.upper() for b in sorted_backends])
        ax.set_xlabel("Composite Score")
        ax.set_xlim(0, 1)
        ax.set_title("Overall Ranking")

        # Add value labels
        for bar, v in zip(bars, values):
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=self.style.tick_size)

        # Highlight winner
        ax.axhline(y=0, color="gold", linewidth=2, alpha=0.3, zorder=0)

    # =========================================================================
    # SUPPLEMENTARY: All metrics table
    # =========================================================================
    def figure_supplementary_all_metrics(self):
        """Generate supplementary table of all metrics."""
        # Get all numeric columns
        metric_cols = [c for c in self.metrics_df.columns
                       if c not in ["backend", "sample_id", "label_source"]
                       and self.metrics_df[c].dtype in ["float64", "int64"]]

        # Compute summary statistics
        summary = self.metrics_df.groupby("backend")[metric_cols].agg(["mean", "std", "median"])
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]

        # Save as CSV
        summary.round(4).to_csv(self.output_dir / "supplementary_all_metrics.csv")

        # Also create a figure version
        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, len(metric_cols) * 0.3 + 1))
        ax.axis("off")

        # Just save the key metrics as a formatted table image
        key_metrics = metric_cols[:10]  # First 10 metrics
        table_data = self.metrics_df.groupby("backend")[key_metrics].mean().round(3)

        table = ax.table(
            cellText=table_data.values,
            rowLabels=[b.upper() for b in table_data.index],
            colLabels=[c.replace("_", "\n") for c in table_data.columns],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.style.tick_size)
        table.scale(1.2, 1.5)

        ax.set_title("Summary Metrics by Backend", fontsize=self.style.title_size, pad=20)

        save_figure(fig, self.output_dir / "supplementary_table")
        plt.close(fig)
        log.info("Generated supplementary materials")


def generate_all_benchmark_figures(
    metrics_csv: str | Path,
    output_dir: str | Path,
    style: NatureStyle | None = None,
) -> SpatialBenchmarkFigures:
    """
    Generate all benchmark figures from metrics CSV.

    Args:
        metrics_csv: Path to all_sample_metrics.csv from aggregation
        output_dir: Directory for figures
        style: Optional style configuration

    Returns:
        SpatialBenchmarkFigures instance
    """
    metrics_df = pd.read_csv(metrics_csv)
    output_dir = Path(output_dir)

    generator = SpatialBenchmarkFigures(
        metrics_df=metrics_df,
        output_dir=output_dir,
        style=style,
    )
    generator.generate_all()

    return generator


def main():
    """CLI for generating publication figures."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate publication-quality spatial benchmark figures"
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        required=True,
        help="Path to all_sample_metrics.csv from aggregation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output figures",
    )
    parser.add_argument(
        "--width",
        type=str,
        default="double",
        choices=["single", "double"],
        help="Figure width (single=88mm, double=180mm)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    style = NatureStyle(
        width=SINGLE_COL_WIDTH if args.width == "single" else DOUBLE_COL_WIDTH
    )

    generate_all_benchmark_figures(
        metrics_csv=args.metrics_csv,
        output_dir=args.output_dir,
        style=style,
    )

    print(f"\nPublication figures saved to {args.output_dir}")
    print("Formats: PNG (300 DPI), PDF (vector), SVG (vector)")


def generate_spatial_figure(
    proportions_by_backend: dict[str, pd.DataFrame],
    spatial_coords: np.ndarray,
    output_dir: str | Path,
    cell_types: list[str] | None = None,
    n_types: int = 6,
    sample_id: str = "",
):
    """
    Generate spatial composition comparison figure.

    Args:
        proportions_by_backend: Dict mapping backend name to proportions DataFrame
        spatial_coords: (n_spots, 2) array of coordinates
        output_dir: Output directory
        cell_types: Specific cell types to plot (default: top by abundance)
        n_types: Number of cell types if cell_types not specified
        sample_id: Sample identifier for title
    """
    apply_nature_style()
    output_dir = Path(output_dir)

    backends = list(proportions_by_backend.keys())
    n_backends = len(backends)

    if cell_types is None:
        # Find most abundant types across all backends
        all_means = pd.concat([p.mean() for p in proportions_by_backend.values()], axis=1)
        cell_types = all_means.mean(axis=1).nlargest(n_types).index.tolist()

    for cell_type in cell_types:
        fig, axes = plt.subplots(1, n_backends, figsize=(3 * n_backends, 3))
        if n_backends == 1:
            axes = [axes]

        vmin, vmax = 0, 0
        for name, props in proportions_by_backend.items():
            if cell_type in props.columns:
                vmax = max(vmax, props[cell_type].max())

        for ax, backend in zip(axes, backends):
            props = proportions_by_backend[backend]
            if cell_type not in props.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(backend.upper())
                ax.axis("off")
                continue

            values = props[cell_type].values
            scatter = ax.scatter(
                spatial_coords[:, 0],
                spatial_coords[:, 1],
                c=values,
                cmap="viridis",
                s=1,
                vmin=0,
                vmax=vmax,
                rasterized=True,  # Important for PDF size
            )
            ax.set_title(backend.upper())
            ax.set_aspect("equal")
            ax.axis("off")

        # Shared colorbar
        fig.colorbar(scatter, ax=axes, shrink=0.6, label="Proportion")
        fig.suptitle(f"{cell_type}" + (f" ({sample_id})" if sample_id else ""))

        safe_name = cell_type.replace("/", "_").replace(" ", "_")
        save_figure(fig, output_dir / f"spatial_{safe_name}")
        plt.close(fig)

    log.info("Generated spatial figures for %d cell types", len(cell_types))


if __name__ == "__main__":
    main()
