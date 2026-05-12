"""Visualization for StageBridge analysis outputs.

Generates publication-quality figures for:
- Flow field metrics (drift, divergence, curl, irreversibility)
- Pathway activity scores
- Proliferation predictions
- Attention patterns

Supports both UMAP/embedding space and spatial tissue coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

from stagebridge.analysis.flow_fields import FlowFieldAnalyzer, FlowFieldMetrics


# Publication-quality colormaps
DIVERGING_CMAP = "RdBu_r"  # Red=positive, Blue=negative
SEQUENTIAL_CMAP = "viridis"
STAGE_COLORS = {
    "Normal": "#2ecc71",
    "AAH": "#f1c40f",
    "AIS": "#e67e22",
    "MIA": "#e74c3c",
    "ADC": "#9b59b6",
    "LUAD": "#9b59b6",
    # Numeric stages
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#9b59b6",
}

PATHWAY_NAMES = [
    "EGFR", "Hypoxia", "JAK-STAT", "MAPK", "NFkB",
    "PI3K", "TGFb", "TNFa", "Trail", "VEGF",
    "WNT", "p53", "Androgen", "Estrogen", "cGAS-STING"
]


@dataclass
class PlotConfig:
    """Configuration for plot styling."""

    figsize: tuple[float, float] = (8, 6)
    dpi: int = 150
    point_size: float = 5.0
    alpha: float = 0.7
    fontsize: int = 12
    title_fontsize: int = 14
    colorbar: bool = True
    save_format: str = "png"


class StageBridgeVisualizer:
    """Generate analysis plots from inference outputs.

    Args:
        output_dir: Directory containing inference outputs
        plot_config: Optional plot configuration
    """

    def __init__(
        self,
        output_dir: Path | str,
        plot_config: PlotConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = plot_config or PlotConfig()

        # Load available outputs
        self._load_outputs()

    def _load_outputs(self) -> None:
        """Load inference outputs."""
        # Required
        pred_path = self.output_dir / "predictions.parquet"
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions.parquet not found in {self.output_dir}")

        self.predictions = pd.read_parquet(pred_path)

        # Optional outputs
        emb_path = self.output_dir / "embeddings.parquet"
        if emb_path.exists():
            emb_df = pd.read_parquet(emb_path)
            self.embeddings = emb_df.values
        else:
            self.embeddings = None

        pathway_path = self.output_dir / "pathway_scores.parquet"
        if pathway_path.exists():
            self.pathway_scores = pd.read_parquet(pathway_path)
        else:
            self.pathway_scores = None

        prolif_path = self.output_dir / "proliferation_scores.parquet"
        if prolif_path.exists():
            self.proliferation_scores = pd.read_parquet(prolif_path)
        else:
            self.proliferation_scores = None

        disp_path = self.output_dir / "displacements.npy"
        if disp_path.exists():
            self.displacements = np.load(disp_path)
        else:
            self.displacements = None

        attn_path = self.output_dir / "attention_weights.npz"
        if attn_path.exists():
            attn_data = np.load(attn_path)
            self.attention_weights = attn_data.get("attention", None)
        else:
            self.attention_weights = None

    def compute_flow_metrics(self) -> FlowFieldMetrics | None:
        """Compute flow field metrics from embeddings and displacements."""
        if self.embeddings is None or self.displacements is None:
            print("Need embeddings and displacements to compute flow metrics")
            return None

        analyzer = FlowFieldAnalyzer()
        return analyzer.compute_metrics(self.embeddings, self.displacements)

    def plot_umap_colored(
        self,
        values: np.ndarray,
        title: str,
        cmap: str = SEQUENTIAL_CMAP,
        vmin: float | None = None,
        vmax: float | None = None,
        umap_coords: np.ndarray | None = None,
        save_path: Path | str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Plot UMAP colored by a scalar value.

        Args:
            values: [N] values to color by
            title: Plot title
            cmap: Colormap name
            vmin, vmax: Color limits
            umap_coords: [N, 2] UMAP coordinates (if None, uses first 2 embedding dims)
            save_path: Path to save figure
            ax: Existing axes to plot on

        Returns:
            Figure if ax was None
        """
        if umap_coords is None:
            if self.embeddings is None:
                raise ValueError("No embeddings or UMAP coordinates provided")
            umap_coords = self.embeddings[:, :2]

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        scatter = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=values,
            cmap=cmap,
            s=self.config.point_size,
            alpha=self.config.alpha,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        if self.config.colorbar:
            plt.colorbar(scatter, ax=ax, label=title)

        ax.set_xlabel("UMAP 1", fontsize=self.config.fontsize)
        ax.set_ylabel("UMAP 2", fontsize=self.config.fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_aspect("equal")

        if save_path and created_fig:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig if created_fig else None

    def plot_spatial_colored(
        self,
        values: np.ndarray,
        spatial_coords: np.ndarray,
        title: str,
        cmap: str = SEQUENTIAL_CMAP,
        vmin: float | None = None,
        vmax: float | None = None,
        save_path: Path | str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Plot spatial coordinates colored by a scalar value.

        Args:
            values: [N] values to color by
            spatial_coords: [N, 2] spatial coordinates
            title: Plot title
            cmap: Colormap name
            vmin, vmax: Color limits
            save_path: Path to save figure
            ax: Existing axes to plot on

        Returns:
            Figure if ax was None
        """
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        scatter = ax.scatter(
            spatial_coords[:, 0],
            spatial_coords[:, 1],
            c=values,
            cmap=cmap,
            s=self.config.point_size,
            alpha=self.config.alpha,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        if self.config.colorbar:
            plt.colorbar(scatter, ax=ax, label=title)

        ax.set_xlabel("X", fontsize=self.config.fontsize)
        ax.set_ylabel("Y", fontsize=self.config.fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Standard histology orientation

        if save_path and created_fig:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig if created_fig else None

    def plot_violin_by_stage(
        self,
        values: np.ndarray,
        stages: np.ndarray,
        title: str,
        ylabel: str = "Score",
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        """Plot violin plots of values by stage.

        Args:
            values: [N] values to plot
            stages: [N] stage labels
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            Figure
        """
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)

        unique_stages = sorted(set(stages))
        data = [values[stages == s] for s in unique_stages]
        colors = [STAGE_COLORS.get(s, "#95a5a6") for s in unique_stages]

        parts = ax.violinplot(data, positions=range(len(unique_stages)), showmeans=True)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(unique_stages)))
        ax.set_xticklabels(unique_stages, fontsize=self.config.fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)

        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        return fig

    def plot_flow_field_panel(
        self,
        flow_metrics: FlowFieldMetrics,
        umap_coords: np.ndarray | None = None,
        save_dir: Path | str | None = None,
    ) -> plt.Figure:
        """Plot 2x2 panel of flow field metrics.

        Args:
            flow_metrics: Computed flow field metrics
            umap_coords: [N, 2] UMAP coordinates
            save_dir: Directory to save individual plots

        Returns:
            Figure with 4 subplots
        """
        if umap_coords is None:
            if self.embeddings is None:
                raise ValueError("No embeddings or UMAP coordinates provided")
            umap_coords = self.embeddings[:, :2]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.config.dpi)

        metrics = [
            ("Drift Magnitude", flow_metrics.drift_magnitude, SEQUENTIAL_CMAP),
            ("Divergence", flow_metrics.divergence, DIVERGING_CMAP),
            ("Curl Magnitude", flow_metrics.curl_magnitude, SEQUENTIAL_CMAP),
            ("Irreversibility", flow_metrics.irreversibility, "magma"),
        ]

        for ax, (name, values, cmap) in zip(axes.flat, metrics):
            # Symmetric limits for divergence
            if "Divergence" in name:
                vmax = np.percentile(np.abs(values), 95)
                vmin = -vmax
            else:
                vmin, vmax = np.percentile(values, [5, 95])

            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=values,
                cmap=cmap,
                s=self.config.point_size,
                alpha=self.config.alpha,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            plt.colorbar(scatter, ax=ax, label=name)
            ax.set_title(name, fontsize=self.config.title_fontsize)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal")

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for name, values, cmap in metrics:
                fname = name.lower().replace(" ", "_")
                self.plot_umap_colored(
                    values, name, cmap,
                    umap_coords=umap_coords,
                    save_path=save_dir / f"{fname}.png",
                )
            fig.savefig(save_dir / "flow_field_panel.png", dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved flow field panel to {save_dir}")

        return fig

    def plot_pathway_panel(
        self,
        pathways: list[str] | None = None,
        umap_coords: np.ndarray | None = None,
        save_dir: Path | str | None = None,
    ) -> plt.Figure | None:
        """Plot panel of pathway activity scores.

        Args:
            pathways: List of pathway names to plot (default: key cancer pathways)
            umap_coords: [N, 2] UMAP coordinates
            save_dir: Directory to save individual plots

        Returns:
            Figure with pathway subplots
        """
        if self.pathway_scores is None:
            print("No pathway scores available")
            return None

        if pathways is None:
            pathways = ["EGFR", "Hypoxia", "TGFb", "JAK-STAT", "NFkB", "TNFa"]

        if umap_coords is None:
            if self.embeddings is None:
                raise ValueError("No embeddings or UMAP coordinates provided")
            umap_coords = self.embeddings[:, :2]

        # Filter to available pathways
        available = [p for p in pathways if p in self.pathway_scores.columns]
        if not available:
            print(f"None of {pathways} found in pathway scores")
            return None

        n_plots = len(available)
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=self.config.dpi)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for ax, pathway in zip(axes, available):
            values = self.pathway_scores[pathway].values
            vmin, vmax = np.percentile(values, [5, 95])

            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=values,
                cmap="RdYlBu_r",
                s=self.config.point_size,
                alpha=self.config.alpha,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            plt.colorbar(scatter, ax=ax, label="Activity")
            ax.set_title(pathway, fontsize=self.config.title_fontsize)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal")

        # Hide unused axes
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for pathway in available:
                values = self.pathway_scores[pathway].values
                self.plot_umap_colored(
                    values, f"{pathway} Activity", "RdYlBu_r",
                    umap_coords=umap_coords,
                    save_path=save_dir / f"pathway_{pathway.lower()}.png",
                )
            fig.savefig(save_dir / "pathway_panel.png", dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved pathway panel to {save_dir}")

        return fig

    def plot_pathway_violins_by_stage(
        self,
        stages: np.ndarray,
        pathways: list[str] | None = None,
        save_dir: Path | str | None = None,
    ) -> plt.Figure | None:
        """Plot violin plots of pathway activities by stage.

        Args:
            stages: [N] stage labels
            pathways: List of pathway names to plot
            save_dir: Directory to save individual plots

        Returns:
            Figure with pathway violin plots
        """
        if self.pathway_scores is None:
            print("No pathway scores available")
            return None

        if pathways is None:
            pathways = ["EGFR", "Hypoxia", "TGFb", "JAK-STAT", "NFkB", "TNFa"]

        available = [p for p in pathways if p in self.pathway_scores.columns]
        if not available:
            return None

        n_plots = len(available)
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=self.config.dpi)
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flat

        unique_stages = sorted(set(stages))
        colors = [STAGE_COLORS.get(s, "#95a5a6") for s in unique_stages]

        for ax, pathway in zip(axes, available):
            values = self.pathway_scores[pathway].values
            data = [values[stages == s] for s in unique_stages]

            parts = ax.violinplot(data, positions=range(len(unique_stages)), showmeans=True)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

            ax.set_xticks(range(len(unique_stages)))
            ax.set_xticklabels(unique_stages)
            ax.set_ylabel("Activity")
            ax.set_title(pathway, fontsize=self.config.title_fontsize)

        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for pathway in available:
                values = self.pathway_scores[pathway].values
                self.plot_violin_by_stage(
                    values, stages, f"{pathway} Activity by Stage",
                    ylabel="Activity Score",
                    save_path=save_dir / f"violin_{pathway.lower()}.png",
                )
            fig.savefig(save_dir / "pathway_violins.png", dpi=self.config.dpi, bbox_inches="tight")
            print(f"Saved pathway violins to {save_dir}")

        return fig

    def generate_all_figures(
        self,
        umap_coords: np.ndarray | None = None,
        spatial_coords: np.ndarray | None = None,
        stages: np.ndarray | None = None,
        save_dir: Path | str | None = None,
    ) -> dict:
        """Generate all available figures.

        Args:
            umap_coords: [N, 2] UMAP coordinates
            spatial_coords: [N, 2] spatial coordinates
            stages: [N] stage labels
            save_dir: Directory to save all figures

        Returns:
            Dictionary of figure names to Figure objects
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        figures = {}

        # Use stage_idx from predictions if not provided
        if stages is None and "stage_idx" in self.predictions.columns:
            stages = self.predictions["stage_idx"].values

        # Flow field metrics
        if self.embeddings is not None and self.displacements is not None:
            flow_metrics = self.compute_flow_metrics()
            if flow_metrics is not None:
                figures["flow_field"] = self.plot_flow_field_panel(
                    flow_metrics, umap_coords,
                    save_dir=save_dir / "flow_fields" if save_dir else None,
                )

                # Violin plots by stage
                if stages is not None:
                    for name, values in [
                        ("drift", flow_metrics.drift_magnitude),
                        ("divergence", flow_metrics.divergence),
                        ("curl", flow_metrics.curl_magnitude),
                        ("irreversibility", flow_metrics.irreversibility),
                    ]:
                        figures[f"violin_{name}"] = self.plot_violin_by_stage(
                            values, stages, f"{name.title()} by Stage",
                            save_path=save_dir / f"violin_{name}.png" if save_dir else None,
                        )

        # Pathway scores
        if self.pathway_scores is not None:
            figures["pathway_panel"] = self.plot_pathway_panel(
                umap_coords=umap_coords,
                save_dir=save_dir / "pathways" if save_dir else None,
            )
            if stages is not None:
                figures["pathway_violins"] = self.plot_pathway_violins_by_stage(
                    stages,
                    save_dir=save_dir / "pathways" if save_dir else None,
                )

        # Proliferation
        if self.proliferation_scores is not None:
            prolif = self.proliferation_scores["proliferation_score"].values
            if umap_coords is not None or self.embeddings is not None:
                figures["proliferation_umap"] = self.plot_umap_colored(
                    prolif, "Proliferation Score", "YlOrRd",
                    umap_coords=umap_coords,
                    save_path=save_dir / "proliferation_umap.png" if save_dir else None,
                )
            if stages is not None:
                figures["proliferation_violin"] = self.plot_violin_by_stage(
                    prolif, stages, "Proliferation by Stage",
                    ylabel="Proliferation Score",
                    save_path=save_dir / "proliferation_violin.png" if save_dir else None,
                )

        # Spatial plots if coordinates provided
        if spatial_coords is not None:
            spatial_dir = save_dir / "spatial" if save_dir else None
            if spatial_dir:
                spatial_dir.mkdir(parents=True, exist_ok=True)

            if self.proliferation_scores is not None:
                figures["proliferation_spatial"] = self.plot_spatial_colored(
                    self.proliferation_scores["proliferation_score"].values,
                    spatial_coords, "Proliferation",
                    cmap="YlOrRd",
                    save_path=spatial_dir / "proliferation.png" if spatial_dir else None,
                )

            if self.pathway_scores is not None:
                for pathway in ["EGFR", "Hypoxia", "TGFb", "NFkB"]:
                    if pathway in self.pathway_scores.columns:
                        figures[f"{pathway}_spatial"] = self.plot_spatial_colored(
                            self.pathway_scores[pathway].values,
                            spatial_coords, pathway,
                            cmap="RdYlBu_r",
                            save_path=spatial_dir / f"{pathway.lower()}.png" if spatial_dir else None,
                        )

            if self.displacements is not None:
                flow_metrics = self.compute_flow_metrics()
                for name, values, cmap in [
                    ("drift", flow_metrics.drift_magnitude, "viridis"),
                    ("divergence", flow_metrics.divergence, DIVERGING_CMAP),
                    ("irreversibility", flow_metrics.irreversibility, "magma"),
                ]:
                    figures[f"{name}_spatial"] = self.plot_spatial_colored(
                        values, spatial_coords, name.title(),
                        cmap=cmap,
                        save_path=spatial_dir / f"{name}.png" if spatial_dir else None,
                    )

        print(f"Generated {len(figures)} figures")
        return figures
