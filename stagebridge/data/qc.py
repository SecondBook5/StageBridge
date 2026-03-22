"""
Quality control filtering and visualization for StageBridge.

This module handles:
- QC threshold configuration
- Per-modality filtering (single-cell, spatial)
- QC metric computation
- QC figure generation (per-donor and dataset-level)
- Integration with doublet detection and ambient RNA correction

Usage:
    from stagebridge.data.qc import QCConfig, run_qc, generate_qc_figures

    config = QCConfig(min_counts=500, max_mito_pct=20.0)
    result = run_qc(adata, config)
    generate_qc_figures(adata, result, output_dir)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QCConfig:
    """Configuration for quality control filtering.

    All thresholds are optional. Set to None to disable a filter.

    Attributes
    ----------
    min_counts : int, optional
        Minimum total counts per cell/spot.
    max_counts : int, optional
        Maximum total counts per cell/spot.
    min_genes : int, optional
        Minimum detected genes per cell/spot.
    max_genes : int, optional
        Maximum detected genes per cell/spot.
    max_mito_pct : float, optional
        Maximum mitochondrial percentage.
    min_cells_per_gene : int, optional
        Minimum cells expressing a gene (for gene filtering).
    max_doublet_score : float, optional
        Maximum doublet score (requires scrublet/doubletfinder).
    spot_tissue_filter : bool
        Whether to filter spots outside tissue (spatial only).
    modality : str
        Data modality ("scrna", "snrna", "spatial").
    """

    # Cell/spot-level thresholds
    min_counts: int | None = 500
    max_counts: int | None = 50000
    min_genes: int | None = 200
    max_genes: int | None = 8000
    max_mito_pct: float | None = 20.0

    # Gene-level thresholds
    min_cells_per_gene: int | None = 3

    # Doublet detection
    max_doublet_score: float | None = None

    # Spatial-specific
    spot_tissue_filter: bool = True

    # Modality
    modality: Literal["scrna", "snrna", "spatial"] = "snrna"

    # Batch column for per-batch QC
    batch_column: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QCConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default_snrna(cls) -> "QCConfig":
        """Default config for snRNA-seq."""
        return cls(
            min_counts=500,
            max_counts=50000,
            min_genes=200,
            max_genes=8000,
            max_mito_pct=20.0,
            min_cells_per_gene=3,
            modality="snrna",
        )

    @classmethod
    def default_spatial(cls) -> "QCConfig":
        """Default config for spatial transcriptomics (Visium)."""
        return cls(
            min_counts=200,
            max_counts=100000,
            min_genes=100,
            max_genes=10000,
            max_mito_pct=30.0,  # Higher threshold for spatial
            min_cells_per_gene=1,
            spot_tissue_filter=True,
            modality="spatial",
        )

    @classmethod
    def lenient(cls) -> "QCConfig":
        """Lenient config for exploratory analysis."""
        return cls(
            min_counts=100,
            max_counts=None,
            min_genes=50,
            max_genes=None,
            max_mito_pct=50.0,
            min_cells_per_gene=1,
        )


@dataclass
class QCMetrics:
    """QC metrics for a single cell/spot."""

    n_counts: int
    n_genes: int
    mito_pct: float
    doublet_score: float | None = None
    in_tissue: bool | None = None


@dataclass
class QCResult:
    """Result of QC filtering."""

    config: QCConfig
    n_cells_pre: int
    n_cells_post: int
    n_genes_pre: int
    n_genes_post: int
    n_filtered_min_counts: int = 0
    n_filtered_max_counts: int = 0
    n_filtered_min_genes: int = 0
    n_filtered_max_genes: int = 0
    n_filtered_mito: int = 0
    n_filtered_doublet: int = 0
    n_filtered_tissue: int = 0
    n_genes_filtered: int = 0
    per_donor_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    per_stage_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = field(default_factory=list)

    @property
    def retention_rate(self) -> float:
        """Percentage of cells retained after filtering."""
        if self.n_cells_pre == 0:
            return 0.0
        return 100.0 * self.n_cells_post / self.n_cells_pre

    @property
    def n_filtered_total(self) -> int:
        """Total number of cells filtered."""
        return self.n_cells_pre - self.n_cells_post

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config.to_dict(),
            "n_cells_pre": self.n_cells_pre,
            "n_cells_post": self.n_cells_post,
            "n_genes_pre": self.n_genes_pre,
            "n_genes_post": self.n_genes_post,
            "n_filtered_min_counts": self.n_filtered_min_counts,
            "n_filtered_max_counts": self.n_filtered_max_counts,
            "n_filtered_min_genes": self.n_filtered_min_genes,
            "n_filtered_max_genes": self.n_filtered_max_genes,
            "n_filtered_mito": self.n_filtered_mito,
            "n_filtered_doublet": self.n_filtered_doublet,
            "n_filtered_tissue": self.n_filtered_tissue,
            "n_genes_filtered": self.n_genes_filtered,
            "retention_rate_pct": self.retention_rate,
            "per_donor_stats": self.per_donor_stats,
            "per_stage_stats": self.per_stage_stats,
            "executed_at": self.executed_at,
            "warnings": self.warnings,
        }

    def save(self, path: Path | str) -> None:
        """Save QC result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# QC metric computation
# ---------------------------------------------------------------------------


def _require_anndata():
    """Import anndata lazily."""
    try:
        import anndata
    except ImportError as e:
        raise ImportError("anndata is required for QC operations") from e
    return anndata


def _require_scanpy():
    """Import scanpy lazily."""
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError("scanpy is required for QC operations") from e
    return sc


def compute_qc_metrics(
    adata: Any,  # AnnData
    *,
    mito_prefix: str = "MT-",
    compute_doublets: bool = False,
) -> None:
    """Compute QC metrics and add to adata.obs.

    Adds columns:
    - n_counts: total counts per cell
    - n_genes: detected genes per cell
    - pct_counts_mito: mitochondrial percentage
    - doublet_score (optional): scrublet doublet score

    Parameters
    ----------
    adata : AnnData
        AnnData object (modified in place).
    mito_prefix : str
        Prefix for mitochondrial genes.
    compute_doublets : bool
        Whether to compute doublet scores using scrublet.
    """
    sc = _require_scanpy()

    # Basic QC metrics via scanpy
    mito_genes = adata.var_names.str.upper().str.startswith(mito_prefix.upper())

    if mito_genes.sum() == 0:
        log.warning(
            "No mitochondrial genes found with prefix '%s'. "
            "Check gene naming convention (expected HGNC symbols like MT-CO1).",
            mito_prefix,
        )

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mito"] if "mito" in adata.var.columns else [],
        percent_top=None,
        inplace=True,
    )

    # Ensure standard column names
    if "total_counts" in adata.obs.columns:
        adata.obs["n_counts"] = adata.obs["total_counts"].astype(int)
    if "n_genes_by_counts" in adata.obs.columns:
        adata.obs["n_genes"] = adata.obs["n_genes_by_counts"].astype(int)

    # Compute mitochondrial percentage manually if not done
    if "pct_counts_mito" not in adata.obs.columns:
        if mito_genes.sum() > 0:
            mito_counts = np.asarray(adata[:, mito_genes].X.sum(axis=1)).ravel()
            total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                pct_mito = np.where(total_counts > 0, 100.0 * mito_counts / total_counts, 0.0)
            adata.obs["pct_counts_mito"] = pct_mito.astype(np.float32)
        else:
            adata.obs["pct_counts_mito"] = 0.0

    # Compute doublet scores if requested
    if compute_doublets:
        try:
            import scrublet as scr

            log.info("Computing doublet scores with scrublet...")
            scrub = scr.Scrublet(adata.X)
            doublet_scores, predicted_doublets = scrub.scrub_doublets()
            adata.obs["doublet_score"] = doublet_scores.astype(np.float32)
            adata.obs["predicted_doublet"] = predicted_doublets
            log.info(
                "Scrublet: predicted %d doublets (%.1f%%)",
                predicted_doublets.sum(),
                100.0 * predicted_doublets.sum() / len(predicted_doublets),
            )
        except ImportError:
            log.warning("scrublet not installed. Skipping doublet detection.")
        except Exception as e:
            log.warning("Doublet detection failed: %s", e)

    if adata.n_obs > 0:
        log.info(
            "Computed QC metrics: n_counts [%d, %d], n_genes [%d, %d], mito%% [%.1f, %.1f]",
            int(adata.obs["n_counts"].min()),
            int(adata.obs["n_counts"].max()),
            int(adata.obs["n_genes"].min()),
            int(adata.obs["n_genes"].max()),
            float(adata.obs["pct_counts_mito"].min()),
            float(adata.obs["pct_counts_mito"].max()),
        )
    else:
        log.info("Computed QC metrics: empty AnnData (0 cells)")


# ---------------------------------------------------------------------------
# QC filtering
# ---------------------------------------------------------------------------


def run_qc(
    adata: Any,  # AnnData
    config: QCConfig,
    *,
    donor_column: str = "donor_id",
    stage_column: str = "stage",
    copy: bool = True,
) -> tuple[Any, QCResult]:
    """Apply QC filtering to AnnData.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    config : QCConfig
        QC configuration with thresholds.
    donor_column : str
        Column name for donor IDs (for per-donor stats).
    stage_column : str
        Column name for stage labels (for per-stage stats).
    copy : bool
        Whether to return a copy (True) or filter in place (False).

    Returns
    -------
    tuple[AnnData, QCResult]
        Filtered AnnData and QC result.
    """
    _require_anndata()

    if copy:
        adata = adata.copy()

    n_cells_pre = adata.n_obs
    n_genes_pre = adata.n_vars

    log.info("Running QC on %d cells, %d genes ...", n_cells_pre, n_genes_pre)

    # Ensure QC metrics are computed
    if "n_counts" not in adata.obs.columns:
        log.info("Computing QC metrics...")
        compute_qc_metrics(adata, compute_doublets=config.max_doublet_score is not None)

    # Initialize filter mask (True = keep)
    keep_mask = np.ones(adata.n_obs, dtype=bool)

    result = QCResult(
        config=config,
        n_cells_pre=n_cells_pre,
        n_cells_post=0,
        n_genes_pre=n_genes_pre,
        n_genes_post=0,
    )

    # Apply cell-level filters
    if config.min_counts is not None:
        fail = adata.obs["n_counts"] < config.min_counts
        result.n_filtered_min_counts = int(fail.sum())
        keep_mask &= ~fail
        log.debug("min_counts filter: %d cells removed", result.n_filtered_min_counts)

    if config.max_counts is not None:
        fail = adata.obs["n_counts"] > config.max_counts
        result.n_filtered_max_counts = int(fail.sum())
        keep_mask &= ~fail
        log.debug("max_counts filter: %d cells removed", result.n_filtered_max_counts)

    if config.min_genes is not None:
        fail = adata.obs["n_genes"] < config.min_genes
        result.n_filtered_min_genes = int(fail.sum())
        keep_mask &= ~fail
        log.debug("min_genes filter: %d cells removed", result.n_filtered_min_genes)

    if config.max_genes is not None:
        fail = adata.obs["n_genes"] > config.max_genes
        result.n_filtered_max_genes = int(fail.sum())
        keep_mask &= ~fail
        log.debug("max_genes filter: %d cells removed", result.n_filtered_max_genes)

    if config.max_mito_pct is not None:
        if "pct_counts_mito" in adata.obs.columns:
            fail = adata.obs["pct_counts_mito"] > config.max_mito_pct
            result.n_filtered_mito = int(fail.sum())
            keep_mask &= ~fail
            log.debug("mito filter: %d cells removed", result.n_filtered_mito)
        else:
            result.warnings.append("pct_counts_mito not found, skipping mito filter")

    if config.max_doublet_score is not None:
        if "doublet_score" in adata.obs.columns:
            fail = adata.obs["doublet_score"] > config.max_doublet_score
            result.n_filtered_doublet = int(fail.sum())
            keep_mask &= ~fail
            log.debug("doublet filter: %d cells removed", result.n_filtered_doublet)
        else:
            result.warnings.append("doublet_score not found, skipping doublet filter")

    # Spatial-specific: in_tissue filter
    if config.spot_tissue_filter and config.modality == "spatial":
        if "in_tissue" in adata.obs.columns:
            fail = ~adata.obs["in_tissue"].astype(bool)
            result.n_filtered_tissue = int(fail.sum())
            keep_mask &= ~fail
            log.debug("tissue filter: %d spots removed", result.n_filtered_tissue)
        else:
            result.warnings.append("in_tissue column not found, skipping tissue filter")

    # Collect per-donor stats before filtering
    if donor_column in adata.obs.columns:
        pre_counts = adata.obs[donor_column].value_counts().to_dict()
    else:
        pre_counts = {}

    # Apply cell filter
    adata_filtered = adata[keep_mask, :].copy()

    # Gene filtering
    if config.min_cells_per_gene is not None and config.min_cells_per_gene > 0:
        gene_counts = np.asarray((adata_filtered.X > 0).sum(axis=0)).ravel()
        gene_mask = gene_counts >= config.min_cells_per_gene
        result.n_genes_filtered = int((~gene_mask).sum())
        adata_filtered = adata_filtered[:, gene_mask].copy()
        log.debug("gene filter: %d genes removed", result.n_genes_filtered)

    result.n_cells_post = adata_filtered.n_obs
    result.n_genes_post = adata_filtered.n_vars

    # Collect per-donor and per-stage stats after filtering
    if donor_column in adata_filtered.obs.columns:
        post_counts = adata_filtered.obs[donor_column].value_counts().to_dict()
        for donor in set(pre_counts.keys()) | set(post_counts.keys()):
            result.per_donor_stats[str(donor)] = {
                "pre_qc": pre_counts.get(donor, 0),
                "post_qc": post_counts.get(donor, 0),
                "filtered": pre_counts.get(donor, 0) - post_counts.get(donor, 0),
            }

    if stage_column in adata_filtered.obs.columns:
        pre_stage = adata.obs[stage_column].value_counts().to_dict()
        post_stage = adata_filtered.obs[stage_column].value_counts().to_dict()
        for stage in set(pre_stage.keys()) | set(post_stage.keys()):
            result.per_stage_stats[str(stage)] = {
                "pre_qc": pre_stage.get(stage, 0),
                "post_qc": post_stage.get(stage, 0),
                "filtered": pre_stage.get(stage, 0) - post_stage.get(stage, 0),
            }

    log.info(
        "QC complete: %d -> %d cells (%.1f%% retained), %d -> %d genes",
        n_cells_pre,
        result.n_cells_post,
        result.retention_rate,
        n_genes_pre,
        result.n_genes_post,
    )

    if result.warnings:
        for warning in result.warnings:
            log.warning("QC warning: %s", warning)

    return adata_filtered, result


# ---------------------------------------------------------------------------
# QC figure generation
# ---------------------------------------------------------------------------


def generate_qc_figures(
    adata: Any,  # AnnData
    result: QCResult,
    output_dir: str | Path,
    *,
    donor_column: str = "donor_id",
    stage_column: str = "stage",
    format: str = "png",
    dpi: int = 150,
) -> list[Path]:
    """Generate QC figures.

    Produces:
    - Per-donor: count/gene/mito distributions, retention bar charts
    - Dataset-level: cells per donor, cells per stage, summary heatmaps

    Parameters
    ----------
    adata : AnnData
        AnnData after QC filtering.
    result : QCResult
        QC result with statistics.
    output_dir : Path
        Directory to save figures.
    donor_column : str
        Column name for donor IDs.
    stage_column : str
        Column name for stage labels.
    format : str
        Figure format (png, pdf, svg).
    dpi : int
        Figure resolution.

    Returns
    -------
    list[Path]
        Paths to generated figures.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        log.warning("matplotlib/seaborn not available, skipping figure generation: %s", e)
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # 1. Distribution plots for QC metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total counts distribution
    if "n_counts" in adata.obs.columns:
        ax = axes[0]
        data = adata.obs["n_counts"].values
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        if result.config.min_counts:
            ax.axvline(
                result.config.min_counts,
                color="red",
                linestyle="--",
                label=f"min={result.config.min_counts}",
            )
        if result.config.max_counts:
            ax.axvline(
                result.config.max_counts,
                color="red",
                linestyle="--",
                label=f"max={result.config.max_counts}",
            )
        ax.set_xlabel("Total counts")
        ax.set_ylabel("Number of cells")
        ax.set_title("Total Counts Distribution")
        ax.legend()

    # Gene counts distribution
    if "n_genes" in adata.obs.columns:
        ax = axes[1]
        data = adata.obs["n_genes"].values
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        if result.config.min_genes:
            ax.axvline(
                result.config.min_genes,
                color="red",
                linestyle="--",
                label=f"min={result.config.min_genes}",
            )
        if result.config.max_genes:
            ax.axvline(
                result.config.max_genes,
                color="red",
                linestyle="--",
                label=f"max={result.config.max_genes}",
            )
        ax.set_xlabel("Detected genes")
        ax.set_ylabel("Number of cells")
        ax.set_title("Gene Count Distribution")
        ax.legend()

    # Mitochondrial percentage distribution
    if "pct_counts_mito" in adata.obs.columns:
        ax = axes[2]
        data = adata.obs["pct_counts_mito"].values
        ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
        if result.config.max_mito_pct:
            ax.axvline(
                result.config.max_mito_pct,
                color="red",
                linestyle="--",
                label=f"max={result.config.max_mito_pct}%",
            )
        ax.set_xlabel("Mitochondrial %")
        ax.set_ylabel("Number of cells")
        ax.set_title("Mitochondrial Percentage Distribution")
        ax.legend()

    plt.tight_layout()
    fig_path = output_dir / f"qc_distributions.{format}"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figures.append(fig_path)
    log.info("Saved QC distribution figure: %s", fig_path)

    # 2. Cells per donor
    if donor_column in adata.obs.columns and result.per_donor_stats:
        fig, ax = plt.subplots(figsize=(12, 6))

        donors = sorted(result.per_donor_stats.keys())
        pre_counts = [result.per_donor_stats[d]["pre_qc"] for d in donors]
        post_counts = [result.per_donor_stats[d]["post_qc"] for d in donors]

        x = np.arange(len(donors))
        width = 0.35

        ax.bar(x - width / 2, pre_counts, width, label="Pre-QC", alpha=0.8)
        ax.bar(x + width / 2, post_counts, width, label="Post-QC", alpha=0.8)

        ax.set_xlabel("Donor")
        ax.set_ylabel("Number of cells")
        ax.set_title("Cells per Donor (Pre/Post QC)")
        ax.set_xticks(x)
        ax.set_xticklabels(donors, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        fig_path = output_dir / f"cells_per_donor.{format}"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig_path)
        log.info("Saved cells per donor figure: %s", fig_path)

    # 3. Cells per stage
    if stage_column in adata.obs.columns and result.per_stage_stats:
        fig, ax = plt.subplots(figsize=(10, 6))

        stages = sorted(result.per_stage_stats.keys())
        pre_counts = [result.per_stage_stats[s]["pre_qc"] for s in stages]
        post_counts = [result.per_stage_stats[s]["post_qc"] for s in stages]

        x = np.arange(len(stages))
        width = 0.35

        ax.bar(x - width / 2, pre_counts, width, label="Pre-QC", alpha=0.8)
        ax.bar(x + width / 2, post_counts, width, label="Post-QC", alpha=0.8)

        ax.set_xlabel("Stage")
        ax.set_ylabel("Number of cells")
        ax.set_title("Cells per Stage (Pre/Post QC)")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        fig_path = output_dir / f"cells_per_stage.{format}"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figures.append(fig_path)
        log.info("Saved cells per stage figure: %s", fig_path)

    # 4. QC filter breakdown (pie chart)
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = []
    sizes = []

    if result.n_filtered_min_counts > 0:
        labels.append(f"Low counts\n({result.n_filtered_min_counts})")
        sizes.append(result.n_filtered_min_counts)
    if result.n_filtered_max_counts > 0:
        labels.append(f"High counts\n({result.n_filtered_max_counts})")
        sizes.append(result.n_filtered_max_counts)
    if result.n_filtered_min_genes > 0:
        labels.append(f"Low genes\n({result.n_filtered_min_genes})")
        sizes.append(result.n_filtered_min_genes)
    if result.n_filtered_max_genes > 0:
        labels.append(f"High genes\n({result.n_filtered_max_genes})")
        sizes.append(result.n_filtered_max_genes)
    if result.n_filtered_mito > 0:
        labels.append(f"High mito\n({result.n_filtered_mito})")
        sizes.append(result.n_filtered_mito)
    if result.n_filtered_doublet > 0:
        labels.append(f"Doublets\n({result.n_filtered_doublet})")
        sizes.append(result.n_filtered_doublet)
    if result.n_filtered_tissue > 0:
        labels.append(f"Outside tissue\n({result.n_filtered_tissue})")
        sizes.append(result.n_filtered_tissue)

    if labels:
        labels.append(f"Retained\n({result.n_cells_post})")
        sizes.append(result.n_cells_post)

        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(f"QC Filter Breakdown\n(Total: {result.n_cells_pre} cells)")
    else:
        ax.text(0.5, 0.5, "No cells filtered", ha="center", va="center", fontsize=14)
        ax.set_title("QC Filter Breakdown")

    plt.tight_layout()
    fig_path = output_dir / f"qc_filter_breakdown.{format}"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figures.append(fig_path)
    log.info("Saved QC filter breakdown figure: %s", fig_path)

    # 5. Per-donor violin plots
    if donor_column in adata.obs.columns:
        donors = adata.obs[donor_column].unique()
        if len(donors) <= 20:  # Only if not too many donors
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            if "n_counts" in adata.obs.columns:
                sns.violinplot(data=adata.obs, x=donor_column, y="n_counts", ax=axes[0])
                axes[0].set_title("Total Counts by Donor")
                axes[0].tick_params(axis="x", rotation=45)

            if "n_genes" in adata.obs.columns:
                sns.violinplot(data=adata.obs, x=donor_column, y="n_genes", ax=axes[1])
                axes[1].set_title("Detected Genes by Donor")
                axes[1].tick_params(axis="x", rotation=45)

            if "pct_counts_mito" in adata.obs.columns:
                sns.violinplot(data=adata.obs, x=donor_column, y="pct_counts_mito", ax=axes[2])
                axes[2].set_title("Mitochondrial % by Donor")
                axes[2].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            fig_path = output_dir / f"qc_metrics_by_donor.{format}"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figures.append(fig_path)
            log.info("Saved per-donor violin plots: %s", fig_path)

    # 6. Donor-stage contingency heatmap
    if donor_column in adata.obs.columns and stage_column in adata.obs.columns:
        contingency = pd.crosstab(adata.obs[donor_column], adata.obs[stage_column])

        if contingency.shape[0] <= 20 and contingency.shape[1] <= 10:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(contingency, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title("Donor-Stage Contingency Table (Post-QC)")
            ax.set_xlabel("Stage")
            ax.set_ylabel("Donor")

            plt.tight_layout()
            fig_path = output_dir / f"donor_stage_contingency.{format}"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figures.append(fig_path)
            log.info("Saved donor-stage contingency heatmap: %s", fig_path)

    log.info("Generated %d QC figures in %s", len(figures), output_dir)
    return figures


# ---------------------------------------------------------------------------
# Per-donor QC figures
# ---------------------------------------------------------------------------


def generate_per_donor_figures(
    adata: Any,  # AnnData
    donor_id: str,
    output_dir: str | Path,
    *,
    donor_column: str = "donor_id",
    format: str = "png",
    dpi: int = 150,
) -> list[Path]:
    """Generate QC figures for a single donor.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    donor_id : str
        Donor identifier.
    output_dir : Path
        Directory to save figures.
    donor_column : str
        Column name for donor IDs.
    format : str
        Figure format.
    dpi : int
        Figure resolution.

    Returns
    -------
    list[Path]
        Paths to generated figures.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping per-donor figures")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subset to donor
    if donor_column not in adata.obs.columns:
        log.warning("Donor column '%s' not found, skipping per-donor figures", donor_column)
        return []

    donor_mask = adata.obs[donor_column].astype(str) == str(donor_id)
    if donor_mask.sum() == 0:
        log.warning("No cells found for donor '%s'", donor_id)
        return []

    donor_obs = adata.obs.loc[donor_mask]
    figures = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total counts
    if "n_counts" in donor_obs.columns:
        axes[0].hist(donor_obs["n_counts"].values, bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Total counts")
        axes[0].set_ylabel("Number of cells")
        axes[0].set_title(f"Total Counts - {donor_id}")

    # Gene counts
    if "n_genes" in donor_obs.columns:
        axes[1].hist(donor_obs["n_genes"].values, bins=30, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Detected genes")
        axes[1].set_ylabel("Number of cells")
        axes[1].set_title(f"Gene Count - {donor_id}")

    # Mitochondrial percentage
    if "pct_counts_mito" in donor_obs.columns:
        axes[2].hist(donor_obs["pct_counts_mito"].values, bins=30, edgecolor="black", alpha=0.7)
        axes[2].set_xlabel("Mitochondrial %")
        axes[2].set_ylabel("Number of cells")
        axes[2].set_title(f"Mito % - {donor_id}")

    plt.tight_layout()
    fig_path = output_dir / f"qc_{donor_id}.{format}"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    figures.append(fig_path)

    log.info("Generated per-donor QC figure for %s: %s", donor_id, fig_path)
    return figures
