"""
Ablation Summary - generate tables, reports, and visualizations from ablation results.

Generates:
- Summary tables (LaTeX, Markdown, HTML)
- Radar charts for multi-metric comparison
- Parallel coordinates for hyperparameter sensitivity
- Ridge plots for metric distributions
- Heatmaps for ablation impact
- Sankey diagrams for flow comparisons (if applicable)
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger
from .metrics import compute_ablation_metrics, compute_degradation, summarize_degradation

log = get_logger(__name__)


def generate_ablation_summary(
    all_results: dict[str, dict[str, Any]],
    baseline_results: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate summary DataFrame comparing all ablations to baseline.

    Parameters
    ----------
    all_results : dict
        Results from all ablations, keyed by name
    baseline_results : dict
        Results from full model baseline

    Returns
    -------
    pd.DataFrame
        Summary table with one row per ablation
    """
    baseline_metrics = compute_ablation_metrics(baseline_results)

    rows = []
    for name, results in all_results.items():
        if "error" in results:
            rows.append(
                {
                    "ablation": name,
                    "status": "failed",
                    "error": results["error"],
                }
            )
            continue

        ablation_metrics = compute_ablation_metrics(results)
        degradation = compute_degradation(baseline_metrics, ablation_metrics)
        degradation_level = summarize_degradation(degradation)

        ablation_info = results.get("ablation", {})

        rows.append(
            {
                "ablation": name,
                "status": "success",
                "tier": ablation_info.get("tier", "unknown"),
                "description": ablation_info.get("description", ""),
                "hypothesis": ablation_info.get("hypothesis", ""),
                "expected_degradation": ablation_info.get("expected_degradation", "unknown"),
                "actual_degradation": degradation_level,
                "transition_loss": ablation_metrics.get("transition_loss"),
                "transition_loss_delta": degradation.get("transition_loss"),
                "donor_accuracy": ablation_metrics.get("donor_held_out_accuracy"),
                "donor_accuracy_delta": degradation.get("donor_held_out_accuracy"),
                "calibration_error": ablation_metrics.get("calibration_error"),
                "calibration_error_delta": degradation.get("calibration_error"),
            }
        )

    return pd.DataFrame(rows)


def generate_ablation_table(
    summary_df: pd.DataFrame,
    output_path: Path,
    format: str = "latex",
) -> str:
    """
    Generate publication-ready ablation table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from generate_ablation_summary
    output_path : Path
        Where to save the table
    format : str
        Output format: "latex", "markdown", or "html"

    Returns
    -------
    str
        Table as string
    """
    # Select columns for publication
    pub_cols = [
        "ablation",
        "description",
        "expected_degradation",
        "actual_degradation",
        "transition_loss_delta",
        "donor_accuracy_delta",
    ]

    pub_df = summary_df[[c for c in pub_cols if c in summary_df.columns]].copy()

    # Rename for publication
    pub_df = pub_df.rename(
        columns={
            "ablation": "Ablation",
            "description": "Description",
            "expected_degradation": "Expected",
            "actual_degradation": "Observed",
            "transition_loss_delta": "ΔLoss",
            "donor_accuracy_delta": "ΔAcc",
        }
    )

    if format == "latex":
        table_str = pub_df.to_latex(index=False, float_format="%.3f")
    elif format == "markdown":
        table_str = pub_df.to_markdown(index=False, floatfmt=".3f")
    else:
        table_str = pub_df.to_html(index=False, float_format="%.3f")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_str)

    return table_str


def save_ablation_report(
    all_results: dict[str, dict[str, Any]],
    baseline_results: dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    Save complete ablation report including summary, tables, and raw data.

    Parameters
    ----------
    all_results : dict
        All ablation results
    baseline_results : dict
        Baseline results
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to saved report directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary
    summary_df = generate_ablation_summary(all_results, baseline_results)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    # Generate tables
    generate_ablation_table(summary_df, output_dir / "ablation_table.tex", format="latex")
    generate_ablation_table(summary_df, output_dir / "ablation_table.md", format="markdown")

    # Save raw results
    with open(output_dir / "ablation_results.json", "w") as f:
        # Convert to JSON-serializable format
        serializable = {k: _make_serializable(v) for k, v in all_results.items()}
        json.dump(serializable, f, indent=2)

    return output_dir


def _make_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    else:
        return obj


# =============================================================================
# VISUALIZATION REPORT GENERATION
# =============================================================================


def generate_ablation_visualizations(
    summary_df: pd.DataFrame,
    output_dir: Path,
    fold_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """
    Generate comprehensive ablation visualizations.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from generate_ablation_summary with columns:
        - ablation, transition_loss, donor_accuracy, calibration_error, etc.
    output_dir : Path
        Directory for saving figures
    fold_results : dict, optional
        Per-fold results for distribution plots (keyed by ablation name)

    Returns
    -------
    dict
        Mapping of figure names to saved paths
    """
    from stagebridge.viz.advanced_plots import (
        plot_radar_chart,
        plot_parallel_coordinates,
        plot_ridge_distributions,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    # Filter successful ablations
    success_df = summary_df[summary_df.get("status", "success") == "success"].copy()
    if success_df.empty:
        log.warning("No successful ablations to visualize")
        return figures

    # Ensure we have a label column
    if "label" not in success_df.columns:
        success_df["label"] = success_df["ablation"].str.replace("_", " ").str.title()

    # -------------------------------------------------------------------------
    # 1. Radar Chart - Multi-metric comparison across ablations
    # -------------------------------------------------------------------------
    radar_metrics = _get_available_metrics(
        success_df,
        [
            "transition_loss",
            "donor_accuracy",
            "calibration_error",
            "wasserstein",
            "mse",
            "mae",
            "stage_f1",
        ],
    )

    if len(radar_metrics) >= 3:
        try:
            # For radar, lower is better for loss/error, higher for accuracy
            # Normalize so higher = better for all
            radar_df = success_df.copy()
            for col in radar_metrics:
                if any(x in col.lower() for x in ["loss", "error", "mse", "mae", "wasserstein"]):
                    # Invert so higher = better
                    max_val = radar_df[col].max()
                    if max_val > 0:
                        radar_df[col] = 1 - (radar_df[col] / max_val)

            fig = plot_radar_chart(
                radar_df,
                metrics=radar_metrics,
                labels_col="label",
                title="Ablation Study: Multi-Metric Comparison",
                normalize=True,
            )
            path = output_dir / "ablation_radar_chart.png"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            figures["radar_chart"] = path
            log.info("Saved radar chart: %s", path)
        except Exception as e:
            log.warning("Failed to generate radar chart: %s", e)

    # -------------------------------------------------------------------------
    # 2. Parallel Coordinates - All metrics side by side
    # -------------------------------------------------------------------------
    parallel_metrics = _get_available_metrics(
        success_df,
        [
            "transition_loss",
            "donor_accuracy",
            "calibration_error",
            "transition_loss_delta",
            "donor_accuracy_delta",
            "calibration_error_delta",
        ],
    )

    if len(parallel_metrics) >= 2:
        try:
            fig = plot_parallel_coordinates(
                success_df,
                metrics=parallel_metrics,
                labels_col="label",
                title="Ablation Impact: Parallel Coordinates",
                normalize=True,
            )
            path = output_dir / "ablation_parallel_coords.png"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            figures["parallel_coordinates"] = path
            log.info("Saved parallel coordinates: %s", path)
        except Exception as e:
            log.warning("Failed to generate parallel coordinates: %s", e)

    # -------------------------------------------------------------------------
    # 3. Ablation Heatmap - Normalized degradation matrix
    # -------------------------------------------------------------------------
    try:
        fig = _plot_ablation_heatmap(success_df, output_dir)
        if fig:
            figures["heatmap"] = output_dir / "ablation_heatmap.png"
    except Exception as e:
        log.warning("Failed to generate heatmap: %s", e)

    # -------------------------------------------------------------------------
    # 4. Ridge Plots - Distribution comparison (if fold results available)
    # -------------------------------------------------------------------------
    if fold_results:
        try:
            # Collect transition loss across folds for each ablation
            loss_distributions = {}
            for ablation_name, fold_df in fold_results.items():
                if "transition_loss" in fold_df.columns:
                    label = ablation_name.replace("_", " ").title()
                    loss_distributions[label] = fold_df["transition_loss"].values

            if len(loss_distributions) >= 2:
                fig = plot_ridge_distributions(
                    loss_distributions,
                    title="Transition Loss Distribution by Ablation",
                )
                path = output_dir / "ablation_ridge_plot.png"
                fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
                plt.close(fig)
                figures["ridge_plot"] = path
                log.info("Saved ridge plot: %s", path)
        except Exception as e:
            log.warning("Failed to generate ridge plot: %s", e)

    # -------------------------------------------------------------------------
    # 5. Degradation Bar Chart - Expected vs Observed
    # -------------------------------------------------------------------------
    try:
        fig = _plot_degradation_comparison(success_df, output_dir)
        if fig:
            figures["degradation_bars"] = output_dir / "ablation_degradation_bars.png"
    except Exception as e:
        log.warning("Failed to generate degradation bars: %s", e)

    # -------------------------------------------------------------------------
    # 6. Calibration Comparison (for new calibration ablations)
    # -------------------------------------------------------------------------
    try:
        fig = _plot_calibration_comparison(success_df, output_dir)
        if fig:
            figures["calibration"] = output_dir / "ablation_calibration.png"
    except Exception as e:
        log.warning("Failed to generate calibration plot: %s", e)

    # -------------------------------------------------------------------------
    # 7. Fusion Strategy Comparison
    # -------------------------------------------------------------------------
    try:
        fig = _plot_fusion_comparison(success_df, output_dir)
        if fig:
            figures["fusion"] = output_dir / "ablation_fusion.png"
    except Exception as e:
        log.warning("Failed to generate fusion plot: %s", e)

    log.info("Generated %d ablation figures in %s", len(figures), output_dir)
    return figures


def _get_available_metrics(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return subset of candidate metrics that exist and have valid data."""
    available = []
    for col in candidates:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0 and vals.std() > 1e-10:  # Has variance
                available.append(col)
    return available


def _plot_ablation_heatmap(df: pd.DataFrame, output_dir: Path) -> plt.Figure | None:
    """Create heatmap of ablation impact on metrics."""
    # Select delta columns (degradation from baseline)
    delta_cols = [c for c in df.columns if c.endswith("_delta") and df[c].notna().any()]
    if not delta_cols:
        # Fall back to absolute metrics
        delta_cols = _get_available_metrics(
            df, ["transition_loss", "donor_accuracy", "calibration_error"]
        )

    if len(delta_cols) < 1:
        return None

    # Build matrix
    ablations = df["ablation"].values
    matrix = df[delta_cols].values.astype(float)

    # Handle NaN
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Normalize columns to [0, 1] for visualization
    col_min = matrix.min(axis=0, keepdims=True)
    col_max = matrix.max(axis=0, keepdims=True)
    matrix_norm = (matrix - col_min) / (col_max - col_min + 1e-8)

    fig, ax = plt.subplots(figsize=(10, max(6, len(ablations) * 0.5)), dpi=150)
    fig.patch.set_facecolor("white")

    im = ax.imshow(matrix_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # Labels
    metric_labels = [c.replace("_delta", "").replace("_", " ").title() for c in delta_cols]
    ax.set_xticks(np.arange(len(delta_cols)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(ablations)))
    ax.set_yticklabels([a.replace("_", " ").title() for a in ablations], fontsize=10)

    # Annotate with actual values
    for i in range(len(ablations)):
        for j in range(len(delta_cols)):
            val = matrix[i, j]
            text_color = "white" if matrix_norm[i, j] > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    # Grid lines
    for i in range(len(ablations) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(len(delta_cols) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_title(
        "Ablation Impact Heatmap (Δ from Baseline)", fontsize=14, fontweight="bold", pad=15
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Impact", fontsize=11)

    plt.tight_layout()

    path = output_dir / "ablation_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap: %s", path)

    return fig


def _plot_degradation_comparison(df: pd.DataFrame, output_dir: Path) -> plt.Figure | None:
    """Bar chart comparing expected vs observed degradation."""
    if "expected_degradation" not in df.columns or "actual_degradation" not in df.columns:
        return None

    # Map degradation levels to numeric
    deg_map = {"minimal": 1, "moderate": 2, "significant": 3, "unknown": 0}
    df = df.copy()
    df["expected_num"] = df["expected_degradation"].map(deg_map).fillna(0)
    df["actual_num"] = df["actual_degradation"].map(deg_map).fillna(0)

    ablations = df["ablation"].values
    x = np.arange(len(ablations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ax.bar(
        x - width / 2,
        df["expected_num"],
        width,
        label="Expected",
        color="#3B82F6",
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar(
        x + width / 2,
        df["actual_num"],
        width,
        label="Observed",
        color="#F97316",
        alpha=0.8,
        edgecolor="white",
        linewidth=1.5,
    )

    ax.set_ylabel("Degradation Level", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ablation", fontsize=12, fontweight="bold")
    ax.set_title("Expected vs Observed Degradation", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [a.replace("_", " ").title() for a in ablations], rotation=45, ha="right", fontsize=10
    )
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Unknown", "Minimal", "Moderate", "Significant"])
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    path = output_dir / "ablation_degradation_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved degradation bars: %s", path)

    return fig


def _plot_calibration_comparison(df: pd.DataFrame, output_dir: Path) -> plt.Figure | None:
    """Plot calibration error comparison for calibration ablations."""
    # Check if we have calibration-related ablations
    calib_ablations = df[df["ablation"].str.contains("calibration", case=False)]
    if len(calib_ablations) < 1:
        return None

    if "calibration_error" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    ablations = df["ablation"].values
    errors = df["calibration_error"].fillna(0).values
    colors = [
        "#10B981" if "temperature" in a else "#EF4444" if "no_calib" in a else "#6B7280"
        for a in ablations
    ]

    bars = ax.bar(
        range(len(ablations)), errors, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5
    )

    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ablation", fontsize=12, fontweight="bold")
    ax.set_title("Confidence Calibration Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(range(len(ablations)))
    ax.set_xticklabels(
        [a.replace("_", " ").title() for a in ablations], rotation=45, ha="right", fontsize=10
    )

    # Add value labels
    for bar, val in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()

    path = output_dir / "ablation_calibration.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration comparison: %s", path)

    return fig


def _plot_fusion_comparison(df: pd.DataFrame, output_dir: Path) -> plt.Figure | None:
    """Plot comparison of fusion strategies."""
    # Check if we have fusion-related ablations
    fusion_ablations = df[df["ablation"].str.contains("fusion|hlca_only|luca_only", case=False)]
    if len(fusion_ablations) < 2:
        return None

    metrics = _get_available_metrics(
        fusion_ablations, ["transition_loss", "donor_accuracy", "wasserstein", "stage_f1"]
    )
    if len(metrics) < 1:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), dpi=150)
    fig.patch.set_facecolor("white")

    if len(metrics) == 1:
        axes = [axes]

    ablations = fusion_ablations["ablation"].values
    colors = plt.cm.Set2(np.linspace(0, 1, len(ablations)))

    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#FAFAFA")
        values = fusion_ablations[metric].fillna(0).values

        bars = ax.bar(
            range(len(ablations)),
            values,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )

        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(ablations)))
        ax.set_xticklabels(
            [a.replace("_", " ").title() for a in ablations], rotation=45, ha="right", fontsize=9
        )

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Fusion Strategy Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = output_dir / "ablation_fusion.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fusion comparison: %s", path)

    return fig


def generate_full_ablation_report(
    all_results: dict[str, dict[str, Any]],
    baseline_results: dict[str, Any],
    output_dir: Path,
    fold_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """
    Generate complete ablation report with tables AND visualizations.

    Parameters
    ----------
    all_results : dict
        All ablation results
    baseline_results : dict
        Baseline results
    output_dir : Path
        Output directory
    fold_results : dict, optional
        Per-fold results for distribution plots

    Returns
    -------
    dict
        Report manifest with paths to all generated artifacts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary DataFrame
    summary_df = generate_ablation_summary(all_results, baseline_results)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    # Generate tables
    generate_ablation_table(summary_df, output_dir / "ablation_table.tex", format="latex")
    generate_ablation_table(summary_df, output_dir / "ablation_table.md", format="markdown")

    # Generate visualizations
    figures = generate_ablation_visualizations(summary_df, output_dir / "figures", fold_results)

    # Save raw results
    with open(output_dir / "ablation_results.json", "w") as f:
        serializable = {k: _make_serializable(v) for k, v in all_results.items()}
        json.dump(serializable, f, indent=2)

    # Create manifest
    manifest = {
        "summary_csv": str(output_dir / "ablation_summary.csv"),
        "table_latex": str(output_dir / "ablation_table.tex"),
        "table_markdown": str(output_dir / "ablation_table.md"),
        "raw_results": str(output_dir / "ablation_results.json"),
        "figures": {k: str(v) for k, v in figures.items()},
        "n_ablations": len(all_results),
        "n_successful": len(summary_df[summary_df.get("status", "success") == "success"]),
        "n_figures": len(figures),
    }

    with open(output_dir / "report_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(
        "Generated full ablation report: %d ablations, %d figures -> %s",
        len(all_results),
        len(figures),
        output_dir,
    )

    return manifest
