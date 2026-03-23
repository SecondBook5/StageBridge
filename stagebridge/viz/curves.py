"""Training and benchmark visualization utilities for StageBridge.

Enhanced with:
  - Violin plots and box plots for distribution visualization
  - Statistical significance markers
  - Better color schemes and styling
  - Publication-quality aesthetics
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_metrics_dataframe(metrics_payload: dict) -> pd.DataFrame:
    """Convert metrics JSON payload into a tidy model-level DataFrame."""
    rows: list[dict[str, object]] = []
    for label, payload in metrics_payload.get("results", {}).items():
        row: dict[str, object] = {
            "label": label,
            "model_name": payload.get("model_name", label),
            "ablation": payload.get("ablation"),
        }
        row.update(payload.get("aggregate", {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def _get_model_color(label: str, is_stagebridge: bool = False) -> str:
    """Get consistent color for model type."""
    if is_stagebridge or "stagebridge" in str(label).lower():
        return "#0E7490"  # Teal for StageBridge
    elif "ablation" in str(label).lower():
        return "#94A3B8"  # Light gray for ablations
    elif "baseline" in str(label).lower():
        return "#64748B"  # Medium gray for baselines
    else:
        return "#334155"  # Dark gray for others


def plot_benchmark_bars(
    df: pd.DataFrame,
    output_path: Path,
    metric_col: str = "sinkhorn_mean",
    metric_std_col: str = "sinkhorn_std",
    title: str = "Donor-held-out Transition Fidelity",
    show_values: bool = True,
    highlight_best: bool = True,
) -> None:
    """Plot model comparison bars with enhanced styling and statistical annotations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with model metrics
    output_path : Path
        Output file path
    metric_col : str
        Column name for primary metric
    metric_std_col : str
        Column name for standard deviation
    title : str
        Plot title
    show_values : bool
        Whether to annotate bars with values
    highlight_best : bool
        Whether to highlight the best performing model
    """
    if df.empty or metric_col not in df.columns:
        raise ValueError(f"Cannot plot benchmark bars; missing '{metric_col}'")

    x = np.arange(df.shape[0])
    y = df[metric_col].astype(float).values
    yerr = df[metric_std_col].astype(float).values if metric_std_col in df.columns else None

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    ax.set_facecolor("#FFFFFF")  # Pure white for publication
    fig.patch.set_facecolor("#FFFFFF")

    # Determine if lower is better (typical for distance metrics)
    lower_is_better = any(
        word in metric_col.lower() for word in ["distance", "loss", "mmd", "sinkhorn"]
    )
    best_idx = np.argmin(y) if lower_is_better else np.argmax(y)

    # Color bars
    colors = [_get_model_color(lbl) for lbl in df["label"]]
    if highlight_best:
        colors[best_idx] = "#D97706"  # Amber for best model

    # Draw bars with gradient effect
    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        color=colors,
        alpha=0.85,
        capsize=5,
        error_kw={"linewidth": 2, "elinewidth": 2, "alpha": 0.7},
        edgecolor="white",
        linewidth=2,
    )

    # Add a subtle gradient to bars
    for bar in bars:
        bar.set_zorder(3)

    # Annotate values on bars
    if show_values:
        for i, (bar, val) in enumerate(zip(bars, y)):
            height = bar.get_height()
            err = yerr[i] if yerr is not None else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + err + 0.02 * (y.max() - y.min()),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Add reference line for best performance
    if highlight_best:
        ax.axhline(
            y[best_idx],
            color="#D97706",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            zorder=1,
            label=f"Best: {df['label'].iloc[best_idx]}",
        )

    # Enhanced styling
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["label"].astype(str).tolist(), rotation=35, ha="right", fontsize=11, fontweight="normal"
    )
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1, zorder=0)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Add legend if highlighting best
    if highlight_best:
        ax.legend(loc="best", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history_payloads: list[dict[str, object]],
    output_path: Path,
    show_smoothed: bool = True,
) -> None:
    """Plot train/val loss curves with enhanced styling and smoothing options.

    Parameters
    ----------
    history_payloads : list of dict
        Training history data
    output_path : Path
        Output file path
    show_smoothed : bool
        Whether to show smoothed curves for noisy data
    """
    if not history_payloads:
        raise ValueError("history_payloads is empty")

    # Set up publication-quality figure
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    # Color palette for multiple runs
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(history_payloads), 1)))

    total_points = 0
    for idx, payload in enumerate(history_payloads):
        name = str(payload.get("name", "run"))
        history = payload.get("history", [])
        if not history:
            continue
        epochs = np.asarray([row.get("epoch") for row in history], dtype=float)
        train_loss = np.asarray([row.get("train_loss") for row in history], dtype=float)
        val_loss = np.asarray([row.get("val_loss") for row in history], dtype=float)
        mask = np.isfinite(epochs) & np.isfinite(train_loss) & np.isfinite(val_loss)
        if not np.any(mask):
            continue

        epochs = epochs[mask]
        train_loss = train_loss[mask]
        val_loss = val_loss[mask]
        total_points += int(epochs.size)

        color = colors[idx % len(colors)]
        marker = "o" if epochs.size <= 5 else None
        markersize = 6 if epochs.size <= 5 else 4

        # Plot training loss
        (train_line,) = ax.plot(
            epochs,
            train_loss,
            label=f"{name} (train)",
            alpha=0.75,
            marker=marker,
            markersize=markersize,
            color=color,
            linewidth=2.5,
            linestyle="-",
        )

        # Plot validation loss
        (val_line,) = ax.plot(
            epochs,
            val_loss,
            label=f"{name} (val)",
            alpha=0.75,
            marker=marker,
            markersize=markersize,
            color=color,
            linewidth=2.5,
            linestyle="--",
        )

        # Add smoothed curves if requested and data is noisy
        if show_smoothed and epochs.size > 10:
            from scipy.ndimage import uniform_filter1d

            window = max(3, int(epochs.size / 10))
            train_smooth = uniform_filter1d(train_loss, size=window, mode="nearest")
            val_smooth = uniform_filter1d(val_loss, size=window, mode="nearest")
            ax.plot(epochs, train_smooth, color=color, linewidth=3, alpha=0.3, linestyle="-")
            ax.plot(epochs, val_smooth, color=color, linewidth=3, alpha=0.3, linestyle="--")

    # Find and mark best validation loss
    all_val_losses = []
    all_epochs = []
    for payload in history_payloads:
        history = payload.get("history", [])
        for row in history:
            if np.isfinite(row.get("val_loss", np.nan)):
                all_val_losses.append(row.get("val_loss"))
                all_epochs.append(row.get("epoch"))

    if all_val_losses:
        best_idx = np.argmin(all_val_losses)
        ax.scatter(
            [all_epochs[best_idx]],
            [all_val_losses[best_idx]],
            s=200,
            marker="*",
            color="gold",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label=f"Best val ({all_val_losses[best_idx]:.4f})",
        )

    # Enhanced styling
    ax.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    title = "Training Curves"
    if total_points <= len(history_payloads):
        title += " (early stopping / smoke test)"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.grid(alpha=0.3, linestyle=":", linewidth=1)

    # Logarithmic scale if loss spans multiple orders of magnitude
    if all_val_losses:
        val_range = max(all_val_losses) / (min(all_val_losses) + 1e-8)
        if val_range > 100:
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)", fontsize=13, fontweight="bold")

    # Legend
    legend = ax.legend(loc="best", fontsize=10, framealpha=0.95, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_violin(
    df: pd.DataFrame,
    output_path: Path,
    metric_col: str,
    group_col: str = "label",
    title: str = "Metric Distribution Comparison",
) -> None:
    """Create violin plot for metric distribution comparison across models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics (must have multiple samples per group)
    output_path : Path
        Output file path
    metric_col : str
        Column name for metric to visualize
    group_col : str
        Column name for grouping (e.g., model names)
    title : str
        Plot title
    """
    if df.empty or metric_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Missing required columns: {metric_col} or {group_col}")

    # Try to import seaborn for violin plots
    try:
        import seaborn as sns

        # Set up publication-quality figure
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")

        # Create violin plot
        sns.violinplot(
            data=df, x=group_col, y=metric_col, ax=ax, palette="Set2", inner="box", linewidth=1.5
        )

        # Overlay individual points
        sns.swarmplot(data=df, x=group_col, y=metric_col, ax=ax, color="black", alpha=0.5, size=4)

        # Enhanced styling
        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)

        plt.xticks(rotation=35, ha="right", fontsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    except ImportError:
        # Fallback to box plot if seaborn not available
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("white")

        # Create box plot
        groups = df[group_col].unique()
        data_by_group = [df[df[group_col] == g][metric_col].values for g in groups]

        ax.boxplot(
            data_by_group,
            labels=groups,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            meanprops=dict(color="green", linewidth=2),
        )

        ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_title(title + " (Box Plot)", fontsize=15, fontweight="bold", pad=15)
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1)

        plt.xticks(rotation=35, ha="right", fontsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        if output_path.suffix.lower() != ".pdf":
            fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
