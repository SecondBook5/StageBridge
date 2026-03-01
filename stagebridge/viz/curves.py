"""Training and benchmark visualization utilities for StageBridge."""
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


def plot_benchmark_bars(
    df: pd.DataFrame,
    output_path: Path,
    metric_col: str = "sinkhorn_mean",
    metric_std_col: str = "sinkhorn_std",
    title: str = "Donor-held-out Transition Fidelity",
) -> None:
    """Plot model comparison bars with optional uncertainty whiskers."""
    if df.empty or metric_col not in df.columns:
        raise ValueError(f"Cannot plot benchmark bars; missing '{metric_col}'")

    x = np.arange(df.shape[0])
    y = df[metric_col].astype(float).values
    yerr = df[metric_std_col].astype(float).values if metric_std_col in df.columns else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#0E7490" if "stagebridge" in str(lbl).lower() else "#334155" for lbl in df["label"]]
    ax.bar(x, y, yerr=yerr, color=colors, alpha=0.9, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].astype(str).tolist(), rotation=25, ha="right")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_training_curves(
    history_payloads: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Plot train/val loss curves across one or more runs."""
    if not history_payloads:
        raise ValueError("history_payloads is empty")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for payload in history_payloads:
        name = str(payload.get("name", "run"))
        history = payload.get("history", [])
        if not history:
            continue
        epochs = [row.get("epoch") for row in history]
        train_loss = [row.get("train_loss") for row in history]
        val_loss = [row.get("val_loss") for row in history]
        ax.plot(epochs, train_loss, label=f"{name} train", alpha=0.85)
        ax.plot(epochs, val_loss, linestyle="--", label=f"{name} val", alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
