"""Figure builders for EA-MIST benchmark outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_method_overview_figure(path: str | Path) -> str:
    """Create a simple publication-facing EA-MIST method overview panel."""
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, 0.14, 0.22, "Spatial lesion\n+ snRNA + WES"),
        (0.24, 0.55, 0.16, 0.22, "Local niche\nconstruction"),
        (0.46, 0.55, 0.16, 0.22, "Local niche\nencoder"),
        (0.68, 0.55, 0.12, 0.22, "Prototype\nbank"),
        (0.84, 0.55, 0.12, 0.22, "Lesion Set\nTransformer"),
        (0.84, 0.18, 0.12, 0.18, "Stage + weak\n displacement\n + aux edges"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#eef3f8", edgecolor="#12344d", linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#12344d")
    arrows = [
        ((0.19, 0.66), (0.24, 0.66)),
        ((0.40, 0.66), (0.46, 0.66)),
        ((0.62, 0.66), (0.68, 0.66)),
        ((0.80, 0.66), (0.84, 0.66)),
        ((0.90, 0.55), (0.90, 0.36)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#12344d"})
    ax.text(0.50, 0.08, "Figure 1. EA-MIST method overview", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_embedding_diagnostics_figure(
    embeddings: pd.DataFrame,
    path: str | Path,
    *,
    color_column: str = "stage",
) -> str:
    """Create diagnostic embedding projections labeled as diagnostics only."""
    path = _ensure_parent(path)
    embedding_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    if len(embedding_cols) < 2:
        raise ValueError("Embedding diagnostics require at least two embedding columns.")
    x = embeddings[embedding_cols].to_numpy(dtype=np.float32)
    labels = embeddings[color_column].astype(str).to_numpy() if color_column in embeddings.columns else np.array(["unknown"] * x.shape[0])
    pca = PCA(n_components=2, random_state=0).fit_transform(x)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0, perplexity=max(5, min(30, x.shape[0] // 3))).fit_transform(x) if x.shape[0] >= 10 else pca
    try:
        import umap

        umap_proj = umap.UMAP(n_components=2, random_state=0).fit_transform(x)
    except Exception:
        umap_proj = pca
    projections = [("PCA", pca), ("UMAP", umap_proj), ("t-SNE", tsne)]
    fig, axes = plt.subplots(1, len(projections), figsize=(14, 4))
    unique_labels = sorted(pd.unique(labels))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(unique_labels), 1)))
    color_map = {label: colors[idx] for idx, label in enumerate(unique_labels)}
    for ax, (title, proj) in zip(axes, projections):
        for label in unique_labels:
            mask = labels == label
            ax.scatter(proj[mask, 0], proj[mask, 1], s=14, alpha=0.8, label=label, color=color_map[label])
        ax.set_title(f"{title} (diagnostic)")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_benchmark_comparison_figure(summary: pd.DataFrame, path: str | Path) -> str:
    """Create a stage/displacement benchmark figure by reference mode and model."""
    path = _ensure_parent(path)
    agg = (
        summary.groupby(["reference_feature_mode", "model_family"], as_index=False)
        .agg(
            stage_macro_f1_mean=("stage_macro_f1", "mean"),
            stage_macro_f1_std=("stage_macro_f1", "std"),
            displacement_spearman_mean=("displacement_spearman", "mean"),
            displacement_spearman_std=("displacement_spearman", "std"),
        )
        .reset_index(drop=True)
    )
    reference_modes = agg["reference_feature_mode"].astype(str).unique().tolist()
    fig, axes = plt.subplots(len(reference_modes), 2, figsize=(12, 4 * max(len(reference_modes), 1)))
    if len(reference_modes) == 1:
        axes = np.asarray([axes])
    for row_idx, reference_mode in enumerate(reference_modes):
        edge_frame = agg[agg["reference_feature_mode"] == reference_mode].sort_values("stage_macro_f1_mean", ascending=False)
        for col_idx, metric in enumerate(("stage_macro_f1", "displacement_spearman")):
            ax = axes[row_idx, col_idx]
            ax.bar(edge_frame["model_family"], edge_frame[f"{metric}_mean"], yerr=edge_frame[f"{metric}_std"].fillna(0.0), color="#4c78a8")
            ax.set_title(f"{reference_mode} {metric.replace('_', ' ').title()}")
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_prototype_interpretation_figure(prototype_frame: pd.DataFrame, path: str | Path) -> str:
    """Create a compact prototype occupancy heatmap."""
    path = _ensure_parent(path)
    if prototype_frame.empty:
        raise ValueError("Prototype interpretation figure requires a non-empty prototype frame.")
    pivot = prototype_frame.pivot_table(index="sample_id", columns="prototype", values="occupancy", fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(dtype=np.float32), aspect="auto", cmap="magma")
    ax.set_title("Figure 4. Prototype occupancy by lesion")
    ax.set_xlabel("Prototype")
    ax.set_ylabel("Sample")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(col) for col in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(idx) for idx in pivot.index], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_ablation_figure(ablation_frame: pd.DataFrame, path: str | Path) -> str:
    """Create a simple ablation comparison panel."""
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ablation_frame["ablation"], ablation_frame["delta_stage_macro_f1"], color="#72b7b2")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta Stage Macro-F1")
    ax.set_title("Figure 5. Ablation effects")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)
