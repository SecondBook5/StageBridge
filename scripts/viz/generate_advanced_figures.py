#!/usr/bin/env python
"""Generate publication-quality advanced figures for EA-MIST benchmark results.

Produces:
  Panel A – Loss curves (train/val, all components)
  Panel B – Confusion matrices (per model, aggregated across folds)
  Panel C – Violin plots of per-fold metrics by model family
  Panel D – UMAP / t-SNE / PCA of pretrained neighborhood embeddings colored by stage
  Panel E – UMAP colored by cell type (receiver state)
  Panel F – Prototype occupancy heatmap (stage x prototype)
  Panel G – Displacement predicted vs true, colored by stage
  Panel H – Per-class stage probability distributions
  Panel I – Cell type composition per stage (stacked bar)
  Panel J – Ring composition heatmap (sender cell types across rings by stage)
  Panel K – HLCA/LuCA feature importance by stage
  Panel L – Attention weight distributions
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)

BENCHMARK_ROOT = Path("outputs/scratch/eamist_v1_20260309/eamist_benchmark")
PRETRAIN_ROOT = Path("outputs/scratch/eamist_v1_20260309/eamist_pretrain")
BAG_PATH = Path("/mnt/e/StageBridge_data/processed/features/eamist_bags.parquet")
OUT_DIR = Path("reports/figures/eamist_advanced")

STAGE_ORDER = ["Normal", "AAH", "AIS", "MIA", "LUAD"]
STAGE_COLORS = {"Normal": "#2ecc71", "AAH": "#f39c12", "AIS": "#e74c3c", "MIA": "#9b59b6", "LUAD": "#2c3e50"}
MODEL_ORDER = ["pooled", "deep_sets", "lesion_set_transformer", "eamist_no_prototypes", "eamist"]
MODEL_LABELS = {"pooled": "Pooled MLP", "deep_sets": "Deep Sets", "lesion_set_transformer": "Set Transformer",
                "eamist_no_prototypes": "EA-MIST (no proto)", "eamist": "EA-MIST"}
MODEL_COLORS = {"pooled": "#95a5a6", "deep_sets": "#3498db", "lesion_set_transformer": "#e67e22",
                "eamist_no_prototypes": "#9b59b6", "eamist": "#e74c3c"}


def _load_all_train_histories() -> dict[str, list[pd.DataFrame]]:
    result: dict[str, list[pd.DataFrame]] = {}
    for model in MODEL_ORDER:
        frames = []
        for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / model).glob("fold_*")):
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                path = seed_dir / "train_history.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df["fold"] = fold_dir.name
                    df["seed"] = seed_dir.name
                    frames.append(df)
        result[model] = frames
    return result


def _load_all_val_histories() -> dict[str, list[pd.DataFrame]]:
    result: dict[str, list[pd.DataFrame]] = {}
    for model in MODEL_ORDER:
        frames = []
        for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / model).glob("fold_*")):
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                path = seed_dir / "val_history.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df["fold"] = fold_dir.name
                    df["seed"] = seed_dir.name
                    frames.append(df)
        result[model] = frames
    return result


def _load_all_metrics() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / model).glob("fold_*")):
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                path = seed_dir / "metrics.json"
                if path.exists():
                    m = json.loads(path.read_text())
                    m["model_family"] = model
                    m["fold"] = fold_dir.name
                    rows.append(m)
    return pd.DataFrame(rows)


def _load_all_confusion_matrices() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for model in MODEL_ORDER:
        matrices = []
        for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / model).glob("fold_*")):
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                path = seed_dir / "confusion_matrix.json"
                if path.exists():
                    cm = json.loads(path.read_text())
                    matrices.append(np.array(cm["matrix"]))
        if matrices:
            result[model] = sum(matrices)
    return result


def _load_all_test_predictions() -> pd.DataFrame:
    frames = []
    for model in MODEL_ORDER:
        for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / model).glob("fold_*")):
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                path = seed_dir / "test_predictions.parquet"
                if path.exists():
                    df = pd.read_parquet(path)
                    df["model_family"] = model
                    df["fold"] = fold_dir.name
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_prototype_data() -> pd.DataFrame:
    frames = []
    for fold_dir in sorted((BENCHMARK_ROOT / "hlca_luca" / "eamist").glob("fold_*")):
        for seed_dir in sorted(fold_dir.glob("seed_*")):
            path = seed_dir / "prototype_composition.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df["fold"] = fold_dir.name
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_loss_curves(train_hist: dict, val_hist: dict, out: Path) -> None:
    """Panel A: Multi-component loss curves for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    loss_cols = ["loss", "stage_loss", "displacement_loss", "edge_loss", "stage_macro_f1", "displacement_spearman"]
    titles = ["Total Loss", "Stage Loss", "Displacement Loss", "Edge Loss", "Stage Macro F1 (val)", "Displacement Spearman (val)"]

    for idx, (col, title) in enumerate(zip(loss_cols, titles)):
        ax = axes[idx // 3, idx % 3]
        for model in MODEL_ORDER:
            is_val_metric = col in ("stage_macro_f1", "displacement_spearman")
            hists = val_hist.get(model, []) if is_val_metric else train_hist.get(model, [])
            if not hists or col not in hists[0].columns:
                continue
            all_epochs = pd.concat(hists)
            mean = all_epochs.groupby("epoch")[col].mean()
            std = all_epochs.groupby("epoch")[col].std().fillna(0)
            ax.plot(mean.index, mean.values, label=MODEL_LABELS[model], color=MODEL_COLORS[model], linewidth=2)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.15, color=MODEL_COLORS[model])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Training Dynamics Across Model Families", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_confusion_matrices(cms: dict, out: Path) -> None:
    """Panel B: Aggregated confusion matrices per model."""
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 4))
    if len(cms) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_ORDER):
        if model not in cms:
            continue
        cm = cms[model].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                pct = cm_norm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center", fontsize=7, color=color, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels(STAGE_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(STAGE_ORDER, fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Stage Classification Confusion Matrices (aggregated across folds)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_metric_violins(metrics_df: pd.DataFrame, out: Path) -> None:
    """Panel C: Violin plots of key metrics across folds."""
    metric_cols = ["stage_macro_f1", "stage_balanced_accuracy", "displacement_spearman", "displacement_mae"]
    metric_labels = ["Stage Macro F1", "Balanced Accuracy", "Displacement Spearman", "Displacement MAE"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, col, label in zip(axes, metric_cols, metric_labels):
        data = []
        positions = []
        colors = []
        for pos, model in enumerate(MODEL_ORDER):
            vals = metrics_df.loc[metrics_df["model_family"] == model, col].dropna().values
            if len(vals) > 0:
                data.append(vals)
                positions.append(pos)
                colors.append(MODEL_COLORS[model])
        if data:
            parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            for idx, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[idx])
                pc.set_alpha(0.7)
            # Overlay individual points
            for idx, (vals, pos) in enumerate(zip(data, positions)):
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
                ax.scatter(pos + jitter, vals, c=colors[idx], s=40, edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Fold Metric Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_embedding_manifolds(out: Path) -> None:
    """Panel D & E: UMAP/t-SNE/PCA of neighborhood embeddings colored by stage and cell type."""
    emb_path = PRETRAIN_ROOT / "neighborhood_embeddings.parquet"
    if not emb_path.exists():
        print("  Skipping embedding manifolds: no embeddings found")
        return

    df = pd.read_parquet(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    # Subsample for visualization
    rng = np.random.default_rng(42)
    n_sample = min(50000, len(df))
    idx = rng.choice(len(df), n_sample, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    X = sub[emb_cols].values.astype(np.float32)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # t-SNE on PCA-50
    pca50 = PCA(n_components=min(50, X.shape[1]), random_state=42)
    X_pca50 = pca50.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA by stage
    for stage in STAGE_ORDER:
        mask = sub["stage"] == stage
        if mask.any():
            axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=STAGE_COLORS[stage], label=stage,
                          s=2, alpha=0.4, rasterized=True)
    axes[0].set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[0].legend(markerscale=5, fontsize=9)

    # t-SNE by stage
    for stage in STAGE_ORDER:
        mask = sub["stage"] == stage
        if mask.any():
            axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=STAGE_COLORS[stage], label=stage,
                          s=2, alpha=0.4, rasterized=True)
    axes[1].set_title("t-SNE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(markerscale=5, fontsize=9)

    fig.suptitle("Pretrained Neighborhood Embeddings by Stage", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out.parent / "panel_D_embedding_stage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: panel_D_embedding_stage.png")

    # Try UMAP if available
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        X_umap = reducer.fit_transform(X_pca50)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        for stage in STAGE_ORDER:
            mask = sub["stage"] == stage
            if mask.any():
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=STAGE_COLORS[stage], label=stage,
                          s=3, alpha=0.4, rasterized=True)
        ax.set_title("UMAP of Pretrained Neighborhood Embeddings", fontsize=14, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=5, fontsize=10)
        fig.tight_layout()
        fig.savefig(out.parent / "panel_D_umap_stage.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: panel_D_umap_stage.png")
    except ImportError:
        print("  Skipping UMAP: umap-learn not installed")


def plot_cell_type_embeddings(out: Path) -> None:
    """Panel E: Embedding colored by receiver cell type."""
    emb_path = PRETRAIN_ROOT / "neighborhood_embeddings.parquet"
    if not emb_path.exists():
        return

    # Load bags to get receiver state labels
    if not BAG_PATH.exists():
        print("  Skipping cell type embeddings: bags parquet not found")
        return

    bags_df = pd.read_parquet(BAG_PATH)
    # Build niche_id -> receiver_state_label lookup
    label_lookup: dict[str, str] = {}
    for _, row in bags_df.iterrows():
        for nid, label in zip(row["niche_ids"], row["receiver_state_labels"]):
            label_lookup[str(nid)] = str(label)

    df = pd.read_parquet(emb_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    rng = np.random.default_rng(42)
    n_sample = min(50000, len(df))
    idx = rng.choice(len(df), n_sample, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    X = sub[emb_cols].values.astype(np.float32)

    # Map niche IDs if available; otherwise skip
    niche_id_col = None
    for col in ["niche_id", "neighborhood_id", "index"]:
        if col in sub.columns:
            niche_id_col = col
            break

    if niche_id_col is None:
        # Use index from original df
        orig_indices = df.index[idx]
        # We can't map directly without niche IDs in embedding frame
        print("  Skipping cell type embedding: no niche_id column in embeddings")
        return

    # PCA for cell type
    pca50 = PCA(n_components=min(50, X.shape[1]), random_state=42)
    X_pca50 = pca50.fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca50)

    cell_types = sub[niche_id_col].map(lambda x: label_lookup.get(str(x), "unknown"))
    top_types = cell_types.value_counts().head(8).index.tolist()
    ct_colors = plt.cm.Set2(np.linspace(0, 1, len(top_types)))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for ct_idx, ct in enumerate(top_types):
        mask = cell_types == ct
        if mask.any():
            ax.scatter(X_tsne[mask.values, 0], X_tsne[mask.values, 1], c=[ct_colors[ct_idx]],
                      label=ct, s=3, alpha=0.4, rasterized=True)
    other_mask = ~cell_types.isin(top_types)
    if other_mask.any():
        ax.scatter(X_tsne[other_mask.values, 0], X_tsne[other_mask.values, 1], c="lightgray",
                  label="Other", s=1, alpha=0.2, rasterized=True)
    ax.set_title("Neighborhood Embeddings by Cell Type", fontsize=14, fontweight="bold")
    ax.legend(markerscale=5, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_prototype_heatmap(proto_df: pd.DataFrame, out: Path) -> None:
    """Panel F: Prototype occupancy heatmap by stage."""
    if proto_df.empty:
        print("  Skipping prototype heatmap: no data")
        return

    pivot = proto_df.groupby(["stage", "prototype"])["occupancy"].mean().unstack(fill_value=0)
    pivot = pivot.reindex([s for s in STAGE_ORDER if s in pivot.index])

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.8), 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"P{int(c)}" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Prototype", fontsize=12)
    ax.set_ylabel("Stage", fontsize=12)
    ax.set_title("EA-MIST Prototype Occupancy by Stage", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Mean Occupancy", shrink=0.8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if val > 0.5 * pivot.values.max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_displacement_scatter(preds_df: pd.DataFrame, out: Path) -> None:
    """Panel G: Predicted vs true displacement, colored by stage."""
    if preds_df.empty:
        return

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(4 * len(MODEL_ORDER), 4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = preds_df[preds_df["model_family"] == model]
        for stage in STAGE_ORDER:
            mask = sub["stage_label"] == stage
            if mask.any():
                ax.scatter(sub.loc[mask, "displacement_target"], sub.loc[mask, "pred_displacement"],
                          c=STAGE_COLORS[stage], label=stage, s=50, edgecolors="white", linewidths=0.5, zorder=5, alpha=0.8)
        lims = [sub["displacement_target"].min() - 0.05, sub["displacement_target"].max() + 0.05]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("True Displacement")
        if ax == axes[0]:
            ax.set_ylabel("Predicted Displacement")
            ax.legend(fontsize=7, markerscale=0.8)
        ax.grid(alpha=0.3)

    fig.suptitle("Displacement Prediction: True vs Predicted", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_stage_probability_distributions(preds_df: pd.DataFrame, out: Path) -> None:
    """Panel H: Per-class probability distributions for EA-MIST predictions."""
    eamist_preds = preds_df[preds_df["model_family"] == "eamist"]
    if eamist_preds.empty:
        return

    prob_cols = [f"prob_{s}" for s in STAGE_ORDER]
    existing_prob_cols = [c for c in prob_cols if c in eamist_preds.columns]
    if not existing_prob_cols:
        return

    fig, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(4 * len(STAGE_ORDER), 4), sharey=True)
    for ax, stage, prob_col in zip(axes, STAGE_ORDER, prob_cols):
        if prob_col not in eamist_preds.columns:
            continue
        for true_stage in STAGE_ORDER:
            mask = eamist_preds["stage_label"] == true_stage
            vals = eamist_preds.loc[mask, prob_col].dropna().values
            if len(vals) > 0:
                ax.hist(vals, bins=15, alpha=0.5, color=STAGE_COLORS[true_stage], label=true_stage, density=True)
        ax.set_title(f"P({stage})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Probability")
        if ax == axes[0]:
            ax.set_ylabel("Density")
            ax.legend(fontsize=7, title="True Stage", title_fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("EA-MIST Stage Probability Distributions by True Stage", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_cell_type_composition(out: Path) -> None:
    """Panel I: Cell type composition per stage (stacked bar)."""
    if not BAG_PATH.exists():
        print("  Skipping cell type composition: bags not found")
        return

    bags_df = pd.read_parquet(BAG_PATH)
    rows = []
    for _, row in bags_df.iterrows():
        stage = str(row["stage_label"])
        labels = row["receiver_state_labels"]
        for label in labels:
            rows.append({"stage": stage, "cell_type": str(label)})
    ct_df = pd.DataFrame(rows)

    # Get top cell types
    top_ct = ct_df["cell_type"].value_counts().head(10).index.tolist()
    ct_df["cell_type_grouped"] = ct_df["cell_type"].where(ct_df["cell_type"].isin(top_ct), "Other")

    pivot = ct_df.groupby(["stage", "cell_type_grouped"]).size().unstack(fill_value=0)
    pivot = pivot.reindex([s for s in STAGE_ORDER if s in pivot.index])
    # Normalize to fractions
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(pivot_norm.columns)))
    pivot_norm.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Fraction of Neighborhoods", fontsize=12)
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_title("Cell Type Composition Across Stages", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Cell Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_ring_composition_heatmap(out: Path) -> None:
    """Panel J: Ring composition heatmap (sender features across rings by stage)."""
    if not BAG_PATH.exists():
        return

    bags_df = pd.read_parquet(BAG_PATH)
    # Aggregate ring compositions per stage
    stage_ring_means: dict[str, np.ndarray] = {}
    for _, row in bags_df.iterrows():
        stage = str(row["stage_label"])
        rings = row["ring_features"]
        # rings is an array of arrays: each element is (num_rings, sender_dim)
        ring_stack = np.stack([np.stack(r) for r in rings])  # (N, num_rings, sender_dim)
        mean_ring = ring_stack.mean(axis=0)  # (num_rings, sender_dim)
        if stage not in stage_ring_means:
            stage_ring_means[stage] = []
        stage_ring_means[stage].append(mean_ring)

    fig, axes = plt.subplots(1, len([s for s in STAGE_ORDER if s in stage_ring_means]),
                              figsize=(4 * len(stage_ring_means), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, stage in zip(axes, [s for s in STAGE_ORDER if s in stage_ring_means]):
        all_means = np.stack(stage_ring_means[stage])  # (n_lesions, num_rings, sender_dim)
        grand_mean = all_means.mean(axis=0)  # (num_rings, sender_dim)
        im = ax.imshow(grand_mean, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Sender Feature", fontsize=9)
        ax.set_ylabel("Ring", fontsize=9)
        ax.set_title(f"{stage}", fontsize=11, fontweight="bold")
        ax.set_yticks(range(grand_mean.shape[0]))
        ax.set_yticklabels([f"Ring {i}" for i in range(grand_mean.shape[0])], fontsize=8)

    fig.suptitle("Spatial Ring Composition by Stage (mean sender cell-type fractions)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_reference_feature_heatmap(out: Path) -> None:
    """Panel K: HLCA/LuCA mean feature profiles by stage."""
    if not BAG_PATH.exists():
        return

    bags_df = pd.read_parquet(BAG_PATH)
    stage_hlca: dict[str, list] = {}
    stage_luca: dict[str, list] = {}

    for _, row in bags_df.iterrows():
        stage = str(row["stage_label"])
        hlca = np.stack([np.asarray(h, dtype=np.float32) for h in row["hlca_features"]])
        luca = np.stack([np.asarray(l, dtype=np.float32) for l in row["luca_features"]])
        stage_hlca.setdefault(stage, []).append(hlca.mean(axis=0))
        stage_luca.setdefault(stage, []).append(luca.mean(axis=0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # HLCA
    stages_present = [s for s in STAGE_ORDER if s in stage_hlca]
    hlca_matrix = np.stack([np.mean(stage_hlca[s], axis=0) for s in stages_present])
    im1 = ax1.imshow(hlca_matrix, cmap="RdBu_r", aspect="auto")
    ax1.set_yticks(range(len(stages_present)))
    ax1.set_yticklabels(stages_present, fontsize=10)
    ax1.set_xlabel("HLCA Latent Dimension", fontsize=10)
    ax1.set_title("HLCA Reference Features by Stage", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.8, label="Mean Activation")

    # LuCA
    luca_matrix = np.stack([np.mean(stage_luca[s], axis=0) for s in stages_present])
    im2 = ax2.imshow(luca_matrix, cmap="RdBu_r", aspect="auto")
    ax2.set_yticks(range(len(stages_present)))
    ax2.set_yticklabels(stages_present, fontsize=10)
    ax2.set_xlabel("LuCA Latent Dimension", fontsize=10)
    ax2.set_title("LuCA Reference Features by Stage", fontsize=13, fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.8, label="Mean Activation")

    fig.suptitle("Reference Atlas Feature Profiles Across Progression Stages", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pathway_features_by_stage(out: Path) -> None:
    """Panel L: LR pathway feature profiles by stage."""
    if not BAG_PATH.exists():
        return

    bags_df = pd.read_parquet(BAG_PATH)
    stage_pathway: dict[str, list] = {}

    for _, row in bags_df.iterrows():
        stage = str(row["stage_label"])
        pw = np.stack([np.asarray(p, dtype=np.float32) for p in row["pathway_features"]])
        stage_pathway.setdefault(stage, []).append(pw.mean(axis=0))

    stages_present = [s for s in STAGE_ORDER if s in stage_pathway]
    pw_matrix = np.stack([np.mean(stage_pathway[s], axis=0) for s in stages_present])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pw_matrix, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(stages_present)))
    ax.set_yticklabels(stages_present, fontsize=10)
    ax.set_xlabel("LR Pathway Dimension", fontsize=10)
    ax.set_title("Ligand-Receptor Pathway Activity by Stage", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Score")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_lesion_size_by_stage(out: Path) -> None:
    """Supplemental: Lesion size (num neighborhoods) distribution by stage."""
    if not BAG_PATH.exists():
        return

    bags_df = pd.read_parquet(BAG_PATH)
    bags_df["num_niches"] = bags_df["niche_ids"].apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage in STAGE_ORDER:
        mask = bags_df["stage_label"] == stage
        vals = bags_df.loc[mask, "num_niches"].values
        if len(vals) > 0:
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(vals))
            stage_idx = STAGE_ORDER.index(stage)
            ax.scatter(stage_idx + jitter, vals, c=STAGE_COLORS[stage], s=60, edgecolors="white",
                      linewidths=0.5, alpha=0.8, label=f"{stage} (n={len(vals)})")
            ax.barh(stage_idx, np.mean(vals), height=0.02, color=STAGE_COLORS[stage], alpha=0.5)

    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels(STAGE_ORDER, fontsize=11)
    ax.set_ylabel("Number of Neighborhoods", fontsize=12)
    ax.set_title("Lesion Size Distribution by Stage", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_patient_stage_matrix(out: Path) -> None:
    """Supplemental: Patient x Stage presence matrix."""
    if not BAG_PATH.exists():
        return

    bags_df = pd.read_parquet(BAG_PATH)
    patient_stage = bags_df.groupby(["donor_id", "stage_label"]).size().unstack(fill_value=0)
    patient_stage = patient_stage.reindex(columns=[s for s in STAGE_ORDER if s in patient_stage.columns], fill_value=0)
    patient_stage = patient_stage.sort_index()

    fig, ax = plt.subplots(figsize=(8, max(6, len(patient_stage) * 0.35)))
    presence = (patient_stage > 0).astype(int)
    im = ax.imshow(presence.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(presence.shape[1]))
    ax.set_xticklabels(presence.columns, fontsize=10, rotation=0)
    ax.set_yticks(range(presence.shape[0]))
    ax.set_yticklabels(presence.index, fontsize=8)
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("Patient", fontsize=12)
    ax.set_title("Patient-Stage Presence Matrix", fontsize=14, fontweight="bold")

    for i in range(presence.shape[0]):
        for j in range(presence.shape[1]):
            count = int(patient_stage.values[i, j])
            if count > 0:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating advanced EA-MIST figures...")

    # Load data
    print("Loading training histories...")
    train_hist = _load_all_train_histories()
    val_hist = _load_all_val_histories()

    print("Loading metrics...")
    metrics_df = _load_all_metrics()

    print("Loading confusion matrices...")
    cms = _load_all_confusion_matrices()

    print("Loading test predictions...")
    preds_df = _load_all_test_predictions()

    print("Loading prototype data...")
    proto_df = _load_prototype_data()

    # Generate figures
    print("\nGenerating panels...")
    plot_loss_curves(train_hist, val_hist, OUT_DIR / "panel_A_loss_curves.png")
    plot_confusion_matrices(cms, OUT_DIR / "panel_B_confusion_matrices.png")
    plot_metric_violins(metrics_df, OUT_DIR / "panel_C_metric_violins.png")
    plot_embedding_manifolds(OUT_DIR / "panel_D_embedding_stage.png")
    plot_cell_type_embeddings(OUT_DIR / "panel_E_celltype_embedding.png")
    plot_prototype_heatmap(proto_df, OUT_DIR / "panel_F_prototype_heatmap.png")
    plot_displacement_scatter(preds_df, OUT_DIR / "panel_G_displacement_scatter.png")
    plot_stage_probability_distributions(preds_df, OUT_DIR / "panel_H_stage_probabilities.png")
    plot_cell_type_composition(OUT_DIR / "panel_I_celltype_composition.png")
    plot_ring_composition_heatmap(OUT_DIR / "panel_J_ring_heatmap.png")
    plot_reference_feature_heatmap(OUT_DIR / "panel_K_reference_features.png")
    plot_pathway_features_by_stage(OUT_DIR / "panel_L_pathway_features.png")
    plot_lesion_size_by_stage(OUT_DIR / "panel_M_lesion_sizes.png")
    plot_patient_stage_matrix(OUT_DIR / "panel_N_patient_stage_matrix.png")

    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
