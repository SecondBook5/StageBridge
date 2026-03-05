#!/usr/bin/env python3
"""Generate biologically interpretable figures from trained StageBridge checkpoints.

Produces:
  1. Real UMAP of HLCA latent space colored by cell type (not k-means clusters)
  2. Cell-type transition heatmaps: what does each HLCA label predict?
  3. Per-transition metric breakdown (Normal→AAH, AAH→AIS, AIS→MIA, MIA→LUAD)
  4. Stage progression trajectory figure with HLCA labels

Usage::

    python scripts/make_biology_figures.py [output_dir=outputs/biology_figures] [run_dir=outputs/hca_poster]
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── colour palettes ────────────────────────────────────────────────────────────
_STAGE_COLORS = {
    "Normal": "#4ADE80", "AAH": "#FACC15", "AIS": "#FB923C",
    "MIA": "#F87171",    "LUAD": "#7F1D1D",
}
_CELTYPE_COLORS = {
    "AT2":                "#2563EB",   # deep blue
    "AT1":                "#93C5FD",   # light blue
    "Basal":              "#DC2626",   # red
    "Secretory":          "#F97316",   # orange
    "Ciliated":           "#16A34A",   # green
    "KRT5- KRT17+ epithelial": "#7C3AED",  # purple
    "Fibroblast lineage": "#D1D5DB",   # light grey
    "T cell lineage":     "#6B7280",   # grey
    "Macrophages":        "#9CA3AF",   # medium grey
    "Capillary":          "#D97706",   # amber
    "Mast cells":         "#BE185D",   # pink
    "unlabeled":          "#E5E7EB",   # very light grey
}
_EPITHELIAL = frozenset(["AT2", "AT1", "Basal", "Secretory", "Ciliated", "KRT5- KRT17+ epithelial"])
_ALL_STAGES = ("Normal", "AAH", "AIS", "MIA", "LUAD")
_TRANSITIONS = [("Normal", "AAH"), ("AAH", "AIS"), ("AIS", "MIA"), ("MIA", "LUAD")]

# ── load data ─────────────────────────────────────────────────────────────────

def _load_adata(data_root: Path):
    """Load snrna_hlca_latent_full.h5ad and attach X as obsm['X_hlca']."""
    import anndata as ad
    path = data_root / "processed" / "anndata" / "snrna_hlca_latent_full.h5ad"
    log.info("Loading adata from %s …", path)
    adata = ad.read_h5ad(path)
    if "X_hlca" not in adata.obsm:
        adata.obsm["X_hlca"] = np.asarray(adata.X, dtype=np.float32)
    return adata


def _load_model(checkpoint_path: Path, adata, device: str = "cpu"):
    """Reconstruct model from checkpoint."""
    from stagebridge.models.stagebridge import StageBridgeModel
    from stagebridge.utils.types import StageBridgeConfig
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    cfg = StageBridgeConfig(**{k: v for k, v in cfg_dict.items()
                                if k in StageBridgeConfig.__dataclass_fields__})
    model = StageBridgeModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model


# ── UMAP ──────────────────────────────────────────────────────────────────────

def _compute_umap(latent: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.3,
                  random_state: int = 42) -> tuple:
    """Returns (coords, reducer_or_None) so caller can call reducer.transform()."""
    try:
        import umap as umap_lib
        log.info("Running UMAP on %d × %d …", *latent.shape)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                 min_dist=min_dist, random_state=random_state)
        coords = np.asarray(reducer.fit_transform(latent), dtype=np.float32)
        return coords, reducer
    except ImportError:
        log.warning("umap-learn not installed; falling back to PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = np.asarray(pca.fit_transform(latent), dtype=np.float32)
        return coords, pca  # PCA also has .transform()


def make_celtype_umap(adata, out_dir: Path, n_cells_per_stage: int = 3_000) -> None:
    """Plot UMAP colored by HLCA cell type with stage facets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(adata.obsm["X_hlca"], dtype=np.float32)
    stages = np.asarray(adata.obs["stage"].astype(str))
    labels = np.asarray(adata.obs["hlca_label"].astype(str))

    # Subsample for UMAP speed
    rng = np.random.default_rng(42)
    idx_list = []
    for stage in _ALL_STAGES:
        stage_idx = np.where(stages == stage)[0]
        n = min(n_cells_per_stage, len(stage_idx))
        idx_list.append(rng.choice(stage_idx, size=n, replace=False))
    idx = np.concatenate(idx_list)

    sub_lat = lat[idx]
    sub_labels = labels[idx]
    sub_stages = stages[idx]

    umap_coords, _ = _compute_umap(sub_lat)

    # Figure 1: colored by cell type, faceted by stage
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), constrained_layout=True)
    fig.suptitle("UMAP of HLCA Latent Space by Cell Type and Stage", fontsize=13, fontweight="bold")

    unique_labels = sorted({l for l in sub_labels if l not in ("unlabeled", "")},
                            key=lambda x: -np.sum(sub_labels == x))

    for ax, stage in zip(axes, _ALL_STAGES):
        # Background: all cells, grey
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c="#E5E7EB", s=1, alpha=0.2, rasterized=True)
        # Foreground: cells from this stage, colored by cell type
        mask = sub_stages == stage
        for ct in unique_labels:
            ct_mask = mask & (sub_labels == ct)
            if not ct_mask.any():
                continue
            color = _CELTYPE_COLORS.get(ct, "#6B7280")
            ax.scatter(umap_coords[ct_mask, 0], umap_coords[ct_mask, 1],
                       c=color, s=4, alpha=0.7, rasterized=True)
        ax.set_title(stage, fontsize=11, color=_STAGE_COLORS.get(stage, "#000"))
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Legend
    handles = [mpatches.Patch(color=_CELTYPE_COLORS.get(ct, "#6B7280"), label=ct)
               for ct in unique_labels if ct in _CELTYPE_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.12), framealpha=0.9)

    fig.savefig(out_dir / "celtype_umap_by_stage.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "celtype_umap_by_stage.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved celtype_umap_by_stage")

    # Store UMAP coords and idx for reuse
    return umap_coords, idx, sub_labels, sub_stages


# ── cell-type transition heatmap ───────────────────────────────────────────────

def _assign_labels_by_nn(pred_latent: np.ndarray, ref_latent: np.ndarray,
                          ref_labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Assign cell-type label to each predicted cell via k-NN majority vote in latent space."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
    nn.fit(ref_latent)
    _, nbrs = nn.kneighbors(pred_latent)
    assigned = []
    for row in nbrs:
        votes = ref_labels[row]
        unique, counts = np.unique(votes, return_counts=True)
        assigned.append(unique[np.argmax(counts)])
    return np.array(assigned)


def make_celtype_transition_heatmap(
    model,
    adata,
    out_dir: Path,
    n_src: int = 500,
    device: str = "cpu",
) -> None:
    """For each stage transition: heatmap of source HLCA label → predicted HLCA label."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(adata.obsm["X_hlca"], dtype=np.float32)
    stages = np.asarray(adata.obs["stage"].astype(str))
    labels = np.asarray(adata.obs["hlca_label"].astype(str))

    from stagebridge.preprocessing.stage_ontology import stage_to_index

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    fig.suptitle("Predicted Cell-Type Transition Heatmaps (HLCA Labels)", fontsize=13, fontweight="bold")

    for ax, (src_stage, tgt_stage) in zip(axes, _TRANSITIONS):
        src_idx = np.where(stages == src_stage)[0]
        tgt_idx = np.where(stages == tgt_stage)[0]
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            ax.set_title(f"{src_stage}→{tgt_stage}\n(no data)")
            continue

        rng = np.random.default_rng(0)
        chosen = rng.choice(src_idx, size=min(n_src, len(src_idx)), replace=False)
        x_src_np = lat[chosen]
        src_labels_sample = labels[chosen]

        x_src_t = torch.tensor(x_src_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            c_s = model.forward_set_context(x_src_t)
            sp = model.encode_stage_pair_tensor(
                stage_to_index(src_stage), stage_to_index(tgt_stage),
                n=len(x_src_t), device=device,
            )
            x_pred_t = model.integrate_euler(x_src_t, c_s, sp, num_steps=8)
        x_pred_np = x_pred_t.cpu().numpy()

        # Assign predicted labels via NN to actual target cells
        tgt_lat = lat[tgt_idx]
        tgt_labels = labels[tgt_idx]
        pred_labels = _assign_labels_by_nn(x_pred_np, tgt_lat, tgt_labels, k=5)

        # Build transition matrix over epithelial cell types
        ep_types = sorted(_EPITHELIAL & set(np.unique(src_labels_sample)) & set(np.unique(pred_labels)))
        if len(ep_types) < 2:
            ep_types = sorted(set(np.unique(src_labels_sample)) & set(np.unique(pred_labels)))
            ep_types = [t for t in ep_types if t not in ("unlabeled",)][:8]

        if not ep_types:
            ax.set_title(f"{src_stage}→{tgt_stage}\n(no cell types)")
            continue

        mat = np.zeros((len(ep_types), len(ep_types)), dtype=np.float32)
        for i, s_ct in enumerate(ep_types):
            mask = src_labels_sample == s_ct
            if not mask.any():
                continue
            for j, p_ct in enumerate(ep_types):
                mat[i, j] = np.sum(pred_labels[mask] == p_ct)
            row_sum = mat[i].sum()
            if row_sum > 0:
                mat[i] /= row_sum

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ep_types)))
        ax.set_yticks(range(len(ep_types)))
        short = [t.replace(" lineage", "").replace("KRT5- KRT17+ ", "").replace(" epithelial", "")[:10]
                 for t in ep_types]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted cell type", fontsize=9)
        ax.set_ylabel("Source cell type", fontsize=9)
        ax.set_title(f"{src_stage} → {tgt_stage}", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")

    fig.savefig(out_dir / "celtype_transition_heatmap.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "celtype_transition_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved celtype_transition_heatmap")


# ── per-transition Sinkhorn comparison ────────────────────────────────────────

def make_per_transition_metrics(run_dir: Path, out_dir: Path, adata, device: str = "cpu") -> None:
    """Re-evaluate all saved checkpoints per transition and plot comparison bar chart."""
    from stagebridge.training.losses import sinkhorn_distance
    from stagebridge.preprocessing.stage_ontology import stage_to_index

    import json
    metrics_path = run_dir / "tables" / "metrics_hca_poster.json"
    if not metrics_path.exists():
        log.warning("metrics json not found: %s", metrics_path)
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    lat = np.asarray(adata.obsm["X_hlca"], dtype=np.float32)
    stages = np.asarray(adata.obs["stage"].astype(str))

    # For each transition, compute per-model Sinkhorn on a fixed subsample
    rng = np.random.default_rng(0)
    n_eval = 256
    src_tgt_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for src_stage, tgt_stage in _TRANSITIONS:
        src_idx = np.where(stages == src_stage)[0]
        tgt_idx = np.where(stages == tgt_stage)[0]
        if len(src_idx) < 20 or len(tgt_idx) < 20:
            continue
        si = rng.choice(src_idx, size=min(n_eval, len(src_idx)), replace=False)
        ti = rng.choice(tgt_idx, size=min(n_eval, len(tgt_idx)), replace=False)
        src_tgt_data[f"{src_stage}→{tgt_stage}"] = (lat[si], lat[ti])

    model_names = [k for k in metrics["results"].keys()
                   if "ablation" not in (metrics["results"][k].get("ablation") or "")]
    model_results: dict[str, dict[str, float]] = {m: {} for m in model_names}

    for model_name in model_names:
        ckpts = metrics["results"][model_name].get("checkpoints", [])
        if not ckpts:
            continue
        ckpt_path = Path(ckpts[0])  # use fold 0
        if not ckpt_path.exists():
            continue
        try:
            model = _load_model(ckpt_path, adata, device=device)
        except Exception as e:
            log.warning("Could not load %s: %s", ckpt_path, e)
            continue

        for transition_key, (x_src_np, x_tgt_np) in src_tgt_data.items():
            src_stage, tgt_stage = transition_key.split("→")
            x_src_t = torch.tensor(x_src_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                c_s = model.forward_set_context(x_src_t)
                sp = model.encode_stage_pair_tensor(
                    stage_to_index(src_stage), stage_to_index(tgt_stage),
                    n=len(x_src_t), device=device,
                )
                x_pred_t = model.integrate_euler(x_src_t, c_s, sp, num_steps=8)
            x_pred_np = x_pred_t.cpu().numpy()

            x_pred_t2 = torch.tensor(x_pred_np, dtype=torch.float32)
            x_tgt_t = torch.tensor(x_tgt_np, dtype=torch.float32)
            sink = float(sinkhorn_distance(x_pred_t2, x_tgt_t, epsilon=0.05, n_iters=50).item())
            model_results[model_name][transition_key] = sink
        log.info("Evaluated %s", model_name)

    # Plot grouped bar chart
    transitions_with_data = [k for k in src_tgt_data if any(k in v for v in model_results.values())]
    if not transitions_with_data:
        log.warning("No per-transition data to plot")
        return

    model_names_with_data = [m for m in model_names if model_results[m]]
    n_models = len(model_names_with_data)
    n_trans = len(transitions_with_data)
    x = np.arange(n_trans)
    width = 0.8 / max(n_models, 1)

    _model_colors = {
        "stagebridge": "#2563EB",
        "stagebridge_sb": "#3B82F6",
        "stagebridge_xattn_sb": "#1D4ED8",
        "deepsets": "#F97316",
        "no_context": "#6B7280",
        "linear": "#D1D5DB",
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(model_names_with_data):
        vals = [model_results[m].get(t, np.nan) for t in transitions_with_data]
        color = _model_colors.get(m, "#94A3B8")
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=m, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(transitions_with_data, fontsize=11)
    ax.set_ylabel("Sinkhorn Distance (lower = better)", fontsize=11)
    ax.set_title("Per-Transition Performance: Sinkhorn Distance by Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "per_transition_sinkhorn.png", dpi=180)
    fig.savefig(out_dir / "per_transition_sinkhorn.pdf")
    plt.close(fig)
    log.info("Saved per_transition_sinkhorn")


# ── AT2 progression ribbon ─────────────────────────────────────────────────────

def make_at2_stage_progression(model, adata, out_dir: Path, device: str = "cpu") -> None:
    """Show the AT2 cell predicted trajectory across all 5 stages as a violin/scatter."""
    from stagebridge.preprocessing.stage_ontology import stage_to_index

    out_dir.mkdir(parents=True, exist_ok=True)
    lat = np.asarray(adata.obsm["X_hlca"], dtype=np.float32)
    stages = np.asarray(adata.obs["stage"].astype(str))
    labels = np.asarray(adata.obs["hlca_label"].astype(str))

    # Compute UMAP on AT2 cells only (fast — ~100k cells → subsample 5k)
    at2_mask = labels == "AT2"
    rng = np.random.default_rng(0)
    n_sub = 2000
    stage_subsets: dict[str, np.ndarray] = {}
    for stage in _ALL_STAGES:
        idx = np.where(at2_mask & (stages == stage))[0]
        if len(idx) == 0:
            continue
        stage_subsets[stage] = rng.choice(idx, size=min(n_sub, len(idx)), replace=False)

    all_idx = np.concatenate(list(stage_subsets.values()))
    all_lat = lat[all_idx]
    all_stage = stages[all_idx]

    umap_coords, reducer = _compute_umap(all_lat, n_neighbors=20, min_dist=0.1)

    # Build a lookup from global index → position in umap_coords
    idx_to_umap = {gi: umap_coords[li] for li, gi in enumerate(all_idx)}

    # Run model: Normal AT2 → predicted AAH AT2, etc.
    # predicted_lat_map: key → (src_latent, pred_latent) arrays
    predicted_lat_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for src_stage, tgt_stage in _TRANSITIONS:
        if src_stage not in stage_subsets:
            continue
        src_sub = stage_subsets[src_stage]
        x_src_np = lat[src_sub]
        x_src_t = torch.tensor(x_src_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            c_s = model.forward_set_context(x_src_t)
            sp = model.encode_stage_pair_tensor(
                stage_to_index(src_stage), stage_to_index(tgt_stage),
                n=len(x_src_t), device=device,
            )
            x_pred_t = model.integrate_euler(x_src_t, c_s, sp, num_steps=8)
        predicted_lat_map[f"{src_stage}→{tgt_stage}"] = (x_src_np, x_pred_t.cpu().numpy())

    arrow_colors = {"Normal→AAH": "#FACC15", "AAH→AIS": "#FB923C",
                    "AIS→MIA": "#F87171", "MIA→LUAD": "#7F1D1D"}

    # Project predicted latent vectors into UMAP space using reducer.transform()
    predicted_umap: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trans_key, (src_lat_sub, pred_lat_sub) in predicted_lat_map.items():
        src_stage = trans_key.split("→")[0]
        # Source cells are in all_lat, so map directly
        src_global = stage_subsets[src_stage]
        src_uv = np.stack([idx_to_umap[gi] for gi in src_global])
        # Predicted cells are new; use UMAP transform to project them
        try:
            pred_uv = np.asarray(reducer.transform(pred_lat_sub), dtype=np.float32)
        except Exception:
            pred_uv = src_uv  # fallback: no displacement if transform fails
        predicted_umap[trans_key] = (src_uv, pred_uv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("AT2 Cell Trajectory: Real Stages vs Predicted Transitions", fontsize=13, fontweight="bold")

    # Left: real AT2 cells colored by stage
    ax = axes[0]
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c="#E5E7EB", s=2, alpha=0.3, rasterized=True)
    for stage in _ALL_STAGES:
        if stage not in stage_subsets:
            continue
        s_idx = np.where(all_stage == stage)[0]
        ax.scatter(umap_coords[s_idx, 0], umap_coords[s_idx, 1],
                   c=_STAGE_COLORS[stage], s=6, alpha=0.7, label=stage, rasterized=True)
    ax.set_title("Real AT2 cells by stage", fontsize=11)
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Right: predicted transitions — per-cell scatter of predicted positions
    # plus thick centroid arrow showing mean displacement per transition
    ax = axes[1]
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c="#E5E7EB", s=2, alpha=0.15, rasterized=True)

    for trans_key, (src_uv, pred_uv) in predicted_umap.items():
        color = arrow_colors.get(trans_key, "#1D4ED8")

        # Scatter predicted cell positions (small dots)
        ax.scatter(pred_uv[:, 0], pred_uv[:, 1], c=color, s=4, alpha=0.4,
                   rasterized=True, zorder=2)

        # Centroid arrow: thick, clearly visible
        src_c = src_uv.mean(axis=0)
        pred_c = pred_uv.mean(axis=0)
        dx, dy = pred_c[0] - src_c[0], pred_c[1] - src_c[1]
        ax.annotate("", xy=(pred_c[0], pred_c[1]), xytext=(src_c[0], src_c[1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                   mutation_scale=20, shrinkA=0, shrinkB=0),
                    zorder=5)
        # Mark source centroid
        ax.scatter(*src_c, c=color, s=80, marker="o", edgecolors="white",
                   linewidths=0.8, zorder=6)

    # Legend for transitions
    handles = [mpatches.Patch(color=c, label=k) for k, c in arrow_colors.items()]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.set_title("Predicted transitions: dots=predicted, arrows=centroid shift", fontsize=10)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    fig.savefig(out_dir / "at2_stage_progression.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "at2_stage_progression.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved at2_stage_progression")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/biology_figures", help="Output directory")
    parser.add_argument("--run-dir", default="outputs/hca_poster", help="Training run directory")
    parser.add_argument("--data-root", default=None, help="Data root (falls back to env/local.yaml)")
    parser.add_argument("--model", default="stagebridge_xattn_sb", help="Which model variant to use for biology figures")
    parser.add_argument("--fold", default=0, type=int, help="Which fold checkpoint to use")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP computation (slow)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    run_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        import os
        env_root = os.environ.get("STAGEBRIDGE_DATA_ROOT")
        if env_root:
            data_root = Path(env_root)
        else:
            local_cfg = _REPO / "configs" / "data" / "local.yaml"
            import yaml
            with open(local_cfg) as fh:
                cfg = yaml.safe_load(fh)
            data_root = Path(cfg.get("data_root") or cfg["paths"]["data_root"])

    adata = _load_adata(data_root)

    # Find checkpoint for requested model
    import json
    metrics_path = run_dir / "tables" / "metrics_hca_poster.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Try requested model, fall back to stagebridge
    model_key = args.model
    if model_key not in metrics["results"]:
        log.warning("Model %s not found in results; using stagebridge", model_key)
        model_key = "stagebridge"

    ckpts = metrics["results"][model_key].get("checkpoints", [])
    if not ckpts or args.fold >= len(ckpts):
        log.error("Checkpoint for %s fold %d not found", model_key, args.fold)
        sys.exit(1)
    ckpt_path = Path(ckpts[args.fold])
    log.info("Loading checkpoint: %s", ckpt_path)
    model = _load_model(ckpt_path, adata, device=args.device)

    # 1. Cell-type UMAP
    if not args.skip_umap:
        log.info("Step 1: Cell-type UMAP …")
        make_celtype_umap(adata, out_dir / "umap")
    else:
        log.info("Skipping UMAP (--skip-umap)")

    # 2. Cell-type transition heatmap
    log.info("Step 2: Cell-type transition heatmap …")
    make_celtype_transition_heatmap(model, adata, out_dir / "transitions", device=args.device)

    # 3. Per-transition Sinkhorn comparison across models
    log.info("Step 3: Per-transition Sinkhorn comparison …")
    make_per_transition_metrics(run_dir, out_dir / "per_transition", adata, device=args.device)

    # 4. AT2 stage progression
    log.info("Step 4: AT2 stage progression …")
    make_at2_stage_progression(model, adata, out_dir / "at2", device=args.device)

    log.info("Biology figures saved to %s", out_dir)
    print(f"\nDone. Figures in: {out_dir}")
    for p in sorted(out_dir.rglob("*.png")):
        print(f"  {p.relative_to(out_dir)}")


if __name__ == "__main__":
    main()
