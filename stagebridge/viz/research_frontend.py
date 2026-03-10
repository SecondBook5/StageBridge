"""Notebook-facing research frontend visualizations for StageBridge."""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from stagebridge.data.luad_evo.stages import CANONICAL_STAGE_ORDER
from stagebridge.evaluation.metrics import rollout_edge_transition
from stagebridge.viz.flows import compute_macroflow_matrix

PALETTE = {
    "ink": "#16202A",
    "muted": "#6B7280",
    "grid": "#D1D9E0",
    "panel": "#F7F3EA",
    "accent": "#A63A2B",
    "teal": "#0F766E",
    "gold": "#D18A00",
    "blue": "#245C73",
    "slate": "#374151",
    "signal": "#C2410C",
    "Normal": "#5CA462",
    "AAH": "#D9A441",
    "AIS": "#D07A3F",
    "MIA": "#B5564B",
    "LUAD": "#6D2E2A",
    "Unknown": "#9CA3AF",
}

MODE_COLORS = {
    "rna_only": "#64748B",
    "pooled": "#B45309",
    "deep_sets": "#A855F7",
    "set_only": "#0F766E",
    "typed_hierarchical_transformer": "#0B5FFF",
    "deep_sets_transformer_hybrid": "#D97706",
    "graph_of_sets": "#7C3AED",
}


def configure_research_style() -> None:
    """Apply a light, print-friendly notebook style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FBF8F1",
            "axes.facecolor": "#FBF8F1",
            "savefig.facecolor": "#FBF8F1",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.35,
            "axes.grid": False,
        }
    )


def _truncate_path(path: str, max_len: int = 50) -> str:
    """Show .../<last few path components> if too long."""
    if len(path) <= max_len:
        return path
    from pathlib import PurePosixPath
    parts = PurePosixPath(path).parts
    truncated = str(PurePosixPath(*parts[-3:])) if len(parts) >= 3 else path
    return f".../{truncated}" if len(truncated) < len(path) else path[-max_len:]


def _stage_palette(stage_name: str) -> str:
    return PALETTE.get(str(stage_name), PALETTE["Unknown"])


def _sorted_stages(values: list[str] | np.ndarray) -> list[str]:
    seen = [str(value) for value in values]
    return [stage for stage in CANONICAL_STAGE_ORDER if stage in seen] + sorted(
        {stage for stage in seen if stage not in CANONICAL_STAGE_ORDER}
    )


def _pca2(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[1] < 2:
        padded = np.zeros((arr.shape[0], 2), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        return padded
    return PCA(n_components=2, random_state=42).fit_transform(arr).astype(np.float32)


def _pca2_with_variance(array: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """PCA with explained variance ratios for axis labels."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[1] < 2:
        padded = np.zeros((arr.shape[0], 2), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        return padded, (100.0, 0.0)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(arr).astype(np.float32)
    var1 = float(pca.explained_variance_ratio_[0] * 100)
    var2 = float(pca.explained_variance_ratio_[1] * 100)
    return coords, (var1, var2)


def _tsne2(array: np.ndarray) -> np.ndarray:
    """t-SNE to 2D, falls back to PCA."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[0] < 5:
        return _pca2(arr)
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30.0, max(2.0, float(arr.shape[0] - 1) / 3.0))
        return TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(arr).astype(np.float32)
    except Exception:
        return _pca2(arr)


def _phate2(array: np.ndarray) -> np.ndarray:
    """PHATE to 2D, falls back to UMAP then PCA."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[0] < 5:
        return _pca2(arr)
    try:
        import phate
        return np.asarray(
            phate.PHATE(n_components=2, random_state=42, n_jobs=1, verbose=0).fit_transform(arr),
            dtype=np.float32,
        )
    except Exception:
        return _umap2(arr)


def _umap2(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={arr.shape}.")
    if arr.shape[0] < 3:
        return _pca2(arr)
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(20, max(2, arr.shape[0] - 1)),
            min_dist=0.15,
            random_state=42,
        )
        return reducer.fit_transform(arr).astype(np.float32)
    except Exception:
        return _pca2(arr)


def _soft_entropy(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    row_sums = arr.sum(axis=1, keepdims=True)
    probs = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)
    probs = np.clip(probs, 1e-8, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def _centroid_distance_matrix(centroid_distances: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    stages: set[str] = set()
    for edge_name in centroid_distances:
        src, tgt = str(edge_name).split("->", 1)
        stages.add(src)
        stages.add(tgt)
    ordered = _sorted_stages(list(stages))
    matrix = np.zeros((len(ordered), len(ordered)), dtype=np.float32)
    lookup = {stage: idx for idx, stage in enumerate(ordered)}
    for edge_name, value in centroid_distances.items():
        src, tgt = str(edge_name).split("->", 1)
        i, j = lookup[src], lookup[tgt]
        matrix[i, j] = float(value)
        matrix[j, i] = float(value)
    return matrix, ordered


def _plot_dense_heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str = "YlOrBr",
    annotate: bool = True,
) -> Any:
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=PALETTE["ink"])
    return im


def _provider_matrix_and_columns(payload: dict[str, Any]) -> tuple[np.ndarray | None, list[str]]:
    mapping = payload.get("mapping_result")
    if mapping is None or mapping.compositions is None:
        return None, []
    matrix = np.asarray(mapping.compositions, dtype=np.float32)
    columns = [str(value) for value in getattr(mapping, "feature_names", ()) or ()]
    return matrix, columns


def plot_reference_frontend(reference_output: dict[str, Any]) -> Figure:
    """Render the active reference branch as an audited alignment/QC figure."""
    configure_research_style()
    cohort = reference_output["cohort"]
    reference = reference_output["reference"]
    diagnostics = reference["diagnostics"]
    label_transfer = reference["label_transfer"]
    latent = np.asarray(cohort.latent, dtype=np.float32)
    coords_pca, pca_var = _pca2_with_variance(latent)
    coords_umap = _umap2(latent)
    stages = cohort.obs["stage"].astype(str).to_numpy()
    labels = cohort.obs.get("hlca_label", pd.Series(["unlabeled"] * len(stages))).astype(str).to_numpy()
    alignment = diagnostics.get("stage_label_alignment", {})
    gate = diagnostics.get("alignment_gate", {})
    gene_overlap = diagnostics.get("gene_overlap", {})
    label_neighborhood = diagnostics.get("label_neighborhood", {})
    stage_probe = diagnostics["stage_preservation"].get("probe", {})
    donor = diagnostics["donor_leakage"]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.88], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.30)
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_umap = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_confusion = fig.add_subplot(gs[1, 0:2])
    ax_centroids = fig.add_subplot(gs[1, 2])

    for stage in _sorted_stages(stages):
        mask = stages == stage
        ax_pca.scatter(
            coords_pca[mask, 0],
            coords_pca[mask, 1],
            s=12,
            alpha=0.72,
            color=_stage_palette(stage),
            label=stage,
            linewidths=0.0,
            rasterized=True,
        )
    ax_pca.set_title("Reference latent: PCA by stage")
    ax_pca.set_xlabel(f"PC 1 ({pca_var[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC 2 ({pca_var[1]:.1f}%)")
    ax_pca.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    ax_pca.grid(True, alpha=0.22)

    unique_labels = pd.Series(labels).value_counts().head(8).index.tolist()
    cmap = mpl.colormaps["tab10"]
    label_palette = {label: cmap(idx % 10) for idx, label in enumerate(unique_labels)}
    for label in unique_labels:
        mask = labels == label
        ax_umap.scatter(
            coords_umap[mask, 0],
            coords_umap[mask, 1],
            s=12,
            alpha=0.70,
            color=label_palette[label],
            label=label,
            linewidths=0.0,
            rasterized=True,
        )
    ax_umap.set_title("UMAP by transferred HLCA label")
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    ax_umap.grid(True, alpha=0.18)

    ax_metrics.axis("off")
    coverage = float(label_transfer.get("coverage", 0.0))
    donor_acc = float(donor.get("logreg_accuracy", float("nan")))
    donor_chance = float(donor.get("chance_accuracy", float("nan")))
    ax_metrics.text(
        0.03,
        0.97,
        "HLCA alignment gate",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    ax_metrics.text(
        0.03,
        0.88,
        "\n".join(
            [
                f"status: {gate.get('status', 'n/a')}",
                f"action: {gate.get('recommended_action', 'n/a')}",
                f"source: {_truncate_path(str(reference['source_path']))}",
                f"latent shape: {tuple(reference['latent_shape'])}",
                f"stage probe acc: {float(stage_probe.get('logreg_accuracy', float('nan'))):.3f}",
                f"stage balanced acc: {float(stage_probe.get('balanced_accuracy', float('nan'))):.3f}",
                f"stage chance: {float(stage_probe.get('chance_accuracy', float('nan'))):.3f}",
                f"donor leakage: {donor_acc:.3f}",
                f"donor chance: {donor_chance:.3f}",
                f"label coverage: {coverage:.3f}",
                f"gene overlap: {float(gene_overlap.get('reference_query_overlap_fraction', float('nan'))):.3f}",
                f"missing genes: {float(gene_overlap.get('missing_gene_fraction', float('nan'))):.3f}",
                f"NN label agreement: {float(label_neighborhood.get('mean_neighbor_label_agreement', float('nan'))):.3f}",
            ]
        ),
        transform=ax_metrics.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7E8", "edgecolor": "#D6C7A1", "alpha": 0.85},
    )
    ax_metrics.text(
        0.03,
        0.18,
        gate.get("interpretation", "No interpretation available."),
        transform=ax_metrics.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=PALETTE["accent"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF0EB", "edgecolor": "#E2B8AA", "alpha": 0.85},
    )

    matrix, ordered_stages = _centroid_distance_matrix(diagnostics["stage_preservation"]["centroid_distances"])
    im = _plot_dense_heatmap(
        ax_centroids,
        matrix,
        xlabels=ordered_stages,
        ylabels=ordered_stages,
        title="Stage centroid separation",
        cmap="YlOrBr",
        annotate=True,
    )
    fig.colorbar(im, ax=ax_centroids, fraction=0.046, pad=0.04)

    confusion = np.asarray(alignment.get("normalized_matrix", []), dtype=np.float32)
    if confusion.size == 0:
        ax_confusion.axis("off")
        ax_confusion.text(0.5, 0.5, "No stage-to-HLCA alignment matrix available", ha="center", va="center", fontsize=12)
    else:
        conf_im = _plot_dense_heatmap(
            ax_confusion,
            confusion,
            xlabels=[str(label) for label in alignment.get("cols", [])],
            ylabels=[str(stage) for stage in alignment.get("rows", [])],
            title="Stage-to-HLCA alignment (row-normalized confusion)",
            cmap="GnBu",
            annotate=True,
        )
        fig.colorbar(conf_im, ax=ax_confusion, fraction=0.022, pad=0.02)

    fig.suptitle("StageBridge v1 research frontend: HLCA reference mapping and alignment gate", fontsize=17, fontweight="bold", x=0.46)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_snrna_preprocessing_frontend(data_output: dict[str, Any]) -> Figure:
    """Render a notebook-facing snRNA cohort preview from the preprocessing step."""
    configure_research_style()
    snrna = data_output["snrna"]
    obs = snrna["obs"]
    raw_pca = np.asarray(snrna.get("pca_embedding"), dtype=np.float32)
    umap_embedding = np.asarray(snrna.get("umap_embedding"), dtype=np.float32)
    stages = obs["stage"].astype(str).to_numpy()
    ordered = _sorted_stages(stages)
    labels = (
        obs["hlca_label"].astype(str).to_numpy()
        if "hlca_label" in obs.columns
        else np.asarray(["unlabeled"] * obs.shape[0], dtype=object)
    )

    pca_embedding, pca_var = _pca2_with_variance(raw_pca if raw_pca.shape[1] > 2 else raw_pca)
    if raw_pca.shape[1] <= 2:
        pca_embedding = raw_pca
    tsne_embedding = _tsne2(raw_pca)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 0.85], wspace=0.26, hspace=0.30)
    ax_pca = fig.add_subplot(gs[0, 0])
    ax_umap = fig.add_subplot(gs[0, 1])
    ax_tsne = fig.add_subplot(gs[0, 2])
    ax_stage_bar = fig.add_subplot(gs[1, 0])
    ax_table_panel = fig.add_subplot(gs[1, 1:])

    for stage in ordered:
        mask = stages == stage
        ax_pca.scatter(
            pca_embedding[mask, 0], pca_embedding[mask, 1],
            s=12, alpha=0.70, color=_stage_palette(stage), label=stage,
            linewidths=0.0, rasterized=True,
        )
    ax_pca.set_title("PCA by stage")
    ax_pca.set_xlabel(f"PC 1 ({pca_var[0]:.1f}%)")
    ax_pca.set_ylabel(f"PC 2 ({pca_var[1]:.1f}%)")
    ax_pca.legend(frameon=False, fontsize=8, loc="best", ncol=2)

    top_labels = pd.Series(labels).value_counts().head(8).index.tolist()
    cmap = mpl.colormaps["tab10"]
    label_palette = {label: cmap(idx % 10) for idx, label in enumerate(top_labels)}
    for label in top_labels:
        mask = labels == label
        ax_umap.scatter(
            umap_embedding[mask, 0], umap_embedding[mask, 1],
            s=12, alpha=0.68, color=label_palette[label], label=label,
            linewidths=0.0, rasterized=True,
        )
    ax_umap.set_title("UMAP by HLCA label")
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.legend(frameon=False, fontsize=7, loc="best")

    for stage in ordered:
        mask = stages == stage
        ax_tsne.scatter(
            tsne_embedding[mask, 0], tsne_embedding[mask, 1],
            s=12, alpha=0.70, color=_stage_palette(stage), label=stage,
            linewidths=0.0, rasterized=True,
        )
    ax_tsne.set_title("t-SNE by stage")
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    ax_tsne.legend(frameon=False, fontsize=8, loc="best", ncol=2)

    stage_counts = pd.Series(snrna.get("stage_counts", {})).reindex(ordered).fillna(0.0)
    ax_stage_bar.bar(
        stage_counts.index.astype(str),
        stage_counts.to_numpy(dtype=np.float32),
        color=[_stage_palette(s) for s in stage_counts.index], alpha=0.9,
    )
    for i, (s, v) in enumerate(stage_counts.items()):
        ax_stage_bar.text(i, float(v) + stage_counts.max() * 0.02, f"{int(v)}", ha="center", fontsize=8)
    ax_stage_bar.set_title("Cells per stage")
    ax_stage_bar.set_ylabel("count")
    ax_stage_bar.tick_params(axis="x", rotation=20)

    sample_stage_counts = snrna.get("sample_stage_counts")
    if isinstance(sample_stage_counts, pd.DataFrame) and not sample_stage_counts.empty:
        preview = sample_stage_counts.fillna(0.0)
        preview = preview.loc[:, [stage for stage in ordered if stage in preview.columns]]
        preview = preview.iloc[: min(12, preview.shape[0])]
        table_im = ax_table_panel.imshow(preview.to_numpy(dtype=np.float32), cmap="YlGnBu", aspect="auto")
        ax_table_panel.set_title("Sample-by-stage cell counts")
        ax_table_panel.set_xticks(np.arange(preview.shape[1]))
        ax_table_panel.set_xticklabels(preview.columns.astype(str), rotation=25, ha="right", fontsize=8)
        ax_table_panel.set_yticks(np.arange(preview.shape[0]))
        ax_table_panel.set_yticklabels(preview.index.astype(str), fontsize=7)
        for i in range(preview.shape[0]):
            for j in range(preview.shape[1]):
                ax_table_panel.text(j, i, f"{int(preview.iloc[i, j])}", ha="center", va="center", fontsize=7, color=PALETTE["ink"])
        fig.colorbar(table_im, ax=ax_table_panel, fraction=0.046, pad=0.03)
    else:
        ax_table_panel.axis("off")
        ax_table_panel.text(0.5, 0.5, "No sample/stage summary available", ha="center", va="center", fontsize=12)

    fig.suptitle(
        f"snRNA-seq cohort  |  {snrna['n_cells']:,} cells  {snrna['n_genes']:,} genes  {snrna['n_donors']} donors",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_spatial_preprocessing_frontend(data_output: dict[str, Any]) -> Figure:
    """Render the raw Visium cohort preview from the preprocessing step."""
    configure_research_style()
    spatial = data_output["spatial"]
    obs = spatial["obs"]
    coords = np.asarray(spatial["coords"], dtype=np.float32)
    stages = obs["stage"].astype(str).to_numpy()
    ordered = _sorted_stages(stages)
    panel = spatial.get("feature_panel", pd.DataFrame())
    feature_genes = list(spatial.get("feature_panel_genes", []))
    roles = dict(spatial.get("feature_panel_roles", {}))
    proxy_used = bool(spatial.get("feature_panel_uses_proxy_genes", False))

    # Determine per-sample breakdown for tissue section mini-maps
    samples = obs["sample_id"].astype(str).to_numpy() if "sample_id" in obs.columns else None
    sample_stages: dict[str, str] = {}
    if samples is not None:
        for s_id in pd.Series(samples).unique():
            sample_stages[str(s_id)] = str(stages[samples == s_id][0])
    unique_samples = sorted(sample_stages.keys(), key=lambda s: (CANONICAL_STAGE_ORDER.index(sample_stages[s]) if sample_stages[s] in CANONICAL_STAGE_ORDER else 99, s)) if sample_stages else []
    n_sample_panels = min(6, len(unique_samples))

    # Layout: row 0 = combined map + bar chart + stats; row 1 = feature genes
    # row 2 (if samples) = per-sample tissue sections
    n_rows = 3 if n_sample_panels > 0 else 2
    n_feature_cols = max(1, len(feature_genes))
    height_ratios = [1.0, 1.0, 0.85] if n_rows == 3 else [1.0, 1.0]
    fig = plt.figure(figsize=(max(16, 3.5 * n_feature_cols), 4.0 * n_rows))
    gs = fig.add_gridspec(
        n_rows, max(n_feature_cols, n_sample_panels, 2),
        height_ratios=height_ratios,
        wspace=0.30, hspace=0.35,
    )

    # Row 0: combined map (left half) + bar chart (right half)
    n_top_cols = max(n_feature_cols, n_sample_panels, 2)
    mid = n_top_cols // 2
    ax_map = fig.add_subplot(gs[0, :mid])
    ax_stage = fig.add_subplot(gs[0, mid:])

    for stage in ordered:
        mask = stages == stage
        ax_map.scatter(
            coords[mask, 1],
            -coords[mask, 0],
            s=18,
            alpha=0.78,
            color=_stage_palette(stage),
            label=stage,
            linewidths=0.0,
            rasterized=True,
        )
    ax_map.set_title("Raw Visium spot layout by stage")
    ax_map.set_xlabel("Visium x")
    ax_map.set_ylabel("Visium y")
    ax_map.legend(frameon=False, fontsize=8, loc="best")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, alpha=0.18)

    stage_counts = pd.Series(spatial.get("stage_counts", {})).reindex(ordered).fillna(0.0)
    ax_stage.bar(stage_counts.index.astype(str), stage_counts.to_numpy(dtype=np.float32), color=[_stage_palette(stage) for stage in stage_counts.index])
    ax_stage.set_title("Spots per stage")
    ax_stage.set_ylabel("spots")
    ax_stage.tick_params(axis="x", rotation=25)
    ax_stage.grid(True, axis="y", alpha=0.22)
    ax_stage.text(
        0.97,
        0.97,
        "\n".join(
            [
                f"spots: {spatial['n_spots']}",
                f"genes: {spatial['n_genes']}",
                f"donors: {spatial['n_donors']}",
                f"samples: {spatial['n_samples']}",
            ]
        ),
        transform=ax_stage.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7E8", "edgecolor": "#D6C7A1", "alpha": 0.85},
    )

    # Row 1: feature gene spatial plots
    subtitle = "Raw spatial feature plots"
    if proxy_used:
        subtitle += " (proxy genes used)"
    feature_axes = [fig.add_subplot(gs[1, j]) for j in range(min(n_feature_cols, len(feature_genes)))]
    for ax, gene in zip(feature_axes, feature_genes, strict=False):
        values = panel[gene].to_numpy(dtype=np.float32) if gene in panel.columns else np.zeros(coords.shape[0], dtype=np.float32)
        scatter = ax.scatter(
            coords[:, 1],
            -coords[:, 0],
            c=values,
            s=14,
            cmap="YlOrRd",
            alpha=0.85,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(f"{roles.get(gene, 'feature')}: {gene}", fontsize=10)
        ax.set_xlabel("Visium x", fontsize=8)
        ax.set_ylabel("Visium y", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.18)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    # Fill remaining row-1 slots
    for j in range(len(feature_genes), gs.ncols):
        ax_empty = fig.add_subplot(gs[1, j])
        ax_empty.axis("off")
    if subtitle:
        fig.text(0.5, 0.365 if n_rows == 3 else 0.48, subtitle, ha="center", fontsize=9, color=PALETTE["muted"])

    # Row 2: per-sample tissue section mini-maps
    if n_sample_panels > 0 and samples is not None:
        sample_axes = [fig.add_subplot(gs[2, j]) for j in range(n_sample_panels)]
        display_samples = unique_samples[:n_sample_panels]
        for ax, s_id in zip(sample_axes, display_samples, strict=False):
            mask = samples == s_id
            s_stage = sample_stages[s_id]
            ax.scatter(
                coords[mask, 1],
                -coords[mask, 0],
                s=12,
                alpha=0.80,
                color=_stage_palette(s_stage),
                linewidths=0.0,
                rasterized=True,
            )
            n_spots = int(mask.sum())
            ax.set_title(f"{s_id} ({s_stage}, n={n_spots})", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)
        for j in range(n_sample_panels, gs.ncols):
            ax_empty = fig.add_subplot(gs[2, j])
            ax_empty.axis("off")

    fig.suptitle("StageBridge v1 research frontend: Visium preprocessing preview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_wes_preprocessing_frontend(data_output: dict[str, Any]) -> Figure:
    """Render the WES cohort preview from the preprocessing step."""
    configure_research_style()
    wes = data_output["wes"]
    frame = wes["frame"].copy()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.35)
    ax_tmb = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_stage_mut = fig.add_subplot(gs[1, 1])

    if not frame.empty:
        ordered = _sorted_stages(frame["stage"].astype(str).tolist())
        stage_tmb = [frame.loc[frame["stage"] == stage, "tmb"].to_numpy(dtype=np.float32) for stage in ordered]
        box = ax_tmb.boxplot(stage_tmb, tick_labels=ordered, patch_artist=True)
        for patch, stage in zip(box["boxes"], ordered, strict=False):
            patch.set_facecolor(_stage_palette(stage))
            patch.set_alpha(0.65)
        for idx, stage in enumerate(ordered, start=1):
            values = frame.loc[frame["stage"] == stage, "tmb"].to_numpy(dtype=np.float32)
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=values.shape[0])
            ax_tmb.scatter(np.full(values.shape[0], idx) + jitter, values, s=18, alpha=0.65, color=PALETTE["ink"], zorder=3)
        ax_tmb.set_title("Tumor mutation burden by stage")
        ax_tmb.set_ylabel("TMB")

        feature_cols = [col for col in wes["feature_columns"] if col != "tmb" and col in frame.columns]
        if feature_cols:
            display_cols = feature_cols[: min(10, len(feature_cols))]
            oncoprint = (
                frame[["patient_id", "stage", *display_cols]]
                .sort_values(["stage", "patient_id"])
                .set_index(["patient_id", "stage"])
            )
            onco_vals = oncoprint.to_numpy(dtype=np.float32)
            im = ax_heat.imshow(onco_vals, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
            ax_heat.set_title("Compact donor-stage oncoprint")
            ax_heat.set_xticks(np.arange(len(display_cols)))
            ax_heat.set_xticklabels(display_cols, rotation=30, ha="right", fontsize=8)
            ax_heat.set_yticks(np.arange(len(oncoprint.index)))
            ax_heat.set_yticklabels(
                [f"{patient}|{stage}" for patient, stage in oncoprint.index.tolist()],
                fontsize=6,
            )
            # Only annotate non-zero cells to avoid clutter
            for i in range(onco_vals.shape[0]):
                for j in range(onco_vals.shape[1]):
                    if onco_vals[i, j] > 0.01:
                        ax_heat.text(j, i, f"{onco_vals[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

            # Mutation frequency bar chart
            freq = frame[display_cols].mean().sort_values(ascending=True)
            ax_freq.barh(freq.index.astype(str), freq.values, color=PALETTE["accent"], alpha=0.88)
            ax_freq.set_title("Mutation frequency across cohort")
            ax_freq.set_xlabel("fraction mutated")

            # Per-stage mutation frequency
            stage_freq = frame.groupby("stage")[display_cols].mean().reindex(ordered).fillna(0.0)
            stage_freq_mat = stage_freq.to_numpy(dtype=np.float32)
            im2 = ax_stage_mut.imshow(stage_freq_mat, cmap="OrRd", aspect="auto", vmin=0.0, vmax=1.0)
            ax_stage_mut.set_title("Mutation frequency by stage")
            ax_stage_mut.set_xticks(np.arange(len(display_cols)))
            ax_stage_mut.set_xticklabels(display_cols, rotation=30, ha="right", fontsize=8)
            ax_stage_mut.set_yticks(np.arange(len(ordered)))
            ax_stage_mut.set_yticklabels(ordered)
            for i in range(stage_freq_mat.shape[0]):
                for j in range(stage_freq_mat.shape[1]):
                    ax_stage_mut.text(j, i, f"{stage_freq_mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=PALETTE["ink"])
            fig.colorbar(im2, ax=ax_stage_mut, fraction=0.046, pad=0.04)
        else:
            ax_heat.axis("off")
            ax_heat.text(0.5, 0.5, "No mutation features available", ha="center", va="center", fontsize=12)
            ax_freq.axis("off")
            ax_stage_mut.axis("off")
    else:
        for ax in [ax_tmb, ax_heat, ax_freq, ax_stage_mut]:
            ax.axis("off")
        ax_tmb.text(0.5, 0.5, "No WES rows for current filter", ha="center", va="center", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(
        "StageBridge v1 research frontend: WES preprocessing preview",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.935,
        f"{wes['n_rows']} samples  |  {wes['n_donors']} donors  |  {len(ordered)} stages  |  mean TMB: {wes.get('tmb_mean', float('nan')):.1f}",
        ha="center", fontsize=10, color=PALETTE["muted"],
    )
    return fig


def plot_spatial_mapping_frontend(spatial_output: dict[str, Any]) -> Figure:
    """Render the active spatial mapping branch as a publication-style summary."""
    configure_research_style()
    mapping = spatial_output["mapping_result"]
    compositions = np.asarray(mapping.compositions, dtype=np.float32)
    coords = np.asarray(mapping.coords, dtype=np.float32)
    feature_names = np.asarray(mapping.feature_names, dtype=object)
    winners = np.argmax(compositions, axis=1)
    confidences = compositions.max(axis=1)
    entropy = _soft_entropy(compositions)
    top_feature_idx = np.argsort(compositions.mean(axis=0))[::-1][: min(7, compositions.shape[1])]
    top_features = [str(feature_names[idx]) for idx in top_feature_idx]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.28)
    ax_winner = fig.add_subplot(gs[:, 0])
    ax_abundance = fig.add_subplot(gs[0, 1])
    ax_qc = fig.add_subplot(gs[1, 1])

    color_cycle = ["#19535F", "#0F766E", "#A63A2B", "#D18A00", "#6B7280", "#7C3AED", "#1D4ED8"]
    for rank, idx in enumerate(top_feature_idx):
        mask = winners == idx
        if not np.any(mask):
            continue
        ax_winner.scatter(
            coords[mask, 1],
            -coords[mask, 0],
            s=18,
            alpha=0.78,
            color=color_cycle[rank % len(color_cycle)],
            label=str(feature_names[idx]),
            linewidths=0.0,
            rasterized=True,
        )
    ax_winner.set_title(f"Spatial winner map: {mapping.method}")
    ax_winner.set_xlabel("Visium x")
    ax_winner.set_ylabel("Visium y")
    ax_winner.set_aspect("equal", adjustable="datalim")
    # Place legend horizontally below the map to avoid overlap
    ax_winner.legend(
        frameon=False, fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.08), ncol=min(4, len(top_features)),
    )
    ax_winner.grid(True, alpha=0.18)

    mean_abundance = compositions[:, top_feature_idx].mean(axis=0)
    ax_abundance.barh(top_features[::-1], mean_abundance[::-1], color=color_cycle[: len(top_features)][::-1], alpha=0.9)
    ax_abundance.set_title("Dominant mapped states")
    ax_abundance.set_xlabel("mean spot abundance")
    qc = spatial_output["spatial_mapping"]["qc"]
    source_path = _truncate_path(str(spatial_output["spatial_mapping"]["source_path"]))
    ax_abundance.text(
        0.03,
        0.95,
        "\n".join(
            [
                f"spots: {int(spatial_output['spatial_mapping']['n_spots'])}",
                f"features: {int(spatial_output['spatial_mapping']['n_features'])}",
                f"mean max assignment: {qc.get('mean_max_assignment', float('nan')):.3f}",
                f"source: {source_path}",
            ]
        ),
        transform=ax_abundance.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EEF7F4", "edgecolor": "#B8D5CD", "alpha": 0.85},
    )

    ax_qc.hist(confidences, bins=25, alpha=0.75, color=PALETTE["teal"], label="max assignment")
    ax_qc.hist(entropy, bins=25, alpha=0.55, color=PALETTE["gold"], label="assignment entropy")
    ax_qc.set_title("Mapping confidence and uncertainty")
    ax_qc.set_xlabel("score")
    ax_qc.set_ylabel("spots")
    ax_qc.legend(frameon=False)
    ax_qc.grid(True, alpha=0.22)

    fig.suptitle("StageBridge v1 research frontend: spatial mapping branch", fontsize=17, fontweight="bold", x=0.46)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_spatial_provider_comparison_frontend(provider_outputs: dict[str, dict[str, Any]]) -> Figure:
    """Render a provider-level comparison across Tangram, TACCO, and DestVI."""
    configure_research_style()
    rows: list[dict[str, Any]] = []
    for method, payload in provider_outputs.items():
        summary = payload.get("spatial_mapping", {})
        qc = summary.get("qc", {}) or {}
        rows.append(
            {
                "method": str(method),
                "status": str(payload.get("status", summary.get("status", "n/a"))),
                "n_spots": float(summary.get("n_spots", 0) or 0),
                "n_features": float(summary.get("n_features", 0) or 0),
                "mean_max_assignment": float(qc.get("mean_max_assignment", 0.0) or 0.0),
                "mean_entropy": float(qc.get("mean_entropy", 0.0) or 0.0),
                "provider_version": str(summary.get("provider_version", "n/a")),
                "execution_mode": str(summary.get("execution_mode", "n/a")),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No provider outputs available for comparison.")

    order = [method for method in ["tangram", "tacco", "destvi"] if method in table["method"].tolist()]
    table["method"] = pd.Categorical(table["method"], categories=order, ordered=True)
    table = table.sort_values("method").reset_index(drop=True)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)
    ax_quality = fig.add_subplot(gs[0, 0])
    ax_coverage = fig.add_subplot(gs[0, 1])
    ax_status = fig.add_subplot(gs[0, 2])

    methods = table["method"].astype(str).tolist()
    colors = [PALETTE["teal"] if status == "complete" else PALETTE["muted"] for status in table["status"]]
    x = np.arange(len(methods), dtype=np.float32)
    width = 0.38

    ax_quality.bar(x - width / 2, table["mean_max_assignment"].to_numpy(dtype=np.float32), width=width, color=colors, alpha=0.9, label="mean max assignment")
    ax_quality.bar(x + width / 2, table["mean_entropy"].to_numpy(dtype=np.float32), width=width, color=PALETTE["gold"], alpha=0.6, label="mean entropy")
    ax_quality.set_title("Provider confidence profile")
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(methods, rotation=20)
    ax_quality.set_ylabel("score")
    ax_quality.legend(frameon=False, fontsize=9)
    ax_quality.grid(True, axis="y", alpha=0.22)
    for idx, row in table.iterrows():
        ax_quality.text(float(x[idx]), max(float(row["mean_max_assignment"]), float(row["mean_entropy"])) + 0.01, row["status"], ha="center", va="bottom", fontsize=8, color=PALETTE["ink"])

    ax_coverage.bar(x - width / 2, table["n_spots"].to_numpy(dtype=np.float32), width=width, color=PALETTE["blue"], alpha=0.88, label="spots")
    ax_coverage.bar(x + width / 2, table["n_features"].to_numpy(dtype=np.float32), width=width, color=PALETTE["accent"], alpha=0.72, label="mapped features")
    ax_coverage.set_title("Provider output coverage")
    ax_coverage.set_xticks(x)
    ax_coverage.set_xticklabels(methods, rotation=20)
    ax_coverage.set_ylabel("count")
    ax_coverage.legend(frameon=False, fontsize=9)
    ax_coverage.grid(True, axis="y", alpha=0.22)

    ax_status.axis("off")
    ax_status.text(0.02, 0.97, "Provider status and provenance", fontsize=14, fontweight="bold", color=PALETTE["ink"], va="top")
    y = 0.82
    for row in table.itertuples(index=False):
        block_color = "#EEF7F4" if row.status == "complete" else "#F3F4F6"
        border_color = "#B8D5CD" if row.status == "complete" else "#D1D5DB"
        ax_status.text(
            0.04,
            y,
            f"{row.method.upper()}",
            fontsize=12,
            fontweight="bold",
            color=PALETTE["ink"],
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": block_color, "edgecolor": border_color},
        )
        ax_status.text(
            0.06,
            y - 0.10,
            "\n".join(
                [
                    f"status: {row.status}",
                    f"version: {row.provider_version}",
                    f"mode: {row.execution_mode}",
                    f"spots/features: {int(row.n_spots)}/{int(row.n_features)}",
                ]
            ),
            fontsize=9,
            color=PALETTE["ink"],
            va="top",
        )
        y -= 0.28

    fig.suptitle("StageBridge v1 research frontend: spatial provider comparison", fontsize=16, fontweight="bold")
    return fig


def plot_spatial_provider_maps_frontend(provider_outputs: dict[str, dict[str, Any]]) -> Figure:
    """Render side-by-side provider winner maps so live mappings are visually auditable."""
    configure_research_style()
    methods = [method for method in ["tangram", "tacco", "destvi"] if method in provider_outputs]
    n_panels = max(1, len(methods))
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.6), squeeze=False)
    axes_list = list(axes[0])

    color_cycle = ["#19535F", "#0F766E", "#A63A2B", "#D18A00", "#6B7280", "#7C3AED", "#1D4ED8", "#E11D48", "#0EA5E9"]
    for ax, method in zip(axes_list, methods, strict=False):
        payload = provider_outputs[method]
        summary = payload.get("spatial_mapping", {})
        mapping = payload.get("mapping_result")
        ax.set_title(method.upper())
        if mapping is None or mapping.compositions is None or mapping.coords is None:
            ax.axis("off")
            ax.text(0.5, 0.5, f"status: {summary.get('status', 'n/a')}", ha="center", va="center", fontsize=12)
            continue

        compositions = np.asarray(mapping.compositions, dtype=np.float32)
        row_sums = compositions.sum(axis=1, keepdims=True)
        probs = np.divide(compositions, row_sums, out=np.zeros_like(compositions), where=row_sums > 0)
        coords = np.asarray(mapping.coords, dtype=np.float32)
        winners = np.argmax(probs, axis=1)
        feature_names = np.asarray(mapping.feature_names, dtype=object)
        for idx in np.unique(winners):
            mask = winners == idx
            ax.scatter(
                coords[mask, 1],
                -coords[mask, 0],
                s=18,
                alpha=0.82,
                color=color_cycle[int(idx) % len(color_cycle)],
                label=str(feature_names[int(idx)]),
                linewidths=0.0,
                rasterized=True,
            )
        ax.set_xlabel("Visium x")
        ax.set_ylabel("Visium y")
        ax.grid(True, alpha=0.18)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"status: {summary.get('status', 'n/a')}",
                    f"mode: {summary.get('execution_mode', 'n/a')}",
                    f"spots/features: {summary.get('n_spots', 0)}/{summary.get('n_features', 0)}",
                    f"mean max assignment: {float((probs.max(axis=1)).mean()):.3f}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7E8", "edgecolor": "#D6C7A1", "alpha": 0.85},
        )
    if methods:
        handles, labels = axes_list[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(5, len(labels)))
    fig.suptitle("StageBridge v1 research frontend: live provider winner maps", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def plot_provider_benchmark_frontend(benchmark_output: dict[str, Any]) -> Figure:
    """Render the hybrid provider benchmark used for downstream selection."""
    configure_research_style()
    table = pd.DataFrame((benchmark_output.get("benchmark") or {}).get("provider_scores", []))
    if table.empty:
        raise ValueError("Provider benchmark output does not contain provider scores.")

    table = table.sort_values("hybrid_rank_score").reset_index(drop=True)
    methods = table["method"].astype(str).tolist()
    colors = [MODE_COLORS.get("set_only", PALETTE["teal"]) if idx == 0 else PALETTE["slate"] for idx in range(len(methods))]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)
    ax_hybrid = fig.add_subplot(gs[0, 0])
    ax_perf = fig.add_subplot(gs[0, 1])
    ax_qc = fig.add_subplot(gs[0, 2])

    ax_hybrid.bar(methods, table["hybrid_rank_score"].astype(float), color=colors, alpha=0.9)
    ax_hybrid.set_title("Hybrid provider score")
    ax_hybrid.set_ylabel("lower is better")
    ax_hybrid.tick_params(axis="x", rotation=20)
    for idx, row in table.iterrows():
        ax_hybrid.text(idx, float(row["hybrid_rank_score"]) + 0.03, f"{float(row['hybrid_rank_score']):.2f}", ha="center", va="bottom", fontsize=9)

    width = 0.36
    x = np.arange(len(methods), dtype=np.float32)
    ax_perf.bar(x - width / 2, table["sinkhorn_mean"].astype(float), width=width, color=PALETTE["accent"], alpha=0.82, label="mean Sinkhorn")
    ax_perf.bar(x + width / 2, table["calibration_mean"].astype(float), width=width, color=PALETTE["gold"], alpha=0.75, label="mean calibration")
    ax_perf.set_title("Downstream provider performance")
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels(methods, rotation=20)
    ax_perf.legend(frameon=False, fontsize=9)
    ax_perf.grid(True, axis="y", alpha=0.22)

    ax_qc.plot(methods, table["mean_max_assignment"].astype(float), marker="o", linewidth=2.0, color=PALETTE["teal"], label="max assignment")
    ax_qc.plot(methods, table["mean_normalized_entropy"].astype(float), marker="s", linewidth=2.0, color=PALETTE["blue"], label="norm entropy")
    ax_qc.plot(methods, table["rows_close_to_one_frac"].astype(float), marker="^", linewidth=2.0, color=PALETTE["signal"], label="rows close to 1")
    ax_qc.set_title("Mapping QC profile")
    ax_qc.set_ylabel("score")
    ax_qc.legend(frameon=False, fontsize=9)
    ax_qc.grid(True, alpha=0.22)
    ax_qc.text(
        0.02,
        0.04,
        "\n".join(
            [
                f"selected: {(benchmark_output.get('benchmark') or {}).get('selected_provider', 'n/a')}",
                f"status: {(benchmark_output.get('benchmark') or {}).get('selection_status', 'n/a')}",
                f"action: {(benchmark_output.get('benchmark') or {}).get('recommended_action', 'n/a')}",
            ]
        ),
        transform=ax_qc.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF7E8", "edgecolor": "#D6C7A1", "alpha": 0.85},
    )

    fig.suptitle("StageBridge v1 research frontend: provider benchmark and winner selection", fontsize=16, fontweight="bold")
    return fig


def plot_spatial_provider_abundance_frontend(provider_outputs: dict[str, dict[str, Any]]) -> Figure:
    """Render abundance and entropy comparisons across live spatial providers."""
    configure_research_style()
    methods = [method for method in ["tangram", "tacco", "destvi"] if method in provider_outputs]
    provider_frames: dict[str, pd.DataFrame] = {}
    provider_entropy: dict[str, np.ndarray] = {}
    for method in methods:
        payload = provider_outputs[method]
        matrix, columns = _provider_matrix_and_columns(payload)
        if matrix is None or not columns:
            continue
        probs = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-8, None)
        provider_frames[method] = pd.DataFrame(probs, columns=columns)
        provider_entropy[method] = _soft_entropy(probs)
    if not provider_frames:
        raise ValueError("No provider matrices available for abundance/entropy plotting.")

    shared_features = sorted(set.intersection(*(set(frame.columns) for frame in provider_frames.values())))
    if not shared_features:
        shared_features = list(next(iter(provider_frames.values())).columns[: min(6, next(iter(provider_frames.values())).shape[1])])
    top_shared = (
        pd.concat([frame[shared_features].mean().rename(method) for method, frame in provider_frames.items()], axis=1)
        .mean(axis=1)
        .sort_values(ascending=False)
        .head(min(6, len(shared_features)))
        .index.tolist()
    )

    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.26)
    ax_abundance = fig.add_subplot(gs[0, 0])
    ax_entropy = fig.add_subplot(gs[0, 1])

    abundance = pd.DataFrame(
        {
            method: provider_frames[method][top_shared].mean()
            for method in methods
            if method in provider_frames
        }
    )
    abundance.plot.bar(ax=ax_abundance, color=[PALETTE["teal"], PALETTE["gold"], PALETTE["accent"]][: abundance.shape[1]], alpha=0.86)
    ax_abundance.set_title("Shared feature abundance across providers")
    ax_abundance.set_ylabel("mean normalized abundance")
    ax_abundance.tick_params(axis="x", rotation=25)
    ax_abundance.legend(frameon=False, fontsize=9, title="provider")
    ax_abundance.grid(True, axis="y", alpha=0.22)

    entropy_data = [provider_entropy[method] for method in methods if method in provider_entropy]
    box = ax_entropy.boxplot(
        entropy_data,
        tick_labels=[method.upper() for method in methods if method in provider_entropy],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], [PALETTE["teal"], PALETTE["gold"], PALETTE["accent"]], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
    ax_entropy.set_title("Spot-level assignment entropy")
    ax_entropy.set_ylabel("entropy")
    ax_entropy.grid(True, axis="y", alpha=0.22)

    fig.suptitle("StageBridge v1 research frontend: provider abundance and entropy audit", fontsize=16, fontweight="bold")
    return fig


def plot_context_frontend(context_output: dict[str, Any]) -> Figure:
    """Render typed niche context as a scientific summary figure."""
    configure_research_style()
    typed = context_output["typed_tokens"]
    tokens = np.asarray(typed.tokens, dtype=np.float32)
    coords = np.asarray(typed.coords, dtype=np.float32)
    obs = typed.obs.copy()
    stages = obs["stage"].astype(str).to_numpy()
    groups = list(typed.schema.typed_feature_names)
    stage_order = _sorted_stages(stages)
    stage_means = pd.DataFrame(tokens, columns=groups).assign(stage=stages).groupby("stage").mean().reindex(stage_order).fillna(0.0)
    dominant_group = np.argmax(tokens, axis=1)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.28)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_diag = fig.add_subplot(gs[1, 1])

    heat = ax_heat.imshow(stage_means.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax_heat.set_title("Stage-wise typed niche composition")
    ax_heat.set_xticks(np.arange(len(groups)))
    ax_heat.set_xticklabels(groups, rotation=20, ha="right")
    ax_heat.set_yticks(np.arange(len(stage_order)))
    ax_heat.set_yticklabels(stage_order)
    for i in range(stage_means.shape[0]):
        for j in range(stage_means.shape[1]):
            ax_heat.text(j, i, f"{stage_means.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
    fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)

    cumulative = np.zeros(len(stage_order), dtype=np.float32)
    group_colors = ["#A63A2B", "#D18A00", "#0F766E", "#245C73"]
    for color, group in zip(group_colors, groups):
        values = stage_means[group].to_numpy(dtype=np.float32)
        ax_stack.bar(stage_order, values, bottom=cumulative, color=color, alpha=0.88, label=group)
        cumulative += values
    ax_stack.set_title("Typed context balance across the disease ladder")
    ax_stack.set_ylabel("mean typed token value")
    ax_stack.legend(frameon=False)
    ax_stack.tick_params(axis="x", rotation=20)

    for idx, group_name in enumerate(groups):
        mask = dominant_group == idx
        ax_map.scatter(
            coords[mask, 1],
            -coords[mask, 0],
            s=16,
            alpha=0.72,
            color=group_colors[idx],
            label=group_name,
            linewidths=0.0,
            rasterized=True,
        )
    ax_map.set_title("Dominant typed niche group per spatial node")
    ax_map.set_xlabel("Visium x")
    ax_map.set_ylabel("Visium y")
    ax_map.legend(frameon=False, fontsize=9)
    ax_map.grid(True, alpha=0.18)

    summary = context_output["context_model"]
    diagnostics = {
        "mode": summary.get("mode", "n/a"),
        "spatial_mapping": summary.get("spatial_mapping_method", "n/a"),
        "token_rows": int(summary.get("typed_token_summary", {}).get("n_tokens", 0)),
        "token_dim": int(summary.get("typed_token_summary", {}).get("token_dim", 0)),
        "context_norm": float(summary.get("example_context_norm", summary.get("graph_context_norm", 0.0))),
        "context_dim": int(summary.get("example_context_dim", summary.get("graph_context_dim", 0))),
    }
    if "graph_num_edges" in summary:
        diagnostics["graph_num_edges"] = int(summary["graph_num_edges"])
        diagnostics["graph_num_nodes"] = int(summary["graph_num_nodes"])
    ax_diag.axis("off")
    ax_diag.text(
        0.02,
        0.95,
        "Context branch diagnostics",
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
    )
    y = 0.82
    for key, value in diagnostics.items():
        ax_diag.text(0.04, y, f"{key}", fontsize=10, color=PALETTE["muted"], va="top")
        ax_diag.text(0.52, y, f"{value}", fontsize=11, color=PALETTE["ink"], va="top", fontweight="bold")
        y -= 0.1
    ax_diag.text(
        0.04,
        0.15,
        "Typed spot tokens feed the local set encoder first.\nGraph propagation is optional and must earn its place.",
        fontsize=10,
        color=PALETTE["accent"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF0EB", "edgecolor": "#E2B8AA", "alpha": 0.85},
    )

    fig.suptitle("StageBridge v1 research frontend: typed niche context branch", fontsize=17, fontweight="bold", x=0.46)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_transition_frontend(transition_output: dict[str, Any], evaluation_output: dict[str, Any]) -> Figure:
    """Render transition dynamics and evaluation as a publication-style summary."""
    configure_research_style()
    x_src = transition_output["x_src_test"]
    x_tgt = transition_output["x_tgt_test"]
    context = transition_output["context"]
    context_tokens = transition_output.get("context_tokens")
    edge_id = int(transition_output["edge_id"])
    model = transition_output["model"]
    x_pred = rollout_edge_transition(
        model,
        x_src,
        context=context,
        context_tokens=context_tokens,
        edge_id=edge_id,
        num_steps=8,
        stochastic=False,
    )

    src_np = x_src.detach().cpu().numpy()
    pred_np = x_pred.detach().cpu().numpy()
    tgt_np = x_tgt.detach().cpu().numpy()
    emb, emb_var = _pca2_with_variance(np.vstack([src_np, pred_np, tgt_np]))
    n_src = src_np.shape[0]
    n_pred = pred_np.shape[0]
    src_emb = emb[:n_src]
    pred_emb = emb[n_src : n_src + n_pred]
    tgt_emb = emb[n_src + n_pred :]
    flow_matrix, source_labels, target_labels = compute_macroflow_matrix(src_np, pred_np, n_clusters=min(6, max(2, src_np.shape[0] // 4)))

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 0.65], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.28)
    ax_embed = fig.add_subplot(gs[:, 0])
    ax_history = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[1, 1])
    ax_metrics = fig.add_subplot(gs[:, 2])

    ax_embed.scatter(src_emb[:, 0], src_emb[:, 1], s=20, alpha=0.45, color="#64748B", label="source")
    ax_embed.scatter(pred_emb[:, 0], pred_emb[:, 1], s=20, alpha=0.65, color=MODE_COLORS.get(transition_output["mode"], PALETTE["teal"]), label="predicted")
    ax_embed.scatter(tgt_emb[:, 0], tgt_emb[:, 1], s=20, alpha=0.45, color=PALETTE["accent"], label="target")
    ax_embed.set_title(f"Edge manifold: {transition_output['edge']} ({transition_output['mode']})")
    ax_embed.set_xlabel(f"PC 1 ({emb_var[0]:.1f}%)")
    ax_embed.set_ylabel(f"PC 2 ({emb_var[1]:.1f}%)")
    ax_embed.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax_embed.grid(True, alpha=0.22)

    history = pd.DataFrame(transition_output.get("training_history", []))
    if not history.empty:
        ax_history.plot(history["epoch"], history["loss_total"], color=PALETTE["teal"], linewidth=2.2, label="total")
        ax_history.plot(history["epoch"], history["loss_drift"], color=PALETTE["gold"], linewidth=1.8, label="drift")
        ax_history.plot(history["epoch"], history["loss_diffusion"], color=PALETTE["accent"], linewidth=1.8, label="diffusion")
    ax_history.set_title("Bridge optimization trajectory")
    ax_history.set_xlabel("epoch")
    ax_history.set_ylabel("loss")
    ax_history.legend(frameon=False)
    ax_history.grid(True, alpha=0.22)
    heldout = evaluation_output["heldout_metrics"]
    calibration = evaluation_output["calibration"]
    context_sensitivity = evaluation_output.get("context_sensitivity") or {}
    attention = transition_output.get("attention_summary") or {}
    auxiliary = transition_output.get("auxiliary_context_shuffle_metrics") or {}
    attention_lines: list[str] = []
    if attention:
        attention_lines.extend(
            [
                f"encoder delta: {transition_output.get('encoder_parameter_delta', 0.0):.4f}",
                f"attention maps: {', '.join(attention.get('available_maps', []))}",
                f"top token types: {', '.join(attention.get('top_token_types', [])) or 'n/a'}",
                f"attention entropy: {attention.get('pma_attention_entropy', float('nan')):.3f}",
                f"confidence-weighted entropy: {attention.get('confidence_weighted_attention_entropy', float('nan')):.3f}",
            ]
        )
    if auxiliary:
        attention_lines.extend(
            [
                f"context shuffle loss: {auxiliary.get('loss', float('nan')):.3f}",
                f"context shuffle accuracy: {auxiliary.get('accuracy', float('nan')):.3f}",
                f"context separation: {auxiliary.get('separation_score', float('nan')):.3f}",
            ]
        )
    ax_metrics.axis("off")
    all_metric_lines = [
        f"sinkhorn: {heldout['sinkhorn']:.3f}",
        f"sinkhorn delta: {heldout['sinkhorn_delta']:.3f}",
        f"mmd: {heldout['mmd_rbf']:.3f}",
        f"auc: {heldout['classifier_auc']:.3f}",
        f"direction cosine: {heldout['direction_cosine']:.3f}",
        f"calibration error: {calibration['mean_abs_shift_error']:.3f}",
        f"context delta: {context_sensitivity.get('context_sensitivity_delta', float('nan')):.3f}",
        *attention_lines,
    ]
    ax_metrics.text(
        0.05,
        0.97,
        "Evaluation metrics",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
        va="top",
        transform=ax_metrics.transAxes,
    )
    ax_metrics.text(
        0.05,
        0.90,
        "\n".join(all_metric_lines),
        transform=ax_metrics.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color=PALETTE["ink"],
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EEF4FB", "edgecolor": "#B6C7DA", "alpha": 0.85},
    )

    heat = ax_flow.imshow(flow_matrix, cmap="magma", aspect="auto")
    ax_flow.set_title("Macroflow from source to predicted states")
    ax_flow.set_xticks(np.arange(len(target_labels)))
    ax_flow.set_xticklabels(target_labels, rotation=30, ha="right", fontsize=8)
    ax_flow.set_yticks(np.arange(len(source_labels)))
    ax_flow.set_yticklabels(source_labels, fontsize=8)
    for i in range(flow_matrix.shape[0]):
        for j in range(flow_matrix.shape[1]):
            ax_flow.text(j, i, f"{flow_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(heat, ax=ax_flow, fraction=0.046, pad=0.04)

    fig.suptitle("StageBridge v1 research frontend: transition and evaluation branch", fontsize=17, fontweight="bold", x=0.46)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_biological_insight_frontend(evaluation_output: dict[str, Any]) -> Figure:
    """Render edge-level typed niche biology as a publication-style summary."""
    configure_research_style()
    biology = evaluation_output.get("biology_summary") or {}
    stage_profiles = biology.get("stage_mean_profiles", {})
    groups = biology.get("typed_groups", [])
    edge_delta = biology.get("edge_delta_by_group", {})
    stages = biology.get("stage_order", [])

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.28)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])

    if stage_profiles and groups:
        stage_frame = pd.DataFrame.from_dict(stage_profiles, orient="index")[groups].reindex(stages).fillna(0.0)
        heat = ax_heat.imshow(stage_frame.to_numpy(), cmap="YlOrBr", aspect="auto")
        ax_heat.set_title("Typed niche profiles across the disease ladder")
        ax_heat.set_xticks(np.arange(len(groups)))
        ax_heat.set_xticklabels(groups, rotation=20, ha="right")
        ax_heat.set_yticks(np.arange(len(stage_frame.index)))
        ax_heat.set_yticklabels(stage_frame.index.tolist())
        for i in range(stage_frame.shape[0]):
            for j in range(stage_frame.shape[1]):
                ax_heat.text(j, i, f"{stage_frame.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
        fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)
    else:
        ax_heat.axis("off")
        ax_heat.text(0.5, 0.5, "No biology summary available", ha="center", va="center")

    if edge_delta:
        delta_series = pd.Series(edge_delta).sort_values()
        colors = [PALETTE["accent"] if value > 0 else PALETTE["blue"] for value in delta_series.values]
        ax_delta.barh(delta_series.index.tolist(), delta_series.values, color=colors, alpha=0.88)
        ax_delta.axvline(0.0, color=PALETTE["ink"], linewidth=1.0)
    ax_delta.set_title(f"Edge shift by typed group: {biology.get('edge', 'n/a')}")
    ax_delta.set_xlabel("target - source typed niche mean")
    interpretation = "\n".join(biology.get("interpretation", []))
    ax_delta.text(
        0.02,
        0.04,
        interpretation or "No interpretation generated.",
        transform=ax_delta.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF4EC", "edgecolor": "#E5C2AF", "alpha": 0.85},
    )

    fig.suptitle("StageBridge v1 research frontend: edge-level biological insight", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_mode_comparison_frontend(mode_table: pd.DataFrame, *, edge: str) -> Figure:
    """Render a matched mode-comparison ladder for one edge."""
    configure_research_style()
    table = mode_table.copy()
    order = [
        mode
        for mode in [
            "rna_only",
            "pooled",
            "deep_sets",
            "set_only",
            "typed_hierarchical_transformer",
            "deep_sets_transformer_hybrid",
            "graph_of_sets",
        ]
        if mode in table["mode"].tolist()
    ]
    table["mode"] = pd.Categorical(table["mode"], categories=order, ordered=True)
    table = table.sort_values("mode").reset_index(drop=True)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax_sink = fig.add_subplot(gs[0, 0])
    ax_cal = fig.add_subplot(gs[0, 1])

    colors = [MODE_COLORS.get(mode, PALETTE["slate"]) for mode in table["mode"].astype(str)]
    ax_sink.bar(table["mode"].astype(str), table["sinkhorn"].astype(float), color=colors, alpha=0.9)
    ax_sink.set_title(f"Mode ladder: held-out Sinkhorn ({edge})")
    ax_sink.set_ylabel("sinkhorn")
    ax_sink.tick_params(axis="x", rotation=20)
    for idx, row in table.iterrows():
        ax_sink.text(idx, float(row["sinkhorn"]) + 0.02, f"{float(row['sinkhorn']):.2f}", ha="center", va="bottom", fontsize=9)

    ax_cal.plot(table["mode"].astype(str), table["calibration_error"].astype(float), marker="o", linewidth=2.0, color=PALETTE["teal"])
    if "context_sensitivity_delta" in table.columns:
        ax_aux = ax_cal.twinx()
        ax_aux.bar(
            table["mode"].astype(str),
            table["context_sensitivity_delta"].fillna(0.0).astype(float),
            alpha=0.22,
            color=PALETTE["gold"],
            label="context delta",
        )
        ax_aux.set_ylabel("context sensitivity delta")
    ax_cal.set_title(f"Calibration and context sensitivity ({edge})")
    ax_cal.set_ylabel("mean absolute shift error")
    ax_cal.tick_params(axis="x", rotation=20)
    ax_cal.grid(True, alpha=0.22)

    fig.suptitle("StageBridge v1 research frontend: matched context-mode comparison", fontsize=16, fontweight="bold")
    return fig


def plot_latent_comparison_frontend(latent_table: pd.DataFrame, *, edge: str, mode: str) -> Figure:
    """Render a compact latent-backend sensitivity comparison."""
    configure_research_style()
    table = latent_table.copy().reset_index(drop=True)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.25)
    ax_sink = fig.add_subplot(gs[0, 0])
    ax_cal = fig.add_subplot(gs[0, 1])

    colors = [PALETTE["teal"] if backend == "hlca" else PALETTE["gold"] for backend in table["backend"].astype(str)]
    ax_sink.bar(table["backend"].astype(str), table["sinkhorn"].astype(float), color=colors, alpha=0.9)
    ax_sink.set_title(f"Latent sensitivity: held-out Sinkhorn ({edge}, {mode})")
    ax_sink.set_ylabel("sinkhorn")
    for idx, row in table.iterrows():
        ax_sink.text(idx, float(row["sinkhorn"]) + 0.02, f"{float(row['sinkhorn']):.2f}", ha="center", va="bottom", fontsize=9)

    ax_cal.plot(table["backend"].astype(str), table["calibration_error"].astype(float), marker="o", linewidth=2.0, color=PALETTE["accent"])
    ax_cal.set_title(f"Latent sensitivity: calibration ({edge}, {mode})")
    ax_cal.set_ylabel("mean absolute shift error")
    ax_cal.grid(True, alpha=0.22)
    summary_lines = [
        f"{row.backend}: +{row.dominant_increase_group or 'n/a'} / -{row.dominant_decrease_group or 'n/a'}"
        for row in table.itertuples(index=False)
    ]
    ax_cal.text(
        0.02,
        0.04,
        "\n".join(summary_lines),
        transform=ax_cal.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF4EC", "edgecolor": "#E5C2AF", "alpha": 0.85},
    )

    fig.suptitle("StageBridge v1 research frontend: latent-backend comparison", fontsize=16, fontweight="bold")
    return fig


def plot_transformer_attention_frontend(context_output: dict[str, Any]) -> Figure:
    """Visualize transformer attention patterns from the hierarchical context encoder.

    Shows: fusion attention heatmap, per-group token counts, relation scores,
    and confidence profiles across biological groups.
    """
    configure_research_style()
    summary = context_output.get("context_model", {})
    diagnostics = summary.get("hierarchical_diagnostics", {})
    fusion_scores = diagnostics.get("fusion_attention_by_group", {})
    relation_scores = diagnostics.get("relation_scores", {})
    group_diagnostics = diagnostics.get("group_diagnostics", [])

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.32)
    ax_fusion = fig.add_subplot(gs[0, 0])
    ax_groups = fig.add_subplot(gs[0, 1])
    ax_relations = fig.add_subplot(gs[1, 0])
    ax_arch = fig.add_subplot(gs[1, 1])

    # Fusion attention heatmap
    if fusion_scores:
        groups = list(fusion_scores.keys())
        query_roles = ["source_stage", "target_stage", "transition"] + groups
        query_roles = query_roles[: summary.get("num_fusion_queries", 7)]
        mat = np.zeros((len(query_roles), len(groups)), dtype=np.float32)
        for j, g in enumerate(groups):
            scores = fusion_scores[g]
            if isinstance(scores, (list, np.ndarray)):
                for i in range(min(len(scores), len(query_roles))):
                    mat[i, j] = float(scores[i]) if i < len(scores) else 0.0
            else:
                mat[0, j] = float(scores)
        im = ax_fusion.imshow(mat, cmap="YlOrBr", aspect="auto")
        ax_fusion.set_xticks(np.arange(len(groups)))
        ax_fusion.set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
        ax_fusion.set_yticks(np.arange(len(query_roles)))
        ax_fusion.set_yticklabels(query_roles, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax_fusion.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax_fusion, fraction=0.046, pad=0.04)
        ax_fusion.set_title("Fusion query attention over typed groups")
    else:
        ax_fusion.axis("off")
        ax_fusion.text(0.5, 0.5, "No fusion attention data", ha="center", va="center", fontsize=12, color=PALETTE["muted"])

    # Per-group token counts and confidence
    if group_diagnostics:
        g_names = [g.get("group_name", f"g{i}") for i, g in enumerate(group_diagnostics)]
        g_counts = [int(g.get("token_count", 0)) for g in group_diagnostics]
        g_conf = [float(g.get("mean_confidence", 0.0)) for g in group_diagnostics]
        group_colors = ["#A63A2B", "#D18A00", "#0F766E", "#245C73"]
        x = np.arange(len(g_names))
        bars = ax_groups.bar(x, g_counts, color=[group_colors[i % len(group_colors)] for i in range(len(g_names))], alpha=0.88)
        for i, (c, conf) in enumerate(zip(g_counts, g_conf)):
            ax_groups.text(i, c + max(g_counts) * 0.02, f"conf={conf:.2f}", ha="center", fontsize=8, color=PALETTE["ink"])
        ax_groups.set_xticks(x)
        ax_groups.set_xticklabels(g_names, rotation=20, ha="right")
        ax_groups.set_title("Token count and confidence by group")
        ax_groups.set_ylabel("tokens")
    else:
        ax_groups.axis("off")
        ax_groups.text(0.5, 0.5, "No group diagnostics", ha="center", va="center", fontsize=12, color=PALETTE["muted"])

    # Relation token scores
    if relation_scores:
        pairs = list(relation_scores.keys())
        scores = [float(relation_scores[p]) for p in pairs]
        colors = [PALETTE["teal"] if s > np.mean(scores) else PALETTE["slate"] for s in scores]
        ax_relations.barh(pairs, scores, color=colors, alpha=0.88)
        ax_relations.set_title("Inter-group relation token scores")
        ax_relations.set_xlabel("score")
    else:
        ax_relations.axis("off")
        ax_relations.text(0.5, 0.5, "No relation scores", ha="center", va="center", fontsize=12, color=PALETTE["muted"])

    # Architecture summary
    ax_arch.axis("off")
    mode = summary.get("mode", "n/a")
    arch_lines = [
        f"Context encoder: {mode}",
        f"Hidden dim: {summary.get('hidden_dim', 'n/a')}",
        f"Attention heads: {summary.get('num_heads', 'n/a')}",
        f"Inducing points: {summary.get('num_inducing_points', 'n/a')}",
        f"Group summary tokens: {summary.get('num_group_summary_tokens', 'n/a')}",
        f"Fusion queries: {summary.get('num_fusion_queries', 'n/a')}",
        f"Spatial RPE: {summary.get('use_spatial_rpe', 'n/a')}",
        f"Confidence gating: {summary.get('use_confidence_gate', 'n/a')}",
        f"Token dropout: {summary.get('token_dropout_rate', 'n/a')}",
        f"Context dim: {summary.get('example_context_dim', summary.get('graph_context_dim', 'n/a'))}",
        f"Context norm: {float(summary.get('example_context_norm', summary.get('graph_context_norm', 0))):.3f}",
    ]
    if summary.get("encoder_parameter_count"):
        arch_lines.append(f"Parameters: {int(summary['encoder_parameter_count']):,}")
    ax_arch.text(
        0.05, 0.95, "Transformer architecture",
        fontsize=14, fontweight="bold", color=PALETTE["ink"], va="top",
    )
    ax_arch.text(
        0.05, 0.82, "\n".join(arch_lines),
        fontsize=10, color=PALETTE["ink"], va="top", family="monospace",
    )

    fig.suptitle("Hierarchical transformer context encoder diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_multi_embedding_frontend(
    latent: np.ndarray,
    stages: np.ndarray,
    *,
    title: str = "Multi-embedding comparison",
) -> Figure:
    """Plot PCA (with variance %), UMAP, t-SNE, and PHATE side by side."""
    configure_research_style()
    ordered = _sorted_stages(stages)

    pca_coords, pca_var = _pca2_with_variance(latent)
    umap_coords = _umap2(latent)
    tsne_coords = _tsne2(latent)
    phate_coords = _phate2(latent)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    embed_data = [
        (pca_coords, f"PCA  (PC1={pca_var[0]:.1f}%, PC2={pca_var[1]:.1f}%)", "PC 1", "PC 2"),
        (umap_coords, "UMAP", "UMAP 1", "UMAP 2"),
        (tsne_coords, "t-SNE", "t-SNE 1", "t-SNE 2"),
        (phate_coords, "PHATE", "PHATE 1", "PHATE 2"),
    ]
    for ax, (coords, subtitle, xlabel, ylabel) in zip(axes, embed_data):
        for stage in ordered:
            mask = stages == stage
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=10, alpha=0.65, color=_stage_palette(stage), label=stage,
                linewidths=0.0, rasterized=True,
            )
        ax.set_title(subtitle, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="best")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


__all__ = [
    "configure_research_style",
    "plot_biological_insight_frontend",
    "plot_context_frontend",
    "plot_latent_comparison_frontend",
    "plot_mode_comparison_frontend",
    "plot_multi_embedding_frontend",
    "plot_reference_frontend",
    "plot_snrna_preprocessing_frontend",
    "plot_spatial_preprocessing_frontend",
    "plot_provider_benchmark_frontend",
    "plot_spatial_provider_comparison_frontend",
    "plot_spatial_provider_abundance_frontend",
    "plot_spatial_provider_maps_frontend",
    "plot_spatial_mapping_frontend",
    "plot_transformer_attention_frontend",
    "plot_transition_frontend",
    "plot_wes_preprocessing_frontend",
]
