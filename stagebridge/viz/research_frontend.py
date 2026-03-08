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


def plot_reference_frontend(reference_output: dict[str, Any]) -> Figure:
    """Render the active reference branch as a four-panel scientific figure."""
    configure_research_style()
    cohort = reference_output["cohort"]
    reference = reference_output["reference"]
    diagnostics = reference["diagnostics"]
    label_transfer = reference["label_transfer"]
    coords = _pca2(cohort.latent)
    stages = cohort.obs["stage"].astype(str).to_numpy()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.1, 1.0], wspace=0.22, hspace=0.28)
    ax_latent = fig.add_subplot(gs[:, 0])
    ax_labels = fig.add_subplot(gs[0, 1])
    ax_centroids = fig.add_subplot(gs[1, 1])

    for stage in _sorted_stages(stages):
        mask = stages == stage
        ax_latent.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.72,
            color=_stage_palette(stage),
            label=stage,
            linewidths=0.0,
            rasterized=True,
        )
    ax_latent.set_title("Reference latent structure by stage")
    ax_latent.set_xlabel("PCA 1")
    ax_latent.set_ylabel("PCA 2")
    ax_latent.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    ax_latent.grid(True, alpha=0.22)

    top_labels = label_transfer.get("top_labels", [])
    if top_labels:
        label_df = pd.DataFrame(top_labels, columns=["label", "count"]).sort_values("count", ascending=True)
        ax_labels.barh(label_df["label"], label_df["count"], color=PALETTE["teal"], alpha=0.88)
    ax_labels.set_title("HLCA label coverage and dominant compartments")
    ax_labels.set_xlabel("cells")
    ax_labels.tick_params(axis="y", labelsize=9)
    coverage = float(label_transfer.get("coverage", 0.0))
    leakage = diagnostics["donor_leakage"]
    ax_labels.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"source: {reference['source_path']}",
                f"shape: {tuple(reference['latent_shape'])}",
                f"label coverage: {coverage:.3f}",
                f"donor leakage: {float(leakage.get('logreg_accuracy', float('nan'))):.3f}",
                f"chance baseline: {float(leakage.get('chance_accuracy', float('nan'))):.3f}",
            ]
        ),
        transform=ax_labels.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF7E8", "edgecolor": "#D6C7A1"},
    )

    matrix, ordered_stages = _centroid_distance_matrix(diagnostics["stage_preservation"]["centroid_distances"])
    im = ax_centroids.imshow(matrix, cmap="YlOrBr")
    ax_centroids.set_title("Stage centroid separation")
    ax_centroids.set_xticks(np.arange(len(ordered_stages)))
    ax_centroids.set_xticklabels(ordered_stages, rotation=35, ha="right")
    ax_centroids.set_yticks(np.arange(len(ordered_stages)))
    ax_centroids.set_yticklabels(ordered_stages)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            ax_centroids.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["ink"])
    fig.colorbar(im, ax=ax_centroids, fraction=0.046, pad=0.04)

    fig.suptitle("StageBridge v1 research frontend: reference latent branch", fontsize=17, fontweight="bold", x=0.45)
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
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)
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
    ax_winner.legend(frameon=False, fontsize=8, loc="upper right")
    ax_winner.grid(True, alpha=0.18)

    mean_abundance = compositions[:, top_feature_idx].mean(axis=0)
    ax_abundance.barh(top_features[::-1], mean_abundance[::-1], color=color_cycle[: len(top_features)][::-1], alpha=0.9)
    ax_abundance.set_title("Dominant mapped states")
    ax_abundance.set_xlabel("mean spot abundance")
    qc = spatial_output["spatial_mapping"]["qc"]
    ax_abundance.text(
        0.03,
        0.95,
        "\n".join(
            [
                f"spots: {int(spatial_output['spatial_mapping']['n_spots'])}",
                f"features: {int(spatial_output['spatial_mapping']['n_features'])}",
                f"mean max assignment: {qc.get('mean_max_assignment', float('nan')):.3f}",
                f"source: {spatial_output['spatial_mapping']['source_path']}",
            ]
        ),
        transform=ax_abundance.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EEF7F4", "edgecolor": "#B8D5CD"},
    )

    ax_qc.hist(confidences, bins=25, alpha=0.75, color=PALETTE["teal"], label="max assignment")
    ax_qc.hist(entropy, bins=25, alpha=0.55, color=PALETTE["gold"], label="assignment entropy")
    ax_qc.set_title("Mapping confidence and uncertainty")
    ax_qc.set_xlabel("score")
    ax_qc.set_ylabel("spots")
    ax_qc.legend(frameon=False)
    ax_qc.grid(True, alpha=0.22)

    fig.suptitle("StageBridge v1 research frontend: spatial mapping branch", fontsize=17, fontweight="bold", x=0.46)
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
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF0EB", "edgecolor": "#E2B8AA"},
    )

    fig.suptitle("StageBridge v1 research frontend: typed niche context branch", fontsize=17, fontweight="bold", x=0.46)
    return fig


def plot_transition_frontend(transition_output: dict[str, Any], evaluation_output: dict[str, Any]) -> Figure:
    """Render transition dynamics and evaluation as a publication-style summary."""
    configure_research_style()
    x_src = transition_output["x_src_test"]
    x_tgt = transition_output["x_tgt_test"]
    context = transition_output["context"]
    edge_id = int(transition_output["edge_id"])
    model = transition_output["model"]
    x_pred = rollout_edge_transition(model, x_src, context=context, edge_id=edge_id, num_steps=8, stochastic=False)

    src_np = x_src.detach().cpu().numpy()
    pred_np = x_pred.detach().cpu().numpy()
    tgt_np = x_tgt.detach().cpu().numpy()
    emb = _pca2(np.vstack([src_np, pred_np, tgt_np]))
    n_src = src_np.shape[0]
    n_pred = pred_np.shape[0]
    src_emb = emb[:n_src]
    pred_emb = emb[n_src : n_src + n_pred]
    tgt_emb = emb[n_src + n_pred :]
    flow_matrix, source_labels, target_labels = compute_macroflow_matrix(src_np, pred_np, n_clusters=min(6, max(2, src_np.shape[0] // 4)))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.28)
    ax_embed = fig.add_subplot(gs[:, 0])
    ax_history = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[1, 1])

    ax_embed.scatter(src_emb[:, 0], src_emb[:, 1], s=20, alpha=0.45, color="#64748B", label="source")
    ax_embed.scatter(pred_emb[:, 0], pred_emb[:, 1], s=20, alpha=0.65, color=MODE_COLORS.get(transition_output["mode"], PALETTE["teal"]), label="predicted")
    ax_embed.scatter(tgt_emb[:, 0], tgt_emb[:, 1], s=20, alpha=0.45, color=PALETTE["accent"], label="target")
    ax_embed.set_title(f"Edge manifold view: {transition_output['edge']} ({transition_output['mode']})")
    ax_embed.set_xlabel("PCA 1")
    ax_embed.set_ylabel("PCA 2")
    ax_embed.legend(frameon=False)
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
    ax_history.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"sinkhorn: {heldout['sinkhorn']:.3f}",
                f"sinkhorn delta: {heldout['sinkhorn_delta']:.3f}",
                f"mmd: {heldout['mmd_rbf']:.3f}",
                f"auc: {heldout['classifier_auc']:.3f}",
                f"direction cosine: {heldout['direction_cosine']:.3f}",
                f"calibration error: {calibration['mean_abs_shift_error']:.3f}",
                f"context delta: {context_sensitivity.get('context_sensitivity_delta', float('nan')):.3f}",
            ]
        ),
        transform=ax_history.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EEF4FB", "edgecolor": "#B6C7DA"},
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
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF4EC", "edgecolor": "#E5C2AF"},
    )

    fig.suptitle("StageBridge v1 research frontend: edge-level biological insight", fontsize=16, fontweight="bold")
    return fig


def plot_mode_comparison_frontend(mode_table: pd.DataFrame, *, edge: str) -> Figure:
    """Render a matched mode-comparison ladder for one edge."""
    configure_research_style()
    table = mode_table.copy()
    order = [mode for mode in ["rna_only", "pooled", "deep_sets", "set_only", "graph_of_sets"] if mode in table["mode"].tolist()]
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
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF4EC", "edgecolor": "#E5C2AF"},
    )

    fig.suptitle("StageBridge v1 research frontend: latent-backend comparison", fontsize=16, fontweight="bold")
    return fig


__all__ = [
    "configure_research_style",
    "plot_biological_insight_frontend",
    "plot_context_frontend",
    "plot_latent_comparison_frontend",
    "plot_mode_comparison_frontend",
    "plot_reference_frontend",
    "plot_spatial_mapping_frontend",
    "plot_transition_frontend",
]
