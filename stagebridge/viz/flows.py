"""Macro-flow visualization helpers (cluster-level Sankey)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def compute_macroflow_matrix(
    x_src: np.ndarray,
    x_tgt_pred: np.ndarray,
    n_clusters: int = 9,
    random_state: int = 0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Cluster source/predicted cells and compute a source->target flow matrix."""
    x0 = np.asarray(x_src, dtype=np.float32)
    x1 = np.asarray(x_tgt_pred, dtype=np.float32)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1]:
        raise ValueError("x_src and x_tgt_pred must be 2D arrays with matching feature dimensions.")
    if x0.shape[0] != x1.shape[0]:
        raise ValueError("x_src and x_tgt_pred must have equal row counts.")

    k = int(max(2, min(n_clusters, x0.shape[0])))
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=min(4096, x0.shape[0]))
    km.fit(np.vstack([x0, x1]))

    src_labels = km.predict(x0)
    tgt_labels = km.predict(x1)

    flow = np.zeros((k, k), dtype=np.float64)
    np.add.at(flow, (src_labels, tgt_labels), 1.0)
    flow = flow / max(1.0, flow.sum())

    src_names = [f"src_C{i}" for i in range(k)]
    tgt_names = [f"pred_C{i}" for i in range(k)]
    return flow.astype(np.float32, copy=False), src_names, tgt_names


def plot_macroflow_sankey(
    flow_matrix: np.ndarray,
    source_labels: list[str],
    target_labels: list[str],
    output_path: Path,
    title: str = "Macro Coupling Sankey",
) -> None:
    """Write a Sankey diagram PNG from a flow matrix."""
    flow = np.asarray(flow_matrix, dtype=np.float32)
    if flow.ndim != 2:
        raise ValueError("flow_matrix must be 2D.")
    if flow.shape != (len(source_labels), len(target_labels)):
        raise ValueError("flow_matrix shape must match source/target label lengths.")

    n_src = len(source_labels)
    labels = [*source_labels, *target_labels]

    src_idx: list[int] = []
    tgt_idx: list[int] = []
    values: list[float] = []
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            val = float(flow[i, j])
            if val <= 0:
                continue
            src_idx.append(i)
            tgt_idx.append(n_src + j)
            values.append(val)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=10,
                        thickness=16,
                        line=dict(color="rgba(60,60,60,0.6)", width=0.5),
                        label=labels,
                        color=["#0E7490"] * n_src + ["#FB923C"] * len(target_labels),
                    ),
                    link=dict(
                        source=src_idx,
                        target=tgt_idx,
                        value=values,
                        color="rgba(59,130,246,0.35)",
                    ),
                )
            ]
        )
        fig.update_layout(title_text=title, font_size=10, width=980, height=620)
        fig.write_image(str(output_path))
    except Exception:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        im = ax.imshow(flow, cmap="viridis", interpolation="nearest")
        ax.set_title(f"{title} (heatmap fallback)")
        ax.set_xlabel("Predicted macro cluster")
        ax.set_ylabel("Source macro cluster")
        ax.set_xticks(np.arange(len(target_labels)))
        ax.set_yticks(np.arange(len(source_labels)))
        ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(source_labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
