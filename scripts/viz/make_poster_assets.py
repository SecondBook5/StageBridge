#!/usr/bin/env python
"""Generate standardized poster assets from StageBridge outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stagebridge.viz.curves import build_metrics_dataframe, plot_benchmark_bars
from stagebridge.viz.spatial_plots import (
    plot_method_schematic,
    plot_metric_heatmap,
    plot_transition_trajectory,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_abstract_support(table_dir: Path) -> None:
    text = [
        "Background: Cross-sectional stage progression in lung disease is unpaired and hard to identify.",
        "Method: StageBridge uses Set Transformer context encoding with OT-coupled flow matching in HLCA-aligned latent space.",
        "Results: Donor-held-out benchmarking compares StageBridge against non-attention baselines and ablations.",
        "Biology: Stage-to-stage trajectory metrics support biologically plausible progression operators for atlas-aligned analysis.",
        "Impact: A reproducible, transformer-first framework for Human Cell Atlas-ready progression modeling.",
    ]
    out = table_dir / "poster_abstract_structure.txt"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(text) + "\n")


def _fallback_panel_c_from_metrics(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Fallback panel C when explicit eval-per-transition JSON is unavailable."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    metric_cols = [c for c in ["sinkhorn_mean", "mmd_rbf_mean", "jsd_composition_mean"] if c in metrics_df.columns]
    if not metric_cols:
        metric_cols = [c for c in metrics_df.columns if c.endswith("_mean")]

    x = np.arange(metrics_df.shape[0])
    for col in metric_cols[:4]:
        ax.plot(x, metrics_df[col].astype(float).values, marker="o", label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["label"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Panel C: Trajectory Fidelity Profile (Model-Level)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build poster assets from StageBridge benchmark files")
    parser.add_argument("metrics_json", type=Path, help="Path to metrics_<run>.json")
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional eval_<run>.json for transition-level panel C",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for poster figures",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("outputs/tables"),
        help="Directory for summary tables",
    )
    args = parser.parse_args()

    metrics_payload = _load_json(args.metrics_json)
    metrics_df = build_metrics_dataframe(metrics_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    # Panel A: method schematic.
    plot_method_schematic(args.output_dir / "poster_panel_a_method_schematic.png")

    # Panel B: benchmark bars.
    plot_benchmark_bars(
        metrics_df,
        output_path=args.output_dir / "poster_panel_b_benchmark.png",
        metric_col="sinkhorn_mean" if "sinkhorn_mean" in metrics_df.columns else metrics_df.columns[-1],
        metric_std_col="sinkhorn_std" if "sinkhorn_std" in metrics_df.columns else "sinkhorn_std",
        title="Panel B: Donor-held-out Benchmark (Primary Metric)",
    )

    # Panel C: transition-level trajectory if eval JSON is available.
    if args.eval_json is not None and args.eval_json.exists():
        eval_payload = _load_json(args.eval_json)
        eval_df = pd.DataFrame(eval_payload.get("results", []))
        if not eval_df.empty and {"stage_src", "stage_tgt", "sinkhorn", "mmd_rbf", "classifier_auc"}.issubset(eval_df.columns):
            plot_transition_trajectory(eval_df, args.output_dir / "poster_panel_c_trajectory.png")
        else:
            _fallback_panel_c_from_metrics(metrics_df, args.output_dir / "poster_panel_c_trajectory.png")
    else:
        _fallback_panel_c_from_metrics(metrics_df, args.output_dir / "poster_panel_c_trajectory.png")

    # Panel D: metric heatmap (interpretability/comparison summary).
    plot_metric_heatmap(metrics_df, args.output_dir / "poster_panel_d_metric_heatmap.png")

    # Export table and abstract support text.
    table_path = args.table_dir / "poster_table_1_metrics.csv"
    metrics_df.to_csv(table_path, index=False)
    _write_abstract_support(args.table_dir)

    print(
        json.dumps(
            {
                "ok": True,
                "figures_dir": str(args.output_dir),
                "tables_dir": str(args.table_dir),
                "rows": int(metrics_df.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
