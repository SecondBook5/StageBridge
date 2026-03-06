#!/usr/bin/env python
"""# path: scripts/qc_niche_tokens.py
Generate quick QC maps for niche-token features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niche_tokens_parquet",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/features/niche_tokens_full.parquet"),
    )
    parser.add_argument(
        "--spatial_h5ad",
        type=Path,
        default=Path("/mnt/e/StageBridge_data/processed/tangram/spatial_tangram_full.h5ad"),
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures/niche_tokens"))
    parser.add_argument("--sample_id", type=str, default="")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args()


def _choose_token_column(df: pd.DataFrame, base_name: str) -> str:
    preferred = [f"tok_smooth_{base_name}", f"tok_{base_name}"]
    for col in preferred:
        if col in df.columns:
            return col
    raise KeyError(f"Missing token column for '{base_name}'. Tried {preferred}.")


def _plot_map(x: np.ndarray, y: np.ndarray, values: np.ndarray, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sc = ax.scatter(x, y, c=values, s=1.5, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("value")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.niche_tokens_parquet.exists():
        raise FileNotFoundError(f"Missing niche tokens parquet: {args.niche_tokens_parquet}")
    if not args.spatial_h5ad.exists():
        raise FileNotFoundError(f"Missing spatial h5ad: {args.spatial_h5ad}")

    df = pd.read_parquet(args.niche_tokens_parquet)
    if "sample_id" not in df.columns:
        raise KeyError("niche tokens parquet missing required 'sample_id' column.")
    if "x" not in df.columns or "y" not in df.columns:
        raise KeyError("niche tokens parquet missing required coordinate columns 'x' and 'y'.")
    if "entropy" not in df.columns:
        raise KeyError("niche tokens parquet missing required 'entropy' column.")

    sample_id = args.sample_id.strip()
    if not sample_id:
        sample_id = str(df["sample_id"].astype(str).iloc[0])
    sub = df[df["sample_id"].astype(str) == sample_id].copy()
    if sub.empty:
        raise ValueError(f"No rows found for sample_id='{sample_id}' in {args.niche_tokens_parquet}.")

    # Validate sample exists in h5ad obs (cheap backed check).
    adata = anndata.read_h5ad(args.spatial_h5ad, backed="r")
    try:
        obs_sample = adata.obs["sample_id"].astype(str) if "sample_id" in adata.obs.columns else None
        if obs_sample is not None and sample_id not in set(obs_sample.unique().tolist()):
            raise ValueError(f"sample_id='{sample_id}' not found in {args.spatial_h5ad}.")
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()

    tok_at2 = _choose_token_column(sub, "AT2")
    tok_cil = _choose_token_column(sub, "Ciliated")

    x = sub["x"].to_numpy(dtype=np.float32)
    y = sub["y"].to_numpy(dtype=np.float32)

    out_dir = args.out_dir
    entropy_png = out_dir / f"entropy_map_{sample_id}.png"
    at2_png = out_dir / f"tok_AT2_map_{sample_id}.png"
    ciliated_png = out_dir / f"tok_Ciliated_map_{sample_id}.png"

    _plot_map(x, y, sub["entropy"].to_numpy(dtype=np.float32), f"Entropy ({sample_id})", entropy_png)
    _plot_map(x, y, sub[tok_at2].to_numpy(dtype=np.float32), f"{tok_at2} ({sample_id})", at2_png)
    _plot_map(x, y, sub[tok_cil].to_numpy(dtype=np.float32), f"{tok_cil} ({sample_id})", ciliated_png)

    summary = {
        "ok": True,
        "sample_id": sample_id,
        "n_spots": int(sub.shape[0]),
        "entropy_png": str(entropy_png),
        "tok_at2_png": str(at2_png),
        "tok_ciliated_png": str(ciliated_png),
        "min_max": {
            "entropy": [float(sub["entropy"].min()), float(sub["entropy"].max())],
            tok_at2: [float(sub[tok_at2].min()), float(sub[tok_at2].max())],
            tok_cil: [float(sub[tok_cil].min()), float(sub[tok_cil].max())],
        },
    }

    if args.json_output:
        print(json.dumps(summary))
    else:
        print(f"entropy_png={entropy_png}")
        print(f"tok_at2_png={at2_png}")
        print(f"tok_ciliated_png={ciliated_png}")
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
