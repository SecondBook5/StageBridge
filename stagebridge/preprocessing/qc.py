"""QC metrics computation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def compute_qc_metrics(
    adata: "ad.AnnData",
) -> pd.DataFrame:
    """Compute QC metrics per cell.

    Args:
        adata: AnnData object

    Returns:
        DataFrame with QC metrics
    """
    import scanpy as sc

    # Mitochondrial genes
    if "pct_counts_mt" not in adata.obs.columns:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

    # Ribosomal genes
    if "pct_counts_ribo" not in adata.obs.columns:
        adata.var["ribo"] = adata.var_names.str.match("^RP[SL]")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["ribo"], percent_top=None, log1p=False, inplace=True
        )

    qc_cols = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]
    available_cols = [c for c in qc_cols if c in adata.obs.columns]

    qc_df = adata.obs[available_cols].copy()
    qc_df["cell_id"] = adata.obs.index

    if "stage" in adata.obs.columns:
        qc_df["stage"] = adata.obs["stage"].values

    return qc_df


def run_qc(
    h5ad_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Run QC metrics pipeline.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory

    Returns:
        Dict of output file paths
    """
    import scanpy as sc

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells")

    outputs = {}

    print("Computing QC metrics...")
    qc_df = compute_qc_metrics(adata)

    out_path = output_dir / "snrna_qc_metrics.parquet"
    qc_df.to_parquet(out_path)
    outputs["qc_metrics"] = out_path
    print(f"  Saved {out_path}")

    # QC by stage
    if "stage" in qc_df.columns:
        qc_cols = ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"]
        available_cols = [c for c in qc_cols if c in qc_df.columns]
        qc_by_stage = qc_df.groupby("stage")[available_cols].agg(["mean", "median", "std"])
        out_path = output_dir / "qc_by_stage.parquet"
        qc_by_stage.to_parquet(out_path)
        outputs["qc_by_stage"] = out_path
        print(f"  Saved {out_path}")

    print("QC complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute QC metrics")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    run_qc(args.h5ad, args.output)
