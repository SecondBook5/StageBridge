"""Differential expression analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def run_de_stage_vs_rest(
    adata: "ad.AnnData",
    stage: str,
    max_rest_cells: int = 50000,
    n_genes: int = 500,
) -> pd.DataFrame:
    """Run DE for one stage vs rest.

    Args:
        adata: AnnData with 'stage' column
        stage: Stage to compare
        max_rest_cells: Max cells to sample from 'rest' group
        n_genes: Number of top genes to return

    Returns:
        DataFrame with DE results
    """
    import scanpy as sc

    stage_mask = adata.obs["stage"] == stage
    n_stage = stage_mask.sum()

    rest_mask = ~stage_mask
    n_rest = min(rest_mask.sum(), max_rest_cells)

    rest_idx = np.random.choice(np.where(rest_mask)[0], size=n_rest, replace=False)
    stage_idx = np.where(stage_mask)[0]

    subset_idx = np.concatenate([stage_idx, rest_idx])
    adata_sub = adata[subset_idx].copy()
    adata_sub.obs["_group"] = (adata_sub.obs["stage"] == stage).map(
        {True: stage, False: "rest"}
    )

    sc.tl.rank_genes_groups(
        adata_sub,
        groupby="_group",
        groups=[stage],
        reference="rest",
        method="wilcoxon",
        n_genes=n_genes,
    )

    return sc.get.rank_genes_groups_df(adata_sub, group=stage)


def run_de_all_stages(
    h5ad_path: str | Path,
    output_dir: str | Path,
    max_rest_cells: int = 50000,
    n_genes: int = 500,
) -> dict[str, Path]:
    """Run DE for all stages vs rest.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        max_rest_cells: Max cells for 'rest' group
        n_genes: Number of genes per stage

    Returns:
        Dict mapping stage to output path
    """
    import scanpy as sc
    from tqdm import tqdm

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    stages = adata.obs["stage"].unique().tolist()
    print(f"  Stages: {stages}")

    outputs = {}
    for stage in tqdm(stages, desc="DE by stage"):
        out_path = output_dir / f"de_stage_{stage}.parquet"

        if out_path.exists():
            tqdm.write(f"  {stage}: SKIP (exists)")
            outputs[stage] = out_path
            continue

        stage_mask = adata.obs["stage"] == stage
        n_stage = stage_mask.sum()
        n_rest = min((~stage_mask).sum(), max_rest_cells)
        tqdm.write(f"  {stage}: {n_stage} vs {n_rest} cells")

        df = run_de_stage_vs_rest(adata, stage, max_rest_cells, n_genes)
        df.to_parquet(out_path)
        outputs[stage] = out_path
        tqdm.write(f"  {stage}: saved {len(df)} genes")

    print("DE complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run differential expression")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-rest", type=int, default=50000, help="Max rest cells")
    parser.add_argument("--n-genes", type=int, default=500, help="Genes per stage")
    args = parser.parse_args()

    run_de_all_stages(args.h5ad, args.output, args.max_rest, args.n_genes)
