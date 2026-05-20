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
    n_genes: int | None = None,
) -> pd.DataFrame:
    """Run DE for one stage vs rest.

    Args:
        adata: AnnData with 'stage' column
        stage: Stage to compare
        max_rest_cells: Max cells to sample from 'rest' group
        n_genes: Number of top genes to return (None = all genes for volcano plots)

    Returns:
        DataFrame with DE results (both up and downregulated)
    """
    import scanpy as sc

    stage_mask = adata.obs["stage"] == stage

    rest_mask = ~stage_mask
    n_rest = min(rest_mask.sum(), max_rest_cells)

    rest_idx = np.random.choice(np.where(rest_mask)[0], size=n_rest, replace=False)
    stage_idx = np.where(stage_mask)[0]

    subset_idx = np.concatenate([stage_idx, rest_idx])
    adata_sub = adata[subset_idx].copy()
    adata_sub.obs["_group"] = (adata_sub.obs["stage"] == stage).map(
        {True: stage, False: "rest"}
    )

    # Use all genes if n_genes not specified (for proper volcano plots)
    if n_genes is None:
        n_genes = adata_sub.n_vars

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
    n_genes: int | None = None,
) -> dict[str, Path]:
    """Run DE for all stages vs rest.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Output directory
        max_rest_cells: Max cells for 'rest' group
        n_genes: Number of genes per stage (None = all genes)

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


def run_pseudobulk_deseq2(
    adata: "ad.AnnData",
    stage: str,
    reference: str | None = None,
    sample_col: str = "donor_id",
    min_cells_per_sample: int = 10,
) -> pd.DataFrame:
    """Run pseudobulk DESeq2 for one stage vs reference.

    Aggregates cells by sample-condition combination (donor + target/reference),
    then runs DESeq2 on pseudobulk counts. This properly handles donors
    that have cells in multiple stages.

    Args:
        adata: AnnData with raw counts in .X and 'stage' column
        stage: Stage to compare (target)
        reference: Reference stage (None = all other stages as "rest")
        sample_col: Column for sample/donor grouping
        min_cells_per_sample: Minimum cells to include a pseudobulk sample

    Returns:
        DataFrame with DESeq2 results (log2FoldChange, pvalue, padj)
        Positive log2FC = higher in target stage
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        raise ImportError("Install pydeseq2: pip install pydeseq2")

    # Filter to relevant stages if pairwise comparison
    if reference is not None:
        mask = adata.obs["stage"].isin([stage, reference])
        adata = adata[mask].copy()
        adata.obs["_condition"] = adata.obs["stage"].map(
            {stage: "target", reference: "rest"}
        )
    else:
        # Stage vs all rest
        adata.obs["_condition"] = (adata.obs["stage"] == stage).map(
            {True: "target", False: "rest"}
        )

    # Create pseudobulk per donor-condition combination
    # This handles donors with cells in both target and rest
    pseudobulk_data = []
    sample_meta = []

    for (donor, condition), group_df in adata.obs.groupby([sample_col, "_condition"]):
        n_cells = len(group_df)
        if n_cells < min_cells_per_sample:
            continue

        # Get cells for this donor-condition
        cell_idx = group_df.index
        subset = adata[cell_idx]

        # Sum counts across cells
        if hasattr(subset.X, "toarray"):
            counts = np.array(subset.X.toarray().sum(axis=0)).flatten()
        else:
            counts = np.array(subset.X.sum(axis=0)).flatten()

        # Create unique sample ID for this donor-condition pair
        sample_id = f"{donor}_{condition}"
        pseudobulk_data.append(counts)
        sample_meta.append({
            "sample": sample_id,
            "donor": donor,
            "condition": condition,
            "n_cells": n_cells,
        })

    if len(pseudobulk_data) == 0:
        raise ValueError(f"No valid pseudobulk samples with >= {min_cells_per_sample} cells")

    # Create count matrix and metadata
    count_matrix = pd.DataFrame(
        np.array(pseudobulk_data),
        index=[m["sample"] for m in sample_meta],
        columns=adata.var_names,
    ).astype(int)

    metadata = pd.DataFrame(sample_meta).set_index("sample")

    # Check we have both conditions
    condition_counts = metadata["condition"].value_counts().to_dict()
    print(f"  Pseudobulk: {len(count_matrix)} samples, {count_matrix.shape[1]} genes")
    print(f"  Conditions: {condition_counts}")

    if "target" not in condition_counts or "rest" not in condition_counts:
        raise ValueError(f"Need both target and rest conditions, got: {condition_counts}")

    if condition_counts["target"] < 2 or condition_counts["rest"] < 2:
        raise ValueError(f"Need at least 2 samples per condition, got: {condition_counts}")

    # Filter low-count genes
    gene_sums = count_matrix.sum(axis=0)
    keep_genes = gene_sums >= 10
    count_matrix = count_matrix.loc[:, keep_genes]
    print(f"  After filtering: {count_matrix.shape[1]} genes")

    # Run DESeq2
    dds = DeseqDataSet(
        counts=count_matrix,
        metadata=metadata[["condition"]],  # Only pass condition column
        design_factors="condition",
        refit_cooks=True,
    )

    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=["condition", "target", "rest"])
    stat_res.summary()

    results = stat_res.results_df.copy()
    results["gene"] = results.index
    results = results.reset_index(drop=True)

    return results[["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]


def run_pseudobulk_all_stages(
    h5ad_path: str | Path,
    output_dir: str | Path,
    sample_col: str = "donor_id",
    min_cells_per_sample: int = 10,
) -> dict[str, Path]:
    """Run pseudobulk DESeq2 for all stages vs rest.

    Args:
        h5ad_path: Path to h5ad with raw counts
        output_dir: Output directory
        sample_col: Sample/donor column
        min_cells_per_sample: Min cells per sample

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
    for stage in tqdm(stages, desc="Pseudobulk DE"):
        out_path = output_dir / f"de_pseudobulk_{stage}.parquet"

        if out_path.exists():
            tqdm.write(f"  {stage}: SKIP (exists)")
            outputs[stage] = out_path
            continue

        tqdm.write(f"  {stage}: running DESeq2...")

        try:
            df = run_pseudobulk_deseq2(adata, stage, None, sample_col, min_cells_per_sample)
            df.to_parquet(out_path)
            outputs[stage] = out_path
            tqdm.write(f"  {stage}: saved {len(df)} genes")
        except Exception as e:
            tqdm.write(f"  {stage}: FAILED - {e}")

    print("Pseudobulk DE complete")
    return outputs


def run_pseudobulk_pairwise(
    h5ad_path: str | Path,
    output_dir: str | Path,
    sample_col: str = "donor_id",
    min_cells_per_sample: int = 10,
) -> dict[str, Path]:
    """Run pseudobulk DESeq2 for all pairwise stage comparisons.

    Args:
        h5ad_path: Path to h5ad with raw counts
        output_dir: Output directory
        sample_col: Sample/donor column
        min_cells_per_sample: Min cells per sample

    Returns:
        Dict mapping comparison to output path
    """
    import scanpy as sc
    from itertools import combinations
    from tqdm import tqdm

    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    stages = adata.obs["stage"].unique().tolist()
    print(f"  Stages: {stages}")

    pairs = list(combinations(stages, 2))
    print(f"  Pairwise comparisons: {len(pairs)}")

    outputs = {}
    all_results = []

    for stage1, stage2 in tqdm(pairs, desc="Pairwise DE"):
        comparison = f"{stage1}_vs_{stage2}"
        out_path = output_dir / f"de_{comparison}.parquet"

        if out_path.exists():
            tqdm.write(f"  {comparison}: SKIP (exists)")
            outputs[comparison] = out_path
            continue

        tqdm.write(f"  {comparison}: running DESeq2...")

        try:
            df = run_pseudobulk_deseq2(adata, stage1, stage2, sample_col, min_cells_per_sample)
            df["comparison"] = comparison
            df.to_parquet(out_path)
            outputs[comparison] = out_path
            all_results.append(df)
            tqdm.write(f"  {comparison}: saved {len(df)} genes")
        except Exception as e:
            tqdm.write(f"  {comparison}: FAILED - {e}")

    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = output_dir / "de_all_pairwise.parquet"
        combined.to_parquet(combined_path)
        outputs["combined"] = combined_path
        print(f"Saved combined: {len(combined)} total gene-comparison pairs")

    print("Pairwise DE complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run differential expression")
    parser.add_argument("--h5ad", required=True, help="Path to h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", choices=["wilcoxon", "pseudobulk", "pairwise"], default="wilcoxon",
                        help="DE method: wilcoxon (cell-level), pseudobulk (stage vs rest), pairwise (all stage pairs)")
    parser.add_argument("--max-rest", type=int, default=50000, help="Max rest cells (wilcoxon)")
    parser.add_argument("--n-genes", type=int, default=None, help="Genes per stage (wilcoxon, omit for all)")
    parser.add_argument("--sample-col", default="donor_id", help="Sample column (pseudobulk/pairwise)")
    parser.add_argument("--min-cells", type=int, default=10, help="Min cells per sample (pseudobulk/pairwise)")
    args = parser.parse_args()

    if args.method == "wilcoxon":
        run_de_all_stages(args.h5ad, args.output, args.max_rest, args.n_genes)
    elif args.method == "pairwise":
        run_pseudobulk_pairwise(args.h5ad, args.output, args.sample_col, args.min_cells)
    else:
        run_pseudobulk_all_stages(args.h5ad, args.output, args.sample_col, args.min_cells)
