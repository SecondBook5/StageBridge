"""GSEA pathway enrichment analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


# Canonical pathway gene sets
HALLMARK_PATHWAYS = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "EMT",
    "HALLMARK_INFLAMMATORY_RESPONSE": "Inflammation",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "IL6_STAT3",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": "TNFa_NFkB",
    "HALLMARK_HYPOXIA": "Hypoxia",
    "HALLMARK_GLYCOLYSIS": "Glycolysis",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "OXPHOS",
    "HALLMARK_P53_PATHWAY": "p53",
    "HALLMARK_APOPTOSIS": "Apoptosis",
    "HALLMARK_G2M_CHECKPOINT": "G2M",
    "HALLMARK_E2F_TARGETS": "E2F",
    "HALLMARK_MYC_TARGETS_V1": "MYC_V1",
}


def run_gsea_prerank(
    gene_stats: pd.Series,
    gene_sets: str = "MSigDB_Hallmark_2020",
    organism: str = "human",
) -> pd.DataFrame:
    """Run GSEA prerank analysis.

    Args:
        gene_stats: Series with gene names as index, stat values as values
        gene_sets: Gene set library name
        organism: Organism for gene sets

    Returns:
        DataFrame with GSEA results
    """
    import gseapy as gp

    result = gp.prerank(
        rnk=gene_stats,
        gene_sets=gene_sets,
        organism=organism,
        permutation_num=100,
        outdir=None,
        seed=42,
        verbose=False,
    )

    return result.res2d


def run_gsea_from_de(
    de_path: str | Path,
    stat_col: str = "scores",
    gene_col: str = "names",
) -> pd.DataFrame:
    """Run GSEA from DE results file.

    Args:
        de_path: Path to DE parquet file
        stat_col: Column with statistics
        gene_col: Column with gene names

    Returns:
        DataFrame with GSEA results
    """
    de_df = pd.read_parquet(de_path)
    gene_stats = pd.Series(de_df[stat_col].values, index=de_df[gene_col].values)
    return run_gsea_prerank(gene_stats)


def run_gsea_all_stages(
    de_dir: str | Path,
    output_dir: str | Path,
    stages: list[str] | None = None,
) -> dict[str, Path]:
    """Run GSEA for all stage DE results.

    Args:
        de_dir: Directory with DE parquet files
        output_dir: Output directory
        stages: List of stages (auto-detect if None)

    Returns:
        Dict mapping stage to output path
    """
    de_dir = Path(de_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find DE files
    if stages is None:
        de_files = list(de_dir.glob("de_stage_*.parquet"))
        stages = [f.stem.replace("de_stage_", "") for f in de_files]

    outputs = {}
    for stage in stages:
        de_path = de_dir / f"de_stage_{stage}.parquet"
        if not de_path.exists():
            print(f"  {stage}: SKIP (no DE file)")
            continue

        print(f"  {stage}...")
        try:
            gsea_df = run_gsea_from_de(de_path)
            out_path = output_dir / f"gsea_{stage}.parquet"
            gsea_df.to_parquet(out_path)
            outputs[stage] = out_path
        except Exception as e:
            print(f"    FAILED: {e}")

    return outputs


def run_gsea(
    de_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Run full GSEA pipeline.

    Args:
        de_dir: Directory with DE parquet files
        output_dir: Output directory

    Returns:
        Dict of output file paths
    """
    de_dir = Path(de_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running GSEA...")
    outputs = run_gsea_all_stages(de_dir, output_dir)

    # Summary table
    if outputs:
        print("Creating summary...")
        summaries = []
        for stage, path in outputs.items():
            df = pd.read_parquet(path)
            df["stage"] = stage
            summaries.append(df)

        summary_df = pd.concat(summaries, ignore_index=True)
        out_path = output_dir / "gsea_summary.parquet"
        summary_df.to_parquet(out_path)
        outputs["summary"] = out_path
        print(f"  Saved {out_path}")

    print("GSEA complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSEA pathway enrichment")
    parser.add_argument("--de-dir", required=True, help="Directory with DE files")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    run_gsea(args.de_dir, args.output)
