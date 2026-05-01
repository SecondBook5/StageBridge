"""Summary statistics for cells and neighborhoods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_celltype_proportions(
    df: pd.DataFrame,
    stage_col: str = "stage",
    celltype_col: str = "cell_type",
) -> pd.DataFrame:
    """Compute cell type proportions by stage.

    Args:
        df: DataFrame with stage and cell_type columns
        stage_col: Column name for stage
        celltype_col: Column name for cell type

    Returns:
        DataFrame with proportions (stage x cell_type)
    """
    return pd.crosstab(df[stage_col], df[celltype_col], normalize="index")


def compute_stage_summary(
    df: pd.DataFrame,
    stage_col: str = "stage",
) -> pd.DataFrame:
    """Compute summary statistics by stage.

    Args:
        df: DataFrame with stage column
        stage_col: Column name for stage

    Returns:
        DataFrame with n_cells per stage
    """
    return df.groupby(stage_col).size().to_frame("n_cells")


def run_summary_stats(
    neighborhoods_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Run full summary statistics pipeline.

    Args:
        neighborhoods_path: Path to neighborhoods.parquet
        output_dir: Output directory

    Returns:
        Dict of output file paths
    """
    neighborhoods_path = Path(neighborhoods_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {neighborhoods_path}...")
    df = pd.read_parquet(neighborhoods_path)
    print(f"  {len(df)} cells")

    outputs = {}

    # Cell type proportions by stage
    celltype_col = "cell_type_luca" if "cell_type_luca" in df.columns else "cell_type"
    if "stage" in df.columns and celltype_col in df.columns:
        print("Computing cell type proportions...")
        props = compute_celltype_proportions(df, "stage", celltype_col)
        out_path = output_dir / "celltype_proportions_by_stage.parquet"
        props.to_parquet(out_path)
        outputs["celltype_proportions"] = out_path
        print(f"  Saved {out_path}")

    # Stage summary
    if "stage" in df.columns:
        print("Computing stage summary...")
        summary = compute_stage_summary(df, "stage")
        out_path = output_dir / "stage_summary.parquet"
        summary.to_parquet(out_path)
        outputs["stage_summary"] = out_path
        print(f"  Saved {out_path}")

    # Donor summary
    donor_col = "donor_id" if "donor_id" in df.columns else "sample_id" if "sample_id" in df.columns else None
    if donor_col and "stage" in df.columns:
        print("Computing donor summary...")
        donor_stage = df.groupby([donor_col, "stage"]).size().unstack(fill_value=0)
        out_path = output_dir / "donor_stage_counts.parquet"
        donor_stage.to_parquet(out_path)
        outputs["donor_stage"] = out_path
        print(f"  Saved {out_path}")

    # Neighborhood size distribution
    if "n_neighbors" in df.columns:
        print("Computing neighborhood stats...")
        nhood_stats = df.groupby("stage")["n_neighbors"].agg(["mean", "median", "std", "min", "max"])
        out_path = output_dir / "neighborhood_size_by_stage.parquet"
        nhood_stats.to_parquet(out_path)
        outputs["nhood_size"] = out_path
        print(f"  Saved {out_path}")

    print("Summary stats complete")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute summary statistics")
    parser.add_argument("--neighborhoods", required=True, help="Path to neighborhoods.parquet")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    run_summary_stats(args.neighborhoods, args.output)
