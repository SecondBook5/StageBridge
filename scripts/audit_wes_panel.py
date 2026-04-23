#!/usr/bin/env python3
"""
Audit candidate WES features for StageBridge v1.

This script creates lesion-level and summary tables from cells.parquet so the
WES panel can be chosen deliberately rather than inherited accidentally from
the current parser output.

The goal is to answer four questions for each candidate feature:
1. Is the feature actually present in the canonical artifact?
2. Is the feature consistently populated at the lesion/patient level?
3. How prevalent is it overall and by stage?
4. Is it common enough to justify inclusion in a v1 progression-focused panel?

This script intentionally aggregates to one row per lesion/sample if possible,
or one row per donor/patient as a fallback, because the genomic variables are
not independent across cells. Repeating the same lesion-level mutation across
thousands of cells would distort prevalence summaries.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_CANDIDATES: List[str] = [
    "tmb",
    "kras_mut",
    "egfr_mut",
    "tp53_mut",
    "stk11_mut",
    "keap1_mut",
    "met_mut",
    "smad4_mut",
    "braf_mut",
    "nfe2l2_mut",
    "rb1_mut",
]


def choose_group_column(df: pd.DataFrame) -> str:
    """Choose the highest-priority grouping column available in the table."""
    priority_order: List[str] = ["lesion_id", "sample_id", "patient_id", "donor_id"]
    for column_name in priority_order:
        if column_name in df.columns:
            return column_name
    raise ValueError(
        "Could not find any grouping column among: lesion_id, sample_id, patient_id, donor_id."
    )


def choose_stage_column(df: pd.DataFrame) -> str | None:
    """Choose the stage column if present."""
    if "stage" in df.columns:
        return "stage"
    return None


def build_group_table(
    df: pd.DataFrame,
    group_col: str,
    stage_col: str | None,
    candidates: List[str],
) -> pd.DataFrame:
    """Build a lesion/sample/patient-level table by collapsing repeated per-cell values."""
    available_candidates: List[str] = [feature for feature in candidates if feature in df.columns]

    if not available_candidates:
        raise ValueError(
            "None of the requested candidate WES features were found in cells.parquet."
        )

    aggregation_map: dict[str, str] = {}

    if "tmb" in available_candidates:
        aggregation_map["tmb"] = "median"

    for feature in available_candidates:
        if feature == "tmb":
            continue
        aggregation_map[feature] = "max"

    if stage_col is not None:
        aggregation_map[stage_col] = "first"

    if "donor_id" in df.columns and group_col != "donor_id":
        aggregation_map["donor_id"] = "first"

    if "patient_id" in df.columns and group_col != "patient_id":
        aggregation_map["patient_id"] = "first"

    grouped_df: pd.DataFrame = (
        df.groupby(group_col, dropna=False)
        .agg(aggregation_map)
        .reset_index()
    )

    return grouped_df


def build_feature_summary(group_df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """Build a feature-level summary table with prevalence and missingness."""
    summary_rows: List[dict] = []

    for feature in candidates:
        if feature not in group_df.columns:
            summary_rows.append(
                {
                    "feature": feature,
                    "present_in_table": False,
                    "n_groups": len(group_df),
                    "n_non_missing": 0,
                    "missing_fraction": 1.0,
                    "positive_fraction": None,
                    "median_value": None,
                }
            )
            continue

        series = group_df[feature]
        n_non_missing: int = int(series.notna().sum())
        missing_fraction: float = float(1.0 - (n_non_missing / max(len(group_df), 1)))

        if feature == "tmb":
            positive_fraction = None
            median_value = float(series.dropna().median()) if n_non_missing > 0 else None
        else:
            positive_fraction = float((series.fillna(0) > 0).mean())
            median_value = float(series.dropna().median()) if n_non_missing > 0 else None

        summary_rows.append(
            {
                "feature": feature,
                "present_in_table": True,
                "n_groups": len(group_df),
                "n_non_missing": n_non_missing,
                "missing_fraction": missing_fraction,
                "positive_fraction": positive_fraction,
                "median_value": median_value,
            }
        )

    summary_df: pd.DataFrame = pd.DataFrame(summary_rows)
    return summary_df


def build_stage_summary(group_df: pd.DataFrame, stage_col: str, candidates: List[str]) -> pd.DataFrame:
    """Build stage-stratified prevalence summaries."""
    rows: List[dict] = []

    for stage_value, stage_df in group_df.groupby(stage_col, dropna=False):
        for feature in candidates:
            if feature not in stage_df.columns:
                continue

            if feature == "tmb":
                rows.append(
                    {
                        "stage": stage_value,
                        "feature": feature,
                        "n_groups": len(stage_df),
                        "positive_fraction": None,
                        "median_value": float(stage_df[feature].dropna().median()) if stage_df[feature].notna().any() else None,
                    }
                )
            else:
                rows.append(
                    {
                        "stage": stage_value,
                        "feature": feature,
                        "n_groups": len(stage_df),
                        "positive_fraction": float((stage_df[feature].fillna(0) > 0).mean()),
                        "median_value": float(stage_df[feature].dropna().median()) if stage_df[feature].notna().any() else None,
                    }
                )

    stage_summary_df: pd.DataFrame = pd.DataFrame(rows)
    return stage_summary_df


def main() -> None:
    """Run the WES panel audit end to end."""
    parser = argparse.ArgumentParser(description="Audit candidate WES features for StageBridge.")

    parser.add_argument(
        "--cells",
        required=True,
        help="Path to canonical cells.parquet",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for audit CSVs",
    )

    parser.add_argument(
        "--candidates",
        nargs="*",
        default=DEFAULT_CANDIDATES,
        help="Candidate WES features to audit",
    )

    args = parser.parse_args()

    cells_path = Path(args.cells)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not cells_path.exists():
        raise FileNotFoundError(f"cells.parquet not found: {cells_path}")

    df = pd.read_parquet(cells_path)

    group_col = choose_group_column(df)
    stage_col = choose_stage_column(df)

    group_df = build_group_table(
        df=df,
        group_col=group_col,
        stage_col=stage_col,
        candidates=args.candidates,
    )

    feature_summary_df = build_feature_summary(
        group_df=group_df,
        candidates=args.candidates,
    )

    group_df.to_csv(outdir / "wes_grouped_table.csv", index=False)
    feature_summary_df.to_csv(outdir / "wes_feature_summary.csv", index=False)

    if stage_col is not None:
        stage_summary_df = build_stage_summary(
            group_df=group_df,
            stage_col=stage_col,
            candidates=args.candidates,
        )
        stage_summary_df.to_csv(outdir / "wes_stage_summary.csv", index=False)

    print(f"Loaded cells.parquet: {cells_path}")
    print(f"Grouping level used: {group_col}")
    print(f"Stage column used: {stage_col}")
    print(f"Wrote: {outdir / 'wes_grouped_table.csv'}")
    print(f"Wrote: {outdir / 'wes_feature_summary.csv'}")
    if stage_col is not None:
        print(f"Wrote: {outdir / 'wes_stage_summary.csv'}")


if __name__ == "__main__":
    main()
