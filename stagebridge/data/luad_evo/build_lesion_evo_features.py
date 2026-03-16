"""Build one lesion-level evolution feature row per lesion for EA-MIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stagebridge.labels.cohort_manifest import build_cleaned_cohort_manifest

from .eamist_common import default_reports_tables_dir, utc_now_iso, write_json


def _resolve_support_path(explicit: Path | None, default_name: str) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    candidate = default_reports_tables_dir() / default_name
    return candidate.resolve() if candidate.exists() else None


def _load_manifest(wes_path: Path, cleaned_manifest: Path | None) -> pd.DataFrame:
    if cleaned_manifest is not None and cleaned_manifest.exists():
        manifest = pd.read_csv(cleaned_manifest)
    else:
        manifest = build_cleaned_cohort_manifest({"data": {"wes_features_path": str(wes_path)}})[
            "cleaned_manifest"
        ]
    if manifest.empty:
        raise ValueError(
            "Cleaned lesion manifest was empty; cannot build lesion-level evolution features."
        )
    if manifest["lesion_id"].duplicated().any():
        duplicates = (
            manifest.loc[manifest["lesion_id"].duplicated(keep=False), "lesion_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"Duplicate lesion ids detected in manifest: {duplicates[:10]}")
    return manifest.loc[:, ["lesion_id", "sample_id", "patient_id", "donor_id", "stage"]].copy()


def _merge_one(
    base: pd.DataFrame, path: Path | None, *, key: str, columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    if path is None or not path.exists():
        return base, []
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = [key, *columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        return base, []
    merge_frame = frame.loc[:, required].copy()
    if merge_frame[key].duplicated().any():
        raise ValueError(f"Support table {path} contains duplicate '{key}' rows.")
    merged = base.merge(merge_frame, on=key, how="left", validate="one_to_one")
    return merged, [column for column in columns if column in merge_frame.columns]


def run(
    wes_path: Path,
    out_path: Path,
    *,
    cleaned_manifest: Path | None = None,
    refined_labels: Path | None = None,
    cna_summary: Path | None = None,
    clone_summary: Path | None = None,
    phylogeny_summary: Path | None = None,
) -> dict[str, object]:
    wes_path = Path(wes_path).resolve()
    out_path = Path(out_path).resolve()
    if not wes_path.exists():
        raise FileNotFoundError(f"Missing WES feature parquet: {wes_path}")

    manifest = _load_manifest(wes_path, cleaned_manifest)
    wes = pd.read_parquet(wes_path).copy()
    if wes.empty:
        raise ValueError("WES feature parquet was empty.")
    required_wes = {"patient_id", "stage"}
    missing_wes = required_wes.difference(wes.columns)
    if missing_wes:
        raise KeyError(f"WES feature parquet is missing required columns: {sorted(missing_wes)}")

    merged = manifest.merge(wes, on=["patient_id", "stage"], how="left", validate="many_to_one")
    mutation_cols = [
        column for column in wes.columns if column not in {"patient_id", "stage", "tmb"}
    ]
    included_features: list[str] = []
    if "tmb" in merged.columns:
        merged["evo_tmb"] = pd.to_numeric(merged["tmb"], errors="coerce").astype(float)
        included_features.append("evo_tmb")
    if mutation_cols:
        merged["evo_driver_burden"] = (
            merged.loc[:, mutation_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        ).astype(float)
        included_features.append("evo_driver_burden")
        for column in mutation_cols:
            merged[f"evo_{column}"] = pd.to_numeric(merged[column], errors="coerce").astype(float)
            included_features.append(f"evo_{column}")

    support_specs = [
        (
            _resolve_support_path(cna_summary, "lesion_cna_summary.csv"),
            [
                "purity",
                "ploidy",
                "fraction_genome_altered",
                "cna_burden",
                "num_focal_events",
                "num_arm_level_events",
                "allele_specific_imbalance",
            ],
        ),
        (
            _resolve_support_path(clone_summary, "lesion_clone_summary.csv"),
            [
                "num_clonal_clusters",
                "dominant_clone_fraction",
                "subclonal_entropy",
                "shared_cluster_count_with_later_lesions",
                "private_cluster_count",
                "driver_cluster_count",
            ],
        ),
        (
            _resolve_support_path(phylogeny_summary, "lesion_phylogeny_summary.csv"),
            [
                "trunk_mutation_burden",
                "branch_count",
                "branch_length_mean",
                "clone_sharing_score",
                "descendant_sharing_score",
                "trunk_membership_score",
                "branch_specificity_score",
                "evidence_of_progression_link",
            ],
        ),
        (
            _resolve_support_path(refined_labels, "lesion_refined_labels.csv"),
            ["progression_risk_score"],
        ),
    ]
    loaded_sources: dict[str, str] = {}
    for path, columns in support_specs:
        if path is None:
            continue
        merged, loaded = _merge_one(merged, path, key="lesion_id", columns=columns)
        if not loaded:
            continue
        stem = path.stem
        loaded_sources[stem] = str(path)
        for column in loaded:
            merged[f"evo_{column}"] = pd.to_numeric(merged[column], errors="coerce").astype(float)
            included_features.append(f"evo_{column}")

    if "evo_subclonal_entropy" in merged.columns:
        merged["evo_clonal_diversity"] = merged["evo_subclonal_entropy"]
        included_features.append("evo_clonal_diversity")
    if "evo_branch_count" in merged.columns:
        merged["evo_branch_complexity"] = merged["evo_branch_count"]
        included_features.append("evo_branch_complexity")
    if {"evo_clone_sharing_score", "evo_trunk_membership_score"}.issubset(merged.columns):
        merged["evo_trunk_shared_clone_score"] = merged[
            ["evo_clone_sharing_score", "evo_trunk_membership_score"]
        ].mean(axis=1)
        included_features.append("evo_trunk_shared_clone_score")

    included_features = list(dict.fromkeys(included_features))
    output_columns = [
        "lesion_id",
        "sample_id",
        "patient_id",
        "donor_id",
        "stage",
        *included_features,
    ]
    output = merged.loc[:, output_columns].copy()
    if output.empty:
        raise ValueError("Lesion evolution feature table was empty.")
    if not included_features:
        raise ValueError(
            "No lesion evolution features could be assembled from the available inputs."
        )
    if output[included_features].isna().all(axis=1).all():
        raise ValueError("Every lesion-level evolution feature row was empty after assembly.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)
    audit = {
        "created_at_utc": utc_now_iso(),
        "wes_path": str(wes_path),
        "manifest_rows": int(manifest.shape[0]),
        "out_path": str(out_path),
        "included_features": included_features,
        "num_rows": int(output.shape[0]),
        "num_empty_rows": int(output[included_features].isna().all(axis=1).sum()),
        "support_sources": loaded_sources,
    }
    write_json(out_path.parent / f"{out_path.stem}.audit.json", audit)
    return audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wes", type=Path, required=True, help="Path to wes_features.parquet")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output parquet path for lesion evolution features"
    )
    parser.add_argument(
        "--cleaned-manifest", type=Path, default=None, help="Optional cleaned cohort manifest CSV"
    )
    parser.add_argument(
        "--refined-labels", type=Path, default=None, help="Optional lesion_refined_labels.csv"
    )
    parser.add_argument(
        "--cna-summary", type=Path, default=None, help="Optional lesion_cna_summary.csv"
    )
    parser.add_argument(
        "--clone-summary", type=Path, default=None, help="Optional lesion_clone_summary.csv"
    )
    parser.add_argument(
        "--phylogeny-summary",
        type=Path,
        default=None,
        help="Optional lesion_phylogeny_summary.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run(
        args.wes,
        args.out,
        cleaned_manifest=args.cleaned_manifest,
        refined_labels=args.refined_labels,
        cna_summary=args.cna_summary,
        clone_summary=args.clone_summary,
        phylogeny_summary=args.phylogeny_summary,
    )
    print(f"built lesion evolution features: {audit}")


if __name__ == "__main__":
    main()
