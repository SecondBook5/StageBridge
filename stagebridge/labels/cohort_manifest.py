"""Cohort normalization and manifest building for label repair."""
from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.data.luad_evo import (
    build_lesion_label_table,
    load_luad_evo_spatial_mapping,
    load_luad_evo_wes_features,
)
from stagebridge.labels.common_schema import (
    COHORT_MANIFEST_COLUMNS,
    DATA_AVAILABILITY_COLUMNS,
    SAMPLE_TO_LESION_COLUMNS,
    empty_frame,
)


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config value from OmegaConf or dict payloads.

    Args:
        cfg: Config tree.
        dotted: Dotted key path.
        default: Fallback value when the key is missing.
    """
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, dotted)
        return default if value is None else value
    current: Any = cfg
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
        if current is None:
            return default
    return current


def _stage_has_later_progression(patient_rows: pd.DataFrame, stage: str) -> bool:
    """Return whether a patient has any later lesion stage than the input stage.

    Args:
        patient_rows: One-patient lesion table.
        stage: Query stage.
    """
    order = {"Normal": 0, "AAH": 1, "AIS": 2, "MIA": 3, "LUAD": 4}
    current_rank = order.get(str(stage), -1)
    return bool((patient_rows["stage"].map(lambda value: order.get(str(value), -1)) > current_rank).any())


def build_cleaned_cohort_manifest(cfg: DictConfig | dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build the normalized lesion cohort manifest for label repair.

    Args:
        cfg: Active StageBridge config payload.
    """
    spatial = load_luad_evo_spatial_mapping(cfg)
    wes = load_luad_evo_wes_features(cfg)
    label_table = build_lesion_label_table(spatial=spatial, wes=wes, cfg=cfg).copy()

    lesion_obs = (
        spatial.obs[["sample_id", "donor_id", "patient_id", "stage"]]
        .drop_duplicates()
        .rename(columns={"sample_id": "lesion_id"})
        .reset_index(drop=True)
    )
    lesion_obs["sample_id"] = lesion_obs["lesion_id"].astype(str)
    spot_counts = spatial.obs.groupby("sample_id", sort=False).size().rename("num_spots").reset_index()
    lesion_obs = lesion_obs.merge(spot_counts, on="sample_id", how="left")
    lesion_obs["num_spots"] = lesion_obs["num_spots"].fillna(0).astype(int)

    label_table = label_table.rename(
        columns={
            "label": "original_label",
            "label_weight": "original_label_weight",
            "label_source": "original_label_source",
            "notes": "original_label_notes",
        }
    )
    merged = lesion_obs.merge(
        label_table[
            [
                "sample_id",
                "edge_label",
                "original_label",
                "original_label_weight",
                "original_label_source",
                "original_label_notes",
            ]
        ],
        on="sample_id",
        how="left",
    )
    merged["edge_label"] = merged["edge_label"].fillna("")

    wes_keys = {
        (str(row.patient_id), str(row.stage))
        for row in wes.frame[["patient_id", "stage"]].drop_duplicates().itertuples(index=False)
    }
    patient_stage_counts = (
        merged.groupby("patient_id", sort=False)["stage"]
        .nunique()
        .rename("num_patient_stages")
        .reset_index()
    )
    patient_lesion_counts = (
        merged.groupby("patient_id", sort=False)["lesion_id"]
        .nunique()
        .rename("num_patient_lesions")
        .reset_index()
    )
    merged = merged.merge(patient_stage_counts, on="patient_id", how="left")
    merged = merged.merge(patient_lesion_counts, on="patient_id", how="left")
    merged["has_spatial"] = True
    merged["has_wes"] = [
        (str(patient_id), str(stage)) in wes_keys
        for patient_id, stage in merged[["patient_id", "stage"]].itertuples(index=False)
    ]
    merged["can_support_phylogeny"] = merged["num_patient_stages"].fillna(0).astype(int) >= int(
        _cfg_select(cfg, "labels.minimum_stages_for_phylogeny", 2)
    )

    availability_trace: list[str] = []
    for row in merged.itertuples(index=False):
        patient_rows = merged.loc[merged["patient_id"].astype(str) == str(row.patient_id)]
        has_later_stage = _stage_has_later_progression(patient_rows, str(row.stage))
        parts = [
            "spatial" if bool(row.has_spatial) else "no_spatial",
            "wes" if bool(row.has_wes) else "no_wes",
            "curated_label" if str(row.original_label_source).startswith("peng_") else "non_curated_label",
            "later_stage" if has_later_stage else "no_later_stage",
            "phylogeny_ready" if bool(row.can_support_phylogeny) else "single_stage_patient",
        ]
        availability_trace.append(";".join(parts))
    merged["availability_trace"] = availability_trace

    cleaned_manifest = merged.loc[:, list(COHORT_MANIFEST_COLUMNS)].copy()
    sample_to_lesion = cleaned_manifest.loc[:, ["sample_id", "lesion_id", "patient_id", "donor_id", "stage", "edge_label"]].copy()
    sample_to_lesion = sample_to_lesion.loc[:, list(SAMPLE_TO_LESION_COLUMNS)]
    donor_summary = (
        cleaned_manifest.groupby(["patient_id", "donor_id"], as_index=False)
        .agg(
            n_lesions=("lesion_id", "nunique"),
            n_stages=("stage", "nunique"),
            n_labeled_lesions=("original_label", lambda values: int(pd.notna(values).sum())),
            n_wes_supported=("has_wes", lambda values: int(pd.Series(values).sum())),
            n_phylogeny_ready=("can_support_phylogeny", lambda values: int(pd.Series(values).sum())),
        )
    )
    availability_matrix = cleaned_manifest.loc[
        :,
        ["lesion_id", "has_spatial", "has_wes", "original_label_source", "can_support_phylogeny"],
    ].copy()
    availability_matrix["has_curated_label"] = availability_matrix["original_label_source"].astype(str).str.startswith("peng_")
    availability_matrix["has_heuristic_label"] = availability_matrix["original_label_source"].astype(str).eq("heuristic_edge_expansion")
    availability_matrix = availability_matrix.drop(columns=["original_label_source"])
    availability_matrix = availability_matrix.loc[:, list(DATA_AVAILABILITY_COLUMNS)]

    if cleaned_manifest.empty:
        raise ValueError("Label-repair cohort manifest is empty after normalization.")
    if cleaned_manifest["lesion_id"].duplicated().any():
        duplicates = cleaned_manifest.loc[cleaned_manifest["lesion_id"].duplicated(keep=False), "lesion_id"].tolist()
        raise ValueError(f"Detected duplicated lesion identifiers in cleaned manifest: {sorted(set(duplicates))}")

    return {
        "cleaned_manifest": cleaned_manifest,
        "sample_to_lesion": sample_to_lesion,
        "donor_summary": donor_summary,
        "availability_matrix": availability_matrix,
        "wes_features": wes.frame.copy() if not wes.frame.empty else empty_frame(tuple(wes.frame.columns)),
    }
