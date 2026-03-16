"""Assemble model-ready lesion bags for EA-MIST from existing StageBridge assets."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import zarr

from stagebridge.logging_utils import get_logger

from .eamist_common import (
    DEFAULT_RING_EDGES,
    EAMIST_BAG_SCHEMA_VERSION,
    WEAK_STAGE_ORDINAL_SUPERVISION,
    align_feature_rows,
    choose_niche_token_columns,
    default_reports_tables_dir,
    default_viability_report_path,
    load_json_if_exists,
    normalize_niche_table,
    numeric_feature_columns,
    stage_index_or_error,
    utc_now_iso,
    write_json,
)
from .feature_builder import (
    build_expression_templates,
    build_lr_pathway_summary,
    build_neighborhood_stats,
    build_receiver_embedding,
    summarize_ring_compositions,
)
from .neighborhood_builder import _resolve_local_neighborhood_geometry, infer_edge_label
from .snrna import load_luad_evo_snrna_latent
from .stages import stage_to_progression_score
from stagebridge.transition_model.disease_edges import edge_id_map

log = get_logger(__name__)


def _assert_consistent_niche_metadata(
    base: pd.DataFrame, feature_df: pd.DataFrame, *, source: str
) -> None:
    compare_columns = [
        column
        for column in ("sample_id", "donor_id", "patient_id", "stage")
        if column in feature_df.columns and column in base.columns
    ]
    if not compare_columns:
        return
    merged = base.loc[:, ["lesion_id", "niche_id", *compare_columns]].merge(
        feature_df.loc[:, ["lesion_id", "niche_id", *compare_columns]],
        on=["lesion_id", "niche_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__source"),
    )
    for column in compare_columns:
        other = f"{column}__source"
        if other not in merged.columns:
            continue
        mismatch = merged[other].notna() & (
            merged[column].astype(str) != merged[other].astype(str)
        )
        if mismatch.any():
            example = (
                merged.loc[mismatch, ["lesion_id", "niche_id", column, other]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"Inconsistent {column} values between niche parquet and {source}, examples={example}"
            )


def _load_optional_labels(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _sidecar_audit_path(path: Path) -> Path:
    return path.parent / f"{path.stem}.audit.json"


def _resolve_viable_edge_labels(viability: dict[str, Any]) -> tuple[str, ...]:
    edges = viability.get("edges", {}) if isinstance(viability, dict) else {}
    if not isinstance(edges, dict):
        return ()
    ordered = edge_id_map()
    viable = [
        str(label)
        for label, payload in edges.items()
        if isinstance(payload, dict) and bool(payload.get("binary_viable", False))
    ]
    return tuple(sorted(viable, key=lambda label: (ordered.get(str(label), 10_000), str(label))))


def _validate_zarr_against_niches(
    zarr_path: Path | None, niche_df: pd.DataFrame
) -> dict[str, Any]:
    if zarr_path is None:
        return {"checked": False}
    zarr_path = Path(zarr_path).resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing niche token bank zarr: {zarr_path}")
    root = zarr.open_group(str(zarr_path), mode="r")
    zarr_samples = {str(row["sample_id"]) for row in root.attrs.get("meta_rows", [])}
    niche_samples = set(niche_df["sample_id"].astype(str))
    if zarr_samples and zarr_samples != niche_samples:
        raise ValueError("Niche token bank sample ids did not match the niche parquet sample ids.")
    return {
        "checked": True,
        "zarr_path": str(zarr_path),
        "n_groups": int(root.attrs.get("n_groups", 0)),
        "n_rows": int(root.attrs.get("n_rows", 0)),
    }


def run(
    niche_bank: Path | None,
    niche_parquet: Path,
    hlca_features: Path,
    luca_features: Path,
    evo_features: Path,
    out_path: Path,
    *,
    snrna_latent: Path | None = None,
    snrna_raw: Path | None = None,
    refined_labels: Path | None = None,
    viability_report: Path | None = None,
    ring_edges: tuple[float, ...] = DEFAULT_RING_EDGES,
    min_instances: int = 3,
    adaptive_neighbor_k: int = 32,
) -> dict[str, object]:
    run_started = perf_counter()
    niche_parquet = Path(niche_parquet).resolve()
    hlca_features = Path(hlca_features).resolve()
    luca_features = Path(luca_features).resolve()
    evo_features = Path(evo_features).resolve()
    out_path = Path(out_path).resolve()
    for path, label in (
        (niche_parquet, "niche parquet"),
        (hlca_features, "HLCA niche features"),
        (luca_features, "LuCA niche features"),
        (evo_features, "lesion evolution features"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required {label}: {path}")

    log.info("Loading EA-MIST inputs from parquet assets.")
    niche_df = normalize_niche_table(pd.read_parquet(niche_parquet))
    token_columns, token_labels, _token_prefix = choose_niche_token_columns(niche_df)
    zarr_audit = _validate_zarr_against_niches(niche_bank, niche_df)

    hlca_df = pd.read_parquet(hlca_features)
    luca_df = pd.read_parquet(luca_features)
    evo_df = pd.read_parquet(evo_features)
    if luca_df.empty:
        raise ValueError("LuCA niche feature table was empty; EA-MIST requires luca_features.")

    merge_base = niche_df.loc[
        :,
        [
            "lesion_id",
            "sample_id",
            "niche_id",
            "donor_id",
            "patient_id",
            "stage",
            "x",
            "y",
            *token_columns,
        ],
    ].copy()
    _assert_consistent_niche_metadata(merge_base, hlca_df, source="HLCA niche features")
    _assert_consistent_niche_metadata(merge_base, luca_df, source="LuCA niche features")
    merge_base = align_feature_rows(merge_base, hlca_df, source="HLCA niche features")
    merge_base = align_feature_rows(merge_base, luca_df, source="LuCA niche features")
    if evo_df["lesion_id"].duplicated().any():
        duplicates = (
            evo_df.loc[evo_df["lesion_id"].duplicated(keep=False), "lesion_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Detected duplicate lesion ids in lesion evolution features: {duplicates[:10]}"
        )
    merged = merge_base.merge(
        evo_df, on="lesion_id", how="left", validate="many_to_one", suffixes=("", "__evo")
    )
    if "stage__evo" in merged.columns:
        stage_mismatch = merged["stage__evo"].notna() & (
            merged["stage"].astype(str) != merged["stage__evo"].astype(str)
        )
        if stage_mismatch.any():
            example = (
                merged.loc[stage_mismatch, ["lesion_id", "stage", "stage__evo"]]
                .drop_duplicates()
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"Inconsistent stage labels between niche inputs and lesion evo features, examples={example}"
            )
    total_lesions = int(merged["lesion_id"].astype(str).nunique())
    total_niches_expected = int(merged.shape[0])
    log.info(
        "Merged EA-MIST inputs for %d lesions and %d niches.",
        total_lesions,
        total_niches_expected,
    )

    hlca_feature_cols = numeric_feature_columns(hlca_df, "hlca_")
    luca_feature_cols = numeric_feature_columns(luca_df, "luca_")
    evo_feature_cols = [
        column
        for column in evo_df.columns
        if str(column).startswith("evo_") and pd.api.types.is_numeric_dtype(evo_df[column])
    ]
    if not hlca_feature_cols:
        raise ValueError("HLCA feature table did not contain numeric 'hlca_' feature columns.")
    if not luca_feature_cols:
        raise ValueError("LuCA feature table did not contain numeric 'luca_' feature columns.")
    if not evo_feature_cols:
        raise ValueError("Lesion evolution feature table did not contain numeric 'evo_' columns.")

    refined_labels = (
        refined_labels.resolve()
        if refined_labels is not None
        else (default_reports_tables_dir() / "lesion_refined_labels.csv").resolve()
    )
    refined = _load_optional_labels(refined_labels)
    if refined is not None and refined["lesion_id"].duplicated().any():
        raise ValueError("Refined label table contains duplicate lesion ids.")
    refined_lookup = (
        {} if refined is None else refined.set_index("lesion_id").to_dict(orient="index")
    )
    viability_path = (
        viability_report.resolve()
        if viability_report is not None
        else default_viability_report_path().resolve()
    )
    viability = load_json_if_exists(viability_path) or {"edges": {}}
    active_edge_labels = _resolve_viable_edge_labels(viability)
    active_edge_lookup = {label: idx for idx, label in enumerate(active_edge_labels)}
    hlca_audit = load_json_if_exists(_sidecar_audit_path(hlca_features)) or {}
    luca_audit = load_json_if_exists(_sidecar_audit_path(luca_features)) or {}

    cfg: dict[str, Any] = {}
    if snrna_latent is not None:
        cfg.setdefault("data", {})["snrna_latent_h5ad"] = str(Path(snrna_latent).resolve())
    log.info("Loading snRNA latent cohort and building expression templates.")
    snrna = load_luad_evo_snrna_latent(cfg)
    templates = build_expression_templates(
        snrna, raw_h5ad_path=None if snrna_raw is None else str(Path(snrna_raw).resolve())
    )
    log.info(
        "Built expression templates from %d snRNA cells across %d template labels.",
        int(snrna.obs.shape[0]),
        len(templates.expression_by_label),
    )

    bag_rows: list[dict[str, object]] = []
    receiver_dim = None
    ring_shape = None
    pathway_dim = None
    stats_dim = None
    max_niches = 0
    total_niches = 0
    evo_nan_fill_count = 0

    for lesion_index, (lesion_id, lesion_df) in enumerate(
        merged.groupby("lesion_id", sort=True), start=1
    ):
        lesion_started = perf_counter()
        lesion_df = lesion_df.sort_values("niche_id").reset_index(drop=True)
        donor_ids = lesion_df["donor_id"].astype(str).unique().tolist()
        patient_ids = lesion_df["patient_id"].astype(str).unique().tolist()
        stages = lesion_df["stage"].astype(str).unique().tolist()
        if len(donor_ids) != 1 or len(patient_ids) != 1 or len(stages) != 1:
            raise ValueError(
                f"Inconsistent lesion metadata for lesion_id={lesion_id}: donors={donor_ids}, patients={patient_ids}, stages={stages}"
            )
        donor_id = donor_ids[0]
        patient_id = patient_ids[0]
        stage = stages[0]
        stage_index = stage_index_or_error(stage)
        displacement_target = float(stage_to_progression_score(stage))
        edge_label = infer_edge_label(stage)
        log.info(
            "Processing lesion %d/%d: lesion_id=%s donor_id=%s patient_id=%s stage=%s niches=%d",
            lesion_index,
            total_lesions,
            lesion_id,
            donor_id,
            patient_id,
            stage,
            int(lesion_df.shape[0]),
        )

        coords = lesion_df.loc[:, ["x", "y"]].to_numpy(dtype=np.float32, copy=False)
        compositions = lesion_df[token_columns].to_numpy(dtype=np.float32, copy=False)
        receiver_features: list[list[float]] = []
        receiver_state_ids: list[int] = []
        receiver_state_labels: list[str] = []
        receiver_confidences: list[float] = []
        ring_features: list[list[list[float]]] = []
        pathway_features_values: list[list[float]] = []
        niche_stats_features: list[list[float]] = []

        for center_index in range(lesion_df.shape[0]):
            local_ring_edges, density = _resolve_local_neighborhood_geometry(
                coords,
                center_index=center_index,
                configured_edges=[float(value) for value in ring_edges],
                neighborhood_radius=float(ring_edges[-1]),
                min_instances=int(min_instances),
                adaptive_neighbor_k=int(adaptive_neighbor_k),
                num_rings=len(ring_edges) - 1,
            )
            center_composition = compositions[center_index]
            receiver_embedding, receiver_label, receiver_confidence = build_receiver_embedding(
                center_composition,
                token_labels,
                templates,
            )
            receiver_state_id = (
                token_labels.index(receiver_label) if receiver_label in token_labels else -1
            )
            ring_compositions = summarize_ring_compositions(
                compositions,
                coords,
                center_index=center_index,
                ring_edges=list(local_ring_edges),
            )
            pathway_summary = build_lr_pathway_summary(
                ring_compositions,
                token_labels,
                templates,
                donor_id=donor_id,
                stage=stage,
                receiver_label=receiver_label,
            )
            niche_stats = build_neighborhood_stats(
                center_composition,
                ring_compositions,
                receiver_confidence=float(receiver_confidence),
                local_density=float(density),
            )
            receiver_features.append(receiver_embedding.astype(np.float32, copy=False).tolist())
            receiver_state_ids.append(int(receiver_state_id))
            receiver_state_labels.append(str(receiver_label))
            receiver_confidences.append(float(receiver_confidence))
            ring_features.append(ring_compositions.astype(np.float32, copy=False).tolist())
            pathway_features_values.append(pathway_summary.astype(np.float32, copy=False).tolist())
            niche_stats_features.append(niche_stats.astype(np.float32, copy=False).tolist())

        hlca_matrix = lesion_df[hlca_feature_cols].to_numpy(dtype=np.float32, copy=False)
        luca_matrix = lesion_df[luca_feature_cols].to_numpy(dtype=np.float32, copy=False)
        evo_row = lesion_df[evo_feature_cols].iloc[0].to_numpy(dtype=np.float32, copy=False)
        evo_nan_fill_count += int(np.isnan(evo_row).sum())
        evo_row = np.nan_to_num(evo_row, nan=0.0).astype(np.float32, copy=False)

        receiver_dim = len(receiver_features[0]) if receiver_features else receiver_dim
        ring_shape = (
            (
                len(ring_features[0]),
                len(ring_features[0][0]) if ring_features and ring_features[0] else 0,
            )
            if ring_features
            else ring_shape
        )
        pathway_dim = len(pathway_features_values[0]) if pathway_features_values else pathway_dim
        stats_dim = len(niche_stats_features[0]) if niche_stats_features else stats_dim
        max_niches = max(max_niches, int(lesion_df.shape[0]))
        total_niches += int(lesion_df.shape[0])

        refined_row = refined_lookup.get(str(lesion_id), {})
        target_binary = None
        label_text = str(refined_row.get("refined_binary_label", "")).lower()
        if label_text == "positive":
            target_binary = 1.0
        elif label_text == "negative":
            target_binary = 0.0
        target_excluded = bool(refined_row.get("exclusion_flag", False))

        edge_meta = {}
        if edge_label and "edges" in viability:
            edge_meta = dict(viability["edges"].get(edge_label, {}))
        edge_targets = np.zeros((len(active_edge_labels),), dtype=np.float32)
        edge_target_mask = np.zeros((len(active_edge_labels),), dtype=bool)
        if edge_label in active_edge_lookup and target_binary is not None and not target_excluded:
            edge_idx = active_edge_lookup[str(edge_label)]
            edge_targets[edge_idx] = float(target_binary)
            edge_target_mask[edge_idx] = True

        bag_rows.append(
            {
                "lesion_id": str(lesion_id),
                "sample_id": str(lesion_df["sample_id"].iloc[0]),
                "donor_id": donor_id,
                "patient_id": patient_id,
                "stage_index": int(stage_index),
                "stage_label": stage,
                "displacement_target": displacement_target,
                "edge_label": edge_label,
                "edge_target_labels": list(active_edge_labels),
                "edge_targets": edge_targets.astype(np.float32, copy=False).tolist(),
                "edge_target_mask": edge_target_mask.astype(bool, copy=False).tolist(),
                "niche_ids": lesion_df["niche_id"].astype(str).tolist(),
                "receiver_state_ids": receiver_state_ids,
                "receiver_state_labels": receiver_state_labels,
                "receiver_confidences": receiver_confidences,
                "receiver_features": receiver_features,
                "ring_features": ring_features,
                "hlca_features": hlca_matrix.astype(np.float32, copy=False).tolist(),
                "luca_features": luca_matrix.astype(np.float32, copy=False).tolist(),
                "pathway_features": pathway_features_values,
                "niche_stats_features": niche_stats_features,
                "lesion_mask": [1] * int(lesion_df.shape[0]),
                "evo_features": evo_row.tolist(),
                "target_binary_label": target_binary,
                "target_progression_risk_score": refined_row.get("progression_risk_score"),
                "target_confidence_tier": refined_row.get("confidence_tier"),
                "target_uncertainty_flag": refined_row.get("uncertainty_flag"),
                "target_exclusion_flag": refined_row.get("exclusion_flag"),
                "edge_binary_viable": edge_meta.get("binary_viable"),
                "edge_continuous_viable": edge_meta.get("continuous_viable"),
                "edge_recommended_target": edge_meta.get("recommended_target"),
                "edge_binary_reason": edge_meta.get("binary_reason"),
                "edge_continuous_reason": edge_meta.get("continuous_reason"),
            }
        )
        log.info(
            "Completed lesion %d/%d: lesion_id=%s elapsed=%.1fs cumulative_niches=%d/%d",
            lesion_index,
            total_lesions,
            lesion_id,
            perf_counter() - lesion_started,
            total_niches,
            total_niches_expected,
        )

    output = pd.DataFrame(bag_rows)
    if output.empty:
        raise ValueError("EA-MIST bag assembly produced no lesion rows.")
    if output["lesion_id"].duplicated().any():
        raise ValueError("Duplicate lesion ids were produced during EA-MIST bag assembly.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Writing EA-MIST bag parquet with %d lesion rows to %s", int(output.shape[0]), out_path
    )
    output.to_parquet(out_path, index=False)

    audit = {
        "created_at_utc": utc_now_iso(),
        "schema_version": EAMIST_BAG_SCHEMA_VERSION,
        "displacement_supervision": WEAK_STAGE_ORDINAL_SUPERVISION,
        "niche_parquet": str(niche_parquet),
        "hlca_features": str(hlca_features),
        "luca_features": str(luca_features),
        "evo_features": str(evo_features),
        "out_path": str(out_path),
        "num_rings": int(len(ring_edges) - 1),
        "ring_edges": [float(edge) for edge in ring_edges],
        "receiver_state_vocabulary": token_labels,
        "active_edge_labels": list(active_edge_labels),
        "num_lesions": int(output.shape[0]),
        "num_niches": int(total_niches),
        "max_niches_per_lesion": int(max_niches),
        "receiver_dim": int(receiver_dim or 0),
        "ring_shape": None if ring_shape is None else [int(ring_shape[0]), int(ring_shape[1])],
        "hlca_dim": int(len(hlca_feature_cols)),
        "luca_dim": int(len(luca_feature_cols)),
        "pathway_dim": int(pathway_dim or 0),
        "niche_stats_dim": int(stats_dim or 0),
        "evo_dim": int(len(evo_feature_cols)),
        "hlca_feature_columns": hlca_feature_cols,
        "luca_feature_columns": luca_feature_cols,
        "evo_feature_columns": evo_feature_cols,
        "refined_labels_used": str(refined_labels) if refined is not None else None,
        "viability_report_used": str(viability_path) if viability else None,
        "evo_nan_values_filled_with_zero": int(evo_nan_fill_count),
        "zarr_validation": zarr_audit,
        "hlca_state_column": hlca_audit.get("chosen_hlca_state_column")
        or hlca_audit.get("chosen_state_column"),
        "luca_state_column": luca_audit.get("chosen_luca_state_column"),
        "luca_scoring_space": luca_audit.get("chosen_scoring_space"),
    }
    write_json(out_path.parent / f"{out_path.stem}.audit.json", audit)
    log.info("EA-MIST bag assembly completed in %.1fs", perf_counter() - run_started)
    return audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niche-bank",
        type=Path,
        default=None,
        help="Optional niche_token_bank.zarr for validation",
    )
    parser.add_argument(
        "--niche-parquet", type=Path, required=True, help="Path to niche_tokens_full.parquet"
    )
    parser.add_argument(
        "--hlca-features", type=Path, required=True, help="Path to niche_hlca_features.parquet"
    )
    parser.add_argument(
        "--luca-features", type=Path, required=True, help="Path to niche_luca_features.parquet"
    )
    parser.add_argument(
        "--evo-features", type=Path, required=True, help="Path to lesion_evo_features.parquet"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output parquet path for lesion bags"
    )
    parser.add_argument(
        "--snrna-latent", type=Path, default=None, help="Optional override for snRNA latent h5ad"
    )
    parser.add_argument(
        "--snrna-raw", type=Path, default=None, help="Optional override for raw snRNA h5ad"
    )
    parser.add_argument(
        "--refined-labels", type=Path, default=None, help="Optional lesion_refined_labels.csv"
    )
    parser.add_argument(
        "--viability-report", type=Path, default=None, help="Optional split_viability_report.json"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run(
        args.niche_bank,
        args.niche_parquet,
        args.hlca_features,
        args.luca_features,
        args.evo_features,
        args.out,
        snrna_latent=args.snrna_latent,
        snrna_raw=args.snrna_raw,
        refined_labels=args.refined_labels,
        viability_report=args.viability_report,
    )
    print(f"built EA-MIST bags: {audit}")


if __name__ == "__main__":
    main()
