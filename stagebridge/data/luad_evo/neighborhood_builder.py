"""Lesion-bag neighborhood construction for EA-MIST."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd

from stagebridge.data.common.schema import LatentCohort, SpatialCohort, WESCohort
from stagebridge.data.luad_evo.eamist_common import (
    EAMIST_BAG_SCHEMA_VERSION,
    WEAK_STAGE_ORDINAL_SUPERVISION,
    load_json_if_exists,
)
from stagebridge.data.luad_evo.feature_builder import (
    build_expression_templates,
    build_lr_pathway_summary,
    build_neighborhood_stats,
    build_receiver_embedding,
    epithelial_columns,
    flatten_neighborhood_features,
    infer_receiver_state,
    summarize_neighborhood_build,
    summarize_ring_compositions,
)
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.stages import stage_to_progression_score, stage_to_index
from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.utils.types import LesionBag, LocalNicheExample

VALID_SOURCE_STAGES: frozenset[str] = frozenset({"AAH", "AIS"})


def _migrate_legacy_bags(bags: list[LesionBag]) -> None:
    """Patch bags loaded from old caches that are missing newer optional fields."""
    for bag in bags:
        if bag.stage_index is None:
            try:
                idx = stage_to_index(bag.stage)
            except ValueError:
                idx = -1
            object.__setattr__(bag, "stage_index", idx)
        if bag.displacement_target is None:
            object.__setattr__(bag, "displacement_target", stage_to_progression_score(bag.stage))
        if bag.edge_target_labels is None:
            object.__setattr__(bag, "edge_target_labels", ())


@dataclass(slots=True, frozen=True)
class NeighborhoodBuildResult:
    """Structured output from lesion-bag preprocessing."""

    bags: list[LesionBag]
    summary: pd.DataFrame
    label_table: pd.DataFrame
    diagnostics: dict[str, object]


def resolve_eamist_bag_parquet_path(cfg: Any | None = None) -> Path:
    """Resolve the canonical prebuilt EA-MIST bag parquet path."""
    import os

    _env_root = os.environ.get("STAGEBRIDGE_DATA_ROOT", "")
    data_root = Path(str(_cfg_get(cfg or {}, "data.data_root", _env_root))).resolve()
    configured = _cfg_get(
        cfg or {}, "data.eamist_bags_parquet", data_root / "processed/features/eamist_bags.parquet"
    )
    return Path(str(configured)).resolve()


def resolve_eamist_bag_audit_path(path: Path) -> Path:
    """Return the audit sidecar path for one EA-MIST bag parquet."""
    return path.parent / f"{path.stem}.audit.json"


def _coerce_vector(value: object, *, label: str, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    return arr.astype(dtype, copy=False)


def _coerce_matrix(value: object, *, label: str, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array-like value, got shape={arr.shape}.")
    return arr.astype(dtype, copy=False)


def _coerce_ring_tensor(value: object, *, label: str, expected_num_rings: int) -> np.ndarray:
    raw = np.asarray(value)
    # Handle object-dtype arrays (e.g. array of arrays from parquet deserialization)
    if raw.dtype == object:
        arr = np.stack([np.asarray(row, dtype=np.float32) for row in raw]).astype(np.float32)
    else:
        arr = raw.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D ring matrix, got shape={arr.shape}.")
    if int(arr.shape[0]) != int(expected_num_rings):
        raise ValueError(
            f"{label} used {arr.shape[0]} rings but the canonical schema requires {expected_num_rings} rings."
        )
    return arr.astype(np.float32, copy=False)


def _edge_targets_from_row(
    row: pd.Series,
    *,
    active_edge_labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    targets_raw = row.get("edge_targets")
    mask_raw = row.get("edge_target_mask")
    if targets_raw is not None and mask_raw is not None:
        targets = _coerce_vector(targets_raw, label="edge_targets", dtype=np.float32)
        mask = _coerce_vector(mask_raw, label="edge_target_mask", dtype=bool).astype(
            bool, copy=False
        )
        if targets.shape[0] != len(active_edge_labels) or mask.shape[0] != len(active_edge_labels):
            raise ValueError(
                f"Edge targets for lesion_id={row.get('lesion_id')} do not match active_edge_labels={active_edge_labels}."
            )
        return targets, mask

    targets = np.zeros((len(active_edge_labels),), dtype=np.float32)
    mask = np.zeros((len(active_edge_labels),), dtype=bool)
    edge_label = row.get("edge_label")
    target_binary = row.get("target_binary_label")
    if edge_label in active_edge_labels and pd.notna(target_binary):
        edge_idx = active_edge_labels.index(str(edge_label))
        targets[edge_idx] = float(target_binary)
        mask[edge_idx] = True
    return targets, mask


def build_lesion_bags_from_parquet(path: Path) -> NeighborhoodBuildResult:
    """Load the canonical prebuilt EA-MIST bag parquet and validate its schema."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"EA-MIST bag parquet not found: {path}")
    audit_path = resolve_eamist_bag_audit_path(path)
    audit = load_json_if_exists(audit_path)
    if audit is None:
        raise FileNotFoundError(f"EA-MIST bag audit sidecar not found: {audit_path}")
    if str(audit.get("schema_version")) != EAMIST_BAG_SCHEMA_VERSION:
        raise ValueError(
            f"Refusing to load stale EA-MIST bags from {path}: expected schema_version="
            f"{EAMIST_BAG_SCHEMA_VERSION!r}, found {audit.get('schema_version')!r}."
        )
    if int(audit.get("num_rings", -1)) != 4:
        raise ValueError(
            f"EA-MIST bag parquet {path} used num_rings={audit.get('num_rings')} instead of the canonical 4."
        )
    if str(audit.get("luca_state_column")) != "cell_type_tumor":
        raise ValueError(
            f"EA-MIST bag parquet {path} was built from LuCA state column {audit.get('luca_state_column')!r}; "
            "expected 'cell_type_tumor'."
        )
    if str(audit.get("displacement_supervision")) != WEAK_STAGE_ORDINAL_SUPERVISION:
        raise ValueError(
            f"EA-MIST bag parquet {path} is missing the expected displacement supervision stamp "
            f"{WEAK_STAGE_ORDINAL_SUPERVISION!r}."
        )
    active_edge_labels = tuple(str(label) for label in audit.get("active_edge_labels", []))
    receiver_vocab = [str(label) for label in audit.get("receiver_state_vocabulary", [])]
    receiver_lookup = {label: idx for idx, label in enumerate(receiver_vocab)}
    ring_edges = [float(edge) for edge in audit.get("ring_edges", [])]

    frame = pd.read_parquet(path)
    required_columns = {
        "lesion_id",
        "sample_id",
        "donor_id",
        "patient_id",
        "stage_label",
        "stage_index",
        "displacement_target",
        "niche_ids",
        "receiver_features",
        "ring_features",
        "hlca_features",
        "luca_features",
        "pathway_features",
        "niche_stats_features",
        "evo_features",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(f"EA-MIST bag parquet is missing required columns: {sorted(missing)}")

    bags: list[LesionBag] = []
    summary_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    edge_lookup = edge_id_map()
    for row in frame.itertuples(index=False):
        row_map = row._asdict()
        lesion_id = str(row_map["lesion_id"])
        stage = str(row_map.get("stage_label") or row_map.get("stage"))
        stage_index = int(row_map["stage_index"])
        displacement_target = float(
            row_map.get("displacement_target", stage_to_progression_score(stage))
        )
        if abs(displacement_target - stage_to_progression_score(stage)) > 1e-6:
            raise ValueError(
                f"EA-MIST bag parquet has inconsistent displacement_target for lesion_id={lesion_id}: "
                f"stage={stage}, stage_score={stage_to_progression_score(stage):.4f}, value={displacement_target:.4f}"
            )

        niche_ids = [str(value) for value in row_map["niche_ids"]]
        receiver_features = [
            _coerce_vector(value, label=f"receiver_features[{idx}]")
            for idx, value in enumerate(row_map["receiver_features"])
        ]
        ring_features = [
            _coerce_ring_tensor(
                value, label=f"ring_features[{idx}]", expected_num_rings=int(audit["num_rings"])
            )
            for idx, value in enumerate(row_map["ring_features"])
        ]
        hlca_features = [
            _coerce_vector(value, label=f"hlca_features[{idx}]")
            for idx, value in enumerate(row_map["hlca_features"])
        ]
        luca_features = [
            _coerce_vector(value, label=f"luca_features[{idx}]")
            for idx, value in enumerate(row_map["luca_features"])
        ]
        pathway_features = [
            _coerce_vector(value, label=f"pathway_features[{idx}]")
            for idx, value in enumerate(row_map["pathway_features"])
        ]
        niche_stats = [
            _coerce_vector(value, label=f"niche_stats_features[{idx}]")
            for idx, value in enumerate(row_map["niche_stats_features"])
        ]
        receiver_state_ids_raw = row_map.get("receiver_state_ids")
        receiver_state_labels_raw = row_map.get("receiver_state_labels")
        if receiver_state_ids_raw is None and receiver_state_labels_raw is None:
            raise ValueError(
                "EA-MIST bag parquet is missing both receiver_state_ids and receiver_state_labels."
            )
        if receiver_state_ids_raw is not None:
            receiver_state_ids = _coerce_vector(
                receiver_state_ids_raw, label="receiver_state_ids", dtype=np.int64
            ).astype(np.int64, copy=False)
        else:
            labels = [str(value) for value in receiver_state_labels_raw]
            receiver_state_ids = np.asarray(
                [receiver_lookup.get(label, -1) for label in labels], dtype=np.int64
            )
        receiver_state_labels = (
            [str(value) for value in receiver_state_labels_raw]
            if receiver_state_labels_raw is not None
            else [
                receiver_vocab[int(idx)] if 0 <= int(idx) < len(receiver_vocab) else "unknown"
                for idx in receiver_state_ids.tolist()
            ]
        )
        receiver_confidences = _coerce_vector(
            row_map.get("receiver_confidences", np.ones(len(niche_ids))),
            label="receiver_confidences",
            dtype=np.float32,
        )

        lengths = {
            "niche_ids": len(niche_ids),
            "receiver_features": len(receiver_features),
            "ring_features": len(ring_features),
            "hlca_features": len(hlca_features),
            "luca_features": len(luca_features),
            "pathway_features": len(pathway_features),
            "niche_stats_features": len(niche_stats),
            "receiver_state_ids": int(receiver_state_ids.shape[0]),
            "receiver_state_labels": len(receiver_state_labels),
            "receiver_confidences": int(receiver_confidences.shape[0]),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent niche list lengths for lesion_id={lesion_id}: {lengths}"
            )

        neighborhoods: list[LocalNicheExample] = []
        for idx, niche_id in enumerate(niche_ids):
            flat = flatten_neighborhood_features(
                receiver_features[idx],
                ring_features[idx],
                hlca_features[idx],
                luca_features[idx],
                pathway_features[idx],
                niche_stats[idx],
            )
            neighborhoods.append(
                LocalNicheExample(
                    lesion_id=lesion_id,
                    sample_id=str(row_map["sample_id"]),
                    donor_id=str(row_map["donor_id"]),
                    patient_id=str(row_map["patient_id"]),
                    stage=stage,
                    edge_label=str(row_map.get("edge_label") or ""),
                    receiver_index=idx,
                    receiver_embedding=receiver_features[idx],
                    receiver_state_id=int(receiver_state_ids[idx]),
                    ring_compositions=ring_features[idx],
                    lr_pathway_summary=pathway_features[idx],
                    neighborhood_stats=niche_stats[idx],
                    flat_features=flat,
                    center_coord=np.asarray([float(idx), float(idx)], dtype=np.float32),
                    hlca_features=hlca_features[idx],
                    luca_features=luca_features[idx],
                    receiver_state_label=receiver_state_labels[idx],
                    receiver_confidence=float(receiver_confidences[idx]),
                    notes=niche_id,
                )
            )

        edge_targets, edge_target_mask = _edge_targets_from_row(
            pd.Series(row_map), active_edge_labels=active_edge_labels
        )
        first_valid_edge = next(
            (
                active_edge_labels[idx]
                for idx, flag in enumerate(edge_target_mask.tolist())
                if bool(flag)
            ),
            str(row_map.get("edge_label") or ""),
        )
        first_valid_target = (
            float(edge_targets[np.argmax(edge_target_mask)]) if edge_target_mask.any() else 0.0
        )
        first_valid_weight = 1.0 if edge_target_mask.any() else 0.0
        bag = LesionBag(
            lesion_id=lesion_id,
            sample_id=str(row_map["sample_id"]),
            donor_id=str(row_map["donor_id"]),
            patient_id=str(row_map["patient_id"]),
            stage=stage,
            edge_id=int(edge_lookup.get(first_valid_edge, -1)),
            edge_label=str(first_valid_edge),
            label=float(first_valid_target),
            label_weight=float(first_valid_weight),
            label_source="prebuilt_eamist_bag",
            neighborhoods=neighborhoods,
            evolution_features=_coerce_vector(
                row_map.get("evo_features", []), label="evo_features", dtype=np.float32
            ),
            stage_index=stage_index,
            displacement_target=displacement_target,
            edge_targets=edge_targets.astype(np.float32, copy=False),
            edge_target_mask=edge_target_mask.astype(bool, copy=False),
            edge_target_labels=tuple(active_edge_labels),
            notes=f"schema={audit.get('schema_version')}",
        )
        bags.append(bag)
        summary_rows.append(
            {
                "lesion_id": bag.lesion_id,
                "sample_id": bag.sample_id,
                "donor_id": bag.donor_id,
                "patient_id": bag.patient_id,
                "stage": bag.stage,
                "stage_index": int(stage_index),
                "displacement_target": float(displacement_target),
                "edge_label": bag.edge_label,
                "label": float(bag.label),
                "label_weight": float(bag.label_weight),
                "label_source": bag.label_source,
                "num_neighborhoods": bag.num_neighborhoods,
                "evolution_feature_dim": 0
                if bag.evolution_features is None
                else int(np.asarray(bag.evolution_features).shape[0]),
                "num_active_edge_targets": int(edge_target_mask.sum()),
            }
        )
        for edge_idx, edge_label in enumerate(active_edge_labels):
            if not bool(edge_target_mask[edge_idx]):
                continue
            label_rows.append(
                {
                    "lesion_id": bag.lesion_id,
                    "sample_id": bag.sample_id,
                    "donor_id": bag.donor_id,
                    "patient_id": bag.patient_id,
                    "stage": bag.stage,
                    "stage_index": int(stage_index),
                    "edge_label": edge_label,
                    "label": float(edge_targets[edge_idx]),
                    "label_weight": 1.0,
                    "label_source": "prebuilt_eamist_bag",
                    "notes": f"schema={audit.get('schema_version')}",
                }
            )

    if not bags:
        raise ValueError(f"EA-MIST bag parquet produced no lesion bags: {path}")
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["stage_index", "donor_id", "sample_id"])
        .reset_index(drop=True)
    )
    label_table = pd.DataFrame(label_rows)
    diagnostics = summarize_neighborhood_build(bags)
    diagnostics["source"] = "prebuilt_bag_parquet"
    diagnostics["bag_parquet"] = str(path)
    diagnostics["bag_audit"] = str(audit_path)
    diagnostics["schema_version"] = str(audit.get("schema_version"))
    diagnostics["ring_edges"] = ring_edges
    diagnostics["active_edge_labels"] = list(active_edge_labels)
    diagnostics["stages"] = sorted(summary["stage"].astype(str).unique().tolist())
    diagnostics["luca_state_column"] = str(audit.get("luca_state_column"))
    diagnostics["displacement_supervision"] = str(audit.get("displacement_supervision"))
    return NeighborhoodBuildResult(
        bags=bags,
        summary=summary,
        label_table=label_table,
        diagnostics=diagnostics,
    )


def _cfg_get(cfg: Any, dotted: str, default: Any) -> Any:
    """Safely read a dotted config key from dict-like configs."""
    current = cfg
    for part in dotted.split("."):
        if hasattr(current, "get"):
            current = current.get(part, None)
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return default
        if current is None:
            return default
    return current


def resolve_curated_labels_path(cfg: Any | None = None) -> Path:
    """Resolve the active curated lesion-label manifest."""
    if cfg is not None:
        configured = _cfg_get(cfg, "data.curated_progression_labels", None)
        if configured:
            return Path(str(configured))
    return Path(__file__).resolve().parent / "curated_progression_labels.csv"


def resolve_lesion_bag_cache_path(cfg: Any | None = None) -> Path | None:
    """Resolve the optional lesion-bag cache path for EA-MIST preprocessing."""
    if cfg is None:
        return None
    if bool(_cfg_get(cfg, "context_model.eamist.use_prebuilt_bags", True)):
        return None
    if not bool(_cfg_get(cfg, "context_model.eamist.cache_lesion_bags", True)):
        return None
    configured = _cfg_get(cfg, "context_model.eamist.lesion_bag_cache_path", None)
    if configured:
        return Path(str(configured))
    output_dir = Path(str(_cfg_get(cfg, "output_dir", "outputs/scratch")))
    run_name = str(_cfg_get(cfg, "run_name", "stagebridge_v1"))
    return output_dir / run_name / "eamist_cache" / "lesion_bags.pkl"


def load_curated_lesion_labels(cfg: Any | None = None) -> pd.DataFrame:
    """Load the curated lesion-level label table if present."""
    path = resolve_curated_labels_path(cfg)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "sample_id",
                "donor_id",
                "stage",
                "edge_label",
                "progression_competent_label",
                "label_source",
                "notes",
            ]
        )
    frame = pd.read_csv(path).copy()
    required = {
        "sample_id",
        "donor_id",
        "stage",
        "edge_label",
        "progression_competent_label",
        "label_source",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Curated lesion labels are missing required columns: {sorted(missing)}")
    if "notes" not in frame.columns:
        frame["notes"] = ""
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["donor_id"] = frame["donor_id"].astype(str)
    frame["stage"] = frame["stage"].astype(str)
    frame["edge_label"] = frame["edge_label"].astype(str)
    frame["progression_competent_label"] = frame["progression_competent_label"].astype(float)
    frame["label_source"] = frame["label_source"].astype(str)
    frame["notes"] = frame["notes"].fillna("").astype(str)
    return frame


def _canonical_sample_key(sample_id: str) -> str:
    """Normalize a lesion/sample identifier across cohort-specific GSM prefixes."""
    normalized = str(sample_id).strip()
    normalized = re.sub(r"^GSM\d+_", "", normalized)
    normalized = re.sub(r"([A-Za-z]+)(\d+)$", r"\1-\2", normalized)
    return normalized


def infer_edge_label(stage: str) -> str | None:
    """Resolve the active transition edge for a lesion stage.

    Returns the canonical edge label for stages that sit on a monitored
    transition boundary.  Stages that are *not* on an active edge (Normal,
    MIA, LUAD) return ``None`` — they still participate in stage
    classification and displacement regression but have no binary edge
    target.
    """
    _stage = str(stage)
    if _stage == "AAH":
        return "AAH->AIS"
    if _stage == "AIS":
        return "AIS->MIA"
    # Normal, MIA, LUAD have no active transition edge.
    return None


def _heuristic_label_for_aah(
    *,
    donor_id: str,
    spatial_obs: pd.DataFrame,
    wes_lookup: dict[tuple[str, str], np.ndarray],
) -> tuple[float, str, str]:
    """Build a low-confidence heuristic lesion label for AAH lesions."""
    donor_mask = spatial_obs["donor_id"].astype(str) == str(donor_id)
    donor_stages = set(spatial_obs.loc[donor_mask, "stage"].astype(str))
    has_later_stage = bool({"AIS", "MIA", "LUAD"}.intersection(donor_stages))
    wes_vec = None
    for stage_name in ("AIS", "MIA", "LUAD", "AAH"):
        key = (str(donor_id), stage_name)
        if key in wes_lookup:
            wes_vec = wes_lookup[key]
            break
    tmb = 0.0 if wes_vec is None or wes_vec.size == 0 else float(wes_vec[0])
    driver_risk = 0.0 if wes_vec is None or wes_vec.size <= 1 else float(np.max(wes_vec[1:]))
    positive = has_later_stage or tmb >= 1.0 or driver_risk > 0.0
    note = (
        f"heuristic_expansion(has_later_stage={has_later_stage}, "
        f"tmb={tmb:.3f}, driver_risk={driver_risk:.3f})"
    )
    return (1.0 if positive else 0.0), "heuristic_edge_expansion", note


def build_lesion_label_table(
    spatial: SpatialCohort,
    *,
    wes: WESCohort | None = None,
    cfg: Any | None = None,
) -> pd.DataFrame:
    """Build curated-plus-heuristic lesion labels for the active source stages."""
    obs = spatial.obs.copy()
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["patient_id"] = obs.get("patient_id", obs["donor_id"]).astype(str)
    obs["stage"] = obs["stage"].astype(str)
    lesion_table = (
        obs.loc[
            obs["stage"].isin(list(VALID_SOURCE_STAGES)),
            ["sample_id", "donor_id", "patient_id", "stage"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    spatial_sample_lookup = {
        _canonical_sample_key(sample_id): str(sample_id)
        for sample_id in lesion_table["sample_id"].astype(str).tolist()
    }
    curated = load_curated_lesion_labels(cfg)
    wes_lookup: dict[tuple[str, str], np.ndarray] = {}
    if wes is not None:
        for row in wes.frame.itertuples(index=False):
            wes_lookup[(str(row.patient_id), str(row.stage))] = np.asarray(
                [getattr(row, col) for col in wes.feature_columns],
                dtype=np.float32,
            )

    rows: list[dict[str, object]] = []
    curated_keys = set()
    for row in curated.itertuples(index=False):
        spatial_sample_id = spatial_sample_lookup.get(_canonical_sample_key(str(row.sample_id)))
        if spatial_sample_id is None:
            donor_stage_matches = lesion_table.loc[
                (lesion_table["donor_id"].astype(str) == str(row.donor_id))
                & (lesion_table["stage"].astype(str) == str(row.stage)),
                "sample_id",
            ].astype(str)
            if donor_stage_matches.nunique() == 1:
                spatial_sample_id = str(donor_stage_matches.iloc[0])
        if spatial_sample_id is None:
            continue
        curated_keys.add((str(spatial_sample_id), str(row.edge_label)))
        rows.append(
            {
                "sample_id": str(spatial_sample_id),
                "donor_id": str(row.donor_id),
                "patient_id": str(getattr(row, "patient_id", row.donor_id)),
                "stage": str(row.stage),
                "edge_label": str(row.edge_label),
                "label": float(row.progression_competent_label),
                "label_weight": 1.0,
                "label_source": str(row.label_source),
                "notes": str(row.notes),
            }
        )

    for lesion in lesion_table.itertuples(index=False):
        edge_label = infer_edge_label(str(lesion.stage))
        if edge_label is None:
            continue
        key = (str(lesion.sample_id), edge_label)
        if key in curated_keys:
            continue
        if str(lesion.stage) != "AAH":
            continue
        label, label_source, notes = _heuristic_label_for_aah(
            donor_id=str(lesion.donor_id),
            spatial_obs=obs,
            wes_lookup=wes_lookup,
        )
        rows.append(
            {
                "sample_id": str(lesion.sample_id),
                "donor_id": str(lesion.donor_id),
                "patient_id": str(lesion.patient_id),
                "stage": str(lesion.stage),
                "edge_label": edge_label,
                "label": float(label),
                "label_weight": 0.5,
                "label_source": label_source,
                "notes": notes,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No lesion labels were available for EA-MIST preprocessing.")
    return table.sort_values(["edge_label", "donor_id", "sample_id"]).reset_index(drop=True)


def _derive_ring_edges(
    sample_coords: np.ndarray,
    *,
    configured_edges: list[float] | None,
    center_index: int,
    neighborhood_radius: float,
    num_rings: int = 4,
) -> list[float]:
    """Resolve ring boundaries for one local neighborhood."""
    if configured_edges:
        return [float(value) for value in configured_edges]
    center = sample_coords[center_index]
    dists = np.linalg.norm(sample_coords - center[None, :], axis=1)
    max_radius = float(min(max(dists.max(initial=0.0), 1e-3), neighborhood_radius))
    return list(np.linspace(0.0, max_radius, num_rings + 1))


def _local_density(
    sample_coords: np.ndarray, *, center_index: int, neighborhood_radius: float
) -> float:
    """Return a compact local density summary around one center spot."""
    center = sample_coords[center_index]
    dists = np.linalg.norm(sample_coords - center[None, :], axis=1)
    return float(np.count_nonzero(dists <= float(neighborhood_radius)))


def _resolve_local_neighborhood_geometry(
    sample_coords: np.ndarray,
    *,
    center_index: int,
    configured_edges: list[float] | None,
    neighborhood_radius: float,
    min_instances: int,
    adaptive_neighbor_k: int,
    num_rings: int = 4,
) -> tuple[list[float], float]:
    """Resolve ring edges and effective density, falling back to adaptive kNN geometry when needed."""
    radius_density = _local_density(
        sample_coords, center_index=center_index, neighborhood_radius=neighborhood_radius
    )
    if radius_density >= float(min_instances):
        return (
            _derive_ring_edges(
                sample_coords,
                configured_edges=configured_edges,
                center_index=center_index,
                neighborhood_radius=neighborhood_radius,
                num_rings=num_rings,
            ),
            radius_density,
        )

    center = sample_coords[center_index]
    dists = np.linalg.norm(sample_coords - center[None, :], axis=1)
    sorted_dists = np.sort(dists.astype(np.float32, copy=False))
    kth_index = (
        min(max(int(adaptive_neighbor_k), int(min_instances), 1), int(sorted_dists.shape[0])) - 1
    )
    effective_radius = float(max(sorted_dists[kth_index], 1e-3))
    ring_edges = list(np.linspace(0.0, effective_radius, num_rings + 1))
    adaptive_density = float(np.count_nonzero(dists <= effective_radius))
    return ring_edges, adaptive_density


def _select_candidate_indices(
    sample_compositions: np.ndarray,
    feature_names: list[str],
    sample_coords: np.ndarray,
    *,
    max_neighborhoods: int,
    strategy: str,
    seed: int,
    neighborhood_radius: float,
) -> np.ndarray:
    """Select center spots for one lesion according to the configured strategy."""
    epi_cols = epithelial_columns(feature_names)
    if epi_cols:
        epithelial_score = (
            sample_compositions[:, epi_cols].sum(axis=1).astype(np.float32, copy=False)
        )
    else:
        epithelial_score = sample_compositions.max(axis=1).astype(np.float32, copy=False)
    if not np.any(epithelial_score > 0.0):
        epithelial_score = sample_compositions.max(axis=1).astype(np.float32, copy=False)
    order = np.argsort(-epithelial_score)
    top_pool = order[: max(max_neighborhoods * 4, max_neighborhoods)]
    rng = np.random.default_rng(int(seed))

    if strategy == "uniform":
        chosen = np.sort(
            rng.choice(top_pool, size=min(max_neighborhoods, top_pool.shape[0]), replace=False)
        )
        return chosen.astype(np.int64, copy=False)

    if strategy == "top_k_dense":
        densities = np.asarray(
            [
                _local_density(
                    sample_coords, center_index=int(idx), neighborhood_radius=neighborhood_radius
                )
                for idx in top_pool
            ],
            dtype=np.float32,
        )
        chosen = top_pool[np.argsort(-densities)[: min(max_neighborhoods, top_pool.shape[0])]]
        return np.sort(chosen).astype(np.int64, copy=False)

    if strategy == "stratified":
        dominant = np.argmax(sample_compositions[top_pool], axis=1)
        chosen_rows: list[int] = []
        for dom in np.unique(dominant):
            dom_rows = top_pool[dominant == dom]
            take = max(1, int(np.ceil(max_neighborhoods / max(len(np.unique(dominant)), 1))))
            if dom_rows.shape[0] <= take:
                chosen_rows.extend(dom_rows.tolist())
            else:
                chosen_rows.extend(rng.choice(dom_rows, size=take, replace=False).tolist())
        chosen = np.asarray(sorted(set(chosen_rows))[:max_neighborhoods], dtype=np.int64)
        return chosen

    raise ValueError(f"Unsupported neighborhood_sampling_strategy '{strategy}'.")


def build_lesion_bags(
    snrna: LatentCohort,
    spatial: SpatialCohort,
    *,
    wes: WESCohort | None = None,
    cfg: Any | None = None,
) -> NeighborhoodBuildResult:
    """Build EA-MIST lesion bags from the current LUAD cohort."""
    label_table = build_lesion_label_table(spatial, wes=wes, cfg=cfg)
    label_lookup = {
        (str(row.sample_id), str(row.edge_label)): row
        for row in label_table.itertuples(index=False)
    }
    raw_h5ad_path = None
    if cfg is not None:
        raw_h5ad_path = str(resolve_luad_evo_paths(cfg).snrna_h5ad)
    base_seed = int(_cfg_get(cfg, "seed", 42))
    template_max_cells_per_group = _cfg_get(
        cfg, "context_model.eamist.template_max_cells_per_group", 512
    )
    templates = build_expression_templates(
        snrna,
        raw_h5ad_path=raw_h5ad_path,
        max_cells_per_group=None
        if template_max_cells_per_group is None
        else int(template_max_cells_per_group),
        seed=base_seed,
    )
    feature_names = [str(name) for name in spatial.feature_names]
    edge_lookup = edge_id_map()

    max_neighborhoods = int(_cfg_get(cfg, "context_model.eamist.max_neighborhoods_per_lesion", 64))
    neighborhood_radius = float(_cfg_get(cfg, "context_model.eamist.neighborhood_radius", 150.0))
    ring_edges_cfg = _cfg_get(cfg, "context_model.eamist.ring_edges", None)
    sampling_strategy = str(
        _cfg_get(cfg, "context_model.eamist.neighborhood_sampling_strategy", "uniform")
    )
    min_instances = int(_cfg_get(cfg, "context_model.eamist.min_cells_per_neighborhood", 3))
    adaptive_neighbor_k = int(_cfg_get(cfg, "context_model.eamist.adaptive_neighbor_k", 32))

    bags: list[LesionBag] = []
    summary_rows: list[dict[str, object]] = []

    grouped = spatial.obs.reset_index(drop=True).groupby("sample_id", sort=False).indices
    wes_lookup: dict[tuple[str, str], np.ndarray] = {}
    if wes is not None:
        for row in wes.frame.itertuples(index=False):
            wes_lookup[(str(row.patient_id), str(row.stage))] = np.asarray(
                [getattr(row, col) for col in wes.feature_columns],
                dtype=np.float32,
            )

    for sample_id, indices in grouped.items():
        sample_obs = spatial.obs.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True)
        stage = str(sample_obs["stage"].iloc[0])
        edge_label = infer_edge_label(stage)  # None for Normal/MIA/LUAD
        # Look up curated label; stages without an active edge still
        # participate via stage classification + displacement regression.
        label_row = None
        if edge_label is not None:
            label_row = label_lookup.get((str(sample_id), edge_label))

        donor_id = str(sample_obs["donor_id"].iloc[0])
        patient_id = str(sample_obs.get("patient_id", sample_obs["donor_id"]).iloc[0])
        sample_coords = np.asarray(
            spatial.coords[np.asarray(indices, dtype=np.int64)], dtype=np.float32
        )
        sample_compositions = np.asarray(
            spatial.compositions[np.asarray(indices, dtype=np.int64)], dtype=np.float32
        )
        selected_centers = _select_candidate_indices(
            sample_compositions,
            feature_names,
            sample_coords,
            max_neighborhoods=max_neighborhoods,
            strategy=sampling_strategy,
            seed=base_seed + int(len(bags)),
            neighborhood_radius=neighborhood_radius,
        )
        neighborhoods: list[LocalNicheExample] = []
        num_rings = int(_cfg_get(cfg, "context_model.eamist.num_rings", 4))
        for local_idx, center_index in enumerate(selected_centers.tolist()):
            ring_edges, density = _resolve_local_neighborhood_geometry(
                sample_coords,
                center_index=center_index,
                configured_edges=ring_edges_cfg,
                neighborhood_radius=neighborhood_radius,
                min_instances=min_instances,
                adaptive_neighbor_k=adaptive_neighbor_k,
                num_rings=num_rings,
            )
            if density < float(min_instances):
                continue
            center_composition = sample_compositions[center_index].astype(np.float32, copy=False)
            receiver_embedding, receiver_label, receiver_confidence = build_receiver_embedding(
                center_composition,
                feature_names,
                templates,
            )
            receiver_state_id, _state_name, _state_score = infer_receiver_state(
                center_composition, feature_names
            )
            ring_compositions = summarize_ring_compositions(
                sample_compositions,
                sample_coords,
                center_index=center_index,
                ring_edges=ring_edges,
            )
            lr_summary = build_lr_pathway_summary(
                ring_compositions,
                feature_names,
                templates,
                donor_id=donor_id,
                stage=stage,
                receiver_label=receiver_label,
            )
            neighborhood_stats = build_neighborhood_stats(
                center_composition,
                ring_compositions,
                receiver_confidence=receiver_confidence,
                local_density=density,
            )
            flat_features = flatten_neighborhood_features(
                receiver_embedding,
                ring_compositions,
                None,
                None,
                lr_summary,
                neighborhood_stats,
            )
            neighborhoods.append(
                LocalNicheExample(
                    lesion_id=str(sample_id),
                    sample_id=str(sample_id),
                    donor_id=donor_id,
                    patient_id=patient_id,
                    stage=stage,
                    edge_label=edge_label or "",
                    receiver_index=local_idx,
                    receiver_embedding=receiver_embedding,
                    receiver_state_id=int(receiver_state_id),
                    ring_compositions=ring_compositions,
                    lr_pathway_summary=lr_summary,
                    neighborhood_stats=neighborhood_stats,
                    flat_features=flat_features,
                    center_coord=sample_coords[center_index].astype(np.float32, copy=False),
                    receiver_confidence=float(receiver_confidence),
                )
            )
        if not neighborhoods:
            continue

        evolution_features = wes_lookup.get((patient_id, stage))
        bag = LesionBag(
            lesion_id=str(sample_id),
            sample_id=str(sample_id),
            donor_id=donor_id,
            patient_id=patient_id,
            stage=stage,
            edge_id=int(edge_lookup.get(edge_label or "", -1)),
            edge_label=edge_label or "",
            label=float(label_row.label) if label_row is not None else 0.0,
            label_weight=float(label_row.label_weight) if label_row is not None else 0.0,
            label_source=str(label_row.label_source)
            if label_row is not None
            else "no_active_edge",
            neighborhoods=neighborhoods,
            evolution_features=None
            if evolution_features is None
            else evolution_features.astype(np.float32, copy=False),
            stage_index=stage_to_index(stage),
            displacement_target=stage_to_progression_score(stage),
            notes=str(label_row.notes) if label_row is not None else f"stage={stage}",
        )
        bags.append(bag)
        summary_rows.append(
            {
                "lesion_id": bag.lesion_id,
                "sample_id": bag.sample_id,
                "donor_id": bag.donor_id,
                "patient_id": bag.patient_id,
                "stage": bag.stage,
                "edge_label": bag.edge_label,
                "label": bag.label,
                "label_weight": bag.label_weight,
                "label_source": bag.label_source,
                "num_neighborhoods": bag.num_neighborhoods,
                "evolution_feature_dim": 0
                if bag.evolution_features is None
                else int(bag.evolution_features.shape[0]),
            }
        )

    if not bags:
        raise ValueError("EA-MIST preprocessing produced no lesion bags.")
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["edge_label", "donor_id", "sample_id"])
        .reset_index(drop=True)
    )
    diagnostics = summarize_neighborhood_build(bags)
    diagnostics["num_labels"] = int(label_table.shape[0])
    diagnostics["edges"] = sorted(summary["edge_label"].astype(str).unique().tolist())
    diagnostics["stages"] = sorted(summary["stage"].astype(str).unique().tolist())
    return NeighborhoodBuildResult(
        bags=bags,
        summary=summary,
        label_table=label_table,
        diagnostics=diagnostics,
    )


def build_lesion_bags_from_config(cfg: Any) -> NeighborhoodBuildResult:
    """Load active cohorts from config and build lesion bags."""
    if bool(_cfg_get(cfg, "context_model.eamist.use_prebuilt_bags", True)):
        bag_path = resolve_eamist_bag_parquet_path(cfg)
        return build_lesion_bags_from_parquet(bag_path)

    cache_path = resolve_lesion_bag_cache_path(cfg)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if not isinstance(cached, NeighborhoodBuildResult):
            raise TypeError(
                f"Lesion-bag cache at {cache_path} did not contain a NeighborhoodBuildResult."
            )
        _migrate_legacy_bags(cached.bags)
        return cached

    from stagebridge.data.luad_evo.snrna import load_luad_evo_snrna_latent
    from stagebridge.data.luad_evo.visium import load_luad_evo_spatial_mapping
    from stagebridge.data.luad_evo.wes import load_luad_evo_wes_features

    paths = resolve_luad_evo_paths(cfg)
    if not paths.snrna_latent_h5ad.exists():
        raise FileNotFoundError(f"Missing snRNA latent file: {paths.snrna_latent_h5ad}")
    if not paths.spatial_tangram_h5ad.exists() and not paths.spatial_h5ad.exists():
        raise FileNotFoundError(f"Missing spatial mapping file: {paths.spatial_tangram_h5ad}")
    snrna = load_luad_evo_snrna_latent(cfg)
    spatial = load_luad_evo_spatial_mapping(cfg)
    wes = load_luad_evo_wes_features(cfg) if paths.wes_features_path.exists() else None
    result = build_lesion_bags(snrna, spatial, wes=wes, cfg=cfg)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle)
    return result
