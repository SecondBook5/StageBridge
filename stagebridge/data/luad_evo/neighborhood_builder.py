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
from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.utils.types import LesionBag, LocalNicheExample

VALID_SOURCE_STAGES: frozenset[str] = frozenset({"AAH", "AIS"})


@dataclass(slots=True, frozen=True)
class NeighborhoodBuildResult:
    """Structured output from lesion-bag preprocessing."""

    bags: list[LesionBag]
    summary: pd.DataFrame
    label_table: pd.DataFrame
    diagnostics: dict[str, object]


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
    """Resolve the active edge for a lesion stage."""
    if str(stage) == "AAH":
        return "AAH->AIS"
    if str(stage) == "AIS":
        return "AIS->MIA"
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
        obs.loc[obs["stage"].isin(list(VALID_SOURCE_STAGES)), ["sample_id", "donor_id", "patient_id", "stage"]]
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
    num_rings: int = 3,
) -> list[float]:
    """Resolve ring boundaries for one local neighborhood."""
    if configured_edges:
        return [float(value) for value in configured_edges]
    center = sample_coords[center_index]
    dists = np.linalg.norm(sample_coords - center[None, :], axis=1)
    max_radius = float(min(max(dists.max(initial=0.0), 1e-3), neighborhood_radius))
    return list(np.linspace(0.0, max_radius, num_rings + 1))


def _local_density(sample_coords: np.ndarray, *, center_index: int, neighborhood_radius: float) -> float:
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
    num_rings: int = 3,
) -> tuple[list[float], float]:
    """Resolve ring edges and effective density, falling back to adaptive kNN geometry when needed."""
    radius_density = _local_density(sample_coords, center_index=center_index, neighborhood_radius=neighborhood_radius)
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
    kth_index = min(max(int(adaptive_neighbor_k), int(min_instances), 1), int(sorted_dists.shape[0])) - 1
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
        epithelial_score = sample_compositions[:, epi_cols].sum(axis=1).astype(np.float32, copy=False)
    else:
        epithelial_score = sample_compositions.max(axis=1).astype(np.float32, copy=False)
    if not np.any(epithelial_score > 0.0):
        epithelial_score = sample_compositions.max(axis=1).astype(np.float32, copy=False)
    order = np.argsort(-epithelial_score)
    top_pool = order[: max(max_neighborhoods * 4, max_neighborhoods)]
    rng = np.random.default_rng(int(seed))

    if strategy == "uniform":
        chosen = np.sort(rng.choice(top_pool, size=min(max_neighborhoods, top_pool.shape[0]), replace=False))
        return chosen.astype(np.int64, copy=False)

    if strategy == "top_k_dense":
        densities = np.asarray(
            [_local_density(sample_coords, center_index=int(idx), neighborhood_radius=neighborhood_radius) for idx in top_pool],
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
    template_max_cells_per_group = _cfg_get(cfg, "context_model.eamist.template_max_cells_per_group", 512)
    templates = build_expression_templates(
        snrna,
        raw_h5ad_path=raw_h5ad_path,
        max_cells_per_group=None if template_max_cells_per_group is None else int(template_max_cells_per_group),
        seed=base_seed,
    )
    feature_names = [str(name) for name in spatial.feature_names]
    edge_lookup = edge_id_map()

    max_neighborhoods = int(_cfg_get(cfg, "context_model.eamist.max_neighborhoods_per_lesion", 64))
    neighborhood_radius = float(_cfg_get(cfg, "context_model.eamist.neighborhood_radius", 150.0))
    ring_edges_cfg = _cfg_get(cfg, "context_model.eamist.ring_edges", None)
    sampling_strategy = str(_cfg_get(cfg, "context_model.eamist.neighborhood_sampling_strategy", "uniform"))
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
        edge_label = infer_edge_label(stage)
        if edge_label is None:
            continue
        label_row = label_lookup.get((str(sample_id), edge_label))
        if label_row is None:
            continue

        donor_id = str(sample_obs["donor_id"].iloc[0])
        patient_id = str(sample_obs.get("patient_id", sample_obs["donor_id"]).iloc[0])
        sample_coords = np.asarray(spatial.coords[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
        sample_compositions = np.asarray(spatial.compositions[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
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
        for local_idx, center_index in enumerate(selected_centers.tolist()):
            ring_edges, density = _resolve_local_neighborhood_geometry(
                sample_coords,
                center_index=center_index,
                configured_edges=ring_edges_cfg,
                neighborhood_radius=neighborhood_radius,
                min_instances=min_instances,
                adaptive_neighbor_k=adaptive_neighbor_k,
                num_rings=3,
            )
            if density < float(min_instances):
                continue
            center_composition = sample_compositions[center_index].astype(np.float32, copy=False)
            receiver_embedding, receiver_label, receiver_confidence = build_receiver_embedding(
                center_composition,
                feature_names,
                templates,
            )
            receiver_state_id, _state_name, _state_score = infer_receiver_state(center_composition, feature_names)
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
                    edge_label=edge_label,
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
            edge_id=int(edge_lookup[edge_label]),
            edge_label=edge_label,
            label=float(label_row.label),
            label_weight=float(label_row.label_weight),
            label_source=str(label_row.label_source),
            neighborhoods=neighborhoods,
            evolution_features=None if evolution_features is None else evolution_features.astype(np.float32, copy=False),
            notes=str(label_row.notes),
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
                "evolution_feature_dim": 0 if bag.evolution_features is None else int(bag.evolution_features.shape[0]),
            }
        )

    if not bags:
        raise ValueError("EA-MIST preprocessing produced no lesion bags.")
    summary = pd.DataFrame(summary_rows).sort_values(["edge_label", "donor_id", "sample_id"]).reset_index(drop=True)
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
    cache_path = resolve_lesion_bag_cache_path(cfg)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if not isinstance(cached, NeighborhoodBuildResult):
            raise TypeError(f"Lesion-bag cache at {cache_path} did not contain a NeighborhoodBuildResult.")
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
