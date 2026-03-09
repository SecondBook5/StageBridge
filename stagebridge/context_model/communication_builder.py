"""Communication-relay example construction for StageBridge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from stagebridge.context_model.token_builder import build_typed_spot_tokens
from stagebridge.data.common.schema import LatentCohort, SpatialCohort, WESCohort
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.stages import normalize_stage_label
from stagebridge.transition_model.disease_edges import edge_id_map
from stagebridge.utils.types import CommunicationBag, CommunicationNeighborhoodExample

EPITHELIAL_LABELS: frozenset[str] = frozenset({"AT2", "Basal", "Secretory", "Ciliated"})


@dataclass(slots=True, frozen=True)
class CommunicationPrior:
    ligand: str
    receptor: str
    family: str
    support: float = 1.0


LUNG_LR_PRIORS: tuple[CommunicationPrior, ...] = (
    CommunicationPrior("IL1B", "IL1R1", "inflammatory", 1.00),
    CommunicationPrior("IL6", "IL6ST", "inflammatory", 0.95),
    CommunicationPrior("TNF", "TNFRSF1A", "inflammatory", 0.92),
    CommunicationPrior("CXCL9", "CXCR3", "chemokine", 0.85),
    CommunicationPrior("CXCL10", "CXCR3", "chemokine", 0.85),
    CommunicationPrior("CXCL12", "CXCR4", "chemokine", 1.00),
    CommunicationPrior("TGFB1", "TGFBR2", "tgfb", 1.00),
    CommunicationPrior("TGFB3", "TGFBR2", "tgfb", 0.80),
    CommunicationPrior("AREG", "EGFR", "growth_factor", 0.95),
    CommunicationPrior("EREG", "EGFR", "growth_factor", 0.85),
    CommunicationPrior("HBEGF", "EGFR", "growth_factor", 0.80),
    CommunicationPrior("HGF", "MET", "growth_factor", 0.90),
    CommunicationPrior("JAG1", "NOTCH1", "notch", 0.85),
    CommunicationPrior("DLL4", "NOTCH1", "notch", 0.80),
    CommunicationPrior("MIF", "CD74", "immune_modulatory", 0.75),
    CommunicationPrior("OSM", "OSMR", "inflammatory", 0.80),
    CommunicationPrior("SPP1", "ITGAV", "ecm", 0.78),
    CommunicationPrior("COL1A1", "ITGB1", "ecm", 0.75),
    CommunicationPrior("FN1", "ITGB1", "ecm", 0.82),
    CommunicationPrior("VEGFA", "KDR", "vascular", 0.78),
    CommunicationPrior("CXCL1", "CXCR2", "chemokine", 0.72),
    CommunicationPrior("CXCL2", "CXCR2", "chemokine", 0.72),
    CommunicationPrior("WNT5A", "FZD7", "developmental", 0.68),
    CommunicationPrior("EGF", "EGFR", "growth_factor", 0.65),
)

RECEIVER_PROGRAMS: dict[str, tuple[str, ...]] = {
    "progenitor": ("KRT8", "CEACAM5", "MUC1", "KRT19"),
    "epithelial_identity": ("EPCAM", "MUC1", "KRT19", "CEACAM5"),
    "stress_emt": ("KRT17", "VIM", "ITGB1", "TGFBR2"),
    "inflammatory_response": ("IL1R1", "TNFRSF1A", "CXCR4", "OSMR"),
    "growth_factor_response": ("EGFR", "ERBB2", "MET", "TGFBR2"),
    "migration_invasion": ("CXCR4", "ITGB1", "VIM", "MMP9"),
}

FAMILY_TO_PROGRAM = {
    "inflammatory": "inflammatory_response",
    "chemokine": "migration_invasion",
    "tgfb": "stress_emt",
    "growth_factor": "progenitor",
    "notch": "progenitor",
    "ecm": "migration_invasion",
    "vascular": "growth_factor_response",
    "immune_modulatory": "inflammatory_response",
    "developmental": "progenitor",
}

RELAY_PROGRAMS: tuple[tuple[str, str], ...] = (
    ("inflammatory_relay", "inflammatory_response"),
    ("tgfb_relay", "stress_emt"),
    ("growth_relay", "progenitor"),
    ("chemokine_relay", "migration_invasion"),
)


def communication_gene_panel() -> list[str]:
    genes = {
        gene
        for prior in LUNG_LR_PRIORS
        for gene in (prior.ligand, prior.receptor)
    }
    for panel in RECEIVER_PROGRAMS.values():
        genes.update(panel)
    return sorted(genes)


def _safe_log1p_dense(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        arr = matrix.toarray()
    else:
        arr = np.asarray(matrix)
    arr = np.asarray(arr, dtype=np.float32)
    return np.log1p(arr).astype(np.float32, copy=False)


def load_expression_panel(
    cell_ids: Iterable[str],
    *,
    raw_h5ad_path: Path | None = None,
    expression_frame: pd.DataFrame | None = None,
    genes: Iterable[str] | None = None,
    cfg: Any | None = None,
) -> pd.DataFrame:
    """Load a compact cell-by-gene expression panel aligned to ``cell_ids``."""
    cell_id_list = [str(cell_id) for cell_id in cell_ids]
    selected_genes = list(genes or communication_gene_panel())
    if expression_frame is not None:
        frame = expression_frame.copy()
        if "cell_id" in frame.columns:
            frame = frame.set_index("cell_id")
        frame.index = frame.index.astype(str)
        return frame.reindex(index=cell_id_list, columns=selected_genes, fill_value=0.0).astype(np.float32)

    if raw_h5ad_path is None:
        raw_h5ad_path = resolve_luad_evo_paths(cfg or {}).snrna_h5ad
    raw = anndata.read_h5ad(raw_h5ad_path, backed="r")
    obs_index = pd.Index(raw.obs_names.astype(str))
    rows = obs_index.get_indexer(cell_id_list)
    if np.any(rows < 0):
        missing = [cell_id_list[idx] for idx, row in enumerate(rows) if row < 0][:5]
        raise KeyError(f"Could not align {len(missing)} cell ids to raw snRNA matrix, examples={missing}")
    var_index = pd.Index(raw.var_names.astype(str))
    available_genes = [gene for gene in selected_genes if gene in var_index]
    gene_rows = var_index.get_indexer(available_genes)
    subset = raw.X[rows][:, gene_rows]
    dense = _safe_log1p_dense(subset)
    frame = pd.DataFrame(dense, index=cell_id_list, columns=available_genes, dtype=np.float32)
    for gene in selected_genes:
        if gene not in frame.columns:
            frame[gene] = np.float32(0.0)
    return frame.loc[:, selected_genes].astype(np.float32, copy=False)


def build_expression_templates(
    cohort: LatentCohort,
    expression_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build donor-stage and fallback sender-expression templates."""
    obs = cohort.obs.copy()
    if "hlca_label" not in obs.columns:
        raise KeyError("Communication relay requires 'hlca_label' in snRNA latent obs.")
    if "cell_id" not in obs.columns:
        raise KeyError("Communication relay requires 'cell_id' in snRNA latent obs.")
    merged = obs[["cell_id", "donor_id", "stage", "hlca_label"]].copy()
    merged["cell_id"] = merged["cell_id"].astype(str)
    merged["stage"] = merged["stage"].astype(str).map(normalize_stage_label)
    merged["donor_id"] = merged["donor_id"].astype(str)
    merged["hlca_label"] = merged["hlca_label"].astype(str)
    merged = merged.merge(expression_panel, left_on="cell_id", right_index=True, how="left")
    genes = expression_panel.columns.tolist()
    donor_stage_label = merged.groupby(["donor_id", "stage", "hlca_label"], dropna=False)[genes].mean()
    stage_label = merged.groupby(["stage", "hlca_label"], dropna=False)[genes].mean()
    label_global = merged.groupby(["hlca_label"], dropna=False)[genes].mean()
    return {
        "donor_stage_label": donor_stage_label,
        "stage_label": stage_label,
        "label_global": label_global,
    }


def _lookup_template_gene(
    templates: dict[str, pd.DataFrame],
    *,
    donor_id: str,
    stage: str,
    hlca_label: str,
    gene: str,
) -> float:
    donor_stage = templates["donor_stage_label"]
    stage_level = templates["stage_label"]
    global_level = templates["label_global"]
    key3 = (str(donor_id), str(stage), str(hlca_label))
    if key3 in donor_stage.index:
        return float(donor_stage.loc[key3, gene])
    key2 = (str(stage), str(hlca_label))
    if key2 in stage_level.index:
        return float(stage_level.loc[key2, gene])
    if str(hlca_label) in global_level.index:
        return float(global_level.loc[str(hlca_label), gene])
    return 0.0


def _weighted_sender_gene_score(
    compositions: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    templates: dict[str, pd.DataFrame],
    *,
    donor_id: str,
    stage: str,
    gene: str,
) -> np.ndarray:
    scores = np.zeros(compositions.shape[0], dtype=np.float32)
    feature_name_list = [str(name) for name in feature_names]
    usable = min(int(compositions.shape[1]), len(feature_name_list))
    for col_idx, label in enumerate(feature_name_list[:usable]):
        gene_score = _lookup_template_gene(
            templates,
            donor_id=str(donor_id),
            stage=str(stage),
            hlca_label=str(label),
            gene=str(gene),
        )
        scores += compositions[:, col_idx].astype(np.float32) * np.float32(gene_score)
    return scores


def _compute_program_scores(
    receiver_expression: pd.Series,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for program_name, genes in RECEIVER_PROGRAMS.items():
        present = [gene for gene in genes if gene in receiver_expression.index]
        if not present:
            out[program_name] = 0.0
            continue
        out[program_name] = float(np.asarray(receiver_expression.loc[present], dtype=np.float32).mean())
    return out


def _family_id_map() -> dict[str, int]:
    families = sorted({prior.family for prior in LUNG_LR_PRIORS})
    return {name: idx for idx, name in enumerate(families)}


def _select_sender_spots(
    spot_df: pd.DataFrame,
    typed_tokens: np.ndarray,
    *,
    feature_names: tuple[str, ...] | list[str],
    max_anchor_spots: int,
    max_sender_spots: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    epithelial_columns = [
        idx for idx, name in enumerate(feature_names)
        if str(name) in EPITHELIAL_LABELS
    ]
    if not epithelial_columns:
        epithelial_score = typed_tokens.max(axis=1)
    else:
        epithelial_score = typed_tokens[:, epithelial_columns].sum(axis=1)
    anchor_rows = np.argsort(-epithelial_score)[: max(1, min(max_anchor_spots, typed_tokens.shape[0]))]
    anchor_centroid = spot_df.iloc[anchor_rows][["x", "y"]].to_numpy(dtype=np.float32).mean(axis=0)
    coords = spot_df[["x", "y"]].to_numpy(dtype=np.float32)
    dists = np.linalg.norm(coords - anchor_centroid[None, :], axis=1)
    order = np.argsort(dists)[: max(1, min(max_sender_spots, typed_tokens.shape[0]))]
    chosen = spot_df.iloc[order].reset_index(drop=True)
    chosen_tokens = typed_tokens[order].astype(np.float32, copy=False)
    chosen_dists = dists[order].astype(np.float32, copy=False)
    return chosen, chosen_tokens, chosen_dists, order.astype(np.int64, copy=False)


def _distance_to_ring(distance: np.ndarray, num_rings: int) -> np.ndarray:
    if distance.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if float(distance.max()) <= 0.0:
        return np.zeros(distance.shape[0], dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, num_rings + 1)
    thresholds = np.quantile(distance, quantiles[1:-1]) if num_rings > 1 else np.array([], dtype=np.float32)
    rings = np.digitize(distance, thresholds, right=True)
    return rings.astype(np.int64, copy=False)


def _build_lr_tokens(
    sender_tokens: np.ndarray,
    sender_dists: np.ndarray,
    sender_types: np.ndarray,
    ring_ids: np.ndarray,
    receiver_expression: pd.Series,
    templates: dict[str, pd.DataFrame],
    *,
    donor_id: str,
    stage: str,
    feature_names: tuple[str, ...] | list[str],
    max_lr_tokens: int,
) -> tuple[np.ndarray, list[str]]:
    family_ids = _family_id_map()
    dominant_strength = sender_tokens.max(axis=1)
    lr_rows: list[np.ndarray] = []
    lr_names: list[str] = []
    for prior in LUNG_LR_PRIORS:
        ligand_activity = _weighted_sender_gene_score(
            sender_tokens,
            feature_names,
            templates,
            donor_id=donor_id,
            stage=stage,
            gene=prior.ligand,
        )
        receptor_activity = np.full(
            ligand_activity.shape[0],
            np.float32(receiver_expression.get(prior.receptor, 0.0)),
            dtype=np.float32,
        )
        support = np.full(ligand_activity.shape[0], np.float32(prior.support), dtype=np.float32)
        proposal_score = ligand_activity * receptor_activity * support
        target_program = FAMILY_TO_PROGRAM.get(prior.family, "progenitor")
        receiver_program = np.float32(
            np.asarray([receiver_expression.get(gene, 0.0) for gene in RECEIVER_PROGRAMS[target_program]], dtype=np.float32).mean()
        )
        family_id = np.full(ligand_activity.shape[0], np.float32(family_ids[prior.family]), dtype=np.float32)
        ring_norm = ring_ids.astype(np.float32) / max(1.0, float(ring_ids.max(initial=0) + 1))
        dist_norm = sender_dists.astype(np.float32) / max(1e-6, float(sender_dists.max(initial=1.0)))
        for idx in range(ligand_activity.shape[0]):
            lr_rows.append(
                np.asarray(
                    [
                        ligand_activity[idx],
                        receptor_activity[idx],
                        proposal_score[idx],
                        support[idx],
                        dist_norm[idx],
                        ring_norm[idx],
                        float(dominant_strength[idx]),
                        float(sender_types[idx]),
                        float(family_id[idx]),
                        float(receiver_program),
                    ],
                    dtype=np.float32,
                )
            )
            lr_names.append(f"{prior.ligand}->{prior.receptor}|{prior.family}|sender_{idx}")
    if not lr_rows:
        return np.zeros((0, 10), dtype=np.float32), []
    stacked = np.stack(lr_rows, axis=0)
    order = np.argsort(-stacked[:, 2])[: max(1, min(max_lr_tokens, stacked.shape[0]))]
    return stacked[order], [lr_names[idx] for idx in order.tolist()]


def _build_response_tokens(
    receiver_program_scores: dict[str, float],
    lr_tokens: np.ndarray,
    *,
    edge_id: int,
) -> tuple[np.ndarray, list[str]]:
    if not receiver_program_scores:
        return np.zeros((0, 5), dtype=np.float32), []
    family_ids = _family_id_map()
    family_score_lookup = {
        family_name: float(lr_tokens[lr_tokens[:, 8] == family_id, 2].mean()) if np.any(lr_tokens[:, 8] == family_id) else 0.0
        for family_name, family_id in family_ids.items()
    } if lr_tokens.size else {family_name: 0.0 for family_name in family_ids}
    rows: list[np.ndarray] = []
    names: list[str] = []
    for prog_idx, (program_name, score) in enumerate(receiver_program_scores.items()):
        linked_family = next((family for family, target in FAMILY_TO_PROGRAM.items() if target == program_name), "growth_factor")
        linked_score = family_score_lookup.get(linked_family, 0.0)
        rows.append(
            np.asarray(
                [
                    np.float32(score),
                    np.float32(linked_score),
                    np.float32(prog_idx),
                    np.float32(len(RECEIVER_PROGRAMS[program_name])),
                    np.float32(edge_id),
                ],
                dtype=np.float32,
            )
        )
        names.append(f"{program_name}|{linked_family}")
    return (np.stack(rows, axis=0), names) if rows else (np.zeros((0, 5), dtype=np.float32), [])


def _build_relay_tokens(
    receiver_program_scores: dict[str, float],
    lr_tokens: np.ndarray,
    sender_tokens: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    family_ids = _family_id_map()
    family_score_lookup = {
        family_name: float(lr_tokens[lr_tokens[:, 8] == family_id, 2].mean()) if lr_tokens.size and np.any(lr_tokens[:, 8] == family_id) else 0.0
        for family_name, family_id in family_ids.items()
    }
    dominant_sender_score = float(sender_tokens.max(axis=1).mean()) if sender_tokens.size else 0.0
    rows: list[np.ndarray] = []
    names: list[str] = []
    for relay_name, program_name in RELAY_PROGRAMS:
        family_name = relay_name.replace("_relay", "")
        inflow = family_score_lookup.get(family_name, 0.0)
        program_score = float(receiver_program_scores.get(program_name, 0.0))
        rows.append(
            np.asarray(
                [
                    np.float32(inflow),
                    np.float32(program_score),
                    np.float32(inflow * program_score),
                    np.float32(dominant_sender_score),
                    np.float32(len(rows)),
                    np.float32(family_ids.get(family_name, 0)),
                ],
                dtype=np.float32,
            )
        )
        names.append(f"{relay_name}|{program_name}")
    return (np.stack(rows, axis=0), names) if rows else (np.zeros((0, 6), dtype=np.float32), [])


def load_curated_progression_labels(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["sample_id", "donor_id", "stage", "edge_label", "progression_competent_label", "label_source", "notes"]
        )
    df = pd.read_csv(path)
    expected = {"sample_id", "edge_label", "progression_competent_label"}
    missing = expected.difference(df.columns)
    if missing:
        raise KeyError(f"Progression label manifest missing columns: {sorted(missing)}")
    return df


def build_progression_label_manifest(
    cohort: LatentCohort,
    *,
    wes: WESCohort | None = None,
    curated_manifest: pd.DataFrame | None = None,
    active_edges: Iterable[str] = ("AAH->AIS", "AIS->MIA"),
) -> pd.DataFrame:
    """Build sample-edge weak labels, preferring curated labels when available."""
    sample_rows = cohort.obs[["sample_id", "donor_id", "stage"]].drop_duplicates().copy()
    sample_rows["stage"] = sample_rows["stage"].astype(str).map(normalize_stage_label)
    curated = curated_manifest if curated_manifest is not None else pd.DataFrame()
    if curated.empty:
        curated = pd.DataFrame(
            columns=["sample_id", "donor_id", "stage", "edge_label", "progression_competent_label", "label_source", "notes"]
        )
    use_curated_only = not curated.empty
    rows: list[dict[str, Any]] = []
    wes_lookup: dict[tuple[str, str], np.ndarray] = {}
    feature_columns: list[str] = []
    if wes is not None:
        from stagebridge.data.luad_evo.wes import build_wes_feature_lookup

        wes_lookup = build_wes_feature_lookup(wes)
        feature_columns = list(wes.feature_columns)
    risk_scores: dict[tuple[str, str], float] = {}
    for _, row in sample_rows.iterrows():
        feat = wes_lookup.get((str(row["donor_id"]), str(row["stage"])))
        if feat is None:
            risk_scores[(str(row["sample_id"]), str(row["stage"]))] = 0.0
            continue
        feat_arr = np.asarray(feat, dtype=np.float32)
        tmb = float(feat_arr[0]) if feat_arr.size else 0.0
        drivers = float(feat_arr[1:].sum()) if feat_arr.size > 1 else 0.0
        risk_scores[(str(row["sample_id"]), str(row["stage"]))] = float(tmb + 0.75 * drivers)

    stage_medians: dict[str, float] = {}
    for stage_name in sample_rows["stage"].astype(str).unique().tolist():
        values = [score for (_sample_id, label_stage), score in risk_scores.items() if label_stage == stage_name]
        stage_medians[stage_name] = float(np.median(values)) if values else 0.0
    donor_stage_sets = sample_rows.groupby("donor_id")["stage"].agg(lambda items: set(items.astype(str).tolist())).to_dict()

    for _, row in sample_rows.iterrows():
        donor_id = str(row["donor_id"])
        sample_id = str(row["sample_id"])
        stage = str(row["stage"])
        donor_stages = donor_stage_sets.get(donor_id, set())
        for label in active_edges:
            src, tgt = [part.strip() for part in str(label).split("->", 1)]
            if stage != src:
                continue
            curated_row = curated[
                (curated["sample_id"].astype(str) == sample_id)
                & (curated["edge_label"].astype(str) == str(label))
            ]
            if not curated_row.empty:
                entry = curated_row.iloc[0].to_dict()
                rows.append(
                    {
                        "sample_id": sample_id,
                        "donor_id": donor_id,
                        "stage": stage,
                        "edge_label": str(label),
                        "progression_competent_label": float(entry["progression_competent_label"]),
                        "label_source": str(entry.get("label_source", "curated_manifest")),
                        "notes": entry.get("notes", ""),
                    }
                )
                continue
            if use_curated_only:
                continue
            risk = risk_scores.get((sample_id, stage), 0.0)
            target_present = tgt in donor_stages
            label_value = 1.0 if (target_present and risk >= stage_medians.get(stage, 0.0)) else 0.0
            rows.append(
                {
                    "sample_id": sample_id,
                    "donor_id": donor_id,
                    "stage": stage,
                    "edge_label": str(label),
                    "progression_competent_label": label_value,
                    "label_source": "heuristic_wes_stage_progression",
                    "notes": f"target_present={int(target_present)};features={','.join(feature_columns) if feature_columns else 'none'}",
                }
            )
    return pd.DataFrame(rows)


def build_communication_bags(
    snrna: LatentCohort,
    spatial: SpatialCohort,
    *,
    wes: WESCohort | None = None,
    cfg: Any | None = None,
    active_edges: Iterable[str] = ("AAH->AIS", "AIS->MIA"),
    curated_manifest_path: Path | str | None = None,
    expression_frame: pd.DataFrame | None = None,
    max_receiver_cells_per_sample: int = 16,
    max_anchor_spots: int = 4,
    max_sender_spots: int = 24,
    max_lr_tokens: int = 12,
    num_distance_rings: int = 3,
    seed: int = 42,
) -> tuple[list[CommunicationBag], pd.DataFrame]:
    """Build weakly supervised communication bags from LUAD snRNA + spatial data."""
    rng = np.random.default_rng(int(seed))
    if "cell_id" not in snrna.obs.columns:
        raise KeyError("snRNA cohort must contain a 'cell_id' column.")
    if "hlca_label" not in snrna.obs.columns:
        raise KeyError("snRNA cohort must contain an 'hlca_label' column.")

    selected_genes = communication_gene_panel()
    expression_panel = load_expression_panel(
        snrna.obs["cell_id"].astype(str).tolist(),
        raw_h5ad_path=None if cfg is None else resolve_luad_evo_paths(cfg).snrna_h5ad,
        expression_frame=expression_frame,
        genes=selected_genes,
        cfg=cfg,
    )
    templates = build_expression_templates(snrna, expression_panel)
    typed = build_typed_spot_tokens(spatial.compositions, spatial.coords, spatial.obs, spatial.feature_names)
    spatial_df = typed.obs.copy()
    spatial_df["x"] = spatial.coords[:, 0]
    spatial_df["y"] = spatial.coords[:, 1]
    curated = (
        load_curated_progression_labels(curated_manifest_path)
        if curated_manifest_path is not None
        else pd.DataFrame()
    )
    label_manifest = build_progression_label_manifest(
        snrna,
        wes=wes,
        curated_manifest=curated,
        active_edges=active_edges,
    )
    label_lookup = {
        (str(row.sample_id), str(row.edge_label)): (float(row.progression_competent_label), str(row.label_source), str(row.notes))
        for row in label_manifest.itertuples(index=False)
    }
    wes_lookup: dict[tuple[str, str], np.ndarray] = {}
    wes_default = np.zeros((0,), dtype=np.float32)
    if wes is not None:
        from stagebridge.data.luad_evo.wes import build_wes_feature_lookup

        wes_lookup = build_wes_feature_lookup(wes)
        wes_default = np.zeros((len(wes.feature_columns),), dtype=np.float32)

    bags: list[CommunicationBag] = []
    metadata_rows: list[dict[str, Any]] = []
    edge_lookup = edge_id_map()
    for edge_label in active_edges:
        stage_src, stage_tgt = [part.strip() for part in str(edge_label).split("->", 1)]
        source_cells = snrna.obs[
            (snrna.obs["stage"].astype(str) == stage_src)
            & (snrna.obs["hlca_label"].astype(str).isin(sorted(EPITHELIAL_LABELS)))
        ].copy()
        for sample_id, sample_cells in source_cells.groupby("sample_id", sort=True, observed=True):
            sample_id = str(sample_id)
            if sample_cells.empty:
                continue
            donor_id = str(sample_cells["donor_id"].iloc[0])
            weak = label_lookup.get((sample_id, str(edge_label)))
            if weak is None:
                continue
            spatial_rows = spatial_df[spatial_df["sample_id"].astype(str) == sample_id].copy()
            typed_rows = np.flatnonzero(typed.obs["sample_id"].astype(str).to_numpy() == sample_id)
            if spatial_rows.empty or typed_rows.size == 0:
                spatial_rows = spatial_df[
                    (spatial_df["donor_id"].astype(str) == donor_id)
                    & (spatial_df["stage"].astype(str).map(normalize_stage_label) == stage_src)
                ].copy()
                typed_rows = np.flatnonzero(
                    (typed.obs["donor_id"].astype(str).to_numpy() == donor_id)
                    & (typed.obs["stage"].astype(str).map(normalize_stage_label).to_numpy() == stage_src)
                )
            if spatial_rows.empty or typed_rows.size == 0:
                continue
            candidate_rows = sample_cells.index.to_numpy(dtype=np.int64, copy=False)
            if candidate_rows.size > max_receiver_cells_per_sample:
                chosen_rows = np.sort(rng.choice(candidate_rows, size=int(max_receiver_cells_per_sample), replace=False))
            else:
                chosen_rows = candidate_rows
            sample_spot_df = spatial_rows.reset_index(drop=True)
            sample_spot_tokens = typed.tokens[typed_rows]
            sample_spot_compositions = spatial.compositions[typed_rows]
            chosen_sender_spots, sender_tokens, sender_dists, chosen_sender_idx = _select_sender_spots(
                sample_spot_df,
                sample_spot_tokens,
                feature_names=typed.schema.typed_feature_names,
                max_anchor_spots=max_anchor_spots,
                max_sender_spots=max_sender_spots,
            )
            sender_compositions = sample_spot_compositions[chosen_sender_idx].astype(np.float32, copy=False)
            sender_coords = chosen_sender_spots[["x", "y"]].to_numpy(dtype=np.float32)
            sender_centroid = sender_coords.mean(axis=0, keepdims=True)
            sender_offsets = sender_coords - sender_centroid
            ring_ids = _distance_to_ring(sender_dists, num_rings=num_distance_rings)
            sender_types = sender_tokens.argmax(axis=1).astype(np.int64, copy=False)
            target_pool = snrna.latent[
                (snrna.obs["donor_id"].astype(str).to_numpy() == donor_id)
                & (snrna.obs["stage"].astype(str).to_numpy() == stage_tgt)
            ]
            if target_pool.shape[0] == 0:
                target_pool = snrna.latent[snrna.obs["stage"].astype(str).to_numpy() == stage_tgt]
            target_latent = (
                target_pool.mean(axis=0).astype(np.float32)
                if target_pool.shape[0] > 0
                else np.zeros(snrna.latent.shape[1], dtype=np.float32)
            )

            examples: list[CommunicationNeighborhoodExample] = []
            for cohort_row in chosen_rows.tolist():
                cell_row = snrna.obs.iloc[int(cohort_row)]
                cell_id = str(cell_row["cell_id"])
                receiver_expression = expression_panel.loc[cell_id]
                receiver_program_scores = _compute_program_scores(receiver_expression)
                receiver_programs = np.asarray(
                    [receiver_program_scores[name] for name in RECEIVER_PROGRAMS],
                    dtype=np.float32,
                )
                lr_tokens, lr_token_names = _build_lr_tokens(
                    sender_compositions,
                    sender_dists,
                    sender_types,
                    ring_ids,
                    receiver_expression,
                    templates,
                    donor_id=donor_id,
                    stage=stage_src,
                    feature_names=spatial.feature_names,
                    max_lr_tokens=max_lr_tokens,
                )
                response_tokens, response_token_names = _build_response_tokens(
                    receiver_program_scores,
                    lr_tokens,
                    edge_id=edge_lookup[str(edge_label)],
                )
                relay_tokens, relay_token_names = _build_relay_tokens(receiver_program_scores, lr_tokens, sender_tokens)
                examples.append(
                    CommunicationNeighborhoodExample(
                        receiver_embedding=np.asarray(snrna.latent[int(cohort_row)], dtype=np.float32),
                        receiver_programs=receiver_programs,
                        sender_embeddings=np.asarray(sender_tokens, dtype=np.float32),
                        sender_types=np.asarray(sender_types, dtype=np.int64),
                        sender_offsets=np.asarray(sender_offsets, dtype=np.float32),
                        ring_ids=np.asarray(ring_ids, dtype=np.int64),
                        lr_token_features=np.asarray(lr_tokens, dtype=np.float32),
                        response_token_features=np.asarray(response_tokens, dtype=np.float32),
                        relay_token_features=np.asarray(relay_tokens, dtype=np.float32),
                        edge_id=int(edge_lookup[str(edge_label)]),
                        sample_id=sample_id,
                        donor_id=donor_id,
                        weak_label=float(weak[0]),
                        receiver_cell_id=cell_id,
                        stage=stage_src,
                        lr_token_names=lr_token_names,
                        response_token_names=response_token_names,
                        relay_token_names=relay_token_names,
                        wes_features=None if not wes_lookup else np.asarray(
                            wes_lookup.get((donor_id, stage_src), wes_default),
                            dtype=np.float32,
                        ),
                        target_latent=target_latent.copy(),
                    )
                )

            if not examples:
                continue
            bag = CommunicationBag(
                sample_id=sample_id,
                donor_id=donor_id,
                edge_id=int(edge_lookup[str(edge_label)]),
                edge_label=str(edge_label),
                weak_label=float(weak[0]),
                examples=examples,
                label_source=str(weak[1]),
                notes=str(weak[2]),
            )
            bags.append(bag)
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "donor_id": donor_id,
                    "stage": stage_src,
                    "edge_label": str(edge_label),
                    "weak_label": float(weak[0]),
                    "label_source": str(weak[1]),
                    "num_queries": len(examples),
                    "num_sender_tokens": int(sender_tokens.shape[0]),
                    "num_lr_tokens": int(examples[0].lr_token_features.shape[0]),
                    "num_response_tokens": int(examples[0].response_token_features.shape[0]),
                    "num_relay_tokens": int(examples[0].relay_token_features.shape[0]),
                }
            )
    bag_table = pd.DataFrame(metadata_rows).sort_values(["edge_label", "sample_id"]).reset_index(drop=True)
    return bags, bag_table


__all__ = [
    "CommunicationPrior",
    "EPITHELIAL_LABELS",
    "FAMILY_TO_PROGRAM",
    "LUNG_LR_PRIORS",
    "RECEIVER_PROGRAMS",
    "build_communication_bags",
    "build_expression_templates",
    "build_progression_label_manifest",
    "communication_gene_panel",
    "load_curated_progression_labels",
    "load_expression_panel",
]
