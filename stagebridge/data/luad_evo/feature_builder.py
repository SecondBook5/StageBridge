"""Compact local niche feature construction for EA-MIST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from stagebridge.data.common.schema import LatentCohort

DEFAULT_EPITHELIAL_LABELS: tuple[str, ...] = (
    "AT2",
    "AT1",
    "Basal",
    "Secretory",
    "Club",
    "Ciliated",
    "Epithelial",
)

LR_FAMILY_PRIORS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "inflammatory": (
        ("IL1B", "IL1R1", 1.00),
        ("IL6", "IL6ST", 0.95),
        ("TNF", "TNFRSF1A", 0.92),
        ("OSM", "OSMR", 0.80),
    ),
    "chemokine": (
        ("CXCL9", "CXCR3", 0.85),
        ("CXCL10", "CXCR3", 0.85),
        ("CXCL12", "CXCR4", 1.00),
        ("CXCL1", "CXCR2", 0.72),
    ),
    "tgfb": (
        ("TGFB1", "TGFBR2", 1.00),
        ("TGFB3", "TGFBR2", 0.80),
    ),
    "growth_factor": (
        ("AREG", "EGFR", 0.95),
        ("EREG", "EGFR", 0.85),
        ("HBEGF", "EGFR", 0.80),
        ("HGF", "MET", 0.90),
    ),
    "ecm": (
        ("SPP1", "ITGAV", 0.78),
        ("COL1A1", "ITGB1", 0.75),
        ("FN1", "ITGB1", 0.82),
    ),
    "vascular": (("VEGFA", "KDR", 0.78),),
}

RECEIVER_PROGRAMS: dict[str, tuple[str, ...]] = {
    "progenitor": ("KRT8", "CEACAM5", "MUC1", "KRT19"),
    "epithelial_identity": ("EPCAM", "MUC1", "KRT19", "CEACAM5"),
    "stress_emt": ("KRT17", "VIM", "ITGB1", "TGFBR2"),
    "inflammatory_response": ("IL1R1", "TNFRSF1A", "CXCR4", "OSMR"),
    "growth_factor_response": ("EGFR", "ERBB2", "MET", "TGFBR2"),
    "migration_invasion": ("CXCR4", "ITGB1", "VIM", "MMP9"),
}


@dataclass(slots=True, frozen=True)
class ExpressionTemplates:
    """Reusable donor/stage/cell-state templates for local feature construction."""

    latent_by_label: dict[str, np.ndarray]
    expression_by_label: dict[str, pd.Series]
    expression_by_donor_stage_label: dict[tuple[str, str, str], pd.Series]


def _safe_log1p_dense(matrix: np.ndarray) -> np.ndarray:
    """Return a log1p-transformed dense float32 matrix."""
    arr = np.asarray(matrix, dtype=np.float32)
    return np.log1p(arr).astype(np.float32, copy=False)


def _build_expression_panel(
    cohort: LatentCohort,
    *,
    genes: Iterable[str],
    raw_h5ad_path: str | None = None,
) -> pd.DataFrame:
    """Load a compact expression panel aligned to the latent cohort cell ids."""
    import anndata
    from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths

    if raw_h5ad_path is None:
        raw_path = resolve_luad_evo_paths({}).snrna_h5ad
    else:
        raw_path = raw_h5ad_path
    raw = anndata.read_h5ad(raw_path, backed="r")
    obs = cohort.obs.copy()
    if "cell_id" not in obs.columns:
        raise KeyError("Latent cohort is missing 'cell_id' needed for expression alignment.")
    cell_ids = obs["cell_id"].astype(str).tolist()
    obs_index = pd.Index(raw.obs_names.astype(str))
    rows = obs_index.get_indexer(cell_ids)
    if np.any(rows < 0):
        missing = [cell_ids[idx] for idx, row in enumerate(rows) if row < 0][:5]
        raise KeyError(f"Could not align latent cell ids to raw snRNA matrix, examples={missing}.")
    gene_list = [str(gene) for gene in genes]
    var_index = pd.Index(raw.var_names.astype(str))
    available = [gene for gene in gene_list if gene in var_index]
    gene_rows = var_index.get_indexer(available)
    dense = _safe_log1p_dense(
        raw.X[rows][:, gene_rows].toarray()
        if hasattr(raw.X[rows][:, gene_rows], "toarray")
        else raw.X[rows][:, gene_rows]
    )
    frame = pd.DataFrame(dense, index=cell_ids, columns=available, dtype=np.float32)
    for gene in gene_list:
        if gene not in frame.columns:
            frame[gene] = np.float32(0.0)
    return frame.loc[:, gene_list].astype(np.float32, copy=False)


def required_panel_genes() -> list[str]:
    """Return the compact gene panel needed for LR-family and receiver programs."""
    genes: set[str] = set()
    for priors in LR_FAMILY_PRIORS.values():
        for ligand, receptor, _support in priors:
            genes.add(str(ligand))
            genes.add(str(receptor))
    for panel in RECEIVER_PROGRAMS.values():
        genes.update(str(gene) for gene in panel)
    return sorted(genes)


def build_expression_templates(
    cohort: LatentCohort,
    *,
    epithelial_labels: Iterable[str] | None = None,
    raw_h5ad_path: str | None = None,
    max_cells_per_group: int | None = 512,
    seed: int = 42,
) -> ExpressionTemplates:
    """Build latent and expression templates for donor/stage/cell-state lookups."""
    obs = cohort.obs.copy()
    required_columns = {"cell_id", "donor_id", "stage", "hlca_label"}
    missing = required_columns.difference(obs.columns)
    if missing:
        raise KeyError(
            f"Latent cohort is missing required columns for EA-MIST templates: {sorted(missing)}"
        )

    epithelial_set = {str(label) for label in (epithelial_labels or DEFAULT_EPITHELIAL_LABELS)}
    mask = obs["hlca_label"].astype(str).isin(epithelial_set).to_numpy()
    if not mask.any():
        mask = np.ones(obs.shape[0], dtype=bool)
    obs = obs.loc[mask].reset_index(drop=True)
    latent = np.asarray(cohort.latent[mask], dtype=np.float32)
    if max_cells_per_group is not None and int(max_cells_per_group) > 0:
        merged_groups = obs[["donor_id", "stage", "hlca_label"]].copy()
        keep_rows: list[np.ndarray] = []
        rng = np.random.default_rng(int(seed))
        for indices in merged_groups.groupby(
            ["donor_id", "stage", "hlca_label"], sort=False
        ).indices.values():
            rows = np.asarray(indices, dtype=np.int64)
            if rows.shape[0] <= int(max_cells_per_group):
                keep_rows.append(rows)
                continue
            keep_rows.append(
                np.sort(rng.choice(rows, size=int(max_cells_per_group), replace=False))
            )
        if keep_rows:
            selected_rows = np.sort(np.concatenate(keep_rows))
            obs = obs.iloc[selected_rows].reset_index(drop=True)
            latent = latent[selected_rows]

    expression_panel = _build_expression_panel(
        LatentCohort(
            latent=latent,
            obs=obs,
            feature_names=cohort.feature_names,
            source_path=cohort.source_path,
            latent_key=cohort.latent_key,
        ),
        genes=required_panel_genes(),
        raw_h5ad_path=raw_h5ad_path,
    )
    merged = obs[["cell_id", "donor_id", "stage", "hlca_label"]].copy()
    merged["cell_id"] = merged["cell_id"].astype(str)
    merged["donor_id"] = merged["donor_id"].astype(str)
    merged["stage"] = merged["stage"].astype(str)
    merged["hlca_label"] = merged["hlca_label"].astype(str)

    latent_by_label: dict[str, np.ndarray] = {}
    expression_by_label: dict[str, pd.Series] = {}
    expression_by_donor_stage_label: dict[tuple[str, str, str], pd.Series] = {}

    for label, label_rows in merged.groupby("hlca_label", sort=False):
        indices = label_rows.index.to_numpy(dtype=np.int64)
        latent_by_label[str(label)] = latent[indices].mean(axis=0).astype(np.float32, copy=False)
        expr = expression_panel.iloc[indices].mean(axis=0).astype(np.float32, copy=False)
        expression_by_label[str(label)] = expr

    donor_stage_label = merged.groupby(["donor_id", "stage", "hlca_label"], sort=False).indices
    for key, indices in donor_stage_label.items():
        expr = (
            expression_panel.iloc[np.asarray(indices, dtype=np.int64)]
            .mean(axis=0)
            .astype(np.float32, copy=False)
        )
        expression_by_donor_stage_label[(str(key[0]), str(key[1]), str(key[2]))] = expr

    return ExpressionTemplates(
        latent_by_label=latent_by_label,
        expression_by_label=expression_by_label,
        expression_by_donor_stage_label=expression_by_donor_stage_label,
    )


def epithelial_columns(feature_names: Iterable[str]) -> list[int]:
    """Return indices for epithelial-like columns from spatial composition names."""
    labels = [str(name) for name in feature_names]
    epithelial_set = {str(label) for label in DEFAULT_EPITHELIAL_LABELS}
    return [idx for idx, label in enumerate(labels) if label in epithelial_set]


def infer_receiver_state(
    center_composition: np.ndarray,
    feature_names: Iterable[str],
) -> tuple[int, str, float]:
    """Infer a compact receiver state from center-spot composition."""
    weights = np.asarray(center_composition, dtype=np.float32)
    names = [str(name) for name in feature_names]
    if weights.ndim != 1 or weights.shape[0] != len(names):
        raise ValueError(
            "Receiver-state inference requires a 1D composition vector aligned to feature names."
        )
    epi_cols = epithelial_columns(names)
    if epi_cols:
        chosen_cols = epi_cols
    else:
        chosen_cols = list(range(len(names)))
    local = weights[chosen_cols]
    winner = int(np.argmax(local)) if local.size else 0
    original_idx = int(chosen_cols[winner]) if chosen_cols else 0
    label = names[original_idx] if names else "unknown"
    score = float(local[winner]) if local.size else 0.0
    return original_idx, str(label), score


def summarize_ring_compositions(
    sample_compositions: np.ndarray,
    sample_coords: np.ndarray,
    *,
    center_index: int,
    ring_edges: list[float],
) -> np.ndarray:
    """Summarize neighborhood sender composition in distance rings around one spot."""
    compositions = np.asarray(sample_compositions, dtype=np.float32)
    coords = np.asarray(sample_coords, dtype=np.float32)
    if compositions.ndim != 2:
        raise ValueError(f"sample_compositions must be 2D, got shape={compositions.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"sample_coords must have shape (n, 2), got {coords.shape}")
    if coords.shape[0] != compositions.shape[0]:
        raise ValueError("coords rows must match composition rows.")
    if center_index < 0 or center_index >= coords.shape[0]:
        raise IndexError(
            f"center_index {center_index} is out of bounds for {coords.shape[0]} spots."
        )
    if len(ring_edges) < 2:
        raise ValueError("ring_edges must define at least one ring boundary.")

    center = coords[center_index]
    dists = np.linalg.norm(coords - center[None, :], axis=1)
    num_rings = len(ring_edges) - 1
    summaries = np.zeros((num_rings, compositions.shape[1]), dtype=np.float32)
    for ring_idx in range(num_rings):
        low = float(ring_edges[ring_idx])
        high = float(ring_edges[ring_idx + 1])
        mask = (
            (dists >= low) & (dists < high)
            if ring_idx < num_rings - 1
            else (dists >= low) & (dists <= high)
        )
        if not mask.any():
            summaries[ring_idx] = compositions[center_index]
        else:
            summaries[ring_idx] = compositions[mask].mean(axis=0).astype(np.float32, copy=False)
    return summaries


def _lookup_expression(
    templates: ExpressionTemplates,
    *,
    donor_id: str,
    stage: str,
    label: str,
) -> pd.Series:
    """Resolve the best available donor/stage/cell-state expression template."""
    key = (str(donor_id), str(stage), str(label))
    if key in templates.expression_by_donor_stage_label:
        return templates.expression_by_donor_stage_label[key]
    if str(label) in templates.expression_by_label:
        return templates.expression_by_label[str(label)]
    return pd.Series(0.0, index=required_panel_genes(), dtype=np.float32)


def build_receiver_embedding(
    center_composition: np.ndarray,
    feature_names: Iterable[str],
    templates: ExpressionTemplates,
) -> tuple[np.ndarray, str, float]:
    """Build a pseudo-receiver latent embedding from epithelial mixture weights."""
    weights = np.asarray(center_composition, dtype=np.float32)
    names = [str(name) for name in feature_names]
    epi_cols = epithelial_columns(names)
    if not epi_cols:
        epi_cols = list(range(len(names)))
    chosen_weights = weights[epi_cols]
    if float(chosen_weights.sum()) <= 0.0:
        chosen_weights = np.ones(len(epi_cols), dtype=np.float32) / max(len(epi_cols), 1)
    else:
        chosen_weights = chosen_weights / np.float32(chosen_weights.sum())
    dim = next(iter(templates.latent_by_label.values())).shape[0]
    receiver = np.zeros(dim, dtype=np.float32)
    winning_label = names[epi_cols[int(np.argmax(chosen_weights))]]
    winning_score = float(chosen_weights.max(initial=0.0))
    for local_idx, col_idx in enumerate(epi_cols):
        label = names[col_idx]
        template = templates.latent_by_label.get(label)
        if template is None:
            continue
        receiver += chosen_weights[local_idx] * template.astype(np.float32, copy=False)
    return receiver.astype(np.float32, copy=False), str(winning_label), winning_score


def build_lr_pathway_summary(
    ring_compositions: np.ndarray,
    feature_names: Iterable[str],
    templates: ExpressionTemplates,
    *,
    donor_id: str,
    stage: str,
    receiver_label: str,
) -> np.ndarray:
    """Build compact LR-family and receiver-program summaries for one niche."""
    names = [str(name) for name in feature_names]
    ring_mean = np.asarray(ring_compositions, dtype=np.float32).mean(axis=0)
    receiver_expr = _lookup_expression(
        templates, donor_id=donor_id, stage=stage, label=receiver_label
    )
    family_scores: list[float] = []
    for family_name, priors in LR_FAMILY_PRIORS.items():
        per_prior: list[float] = []
        for ligand, receptor, support in priors:
            ligand_score = 0.0
            for idx, label in enumerate(names[: ring_mean.shape[0]]):
                expr = _lookup_expression(templates, donor_id=donor_id, stage=stage, label=label)
                ligand_score += float(ring_mean[idx]) * float(expr.get(ligand, 0.0))
            receptor_score = float(receiver_expr.get(receptor, 0.0))
            per_prior.append(float(ligand_score * receptor_score * support))
        family_scores.append(float(np.mean(per_prior)) if per_prior else 0.0)
    program_scores = [
        float(np.mean([float(receiver_expr.get(gene, 0.0)) for gene in genes]))
        for genes in RECEIVER_PROGRAMS.values()
    ]
    return np.asarray([*family_scores, *program_scores], dtype=np.float32)


def build_neighborhood_stats(
    center_composition: np.ndarray,
    ring_compositions: np.ndarray,
    *,
    receiver_confidence: float,
    local_density: float,
) -> np.ndarray:
    """Build compact density/diversity/uncertainty summary features."""
    center = np.asarray(center_composition, dtype=np.float32)
    ring_mean = np.asarray(ring_compositions, dtype=np.float32).mean(axis=0)
    safe_center = np.clip(center, 1e-8, 1.0)
    safe_ring = np.clip(ring_mean, 1e-8, 1.0)
    center_entropy = float(-(safe_center * np.log(safe_center)).sum())
    ring_entropy = float(-(safe_ring * np.log(safe_ring)).sum())
    diversity = float(np.count_nonzero(center > 0.05) / max(center.shape[0], 1))
    epithelial_mass = float(center.sum())
    stats = np.asarray(
        [
            float(local_density),
            float(center_entropy),
            float(ring_entropy),
            float(diversity),
            float(receiver_confidence),
            float(epithelial_mass),
        ],
        dtype=np.float32,
    )
    return stats


def flatten_neighborhood_features(
    receiver_embedding: np.ndarray,
    ring_compositions: np.ndarray,
    hlca_features: np.ndarray | None,
    luca_features: np.ndarray | None,
    lr_pathway_summary: np.ndarray,
    neighborhood_stats: np.ndarray,
) -> np.ndarray:
    """Flatten the structured neighborhood representation for MLP/SSL usage."""
    pieces = [
        np.asarray(receiver_embedding, dtype=np.float32).reshape(-1),
        np.asarray(ring_compositions, dtype=np.float32).reshape(-1),
    ]
    if hlca_features is not None:
        pieces.append(np.asarray(hlca_features, dtype=np.float32).reshape(-1))
    if luca_features is not None:
        pieces.append(np.asarray(luca_features, dtype=np.float32).reshape(-1))
    pieces.extend(
        [
            np.asarray(lr_pathway_summary, dtype=np.float32).reshape(-1),
            np.asarray(neighborhood_stats, dtype=np.float32).reshape(-1),
        ]
    )
    return np.concatenate(
        pieces,
        axis=0,
    ).astype(np.float32, copy=False)


def summarize_neighborhood_build(
    bags: Iterable[object],
) -> dict[str, float]:
    """Return compact summary diagnostics for lesion-bag preprocessing."""
    num_bags = 0
    num_instances = 0
    neighborhoods_per_bag: list[int] = []
    for bag in bags:
        num_bags += 1
        count = int(getattr(bag, "num_neighborhoods", len(getattr(bag, "neighborhoods", []))))
        num_instances += count
        neighborhoods_per_bag.append(count)
    return {
        "num_bags": float(num_bags),
        "num_instances": float(num_instances),
        "mean_neighborhoods_per_bag": float(np.mean(neighborhoods_per_bag))
        if neighborhoods_per_bag
        else 0.0,
        "median_neighborhoods_per_bag": float(np.median(neighborhoods_per_bag))
        if neighborhoods_per_bag
        else 0.0,
    }
