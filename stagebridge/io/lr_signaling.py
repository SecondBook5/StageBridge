"""Ligand-receptor signaling feature computation for StageBridge.

Computes per-(patient, stage) ligand-receptor interaction scores from spatial
Visium data using a product approximation:
    LR_score(spot) = L_expression(spot) × R_expression(spot)

Aggregated statistics (mean, max, fraction_active) over spots provide
interpretable per-sample LR signaling features that condition the drift head
alongside WES somatic features.

Reference LR pairs are curated for lung adenocarcinoma progression with
known roles in angiogenesis, EMT, immune evasion, and stromal remodeling.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Curated lung adenocarcinoma LR pair database
# Format: (ligand, receptor, pathway, biological_role)
# ---------------------------------------------------------------------------
LUNG_LR_PAIRS: list[tuple[str, str, str, str]] = [
    # Angiogenesis
    ("VEGFA", "KDR",     "VEGF",    "angiogenesis"),
    ("VEGFA", "FLT1",    "VEGF",    "angiogenesis"),
    ("ANGPT2", "TEK",    "ANGPT",   "angiogenesis"),
    # EMT / immunosuppression
    ("TGFB1", "TGFBR1",  "TGF-beta", "EMT"),
    ("TGFB1", "TGFBR2",  "TGF-beta", "EMT"),
    ("TGFB2", "TGFBR1",  "TGF-beta", "EMT"),
    # Proliferation / EGFR
    ("EGF",   "EGFR",    "EGF",     "proliferation"),
    ("AREG",  "EGFR",    "EGF",     "proliferation"),
    ("EREG",  "EGFR",    "EGF",     "proliferation"),
    # Migration / invasion
    ("CXCL12","CXCR4",   "CXCL12",  "migration"),
    ("HGF",   "MET",     "HGF",     "invasion"),
    ("SPP1",  "CD44",    "SPP1",    "invasion"),
    # Immune recruitment / evasion
    ("CCL2",  "CCR2",    "CCL2",    "immune_recruitment"),
    ("CXCL10","CXCR3",   "CXCL10",  "immune_recruitment"),
    ("CD274", "PDCD1",   "PDL1-PD1","immune_checkpoint"),
    ("LGALS9","HAVCR2",  "TIM3",    "immune_checkpoint"),
    ("CD80",  "CTLA4",   "CD80",    "immune_checkpoint"),
    # Stromal / fibroblast activation
    ("PDGFA", "PDGFRA",  "PDGF",    "stromal_activation"),
    ("PDGFB", "PDGFRB",  "PDGF",    "stromal_activation"),
    ("FGF2",  "FGFR1",   "FGF",     "stromal_activation"),
    # Stemness / Wnt
    ("WNT5A", "FZD1",    "WNT",     "stemness"),
    ("WNT3A", "FZD4",    "WNT",     "stemness"),
    # Notch / cell fate
    ("JAG1",  "NOTCH1",  "NOTCH",   "cell_fate"),
    ("DLL4",  "NOTCH1",  "NOTCH",   "cell_fate"),
]

# Column name generated from each LR pair
LR_FEATURE_COLS: list[str] = [f"{ligand}_{receptor}" for ligand, receptor, _, _ in LUNG_LR_PAIRS]

# Columns in the raw obs/metadata expected on spatial AnnData
_DONOR_COL = "patient_id"
_STAGE_COL = "stage"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_lr_scores_from_spatial(
    adata_spatial: Any,
    lr_pairs: list[tuple[str, str, str, str]] = LUNG_LR_PAIRS,
    layer: str = "log1p",
    donor_col: str = _DONOR_COL,
    stage_col: str = _STAGE_COL,
    active_threshold: float = 0.1,
) -> pd.DataFrame:
    """Compute per-spot LR interaction scores and aggregate by patient × stage.

    Parameters
    ----------
    adata_spatial : AnnData
        Visium data with log1p-normalised expression in ``layers[layer]``.
        Must have ``obs[donor_col]`` and ``obs[stage_col]``.
    lr_pairs : list
        LR pairs to score.  Pairs where one or both genes are absent in the
        feature matrix are silently skipped with a debug-level log.
    layer : str
        Expression layer to use (should be log1p-normalised).
    donor_col, stage_col : str
        Column names in ``adata_spatial.obs``.
    active_threshold : float
        Expression threshold above which a spot is considered "active" for
        computing the *fraction_active* statistic.

    Returns
    -------
    pd.DataFrame
        Rows = (patient_id, stage) groups.
        Columns = per-LR-pair statistics (``{L}_{R}_mean``, ``{L}_{R}_max``,
        ``{L}_{R}_frac``).
    """
    import scipy.sparse as sp

    if layer not in adata_spatial.layers:
        raise KeyError(
            f"Layer '{layer}' not in adata_spatial.layers. "
            f"Available: {list(adata_spatial.layers.keys())}"
        )
    if donor_col not in adata_spatial.obs.columns:
        raise KeyError(f"Donor column '{donor_col}' missing from adata_spatial.obs.")
    if stage_col not in adata_spatial.obs.columns:
        raise KeyError(f"Stage column '{stage_col}' missing from adata_spatial.obs.")

    gene_names = np.asarray(adata_spatial.var_names)
    gene_index: dict[str, int] = {g: i for i, g in enumerate(gene_names)}

    # Build (N_spots, N_genes) dense float32 matrix — Visium spots are few (~3k)
    X = adata_spatial.layers[layer]
    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    donors = np.asarray(adata_spatial.obs[donor_col])
    stages = np.asarray(adata_spatial.obs[stage_col])

    rows: list[dict] = []
    for donor in np.unique(donors):
        for stage in np.unique(stages[donors == donor]):
            mask = (donors == donor) & (stages == stage)
            X_group = X[mask]  # (n_spots, n_genes)

            row: dict = {"patient_id": donor, "stage": stage}
            for lig, rec, _, _ in lr_pairs:
                col = f"{lig}_{rec}"
                l_idx = gene_index.get(lig.upper()) if lig.upper() in gene_index else gene_index.get(lig)
                r_idx = gene_index.get(rec.upper()) if rec.upper() in gene_index else gene_index.get(rec)

                if l_idx is None or r_idx is None:
                    row[f"{col}_mean"] = float("nan")
                    row[f"{col}_max"] = float("nan")
                    row[f"{col}_frac"] = float("nan")
                    continue

                lr_scores = X_group[:, l_idx] * X_group[:, r_idx]  # (n_spots,)
                row[f"{col}_mean"] = float(lr_scores.mean())
                row[f"{col}_max"]  = float(lr_scores.max())
                row[f"{col}_frac"] = float((lr_scores > active_threshold).mean())

            rows.append(row)

    df = pd.DataFrame(rows)
    log.info(
        "Computed LR scores for %d (patient, stage) groups, %d LR pairs.",
        len(df),
        len(lr_pairs),
    )
    return df


def build_lr_feature_matrix(
    adata_spatial: Any,
    lr_pairs: list[tuple[str, str, str, str]] = LUNG_LR_PAIRS,
    stat: str = "mean",
    output_path: Path | str | None = None,
    layer: str = "log1p",
) -> pd.DataFrame:
    """Build a compact per-(patient_id, stage) LR feature matrix.

    Parameters
    ----------
    adata_spatial : AnnData
        Visium spatial data with expression and metadata.
    lr_pairs : list
        LR pairs to include.
    stat : {"mean", "max", "frac"}
        Which per-group statistic to use as the feature value.
    output_path : path-like or None
        If provided, saves the DataFrame as a Parquet file.
    layer : str
        Expression layer.

    Returns
    -------
    pd.DataFrame
        Columns: patient_id, stage, + one column per LR pair (feature values).
        NaN values (missing genes) are filled with 0.
    """
    if stat not in {"mean", "max", "frac"}:
        raise ValueError(f"stat must be 'mean', 'max', or 'frac'; got '{stat}'.")

    full_df = compute_lr_scores_from_spatial(
        adata_spatial=adata_spatial,
        lr_pairs=lr_pairs,
        layer=layer,
    )

    # Select the requested statistic columns
    id_cols = ["patient_id", "stage"]
    stat_cols = [f"{ligand}_{receptor}_{stat}" for ligand, receptor, _, _ in lr_pairs]
    present_stat_cols = [c for c in stat_cols if c in full_df.columns]
    compact = full_df[id_cols + present_stat_cols].copy()

    # Rename: {l}_{r}_{stat} → {l}_{r}
    rename_map = {
        f"{ligand}_{receptor}_{stat}": f"{ligand}_{receptor}"
        for ligand, receptor, _, _ in lr_pairs
    }
    compact = compact.rename(columns=rename_map)
    compact = compact.fillna(0.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compact.to_parquet(output_path, index=False)
        log.info("Saved LR feature matrix to %s (%d rows).", output_path, len(compact))

    return compact


def load_lr_features(
    parquet_path: Path | str,
    feature_cols: list[str] | None = None,
) -> dict[tuple[str, str], np.ndarray]:
    """Load LR features from Parquet into a fast lookup dict.

    Parameters
    ----------
    parquet_path : path-like
        Path to the LR feature Parquet file produced by :func:`build_lr_feature_matrix`.
    feature_cols : list[str] or None
        Subset of LR feature columns to load.  Defaults to all LR columns
        (excludes ``patient_id`` and ``stage``).

    Returns
    -------
    dict[(patient_id, stage), np.ndarray]
        Maps ``(patient_id, stage)`` tuples to 1-D float32 feature arrays.
    """
    df = pd.read_parquet(parquet_path)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("patient_id", "stage")]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("LR feature columns not found in Parquet (filling zeros): %s", missing)
        for c in missing:
            df[c] = 0.0

    lookup: dict[tuple[str, str], np.ndarray] = {}
    for _, row in df.iterrows():
        key = (str(row["patient_id"]), str(row["stage"]))
        lookup[key] = row[feature_cols].to_numpy(dtype=np.float32)

    log.info(
        "Loaded LR features: %d (patient, stage) entries, %d features.",
        len(lookup),
        len(feature_cols),
    )
    return lookup


def get_lr_feature_dim(lr_pairs: list[tuple[str, str, str, str]] = LUNG_LR_PAIRS) -> int:
    """Return the number of LR feature dimensions for the given pair list."""
    return len(lr_pairs)


def lr_pathway_groups(
    lr_pairs: list[tuple[str, str, str, str]] = LUNG_LR_PAIRS,
) -> dict[str, list[int]]:
    """Return a dict mapping pathway name to feature column indices."""
    groups: dict[str, list[int]] = {}
    for i, (_, _, pathway, _) in enumerate(lr_pairs):
        groups.setdefault(pathway, []).append(i)
    return groups
