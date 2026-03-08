"""Reference-layer diagnostic helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def summarize_latent(embedding: Any) -> dict[str, object]:
    """Return a small numeric summary for a latent matrix-like object."""
    arr = np.asarray(embedding, dtype=np.float32)
    return {
        "shape": tuple(arr.shape),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def stage_preservation_diagnostics(
    latent: np.ndarray,
    obs: pd.DataFrame,
    *,
    stage_col: str = "stage",
    max_cells: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute lightweight stage-separation diagnostics for the latent space."""
    arr = np.asarray(latent, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D latent array, got shape={arr.shape}.")
    if stage_col not in obs.columns:
        raise KeyError(f"Missing stage column '{stage_col}' in obs.")

    stage_counts = obs.groupby(stage_col).size().to_dict()
    centroids: dict[str, np.ndarray] = {}
    for stage_name, idx in obs.groupby(stage_col).groups.items():
        rows = np.asarray(list(idx), dtype=np.int64)
        centroids[str(stage_name)] = arr[rows].mean(axis=0)

    centroid_distances: dict[str, float] = {}
    stage_names = sorted(centroids)
    for i, src in enumerate(stage_names):
        for tgt in stage_names[i + 1 :]:
            centroid_distances[f"{src}->{tgt}"] = float(np.linalg.norm(centroids[src] - centroids[tgt]))

    probe = _stage_probe_diagnostics(
        arr,
        obs[stage_col].astype(str).to_numpy(),
        max_cells=max_cells,
        seed=seed,
    )

    return {
        "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        "centroid_distances": centroid_distances,
        "n_stages": int(len(stage_counts)),
        "probe": probe,
    }


def donor_leakage_diagnostics(
    latent: np.ndarray,
    obs: pd.DataFrame,
    *,
    donor_col: str = "donor_id",
    max_cells: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate residual donor leakage from the latent space."""
    if donor_col not in obs.columns:
        raise KeyError(f"Missing donor column '{donor_col}' in obs.")

    arr = np.asarray(latent, dtype=np.float32)
    donors = obs[donor_col].astype(str).to_numpy()
    if arr.shape[0] != donors.shape[0]:
        raise ValueError("latent rows and obs rows must match.")

    rng = np.random.default_rng(int(seed))
    if arr.shape[0] > max_cells:
        keep = rng.choice(arr.shape[0], size=int(max_cells), replace=False)
        arr = arr[keep]
        donors = donors[keep]

    unique_donors = np.unique(donors)
    chance = 1.0 / max(len(unique_donors), 1)
    if len(unique_donors) < 2:
        return {
            "n_donors": int(len(unique_donors)),
            "chance_accuracy": float(chance),
            "logreg_accuracy": float("nan"),
        }
    counts = pd.Series(donors).value_counts()
    if int(counts.min()) < 2:
        return {
            "n_donors": int(len(unique_donors)),
            "chance_accuracy": float(chance),
            "logreg_accuracy": float("nan"),
            "status": "insufficient_per_donor_counts",
        }

    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        arr,
        donors,
        test_size=0.25,
        random_state=int(seed),
        stratify=donors,
    )
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return {
        "n_donors": int(len(unique_donors)),
        "chance_accuracy": float(chance),
        "logreg_accuracy": float(accuracy_score(y_test, pred)),
    }


def gene_overlap_diagnostics(
    *,
    query_h5ad_path: str | Path | None,
    reference_h5ad_path: str | Path | None,
) -> dict[str, Any]:
    """Compare query and reference gene coverage."""
    if query_h5ad_path is None or reference_h5ad_path is None:
        return {
            "status": "missing_paths",
            "query_gene_count": 0,
            "reference_gene_count": 0,
            "shared_gene_count": 0,
            "reference_query_overlap_fraction": float("nan"),
            "missing_gene_fraction": float("nan"),
        }
    query_path = Path(query_h5ad_path)
    reference_path = Path(reference_h5ad_path)
    if not query_path.exists() or not reference_path.exists():
        return {
            "status": "missing_files",
            "query_gene_count": 0,
            "reference_gene_count": 0,
            "shared_gene_count": 0,
            "reference_query_overlap_fraction": float("nan"),
            "missing_gene_fraction": float("nan"),
            "query_path": str(query_path),
            "reference_path": str(reference_path),
        }

    query = anndata.read_h5ad(query_path, backed="r")
    reference = anndata.read_h5ad(reference_path, backed="r")
    try:
        query_genes = pd.Index(query.var_names.astype(str))
        reference_genes = pd.Index(reference.var_names.astype(str))
        shared = query_genes.intersection(reference_genes)
    finally:
        try:
            query.file.close()
        except Exception:
            pass
        try:
            reference.file.close()
        except Exception:
            pass

    query_n = int(query_genes.shape[0])
    ref_n = int(reference_genes.shape[0])
    shared_n = int(shared.shape[0])
    overlap_fraction = float(shared_n / max(query_n, 1))
    return {
        "status": "complete",
        "query_gene_count": query_n,
        "reference_gene_count": ref_n,
        "shared_gene_count": shared_n,
        "reference_query_overlap_fraction": overlap_fraction,
        "missing_gene_fraction": float(1.0 - overlap_fraction),
    }


def nearest_neighbor_label_agreement(
    latent: np.ndarray,
    obs: pd.DataFrame,
    *,
    label_col: str = "hlca_label",
    n_neighbors: int = 10,
) -> dict[str, Any]:
    """Estimate local label consistency in the latent space."""
    if label_col not in obs.columns:
        return {
            "status": "missing_labels",
            "n_labeled_cells": 0,
            "mean_neighbor_label_agreement": float("nan"),
        }
    labels = obs[label_col]
    valid = labels.notna() & labels.astype(str).ne("") & labels.astype(str).ne("nan")
    if int(valid.sum()) < 3:
        return {
            "status": "insufficient_labels",
            "n_labeled_cells": int(valid.sum()),
            "mean_neighbor_label_agreement": float("nan"),
        }

    arr = np.asarray(latent[valid.to_numpy()], dtype=np.float32)
    kept = labels.loc[valid].astype(str).to_numpy()
    k = max(2, min(int(n_neighbors), arr.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(arr)
    indices = nn.kneighbors(arr, return_distance=False)[:, 1:]
    agreement = np.mean(kept[indices] == kept[:, None], axis=1)
    return {
        "status": "complete",
        "n_labeled_cells": int(arr.shape[0]),
        "n_neighbors": int(k),
        "mean_neighbor_label_agreement": float(np.mean(agreement)),
        "median_neighbor_label_agreement": float(np.median(agreement)),
    }


def stage_label_alignment(
    obs: pd.DataFrame,
    *,
    stage_col: str = "stage",
    label_col: str = "hlca_label",
    top_n_labels: int = 8,
) -> dict[str, Any]:
    """Summarize stage-vs-label alignment as a compact confusion payload."""
    if stage_col not in obs.columns or label_col not in obs.columns:
        return {
            "status": "missing_columns",
            "rows": [],
            "cols": [],
            "matrix": [],
            "normalized_matrix": [],
        }
    frame = obs[[stage_col, label_col]].copy()
    frame = frame.loc[
        frame[stage_col].notna()
        & frame[label_col].notna()
        & frame[label_col].astype(str).ne("")
        & frame[label_col].astype(str).ne("nan")
    ]
    if frame.empty:
        return {
            "status": "no_overlap",
            "rows": [],
            "cols": [],
            "matrix": [],
            "normalized_matrix": [],
        }
    frame[stage_col] = frame[stage_col].astype(str)
    frame[label_col] = frame[label_col].astype(str)
    top_labels = frame[label_col].value_counts().head(int(top_n_labels)).index.astype(str).tolist()
    table = pd.crosstab(frame[stage_col], frame[label_col]).reindex(columns=top_labels, fill_value=0)
    row_norm = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    dominant = {
        str(stage): {
            "dominant_label": str(row.idxmax()),
            "dominant_fraction": float(row.max() / max(row.sum(), 1)),
        }
        for stage, row in table.iterrows()
    }
    return {
        "status": "complete",
        "rows": [str(idx) for idx in table.index.tolist()],
        "cols": [str(col) for col in table.columns.tolist()],
        "matrix": table.to_numpy(dtype=np.float32).tolist(),
        "normalized_matrix": row_norm.to_numpy(dtype=np.float32).tolist(),
        "dominant_label_per_stage": dominant,
    }


def reference_alignment_gate(
    *,
    stage_preservation: dict[str, Any],
    donor_leakage: dict[str, Any],
    label_transfer: dict[str, Any],
    gene_overlap: dict[str, Any],
    label_neighborhood: dict[str, Any],
) -> dict[str, Any]:
    """Assign a conservative gate status to the reference latent branch."""
    balanced = float(stage_preservation.get("probe", {}).get("balanced_accuracy", float("nan")))
    chance = float(stage_preservation.get("probe", {}).get("chance_accuracy", float("nan")))
    donor_acc = float(donor_leakage.get("logreg_accuracy", float("nan")))
    donor_chance = float(donor_leakage.get("chance_accuracy", float("nan")))
    coverage = float(label_transfer.get("coverage", float("nan")))
    overlap = float(gene_overlap.get("reference_query_overlap_fraction", float("nan")))
    nn_agreement = float(label_neighborhood.get("mean_neighbor_label_agreement", float("nan")))

    status = "fail"
    interpretation = "Reference alignment is too weak to trust downstream conditioning."
    recommended_action = "remove_from_v1_claims"
    if (
        np.isfinite(balanced)
        and np.isfinite(chance)
        and balanced >= chance + 0.15
        and (not np.isfinite(donor_acc) or not np.isfinite(donor_chance) or donor_acc <= donor_chance + 0.20)
        and (not np.isfinite(coverage) or coverage >= 0.90)
        and (not np.isfinite(overlap) or overlap >= 0.50)
        and (not np.isfinite(nn_agreement) or nn_agreement >= 0.45)
    ):
        status = "pass"
        interpretation = "Reference latent preserves stage structure above chance without dominant donor leakage."
        recommended_action = "keep"
    elif (
        np.isfinite(balanced)
        and np.isfinite(chance)
        and balanced >= chance + 0.05
        and (not np.isfinite(coverage) or coverage >= 0.75)
    ):
        status = "weak_pass"
        interpretation = "Reference latent carries useful stage signal, but residual leakage or label instability still warrants caution."
        recommended_action = "keep_as_optional"

    return {
        "status": status,
        "interpretation": interpretation,
        "recommended_action": recommended_action,
        "metrics": {
            "stage_probe_balanced_accuracy": balanced,
            "stage_probe_chance": chance,
            "donor_leakage_accuracy": donor_acc,
            "donor_leakage_chance": donor_chance,
            "label_coverage": coverage,
            "gene_overlap_fraction": overlap,
            "neighbor_label_agreement": nn_agreement,
        },
    }


def _stage_probe_diagnostics(
    latent: np.ndarray,
    stages: np.ndarray,
    *,
    max_cells: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate how well stage identity is preserved in the latent space."""
    arr = np.asarray(latent, dtype=np.float32)
    labels = np.asarray(stages, dtype=object).astype(str)
    if arr.shape[0] != labels.shape[0]:
        raise ValueError("latent rows and stage labels must match.")

    rng = np.random.default_rng(int(seed))
    if arr.shape[0] > max_cells:
        keep = rng.choice(arr.shape[0], size=int(max_cells), replace=False)
        arr = arr[keep]
        labels = labels[keep]

    unique_labels = np.unique(labels)
    counts = pd.Series(labels).value_counts()
    chance = 1.0 / max(len(unique_labels), 1)
    result: dict[str, Any] = {
        "n_stages": int(len(unique_labels)),
        "chance_accuracy": float(chance),
    }
    if len(unique_labels) < 2:
        result["logreg_accuracy"] = float("nan")
        result["balanced_accuracy"] = float("nan")
        result["status"] = "single_stage_only"
        return result
    if int(counts.min()) < 2:
        result["logreg_accuracy"] = float("nan")
        result["balanced_accuracy"] = float("nan")
        result["status"] = "insufficient_per_stage_counts"
        return result

    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split(
        arr,
        labels,
        test_size=0.25,
        random_state=int(seed),
        stratify=labels,
    )
    clf = LogisticRegression(max_iter=400)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    result["logreg_accuracy"] = float(accuracy_score(y_test, pred))
    result["balanced_accuracy"] = float(balanced_accuracy_score(y_test, pred))
    result["status"] = "ok"
    return result
