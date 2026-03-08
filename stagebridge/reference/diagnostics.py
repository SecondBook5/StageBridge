"""Reference-layer diagnostic helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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

    return {
        "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        "centroid_distances": centroid_distances,
        "n_stages": int(len(stage_counts)),
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
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

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
