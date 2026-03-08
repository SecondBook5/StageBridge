"""Reference label-transfer helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd


def transfer_reference_labels(obs: pd.DataFrame, *, label_col: str = "hlca_label") -> dict[str, Any]:
    """Expose the active reference labels without hiding coverage or missingness."""
    if label_col not in obs.columns:
        return {
            "ok": False,
            "status": "missing_labels",
            "label_col": label_col,
            "coverage": 0.0,
            "top_labels": [],
        }
    raw = obs[label_col]
    valid = raw.notna() & raw.astype(str).ne("") & raw.astype(str).ne("nan")
    labels = raw.loc[valid].astype(str)
    counts = labels.value_counts().head(10)
    coverage = float(valid.mean())
    return {
        "ok": True,
        "status": "complete",
        "label_col": label_col,
        "coverage": coverage,
        "top_labels": [(str(idx), int(val)) for idx, val in counts.items()],
    }
