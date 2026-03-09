"""Target-support and split-viability checks for label-repair outputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from stagebridge.labels.common_schema import DONOR_SUPPORT_COLUMNS, EDGE_SUPPORT_COLUMNS


def _cfg_select(cfg: DictConfig | dict[str, Any], dotted: str, default: Any) -> Any:
    """Read a dotted config value from OmegaConf or dict payloads.

    Args:
        cfg: Config tree.
        dotted: Dotted key path.
        default: Fallback when the key is missing.
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


def _binary_support(relevant: pd.DataFrame, *, num_folds: int) -> tuple[bool, str]:
    """Return binary viability and explanation for one edge label.

    Args:
        relevant: Refined label subset for one edge.
        num_folds: Requested donor-held-out fold count.
    """
    usable = relevant.loc[
        (~relevant["exclusion_flag"].astype(bool))
        & (~relevant["uncertainty_flag"].astype(bool))
        & (relevant["refined_binary_label"].isin(["positive", "negative"]))
    ].copy()
    donor_support = (
        usable.groupby(["refined_binary_label"], sort=False)["donor_id"]
        .nunique()
        .to_dict()
    )
    positive_donors = int(donor_support.get("positive", 0))
    negative_donors = int(donor_support.get("negative", 0))
    if positive_donors == 0 or negative_donors == 0:
        return False, "At least one class has zero donor support after refinement."
    if positive_donors < int(num_folds) or negative_donors < int(num_folds):
        return False, (
            f"Donor-held-out {num_folds}-fold CV requires at least {num_folds} donors per class. "
            f"Observed positive_donors={positive_donors}, negative_donors={negative_donors}."
        )
    return True, "Both classes have enough donor support for donor-held-out binary evaluation."


def _continuous_support(relevant: pd.DataFrame) -> tuple[bool, str]:
    """Return continuous-target viability and explanation for one edge label.

    Args:
        relevant: Refined label subset for one edge.
    """
    usable = relevant.loc[~relevant["exclusion_flag"].astype(bool)].copy()
    unique_scores = pd.to_numeric(usable["progression_risk_score"], errors="coerce").dropna().nunique()
    donor_count = usable["donor_id"].astype(str).nunique()
    if usable.shape[0] < 5:
        return False, "Too few usable lesions for a continuous target."
    if unique_scores < 3:
        return False, "Too few unique risk scores for a continuous target."
    if donor_count < 3:
        return False, "Too few donors for a stable continuous target."
    return True, "Continuous risk target is supported by lesion count, donor count, and score diversity."


def evaluate_label_support(
    manifest: pd.DataFrame,
    refined_labels: pd.DataFrame,
    cfg: DictConfig | dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate binary and continuous target viability after refinement.

    Args:
        manifest: Cleaned lesion manifest.
        refined_labels: Refined label table.
        cfg: Active config tree.
    """
    num_folds = int(_cfg_select(cfg, "labels.viability.num_folds", 3))
    edge_rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    split_report: dict[str, Any] = {"requested_num_folds": num_folds, "edges": {}}

    for edge_label in sorted(refined_labels["edge_label"].dropna().astype(str).unique().tolist()):
        relevant = refined_labels.loc[refined_labels["edge_label"].astype(str) == edge_label].copy()
        binary_viable, binary_reason = _binary_support(relevant, num_folds=num_folds)
        continuous_viable, continuous_reason = _continuous_support(relevant)
        recommended = "exclude"
        reason = binary_reason
        if binary_viable:
            recommended = "binary_classification"
            reason = binary_reason
        elif continuous_viable:
            recommended = "continuous_risk"
            reason = continuous_reason
        elif relevant.loc[~relevant["exclusion_flag"].astype(bool)].shape[0] > 0:
            recommended = "descriptive_only"
            reason = "Edge retains lesions for descriptive analysis, but target support is insufficient for supervised evaluation."

        donor_support_frame = relevant.groupby("donor_id", sort=False).agg(
            n_lesions=("lesion_id", "nunique"),
            positive_lesions=("refined_binary_label", lambda values: int(pd.Series(values).eq("positive").sum())),
            negative_lesions=("refined_binary_label", lambda values: int(pd.Series(values).eq("negative").sum())),
            uncertain_lesions=("uncertainty_flag", lambda values: int(pd.Series(values).astype(bool).sum())),
            excluded_lesions=("exclusion_flag", lambda values: int(pd.Series(values).astype(bool).sum())),
        ).reset_index()
        donor_support_frame["edge_label"] = edge_label
        donor_support_frame["binary_support_status"] = np.where(
            (donor_support_frame["positive_lesions"] > 0) & (donor_support_frame["negative_lesions"] > 0),
            "mixed",
            np.where(donor_support_frame["positive_lesions"] > 0, "positive_only", np.where(donor_support_frame["negative_lesions"] > 0, "negative_only", "uncertain_only")),
        )
        donor_rows.extend(donor_support_frame.loc[:, list(DONOR_SUPPORT_COLUMNS)].to_dict(orient="records"))

        usable_binary = relevant.loc[
            (~relevant["exclusion_flag"].astype(bool))
            & (~relevant["uncertainty_flag"].astype(bool))
            & (relevant["refined_binary_label"].isin(["positive", "negative"]))
        ].copy()
        edge_rows.append(
            {
                "edge_label": edge_label,
                "target_kind": "refined",
                "n_lesions": int(relevant.shape[0]),
                "n_donors": int(relevant["donor_id"].astype(str).nunique()),
                "positive_lesions": int(usable_binary["refined_binary_label"].eq("positive").sum()),
                "negative_lesions": int(usable_binary["refined_binary_label"].eq("negative").sum()),
                "uncertain_lesions": int(relevant["uncertainty_flag"].astype(bool).sum()),
                "excluded_lesions": int(relevant["exclusion_flag"].astype(bool).sum()),
                "positive_donors": int(usable_binary.loc[usable_binary["refined_binary_label"] == "positive", "donor_id"].astype(str).nunique()),
                "negative_donors": int(usable_binary.loc[usable_binary["refined_binary_label"] == "negative", "donor_id"].astype(str).nunique()),
                "continuous_unique_scores": int(pd.to_numeric(relevant["progression_risk_score"], errors="coerce").dropna().nunique()),
                "binary_viable": bool(binary_viable),
                "continuous_viable": bool(continuous_viable),
                "recommended_target": recommended,
                "reason": reason,
            }
        )
        split_report["edges"][edge_label] = {
            "binary_viable": bool(binary_viable),
            "binary_reason": binary_reason,
            "continuous_viable": bool(continuous_viable),
            "continuous_reason": continuous_reason,
            "recommended_target": recommended,
        }

    edge_support = pd.DataFrame(edge_rows, columns=list(EDGE_SUPPORT_COLUMNS))
    donor_support = pd.DataFrame(donor_rows, columns=list(DONOR_SUPPORT_COLUMNS))
    return edge_support, donor_support, split_report
