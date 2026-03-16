"""Build niche-level LuCA similarity features for EA-MIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stagebridge.logging_utils import get_logger

from .eamist_common import (
    DEFAULT_LUCA_TOP_K,
    choose_niche_token_columns,
    cosine_similarity_rows,
    entropy_from_rows,
    normalize_niche_table,
    topk_labels_and_scores,
    utc_now_iso,
    write_json,
)

log = get_logger(__name__)


def _state_token_matrix(summary: pd.DataFrame, token_labels: list[str]) -> np.ndarray:
    columns = [f"token_weight__{label}" for label in token_labels]
    missing = [column for column in columns if column not in summary.columns]
    if missing:
        raise KeyError(f"LuCA state summary is missing token-weight columns: {missing}")
    matrix = summary[columns].to_numpy(dtype=np.float32, copy=False)
    row_sum = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    return matrix.astype(np.float32, copy=False)


def run(
    niche_parquet: Path,
    luca_centroids: Path,
    luca_summary: Path,
    out_path: Path,
    *,
    top_k: int = DEFAULT_LUCA_TOP_K,
) -> dict[str, object]:
    niche_parquet = Path(niche_parquet).resolve()
    luca_centroids = Path(luca_centroids).resolve()
    luca_summary = Path(luca_summary).resolve()
    out_path = Path(out_path).resolve()
    if not niche_parquet.exists():
        raise FileNotFoundError(f"Missing niche parquet: {niche_parquet}")
    if not luca_centroids.exists():
        raise FileNotFoundError(f"Missing LuCA centroid parquet: {luca_centroids}")
    if not luca_summary.exists():
        raise FileNotFoundError(f"Missing LuCA summary parquet: {luca_summary}")

    log.info("Building niche-level LuCA features from %s", luca_summary)
    niche_df = normalize_niche_table(pd.read_parquet(niche_parquet))
    token_columns, token_labels, token_prefix = choose_niche_token_columns(niche_df)
    niche_matrix = niche_df[token_columns].to_numpy(dtype=np.float32, copy=False)

    centroids = pd.read_parquet(luca_centroids)
    summary = pd.read_parquet(luca_summary)
    if centroids.empty or summary.empty:
        raise ValueError("LuCA reference inputs were empty; luca_features cannot be built.")
    if centroids["luca_state"].duplicated().any() or summary["luca_state"].duplicated().any():
        raise ValueError("LuCA reference states must be unique.")
    summary = summary.merge(
        centroids.loc[:, ["luca_state", "dispersion"]],
        on="luca_state",
        how="left",
        suffixes=("", "__centroid"),
    )

    state_matrix = _state_token_matrix(summary, token_labels)
    similarity = cosine_similarity_rows(niche_matrix, state_matrix)
    state_labels = summary["luca_state"].astype(str).tolist()
    top_scores, top_labels = topk_labels_and_scores(similarity, state_labels, int(top_k))

    malignant_mask = summary["malignant_flag"].fillna(False).astype(bool).to_numpy()
    immune_mask = summary["immune_flag"].fillna(False).astype(bool).to_numpy()
    stromal_mask = (
        summary["stromal_flag"].fillna(False).astype(bool).to_numpy()
        | summary["compartment_group"].astype(str).eq("stromal").to_numpy()
    )
    invasive_mask = summary["invasive_like_flag"].fillna(False).astype(bool).to_numpy()
    epithelial_mask = summary["epithelial_flag"].fillna(False).astype(bool).to_numpy()

    def _masked_mean(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros(similarity.shape[0], dtype=np.float32)
        return similarity[:, mask].mean(axis=1).astype(np.float32, copy=False)

    malignant_mean = _masked_mean(malignant_mask)
    immune_mean = _masked_mean(immune_mask)
    stromal_mean = _masked_mean(stromal_mask)
    epithelial_mean = _masked_mean(epithelial_mask)
    invasive_mean = _masked_mean(invasive_mask)

    top_entropy = entropy_from_rows(np.clip(top_scores, 0.0, None))
    result = niche_df.loc[
        :, ["lesion_id", "sample_id", "niche_id", "donor_id", "patient_id", "stage"]
    ].copy()
    for idx in range(top_scores.shape[1]):
        result[f"luca_top{idx + 1}_similarity"] = top_scores[:, idx].astype(np.float32, copy=False)
        result[f"luca_top{idx + 1}_state"] = top_labels[:, idx].astype(str)
    result["luca_tumor_adoption_score"] = malignant_mean
    result["luca_invasive_like_score"] = invasive_mean
    result["luca_tumor_immune_stromal_ecosystem_score"] = (
        malignant_mean * np.maximum(immune_mean, stromal_mean)
    ).astype(np.float32, copy=False)
    result["luca_max_state_similarity"] = similarity.max(axis=1).astype(np.float32, copy=False)
    result["luca_topk_entropy"] = top_entropy
    result["luca_malignant_state_similarity"] = malignant_mean
    result["luca_immune_state_similarity"] = immune_mean
    result["luca_stromal_state_similarity"] = stromal_mean
    result["luca_epithelial_state_similarity"] = epithelial_mean
    result["luca_state_count"] = np.int32(summary.shape[0])
    result.to_parquet(out_path, index=False)

    excluded_states: list[str] = []
    if not np.any(invasive_mask):
        excluded_states.append("invasive_like_states_not_detected")
    audit = {
        "created_at_utc": utc_now_iso(),
        "niche_parquet": str(niche_parquet),
        "luca_centroids": str(luca_centroids),
        "luca_summary": str(luca_summary),
        "out_path": str(out_path),
        "chosen_scoring_space": "token_composition_space",
        "token_columns_used": token_columns,
        "token_prefix_used": token_prefix,
        "chosen_luca_state_column": str(summary["state_annotation_column"].dropna().iloc[0])
        if "state_annotation_column" in summary.columns
        and not summary["state_annotation_column"].dropna().empty
        else "unknown",
        "chosen_top_k": int(min(int(top_k), int(summary.shape[0]))),
        "num_niches_scored": int(result.shape[0]),
        "missing_value_count": int(result.isna().sum().sum()),
        "excluded_luca_states": excluded_states,
        "num_luca_states": int(summary.shape[0]),
        "centroid_dimension": int(len(centroids["centroid_vector"].iloc[0]))
        if not centroids.empty
        else 0,
    }
    write_json(out_path.parent / f"{out_path.stem}.audit.json", audit)
    log.info(
        "Built LuCA niche features for %d niches using %d states and top_k=%d",
        int(result.shape[0]),
        int(summary.shape[0]),
        int(min(int(top_k), int(summary.shape[0]))),
    )
    return audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--niche-parquet", type=Path, required=True, help="Path to niche_tokens_full.parquet"
    )
    parser.add_argument(
        "--luca-centroids", type=Path, required=True, help="Path to luca_state_centroids.parquet"
    )
    parser.add_argument(
        "--luca-summary", type=Path, required=True, help="Path to luca_state_summary.parquet"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output parquet path for niche-level LuCA features"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_LUCA_TOP_K,
        help="Top-k LuCA state similarities to store per niche",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run(
        args.niche_parquet,
        args.luca_centroids,
        args.luca_summary,
        args.out,
        top_k=int(args.top_k),
    )
    print(f"built LuCA niche features: {audit}")


if __name__ == "__main__":
    main()
