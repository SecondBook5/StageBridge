"""Build niche-level HLCA healthy-reference features for EA-MIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from .eamist_common import (
    DEFAULT_HLCA_TOP_K,
    TOKEN_LINEAGES,
    choose_niche_token_columns,
    cosine_similarity_rows,
    entropy_from_rows,
    normalize_niche_table,
    topk_labels_and_scores,
    utc_now_iso,
    write_json,
)


def _lineage_sum(
    frame: pd.DataFrame, token_columns: list[str], labels: list[str], lineage: str
) -> np.ndarray:
    members = set(TOKEN_LINEAGES[lineage])
    selected = [column for column, label in zip(token_columns, labels) if label in members]
    if not selected:
        return np.zeros(frame.shape[0], dtype=np.float32)
    return frame[selected].sum(axis=1).to_numpy(dtype=np.float32, copy=False)


def run(
    hlca_labels_path: Path,
    hlca_latent_path: Path,
    niche_parquet: Path,
    out_path: Path,
    *,
    top_k: int = DEFAULT_HLCA_TOP_K,
) -> dict[str, object]:
    hlca_labels_path = Path(hlca_labels_path).resolve()
    hlca_latent_path = Path(hlca_latent_path).resolve()
    niche_parquet = Path(niche_parquet).resolve()
    out_path = Path(out_path).resolve()
    if not hlca_labels_path.exists():
        raise FileNotFoundError(f"Missing HLCA labels parquet: {hlca_labels_path}")
    if not hlca_latent_path.exists():
        raise FileNotFoundError(f"Missing HLCA latent h5ad: {hlca_latent_path}")
    if not niche_parquet.exists():
        raise FileNotFoundError(f"Missing niche parquet: {niche_parquet}")

    niche_df = normalize_niche_table(pd.read_parquet(niche_parquet))
    token_columns, token_labels, token_prefix = choose_niche_token_columns(niche_df)
    niche_matrix = niche_df[token_columns].to_numpy(dtype=np.float32, copy=False)

    labels_df = pd.read_parquet(hlca_labels_path)
    if labels_df.empty:
        raise ValueError("HLCA label parquet was empty.")
    if "hlca_label" in labels_df.columns:
        state_column = "hlca_label"
    else:
        string_cols = [
            column
            for column in labels_df.columns
            if labels_df[column].dtype == object or pd.api.types.is_string_dtype(labels_df[column])
        ]
        if not string_cols:
            raise ValueError("Could not detect a useful HLCA state column.")
        state_column = str(string_cols[0])

    latent = anndata.read_h5ad(hlca_latent_path, backed="r")
    try:
        obs = latent.obs.copy()
    finally:
        if getattr(latent, "isbacked", False):
            latent.file.close()
    if state_column not in obs.columns:
        if state_column in labels_df.columns:
            obs = obs.join(labels_df[[state_column]], how="left")
        else:
            raise KeyError(
                f"HLCA latent obs and labels parquet were both missing state column '{state_column}'."
            )
    if "stage" not in obs.columns:
        raise KeyError("HLCA latent file is missing 'stage' in obs.")

    obs[state_column] = obs[state_column].astype(str)
    obs["stage"] = obs["stage"].astype(str)
    healthy_mask = obs["stage"].eq("Normal")
    baseline_source = "Normal"
    baseline_counts = obs.loc[healthy_mask, state_column].value_counts()
    if baseline_counts.sum() <= 0:
        baseline_source = "all_cells"
        baseline_counts = obs[state_column].value_counts()
    matched_states = [
        label for label in token_labels if float(baseline_counts.get(label, 0.0)) > 0.0
    ]
    if not matched_states:
        raise ValueError("No HLCA states overlapped with niche token labels.")

    token_index = {label: idx for idx, label in enumerate(token_labels)}
    state_similarity = np.column_stack(
        [niche_matrix[:, token_index[label]] for label in matched_states]
    ).astype(np.float32, copy=False)
    top_scores, top_labels = topk_labels_and_scores(state_similarity, matched_states, int(top_k))
    baseline_vector = np.asarray(
        [float(baseline_counts.get(label, 0.0)) for label in matched_states], dtype=np.float32
    )[None, :]
    normal_likeness = (
        cosine_similarity_rows(state_similarity, baseline_vector)
        .reshape(-1)
        .astype(np.float32, copy=False)
    )
    max_state_similarity = state_similarity.max(axis=1).astype(np.float32, copy=False)

    epithelial_summary = _lineage_sum(niche_df, token_columns, token_labels, "epithelial")
    immune_summary = _lineage_sum(niche_df, token_columns, token_labels, "immune")
    stromal_summary = _lineage_sum(niche_df, token_columns, token_labels, "stromal_endothelial")
    dominant_lineage = np.maximum.reduce([epithelial_summary, immune_summary, stromal_summary])
    lineage_fidelity = np.divide(
        max_state_similarity,
        np.clip(dominant_lineage, 1e-6, None),
        out=np.zeros_like(max_state_similarity),
        where=dominant_lineage > 0,
    ).astype(np.float32, copy=False)

    result = niche_df.loc[
        :, ["lesion_id", "sample_id", "niche_id", "donor_id", "patient_id", "stage"]
    ].copy()
    for idx in range(top_scores.shape[1]):
        result[f"hlca_top{idx + 1}_similarity"] = top_scores[:, idx].astype(np.float32, copy=False)
        result[f"hlca_top{idx + 1}_state"] = top_labels[:, idx].astype(str)
    result["hlca_normal_likeness_score"] = normal_likeness
    result["hlca_deviation_from_normal_score"] = (1.0 - normal_likeness).astype(
        np.float32, copy=False
    )
    result["hlca_lineage_fidelity_score"] = lineage_fidelity
    result["hlca_max_state_similarity"] = max_state_similarity
    result["hlca_topk_entropy"] = entropy_from_rows(np.clip(top_scores, 0.0, None))
    result["hlca_epithelial_like_similarity"] = epithelial_summary
    result["hlca_immune_like_similarity"] = immune_summary
    result["hlca_stromal_endothelial_like_similarity"] = stromal_summary
    result.to_parquet(out_path, index=False)

    audit = {
        "created_at_utc": utc_now_iso(),
        "hlca_labels": str(hlca_labels_path),
        "hlca_latent": str(hlca_latent_path),
        "niche_parquet": str(niche_parquet),
        "out_path": str(out_path),
        "chosen_hlca_state_column": state_column,
        "chosen_scoring_space": "token_composition_space_aligned_to_hlca_labels",
        "baseline_source": baseline_source,
        "matched_states": matched_states,
        "top_k": int(min(int(top_k), len(matched_states))),
        "token_columns_used": token_columns,
        "token_prefix_used": token_prefix,
        "num_niches_scored": int(result.shape[0]),
        "missing_value_count": int(result.isna().sum().sum()),
    }
    write_json(out_path.parent / f"{out_path.stem}.audit.json", audit)
    return audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hlca-labels", type=Path, required=True, help="Path to snrna_full_hlca_labels.parquet"
    )
    parser.add_argument(
        "--hlca-latent", type=Path, required=True, help="Path to snrna_hlca_latent_full.h5ad"
    )
    parser.add_argument(
        "--niche-parquet", type=Path, required=True, help="Path to niche_tokens_full.parquet"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output parquet for niche HLCA features"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_HLCA_TOP_K,
        help="Top-k HLCA state similarities to keep per niche",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run(
        args.hlca_labels,
        args.hlca_latent,
        args.niche_parquet,
        args.out,
        top_k=int(args.top_k),
    )
    print(f"built HLCA niche features: {audit}")


if __name__ == "__main__":
    main()
