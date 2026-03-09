"""Build LuCA state centroids and state summaries for EA-MIST."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from stagebridge.logging_utils import get_logger

from .eamist_common import (
    SelectedEmbedding,
    choose_best_embedding,
    infer_state_grouping,
    infer_token_profile,
    read_matrix_chunk,
    read_obs_frame_h5ad,
    select_luca_columns,
    utc_now_iso,
    write_json,
)

log = get_logger(__name__)


def _mode_or_none(series: pd.Series) -> str | None:
    values = series.dropna().astype(str)
    if values.empty:
        return None
    mode = values.mode(dropna=True)
    if mode.empty:
        return None
    return str(mode.iloc[0])


def _load_selected_obs(atlas_path: Path) -> tuple[pd.DataFrame, object]:
    selected = select_luca_columns(atlas_path)
    columns = [selected.state_column]
    for optional in (
        selected.major_celltype_column,
        selected.malignant_column,
        *selected.dataset_columns,
        *selected.sample_columns,
        *selected.patient_columns,
        *selected.epithelial_subtype_columns,
    ):
        if optional is not None and optional not in columns:
            columns.append(optional)
    obs = read_obs_frame_h5ad(atlas_path, columns)
    return obs, selected


def _accumulate_centroids_from_obsm(
    atlas_path: Path,
    embedding: SelectedEmbedding,
    state_codes: np.ndarray,
    n_states: int,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_states, dtype=np.int64)
    sums = np.zeros((n_states, int(embedding.shape[1])), dtype=np.float64)
    sumsq = np.zeros_like(sums)
    total_chunks = max((int(embedding.shape[0]) + int(chunk_size) - 1) // int(chunk_size), 1)
    progress_every = max(total_chunks // 10, 1)
    with h5py.File(atlas_path, "r") as handle:
        matrix_obj = handle["obsm"][embedding.key]
        for chunk_index, start in enumerate(range(0, int(embedding.shape[0]), int(chunk_size)), start=1):
            stop = min(start + int(chunk_size), int(embedding.shape[0]))
            block = read_matrix_chunk(matrix_obj, start, stop)
            block_codes = state_codes[start:stop]
            for code in np.unique(block_codes):
                if code < 0:
                    continue
                mask = block_codes == code
                rows = block[mask]
                if rows.size == 0:
                    continue
                counts[int(code)] += int(rows.shape[0])
                sums[int(code)] += rows.sum(axis=0, dtype=np.float64)
                sumsq[int(code)] += np.square(rows, dtype=np.float64).sum(axis=0, dtype=np.float64)
            if chunk_index == 1 or chunk_index % progress_every == 0 or chunk_index == total_chunks:
                log.info(
                    "Accumulating LuCA centroids from obsm '%s': chunk %d/%d",
                    embedding.key,
                    chunk_index,
                    total_chunks,
                )
    return counts, sums, sumsq


def _accumulate_centroids_from_x(
    atlas_path: Path,
    embedding: SelectedEmbedding,
    state_codes: np.ndarray,
    n_states: int,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_states, dtype=np.int64)
    sums = np.zeros((n_states, int(embedding.shape[1])), dtype=np.float64)
    sumsq = np.zeros_like(sums)
    total_chunks = max((int(embedding.shape[0]) + int(chunk_size) - 1) // int(chunk_size), 1)
    progress_every = max(total_chunks // 10, 1)
    adata = anndata.read_h5ad(atlas_path, backed="r")
    try:
        for chunk_index, start in enumerate(range(0, int(embedding.shape[0]), int(chunk_size)), start=1):
            stop = min(start + int(chunk_size), int(embedding.shape[0]))
            block = adata.X[start:stop]
            if sp.issparse(block):
                block = block.toarray()
            block = np.asarray(block, dtype=np.float32)
            block_codes = state_codes[start:stop]
            for code in np.unique(block_codes):
                if code < 0:
                    continue
                mask = block_codes == code
                rows = block[mask]
                if rows.size == 0:
                    continue
                counts[int(code)] += int(rows.shape[0])
                sums[int(code)] += rows.sum(axis=0, dtype=np.float64)
                sumsq[int(code)] += np.square(rows, dtype=np.float64).sum(axis=0, dtype=np.float64)
            if chunk_index == 1 or chunk_index % progress_every == 0 or chunk_index == total_chunks:
                log.info(
                    "Accumulating LuCA centroids from X: chunk %d/%d",
                    chunk_index,
                    total_chunks,
                )
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    return counts, sums, sumsq


def run(atlas_path: Path, outdir: Path, *, chunk_size: int = 8192) -> dict[str, object]:
    atlas_path = Path(atlas_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Missing LuCA atlas file: {atlas_path}")

    obs, selected = _load_selected_obs(atlas_path)
    log.info(
        "Selected LuCA state column=%s major_celltype_column=%s malignant_column=%s",
        selected.state_column,
        selected.major_celltype_column,
        selected.malignant_column,
    )
    state_series = obs[selected.state_column].astype(str).replace({"None": np.nan, "nan": np.nan, "": np.nan})
    if state_series.dropna().empty:
        raise ValueError(f"Selected LuCA state column '{selected.state_column}' did not contain usable values.")
    state_categories = sorted(state_series.dropna().astype(str).unique().tolist())
    state_to_code = {state: idx for idx, state in enumerate(state_categories)}
    state_codes = np.asarray([state_to_code.get(value, -1) if pd.notna(value) else -1 for value in state_series], dtype=np.int32)

    embedding = choose_best_embedding(atlas_path)
    if int(embedding.shape[0]) != int(obs.shape[0]):
        raise ValueError(
            f"LuCA embedding row count {embedding.shape[0]} does not match obs rows {obs.shape[0]}."
        )
    log.info(
        "Using LuCA embedding key=%s source=%s shape=%s",
        embedding.key,
        embedding.source,
        embedding.shape,
    )
    if embedding.source == "obsm":
        counts, sums, sumsq = _accumulate_centroids_from_obsm(
            atlas_path,
            embedding,
            state_codes,
            len(state_categories),
            chunk_size=int(chunk_size),
        )
    else:
        counts, sums, sumsq = _accumulate_centroids_from_x(
            atlas_path,
            embedding,
            state_codes,
            len(state_categories),
            chunk_size=int(chunk_size),
        )

    rows_centroids: list[dict[str, object]] = []
    rows_summary: list[dict[str, object]] = []
    group_cols = [selected.state_column]
    if selected.major_celltype_column is not None:
        group_cols.append(selected.major_celltype_column)
    if selected.malignant_column is not None:
        group_cols.append(selected.malignant_column)
    for column in (*selected.dataset_columns, *selected.sample_columns, *selected.patient_columns, *selected.epithelial_subtype_columns):
        if column not in group_cols:
            group_cols.append(column)
    grouped = obs[group_cols].copy()

    for state, code in state_to_code.items():
        count = int(counts[int(code)])
        if count <= 0:
            continue
        centroid = (sums[int(code)] / float(count)).astype(np.float32, copy=False)
        variance = np.maximum((sumsq[int(code)] / float(count)) - np.square(centroid, dtype=np.float32), 0.0)
        dispersion = float(np.mean(variance, dtype=np.float64))
        state_rows = grouped.loc[state_series == state]
        major_value = _mode_or_none(state_rows[selected.major_celltype_column]) if selected.major_celltype_column is not None else None
        malignant_value = _mode_or_none(state_rows[selected.malignant_column]) if selected.malignant_column is not None else None
        epithelial_value = None
        for column in selected.epithelial_subtype_columns:
            epithelial_value = _mode_or_none(state_rows[column])
            if epithelial_value is not None:
                break
        grouping = infer_state_grouping(state, major_value, malignant_value, epithelial_value)
        token_profile = infer_token_profile(state, major_value, malignant_value, epithelial_value)
        dataset_values = {column: _mode_or_none(state_rows[column]) for column in selected.dataset_columns}
        rows_centroids.append(
            {
                "luca_state": str(state),
                "centroid_vector": centroid.tolist(),
                "count": count,
                "dispersion": dispersion,
                "major_lineage_tag": grouping["major_lineage_tag"],
                "compartment_group": grouping["compartment_group"],
                "malignant_flag": bool(grouping["malignant_flag"]),
                "immune_flag": bool(grouping["immune_flag"]),
                "stromal_flag": bool(grouping["stromal_flag"]),
                "epithelial_flag": bool(grouping["epithelial_flag"]),
                "invasive_like_flag": bool(grouping["invasive_like_flag"]),
                "epithelial_subtype_label": grouping["epithelial_subtype_label"],
            }
        )
        summary_row: dict[str, object] = {
            "luca_state": str(state),
            "state_annotation_column": selected.state_column,
            "state_annotation_value": str(state),
            "major_celltype_column": selected.major_celltype_column,
            "major_celltype_value": major_value,
            "malignant_column": selected.malignant_column,
            "malignant_value": malignant_value,
            "count": count,
            "dispersion": dispersion,
            "major_lineage_tag": grouping["major_lineage_tag"],
            "compartment_group": grouping["compartment_group"],
            "malignant_flag": bool(grouping["malignant_flag"]),
            "immune_flag": bool(grouping["immune_flag"]),
            "stromal_flag": bool(grouping["stromal_flag"]),
            "epithelial_flag": bool(grouping["epithelial_flag"]),
            "invasive_like_flag": bool(grouping["invasive_like_flag"]),
            "epithelial_subtype_label": grouping["epithelial_subtype_label"],
        }
        for column, value in dataset_values.items():
            summary_row[f"dataset_mode__{column}"] = value
        for label in token_profile:
            summary_row[f"token_weight__{label}"] = float(token_profile[label])
        rows_summary.append(summary_row)

    centroids = pd.DataFrame(rows_centroids).sort_values(["count", "luca_state"], ascending=[False, True]).reset_index(drop=True)
    summary = pd.DataFrame(rows_summary).sort_values(["count", "luca_state"], ascending=[False, True]).reset_index(drop=True)
    if centroids.empty or summary.empty:
        raise ValueError("LuCA reference construction produced no states.")

    centroids.to_parquet(outdir / "luca_state_centroids.parquet", index=False)
    summary.to_parquet(outdir / "luca_state_summary.parquet", index=False)

    manifest = {
        "created_at_utc": utc_now_iso(),
        "atlas_path": str(atlas_path),
        "state_column": selected.state_column,
        "major_celltype_column": selected.major_celltype_column,
        "malignant_column": selected.malignant_column,
        "embedding_key": embedding.key,
        "embedding_source": embedding.source,
        "embedding_shape": [int(embedding.shape[0]), int(embedding.shape[1])],
        "number_of_states": int(summary.shape[0]),
        "top_states_by_abundance": summary.loc[:, ["luca_state", "count"]].head(20).to_dict(orient="records"),
    }
    write_json(outdir / "luca_reference_manifest.json", manifest)
    log.info(
        "Built LuCA reference with %d states; top state=%s",
        int(summary.shape[0]),
        str(summary.iloc[0]["luca_state"]),
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True, help="Path to the LuCA atlas h5ad")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for centroid/state summary outputs")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Row chunk size for centroid accumulation")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run(args.atlas, args.outdir, chunk_size=int(args.chunk_size))
    print(f"built LuCA reference: {manifest}")


if __name__ == "__main__":
    main()
