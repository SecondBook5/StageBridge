"""Niche token feature extraction and token-bank utilities for Tangram outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import anndata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import diags
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import zarr

from stagebridge.logging_utils import get_logger
from stagebridge.context_model.token_schema import TypedTokenSchema, default_typed_token_schema

log = get_logger(__name__)

INDEX_COLUMN = "__index_level_0__"
DEFAULT_TOKEN_PREFIX = "tok_"
DEFAULT_SMOOTH_TOKEN_PREFIX = "tok_smooth_"


@dataclass(slots=True, frozen=True)
class TypedTokenResult:
    """Typed token table produced from spot-level spatial mapping outputs."""

    tokens: np.ndarray
    coords: np.ndarray
    obs: pd.DataFrame
    schema: TypedTokenSchema

    def summary(self) -> dict[str, object]:
        return {
            "n_tokens": int(self.tokens.shape[0]),
            "token_dim": int(self.tokens.shape[1]),
            "typed_feature_names": list(self.schema.typed_feature_names),
            "stage_counts": {str(k): int(v) for k, v in self.obs.groupby("stage").size().items()},
        }


def build_typed_spot_tokens(
    compositions: np.ndarray,
    coords: np.ndarray,
    obs: pd.DataFrame,
    raw_feature_names: list[str] | tuple[str, ...],
    *,
    schema: TypedTokenSchema | None = None,
    normalize_rows: bool = True,
) -> TypedTokenResult:
    """Aggregate raw Tangram cell-state outputs into typed niche token groups."""
    raw = np.asarray(compositions, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D compositions matrix, got shape={raw.shape}.")
    if coords.shape[0] != raw.shape[0]:
        raise ValueError("coords rows must match compositions rows.")
    if obs.shape[0] != raw.shape[0]:
        raise ValueError("obs rows must match compositions rows.")

    schema = schema or default_typed_token_schema(raw_feature_names)
    feature_names = [str(name) for name in raw_feature_names]
    if len(feature_names) != raw.shape[1]:
        raise ValueError(
            f"Feature-name length mismatch: {len(feature_names)} names for {raw.shape[1]} columns."
        )

    values = raw.copy()
    if normalize_rows:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_sums,
            out=np.zeros_like(values),
            where=row_sums > 0,
        )

    typed = np.zeros((values.shape[0], len(schema.typed_feature_names)), dtype=np.float32)
    group_to_col = {name: idx for idx, name in enumerate(schema.typed_feature_names)}
    for col_idx, feature_name in enumerate(feature_names):
        group_name = schema.group_for(feature_name)
        typed[:, group_to_col[group_name]] += values[:, col_idx]

    return TypedTokenResult(
        tokens=typed,
        coords=np.asarray(coords, dtype=np.float32),
        obs=obs.reset_index(drop=True).copy(),
        schema=schema,
    )


@dataclass(slots=True)
class NicheTokenBuildResult:
    """In-memory result from niche-token feature construction."""

    df: pd.DataFrame
    score_columns: list[str]
    token_columns: list[str]
    smooth_token_columns: list[str]
    audit: dict[str, Any]


@dataclass(slots=True)
class NicheTokenBankBuildResult:
    """Summary from writing a Zarr token bank."""

    output_path: Path
    sample_ids: list[str]
    token_columns: list[str]
    n_rows: int
    n_groups: int
    audit: dict[str, Any]


def _parse_obs_name(obs_name: str) -> tuple[str, str]:
    """Parse ``sample_id`` and barcode-like suffix from canonical spot name."""
    text = str(obs_name)
    if ":" in text:
        sample_id, barcode = text.split(":", 1)
        return sample_id, barcode
    return text, text


def _parse_gsm_from_sample_id(sample_id: str) -> str:
    m = re.match(r"^(GSM\d+)", str(sample_id))
    if m:
        return m.group(1)
    return str(sample_id).split("_")[0]


def _arrow_scores_to_arrays(
    scores_parquet: Path,
    expected_columns: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load scores parquet via pyarrow and return index, values, and column names."""
    table = pq.read_table(scores_parquet)
    col_names = list(table.column_names)
    if INDEX_COLUMN not in col_names:
        raise KeyError(
            f"Expected parquet index column '{INDEX_COLUMN}' in {scores_parquet}. "
            f"Found columns: {col_names}"
        )

    score_cols = [c for c in col_names if c != INDEX_COLUMN]
    if expected_columns is not None:
        want = [str(c) for c in expected_columns]
        if score_cols != want:
            raise ValueError(
                "Tangram score columns mismatch. "
                f"Expected {want}, found {score_cols}."
            )

    index = table[INDEX_COLUMN].to_numpy(zero_copy_only=False).astype(str)
    values = np.column_stack(
        [table[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in score_cols]
    ).astype(np.float32, copy=False)
    return index, values, score_cols


def _validate_alignment(obs_names: np.ndarray, score_index: np.ndarray) -> None:
    """Hard-fail if score rows do not align 1:1 with AnnData obs."""
    if obs_names.shape[0] != score_index.shape[0]:
        raise ValueError(
            "Scores/AnnData row count mismatch: "
            f"{score_index.shape[0]} scores rows vs {obs_names.shape[0]} obs rows."
        )
    mismatch = np.flatnonzero(obs_names != score_index)
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            "Scores index does not match AnnData obs_names exactly. "
            f"First mismatch at row {i}: score_index='{score_index[i]}' vs obs_name='{obs_names[i]}'."
        )


def _resolve_metadata(
    adata_obs: pd.DataFrame,
    obs_names: np.ndarray,
    coords: np.ndarray,
    group_key: str,
) -> tuple[pd.DataFrame, str]:
    """Build metadata DataFrame with required keys and resolved grouping column."""
    n = obs_names.shape[0]
    sample_from_index = np.empty(n, dtype=object)
    barcode_from_index = np.empty(n, dtype=object)
    gsm_from_index = np.empty(n, dtype=object)
    for i, name in enumerate(obs_names.tolist()):
        sample_id, barcode = _parse_obs_name(name)
        sample_from_index[i] = sample_id
        barcode_from_index[i] = barcode
        gsm_from_index[i] = _parse_gsm_from_sample_id(sample_id)

    df = pd.DataFrame(index=pd.Index(obs_names, name="spot_obs_name"))
    df["spot_id"] = adata_obs["spot_id"].astype(str).to_numpy() if "spot_id" in adata_obs.columns else barcode_from_index
    df["barcode"] = adata_obs["barcode"].astype(str).to_numpy() if "barcode" in adata_obs.columns else barcode_from_index
    df["donor_id"] = (
        adata_obs["donor_id"].astype(str).to_numpy()
        if "donor_id" in adata_obs.columns
        else (
            adata_obs["patient_id"].astype(str).to_numpy()
            if "patient_id" in adata_obs.columns
            else np.array(["unknown_donor"] * n, dtype=object)
        )
    )
    df["stage"] = (
        adata_obs["stage"].astype(str).to_numpy()
        if "stage" in adata_obs.columns
        else np.array(["Unknown"] * n, dtype=object)
    )
    df["sample_id"] = (
        adata_obs["sample_id"].astype(str).to_numpy()
        if "sample_id" in adata_obs.columns
        else sample_from_index
    )
    df["gsm_id"] = (
        adata_obs["gsm_id"].astype(str).to_numpy()
        if "gsm_id" in adata_obs.columns
        else gsm_from_index
    )
    df["x"] = coords[:, 0].astype(np.float32, copy=False)
    df["y"] = coords[:, 1].astype(np.float32, copy=False)

    resolved_group_key = group_key
    if resolved_group_key not in df.columns:
        if resolved_group_key == "sample_id":
            resolved_group_key = "gsm_id"
        else:
            raise KeyError(
                f"group_key='{group_key}' not found in metadata columns {list(df.columns)}."
            )
    return df, resolved_group_key


def _row_normalize_prob(
    scores: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Row-normalize score matrix into probabilities with zero-row handling."""
    scores = np.asarray(scores, dtype=np.float32)
    n, m = scores.shape
    row_sum = scores.sum(axis=1, dtype=np.float32)
    zero_mask = row_sum <= np.float32(eps)
    safe_sum = row_sum.copy()
    safe_sum[zero_mask] = 1.0
    probs = scores / safe_sum[:, None]
    if np.any(zero_mask):
        probs[zero_mask] = np.float32(1.0 / float(m))
    return probs.astype(np.float32, copy=False), int(zero_mask.sum())


def _entropy_confidence(
    probs: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row entropy and max-prob confidence."""
    p = np.asarray(probs, dtype=np.float32)
    entropy = -np.sum(p * np.log(p + np.float32(eps)), axis=1, dtype=np.float32)
    confidence = np.max(p, axis=1).astype(np.float32, copy=False)
    return entropy.astype(np.float32, copy=False), confidence


def _groupwise_smoothing(
    probs: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    k_neighbors: int,
    show_progress: bool,
) -> np.ndarray:
    """Compute group-wise kNN smoothing ``p_smooth = A @ p`` with CSR adjacency."""
    p = np.asarray(probs, dtype=np.float32)
    xy = np.asarray(coords, dtype=np.float32)
    out = np.zeros_like(p, dtype=np.float32)

    group_values, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_ids = np.arange(group_values.shape[0], dtype=np.int64)
    iterator: Iterable[np.int64] = group_ids
    if show_progress:
        iterator = tqdm(group_ids, desc="Smoothing groups", unit="group")

    for gid in iterator:
        idx = np.where(inverse == gid)[0]
        n = idx.shape[0]
        if n == 0:
            continue
        if n == 1:
            out[idx] = p[idx]
            continue

        k_eff = min(max(1, int(k_neighbors)), n - 1)
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
        nn.fit(xy[idx])
        adjacency = nn.kneighbors_graph(mode="connectivity")  # csr, excludes self
        adjacency = adjacency.astype(np.float32, copy=False) + diags(
            np.ones(n, dtype=np.float32), format="csr"
        )
        row_sum = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32, copy=False)
        inv_row_sum = np.divide(
            np.float32(1.0),
            row_sum,
            out=np.zeros_like(row_sum, dtype=np.float32),
            where=row_sum > 0,
        )
        A = diags(inv_row_sum, format="csr") @ adjacency
        out[idx] = (A @ p[idx]).astype(np.float32, copy=False)
    return out


def _numeric_audit(
    df: pd.DataFrame,
    token_columns: list[str],
) -> dict[str, Any]:
    """Compute numeric audit statistics for the output token table."""
    numeric_cols = ["entropy", "confidence", *token_columns]
    arr = df[numeric_cols].to_numpy(dtype=np.float32)
    nan_counts = np.isnan(arr).sum(axis=0)
    inf_counts = np.isinf(arr).sum(axis=0)
    min_vals = np.nanmin(arr, axis=0)
    max_vals = np.nanmax(arr, axis=0)

    q = np.quantile(
        df[["entropy", "confidence"]].to_numpy(dtype=np.float32),
        q=[0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0],
        axis=0,
    )
    return {
        "nan_count_per_column": {
            col: int(v) for col, v in zip(numeric_cols, nan_counts.tolist())
        },
        "inf_count_per_column": {
            col: int(v) for col, v in zip(numeric_cols, inf_counts.tolist())
        },
        "min_per_column": {
            col: float(v) for col, v in zip(numeric_cols, min_vals.tolist())
        },
        "max_per_column": {
            col: float(v) for col, v in zip(numeric_cols, max_vals.tolist())
        },
        "entropy_quantiles": {
            "q00": float(q[0, 0]),
            "q05": float(q[1, 0]),
            "q25": float(q[2, 0]),
            "q50": float(q[3, 0]),
            "q75": float(q[4, 0]),
            "q95": float(q[5, 0]),
            "q100": float(q[6, 0]),
        },
        "confidence_quantiles": {
            "q00": float(q[0, 1]),
            "q05": float(q[1, 1]),
            "q25": float(q[2, 1]),
            "q50": float(q[3, 1]),
            "q75": float(q[4, 1]),
            "q95": float(q[5, 1]),
            "q100": float(q[6, 1]),
        },
    }


def build_niche_tokens_dataframe(
    *,
    spatial_h5ad: Path,
    scores_parquet: Path,
    group_key: str = "sample_id",
    smooth: bool = True,
    k_neighbors: int = 6,
    eps: float = 1e-12,
    show_progress: bool = True,
) -> NicheTokenBuildResult:
    """Build the niche-token DataFrame from Tangram scores and spatial metadata."""
    spatial_h5ad = Path(spatial_h5ad)
    scores_parquet = Path(scores_parquet)
    if not spatial_h5ad.exists():
        raise FileNotFoundError(f"Missing spatial h5ad: {spatial_h5ad}")
    if not scores_parquet.exists():
        raise FileNotFoundError(f"Missing scores parquet: {scores_parquet}")

    adata = anndata.read_h5ad(spatial_h5ad, backed="r")
    try:
        if "spatial" not in adata.obsm:
            raise KeyError(f"Missing required obsm['spatial'] in {spatial_h5ad}")

        obs_names = adata.obs_names.astype(str).to_numpy()
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        score_index, scores, score_cols = _arrow_scores_to_arrays(scores_parquet=scores_parquet)
        _validate_alignment(obs_names=obs_names, score_index=score_index)

        metadata_df, resolved_group_key = _resolve_metadata(
            adata_obs=adata.obs,
            obs_names=obs_names,
            coords=coords,
            group_key=group_key,
        )

        probs, zero_row_count = _row_normalize_prob(scores, eps=float(eps))
        entropy, confidence = _entropy_confidence(probs, eps=float(eps))

        token_cols = [f"{DEFAULT_TOKEN_PREFIX}{col}" for col in score_cols]
        for j, col in enumerate(token_cols):
            metadata_df[col] = probs[:, j]

        smooth_cols: list[str] = []
        if smooth:
            groups = metadata_df[resolved_group_key].astype(str).to_numpy()
            p_smooth = _groupwise_smoothing(
                probs=probs,
                coords=coords,
                groups=groups,
                k_neighbors=int(k_neighbors),
                show_progress=bool(show_progress),
            )
            smooth_cols = [f"{DEFAULT_SMOOTH_TOKEN_PREFIX}{col}" for col in score_cols]
            for j, col in enumerate(smooth_cols):
                metadata_df[col] = p_smooth[:, j]

        metadata_df["entropy"] = entropy
        metadata_df["confidence"] = confidence

        all_token_cols = token_cols + smooth_cols
        tok_sum = metadata_df[token_cols].sum(axis=1).to_numpy(dtype=np.float32)
        tok_close = np.abs(tok_sum - 1.0) <= 1e-3

        audit = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "spatial_h5ad": str(spatial_h5ad),
                "scores_parquet": str(scores_parquet),
                "group_key_requested": group_key,
                "group_key_used": resolved_group_key,
                "k_neighbors": int(k_neighbors),
                "smooth": bool(smooth),
                "eps": float(eps),
            },
            "shape": {
                "n_rows": int(metadata_df.shape[0]),
                "n_cols": int(metadata_df.shape[1]),
                "n_score_cols": int(len(score_cols)),
                "n_token_cols": int(len(token_cols)),
                "n_smooth_token_cols": int(len(smooth_cols)),
                "n_groups": int(metadata_df[resolved_group_key].nunique()),
            },
            "zero_rows_replaced_with_uniform": int(zero_row_count),
            "token_sum_within_1pm1e-3_fraction": float(np.mean(tok_close)),
            "token_sum_stats": {
                "min": float(np.min(tok_sum)),
                "median": float(np.median(tok_sum)),
                "max": float(np.max(tok_sum)),
            },
        }
        audit.update(_numeric_audit(metadata_df, token_columns=all_token_cols))

        return NicheTokenBuildResult(
            df=metadata_df,
            score_columns=score_cols,
            token_columns=token_cols,
            smooth_token_columns=smooth_cols,
            audit=audit,
        )
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def write_niche_token_outputs(
    *,
    result: NicheTokenBuildResult,
    out_parquet: Path,
    out_audit_json: Path,
) -> None:
    """Write niche-token DataFrame and audit JSON."""
    out_parquet = Path(out_parquet)
    out_audit_json = Path(out_audit_json)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_audit_json.parent.mkdir(parents=True, exist_ok=True)

    result.df.to_parquet(out_parquet, index=True, engine="pyarrow")
    out_audit_json.write_text(json.dumps(result.audit, indent=2), encoding="utf-8")


def write_tokens_to_h5ad(
    *,
    spatial_h5ad: Path,
    out_h5ad: Path,
    tokens_df: pd.DataFrame,
    token_columns: list[str],
) -> None:
    """Write an h5ad copy with token-derived columns and ``obsm['X_niche_tokens']``."""
    spatial_h5ad = Path(spatial_h5ad)
    out_h5ad = Path(out_h5ad)
    adata = anndata.read_h5ad(spatial_h5ad)
    try:
        if not adata.obs_names.astype(str).equals(tokens_df.index.astype(str)):  # type: ignore[attr-defined]
            raise ValueError("Token DataFrame index does not align with AnnData obs_names.")
        adata.obs["entropy"] = tokens_df["entropy"].to_numpy(dtype=np.float32)
        adata.obs["confidence"] = tokens_df["confidence"].to_numpy(dtype=np.float32)
        adata.obsm["X_niche_tokens"] = tokens_df[token_columns].to_numpy(dtype=np.float32)
        adata.uns["niche_token_columns"] = [str(c) for c in token_columns]
        out_h5ad.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_h5ad, compression="lzf")
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def _safe_group_name(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("_")
    return safe or "sample"


def build_zarr_token_bank(
    *,
    niche_tokens_parquet: Path,
    output_zarr: Path,
    group_key: str = "sample_id",
    token_prefix: str = DEFAULT_SMOOTH_TOKEN_PREFIX,
) -> NicheTokenBankBuildResult:
    """Build a per-sample Zarr token bank for fast random token sampling."""
    niche_tokens_parquet = Path(niche_tokens_parquet)
    output_zarr = Path(output_zarr)
    if not niche_tokens_parquet.exists():
        raise FileNotFoundError(f"Missing niche tokens parquet: {niche_tokens_parquet}")

    df = pd.read_parquet(niche_tokens_parquet)
    if group_key not in df.columns:
        raise KeyError(f"Missing group key column '{group_key}' in {niche_tokens_parquet}.")
    if "x" not in df.columns or "y" not in df.columns:
        raise KeyError("Expected coordinate columns 'x' and 'y' in niche tokens parquet.")

    token_cols = [c for c in df.columns if str(c).startswith(str(token_prefix))]
    if not token_cols and token_prefix == DEFAULT_SMOOTH_TOKEN_PREFIX:
        token_cols = [c for c in df.columns if str(c).startswith(DEFAULT_TOKEN_PREFIX)]
        token_prefix = DEFAULT_TOKEN_PREFIX
    if not token_cols:
        raise ValueError(
            f"No token columns found with prefix '{token_prefix}' in {niche_tokens_parquet}."
        )

    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_zarr), mode="w", zarr_format=2)

    sample_ids: list[str] = []
    meta_rows: list[dict[str, Any]] = []
    for sample_id, sub in df.groupby(group_key, sort=True):
        sample = str(sample_id)
        group_name = _safe_group_name(sample)
        grp = root.require_group(group_name)

        tokens = sub[token_cols].to_numpy(dtype=np.float32, copy=False)
        coords = sub[["x", "y"]].to_numpy(dtype=np.float32, copy=False)
        spot_names = sub.index.astype(str).to_numpy()

        grp.create_array(
            "tokens",
            data=tokens,
            chunks=(min(8192, max(1, tokens.shape[0])), tokens.shape[1]),
            overwrite=True,
        )
        grp.create_array(
            "coords",
            data=coords,
            chunks=(min(8192, max(1, coords.shape[0])), 2),
            overwrite=True,
        )
        grp.create_array(
            "spot_names",
            data=spot_names.astype(str),
            overwrite=True,
        )
        grp.attrs["sample_id"] = sample
        grp.attrs["donor_id"] = str(sub["donor_id"].iloc[0]) if "donor_id" in sub.columns else "unknown_donor"
        grp.attrs["stage"] = str(sub["stage"].iloc[0]) if "stage" in sub.columns else "Unknown"
        grp.attrs["gsm_id"] = str(sub["gsm_id"].iloc[0]) if "gsm_id" in sub.columns else _parse_gsm_from_sample_id(sample)
        grp.attrs["n_spots"] = int(sub.shape[0])
        grp.attrs["token_dim"] = int(tokens.shape[1])

        sample_ids.append(sample)
        meta_rows.append(
            {
                "sample_id": sample,
                "group_name": group_name,
                "donor_id": grp.attrs["donor_id"],
                "stage": grp.attrs["stage"],
                "gsm_id": grp.attrs["gsm_id"],
                "n_spots": int(sub.shape[0]),
            }
        )

    root.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    root.attrs["group_key"] = group_key
    root.attrs["token_columns"] = [str(c) for c in token_cols]
    root.attrs["token_prefix"] = token_prefix
    root.attrs["n_groups"] = len(sample_ids)
    root.attrs["n_rows"] = int(df.shape[0])
    root.attrs["meta_rows"] = meta_rows

    audit = {
        "created_at_utc": root.attrs["created_at_utc"],
        "input_parquet": str(niche_tokens_parquet),
        "output_zarr": str(output_zarr),
        "group_key": group_key,
        "token_prefix_used": token_prefix,
        "n_rows": int(df.shape[0]),
        "n_groups": int(len(sample_ids)),
        "token_dim": int(len(token_cols)),
        "samples": meta_rows,
    }
    return NicheTokenBankBuildResult(
        output_path=output_zarr,
        sample_ids=sample_ids,
        token_columns=[str(c) for c in token_cols],
        n_rows=int(df.shape[0]),
        n_groups=len(sample_ids),
        audit=audit,
    )


class NicheTokenBank:
    """Reader and sampler for Zarr-based niche-token banks."""

    def __init__(
        self,
        zarr_path: Path,
        random_seed: int = 42,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Niche token bank not found: {self.zarr_path}")

        self.root = zarr.open_group(str(self.zarr_path), mode="r")
        self.token_columns = [str(x) for x in self.root.attrs.get("token_columns", [])]
        if not self.token_columns:
            raise ValueError(f"Token bank at {self.zarr_path} is missing token_columns metadata.")
        self.token_dim = len(self.token_columns)

        self._rng = np.random.default_rng(int(random_seed))
        self._sample_meta: dict[str, dict[str, Any]] = {}
        self._sample_tokens_cache: dict[str, np.ndarray] = {}
        self._sample_sizes: dict[str, int] = {}

        meta_rows = self.root.attrs.get("meta_rows", [])
        if meta_rows:
            for row in meta_rows:
                sample_id = str(row["sample_id"])
                self._sample_meta[sample_id] = dict(row)
                self._sample_sizes[sample_id] = int(row["n_spots"])
        else:
            for group_name in self.root.group_keys():
                grp = self.root[group_name]
                sample_id = str(grp.attrs.get("sample_id", group_name))
                row = {
                    "sample_id": sample_id,
                    "group_name": group_name,
                    "donor_id": str(grp.attrs.get("donor_id", "unknown_donor")),
                    "stage": str(grp.attrs.get("stage", "Unknown")),
                    "gsm_id": str(grp.attrs.get("gsm_id", _parse_gsm_from_sample_id(sample_id))),
                    "n_spots": int(grp.attrs.get("n_spots", grp["tokens"].shape[0])),
                }
                self._sample_meta[sample_id] = row
                self._sample_sizes[sample_id] = int(row["n_spots"])

        self._donor_stage_to_samples: dict[tuple[str, str], list[str]] = {}
        self._donor_to_samples: dict[str, list[str]] = {}
        for sample_id, row in self._sample_meta.items():
            donor = str(row.get("donor_id", "unknown_donor"))
            stage = str(row.get("stage", "Unknown"))
            self._donor_stage_to_samples.setdefault((donor, stage), []).append(sample_id)
            self._donor_to_samples.setdefault(donor, []).append(sample_id)
        self._all_samples = sorted(self._sample_meta.keys())

        self._kmeans_cache: dict[tuple[tuple[str, ...], int], np.ndarray] = {}

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self._sample_meta.keys())

    @property
    def sample_meta(self) -> dict[str, dict[str, Any]]:
        return dict(self._sample_meta)

    def _sample_group_name(self, sample_id: str) -> str:
        row = self._sample_meta.get(sample_id)
        if row is None:
            raise KeyError(f"Unknown sample_id in token bank: {sample_id}")
        return str(row.get("group_name") or _safe_group_name(sample_id))

    def _load_sample_tokens(self, sample_id: str) -> np.ndarray:
        if sample_id in self._sample_tokens_cache:
            return self._sample_tokens_cache[sample_id]
        group_name = self._sample_group_name(sample_id)
        arr = np.asarray(self.root[group_name]["tokens"][:], dtype=np.float32)
        self._sample_tokens_cache[sample_id] = arr
        self._sample_sizes[sample_id] = int(arr.shape[0])
        return arr

    def _load_sample_coords(self, sample_id: str) -> np.ndarray | None:
        """Load spatial (x, y) coordinates for a sample, or None if not stored."""
        group_name = self._sample_group_name(sample_id)
        grp = self.root[group_name]
        if "coords" not in grp:
            return None
        return np.asarray(grp["coords"][:], dtype=np.float32)

    def _candidate_samples(
        self,
        donor_id: str,
        stage: str,
        fallback: str,
    ) -> list[str]:
        key = (str(donor_id), str(stage))
        samples = list(self._donor_stage_to_samples.get(key, []))
        if samples:
            return samples

        fallback_norm = str(fallback).lower()
        if fallback_norm in {"donor", "donor_stage", "donor_any"}:
            donor_samples = list(self._donor_to_samples.get(str(donor_id), []))
            if donor_samples:
                return donor_samples
        if fallback_norm in {"global", "donor", "donor_stage", "donor_any"}:
            return list(self._all_samples)
        return samples

    def sample_tokens(
        self,
        *,
        donor_id: str,
        stage: str,
        m_niche: int,
        strategy: str = "random_m",
        fallback: str = "donor",
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        """Sample niche tokens and spatial coordinates for the requested donor/stage.

        Returns
        -------
        tokens : (m_niche, token_dim) float32
        coords : (m_niche, 2) float32 or None if coordinates not stored in bank
        meta : dict with provenance information
        """
        m = max(1, int(m_niche))
        samples = self._candidate_samples(donor_id=donor_id, stage=stage, fallback=fallback)
        if not samples:
            raise RuntimeError(
                f"No niche tokens available for donor='{donor_id}', stage='{stage}', fallback='{fallback}'."
            )
        samples = sorted(samples)
        strategy_norm = str(strategy).lower()

        if strategy_norm == "kmeans_m":
            key = (tuple(samples), m)
            centers = self._kmeans_cache.get(key)
            if centers is None:
                pooled = np.concatenate([self._load_sample_tokens(s) for s in samples], axis=0)
                if pooled.shape[0] <= m:
                    centers = pooled.astype(np.float32, copy=False)
                else:
                    from sklearn.cluster import MiniBatchKMeans

                    km = MiniBatchKMeans(
                        n_clusters=m,
                        random_state=0,
                        batch_size=min(8192, pooled.shape[0]),
                        n_init=1,
                    )
                    centers = km.fit(pooled).cluster_centers_.astype(np.float32, copy=False)
                self._kmeans_cache[key] = centers
            if centers.shape[0] < m:
                idx = self._rng.integers(0, centers.shape[0], size=m)
                sampled = centers[idx]
            else:
                sampled = centers[:m]
            # k-means centers don't have meaningful spatial coords
            return sampled.astype(np.float32, copy=False), None, {
                "samples_used": samples,
                "strategy": strategy_norm,
                "fallback": fallback,
            }

        if strategy_norm != "random_m":
            raise ValueError(f"Unsupported niche sampling strategy: {strategy!r}")

        sizes = np.asarray([self._sample_sizes.get(s, 0) for s in samples], dtype=np.float64)
        if np.any(sizes <= 0):
            sizes = np.asarray([self._load_sample_tokens(s).shape[0] for s in samples], dtype=np.float64)
        probs = sizes / sizes.sum()

        sample_choice = self._rng.choice(len(samples), size=m, replace=True, p=probs)
        out_tokens = np.empty((m, self.token_dim), dtype=np.float32)
        # Check if coords are available in the first candidate sample
        _has_coords = self._load_sample_coords(samples[0]) is not None
        out_coords: np.ndarray | None = np.empty((m, 2), dtype=np.float32) if _has_coords else None

        for sid in np.unique(sample_choice):
            sample_id = samples[int(sid)]
            arr = self._load_sample_tokens(sample_id)
            pos = np.where(sample_choice == sid)[0]
            rows = self._rng.integers(0, arr.shape[0], size=pos.shape[0])
            out_tokens[pos] = arr[rows]
            if out_coords is not None:
                c = self._load_sample_coords(sample_id)
                if c is not None:
                    out_coords[pos] = c[rows]
                else:
                    out_coords[pos] = 0.0

        return out_tokens, out_coords, {
            "samples_used": samples,
            "strategy": strategy_norm,
            "fallback": fallback,
        }
