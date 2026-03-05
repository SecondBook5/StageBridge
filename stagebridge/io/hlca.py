"""HLCA reference loading and latent-space alignment helpers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.request import urlretrieve

import anndata
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from tqdm.auto import tqdm

from stagebridge import config
from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import normalize_stage_series

log = get_logger(__name__)


HLCA_FULL_URL = (
    "https://datasets.cellxgene.cziscience.com/"
    "dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad"
)
HLCA_FULL_FILENAME = "hlca_full_v1.h5ad"


@dataclass(slots=True)
class HLCAReference:
    """Reference container for HLCA artifacts."""

    adata: Any
    source_path: Path
    latent_key: str = "X_hlca"


@dataclass(slots=True)
class HLCAMappingResult:
    """Artifacts and summary metrics from full-scale HLCA mapping."""

    run_id: str
    latent_h5ad_path: Path
    labels_parquet_path: Path
    mapping_report_path: Path
    gene_report_path: Path
    overlap_percent: float
    latent_shape: tuple[int, int]
    top10_labels: list[tuple[str, int]]
    peak_rss_mb: float
    wall_time_seconds: float
    report: dict[str, Any]


@dataclass(slots=True)
class HLCAEvaluationResult:
    """Artifacts and summary metrics from HLCA mapping evaluation."""

    run_id: str
    report_path: Path
    report: dict[str, Any]


def hlca_reference_dir() -> Path:
    """Return the canonical HLCA reference directory inside data root."""
    return config.resolve_path("data", "reference", "hlca")


def hlca_reference_h5ad(default_filename: str = HLCA_FULL_FILENAME) -> Path:
    """Return canonical path to the HLCA full h5ad."""
    return hlca_reference_dir() / default_filename


def download_hlca_reference(
    output_path: Path | None = None,
    url: str = HLCA_FULL_URL,
) -> Path:
    """Download the HLCA full reference h5ad to the configured reference dir."""
    dst = output_path or hlca_reference_h5ad()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log.info("HLCA file already exists at %s", dst)
        return dst
    log.info("Downloading HLCA reference from %s", url)
    urlretrieve(url, dst)  # nosec: trusted URL is user-provided in project docs
    log.info("Downloaded HLCA reference to %s", dst)
    return dst


def load_hlca_reference(
    h5ad_path: Path | None = None,
    backed: str | None = None,
    latent_key: str = "X_hlca",
) -> HLCAReference:
    """Load HLCA reference AnnData from disk."""
    try:
        import anndata
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "anndata is required to load HLCA reference. "
            "Install dependencies from environment.yml first."
        ) from exc

    path = Path(h5ad_path) if h5ad_path is not None else hlca_reference_h5ad()
    if not path.exists():
        raise FileNotFoundError(
            f"HLCA reference not found at {path}\n"
            "Expected full atlas file (.h5ad) at this path.\n"
            f"You can use download_hlca_reference() with URL:\n{HLCA_FULL_URL}"
        )

    adata = anndata.read_h5ad(path, backed=backed)
    log.info("Loaded HLCA reference: %s with shape %s", path, adata.shape)
    return HLCAReference(adata=adata, source_path=path, latent_key=latent_key)


def harmonize_hlca_metadata(
    adata: Any,
    stage_col: str = "stage",
    donor_col: str = "patient_id",
    sample_col: str = "sample_id",
    modality_col: str = "modality",
) -> None:
    """Standardize key metadata columns in ``adata.obs`` in place."""
    obs = adata.obs

    if stage_col in obs.columns:
        obs[stage_col] = normalize_stage_series(obs[stage_col])
    else:
        obs[stage_col] = "Unknown"

    if donor_col not in obs.columns:
        obs[donor_col] = "unknown_donor"
    if sample_col not in obs.columns:
        obs[sample_col] = "unknown_sample"
    if modality_col not in obs.columns:
        obs[modality_col] = "unknown_modality"

    # Common HLCA naming alternatives
    if "donor_id" in obs.columns and donor_col not in obs.columns:
        obs[donor_col] = obs["donor_id"].astype(str)
    if "sample" in obs.columns and sample_col not in obs.columns:
        obs[sample_col] = obs["sample"].astype(str)


def map_to_hlca_latent(
    adata: Any,
    reference: HLCAReference | None = None,
    input_key: str = "X_pca",
    output_key: str = "X_hlca",
    n_components: int = 64,
) -> None:
    """Create ``adata.obsm[output_key]`` with HLCA-aligned latent representation.

    Strategy:
    1. Use existing ``obsm[input_key]`` as source representation.
    2. If a reference with ``obsm[reference.latent_key]`` is available, match source
       latent to reference scale (z-score -> reference mean/std) for compatibility.
    3. If no reference latent is available, keep deterministic PCA fallback.
    """
    if input_key not in adata.obsm:
        raise KeyError(
            f"Input latent key '{input_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    X = np.asarray(adata.obsm[input_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D latent array in obsm['{input_key}'], got {X.shape}")

    n_components = min(n_components, X.shape[1])
    X = X[:, :n_components]

    if reference is not None and reference.latent_key in reference.adata.obsm:
        X_ref = np.asarray(reference.adata.obsm[reference.latent_key], dtype=np.float32)
        if X_ref.ndim == 2 and X_ref.shape[1] >= n_components:
            X_ref = X_ref[:, :n_components]
            src_mu = X.mean(axis=0, keepdims=True)
            src_sd = X.std(axis=0, keepdims=True) + 1e-6
            ref_mu = X_ref.mean(axis=0, keepdims=True)
            ref_sd = X_ref.std(axis=0, keepdims=True) + 1e-6
            X = (X - src_mu) / src_sd
            X = X * ref_sd + ref_mu
            log.info(
                "Mapped to HLCA-aligned latent (%s) using reference scaling.",
                reference.latent_key,
            )
        else:
            log.warning(
                "Reference latent key '%s' unavailable or mismatched; using fallback latent.",
                reference.latent_key,
            )
    else:
        log.info("No HLCA reference latent provided; using deterministic fallback latent.")

    adata.obsm[output_key] = X.astype(np.float32)


_ENSG_RE = re.compile(r"^(ENSG\d+)(?:\.\d+)?$")


def _now_rss_mb(process: psutil.Process) -> float:
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def _hash_string_array(values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for value in values.tolist():
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _canonicalize_ensg(value: str) -> str | None:
    m = _ENSG_RE.match(value)
    if m is None:
        return None
    return m.group(1)


def _canonicalize_ensg_array(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape[0], dtype=object)
    for i, v in enumerate(values.tolist()):
        out[i] = _canonicalize_ensg(str(v))
    return out


def _normalize_symbol(value: str) -> str:
    return str(value).strip().upper()


def _coerce_predict_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def _build_gene_lookup_with_cache(
    query_var_names: pd.Index,
    ref_adata: anndata.AnnData,
    cache_path: Path,
    mapping_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    query_vars = query_var_names.astype(str).to_numpy()
    ref_genes = ref_adata.var_names.astype(str).to_numpy()
    ref_index = pd.Index(ref_genes)

    query_hash = _hash_string_array(query_vars)
    ref_feature_name = (
        ref_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in ref_adata.var.columns
        else np.array([""] * len(ref_genes), dtype=object)
    )
    ref_sig = _hash_string_array(np.asarray([f"{g}|{n}" for g, n in zip(ref_genes, ref_feature_name)]))
    mapping_version = f"{mapping_source}:{ref_sig}"

    direct_lookup = {}
    canonical_query_ensg = _canonicalize_ensg_array(query_vars)
    for qidx, gene in enumerate(canonical_query_ensg.tolist()):
        if gene is None or gene in direct_lookup:
            continue
        direct_lookup[gene] = qidx
    direct_col_lookup = np.array([direct_lookup.get(g, -1) for g in ref_genes], dtype=np.int64)
    direct_overlap = float((direct_col_lookup >= 0).mean())

    cache_used = False
    mapped_ensg = np.full(query_vars.shape[0], None, dtype=object)
    if cache_path.exists():
        try:
            cache_df = pd.read_parquet(cache_path)
            cached = cache_df[
                (cache_df["query_hash"] == query_hash)
                & (cache_df["mapping_version"] == mapping_version)
            ].copy()
            if len(cached) == len(query_vars):
                cached = cached.sort_values("query_pos", kind="stable")
                mapped_ensg = cached["mapped_ensg"].replace({np.nan: None}).to_numpy(dtype=object)
                cache_used = True
        except Exception:
            cache_used = False

    duplicate_ref_symbols = 0
    if not cache_used:
        ref_symbol_df = pd.DataFrame(
            {
                "ensg": ref_genes,
                "symbol_norm": np.asarray([_normalize_symbol(x) for x in ref_feature_name], dtype=object),
            }
        )
        ref_symbol_df = ref_symbol_df[ref_symbol_df["symbol_norm"] != ""].copy()
        duplicate_ref_symbols = int(ref_symbol_df["symbol_norm"].duplicated(keep=False).sum())
        ref_symbol_df = ref_symbol_df.sort_values(["symbol_norm", "ensg"], kind="stable")
        ref_symbol_df = ref_symbol_df.drop_duplicates("symbol_norm", keep="first")
        symbol_to_ensg = pd.Series(ref_symbol_df["ensg"].to_numpy(), index=ref_symbol_df["symbol_norm"].to_numpy())

        query_symbol_norm = pd.Series([_normalize_symbol(x) for x in query_vars])
        mapped_from_symbol = query_symbol_norm.map(symbol_to_ensg).replace({np.nan: None}).to_numpy(dtype=object)
        has_canonical_ensg = np.array([ensg is not None for ensg in canonical_query_ensg], dtype=bool)
        mapped_ensg = np.where(has_canonical_ensg, canonical_query_ensg, mapped_from_symbol)

        # cache mapping for deterministic resume/reuse
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_rows = pd.DataFrame(
            {
                "query_hash": query_hash,
                "mapping_version": mapping_version,
                "query_pos": np.arange(len(query_vars), dtype=np.int64),
                "original_var": query_vars,
                "mapped_ensg": mapped_ensg,
            }
        )
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)
            existing = existing[
                ~(
                    (existing["query_hash"] == query_hash)
                    & (existing["mapping_version"] == mapping_version)
                )
            ]
            cache_rows = pd.concat([existing, cache_rows], ignore_index=True)
        cache_rows.to_parquet(cache_path, index=False)

    mapping_df = pd.DataFrame(
        {
            "mapped_ensg": mapped_ensg,
            "query_pos": np.arange(len(query_vars), dtype=np.int64),
        }
    )
    mapping_df = mapping_df.dropna(subset=["mapped_ensg"]).copy()
    duplicate_query_mapped = int(mapping_df["mapped_ensg"].duplicated(keep=False).sum())
    mapping_df = mapping_df.drop_duplicates("mapped_ensg", keep="first")
    ensg_to_qidx = pd.Series(mapping_df["query_pos"].to_numpy(), index=mapping_df["mapped_ensg"].to_numpy())

    mapped_col_lookup = ensg_to_qidx.reindex(ref_index).fillna(-1).astype(np.int64).to_numpy()
    mapped_overlap = float((mapped_col_lookup >= 0).mean())

    report = {
        "query_gene_count": int(len(query_vars)),
        "reference_gene_count": int(len(ref_genes)),
        "query_hash": query_hash,
        "mapping_source": mapping_source,
        "mapping_version": mapping_version,
        "direct_overlap_ratio": direct_overlap,
        "mapped_overlap_ratio": mapped_overlap,
        "cache_used": bool(cache_used),
        "duplicate_query_mapped_ensg_count": duplicate_query_mapped,
        "duplicate_reference_symbol_count": duplicate_ref_symbols,
        "duplicate_rule": "drop_duplicates_keep_first",
        "effective_n_genes_used": int((mapped_col_lookup >= 0).sum()),
        "missing_reference_genes": int((mapped_col_lookup < 0).sum()),
    }
    return mapped_col_lookup, report


def _slice_source_to_csr(
    source_matrix: Any,
    row_indexer: slice | np.ndarray,
    col_indexer: np.ndarray,
    dense_max_elements: int,
    state: dict[str, Any],
) -> sp.csr_matrix:
    chunk = source_matrix[row_indexer, col_indexer]
    if sp.issparse(chunk):
        return chunk.tocsr()

    dense = np.asarray(chunk)
    if dense.size > dense_max_elements:
        raise MemoryError(
            "Dense matrix materialization would exceed safety threshold: "
            f"{dense.size} elements > {dense_max_elements}."
        )
    state["densified"] = True
    return sp.csr_matrix(dense)


def _build_ref_aligned_chunk(
    source_matrix: Any,
    row_indexer: slice | np.ndarray,
    col_lookup: np.ndarray,
    dense_max_elements: int,
    state: dict[str, Any],
) -> sp.csr_matrix:
    present_positions = np.flatnonzero(col_lookup >= 0)
    present_query_cols = col_lookup[present_positions]

    X_present = _slice_source_to_csr(
        source_matrix=source_matrix,
        row_indexer=row_indexer,
        col_indexer=present_query_cols,
        dense_max_elements=dense_max_elements,
        state=state,
    ).astype(np.float32, copy=False)

    if present_positions.size == col_lookup.size:
        return X_present

    mapped_indices = present_positions[X_present.indices]
    X_full = sp.csr_matrix(
        (X_present.data, mapped_indices, X_present.indptr.copy()),
        shape=(X_present.shape[0], col_lookup.size),
        dtype=np.float32,
    )
    return X_full


def _build_query_chunk_adata(
    source_matrix: Any,
    row_indexer: slice | np.ndarray,
    col_lookup: np.ndarray,
    ref_var_df: pd.DataFrame,
    obs_index: pd.Index,
    dataset_values: np.ndarray,
    dense_max_elements: int,
    state: dict[str, Any],
) -> anndata.AnnData:
    X_chunk = _build_ref_aligned_chunk(
        source_matrix=source_matrix,
        row_indexer=row_indexer,
        col_lookup=col_lookup,
        dense_max_elements=dense_max_elements,
        state=state,
    )

    if isinstance(row_indexer, slice):
        idx = np.arange(row_indexer.start or 0, row_indexer.stop, dtype=np.int64)
    else:
        idx = np.asarray(row_indexer, dtype=np.int64)
    obs_chunk = pd.DataFrame(index=obs_index[idx].copy())
    obs_chunk["dataset"] = pd.Categorical(dataset_values[idx])
    obs_chunk["scanvi_label"] = "unlabeled"

    return anndata.AnnData(X=X_chunk, obs=obs_chunk, var=ref_var_df.copy())


def _stratified_subset_indices(
    donor_values: np.ndarray,
    stage_values: np.ndarray,
    max_cells: int,
    seed: int = 0,
) -> np.ndarray:
    n_obs = donor_values.shape[0]
    if max_cells <= 0 or max_cells >= n_obs:
        return np.arange(n_obs, dtype=np.int64)

    donor_codes, _ = pd.factorize(donor_values, sort=True)
    stage_codes, _ = pd.factorize(stage_values, sort=True)
    n_stage = int(stage_codes.max() + 1)
    group_codes = donor_codes.astype(np.int64) * (n_stage + 1) + stage_codes.astype(np.int64)
    _, inverse = np.unique(group_codes, return_inverse=True)

    counts = np.bincount(inverse)
    frac = float(max_cells) / float(n_obs)
    raw_targets = counts.astype(np.float64) * frac
    targets = np.floor(raw_targets).astype(np.int64)
    targets = np.maximum(targets, 1)
    targets = np.minimum(targets, counts)

    total = int(targets.sum())
    if total > max_cells:
        reducible = targets - 1
        need = total - max_cells
        for gid in np.argsort(-reducible):
            if need <= 0:
                break
            drop = int(min(reducible[gid], need))
            targets[gid] -= drop
            need -= drop
    elif total < max_cells:
        capacity = counts - targets
        need = max_cells - total
        residual = raw_targets - targets
        for gid in np.argsort(-residual):
            if need <= 0:
                break
            add = int(min(capacity[gid], need))
            targets[gid] += add
            need -= add

    rng = np.random.default_rng(seed)
    rand = rng.random(n_obs, dtype=np.float32)
    order = np.lexsort((rand, inverse))
    inv_sorted = inverse[order]
    starts = np.flatnonzero(np.r_[True, inv_sorted[1:] != inv_sorted[:-1]])
    ranks = np.arange(n_obs, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, n_obs]))
    keep_sorted = ranks < targets[inv_sorted]
    keep = np.sort(order[keep_sorted]).astype(np.int64)
    return keep


def _extract_label_categories(query_model: Any) -> np.ndarray:
    state = query_model.adata_manager.get_state_registry("labels")
    cats = np.asarray(state.categorical_mapping, dtype=object)
    return cats.astype(str)


def _load_progress(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_progress(progress_path: Path, payload: dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _open_or_create_memmap(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    expected_nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    mode = "r+" if path.exists() and path.stat().st_size == expected_nbytes else "w+"
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


def _run_optional_knn_label_transfer(
    ref_adata: anndata.AnnData,
    query_latent: np.ndarray,
    label_levels: list[str],
    batch_size: int,
    show_progress: bool,
) -> dict[str, np.ndarray]:
    from pynndescent import NNDescent

    if "X_scanvi_emb" not in ref_adata.obsm:
        raise KeyError("Reference AnnData missing obsm['X_scanvi_emb'] for KNN label transfer.")
    ref_latent = np.asarray(ref_adata.obsm["X_scanvi_emb"], dtype=np.float32)
    nn_index = NNDescent(ref_latent, n_neighbors=64, metric="euclidean", random_state=0)

    result: dict[str, np.ndarray] = {}
    n_obs = query_latent.shape[0]
    for level in label_levels:
        if level not in ref_adata.obs.columns:
            continue
        ref_cat = pd.Categorical(ref_adata.obs[level].astype(str))
        ref_codes = ref_cat.codes.astype(np.int32, copy=False)
        categories = np.asarray(ref_cat.categories, dtype=object)

        pred_codes = np.full(n_obs, -1, dtype=np.int32)
        uncertainty = np.full(n_obs, np.nan, dtype=np.float32)

        starts = range(0, n_obs, batch_size)
        if show_progress:
            starts = tqdm(
                starts,
                total=(n_obs + batch_size - 1) // batch_size,
                desc=f"HLCA KNN transfer [{level}]",
                unit="chunk",
            )
        for start in starts:
            end = min(start + batch_size, n_obs)
            idxs, dists = nn_index.query(query_latent[start:end], k=64)
            weights = 1.0 / (dists + 1e-6)
            neigh_codes = ref_codes[idxs]
            m, k = neigh_codes.shape

            score = np.zeros((m, len(categories)), dtype=np.float32)
            row = np.repeat(np.arange(m), k)
            col = neigh_codes.reshape(-1)
            w = weights.reshape(-1)
            valid = col >= 0
            np.add.at(score, (row[valid], col[valid]), w[valid])

            denom = score.sum(axis=1, keepdims=True) + 1e-8
            probs = score / denom
            best = probs.argmax(axis=1)
            pred_codes[start:end] = best
            uncertainty[start:end] = 1.0 - probs[np.arange(m), best]

        result[f"{level}_knn_label"] = categories[pred_codes]
        result[f"{level}_knn_uncertainty"] = uncertainty
    return result


def _select_ref_latent_key(ref_adata: anndata.AnnData) -> str:
    for key in ("X_scanvi_emb", "X_scANVI", "X_scVI", "X_hlca"):
        if key in ref_adata.obsm:
            return key
    raise KeyError(
        "Reference AnnData is missing a usable latent key. "
        "Expected one of: X_scanvi_emb, X_scANVI, X_scVI, X_hlca."
    )


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _stratified_sample_indices(
    labels: np.ndarray,
    max_n: int,
    seed: int,
) -> np.ndarray:
    n = labels.shape[0]
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    label_codes, _ = pd.factorize(labels, sort=True)
    counts = np.bincount(label_codes)

    frac = float(max_n) / float(n)
    raw_targets = counts.astype(np.float64) * frac
    targets = np.floor(raw_targets).astype(np.int64)
    targets = np.maximum(targets, 1)
    targets = np.minimum(targets, counts)

    total = int(targets.sum())
    if total > max_n:
        reducible = targets - 1
        need = total - max_n
        for gid in np.argsort(-reducible):
            if need <= 0:
                break
            drop = int(min(reducible[gid], need))
            targets[gid] -= drop
            need -= drop
    elif total < max_n:
        capacity = counts - targets
        need = max_n - total
        residual = raw_targets - targets
        for gid in np.argsort(-residual):
            if need <= 0:
                break
            add = int(min(capacity[gid], need))
            targets[gid] += add
            need -= add

    rand = rng.random(n, dtype=np.float32)
    order = np.lexsort((rand, label_codes))
    codes_sorted = label_codes[order]
    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    ranks = np.arange(n, dtype=np.int64) - np.repeat(starts, np.diff(np.r_[starts, n]))
    keep_sorted = ranks < targets[codes_sorted]
    return np.sort(order[keep_sorted]).astype(np.int64)


def _coerce_dense_float32(x: Any) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def evaluate_hlca_mapping_outputs(
    *,
    run_id: str,
    latent_h5ad_path: Path,
    labels_parquet_path: Path,
    report_path: Path,
    processed_hlca_dir: Path,
    eval_cfg: dict[str, Any] | None = None,
) -> HLCAEvaluationResult:
    """Quantitatively evaluate HLCA label quality against reference neighborhoods."""
    from scvi.hub import HubModel
    from sklearn.neighbors import NearestNeighbors

    eval_cfg = dict(eval_cfg or {})
    wall_t0 = time.perf_counter()
    stage_times: dict[str, float] = {}
    process = psutil.Process()
    peak_rss_mb = _now_rss_mb(process)

    def mark_peak() -> None:
        nonlocal peak_rss_mb
        peak_rss_mb = max(peak_rss_mb, _now_rss_mb(process))

    def stage_start() -> float:
        return time.perf_counter()

    def stage_done(name: str, t0: float) -> None:
        stage_times[name] = time.perf_counter() - t0

    min_label_coverage = float(eval_cfg.get("min_label_coverage", 0.95))
    min_majority_agreement = float(eval_cfg.get("min_majority_agreement", 0.55))
    min_neighbor_agreement = float(eval_cfg.get("min_neighbor_agreement", 0.35))
    max_query_cells = int(eval_cfg.get("max_query_cells_knn", 100_000))
    max_ref_cells = int(eval_cfg.get("max_ref_cells_knn", 300_000))
    knn_k = int(eval_cfg.get("knn_k", 30))
    random_seed = int(eval_cfg.get("random_seed", 0))

    repo_id = str(eval_cfg.get("hub_repo_id", "scvi-tools/human-lung-cell-atlas-scanvi"))
    model_cache_dir = Path(str(eval_cfg.get("model_cache_dir", processed_hlca_dir / "hub_cache")))

    latent_h5ad_path = Path(latent_h5ad_path)
    labels_parquet_path = Path(labels_parquet_path)
    report_path = Path(report_path)
    processed_hlca_dir = Path(processed_hlca_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not latent_h5ad_path.exists():
        raise FileNotFoundError(f"Missing latent h5ad: {latent_h5ad_path}")
    if not labels_parquet_path.exists():
        raise FileNotFoundError(f"Missing labels parquet: {labels_parquet_path}")

    t0 = stage_start()
    latent_adata = anndata.read_h5ad(latent_h5ad_path, backed="r")
    obs_index = latent_adata.obs_names.astype(str)
    n_obs = int(latent_adata.n_obs)
    n_latent = int(latent_adata.n_vars)
    stage_done("load_latent", t0)
    mark_peak()

    t0 = stage_start()
    labels_df = pd.read_parquet(labels_parquet_path)
    labels_df.index = labels_df.index.astype(str)
    label_col = "hlca_label" if "hlca_label" in labels_df.columns else None
    if label_col is None:
        raise KeyError("labels parquet is missing required column 'hlca_label'.")

    aligned_labels = pd.Series(index=obs_index, dtype=object, name="hlca_label")
    overlap_idx = obs_index.intersection(labels_df.index)
    aligned_labels.loc[overlap_idx] = labels_df.loc[overlap_idx, label_col].astype(str).to_numpy()
    stage_done("load_labels", t0)
    mark_peak()

    label_non_null = aligned_labels.notna()
    label_coverage = float(label_non_null.mean())
    label_counts = aligned_labels[label_non_null].astype(str).value_counts()
    top10_labels = [(str(k), int(v)) for k, v in label_counts.head(10).items()]

    prob_col = None
    entropy_col = None
    for candidate in ("hlca_max_prob", "max_prob"):
        if candidate in labels_df.columns:
            prob_col = candidate
            break
    for candidate in ("hlca_entropy", "entropy"):
        if candidate in labels_df.columns:
            entropy_col = candidate
            break

    aligned_prob = None
    aligned_entropy = None
    if prob_col is not None:
        aligned_prob = pd.Series(index=obs_index, dtype="float32")
        aligned_prob.loc[overlap_idx] = labels_df.loc[overlap_idx, prob_col].astype("float32").to_numpy()
    if entropy_col is not None:
        aligned_entropy = pd.Series(index=obs_index, dtype="float32")
        aligned_entropy.loc[overlap_idx] = labels_df.loc[overlap_idx, entropy_col].astype("float32").to_numpy()

    # Pull HLCA reference and build a nearest-neighbor agreement check.
    t0 = stage_start()
    hubmodel = HubModel.pull_from_huggingface_hub(repo_id, cache_dir=model_cache_dir)
    ref_adata = hubmodel.adata
    ref_latent_key = _select_ref_latent_key(ref_adata)
    if "scanvi_label" not in ref_adata.obs.columns:
        raise KeyError("Reference AnnData is missing obs['scanvi_label'].")
    stage_done("pull_reference", t0)
    mark_peak()

    t0 = stage_start()
    ref_latent = np.asarray(ref_adata.obsm[ref_latent_key], dtype=np.float32)
    ref_labels = ref_adata.obs["scanvi_label"].astype(str).to_numpy()
    ref_keep = _stratified_sample_indices(ref_labels, max_ref_cells, random_seed)
    ref_latent_sub = ref_latent[ref_keep]
    ref_labels_sub = ref_labels[ref_keep]
    stage_done("reference_sampling", t0)
    mark_peak()

    t0 = stage_start()
    query_candidates = np.flatnonzero(label_non_null.to_numpy())
    if query_candidates.size == 0:
        raise RuntimeError("No non-null hlca_label values found for evaluation.")
    rng = np.random.default_rng(random_seed)
    if max_query_cells > 0 and query_candidates.size > max_query_cells:
        query_keep = np.sort(rng.choice(query_candidates, size=max_query_cells, replace=False))
    else:
        query_keep = query_candidates

    query_latent = _coerce_dense_float32(latent_adata.X[query_keep, :])
    query_labels = aligned_labels.iloc[query_keep].astype(str).to_numpy()
    stage_done("query_sampling", t0)
    mark_peak()

    t0 = stage_start()
    knn = NearestNeighbors(
        n_neighbors=max(1, knn_k),
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    if query_latent.shape[1] != ref_latent_sub.shape[1]:
        used_dim = int(min(query_latent.shape[1], ref_latent_sub.shape[1]))
        query_latent = query_latent[:, :used_dim]
        ref_latent_sub = ref_latent_sub[:, :used_dim]
    else:
        used_dim = int(query_latent.shape[1])
    knn.fit(ref_latent_sub)
    neigh_dist, neigh_idx = knn.kneighbors(query_latent, return_distance=True)
    neigh_labels = ref_labels_sub[neigh_idx]

    ref_categories = pd.Index(np.unique(ref_labels_sub))
    cat_to_code = pd.Series(np.arange(len(ref_categories), dtype=np.int32), index=ref_categories)
    query_codes = cat_to_code.reindex(query_labels).fillna(-1).astype(np.int32).to_numpy()
    neigh_codes = (
        cat_to_code.reindex(neigh_labels.reshape(-1))
        .fillna(-1)
        .astype(np.int32)
        .to_numpy()
        .reshape(neigh_labels.shape)
    )

    valid_pred = query_codes >= 0
    per_cell_neighbor_match = np.full(query_keep.shape[0], np.nan, dtype=np.float32)
    if np.any(valid_pred):
        match = neigh_codes == query_codes[:, None]
        per_cell_neighbor_match[valid_pred] = match[valid_pred].mean(axis=1).astype(np.float32)
        mean_neighbor_agreement = float(np.nanmean(per_cell_neighbor_match))
    else:
        mean_neighbor_agreement = float("nan")

    rows = np.repeat(np.arange(neigh_codes.shape[0]), neigh_codes.shape[1])
    cols = neigh_codes.reshape(-1)
    valid_cols = cols >= 0
    votes = np.zeros((neigh_codes.shape[0], len(ref_categories)), dtype=np.int32)
    np.add.at(votes, (rows[valid_cols], cols[valid_cols]), 1)
    majority_codes = votes.argmax(axis=1)
    majority_labels = ref_categories.to_numpy(dtype=object)[majority_codes]
    if np.any(valid_pred):
        majority_agreement = float(np.mean(majority_labels[valid_pred] == query_labels[valid_pred]))
    else:
        majority_agreement = float("nan")
    stage_done("knn_agreement", t0)
    mark_peak()

    # Donor/stage composition drift checks in predicted label space.
    t0 = stage_start()
    obs_df = latent_adata.obs
    donor_col = "donor_id" if "donor_id" in obs_df.columns else None
    stage_col = "stage" if "stage" in obs_df.columns else None

    label_space = pd.Index(label_counts.index.astype(str))
    global_counts = label_counts.reindex(label_space, fill_value=0).to_numpy(dtype=np.float64)
    global_dist = global_counts / max(global_counts.sum(), 1.0)

    donor_js: dict[str, float] = {}
    if donor_col is not None and label_space.size > 0:
        donor_values = obs_df[donor_col].astype(str).to_numpy()
        label_values = aligned_labels.astype(str).to_numpy()
        for donor in np.unique(donor_values):
            donor_mask = donor_values == donor
            donor_labels = pd.Series(label_values[donor_mask])
            donor_labels = donor_labels[donor_labels != "nan"]
            if donor_labels.empty:
                continue
            donor_counts = donor_labels.value_counts().reindex(label_space, fill_value=0).to_numpy(dtype=np.float64)
            donor_dist = donor_counts / max(donor_counts.sum(), 1.0)
            donor_js[str(donor)] = _js_divergence(donor_dist, global_dist)

    stage_js: dict[str, float] = {}
    if stage_col is not None and label_space.size > 0:
        stage_values = obs_df[stage_col].astype(str).to_numpy()
        label_values = aligned_labels.astype(str).to_numpy()
        for stage in np.unique(stage_values):
            stage_mask = stage_values == stage
            stage_labels = pd.Series(label_values[stage_mask])
            stage_labels = stage_labels[stage_labels != "nan"]
            if stage_labels.empty:
                continue
            stage_counts = stage_labels.value_counts().reindex(label_space, fill_value=0).to_numpy(dtype=np.float64)
            stage_dist = stage_counts / max(stage_counts.sum(), 1.0)
            stage_js[str(stage)] = _js_divergence(stage_dist, global_dist)

    stage_done("composition_drift", t0)
    mark_peak()

    prob_summary: dict[str, float] = {}
    corr_prob_vs_knn = None
    if aligned_prob is not None:
        prob_values = aligned_prob.iloc[query_keep].to_numpy(dtype=np.float32)
        valid_prob = np.isfinite(prob_values) & np.isfinite(per_cell_neighbor_match)
        if np.any(valid_prob):
            prob_summary = {
                "mean": float(np.nanmean(prob_values[valid_prob])),
                "median": float(np.nanmedian(prob_values[valid_prob])),
                "p10": float(np.nanpercentile(prob_values[valid_prob], 10)),
                "p90": float(np.nanpercentile(prob_values[valid_prob], 90)),
            }
            corr_prob_vs_knn = float(
                pd.Series(prob_values[valid_prob]).corr(
                    pd.Series(per_cell_neighbor_match[valid_prob]),
                    method="spearman",
                )
            )
    entropy_summary: dict[str, float] = {}
    if aligned_entropy is not None:
        ent_values = aligned_entropy.iloc[query_keep].to_numpy(dtype=np.float32)
        valid_ent = np.isfinite(ent_values)
        if np.any(valid_ent):
            entropy_summary = {
                "mean": float(np.nanmean(ent_values[valid_ent])),
                "median": float(np.nanmedian(ent_values[valid_ent])),
                "p10": float(np.nanpercentile(ent_values[valid_ent], 10)),
                "p90": float(np.nanpercentile(ent_values[valid_ent], 90)),
            }

    checks = {
        "label_coverage": bool(label_coverage >= min_label_coverage),
        "majority_agreement": bool(majority_agreement >= min_majority_agreement),
        "mean_neighbor_agreement": bool(mean_neighbor_agreement >= min_neighbor_agreement),
    }
    all_checks_pass = all(checks.values())

    report = {
        "ok": bool(all_checks_pass),
        "run_id": run_id,
        "inputs": {
            "latent_h5ad": str(latent_h5ad_path),
            "labels_parquet": str(labels_parquet_path),
            "reference_repo_id": repo_id,
            "reference_latent_key": ref_latent_key,
        },
        "counts": {
            "n_obs_total": n_obs,
            "n_latent_dim": n_latent,
            "n_labeled_cells": int(label_non_null.sum()),
            "label_coverage": label_coverage,
            "top10_labels": top10_labels,
        },
        "knn_validation": {
            "k": int(knn_k),
            "n_query_cells_evaluated": int(query_keep.shape[0]),
            "n_reference_cells_used": int(ref_latent_sub.shape[0]),
            "latent_dim_used": used_dim,
            "majority_agreement": majority_agreement,
            "mean_neighbor_agreement": mean_neighbor_agreement,
            "mean_neighbor_distance": float(np.mean(neigh_dist)),
        },
        "uncertainty": {
            "probability_column": prob_col,
            "entropy_column": entropy_col,
            "probability_summary": prob_summary,
            "entropy_summary": entropy_summary,
            "spearman_prob_vs_neighbor_agreement": corr_prob_vs_knn,
        },
        "composition_drift": {
            "donor_js_mean": float(np.mean(list(donor_js.values()))) if donor_js else None,
            "donor_js_max": float(np.max(list(donor_js.values()))) if donor_js else None,
            "stage_js_mean": float(np.mean(list(stage_js.values()))) if stage_js else None,
            "stage_js_max": float(np.max(list(stage_js.values()))) if stage_js else None,
            "donor_js": donor_js,
            "stage_js": stage_js,
        },
        "thresholds": {
            "min_label_coverage": min_label_coverage,
            "min_majority_agreement": min_majority_agreement,
            "min_mean_neighbor_agreement": min_neighbor_agreement,
        },
        "checks": checks,
        "timing_seconds": stage_times,
        "total_wall_time_seconds": time.perf_counter() - wall_t0,
        "peak_rss_mb": peak_rss_mb,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if hasattr(latent_adata, "isbacked") and latent_adata.isbacked:
        latent_adata.file.close()

    return HLCAEvaluationResult(
        run_id=run_id,
        report_path=report_path,
        report=report,
    )


def map_full_snrna_with_hlca(
    *,
    run_id: str,
    snrna_h5ad_path: Path,
    output_latent_h5ad_path: Path,
    output_labels_parquet_path: Path,
    mapping_report_path: Path,
    gene_report_path: Path,
    progress_path: Path,
    processed_hlca_dir: Path,
    hlca_cfg: dict[str, Any],
) -> HLCAMappingResult:
    """Map full snRNA AnnData into HLCA latent and label space at scale."""
    from scvi.hub import HubModel
    from scvi.model import SCANVI

    wall_t0 = time.perf_counter()
    stage_times: dict[str, float] = {}
    process = psutil.Process()
    peak_rss_mb = _now_rss_mb(process)
    state: dict[str, Any] = {"used_backed": True, "densified": False}

    def mark_peak() -> None:
        nonlocal peak_rss_mb
        peak_rss_mb = max(peak_rss_mb, _now_rss_mb(process))

    def stage_start() -> float:
        return time.perf_counter()

    def stage_done(name: str, t0: float) -> None:
        stage_times[name] = time.perf_counter() - t0

    snrna_h5ad_path = Path(snrna_h5ad_path)
    output_latent_h5ad_path = Path(output_latent_h5ad_path)
    output_labels_parquet_path = Path(output_labels_parquet_path)
    mapping_report_path = Path(mapping_report_path)
    gene_report_path = Path(gene_report_path)
    progress_path = Path(progress_path)
    processed_hlca_dir = Path(processed_hlca_dir)

    output_latent_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_report_path.parent.mkdir(parents=True, exist_ok=True)
    gene_report_path.parent.mkdir(parents=True, exist_ok=True)
    processed_hlca_dir.mkdir(parents=True, exist_ok=True)

    repo_id = str(hlca_cfg.get("hub_repo_id", "scvi-tools/human-lung-cell-atlas-scanvi"))
    model_cache_dir = Path(str(hlca_cfg.get("model_cache_dir", processed_hlca_dir / "hub_cache")))
    query_model_dir = Path(str(hlca_cfg.get("query_model_dir", processed_hlca_dir / "query_model_full")))
    surgery_epochs = int(hlca_cfg.get("surgery_epochs", 500))
    batch_size_infer = int(hlca_cfg.get("batch_size_infer", 1024))
    inference_chunk_size = int(hlca_cfg.get("inference_chunk_size", batch_size_infer * 8))
    dense_max_elements = int(hlca_cfg.get("dense_max_elements", 200_000_000))
    train_max_cells = int(hlca_cfg.get("train_max_cells", 200_000))
    query_dataset_mode = str(hlca_cfg.get("query_dataset_mode", "donor"))
    query_dataset_constant = str(hlca_cfg.get("query_dataset_constant", "stagebridge_query"))
    export_probs = bool(hlca_cfg.get("export_probs", False))
    knn_label_transfer_levels = bool(hlca_cfg.get("knn_label_transfer_levels", False))
    label_levels = [str(x) for x in hlca_cfg.get("label_levels", [])]
    keep_intermediates = bool(hlca_cfg.get("keep_intermediates", False))
    resume_enabled = bool(hlca_cfg.get("resume", True))
    show_progress = bool(hlca_cfg.get("show_progress", True))

    # Pull model and inspect registry
    t0 = stage_start()
    hubmodel = HubModel.pull_from_huggingface_hub(repo_id, cache_dir=model_cache_dir)
    ref_adata = hubmodel.adata
    ref_model = hubmodel.model
    stage_done("pull_model", t0)
    mark_peak()

    setup_args = ref_model.adata_manager.registry.get("setup_args", {})
    if setup_args.get("batch_key") != "dataset":
        raise ValueError(f"Reference batch_key mismatch: expected 'dataset', got {setup_args.get('batch_key')!r}")
    if setup_args.get("labels_key") != "scanvi_label":
        raise ValueError(
            f"Reference labels_key mismatch: expected 'scanvi_label', got {setup_args.get('labels_key')!r}"
        )
    if setup_args.get("unlabeled_category") != "unlabeled":
        raise ValueError(
            "Reference unlabeled_category mismatch: "
            f"expected 'unlabeled', got {setup_args.get('unlabeled_category')!r}"
        )
    if setup_args.get("layer", None) is not None:
        raise ValueError(f"Reference layer mismatch: expected None, got {setup_args.get('layer')!r}")

    adata_full = anndata.read_h5ad(snrna_h5ad_path, backed="r")
    n_obs = int(adata_full.n_obs)
    obs_index = adata_full.obs_names.copy()

    if query_dataset_mode not in {"donor", "constant"}:
        raise ValueError(f"Unsupported query_dataset_mode: {query_dataset_mode!r}")
    donor_values = adata_full.obs["donor_id"].astype(str).to_numpy()
    stage_values = adata_full.obs["stage"].astype(str).to_numpy()
    gsm_values = adata_full.obs["gsm_id"].astype(str).to_numpy()
    sample_values = adata_full.obs["sample_id"].astype(str).to_numpy()
    dataset_values = donor_values if query_dataset_mode == "donor" else np.full(n_obs, query_dataset_constant)

    # Count-layer routing
    source_layer = None
    source_matrix = adata_full.X
    if "counts" in adata_full.layers:
        probe = adata_full.X[: min(128, n_obs), : min(128, adata_full.n_vars)]
        count_like = True
        if sp.issparse(probe):
            data = np.asarray(probe.data)
            if data.size and (np.any(data < 0) or not np.allclose(data, np.round(data), atol=1e-6)):
                count_like = False
        else:
            arr = np.asarray(probe)
            if arr.size and (np.any(arr < 0) or not np.allclose(arr, np.round(arr), atol=1e-6)):
                count_like = False
        if not count_like:
            source_layer = "counts"
            source_matrix = adata_full.layers["counts"]

    # Gene overlap check + deterministic mapping
    t0 = stage_start()
    query_vars = adata_full.var_names.astype(str).to_numpy()
    ref_genes = ref_adata.var_names.astype(str).to_numpy()
    canonical_query = _canonicalize_ensg_array(query_vars)
    direct_lookup = {}
    for qidx, gene in enumerate(canonical_query.tolist()):
        if gene is None or gene in direct_lookup:
            continue
        direct_lookup[gene] = qidx
    direct_overlap = float(np.mean(np.array([g in direct_lookup for g in ref_genes], dtype=np.float32)))
    stage_done("gene_overlap_check", t0)

    t0 = stage_start()
    cache_path = processed_hlca_dir / "gene_map_cache.parquet"
    col_lookup, gene_report = _build_gene_lookup_with_cache(
        query_var_names=adata_full.var_names,
        ref_adata=ref_adata,
        cache_path=cache_path,
        mapping_source="hlca_ref_feature_name",
    )
    overlap_ratio = float((col_lookup >= 0).mean())
    n_ref_genes_total = int(len(col_lookup))
    n_ref_genes_present = int((col_lookup >= 0).sum())
    gene_report["overlap_ratio_final"] = overlap_ratio
    gene_report["overlap_percent_final"] = 100.0 * overlap_ratio
    gene_report["fails_overlap_threshold_0_70"] = bool(overlap_ratio < 0.70)
    gene_report["n_reference_genes_total"] = n_ref_genes_total
    gene_report["n_reference_genes_present_in_query"] = n_ref_genes_present
    gene_report["effective_n_genes_used_after_zero_fill"] = n_ref_genes_total
    gene_report_path.write_text(json.dumps(gene_report, indent=2), encoding="utf-8")
    stage_done("gene_id_mapping", t0)
    mark_peak()
    if overlap_ratio < 0.70:
        raise RuntimeError(
            "Gene overlap with HLCA reference remained below 0.70 after mapping. "
            f"See report: {gene_report_path}"
        )

    ref_var_df = ref_adata.var.copy()
    ref_var_df.index = ref_adata.var_names.astype(str)

    # Subset selection for query surgery training
    t0 = stage_start()
    train_idx = _stratified_subset_indices(
        donor_values=donor_values,
        stage_values=stage_values,
        max_cells=train_max_cells,
        seed=0,
    )
    stage_done("subset_selection", t0)
    mark_peak()

    # Build and train query model
    t0 = stage_start()
    train_chunk = _build_query_chunk_adata(
        source_matrix=source_matrix,
        row_indexer=train_idx,
        col_lookup=col_lookup,
        ref_var_df=ref_var_df,
        obs_index=obs_index,
        dataset_values=dataset_values,
        dense_max_elements=dense_max_elements,
        state=state,
    )
    SCANVI.prepare_query_anndata(train_chunk, ref_model)
    query_model = SCANVI.load_query_data(train_chunk, ref_model)
    query_model.train(
        max_epochs=surgery_epochs,
        early_stopping=bool(hlca_cfg.get("early_stopping", True)),
        early_stopping_monitor=str(hlca_cfg.get("early_stopping_monitor", "elbo_train")),
        early_stopping_patience=int(hlca_cfg.get("early_stopping_patience", 10)),
        early_stopping_min_delta=float(hlca_cfg.get("early_stopping_min_delta", 0.001)),
        plan_kwargs={"weight_decay": float(hlca_cfg.get("weight_decay", 0.0))},
        enable_progress_bar=show_progress,
    )
    query_model_dir.parent.mkdir(parents=True, exist_ok=True)
    query_model.save(query_model_dir, overwrite=True)
    stage_done("query_train", t0)
    mark_peak()

    latent_dim = int(getattr(query_model.module, "n_latent", 30))
    temp_dir = processed_hlca_dir / "tmp" / run_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    latent_memmap_path = temp_dir / "latent.float32.mmap"
    label_codes_memmap_path = temp_dir / "label_codes.int32.mmap"
    max_prob_memmap_path = temp_dir / "max_prob.float32.mmap"
    entropy_memmap_path = temp_dir / "entropy.float32.mmap"

    progress = _load_progress(progress_path) if resume_enabled else {}
    if progress.get("n_obs") != n_obs or progress.get("latent_dim") != latent_dim:
        progress = {
            "run_id": run_id,
            "n_obs": n_obs,
            "latent_dim": latent_dim,
            "latent_completed_rows": 0,
            "labels_completed_rows": 0,
            "status": "initialized",
        }
        _save_progress(progress_path, progress)
    elif progress.get("latent_completed_rows", 0) > 0 or progress.get("labels_completed_rows", 0) > 0:
        log.info(
            "Resuming HLCA mapping run_id=%s from latent_rows=%s labels_rows=%s",
            run_id,
            progress.get("latent_completed_rows", 0),
            progress.get("labels_completed_rows", 0),
        )

    latent_mm = _open_or_create_memmap(
        latent_memmap_path,
        dtype="float32",
        shape=(n_obs, latent_dim),
    )
    label_codes_mm = _open_or_create_memmap(
        label_codes_memmap_path,
        dtype="int32",
        shape=(n_obs,),
    )
    if export_probs:
        max_prob_mm = _open_or_create_memmap(max_prob_memmap_path, dtype="float32", shape=(n_obs,))
        entropy_mm = _open_or_create_memmap(entropy_memmap_path, dtype="float32", shape=(n_obs,))
    else:
        max_prob_mm = None
        entropy_mm = None

    # Full latent inference
    t0 = stage_start()
    latent_done = int(progress.get("latent_completed_rows", 0))
    latent_starts = range(latent_done, n_obs, inference_chunk_size)
    if show_progress:
        latent_starts = tqdm(
            latent_starts,
            total=max(0, (n_obs - latent_done + inference_chunk_size - 1) // inference_chunk_size),
            desc="HLCA latent inference",
            unit="chunk",
        )
    for start in latent_starts:
        end = min(start + inference_chunk_size, n_obs)
        query_chunk = _build_query_chunk_adata(
            source_matrix=source_matrix,
            row_indexer=slice(start, end),
            col_lookup=col_lookup,
            ref_var_df=ref_var_df,
            obs_index=obs_index,
            dataset_values=dataset_values,
            dense_max_elements=dense_max_elements,
            state=state,
        )
        SCANVI.prepare_query_anndata(query_chunk, ref_model)
        latent_batch = query_model.get_latent_representation(query_chunk, batch_size=batch_size_infer)
        latent_mm[start:end, :] = np.asarray(latent_batch, dtype=np.float32)
        progress["latent_completed_rows"] = end
        progress["status"] = "latent_inference"
        _save_progress(progress_path, progress)
        mark_peak()
    latent_mm.flush()
    stage_done("latent_inference_full", t0)

    # Full label inference
    t0 = stage_start()
    label_categories = _extract_label_categories(query_model)
    labels_done = int(progress.get("labels_completed_rows", 0))
    label_starts = range(labels_done, n_obs, inference_chunk_size)
    if show_progress:
        label_starts = tqdm(
            label_starts,
            total=max(0, (n_obs - labels_done + inference_chunk_size - 1) // inference_chunk_size),
            desc="HLCA label inference",
            unit="chunk",
        )
    for start in label_starts:
        end = min(start + inference_chunk_size, n_obs)
        query_chunk = _build_query_chunk_adata(
            source_matrix=source_matrix,
            row_indexer=slice(start, end),
            col_lookup=col_lookup,
            ref_var_df=ref_var_df,
            obs_index=obs_index,
            dataset_values=dataset_values,
            dense_max_elements=dense_max_elements,
            state=state,
        )
        SCANVI.prepare_query_anndata(query_chunk, ref_model)

        pred = _coerce_predict_output(query_model.predict(query_chunk, batch_size=batch_size_infer))
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0].to_numpy()
        pred = np.asarray(pred, dtype=object).astype(str)
        codes = pd.Categorical(pred, categories=label_categories).codes.astype(np.int32, copy=False)
        label_codes_mm[start:end] = codes

        if export_probs and max_prob_mm is not None and entropy_mm is not None:
            probs = _coerce_predict_output(
                query_model.predict(query_chunk, soft=True, batch_size=batch_size_infer)
            )
            if isinstance(probs, pd.DataFrame):
                probs = probs.to_numpy(dtype=np.float32)
            probs = np.asarray(probs, dtype=np.float32)
            max_prob_mm[start:end] = probs.max(axis=1)
            entropy_mm[start:end] = -(probs * np.log(probs + 1e-12)).sum(axis=1)

        progress["labels_completed_rows"] = end
        progress["status"] = "label_inference"
        _save_progress(progress_path, progress)
        mark_peak()
    label_codes_mm.flush()
    if export_probs and max_prob_mm is not None and entropy_mm is not None:
        max_prob_mm.flush()
        entropy_mm.flush()
    stage_done("label_inference_full", t0)

    # Optional KNN label transfer on additional hierarchy levels
    t0 = stage_start()
    knn_outputs: dict[str, np.ndarray] = {}
    if knn_label_transfer_levels:
        knn_outputs = _run_optional_knn_label_transfer(
            ref_adata=ref_adata,
            query_latent=np.asarray(latent_mm, dtype=np.float32),
            label_levels=label_levels,
            batch_size=inference_chunk_size,
            show_progress=show_progress,
        )
    stage_done("knn_label_transfer_levels", t0)

    # Final labels/obs tables
    label_codes = np.asarray(label_codes_mm, dtype=np.int32)
    hlca_label = pd.Categorical.from_codes(label_codes, categories=label_categories)
    label_series = pd.Series(hlca_label, index=obs_index, name="hlca_label")

    labels_df = pd.DataFrame(index=obs_index.copy())
    labels_df.index.name = "cell_id"
    labels_df["hlca_label"] = label_series
    if export_probs and max_prob_mm is not None and entropy_mm is not None:
        labels_df["hlca_max_prob"] = np.asarray(max_prob_mm, dtype=np.float32)
        labels_df["hlca_entropy"] = np.asarray(entropy_mm, dtype=np.float32)
        labels_df["hlca_uncertain"] = (
            labels_df["hlca_max_prob"] < float(hlca_cfg.get("uncertainty_threshold", 0.2))
        )
    for col, values in knn_outputs.items():
        labels_df[col] = values

    labels_df.to_parquet(output_labels_parquet_path, index=True, engine="pyarrow")

    latent_obs = pd.DataFrame(index=obs_index.copy())
    latent_obs.index.name = "cell_id"
    latent_obs["donor_id"] = pd.Categorical(donor_values)
    latent_obs["stage"] = pd.Categorical(stage_values)
    latent_obs["gsm_id"] = pd.Categorical(gsm_values)
    latent_obs["sample_id"] = pd.Categorical(sample_values)
    latent_obs["hlca_label"] = pd.Categorical(label_series)
    if export_probs and max_prob_mm is not None and entropy_mm is not None:
        latent_obs["hlca_max_prob"] = np.asarray(max_prob_mm, dtype=np.float32)
        latent_obs["hlca_entropy"] = np.asarray(entropy_mm, dtype=np.float32)
        latent_obs["hlca_uncertain"] = (
            latent_obs["hlca_max_prob"] < float(hlca_cfg.get("uncertainty_threshold", 0.2))
        )
    for col, values in knn_outputs.items():
        latent_obs[col] = values

    latent_var = pd.DataFrame(index=pd.Index([f"latent_{i}" for i in range(latent_dim)], name="latent"))
    latent_adata = anndata.AnnData(X=np.asarray(latent_mm, dtype=np.float32), obs=latent_obs, var=latent_var)
    latent_adata.write_h5ad(output_latent_h5ad_path, compression="lzf")

    progress["status"] = "completed"
    progress["finished_at_unix"] = time.time()
    _save_progress(progress_path, progress)

    if not keep_intermediates:
        try:
            latent_memmap_path.unlink(missing_ok=True)
            label_codes_memmap_path.unlink(missing_ok=True)
            max_prob_memmap_path.unlink(missing_ok=True)
            entropy_memmap_path.unlink(missing_ok=True)
            temp_dir.rmdir()
        except OSError:
            pass

    label_counts = label_series.astype(str).value_counts().head(10)
    top10 = [(str(idx), int(val)) for idx, val in label_counts.items()]

    report = {
        "ok": True,
        "run_id": run_id,
        "inputs": {
            "snrna_h5ad": str(snrna_h5ad_path),
            "source_layer_used": source_layer,
            "n_obs": n_obs,
            "n_vars_input": int(adata_full.n_vars),
            "repo_id": repo_id,
        },
        "outputs": {
            "latent_h5ad": str(output_latent_h5ad_path),
            "labels_parquet": str(output_labels_parquet_path),
            "query_model_dir": str(query_model_dir),
            "progress_json": str(progress_path),
        },
        "wall_time_seconds": {
            "pull_model": float(stage_times.get("pull_model", 0.0)),
            "gene_overlap_check": float(stage_times.get("gene_overlap_check", 0.0)),
            "gene_id_mapping": float(stage_times.get("gene_id_mapping", 0.0)),
            "subset_selection": float(stage_times.get("subset_selection", 0.0)),
            "query_train": float(stage_times.get("query_train", 0.0)),
            "latent_inference_full": float(stage_times.get("latent_inference_full", 0.0)),
            "label_inference_full": float(stage_times.get("label_inference_full", 0.0)),
            "knn_label_transfer_levels": float(stage_times.get("knn_label_transfer_levels", 0.0)),
        },
        "total_wall_time_seconds": time.perf_counter() - wall_t0,
        "peak_rss_mb": peak_rss_mb,
        "used_backed": bool(state["used_backed"]),
        "densified": bool(state["densified"]),
        "hlca_mapping": {
            "overlap_percent": 100.0 * overlap_ratio,
            "direct_overlap_percent": 100.0 * direct_overlap,
            "effective_n_genes_used": n_ref_genes_total,
            "n_reference_genes_present_in_query": n_ref_genes_present,
            "latent_shape": [int(n_obs), int(latent_dim)],
            "batch_size_infer": batch_size_infer,
            "inference_chunk_size": inference_chunk_size,
            "subset_size_used_for_training": int(len(train_idx)),
            "label_top10": top10,
        },
        "knn_label_transfer_levels_enabled": knn_label_transfer_levels,
    }
    mapping_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if hasattr(adata_full, "isbacked") and adata_full.isbacked:
        adata_full.file.close()

    return HLCAMappingResult(
        run_id=run_id,
        latent_h5ad_path=output_latent_h5ad_path,
        labels_parquet_path=output_labels_parquet_path,
        mapping_report_path=mapping_report_path,
        gene_report_path=gene_report_path,
        overlap_percent=100.0 * overlap_ratio,
        latent_shape=(int(n_obs), int(latent_dim)),
        top10_labels=top10,
        peak_rss_mb=peak_rss_mb,
        wall_time_seconds=report["total_wall_time_seconds"],
        report=report,
    )
