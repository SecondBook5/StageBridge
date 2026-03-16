"""
GEO snRNA-seq I/O for StageBridge.

Custom dense-counts format
--------------------------
Files named  *.raw_counts.mtx.txt.gz  are NOT standard Matrix Market.

Format:
  Line 1 : whitespace-delimited cell barcodes (columns of the matrix).
  Line 2+: GENE_SYMBOL  count_cell0  count_cell1  ... count_cellN

This module parses that format into a sparse CSR AnnData WITHOUT ever
materialising the full dense matrix.

Usage (CLI):
    python -m stagebridge.data.luad_evo.snrna convert <input.mtx.txt.gz> <output.h5ad>
    python -m stagebridge.data.luad_evo.snrna manifest <extracted_dir> <manifest.csv>
    python -m stagebridge.data.luad_evo.snrna merge    <manifest.csv>  <merged.h5ad>
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm

from stagebridge.logging_utils import get_logger
from stagebridge.data.common.schema import LatentCohort
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.stages import (
    CANONICAL_STAGE_ORDER,
    normalize_stage_label,
)

log = get_logger(__name__)


def resolve_snrna_latent_path(cfg: Any | None = None) -> Path:
    """Resolve the active LUAD latent path with fallbacks to real local assets."""
    if cfg is not None:
        paths = resolve_luad_evo_paths(cfg)
        candidates = [
            paths.snrna_latent_h5ad,
            paths.snrna_h5ad,
        ]
    else:
        from stagebridge.config import get_data_root

        root = get_data_root()
        candidates = [
            root / "processed" / "anndata" / "snrna_hlca_latent_full.h5ad",
            root / "interim" / "anndata" / "snrna" / "snrna_full.h5ad",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve an snRNA latent file for luad_evo. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def load_luad_evo_snrna_latent(
    cfg: Any | None = None,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_cells_per_stage: int | None = None,
    seed: int = 42,
) -> LatentCohort:
    """Load the active LUAD snRNA latent table used by Mission 3 pipelines."""
    latent_path = resolve_snrna_latent_path(cfg)
    adata = anndata.read_h5ad(latent_path)
    obs = adata.obs.copy()
    if "cell_id" not in obs.columns:
        obs["cell_id"] = adata.obs_names.astype(str)
    if "stage" not in obs.columns:
        raise KeyError(f"Missing 'stage' column in {latent_path}.")

    if "donor_id" not in obs.columns and "patient_id" in obs.columns:
        obs["donor_id"] = obs["patient_id"].astype(str)
    if "patient_id" not in obs.columns and "donor_id" in obs.columns:
        obs["patient_id"] = obs["donor_id"].astype(str)
    if "sample_id" not in obs.columns:
        obs["sample_id"] = obs.index.astype(str)
    obs["stage"] = obs["stage"].astype(str).map(normalize_stage_label)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["patient_id"] = obs["patient_id"].astype(str)

    mask = np.ones(adata.n_obs, dtype=bool)
    if stages:
        wanted = {normalize_stage_label(stage) for stage in stages}
        mask &= obs["stage"].isin(wanted).to_numpy()
    if donors:
        wanted_donors = {str(donor) for donor in donors}
        mask &= obs["donor_id"].isin(wanted_donors).to_numpy()

    if max_cells_per_stage is not None and max_cells_per_stage > 0:
        rng = np.random.default_rng(int(seed))
        chosen = np.zeros(adata.n_obs, dtype=bool)
        masked_positions = np.flatnonzero(mask)
        masked_stages = obs.iloc[masked_positions]["stage"].to_numpy()
        for stage_name in pd.unique(masked_stages):
            stage_rows = masked_positions[masked_stages == stage_name]
            if stage_rows.shape[0] <= max_cells_per_stage:
                chosen[stage_rows] = True
                continue
            keep = rng.choice(stage_rows, size=int(max_cells_per_stage), replace=False)
            chosen[keep] = True
        mask &= chosen

    latent = np.asarray(adata.X[mask], dtype=np.float32)
    obs = obs.loc[mask].reset_index(drop=True)
    feature_names = tuple(
        f"{(getattr(cfg, 'data', {}) or {}).get('latent_key', 'X_hlca')}_{i}"
        for i in range(latent.shape[1])
    )
    return LatentCohort(
        latent=latent,
        obs=obs,
        feature_names=feature_names,
        source_path=latent_path,
        latent_key="X_hlca",
    )


def load_luad_evo_snrna_pca_latent(
    cfg: Any | None = None,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_cells_per_stage: int | None = None,
    n_components: int = 32,
    seed: int = 42,
    log1p_transform: bool = True,
) -> LatentCohort:
    """Fit a lightweight PCA/SVD latent directly from the active snRNA matrix.

    This is the deliberate fallback backend for latent-sensitivity testing.
    It uses the raw snRNA h5ad rather than the precomputed HLCA latent file.
    """
    paths = resolve_luad_evo_paths(cfg or {})
    raw_path = paths.snrna_h5ad
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw snRNA h5ad not found: {raw_path}")

    adata = anndata.read_h5ad(raw_path, backed="r")
    obs = adata.obs.copy()
    if "cell_id" not in obs.columns:
        obs["cell_id"] = adata.obs_names.astype(str)
    if "stage" not in obs.columns:
        raise KeyError(f"Missing 'stage' column in {raw_path}.")
    if "donor_id" not in obs.columns and "patient_id" in obs.columns:
        obs["donor_id"] = obs["patient_id"].astype(str)
    if "patient_id" not in obs.columns and "donor_id" in obs.columns:
        obs["patient_id"] = obs["donor_id"].astype(str)
    if "sample_id" not in obs.columns:
        obs["sample_id"] = obs.index.astype(str)
    obs["stage"] = obs["stage"].astype(str).map(normalize_stage_label)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["patient_id"] = obs["patient_id"].astype(str)

    mask = np.ones(adata.n_obs, dtype=bool)
    if stages:
        wanted = {normalize_stage_label(stage) for stage in stages}
        mask &= obs["stage"].isin(wanted).to_numpy()
    if donors:
        wanted_donors = {str(donor) for donor in donors}
        mask &= obs["donor_id"].isin(wanted_donors).to_numpy()

    if max_cells_per_stage is not None and max_cells_per_stage > 0:
        rng = np.random.default_rng(int(seed))
        chosen = np.zeros(adata.n_obs, dtype=bool)
        masked_positions = np.flatnonzero(mask)
        masked_stages = obs.iloc[masked_positions]["stage"].to_numpy()
        for stage_name in pd.unique(masked_stages):
            stage_rows = masked_positions[masked_stages == stage_name]
            if stage_rows.shape[0] <= max_cells_per_stage:
                chosen[stage_rows] = True
                continue
            keep = rng.choice(stage_rows, size=int(max_cells_per_stage), replace=False)
            chosen[keep] = True
        mask &= chosen

    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError("No snRNA rows remained after PCA latent filtering.")

    matrix = adata.X[rows]
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float32, copy=False)
        if log1p_transform:
            matrix = matrix.copy()
            matrix.data = np.log1p(matrix.data)
        n_eff = max(2, min(int(n_components), int(matrix.shape[0]) - 1, int(matrix.shape[1]) - 1))
        latent = (
            TruncatedSVD(n_components=n_eff, random_state=int(seed))
            .fit_transform(matrix)
            .astype(np.float32)
        )
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
        if log1p_transform:
            matrix = np.log1p(matrix)
        n_eff = max(2, min(int(n_components), int(matrix.shape[0]) - 1, int(matrix.shape[1]) - 1))
        latent = (
            PCA(n_components=n_eff, random_state=int(seed))
            .fit_transform(matrix)
            .astype(np.float32)
        )

    obs = obs.iloc[rows].reset_index(drop=True)
    feature_names = tuple(f"X_pca_{i}" for i in range(latent.shape[1]))
    return LatentCohort(
        latent=latent,
        obs=obs,
        feature_names=feature_names,
        source_path=raw_path,
        latent_key="X_pca",
    )


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Handles stems like:
#   GSM9237901_P3_Normal
#   GSM9237905_P4_Normal1
#   GSM9237910_P7_AAH
#   GSM9237915_P8_MIA
_STEM_RE = re.compile(
    r"^(?P<gsm>GSM\d+)"
    r"_(?P<patient_id>P\d+)"
    r"_(?P<stage_raw>.+?)$"
)


def _normalize_stage(stage_raw: str) -> str:
    """Return canonical lung stage label or ``Unknown`` when not mappable."""
    # Strip trailing digits (Normal1 -> Normal) then normalize via ontology.
    stripped = re.sub(r"\d+$", "", stage_raw).strip("_")
    canonical = normalize_stage_label(stripped)
    if canonical in CANONICAL_STAGE_ORDER:
        return canonical
    return "Unknown"


def parse_sample_info_from_filename(stem: str) -> dict:
    """Parse a sample filename stem into its components.

    Parameters
    ----------
    stem : str
        Filename without extension(s), e.g. ``GSM9237901_P3_Normal``.

    Returns
    -------
    dict with keys: gsm, patient_id, stage_raw, stage_normalized, sample_id
    """
    m = _STEM_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Cannot parse sample info from filename stem: {stem!r}\n"
            f"Expected pattern: GSMxxxxxxx_Pn_Stage[suffix]\n"
            f"Examples: GSM9237901_P3_Normal  GSM9237905_P4_Normal1"
        )
    d = m.groupdict()
    d["stage_normalized"] = _normalize_stage(d["stage_raw"])
    d["sample_id"] = stem
    return d


# ---------------------------------------------------------------------------
# Core conversion: dense-counts gz → sparse AnnData
# ---------------------------------------------------------------------------


def _iter_lines(input_path: Path) -> Iterator[bytes]:
    """Yield raw byte lines from a gzip file."""
    with gzip.open(input_path, "rb") as fh:
        yield from fh


def _sample_stem_from_path(input_path: Path) -> str:
    """Return canonical sample stem from GEO snRNA filename."""
    stem = input_path.name
    for ext in (".gz", ".txt", ".mtx", ".raw_counts"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return stem


def discover_snrna_files(raw_dir: Path) -> pd.DataFrame:
    """Discover and parse snRNA raw files in *raw_dir*.

    Returns
    -------
    pd.DataFrame
        Columns:
        - sample_id
        - input_path
        - gsm_id
        - donor_id
        - stage
        - stage_raw
        - file_size_bytes
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"snRNA extracted directory not found: {raw_dir}\n"
            f"Expected GEO files named *.raw_counts.mtx.txt.gz in this directory."
        )

    files = sorted(raw_dir.glob("*.raw_counts.mtx.txt.gz"))
    if not files:
        raise FileNotFoundError(f"No *.raw_counts.mtx.txt.gz files found in: {raw_dir}")

    rows: list[dict[str, Any]] = []
    for fpath in files:
        stem = _sample_stem_from_path(fpath)
        info = parse_sample_info_from_filename(stem)
        rows.append(
            {
                "sample_id": info["sample_id"],
                "input_path": str(fpath),
                "gsm_id": info["gsm"],
                "donor_id": info["patient_id"],
                "stage": info["stage_normalized"],
                "stage_raw": info["stage_raw"],
                "file_size_bytes": int(fpath.stat().st_size),
            }
        )

    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def apply_snrna_smoke_limits(
    manifest_df: pd.DataFrame,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
) -> pd.DataFrame:
    """Select a deterministic smoke subset from a discovered snRNA manifest."""
    df = manifest_df.copy()
    if df.empty:
        return df

    if max_donors is not None and max_donors > 0:
        donor_order = (
            df.groupby("donor_id", as_index=True)["file_size_bytes"]
            .sum()
            .sort_values(kind="stable")
            .index.tolist()
        )
        keep_donors = set(donor_order[:max_donors])
        df = df[df["donor_id"].isin(keep_donors)].copy()

    if max_samples_per_stage is not None and max_samples_per_stage > 0:
        df = (
            df.sort_values(["stage", "file_size_bytes", "sample_id"], kind="stable")
            .groupby("stage", as_index=False, group_keys=False)
            .head(max_samples_per_stage)
            .copy()
        )

    return df.sort_values("sample_id", kind="stable").reset_index(drop=True)


def load_snrna_sample(
    input_path: Path,
    max_cells_per_sample: int | None = None,
) -> anndata.AnnData:
    """Load one GEO snRNA dense-counts sample into AnnData.

    Input format by inspection:
    - line 1: barcode header (whitespace-delimited columns)
    - line 2+: GENE_SYMBOL + dense integer counts across all barcodes
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"snRNA input file not found: {input_path}")

    sample_stem = _sample_stem_from_path(input_path)
    info = parse_sample_info_from_filename(sample_stem)

    with gzip.open(input_path, "rt", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        if not header:
            raise ValueError(f"Missing barcode header line in {input_path}")

        n_total_cells = len(header)
        if max_cells_per_sample is not None and max_cells_per_sample > 0:
            n_keep = min(int(max_cells_per_sample), n_total_cells)
        else:
            n_keep = n_total_cells

        selected_idx = np.arange(n_keep, dtype=np.int64)
        selected_barcodes = [header[i] for i in selected_idx.tolist()]

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        data_chunks: list[np.ndarray] = []
        var_names: list[str] = []

        gene_idx = 0
        for line in fh:
            s = line.rstrip("\n")
            if not s:
                continue

            parts = s.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed gene row in {input_path} at gene index {gene_idx}: {s[:120]!r}"
                )
            gene_symbol, count_blob = parts
            counts_full = np.fromstring(count_blob, sep=" ", dtype=np.int32)
            if counts_full.size != n_total_cells:
                raise ValueError(
                    f"Gene '{gene_symbol}' in {input_path} has {counts_full.size} counts, "
                    f"expected {n_total_cells} from header."
                )

            counts = counts_full[selected_idx]
            var_names.append(gene_symbol)

            nz = np.flatnonzero(counts)
            if nz.size:
                row_chunks.append(np.full(nz.size, gene_idx, dtype=np.int32))
                col_chunks.append(nz.astype(np.int32, copy=False))
                data_chunks.append(counts[nz].astype(np.float32, copy=False))
            gene_idx += 1

    n_genes = len(var_names)
    if n_genes == 0:
        raise ValueError(f"No gene rows parsed from {input_path}")

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        vals = np.concatenate(data_chunks)
    else:
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)
        vals = np.array([], dtype=np.float32)

    X_genes_cells = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_genes, n_keep),
        dtype=np.float32,
    )
    X = X_genes_cells.T.tocsr()

    obs_index = pd.Index(
        [f"{info['sample_id']}:{bc}" for bc in selected_barcodes],
        name="cell_id",
    )
    obs = pd.DataFrame(index=obs_index)
    obs["barcode"] = selected_barcodes
    obs["donor_id"] = info["patient_id"]
    obs["patient_id"] = info["patient_id"]
    obs["stage"] = info["stage_normalized"]
    obs["stage_raw"] = info["stage_raw"]
    obs["gsm_id"] = info["gsm"]
    obs["gsm"] = info["gsm"]
    obs["sample_id"] = info["sample_id"]
    obs["modality"] = "snrna"

    var = pd.DataFrame(index=pd.Index(var_names, name="gene"))
    if var.index.to_series().str.fullmatch(r"\d+").all():
        var["gene_id_raw"] = var.index.astype(str)

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    return adata


def load_snrna_dataset(
    raw_dir: Path,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
    max_cells_per_sample: int | None = None,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Load and concatenate GEO snRNA samples from *raw_dir*."""
    manifest = discover_snrna_files(raw_dir)
    selected = apply_snrna_smoke_limits(
        manifest,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
    )
    if selected.empty:
        raise RuntimeError("No snRNA samples selected after applying limits.")

    adatas: list[anndata.AnnData] = []
    for row in selected.itertuples(index=False):
        ad = load_snrna_sample(
            Path(row.input_path),
            max_cells_per_sample=max_cells_per_sample,
        )
        adatas.append(ad)

    merged = anndata.concat(adatas, join="outer", merge="same")
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    merged.layers["counts"] = merged.X.copy()
    return merged, selected


def convert_snrna_dense_counts_to_h5ad(
    input_path: Path,
    output_path: Path,
    genes_per_chunk: int = 256,
) -> None:
    """Parse a custom dense-counts .mtx.txt.gz and write an h5ad file.

    The format (NOT standard MTX):
      - Line 1  : whitespace-delimited barcodes (one per cell / column).
      - Line 2+ : GENE_SYMBOL  count0  count1  … countN-1

    Parameters
    ----------
    input_path : Path
        Path to the ``*.raw_counts.mtx.txt.gz`` file.
    output_path : Path
        Destination ``.h5ad`` file (parent dirs created automatically).
    genes_per_chunk : int
        Number of gene rows to accumulate per COO chunk before appending
        to the running sparse-triplet lists.  Tune to fit RAM.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\nExpected a gzip-compressed dense-counts matrix."
        )

    log.info("Converting: %s → %s", input_path, output_path)

    # --- Parse sample metadata from filename ---
    stem = input_path.name
    # Strip extensions: .raw_counts.mtx.txt.gz
    for ext in (".gz", ".txt", ".mtx", ".raw_counts"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    sample_info = parse_sample_info_from_filename(stem)
    log.info("Sample info: %s", sample_info)

    line_iter = _iter_lines(input_path)

    # --- Line 1: barcodes ---
    header_line = next(line_iter).decode("utf-8").rstrip("\n")
    obs_names = header_line.split()
    n_cells = len(obs_names)
    if n_cells == 0:
        raise ValueError(
            f"First line of {input_path} is empty — expected whitespace-separated barcodes."
        )
    log.info("n_cells (barcodes): %d", n_cells)

    # --- Lines 2+: gene rows ---
    # Build sparse matrix using lil_matrix for row-by-row insertion
    # Switch to COO accumulation for memory efficiency.
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    data_vals: list[np.ndarray] = []
    var_names: list[str] = []

    gene_idx = 0
    chunk_rows: list[int] = []
    chunk_cols: list[int] = []
    chunk_data: list[int] = []

    def _flush_chunk() -> None:
        if chunk_rows:
            row_indices.append(np.array(chunk_rows, dtype=np.int32))
            col_indices.append(np.array(chunk_cols, dtype=np.int32))
            data_vals.append(np.array(chunk_data, dtype=np.float32))
            chunk_rows.clear()
            chunk_cols.clear()
            chunk_data.clear()

    for raw_line in tqdm(line_iter, desc="Parsing gene rows", unit="genes"):
        line = raw_line.decode("utf-8").rstrip("\n")
        if not line:
            continue
        tokens = line.split()
        gene = tokens[0]
        counts_str = tokens[1:]
        if len(counts_str) != n_cells:
            raise ValueError(
                f"Gene '{gene}' at row {gene_idx + 1} has {len(counts_str)} values "
                f"but expected {n_cells} (= number of barcodes in header).\n"
                f"File: {input_path}"
            )
        var_names.append(gene)
        counts = np.fromiter((int(c) for c in counts_str), dtype=np.int32, count=n_cells)
        nz_mask = counts != 0
        nz_cols = np.where(nz_mask)[0]
        for c in nz_cols:
            chunk_rows.append(gene_idx)
            chunk_cols.append(int(c))
            chunk_data.append(int(counts[c]))

        gene_idx += 1
        if gene_idx % genes_per_chunk == 0:
            _flush_chunk()

    _flush_chunk()

    n_genes = gene_idx
    log.info("n_genes: %d", n_genes)

    if n_genes == 0:
        raise ValueError(f"No gene rows found in {input_path}.  File may be malformed.")

    # Build CSR (genes x cells first, then transpose to cells x genes)
    log.info("Building sparse CSR matrix (cells=%d, genes=%d)...", n_cells, n_genes)
    all_rows = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int32)
    all_cols = np.concatenate(col_indices) if col_indices else np.array([], dtype=np.int32)
    all_data = np.concatenate(data_vals) if data_vals else np.array([], dtype=np.float32)

    # genes × cells
    X_genes_cells = sp.csr_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(n_genes, n_cells),
        dtype=np.float32,
    )
    # Transpose → cells × genes  (standard AnnData layout)
    X = X_genes_cells.T.tocsr()

    nnz = X.nnz
    log.info("nnz: %d  (density=%.4f%%)", nnz, 100 * nnz / (n_cells * n_genes))

    # --- Build AnnData ---
    obs = pd.DataFrame(
        {
            "gsm": sample_info["gsm"],
            "patient_id": sample_info["patient_id"],
            "stage_raw": sample_info["stage_raw"],
            "stage": sample_info["stage_normalized"],
            "sample_id": sample_info["sample_id"],
        },
        index=obs_names,
    )
    var = pd.DataFrame(index=var_names)
    var.index.name = "gene_symbols"

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing h5ad: %s", output_path)
    adata.write_h5ad(output_path)

    print(
        f"\n{'=' * 60}\n"
        f"  Sample : {sample_info['sample_id']}\n"
        f"  n_cells: {n_cells}\n"
        f"  n_genes: {n_genes}\n"
        f"  nnz    : {nnz}\n"
        f"  Output : {output_path}\n"
        f"{'=' * 60}\n"
    )


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_snrna_manifest(raw_dir: Path, output_csv: Path) -> None:
    """Scan *raw_dir* for ``*.raw_counts.mtx.txt.gz`` and write a manifest CSV.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the extracted GEO snRNA files.
    output_csv : Path
        Destination CSV path (parent dirs created if needed).

    CSV columns: sample_id, input_path, gsm, patient_id, stage
    """
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"snRNA extracted directory not found: {raw_dir}\n"
            f"Expected the GEO GSE308103 extracted files at this location."
        )

    files = sorted(raw_dir.glob("*.raw_counts.mtx.txt.gz"))
    if not files:
        raise FileNotFoundError(f"No *.raw_counts.mtx.txt.gz files found in: {raw_dir}")

    rows = []
    for fpath in files:
        stem = fpath.name
        for ext in (".gz", ".txt", ".mtx", ".raw_counts"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        try:
            info = parse_sample_info_from_filename(stem)
        except ValueError as exc:
            log.warning("Skipping unparseable filename %s: %s", fpath.name, exc)
            continue
        rows.append(
            {
                "sample_id": info["sample_id"],
                "input_path": str(fpath),
                "gsm": info["gsm"],
                "patient_id": info["patient_id"],
                "stage": info["stage_normalized"],
            }
        )

    if not rows:
        raise RuntimeError(
            f"No parseable snRNA files found in {raw_dir}.\n"
            f"Check filename convention: GSMxxxxxxx_Pn_Stage.raw_counts.mtx.txt.gz"
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Manifest written (%d samples): %s", len(df), output_csv)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_snrna_h5ad(manifest_csv: Path, output_h5ad: Path) -> None:
    """Concatenate per-sample h5ad files listed in *manifest_csv*.

    Each sample's h5ad must already exist (run convert_snrna_dense_counts_to_h5ad
    for each row first).

    Parameters
    ----------
    manifest_csv : Path
        CSV produced by :func:`build_snrna_manifest`.
    output_h5ad : Path
        Destination merged h5ad.
    """
    manifest_csv = Path(manifest_csv)
    output_h5ad = Path(output_h5ad)

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest CSV not found: {manifest_csv}\nRun build_snrna_manifest() first."
        )

    df = pd.read_csv(manifest_csv)
    required_cols = {"sample_id", "input_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest CSV missing columns {missing}.\n"
            f"Expected columns: sample_id, input_path, gsm, patient_id, stage"
        )

    from stagebridge.config import interim_snrna_dir

    adatas = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        h5ad_path = interim_snrna_dir() / f"{sample_id}.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(
                f"Per-sample h5ad not found: {h5ad_path}\n"
                f"Convert sample '{sample_id}' first:\n"
                f"  python -m stagebridge.data.luad_evo.snrna convert "
                f"  {row['input_path']}  {h5ad_path}"
            )
        log.info("Loading %s", h5ad_path)
        adata = anndata.read_h5ad(h5ad_path)
        adatas.append(adata)

    if not adatas:
        raise RuntimeError("No AnnData objects loaded — manifest may be empty.")

    log.info("Concatenating %d samples...", len(adatas))
    merged = anndata.concat(adatas, join="outer", label="sample_id", merge="same")
    merged.obs_names_make_unique()

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing merged h5ad (%d cells, %d genes): %s", *merged.shape, output_h5ad)
    merged.write_h5ad(output_h5ad)
    print(f"Merged snRNA: {merged.shape[0]} cells × {merged.shape[1]} genes → {output_h5ad}")


# ---------------------------------------------------------------------------
# CLI __main__
# ---------------------------------------------------------------------------


def _usage() -> None:
    print(
        "Usage:\n"
        "  python -m stagebridge.data.luad_evo.snrna convert  <input.raw_counts.mtx.txt.gz> <output.h5ad>\n"
        "  python -m stagebridge.data.luad_evo.snrna manifest <extracted_dir>               <manifest.csv>\n"
        "  python -m stagebridge.data.luad_evo.snrna merge    <manifest.csv>                <merged.h5ad>\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "convert":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        convert_snrna_dense_counts_to_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "manifest":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        build_snrna_manifest(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "merge":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        merge_snrna_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(f"Unknown command: {cmd!r}")
        _usage()
        sys.exit(1)
