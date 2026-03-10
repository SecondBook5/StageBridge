"""
GEO spatial (10x Visium) I/O for StageBridge.

Primary loader: ``squidpy.read.visium()``
-----------------------------------------
squidpy natively handles the 10x Visium directory layout:
  filtered_feature_bc_matrix/  matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
  spatial/                     tissue_positions*.csv, tissue_lowres_image.png,
                               scalefactors_json.json

If squidpy is not available (e.g. during a minimal install), a manual fallback
parser is used instead.  Both paths produce an AnnData with:
  - X               : CSR float32 counts (cells × genes)
  - layers["counts"]: copy of raw X
  - obsm["spatial"] : (n_spots, 2) pixel coordinates if available

Responsibilities
----------------
1. Expand per-sample .tar.gz archives into samples/<SAMPLE_ID>/ directories.
2. Load each sample as AnnData via squidpy (or manual fallback).
3. Write per-sample .h5ad files and build a manifest CSV.
4. Concatenate into a merged .h5ad.

Usage (CLI):
    python -m stagebridge.data.luad_evo.visium expand   <extracted_dir> <samples_dir>
    python -m stagebridge.data.luad_evo.visium load     <sample_dir>    <output.h5ad>
    python -m stagebridge.data.luad_evo.visium manifest <samples_dir>   <manifest.csv>
    python -m stagebridge.data.luad_evo.visium merge    <manifest.csv>  <merged.h5ad>
"""
from __future__ import annotations

import gzip
import io
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import io as sio

from stagebridge.logging_utils import get_logger
from stagebridge.data.common.schema import SpatialCohort
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths
from stagebridge.data.luad_evo.stages import (
    CANONICAL_STAGE_ORDER,
    normalize_stage_label,
)

log = get_logger(__name__)


def resolve_spatial_tangram_path(cfg: Any | None = None) -> Path:
    """Resolve the active Tangram spatial output for LUAD evolution."""
    if cfg is not None:
        paths = resolve_luad_evo_paths(cfg)
        candidates = [paths.spatial_tangram_h5ad, paths.spatial_h5ad]
    else:
        from stagebridge.config import get_data_root
        root = get_data_root()
        candidates = [
            root / "processed" / "tangram" / "spatial_tangram_full.h5ad",
            root / "interim" / "anndata" / "spatial" / "spatial_full.h5ad",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve a spatial LUAD file. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def load_luad_evo_spatial_mapping(
    cfg: Any | None = None,
    *,
    mapping_h5ad_path: Path | None = None,
    composition_key: str = "X_tangram_ct",
    columns_key: str = "tangram_ct_columns",
    stages: list[str] | None = None,
    donors: list[str] | None = None,
    max_spots_per_stage: int | None = None,
    seed: int = 42,
) -> SpatialCohort:
    """Load a LUAD spatial provider output with standardized filtering."""
    spatial_path = Path(mapping_h5ad_path) if mapping_h5ad_path is not None else resolve_spatial_tangram_path(cfg)
    adata = anndata.read_h5ad(spatial_path)
    if composition_key not in adata.obsm:
        raise KeyError(
            f"Expected '{composition_key}' in {spatial_path}. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    if "spatial" not in adata.obsm:
        raise KeyError(f"Expected 'spatial' coordinates in {spatial_path}.")

    obs = adata.obs.copy()
    if "patient_id" not in obs.columns and "donor_id" in obs.columns:
        obs["patient_id"] = obs["donor_id"].astype(str)
    if "donor_id" not in obs.columns and "patient_id" in obs.columns:
        obs["donor_id"] = obs["patient_id"].astype(str)
    obs["stage"] = obs["stage"].astype(str).map(normalize_stage_label)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["patient_id"] = obs["patient_id"].astype(str)
    if "sample_id" not in obs.columns:
        obs["sample_id"] = obs.index.astype(str)

    mask = np.ones(adata.n_obs, dtype=bool)
    if stages:
        wanted = {normalize_stage_label(stage) for stage in stages}
        mask &= obs["stage"].isin(wanted).to_numpy()
    if donors:
        wanted_donors = {str(donor) for donor in donors}
        mask &= obs["donor_id"].isin(wanted_donors).to_numpy()

    if max_spots_per_stage is not None and max_spots_per_stage > 0:
        rng = np.random.default_rng(int(seed))
        chosen = np.zeros(adata.n_obs, dtype=bool)
        masked_positions = np.flatnonzero(mask)
        masked_stages = obs.iloc[masked_positions]["stage"].to_numpy()
        for stage_name in pd.unique(masked_stages):
            rows = masked_positions[masked_stages == stage_name]
            if rows.shape[0] <= max_spots_per_stage:
                chosen[rows] = True
                continue
            keep = rng.choice(rows, size=int(max_spots_per_stage), replace=False)
            chosen[keep] = True
        mask &= chosen

    feature_names = tuple(str(name) for name in adata.uns.get(columns_key, adata.obsm[composition_key].dtype.names or []))
    if not feature_names:
        feature_names = tuple(str(name) for name in getattr(adata, "var_names", [])[: adata.obsm[composition_key].shape[1]])
    if not feature_names or len(feature_names) != adata.obsm[composition_key].shape[1]:
        feature_names = tuple(f"ct_{i}" for i in range(adata.obsm[composition_key].shape[1]))

    return SpatialCohort(
        compositions=np.asarray(adata.obsm[composition_key][mask], dtype=np.float32),
        coords=np.asarray(adata.obsm["spatial"][mask], dtype=np.float32),
        obs=obs.loc[mask].reset_index(drop=True),
        feature_names=feature_names,
        source_path=spatial_path,
    )

# ---------------------------------------------------------------------------
# squidpy import (optional — graceful fallback)
# ---------------------------------------------------------------------------
try:
    import squidpy as sq
    _SQUIDPY_AVAILABLE = True
    log.info("squidpy %s available — using sq.read.visium() as primary loader.", sq.__version__)
except ImportError:
    _SQUIDPY_AVAILABLE = False
    log.warning(
        "squidpy not installed — falling back to manual Visium parser.\n"
        "Install with:  pip install squidpy>=1.4"
    )

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_STEM_RE = re.compile(r"^(?P<gsm>GSM\d+)_(?P<patient_id>P\d+)_(?P<stage_raw>.+?)$")

def _normalize_stage(stage_raw: str) -> str:
    """Return canonical lung stage label or ``Unknown`` when not mappable."""
    stripped = re.sub(r"\d+$", "", stage_raw).strip("_")
    canonical = normalize_stage_label(stripped)
    if canonical in CANONICAL_STAGE_ORDER:
        return canonical
    return "Unknown"


def _parse_stem(stem: str) -> dict:
    m = _STEM_RE.match(stem)
    if m is None:
        raise ValueError(
            f"Cannot parse spatial sample stem: {stem!r}\n"
            f"Expected: GSMxxxxxxx_Pn_Stage  (e.g. GSM9234567_P1_Normal)"
        )
    d = m.groupdict()
    d["stage_normalized"] = _normalize_stage(d["stage_raw"])
    d["sample_id"] = stem
    return d


def _attach_sample_obs(adata: anndata.AnnData, stem: str) -> None:
    """Add gsm / patient_id / stage / sample_id columns to obs in-place."""
    try:
        info = _parse_stem(stem)
        adata.obs["gsm"]        = info["gsm"]
        adata.obs["patient_id"] = info["patient_id"]
        adata.obs["stage_raw"]  = info["stage_raw"]
        adata.obs["stage"]      = info["stage_normalized"]
        adata.obs["sample_id"]  = info["sample_id"]
    except ValueError as exc:
        log.warning("Could not parse sample info from stem %r: %s", stem, exc)


# ---------------------------------------------------------------------------
# Tarball expansion
# ---------------------------------------------------------------------------

def expand_spatial_tarballs(extracted_dir: Path, samples_dir: Path) -> None:
    """Extract each GSM*.tar.gz in *extracted_dir* into *samples_dir*/<SAMPLE_ID>/.

    Idempotent: skips samples whose destination directory already exists and
    is non-empty.

    Parameters
    ----------
    extracted_dir : Path
        Directory containing the downloaded ``*.tar.gz`` files.
    samples_dir : Path
        Destination root; one sub-directory is created per sample.
    """
    extracted_dir = Path(extracted_dir)
    samples_dir   = Path(samples_dir)

    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Spatial extracted dir not found: {extracted_dir}\n"
            f"Expected the GEO GSE307534 downloaded tarballs at this path."
        )

    tarballs = sorted(extracted_dir.glob("GSM*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(
            f"No GSM*.tar.gz files found in: {extracted_dir}"
        )

    log.info("Found %d tarballs in %s", len(tarballs), extracted_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    for tb in tarballs:
        stem = tb.name
        for ext in (".tar.gz", ".tgz"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        dest = samples_dir / stem
        if dest.exists() and any(dest.iterdir()):
            log.info("Already extracted (skipping): %s", dest)
            continue

        dest.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s → %s", tb.name, dest)
        try:
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(path=dest)
        except tarfile.TarError as exc:
            raise RuntimeError(f"Failed to extract {tb}:\n  {exc}") from exc
        log.info("Done: %s", tb.name)


# ---------------------------------------------------------------------------
# squidpy-based loader (primary)
# ---------------------------------------------------------------------------

def _find_matrix_dir_name(sample_dir: Path) -> str:
    """Return the relative name of the matrix sub-directory."""
    return _find_matrix_dir(sample_dir).name


def _load_with_squidpy(sample_dir: Path) -> anndata.AnnData:
    """Load a Visium sample using ``squidpy.read.visium()``.

    squidpy handles:
    - filtered_feature_bc_matrix/ MTX layout
    - spatial/tissue_positions*.csv  (pixel coordinates)
    - spatial/scalefactors_json.json
    - spatial/tissue_lowres_image.png (stored in adata.uns["spatial"])

    Returns AnnData with X as CSR float32, layers["counts"], and obsm["spatial"].
    """
    matrix_subdir = _find_matrix_dir_name(sample_dir)
    log.info("Loading with squidpy: %s  (matrix dir: %s)", sample_dir.name, matrix_subdir)

    adata = sq.read.visium(
        path=sample_dir,
        counts_file=f"{matrix_subdir}/matrix.mtx.gz",
        library_id=sample_dir.name,
    )

    if sp.issparse(adata.X):
        adata.X = adata.X.astype(np.float32).tocsr()
    else:
        adata.X = sp.csr_matrix(np.array(adata.X, dtype=np.float32))

    adata.layers["counts"] = adata.X.copy()

    if "spatial" in adata.obsm:
        log.info("Spatial coords from squidpy: shape=%s", adata.obsm["spatial"].shape)
    else:
        log.warning("squidpy did not attach spatial coords for %s", sample_dir.name)

    return adata


# ---------------------------------------------------------------------------
# Manual fallback loader (used when squidpy is absent)
# ---------------------------------------------------------------------------

def _find_matrix_dir(sample_dir: Path) -> Path:
    for name in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
        d = sample_dir / name
        if d.is_dir():
            return d
        for sub in sample_dir.iterdir():
            if sub.is_dir():
                d2 = sub / name
                if d2.is_dir():
                    return d2
    raise FileNotFoundError(
        f"No feature-barcode matrix directory found in: {sample_dir}\n"
        f"Contents: {[p.name for p in sample_dir.iterdir()]}"
    )


def _read_tsv_gz_col(path: Path, col: int = 0) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [line.rstrip("\n").split("\t")[col] for line in fh if line.strip()]


def _read_features(matrix_dir: Path) -> list[str]:
    for name in ("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"):
        p = matrix_dir / name
        if p.exists():
            # features.tsv has 3 cols: Ensembl_id, gene_symbol, feature_type
            col = 1 if name.startswith("features") else 0
            return _read_tsv_gz_col(p, col=col)
    raise FileNotFoundError(
        f"No features/genes TSV found in: {matrix_dir}\n"
        f"Expected features.tsv.gz, features.tsv, genes.tsv.gz, or genes.tsv"
    )


def _read_mtx_sparse(path: Path) -> tuple[sp.csr_matrix, int, int]:
    """Read an MTX file; return (genes x cells CSR, n_genes, n_cells)."""
    from scipy.io import mmread

    mat = mmread(str(path)).astype(np.float32).tocsr()
    return mat, mat.shape[0], mat.shape[1]


def _load_spatial_coords_manual(sample_dir: Path, barcodes: list[str]) -> np.ndarray | None:
    """Read tissue_positions CSV and return pixel coords aligned to *barcodes*."""
    spatial_dir = sample_dir / "spatial"
    if not spatial_dir.is_dir():
        for sub in sample_dir.iterdir():
            if sub.is_dir() and (sub / "spatial").is_dir():
                spatial_dir = sub / "spatial"
                break
        else:
            return None

    for fname in (
        "tissue_positions.csv",
        "tissue_positions_list.csv",
        "tissue_positions.csv.gz",
    ):
        tp = spatial_dir / fname
        if not tp.exists():
            continue

        opener = gzip.open if tp.suffix == ".gz" else open
        with opener(tp, "rt") as fh:
            first = fh.readline()

        # Detect header: if first token is a barcode-like string (alphanumeric/dash)
        # it may or may not have a header row depending on Visium software version.
        col_names = ["barcode", "in_tissue", "array_row", "array_col",
                     "pxl_row_in_fullres", "pxl_col_in_fullres"]
        first_col = first.split(",")[0].strip()
        has_header = not first_col.replace("-", "").replace("_", "").isdigit()

        df = pd.read_csv(
            tp,
            header=0 if has_header else None,
            names=None if has_header else col_names,
            index_col=0,
        )

        if "pxl_row_in_fullres" in df.columns:
            coord_cols = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
        elif "array_row" in df.columns:
            coord_cols = ["array_row", "array_col"]
        else:
            log.warning("tissue_positions has unrecognised columns: %s", list(df.columns))
            return None

        df_aligned = df.reindex(barcodes)[coord_cols]
        if df_aligned.isna().all(axis=None):
            log.warning(
                "Barcodes in tissue_positions do not match matrix barcodes — "
                "spatial coords will not be set."
            )
            return None
        return df_aligned.values.astype(float)

    return None


def _load_with_manual_fallback(sample_dir: Path) -> anndata.AnnData:
    """Manual Visium loader — used when squidpy is not installed."""
    log.info("Loading with manual parser: %s", sample_dir.name)
    matrix_dir = _find_matrix_dir(sample_dir)

    bc_path = next(
        (matrix_dir / n for n in ("barcodes.tsv.gz", "barcodes.tsv")
         if (matrix_dir / n).exists()),
        None,
    )
    if bc_path is None:
        raise FileNotFoundError(f"barcodes.tsv[.gz] not found in: {matrix_dir}")
    barcodes = _read_tsv_gz_col(bc_path, col=0)
    gene_names = _read_features(matrix_dir)

    mtx_path = next(
        (matrix_dir / n for n in ("matrix.mtx.gz", "matrix.mtx")
         if (matrix_dir / n).exists()),
        None,
    )
    if mtx_path is None:
        raise FileNotFoundError(f"matrix.mtx[.gz] not found in: {matrix_dir}")

    X_genes_cells, n_genes, n_cells = _read_mtx_sparse(mtx_path)
    if len(barcodes) != n_cells or len(gene_names) != n_genes:
        raise ValueError(
            f"MTX header says ({n_genes} genes, {n_cells} cells) but "
            f"barcodes file has {len(barcodes)} entries and "
            f"features file has {len(gene_names)} entries."
        )

    X = X_genes_cells.T.tocsr()  # cells × genes
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var.index.name = "gene_symbols"
    adata.layers["counts"] = X.copy()

    coords = _load_spatial_coords_manual(sample_dir, barcodes)
    if coords is not None:
        adata.obsm["spatial"] = coords
        log.info("Spatial coords attached (manual): %s", coords.shape)
    else:
        log.warning(
            "No spatial coordinates found in %s — obsm['spatial'] not set.",
            sample_dir.name,
        )
    return adata


# ---------------------------------------------------------------------------
# Public: load one sample
# ---------------------------------------------------------------------------

def load_visium_sample(sample_dir: Path) -> anndata.AnnData:
    """Load a 10x Visium sample directory into AnnData.

    Uses ``squidpy.read.visium()`` if squidpy is installed; otherwise falls
    back to the manual parser.  Both produce equivalent AnnData objects.

    Parameters
    ----------
    sample_dir : Path
        Root of the extracted sample directory.

    Returns
    -------
    AnnData
        spots × genes, with obsm["spatial"] if coordinates were found.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    if _SQUIDPY_AVAILABLE:
        try:
            adata = _load_with_squidpy(sample_dir)
        except Exception as exc:
            log.warning(
                "squidpy.read.visium() failed for %s (%s) — "
                "falling back to manual parser.",
                sample_dir.name, exc,
            )
            adata = _load_with_manual_fallback(sample_dir)
    else:
        adata = _load_with_manual_fallback(sample_dir)

    _attach_sample_obs(adata, sample_dir.name)
    log.info(
        "Loaded %s: %d spots × %d genes  | spatial=%s  | loader=%s",
        sample_dir.name,
        adata.n_obs,
        adata.n_vars,
        "yes" if "spatial" in adata.obsm else "no",
        "squidpy" if _SQUIDPY_AVAILABLE else "manual",
    )
    return adata


# ---------------------------------------------------------------------------
# Write single sample h5ad
# ---------------------------------------------------------------------------

def write_spatial_h5ad(sample_dir: Path, output_path: Path) -> None:
    """Load *sample_dir* and write to *output_path* as h5ad."""
    adata = load_visium_sample(Path(sample_dir))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(
        f"\n{'='*60}\n"
        f"  Sample     : {Path(sample_dir).name}\n"
        f"  n_spots    : {adata.n_obs}\n"
        f"  n_genes    : {adata.n_vars}\n"
        f"  has_spatial: {'spatial' in adata.obsm}\n"
        f"  loader     : {'squidpy' if _SQUIDPY_AVAILABLE else 'manual'}\n"
        f"  Output     : {output_path}\n"
        f"{'='*60}\n"
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_spatial_manifest(samples_dir: Path, output_csv: Path) -> None:
    """Write a manifest CSV for extracted spatial samples.

    CSV columns: sample_id, sample_dir, gsm, patient_id, stage
    """
    samples_dir = Path(samples_dir)
    output_csv  = Path(output_csv)

    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"Spatial samples directory not found: {samples_dir}\n"
            f"Run expand_spatial_tarballs() first."
        )

    sample_dirs = sorted(
        p for p in samples_dir.iterdir()
        if p.is_dir() and p.name.startswith("GSM")
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No GSM* sub-directories found in: {samples_dir}")

    rows = []
    for sd in sample_dirs:
        try:
            info = _parse_stem(sd.name)
        except ValueError as exc:
            log.warning("Skipping %s: %s", sd.name, exc)
            continue
        rows.append({
            "sample_id":  info["sample_id"],
            "sample_dir": str(sd),
            "gsm":        info["gsm"],
            "patient_id": info["patient_id"],
            "stage":      info["stage_normalized"],
        })

    if not rows:
        raise RuntimeError(f"No parseable sample directories in {samples_dir}.")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log.info("Spatial manifest written (%d samples): %s", len(df), output_csv)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_spatial_h5ad(manifest_csv: Path, output_h5ad: Path) -> None:
    """Concatenate per-sample spatial h5ad files listed in *manifest_csv*."""
    manifest_csv = Path(manifest_csv)
    output_h5ad  = Path(output_h5ad)

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Spatial manifest CSV not found: {manifest_csv}\n"
            f"Run build_spatial_manifest() first."
        )

    from stagebridge.config import interim_spatial_dir

    df = pd.read_csv(manifest_csv)
    adatas = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        h5ad_path = interim_spatial_dir() / f"{sample_id}.h5ad"
        if not h5ad_path.exists():
            raise FileNotFoundError(
                f"Per-sample spatial h5ad not found: {h5ad_path}\n"
                f"Run write_spatial_h5ad() for sample '{sample_id}' first."
            )
        log.info("Loading %s", h5ad_path)
        ad = anndata.read_h5ad(h5ad_path)
        # Some Visium feature tables contain repeated gene symbols; concat requires unique indices.
        if not ad.var_names.is_unique:
            ad.var_names_make_unique()
        adatas.append(ad)

    log.info("Concatenating %d spatial samples...", len(adatas))
    merged = anndata.concat(adatas, join="outer", label="sample_id", merge="same")
    merged.obs_names_make_unique()

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Writing merged spatial h5ad (%d spots, %d genes): %s",
        *merged.shape, output_h5ad,
    )
    merged.write_h5ad(output_h5ad)
    print(
        f"Merged spatial: {merged.shape[0]} spots × {merged.shape[1]} genes "
        f"→ {output_h5ad}"
    )


def _sample_stem_from_tar_path(input_path: Path) -> str:
    """Return canonical stem from a GSM spatial tarball path."""
    stem = input_path.name
    for ext in (".tar.gz", ".tgz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def discover_spatial_tarballs(extracted_dir: Path) -> pd.DataFrame:
    """Discover and parse spatial GEO tarballs in *extracted_dir*."""
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Spatial extracted directory not found: {extracted_dir}")

    tarballs = sorted(extracted_dir.glob("GSM*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(f"No GSM*.tar.gz files found in: {extracted_dir}")

    rows: list[dict[str, Any]] = []
    for tb in tarballs:
        stem = _sample_stem_from_tar_path(tb)
        info = _parse_stem(stem)
        rows.append(
            {
                "sample_id": info["sample_id"],
                "input_path": str(tb),
                "gsm_id": info["gsm"],
                "donor_id": info["patient_id"],
                "stage": info["stage_normalized"],
                "stage_raw": info["stage_raw"],
                "file_size_bytes": int(tb.stat().st_size),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def apply_spatial_smoke_limits(
    manifest_df: pd.DataFrame,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
) -> pd.DataFrame:
    """Select a deterministic smoke subset from discovered spatial tarballs."""
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
        keep_donors = set(donor_order[: max_donors])
        df = df[df["donor_id"].isin(keep_donors)].copy()

    if max_samples_per_stage is not None and max_samples_per_stage > 0:
        df = (
            df.sort_values(["stage", "file_size_bytes", "sample_id"], kind="stable")
            .groupby("stage", as_index=False, group_keys=False)
            .head(max_samples_per_stage)
            .copy()
        )

    return df.sort_values("sample_id", kind="stable").reset_index(drop=True)


def inspect_spatial_tarball_format(tar_path: Path) -> dict[str, Any]:
    """Inspect a spatial tarball and report detected format details."""
    tar_path = Path(tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Spatial tarball not found: {tar_path}")

    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m.name for m in tf.getmembers() if m.isfile()]

    has_matrix = any(
        m.endswith("filtered_feature_bc_matrix/matrix.mtx.gz")
        or m.endswith("filtered_feature_bc_matrix/matrix.mtx")
        or m.endswith("raw_feature_bc_matrix/matrix.mtx.gz")
        or m.endswith("raw_feature_bc_matrix/matrix.mtx")
        for m in members
    )
    has_barcodes = any(
        m.endswith("filtered_feature_bc_matrix/barcodes.tsv.gz")
        or m.endswith("filtered_feature_bc_matrix/barcodes.tsv")
        or m.endswith("raw_feature_bc_matrix/barcodes.tsv.gz")
        or m.endswith("raw_feature_bc_matrix/barcodes.tsv")
        for m in members
    )
    has_features = any(
        m.endswith("filtered_feature_bc_matrix/features.tsv.gz")
        or m.endswith("filtered_feature_bc_matrix/features.tsv")
        or m.endswith("filtered_feature_bc_matrix/genes.tsv.gz")
        or m.endswith("filtered_feature_bc_matrix/genes.tsv")
        or m.endswith("raw_feature_bc_matrix/features.tsv.gz")
        or m.endswith("raw_feature_bc_matrix/features.tsv")
        or m.endswith("raw_feature_bc_matrix/genes.tsv.gz")
        or m.endswith("raw_feature_bc_matrix/genes.tsv")
        for m in members
    )
    has_spatial_dir = any("/spatial/" in m for m in members)

    format_name = "visium_10x" if (has_matrix and has_barcodes and has_features and has_spatial_dir) else "unknown"
    return {
        "format": format_name,
        "file_count": len(members),
        "members_preview": sorted(members)[:40],
    }


def _open_tar_text_member(tf: tarfile.TarFile, member_name: str):
    """Open a tar member as text stream (auto-handles .gz members)."""
    raw = tf.extractfile(member_name)
    if raw is None:
        raise FileNotFoundError(f"Could not open member from tar: {member_name}")
    if member_name.endswith(".gz"):
        return gzip.open(raw, "rt", encoding="utf-8")
    return io.TextIOWrapper(raw, encoding="utf-8")


def _read_tar_tsv_column(tf: tarfile.TarFile, member_name: str, col: int = 0) -> list[str]:
    with _open_tar_text_member(tf, member_name) as fh:
        out: list[str] = []
        for line in fh:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if col >= len(parts):
                raise ValueError(
                    f"TSV member {member_name} has only {len(parts)} columns on line: {line[:120]!r}"
                )
            out.append(parts[col])
    return out


def _read_tar_mtx_sparse(tf: tarfile.TarFile, member_name: str) -> sp.csr_matrix:
    raw = tf.extractfile(member_name)
    if raw is None:
        raise FileNotFoundError(f"Could not open matrix member from tar: {member_name}")
    if member_name.endswith(".gz"):
        with gzip.GzipFile(fileobj=raw, mode="rb") as fh:
            mat = sio.mmread(fh)
    else:
        mat = sio.mmread(raw)
    if sp.issparse(mat):
        return mat.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(mat, dtype=np.float32))


def _find_required_member(members: list[str], suffixes: tuple[str, ...], label: str) -> str:
    for suffix in suffixes:
        for member_name in members:
            if member_name.endswith(suffix):
                return member_name
    raise FileNotFoundError(
        f"Could not find {label} in tarball. Tried suffixes: {suffixes}. "
        f"Found files (first 50): {sorted(members)[:50]}"
    )


def _load_spatial_coords_from_tar(
    tf: tarfile.TarFile,
    member_names: list[str],
    barcodes: list[str],
    sample_id: str,
) -> np.ndarray:
    expected_coord_files = (
        "tissue_positions.csv",
        "tissue_positions_list.csv",
        "tissue_positions.csv.gz",
        "tissue_positions_list.csv.gz",
    )

    coord_member: str | None = None
    for member_name in member_names:
        base = Path(member_name).name
        if base.startswith("._"):
            continue
        if base in expected_coord_files and "/spatial/" in member_name:
            coord_member = member_name
            break

    if coord_member is None:
        found_spatial_files = sorted(
            Path(m).name
            for m in member_names
            if "/spatial/" in m and not Path(m).name.startswith("._")
        )
        raise ValueError(
            f"{sample_id}: spatial coordinates not found in tarball.\n"
            f"Expected one of {list(expected_coord_files)} under a spatial/ directory.\n"
            f"Found spatial files: {found_spatial_files}"
        )

    with _open_tar_text_member(tf, coord_member) as fh:
        coord_text = fh.read()
    if not coord_text.strip():
        raise ValueError(f"{sample_id}: coordinate file is empty: {coord_member}")

    first_line = coord_text.splitlines()[0].strip().lower()
    has_header = (
        "barcode" in first_line
        or "pxl_row_in_fullres" in first_line
        or "array_row" in first_line
    )

    if has_header:
        df = pd.read_csv(io.StringIO(coord_text))
        if "barcode" not in df.columns:
            df = df.rename(columns={df.columns[0]: "barcode"})
    else:
        df = pd.read_csv(
            io.StringIO(coord_text),
            header=None,
            names=[
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_row_in_fullres",
                "pxl_col_in_fullres",
            ],
        )

    if "barcode" not in df.columns:
        raise ValueError(
            f"{sample_id}: unable to identify barcode column in coordinate file {coord_member}. "
            f"Columns found: {list(df.columns)}"
        )
    df = df.set_index("barcode")

    if {"pxl_col_in_fullres", "pxl_row_in_fullres"}.issubset(df.columns):
        coord_cols = ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    elif {"array_col", "array_row"}.issubset(df.columns):
        coord_cols = ["array_col", "array_row"]
    else:
        raise ValueError(
            f"{sample_id}: coordinate file {coord_member} lacks usable coordinate columns.\n"
            f"Expected either ['pxl_col_in_fullres','pxl_row_in_fullres'] "
            f"or ['array_col','array_row'].\n"
            f"Found columns: {list(df.columns)}"
        )

    aligned = df.reindex(barcodes)[coord_cols]
    missing_mask = aligned.isna().any(axis=1)
    if bool(missing_mask.any()):
        missing_count = int(missing_mask.sum())
        raise ValueError(
            f"{sample_id}: {missing_count} / {len(barcodes)} barcodes are missing coordinates in {coord_member}."
        )
    return aligned.to_numpy(dtype=np.float32, copy=False)


def load_spatial_sample_from_tarball(
    tar_path: Path,
    max_spots_per_sample: int | None = None,
) -> anndata.AnnData:
    """Load one GEO spatial tarball into AnnData (spots x genes)."""
    tar_path = Path(tar_path)
    inspect = inspect_spatial_tarball_format(tar_path)
    if inspect["format"] != "visium_10x":
        raise ValueError(
            f"Unsupported spatial tar format for {tar_path}.\n"
            f"Detected format: {inspect['format']}\n"
            f"Members preview: {inspect['members_preview']}"
        )

    sample_stem = _sample_stem_from_tar_path(tar_path)
    info = _parse_stem(sample_stem)

    with tarfile.open(tar_path, "r:gz") as tf:
        member_names = [m.name for m in tf.getmembers() if m.isfile() and not m.name.endswith("/")]

        matrix_member = _find_required_member(
            member_names,
            (
                "filtered_feature_bc_matrix/matrix.mtx.gz",
                "filtered_feature_bc_matrix/matrix.mtx",
                "raw_feature_bc_matrix/matrix.mtx.gz",
                "raw_feature_bc_matrix/matrix.mtx",
            ),
            "matrix file",
        )
        barcode_member = _find_required_member(
            member_names,
            (
                "filtered_feature_bc_matrix/barcodes.tsv.gz",
                "filtered_feature_bc_matrix/barcodes.tsv",
                "raw_feature_bc_matrix/barcodes.tsv.gz",
                "raw_feature_bc_matrix/barcodes.tsv",
            ),
            "barcodes file",
        )
        feature_member = _find_required_member(
            member_names,
            (
                "filtered_feature_bc_matrix/features.tsv.gz",
                "filtered_feature_bc_matrix/features.tsv",
                "filtered_feature_bc_matrix/genes.tsv.gz",
                "filtered_feature_bc_matrix/genes.tsv",
                "raw_feature_bc_matrix/features.tsv.gz",
                "raw_feature_bc_matrix/features.tsv",
                "raw_feature_bc_matrix/genes.tsv.gz",
                "raw_feature_bc_matrix/genes.tsv",
            ),
            "features file",
        )

        barcodes = _read_tar_tsv_column(tf, barcode_member, col=0)
        feature_col = 1 if Path(feature_member).name.startswith("features") else 0
        features = _read_tar_tsv_column(tf, feature_member, col=feature_col)
        X_genes_spots = _read_tar_mtx_sparse(tf, matrix_member)

        n_genes, n_spots = X_genes_spots.shape
        if len(features) != n_genes or len(barcodes) != n_spots:
            raise ValueError(
                f"{sample_stem}: matrix shape is ({n_genes} genes, {n_spots} spots), "
                f"but parsed {len(features)} features and {len(barcodes)} barcodes."
            )

        coords = _load_spatial_coords_from_tar(
            tf=tf,
            member_names=member_names,
            barcodes=barcodes,
            sample_id=sample_stem,
        )

    if max_spots_per_sample is not None and max_spots_per_sample > 0:
        n_keep = min(int(max_spots_per_sample), len(barcodes))
    else:
        n_keep = len(barcodes)

    keep_idx = np.arange(n_keep, dtype=np.int64)
    X = X_genes_spots[:, keep_idx].T.tocsr()
    kept_barcodes = [barcodes[i] for i in keep_idx.tolist()]
    kept_coords = coords[keep_idx, :]

    obs_index = pd.Index(
        [f"{info['sample_id']}:{bc}" for bc in kept_barcodes],
        name="spot_obs_id",
    )
    obs = pd.DataFrame(index=obs_index)
    obs["spot_id"] = kept_barcodes
    obs["barcode"] = kept_barcodes
    obs["donor_id"] = info["patient_id"]
    obs["patient_id"] = info["patient_id"]
    obs["stage"] = info["stage_normalized"]
    obs["stage_raw"] = info["stage_raw"]
    obs["gsm_id"] = info["gsm"]
    obs["gsm"] = info["gsm"]
    obs["sample_id"] = info["sample_id"]
    obs["modality"] = "spatial"

    var = pd.DataFrame(index=pd.Index(features, name="gene"))
    if var.index.to_series().str.fullmatch(r"\d+").all():
        var["gene_id_raw"] = var.index.astype(str)

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["spatial"] = kept_coords.astype(np.float32, copy=False)
    return adata


def load_spatial_dataset(
    extracted_dir: Path,
    max_donors: int | None = None,
    max_samples_per_stage: int | None = None,
    max_spots_per_sample: int | None = None,
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Load and concatenate spatial tarballs from *extracted_dir*."""
    manifest = discover_spatial_tarballs(extracted_dir)
    selected = apply_spatial_smoke_limits(
        manifest,
        max_donors=max_donors,
        max_samples_per_stage=max_samples_per_stage,
    )
    if selected.empty:
        raise RuntimeError("No spatial samples selected after applying limits.")

    adatas: list[anndata.AnnData] = []
    for row in selected.itertuples(index=False):
        ad = load_spatial_sample_from_tarball(
            Path(row.input_path),
            max_spots_per_sample=max_spots_per_sample,
        )
        adatas.append(ad)

    merged = anndata.concat(adatas, join="outer", merge="same")
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    merged.layers["counts"] = merged.X.copy()

    if "spatial" not in merged.obsm:
        raise ValueError("Merged spatial AnnData is missing obsm['spatial'].")

    return merged, selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(
        "Usage:\n"
        "  python -m stagebridge.data.luad_evo.visium expand   <extracted_dir>  <samples_dir>\n"
        "  python -m stagebridge.data.luad_evo.visium load     <sample_dir>     <output.h5ad>\n"
        "  python -m stagebridge.data.luad_evo.visium manifest <samples_dir>    <manifest.csv>\n"
        "  python -m stagebridge.data.luad_evo.visium merge    <manifest.csv>   <merged.h5ad>\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "expand":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        expand_spatial_tarballs(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "load":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        write_spatial_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "manifest":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        build_spatial_manifest(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "merge":
        if len(sys.argv) != 4:
            _usage()
            sys.exit(1)
        merge_spatial_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(f"Unknown command: {cmd!r}")
        _usage()
        sys.exit(1)
