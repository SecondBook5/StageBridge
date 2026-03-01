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
    python -m stagebridge.io.geo_spatial expand   <extracted_dir> <samples_dir>
    python -m stagebridge.io.geo_spatial load     <sample_dir>    <output.h5ad>
    python -m stagebridge.io.geo_spatial manifest <samples_dir>   <manifest.csv>
    python -m stagebridge.io.geo_spatial merge    <manifest.csv>  <merged.h5ad>
"""
from __future__ import annotations

import gzip
import re
import sys
import tarfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from stagebridge.logging_utils import get_logger
from stagebridge.preprocessing.stage_ontology import (
    CANONICAL_STAGE_ORDER,
    normalize_stage_label,
)

log = get_logger(__name__)

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
    for name in ("filtered_feature_bc_matrix", "raw_feature_bc_matrix"):
        if (sample_dir / name).is_dir():
            return name
        for sub in sample_dir.iterdir():
            if sub.is_dir() and (sub / name).is_dir():
                return name
    raise FileNotFoundError(
        f"No filtered_feature_bc_matrix/ or raw_feature_bc_matrix/ found in:\n"
        f"  {sample_dir}\n"
        f"Contents: {[p.name for p in sample_dir.iterdir()]}"
    )


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
    """Read an MTX file; return (genes×cells CSR, n_genes, n_cells)."""
    opener = gzip.open if path.suffix == ".gz" else open
    rows, cols, data = [], [], []
    header_done = False
    n_genes = n_cells = 0
    with opener(path, "rt") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if not header_done:
                n_genes, n_cells = int(parts[0]), int(parts[1])
                header_done = True
                continue
            rows.append(int(parts[0]) - 1)
            cols.append(int(parts[1]) - 1)
            data.append(float(parts[2]))
    mat = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_genes, n_cells), dtype=np.float32
    )
    return mat, n_genes, n_cells


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(
        "Usage:\n"
        "  python -m stagebridge.io.geo_spatial expand   <extracted_dir>  <samples_dir>\n"
        "  python -m stagebridge.io.geo_spatial load     <sample_dir>     <output.h5ad>\n"
        "  python -m stagebridge.io.geo_spatial manifest <samples_dir>    <manifest.csv>\n"
        "  python -m stagebridge.io.geo_spatial merge    <manifest.csv>   <merged.h5ad>\n"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage(); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "expand":
        if len(sys.argv) != 4: _usage(); sys.exit(1)
        expand_spatial_tarballs(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "load":
        if len(sys.argv) != 4: _usage(); sys.exit(1)
        write_spatial_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "manifest":
        if len(sys.argv) != 4: _usage(); sys.exit(1)
        build_spatial_manifest(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "merge":
        if len(sys.argv) != 4: _usage(); sys.exit(1)
        merge_spatial_h5ad(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(f"Unknown command: {cmd!r}"); _usage(); sys.exit(1)
