"""IO for the NSCLC brain metastasis dataset (GSE223499 / GSE223501 / GSE223502).

Three data modalities from Rossi et al.:

  GSE223499 — snRNA-seq (10x Chromium, per-patient h5 + integrated metadata CSV)
  GSE223501 — Slide-seq V2 spatial (coords CSV + counts CSV per sample)
  GSE223502 — lpWGS copy-number (ichorCNA .cna.seg files per patient)

Patient IDs: PA001–PA141, KRAS_6–KRAS_17, STK_1–STK_22, N254/N561/N586
Tissue types: PRIMARY, BRAIN_METS, CHEST_WALL_MET
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# snRNA-seq: integrated metadata CSV (GSE223499_sn_integrated_data.csv.gz)
# ---------------------------------------------------------------------------

_TISSUE_MAP: dict[str, str] = {
    "PRIMARY": "LUAD",
    "BRAIN_METS": "BrainMet",
    "CHEST_WALL_MET": "ChestWallMet",
}


def load_brainmets_metadata(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Load the integrated metadata CSV (no gene expression, just obs).

    Returns DataFrame with columns normalised for StageBridge:
      - barcode, patient_id, tissue_type, stage (mapped), stk11_status,
        tumor_nontumor_major, tumor_nontumor_finer, nCount_RNA, nFeature_RNA, ...
    """
    df = pd.read_csv(csv_path, index_col=0)
    df = df.rename(
        columns={
            "orig.ident": "patient_id_raw",
            "PRIMARY vs BRAIN_METS vs CHEST_WALL_MET": "tissue_type",
            "STK11-MUT vs STK11-WT": "stk11_status",
            "patient": "patient_id",
        }
    )
    df.index.name = "barcode"
    df = df.reset_index()

    # Map tissue type to extended stage ontology names
    df["stage"] = df["tissue_type"].map(_TISSUE_MAP)
    unmapped = df["stage"].isna().sum()
    if unmapped > 0:
        log.warning("%d cells have unmapped tissue type", unmapped)

    return df


def load_brainmets_snrna_h5(
    tar_path: str | Path,
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Load per-patient 10x h5 files from GSE223499_RAW.tar.

    Parameters
    ----------
    tar_path : path to GSE223499_RAW.tar
    patient_id : if specified, only load this patient (e.g. "PA001")

    Returns
    -------
    dict mapping patient_id -> dict with keys:
        "barcodes": np.ndarray of str
        "features": np.ndarray of str (gene names)
        "matrix": scipy.sparse.csc_matrix (genes x cells)
    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError("h5py required for reading 10x h5 files") from err

    tar_path = Path(tar_path)
    results: dict[str, Any] = {}

    # Match pattern: GSMxxxxxxx_NSCLC_<PATIENT>_sn_raw_feature_bc_matrix.h5
    h5_re = re.compile(r"GSM\d+_NSCLC_(\w+)_sn_raw_feature_bc_matrix\.h5$")

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            m = h5_re.search(member.name)
            if m is None:
                continue
            pid = m.group(1)
            if patient_id is not None and pid != patient_id:
                continue

            log.info("Loading h5 for patient %s", pid)
            fobj = tf.extractfile(member)
            if fobj is None:
                continue

            data = fobj.read()
            buf = io.BytesIO(data)
            with h5py.File(buf, "r") as f:
                # 10x h5 format: /matrix/{barcodes, features/name, data, indices, indptr, shape}
                grp = f["matrix"]
                barcodes = np.array(grp["barcodes"]).astype(str)
                features = np.array(grp["features"]["name"]).astype(str)
                mat = sp.csc_matrix(
                    (np.array(grp["data"]), np.array(grp["indices"]), np.array(grp["indptr"])),
                    shape=tuple(grp["shape"]),
                )
                results[pid] = {
                    "barcodes": barcodes,
                    "features": features,
                    "matrix": mat,
                }

            if patient_id is not None:
                break

    return results


# ---------------------------------------------------------------------------
# Slide-seq V2 spatial (GSE223501)
# ---------------------------------------------------------------------------

_SLIDESEQ_COORD_RE = re.compile(r"GSM\d+_NSCLC_(\w+)_slideseq_coords\.csv\.gz$")
_SLIDESEQ_COUNT_RE = re.compile(r"GSM\d+_NSCLC_(\w+)_slideseq_counts\.csv\.gz$")


def load_slideseq_sample(
    tar_path: str | Path,
    sample_id: str,
) -> dict[str, Any]:
    """Load one Slide-seq sample (coords + counts) from GSE223501_RAW.tar.

    Parameters
    ----------
    tar_path : path to GSE223501_RAW.tar
    sample_id : e.g. "PA019", "STK_14", "KRAS_10"

    Returns
    -------
    dict with keys:
        "barcodes": np.ndarray of str
        "genes": np.ndarray of str
        "counts": scipy.sparse.csc_matrix (genes x beads)
        "coords": np.ndarray (beads, 2) — x, y spatial coords
    """
    tar_path = Path(tar_path)
    coords_df = None
    counts_mat = None
    genes = None
    count_barcodes = None

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            # Coords
            m = _SLIDESEQ_COORD_RE.search(member.name)
            if m and m.group(1) == sample_id:
                fobj = tf.extractfile(member)
                if fobj:
                    coords_df = pd.read_csv(
                        io.BytesIO(fobj.read()),
                        compression="gzip",
                        index_col=0,
                    )

            # Counts (gene x bead CSV, potentially large)
            m = _SLIDESEQ_COUNT_RE.search(member.name)
            if m and m.group(1) == sample_id:
                fobj = tf.extractfile(member)
                if fobj:
                    raw = fobj.read()
                    # Read header line for barcodes
                    header_end = raw.index(b"\n")
                    header = gzip.decompress(raw[: max(header_end + 4096, len(raw))])
                    # Actually, decompress fully — Slide-seq samples are ~50k beads
                    text = gzip.decompress(raw).decode("utf-8")
                    lines = text.split("\n")
                    header_cols = lines[0].split(",")
                    count_barcodes = np.array([c.strip('"') for c in header_cols[1:]])

                    gene_names = []
                    data_rows = []
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        parts = line.split(",")
                        gene_names.append(parts[0].strip('"'))
                        data_rows.append(np.array([int(x) for x in parts[1:]], dtype=np.int32))

                    genes = np.array(gene_names)
                    counts_mat = sp.csc_matrix(np.array(data_rows))

    if coords_df is None or counts_mat is None:
        raise FileNotFoundError(f"Sample {sample_id} not found in {tar_path}")

    # Align barcodes between coords and counts
    coord_barcodes = coords_df.index.values.astype(str)
    # Replace dots with dashes to match (R export replaces - with .)
    if count_barcodes is not None:
        count_barcodes_fixed = np.array([b.replace(".", "-") for b in count_barcodes])
    else:
        count_barcodes_fixed = coord_barcodes

    return {
        "barcodes": coord_barcodes,
        "genes": genes,
        "counts": counts_mat,  # (genes, beads)
        "coords": coords_df[["x", "y"]].values.astype(np.float32),
    }


def list_slideseq_samples(tar_path: str | Path) -> list[str]:
    """List available Slide-seq sample IDs in the tar."""
    samples = set()
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            m = _SLIDESEQ_COORD_RE.search(member.name)
            if m:
                samples.add(m.group(1))
    return sorted(samples)


# ---------------------------------------------------------------------------
# lpWGS copy-number (GSE223502) — ichorCNA output
# ---------------------------------------------------------------------------

_LPWGS_CNA_RE = re.compile(r"GSM\d+_NSCLC_(\w+)_lpwgs_\S+_tumor\.cna\.seg\.gz$")

# Chromosome arm lengths (GRCh38, Mb) for normalised CNA features
_CHROMOSOME_ARMS = 44  # 22 autosomes x 2 arms

# Key oncogene/tumor-suppressor loci for arm-level CNA features
_CNA_GENE_LOCI: dict[str, tuple[str, int, int]] = {
    "myc_amp": ("8", 127_735_434, 127_742_951),  # MYC 8q24
    "egfr_amp": ("7", 55_019_017, 55_211_628),  # EGFR 7p11
    "cdkn2a_del": ("9", 21_967_751, 21_995_301),  # CDKN2A 9p21
    "rb1_del": ("13", 48_303_751, 48_481_890),  # RB1 13q14
    "pten_del": ("10", 87_863_113, 87_971_930),  # PTEN 10q23
    "nkx2_1_amp": ("14", 36_985_602, 36_989_163),  # NKX2-1/TTF1 14q13
    "kras_amp": ("12", 25_205_246, 25_250_929),  # KRAS 12p12
    "stk11_del": ("19", 1_205_866, 1_228_675),  # STK11 19p13
}


def parse_lpwgs_features_from_tar(
    tar_path: str | Path,
) -> pd.DataFrame:
    """Parse all lpWGS CNA segment files and compute per-patient features.

    Features extracted:
      - fraction_genome_altered (FGA): fraction of genome with |log2ratio| > 0.2
      - num_segments: total CNA segments (genomic instability proxy)
      - mean_ploidy: weighted-average copy number
      - gain_fraction: fraction of genome with GAIN
      - loss_fraction: fraction of genome with LOSS
      - Per-locus binary flags: myc_amp, egfr_amp, cdkn2a_del, rb1_del,
        pten_del, nkx2_1_amp, kras_amp, stk11_del

    Returns
    -------
    pd.DataFrame with columns: patient_id + feature columns
    """
    tar_path = Path(tar_path)
    rows: list[dict[str, Any]] = []

    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            m = _LPWGS_CNA_RE.search(member.name)
            if m is None:
                continue
            patient_id = m.group(1)
            log.info("Parsing lpWGS CNA for %s", patient_id)

            fobj = tf.extractfile(member)
            if fobj is None:
                continue

            df = pd.read_csv(fobj, sep="\t", compression="gzip")
            feats = _compute_cna_features(df, patient_id)
            rows.append(feats)

    if not rows:
        raise ValueError(f"No CNA files parsed from {tar_path}")

    result = pd.DataFrame(rows)
    return result.sort_values("patient_id").reset_index(drop=True)


def _compute_cna_features(df: pd.DataFrame, patient_id: str) -> dict[str, Any]:
    """Compute CNA summary features from an ichorCNA .cna.seg DataFrame."""
    feats: dict[str, Any] = {"patient_id": patient_id}

    # Column names vary — find the right ones
    cn_col = [c for c in df.columns if "copy.number" in c.lower() or "copy_number" in c.lower()]
    event_col = [c for c in df.columns if "event" in c.lower()]
    logr_col = [c for c in df.columns if "logr" in c.lower() and "copy" not in c.lower()]

    # Segment sizes
    df["seg_size"] = df["end"] - df["start"]
    total_genome = df["seg_size"].sum()

    if total_genome == 0:
        feats.update(
            {
                "fga": 0.0,
                "num_segments": 0,
                "mean_ploidy": 2.0,
                "gain_fraction": 0.0,
                "loss_fraction": 0.0,
            }
        )
        for locus in _CNA_GENE_LOCI:
            feats[locus] = 0.0
        return feats

    # FGA: fraction with |logR| > 0.2
    if logr_col:
        logr = pd.to_numeric(df[logr_col[0]], errors="coerce").fillna(0)
        altered = df.loc[logr.abs() > 0.2, "seg_size"].sum()
        feats["fga"] = float(altered / total_genome)
    else:
        feats["fga"] = 0.0

    feats["num_segments"] = len(df)

    # Mean ploidy (weighted by segment size)
    if cn_col:
        cn = pd.to_numeric(df[cn_col[0]], errors="coerce").fillna(2)
        feats["mean_ploidy"] = float((cn * df["seg_size"]).sum() / total_genome)
    else:
        feats["mean_ploidy"] = 2.0

    # Gain/loss fraction
    if event_col:
        events = df[event_col[0]].astype(str).str.upper()
        feats["gain_fraction"] = float(
            df.loc[events.str.contains("GAIN", na=False), "seg_size"].sum() / total_genome
        )
        feats["loss_fraction"] = float(
            df.loc[events.str.contains("LOSS|HLOSS|DEL", na=False), "seg_size"].sum()
            / total_genome
        )
    else:
        feats["gain_fraction"] = 0.0
        feats["loss_fraction"] = 0.0

    # Per-locus flags
    for locus_name, (chrom, start, end) in _CNA_GENE_LOCI.items():
        mask = (df["chr"].astype(str) == chrom) & (df["start"] <= end) & (df["end"] >= start)
        if mask.any() and cn_col:
            cn_at_locus = pd.to_numeric(df.loc[mask, cn_col[0]], errors="coerce").mean()
            if "amp" in locus_name:
                feats[locus_name] = float(cn_at_locus > 2.5 if not np.isnan(cn_at_locus) else 0)
            else:  # deletion
                feats[locus_name] = float(cn_at_locus < 1.5 if not np.isnan(cn_at_locus) else 0)
        else:
            feats[locus_name] = 0.0

    return feats


LPWGS_FEATURE_COLS: list[str] = [
    "fga",
    "num_segments",
    "mean_ploidy",
    "gain_fraction",
    "loss_fraction",
    "myc_amp",
    "egfr_amp",
    "cdkn2a_del",
    "rb1_del",
    "pten_del",
    "nkx2_1_amp",
    "kras_amp",
    "stk11_del",
]
