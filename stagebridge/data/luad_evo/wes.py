"""WES somatic variant feature extraction for StageBridge.

Reads GATK Mutect2 PASS-filtered VCFs from GSE307529_RAW.tar and computes
per-(patient, stage) genomic features:

    - tmb: tumor mutation burden (filtered SNVs per Mb)
    - kras_mut: KRAS codon 12/13 hotspot mutation (binary)
    - egfr_mut: EGFR L858R or exon-19 deletion (binary)
    - tp53_mut: any coding TP53 variant (binary)
    - stk11_mut: any coding STK11 variant (binary)
    - keap1_mut: any coding KEAP1 variant (binary)
    - smad4_mut: any coding SMAD4 variant (binary)
    - braf_mut: BRAF V600E hotspot (binary)

Usage::

    from stagebridge.data.luad_evo.wes import parse_wes_features_from_tar
    df = parse_wes_features_from_tar("$STAGEBRIDGE_DATA_ROOT/raw/geo/GSE307529_RAW.tar")
    df.to_parquet("wes_features.parquet", index=False)
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from stagebridge.data.common.schema import WESCohort
from stagebridge.data.luad_evo.metadata import resolve_luad_evo_paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exome size for TMB normalisation (GRCh38 coding+UTR ≈ 38 Mb)
_EXOME_MB: float = 38.0

# Minimum quality filters for TMB computation
_TMB_MIN_DP: int = 8
_TMB_MIN_AF: float = 0.05

# ---------------------------------------------------------------------------
# Gene coding-region intervals (GRCh38 / hg38) used for binary mutation flags.
# Tuple: (chrom, start, end)  — half-open, 0-based
# Coordinates verified from UCSC Genome Browser / Ensembl GRCh38
_GENE_REGIONS: dict[str, tuple[str, int, int]] = {
    "tp53": ("chr17", 7_668_421, 7_687_490),      # GRCh38: TP53 coding region
    "stk11": ("chr19", 1_205_798, 1_228_434),     # GRCh38: STK11/LKB1
    "keap1": ("chr19", 10_596_796, 10_614_367),   # GRCh38: KEAP1
    "smad4": ("chr18", 51_030_213, 51_085_042),   # GRCh38: SMAD4
}

# KRAS codon 12/13 hotspot positions (chr12, GRCh38)
# G12: chr12:25,245,350 (G12C/D/V/A/S/R)
# G13: chr12:25,245,347 (G13D/C)
_KRAS_HOTSPOT_CHROM = "chr12"
_KRAS_HOTSPOT_POSITIONS: frozenset[int] = frozenset({
    25_245_347,  # G13 codon position 1
    25_245_348,  # G13 codon position 2
    25_245_349,  # G13 codon position 3
    25_245_350,  # G12 codon position 1
    25_245_351,  # G12 codon position 2
    25_245_352,  # G12 codon position 3
})

# EGFR L858R (rs121434568): GRCh38 = chr7:55,191,822 (T>G)
# EGFR exon-19 deletions: GRCh38 = chr7:55,174,721-55,174,820 (ELREA deletion region)
_EGFR_L858R = ("chr7", 55_191_822)
_EGFR_EXON19_REGION = ("chr7", 55_174_721, 55_174_820)

# BRAF V600E: chr7:140,753,336 in GRCh38 (A>T, minus strand = T>A on plus)
_BRAF_V600E = ("chr7", 140_753_336)

# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(r"GSM\d+_(P\d+[^_]*)_([\w-]+)\.WES\.PASS\.recode\.vcf\.gz$")


def _parse_patient_stage(filename: str) -> tuple[str, str] | None:
    """Return (patient_id, stage_raw) from a VCF filename, or None."""
    m = _FILENAME_RE.search(filename)
    if m is None:
        return None
    patient_id = m.group(1)  # e.g. "P1", "P21"
    stage_raw = m.group(2)  # e.g. "AAH", "AIS-1", "LUAD"
    return patient_id, stage_raw


def _normalize_stage(stage_raw: str) -> str:
    """Collapse multi-region suffixes: 'AIS-1' → 'AIS', 'LUAD-1' → 'LUAD'."""
    return re.sub(r"-\d+$", "", stage_raw)


def _iter_vcf_lines(fileobj: io.BufferedIOBase) -> Iterator[list[str]]:
    """Yield parsed data-line fields from a gzipped VCF file object."""
    import gzip

    with gzip.open(fileobj, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            yield fields


def _extract_af_dp(format_field: str, sample_field: str) -> tuple[float, int]:
    """Return (allele_fraction, depth) from FORMAT/SAMPLE columns; (-1, 0) on failure."""
    try:
        fmt = format_field.split(":")
        vals = sample_field.split(":")
        d = dict(zip(fmt, vals))
        af = float(d.get("AF", "-1").split(",")[0])
        dp = int(d.get("DP", "0"))
        return af, dp
    except Exception:
        return -1.0, 0


def _compute_features_from_vcf(fileobj: io.BufferedIOBase) -> dict[str, float]:
    """Parse one gzipped VCF stream and return feature dict."""
    tmb_count = 0
    kras_mut = 0
    egfr_mut = 0
    tp53_mut = 0
    stk11_mut = 0
    keap1_mut = 0
    smad4_mut = 0
    braf_mut = 0

    for fields in _iter_vcf_lines(fileobj):
        chrom = fields[0]
        try:
            pos = int(fields[1])  # 1-based VCF position
        except ValueError:
            continue
        ref = fields[3]
        alt = fields[4]

        # Quality filter for TMB
        af, dp = _extract_af_dp(fields[8], fields[9])
        is_snv = len(ref) == 1 and len(alt) == 1 and alt != "."
        if is_snv and dp >= _TMB_MIN_DP and af >= _TMB_MIN_AF:
            tmb_count += 1

        # Convert to 0-based for range checks
        pos0 = pos - 1

        # KRAS codon 12/13
        if chrom == _KRAS_HOTSPOT_CHROM and pos in _KRAS_HOTSPOT_POSITIONS:
            if af >= _TMB_MIN_AF:
                kras_mut = 1

        # EGFR L858R
        if chrom == _EGFR_L858R[0] and pos == _EGFR_L858R[1] and ref == "T" and alt == "G":
            if af >= _TMB_MIN_AF:
                egfr_mut = 1
        # EGFR exon-19 deletion region (indels acceptable here)
        _e19_c, _e19_s, _e19_e = _EGFR_EXON19_REGION
        if chrom == _e19_c and _e19_s <= pos0 < _e19_e and len(ref) != len(alt):
            if af >= _TMB_MIN_AF:
                egfr_mut = 1

        # BRAF V600E
        if chrom == _BRAF_V600E[0] and pos == _BRAF_V600E[1] and ref == "A" and alt == "T":
            if af >= _TMB_MIN_AF:
                braf_mut = 1

        # Gene-region flags
        for flag_name, (g_chrom, g_start, g_end) in _GENE_REGIONS.items():
            if chrom == g_chrom and g_start <= pos0 < g_end and af >= _TMB_MIN_AF:
                if flag_name == "tp53":
                    tp53_mut = 1
                elif flag_name == "stk11":
                    stk11_mut = 1
                elif flag_name == "keap1":
                    keap1_mut = 1
                elif flag_name == "smad4":
                    smad4_mut = 1

    tmb = tmb_count / _EXOME_MB
    return {
        "tmb": tmb,
        "kras_mut": float(kras_mut),
        "egfr_mut": float(egfr_mut),
        "tp53_mut": float(tp53_mut),
        "stk11_mut": float(stk11_mut),
        "keap1_mut": float(keap1_mut),
        "smad4_mut": float(smad4_mut),
        "braf_mut": float(braf_mut),
    }


def parse_wes_features_from_tar(tar_path: str | Path) -> pd.DataFrame:
    """Parse all WES VCFs from a RAW tar and return a feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: patient_id, stage, tmb, kras_mut, egfr_mut, tp53_mut,
        stk11_mut, keap1_mut, smad4_mut, braf_mut.
        One row per (patient_id, stage_raw) VCF file; when a patient has
        multiple lesions of the same normalised stage (e.g. AIS and AIS-1),
        features are averaged so there is exactly one row per
        (patient_id, stage) in the returned DataFrame.
    """
    tar_path = Path(tar_path)
    rows: list[dict] = []

    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        n = len(members)
        for i, member in enumerate(members):
            parsed = _parse_patient_stage(member.name)
            if parsed is None:
                log.debug("Skipping unrecognised filename: %s", member.name)
                continue
            patient_id, stage_raw = parsed
            stage = _normalize_stage(stage_raw)
            log.info("[%d/%d] Parsing %s (%s → %s)", i + 1, n, member.name, stage_raw, stage)

            fobj = tf.extractfile(member)
            if fobj is None:
                log.warning("Could not extract %s", member.name)
                continue

            feats = _compute_features_from_vcf(fobj)
            feats["patient_id"] = patient_id
            feats["stage"] = stage
            rows.append(feats)

    if not rows:
        raise ValueError(f"No VCF files parsed from {tar_path}")

    df = pd.DataFrame(rows)

    # Average multi-region duplicates (same patient + stage)
    numeric_cols = [c for c in df.columns if c not in ("patient_id", "stage")]
    df = df.groupby(["patient_id", "stage"], as_index=False)[numeric_cols].mean()
    df = df.sort_values(["patient_id", "stage"]).reset_index(drop=True)
    return df


def load_wes_features(path: str | Path) -> pd.DataFrame:
    """Load a pre-computed wes_features.parquet."""
    return pd.read_parquet(path)


def build_wes_feature_tensor(
    wes_df: pd.DataFrame,
    patient_id: str,
    stage: str,
    feature_cols: list[str] | None = None,
) -> "np.ndarray":
    """Return feature vector for (patient_id, stage), or zeros if unknown.

    Parameters
    ----------
    wes_df:
        DataFrame from :func:`load_wes_features`.
    patient_id:
        Donor ID string, e.g. ``"P1"``.
    stage:
        Stage string, e.g. ``"AAH"`` or ``"LUAD"``.
    feature_cols:
        Column subset to use.  Defaults to all numeric columns.
    """
    if feature_cols is None:
        feature_cols = [c for c in wes_df.columns if c not in ("patient_id", "stage")]

    mask = (wes_df["patient_id"] == patient_id) & (wes_df["stage"] == stage)
    rows = wes_df.loc[mask, feature_cols]
    if rows.empty:
        return np.zeros(len(feature_cols), dtype=np.float32)
    return rows.iloc[0].values.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature column names (stable, used for model config)
WES_FEATURE_COLS: list[str] = [
    "tmb",
    "kras_mut",
    "egfr_mut",
    "tp53_mut",
    "stk11_mut",
    "keap1_mut",
    "smad4_mut",
    "braf_mut",
]


def resolve_wes_features_path(cfg: object | None = None) -> Path:
    """Resolve the active LUAD WES parquet path."""
    if cfg is not None:
        return resolve_luad_evo_paths(cfg).wes_features_path
    from stagebridge.config import get_data_root

    return get_data_root() / "processed" / "features" / "wes_features.parquet"


def load_luad_evo_wes_features(
    cfg: object | None = None,
    *,
    stages: list[str] | None = None,
    donors: list[str] | None = None,
) -> WESCohort:
    """Load the active LUAD WES feature table."""
    path = resolve_wes_features_path(cfg)
    df = pd.read_parquet(path).copy()
    if stages:
        df = df[df["stage"].isin(stages)].copy()
    if donors:
        df = df[df["patient_id"].isin([str(donor) for donor in donors])].copy()
    return WESCohort(
        frame=df.reset_index(drop=True),
        feature_columns=tuple(WES_FEATURE_COLS),
        source_path=path,
    )


def build_wes_feature_lookup(
    wes: WESCohort | pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    """Build a `(donor_id, stage)` lookup for WES regularization."""
    if isinstance(wes, WESCohort):
        frame = wes.frame
        feature_columns = list(wes.feature_columns)
    else:
        frame = wes
        feature_columns = [col for col in WES_FEATURE_COLS if col in frame.columns]
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for row in frame.itertuples(index=False):
        lookup[(str(row.patient_id), str(row.stage))] = np.asarray(
            [getattr(row, col) for col in feature_columns],
            dtype=np.float32,
        )
    return lookup
