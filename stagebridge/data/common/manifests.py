"""
Manifest helpers — shared utilities for building and validating
sample manifests used by both snRNA and spatial pipelines.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from stagebridge.logging_utils import get_logger

log = get_logger(__name__)

REQUIRED_SNRNA_COLS = {"sample_id", "input_path", "gsm", "patient_id", "stage"}
REQUIRED_SPATIAL_COLS = {"sample_id", "sample_dir", "gsm", "patient_id", "stage"}


def load_manifest(csv_path: Path, required_cols: set[str] | None = None) -> pd.DataFrame:
    """Load and validate a manifest CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the manifest CSV.
    required_cols : set[str] or None
        If provided, raises ValueError if any column is missing.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest CSV not found: {csv_path}\n"
            f"Generate it first with build_snrna_manifest() or build_spatial_manifest()."
        )
    df = pd.read_csv(csv_path)
    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Manifest {csv_path} is missing required columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )
    log.info("Loaded manifest: %d rows from %s", len(df), csv_path)
    return df


def validate_snrna_manifest(csv_path: Path) -> pd.DataFrame:
    """Load and validate a snRNA manifest CSV."""
    return load_manifest(csv_path, required_cols=REQUIRED_SNRNA_COLS)


def validate_spatial_manifest(csv_path: Path) -> pd.DataFrame:
    """Load and validate a spatial manifest CSV."""
    return load_manifest(csv_path, required_cols=REQUIRED_SPATIAL_COLS)


def resolve_git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return current git commit hash when available, else ``None``."""
    repo_root = Path(repo_root or Path.cwd())
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def summarize_anndata(
    adata: Any,
    donor_col: str = "donor_id",
    stage_col: str = "stage",
) -> dict[str, Any]:
    """Build a compact summary block for an AnnData object."""
    donors: list[str] = []
    stages: list[str] = []

    if donor_col in adata.obs.columns:
        donors = sorted({str(x) for x in adata.obs[donor_col].astype(str).tolist()})
    elif "patient_id" in adata.obs.columns:
        donors = sorted({str(x) for x in adata.obs["patient_id"].astype(str).tolist()})

    if stage_col in adata.obs.columns:
        stages = sorted({str(x) for x in adata.obs[stage_col].astype(str).tolist()})

    return {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "donors": donors,
        "stages": stages,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write *payload* to *path* as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def write_resolved_config_yaml(path: Path, cfg: Any) -> Path:
    """Write fully resolved Hydra config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from omegaconf import OmegaConf

        text = OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        try:
            import yaml

            text = yaml.safe_dump(cfg, sort_keys=False)
        except Exception:
            text = str(cfg)
    header = f"# {path}\n"
    if not text.startswith("# "):
        text = header + text
    path.write_text(text, encoding="utf-8")
    return path
