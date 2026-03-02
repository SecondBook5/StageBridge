"""
stagebridge/io/paths.py
Single source of truth for resolving all filesystem paths from a Hydra config.

Usage
-----
    from stagebridge.io.paths import StageBridgePaths
    paths = StageBridgePaths.from_cfg(cfg)

    adata = anndata.read_h5ad(paths.snrna_merged)
    checkpoint_path = paths.checkpoint(run_id="my_run", fold=0)

All methods return ``pathlib.Path`` objects and create parent directories
only when ``mkdir=True`` is passed explicitly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_root(cfg_data_root: Optional[str] = None) -> Path:
    """
    Determine the data root with a clear precedence chain:
    1. cfg_data_root argument (from Hydra cfg.data.data_root)
    2. STAGEBRIDGE_DATA_ROOT environment variable
    3. Raise ValueError with a helpful message.
    """
    raw = cfg_data_root or os.environ.get("STAGEBRIDGE_DATA_ROOT")
    if raw:
        return Path(raw)
    raise ValueError(
        "Data root is not configured.  Either:\n"
        "  a) export STAGEBRIDGE_DATA_ROOT=/path/to/data\n"
        "  b) copy configs/data/local.yaml.example → configs/data/local.yaml "
        "     and run with `data=local`"
    )


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

@dataclass
class StageBridgePaths:
    """
    Centralised path resolver for StageBridge artefacts.

    Instantiate via :meth:`from_cfg` (recommended) or directly by providing
    ``data_root`` and ``output_dir``.

    Parameters
    ----------
    data_root:
        Root of the external data directory (outside the repo).
    output_dir:
        Root of the in-repo output directory (default: ``outputs/``).
    """

    data_root: Path
    output_dir: Path

    # ── external data: raw ─────────────────────────────────────────────────

    @property
    def snrna_raw_dir(self) -> Path:
        """GSE308103 snRNA-seq raw download directory."""
        return self.data_root / "data" / "raw" / "geo" / "GSE308103_snrna" / "extracted"

    @property
    def spatial_raw_dir(self) -> Path:
        """GSE307534 Visium raw download / extracted directory."""
        return self.data_root / "data" / "raw" / "geo" / "GSE307534_spatial"

    # ── external data: interim per-sample AnnData ──────────────────────────

    @property
    def snrna_interim_dir(self) -> Path:
        return self.data_root / "interim" / "anndata" / "snrna"

    @property
    def spatial_interim_dir(self) -> Path:
        return self.data_root / "interim" / "anndata" / "spatial"

    def snrna_sample(self, sample_id: str) -> Path:
        return self.snrna_interim_dir / f"{sample_id}.h5ad"

    def spatial_sample(self, sample_id: str) -> Path:
        return self.spatial_interim_dir / f"{sample_id}.h5ad"

    # ── external data: merged processed AnnData ────────────────────────────

    @property
    def snrna_merged(self) -> Path:
        return self.data_root / "processed" / "anndata" / "snrna_merged.h5ad"

    @property
    def spatial_merged(self) -> Path:
        return self.data_root / "processed" / "anndata" / "spatial_merged.h5ad"

    # ── external data: reference atlas ─────────────────────────────────────

    @property
    def hlca_h5ad(self) -> Path:
        return self.data_root / "data" / "reference" / "hlca" / "hlca_full_v1.h5ad"

    # ── in-repo outputs ────────────────────────────────────────────────────

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def figure(self, name: str) -> Path:
        return self.figures_dir / name

    def table(self, name: str) -> Path:
        return self.tables_dir / name

    def checkpoint(self, run_id: str, fold: int, tag: str = "best") -> Path:
        return self.checkpoints_dir / f"{run_id}_fold{fold}_{tag}.pt"

    def history(self, run_id: str, fold: int) -> Path:
        return self.output_dir / f"history_{run_id}_fold{fold}.json"

    def run_manifest(self, run_id: str) -> Path:
        return self.tables_dir / f"run_manifest_{run_id}.json"

    def metrics(self, run_id: str) -> Path:
        return self.tables_dir / f"metrics_{run_id}.json"

    def env_check(self) -> Path:
        return self.tables_dir / "env_check.json"

    def data_audit(self) -> Path:
        return self.tables_dir / "data_audit.json"

    # ── mkdir helpers ───────────────────────────────────────────────────────

    def ensure_output_dirs(self) -> None:
        """Create all in-repo output directories (idempotent)."""
        for d in (self.figures_dir, self.tables_dir, self.checkpoints_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_cfg(cls, cfg: object) -> "StageBridgePaths":
        """
        Build a :class:`StageBridgePaths` from a Hydra ``DictConfig``.

        Looks up ``cfg.data.data_root`` and ``cfg.output_dir``.  Falls back
        to the ``STAGEBRIDGE_DATA_ROOT`` environment variable if
        ``cfg.data.data_root`` is not set.
        """
        data_cfg = getattr(cfg, "data", None)
        raw_data_root: Optional[str] = None
        if data_cfg is not None:
            raw_data_root = str(getattr(data_cfg, "data_root", None) or "").strip() or None

        data_root = _resolve_data_root(raw_data_root)

        raw_output_dir = str(getattr(cfg, "output_dir", None) or "outputs").strip()
        output_dir = Path(raw_output_dir)

        return cls(data_root=data_root, output_dir=output_dir)

    @classmethod
    def from_env(cls, output_dir: str = "outputs") -> "StageBridgePaths":
        """
        Build from the ``STAGEBRIDGE_DATA_ROOT`` environment variable only.
        Useful in scripts that don't use Hydra.
        """
        return cls(data_root=_resolve_data_root(), output_dir=Path(output_dir))
