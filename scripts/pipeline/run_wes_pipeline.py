#!/usr/bin/env python3
"""Extract WES features from GSE307529 VCF tar and save as parquet.

Usage
-----
python scripts/run_wes_pipeline.py

The script reads STAGEBRIDGE_DATA_ROOT from environment (or configs/data/local.yaml
if present) and writes:

    $DATA_ROOT/processed/features/wes_features.parquet

This file is later loaded by the StageBridge trainer when use_wes_features=true.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add repo root to path so stagebridge package is importable
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
log = logging.getLogger("run_wes_pipeline")


def _resolve_data_root() -> Path:
    data_root = os.environ.get("STAGEBRIDGE_DATA_ROOT")
    if data_root:
        return Path(data_root)
    local_cfg = _REPO / "configs" / "data" / "local.yaml"
    if local_cfg.exists():
        import yaml  # type: ignore[import]
        with open(local_cfg) as fh:
            cfg = yaml.safe_load(fh)
        root = cfg.get("data_root") or cfg.get("paths", {}).get("data_root")
        if root:
            return Path(root)
    raise RuntimeError(
        "Cannot find data root. Set STAGEBRIDGE_DATA_ROOT env var or create "
        "configs/data/local.yaml with a data_root key."
    )


def main() -> None:
    data_root = _resolve_data_root()
    tar_path = data_root / "raw" / "geo" / "GSE307529_wes" / "downloads" / "GSE307529_RAW.tar"
    out_path = data_root / "processed" / "features" / "wes_features.parquet"

    if not tar_path.exists():
        log.error("WES tar not found: %s", tar_path)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Parsing WES features from %s …", tar_path)
    from stagebridge.io.geo_wes import parse_wes_features_from_tar

    df = parse_wes_features_from_tar(tar_path)

    log.info("Parsed %d (patient, stage) rows", len(df))
    log.info("\n%s", df.to_string(index=False))

    df.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
