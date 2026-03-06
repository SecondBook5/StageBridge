#!/usr/bin/env python
"""Thin entrypoint wrapper for `stagebridge.workflows.pipeline.map_hlca`."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stagebridge.logging_utils import configure_root_logger
from stagebridge.workflows.pipeline import map_hlca

configure_root_logger()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(json.dumps(map_hlca(cfg)))


if __name__ == "__main__":
    main()
