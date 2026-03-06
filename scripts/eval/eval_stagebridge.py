#!/usr/bin/env python
"""Thin entrypoint wrapper for `stagebridge.workflows.eval.run_evaluation`."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stagebridge.logging_utils import configure_root_logger
from stagebridge.workflows.eval import run_evaluation

configure_root_logger()


@hydra.main(config_path="../../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_evaluation(cfg)))


if __name__ == "__main__":
    main()
