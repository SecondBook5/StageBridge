#!/usr/bin/env python
"""Run the full label-repair workflow and generate reports."""
from __future__ import annotations

from stagebridge.notebook_api import compose_config
from stagebridge.pipelines.run_label_repair import run_label_repair


if __name__ == "__main__":
    cfg = compose_config(overrides=["labels=repair"])
    run_label_repair(cfg)
