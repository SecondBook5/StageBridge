#!/usr/bin/env python
"""Run the configured clonal backend or parse existing summaries."""
from __future__ import annotations

from stagebridge.labels.cohort_manifest import build_cleaned_cohort_manifest
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines.run_label_repair import run_label_clonal


if __name__ == "__main__":
    cfg = compose_config(overrides=["labels=repair"])
    manifest = build_cleaned_cohort_manifest(cfg)["cleaned_manifest"]
    run_label_clonal(cfg, manifest=manifest)
