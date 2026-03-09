#!/usr/bin/env python
"""Derive refined lesion labels and continuous risk scores."""
from __future__ import annotations

from stagebridge.labels.cohort_manifest import build_cleaned_cohort_manifest
from stagebridge.notebook_api import compose_config
from stagebridge.pipelines.run_label_repair import run_label_refinement


if __name__ == "__main__":
    cfg = compose_config(overrides=["labels=repair"])
    manifest_outputs = build_cleaned_cohort_manifest(cfg)
    run_label_refinement(cfg, cached=manifest_outputs)
