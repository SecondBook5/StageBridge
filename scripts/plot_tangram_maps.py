#!/usr/bin/env python
"""Compatibility wrapper.

Canonical implementation: scripts/pipeline/plot_tangram_maps.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "pipeline/plot_tangram_maps.py"), run_name="__main__")
