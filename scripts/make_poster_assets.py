#!/usr/bin/env python
"""Compatibility wrapper.

Canonical implementation: scripts/viz/make_poster_assets.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "viz/make_poster_assets.py"), run_name="__main__")
