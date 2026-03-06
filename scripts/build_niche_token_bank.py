#!/usr/bin/env python
"""Compatibility wrapper.

Canonical implementation: scripts/pipeline/build_niche_token_bank.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "pipeline/build_niche_token_bank.py"), run_name="__main__")
