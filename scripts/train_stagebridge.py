#!/usr/bin/env python
"""Compatibility wrapper.

Canonical implementation: scripts/train/train_stagebridge.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "train/train_stagebridge.py"), run_name="__main__")
