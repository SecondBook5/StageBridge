#!/usr/bin/env python
"""Compatibility wrapper.

Canonical implementation: scripts/eval/check_env.py
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "eval/check_env.py"), run_name="__main__")
