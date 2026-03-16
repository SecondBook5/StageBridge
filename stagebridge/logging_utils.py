"""
Centralised logging configuration for StageBridge.

Usage (at the top of each module or in the notebook):
    from stagebridge.logging_utils import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_root_configured = False


def configure_root_logger(level: int = logging.INFO) -> None:
    """Configure the root logger once for the whole process.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _root_configured
    if _root_configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _root_configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger, ensuring the root is configured."""
    configure_root_logger()
    return logging.getLogger(name)
