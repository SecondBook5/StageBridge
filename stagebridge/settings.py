"""Global settings for StageBridge.

Provides user-configurable defaults for logging, computation, and visualization.
Follows scanpy/scvi-tools patterns for publication-quality tools.

Usage:
    import stagebridge as sb

    # Set verbosity (0=error, 1=warning, 2=info, 3=debug)
    sb.settings.verbosity = 1

    # Set default device
    sb.settings.device = "cuda"

    # Set number of parallel jobs
    sb.settings.n_jobs = 4

    # Configure plot defaults
    sb.settings.figure_dir = "./figures"
    sb.settings.autosave = True
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _get_default_cache_dir() -> Path:
    """Get default cache directory, respecting XDG spec."""
    if "STAGEBRIDGE_CACHE_DIR" in os.environ:
        return Path(os.environ["STAGEBRIDGE_CACHE_DIR"])
    xdg_cache = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg_cache) / "stagebridge"


def _get_default_device() -> str:
    """Detect default device."""
    if "STAGEBRIDGE_DEVICE" in os.environ:
        return os.environ["STAGEBRIDGE_DEVICE"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class StageBridgeSettings:
    """Global settings for StageBridge.

    Attributes
    ----------
    verbosity
        Logging verbosity level:
        - 0: errors only
        - 1: warnings
        - 2: info (default)
        - 3: debug
    n_jobs
        Number of parallel jobs for data loading (-1 = all CPUs)
    device
        Default device for model inference ("auto", "cuda", "cpu")
    cache_dir
        Directory for caching downloaded models and data
    figure_dir
        Default directory for saving figures
    autosave
        Whether to automatically save figures
    plot_suffix
        Suffix added to saved figure filenames
    autoshow
        Whether to automatically show figures
    dpi
        Default DPI for figures
    figsize
        Default figure size (width, height) in inches

    Examples
    --------
    >>> import stagebridge as sb
    >>> sb.settings.verbosity = 1  # Only warnings and errors
    >>> sb.settings.device = "cuda"
    >>> sb.settings.figure_dir = "./my_figures"
    """

    # Logging
    verbosity: int = 2

    # Computation
    n_jobs: int = 1
    device: str = field(default_factory=_get_default_device)
    batch_size: int = 256

    # Caching
    cache_dir: Path = field(default_factory=_get_default_cache_dir)

    # Visualization
    figure_dir: Path = field(default_factory=lambda: Path("./figures"))
    autosave: bool = False
    plot_suffix: str = ""
    autoshow: bool = True
    dpi: int = 150
    figsize: tuple[float, float] = (6.0, 4.0)

    # Model registry
    model_registry_url: str = "https://github.com/SecondBook5/StageBridge/releases/download"

    def __post_init__(self):
        """Convert paths and validate settings."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.figure_dir, str):
            self.figure_dir = Path(self.figure_dir)

    @property
    def log_level(self) -> int:
        """Convert verbosity to logging level."""
        levels = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
        }
        return levels.get(self.verbosity, logging.INFO)

    def reset(self) -> None:
        """Reset all settings to defaults."""
        defaults = StageBridgeSettings()
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(defaults, field_name))


# Global settings instance
settings = StageBridgeSettings()


def setup_logging(verbosity: int | None = None) -> logging.Logger:
    """Configure logging for StageBridge.

    Parameters
    ----------
    verbosity
        Override settings.verbosity if provided

    Returns
    -------
    Logger instance for stagebridge
    """
    level = settings.log_level if verbosity is None else {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }.get(verbosity, logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root stagebridge logger
    logger = logging.getLogger("stagebridge")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Add stream handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a stagebridge submodule.

    Parameters
    ----------
    name
        Module name (e.g., "stagebridge.api")

    Returns
    -------
    Logger instance
    """
    logger = logging.getLogger(name)
    # Inherit level from root if not set
    if logger.level == logging.NOTSET:
        logger.setLevel(settings.log_level)
    return logger


# Initialize logging on import
_root_logger = setup_logging()
