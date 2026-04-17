"""Path configuration for StageBridge - supports local and HPC environments.

Usage:
    from stagebridge.config.paths import get_paths

    paths = get_paths()  # Auto-detects environment
    paths = get_paths("hpc")  # Force HPC paths
    paths = get_paths("local")  # Force local paths

    print(paths.hlca_dir)
    print(paths.luca_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PathConfig:
    """Path configuration for a specific environment."""

    data_root: Path
    hlca_dir: Path
    luca_dir: Path
    evo_dir: Path
    output_dir: Path
    results_dir: Path
    figures_dir: Path

    def __post_init__(self):
        # Expand ~ and convert to Path objects
        self.data_root = Path(self.data_root).expanduser()
        self.hlca_dir = Path(self.hlca_dir).expanduser()
        self.luca_dir = Path(self.luca_dir).expanduser()
        self.evo_dir = Path(self.evo_dir).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        self.results_dir = Path(self.results_dir).expanduser()
        self.figures_dir = Path(self.figures_dir).expanduser()

    @property
    def hlca_path(self) -> Path | None:
        """Get HLCA h5ad file path if it exists."""
        candidates = [
            self.hlca_dir / "hlca_scanvi.h5ad",
            self.hlca_dir / "hlca_core.h5ad",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @property
    def luca_path(self) -> Path | None:
        """Get LuCA h5ad file path if it exists."""
        candidates = [
            self.luca_dir / "luca_extended.h5ad",
            self.luca_dir / "f678fb47-e51b-4dc5-b23f-f9df43a67ee5.h5ad",
            self.luca_dir / "luca_core.h5ad",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @property
    def evo_path(self) -> Path | None:
        """Get evolutionary snRNA-seq h5ad file path if it exists."""
        candidates = [
            self.evo_dir / "snrna_merged.h5ad",
            self.evo_dir / "cells.h5ad",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, str]:
        """Check status of all data files."""
        return {
            "hlca": f"✓ {self.hlca_path.name}" if self.hlca_path else "✗ Not found",
            "luca": f"✓ {self.luca_path.name}" if self.luca_path else "✗ Not found",
            "evo": f"✓ {self.evo_path.name}" if self.evo_path else "✗ Not found",
        }


def detect_environment() -> Literal["local", "hpc"]:
    """Auto-detect environment based on hostname or env var."""
    # Check explicit environment variable first
    env = os.environ.get("STAGEBRIDGE_ENV", "").lower()
    if env in ("local", "hpc"):
        return env

    # Check if on HPC by looking for data1 directory
    if Path("/data1/chaunzt1").exists():
        return "hpc"

    # Check hostname
    import socket

    hostname = socket.gethostname().lower()
    if "mskcc" in hostname or "lilac" in hostname or "login" in hostname:
        return "hpc"

    return "local"


def get_paths(env: Literal["local", "hpc"] | None = None) -> PathConfig:
    """Get path configuration for the specified or detected environment.

    Args:
        env: Environment name ("local" or "hpc"). If None, auto-detects.

    Returns:
        PathConfig with all paths for the environment.
    """
    if env is None:
        env = detect_environment()

    # Find config file
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "paths.yaml",
        Path("config/paths.yaml"),
        Path.home() / ".stagebridge" / "paths.yaml",
    ]

    config_file = None
    for p in config_paths:
        if p.exists():
            config_file = p
            break

    if config_file is None:
        # Use defaults if no config file
        if env == "hpc":
            return PathConfig(
                data_root="/data1/chaunzt1/stagebridge/processed",
                hlca_dir="/data1/chaunzt1/stagebridge/processed/HLCA",
                luca_dir="/data1/chaunzt1/stagebridge/processed/LuCA",
                evo_dir="/data1/chaunzt1/stagebridge/processed/luad_evo",
                output_dir="/data1/chaunzt1/stagebridge/outputs/v1_demo",
                results_dir="/data1/chaunzt1/stagebridge/results",
                figures_dir="/data1/chaunzt1/stagebridge/figures",
            )
        else:
            return PathConfig(
                data_root=Path.home() / "data" / "stagebridge" / "processed",
                hlca_dir=Path.home() / "data" / "stagebridge" / "processed" / "HLCA",
                luca_dir=Path.home() / "data" / "stagebridge" / "processed" / "LuCA",
                evo_dir=Path.home() / "data" / "stagebridge" / "processed" / "luad_evo",
                output_dir=Path("data/outputs/v1_demo"),
                results_dir=Path("results"),
                figures_dir=Path("figures"),
            )

    # Load from config file
    with open(config_file) as f:
        config = yaml.safe_load(f)

    env_config = config.get(env, {})
    return PathConfig(
        data_root=env_config.get("data_root", ""),
        hlca_dir=env_config.get("hlca_dir", ""),
        luca_dir=env_config.get("luca_dir", ""),
        evo_dir=env_config.get("evo_dir", ""),
        output_dir=env_config.get("output_dir", ""),
        results_dir=env_config.get("results_dir", ""),
        figures_dir=env_config.get("figures_dir", ""),
    )


def print_status():
    """Print current environment and data status."""
    env = detect_environment()
    paths = get_paths(env)

    print(f"Environment: {env.upper()}")
    print(f"Data root: {paths.data_root}")
    print()
    print("Data files:")
    for name, status in paths.status().items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    print_status()
