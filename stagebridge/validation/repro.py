"""
Reproducibility validation for StageBridge.

Ensures runs can be exactly reproduced given the same config and seed.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def capture_environment() -> dict[str, Any]:
    """
    Capture full environment for reproducibility.

    Returns
    -------
    dict
        Environment snapshot
    """
    env = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": sys.platform,
    }

    # Git info
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        env["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        env["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        env["git_commit"] = "unknown"
        env["git_branch"] = "unknown"
        env["git_dirty"] = None

    # Key packages
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        env["pandas_version"] = pandas.__version__
    except ImportError:
        pass

    try:
        import anndata
        env["anndata_version"] = anndata.__version__
    except ImportError:
        pass

    try:
        import scvi
        env["scvi_version"] = scvi.__version__
    except ImportError:
        pass

    return env


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic hash of config for reproducibility tracking."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def save_repro_manifest(
    config: dict[str, Any],
    seed: int,
    output_dir: Path,
    extra_info: dict[str, Any] | None = None,
) -> Path:
    """
    Save reproducibility manifest for a run.

    Parameters
    ----------
    config : dict
        Run configuration
    seed : int
        Random seed used
    output_dir : Path
        Directory to save manifest
    extra_info : dict, optional
        Additional info to include

    Returns
    -------
    Path
        Path to saved manifest
    """
    manifest = {
        "seed": seed,
        "config_hash": compute_config_hash(config),
        "config": config,
        "environment": capture_environment(),
    }

    if extra_info:
        manifest["extra"] = extra_info

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "repro_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info(f"Saved reproducibility manifest to {manifest_path}")
    return manifest_path


def verify_reproducibility(
    manifest_path: Path,
    current_config: dict[str, Any],
    current_seed: int,
) -> dict[str, Any]:
    """
    Verify that current config matches saved manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to saved manifest
    current_config : dict
        Current configuration
    current_seed : int
        Current seed

    Returns
    -------
    dict
        Verification report
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    report = {
        "reproducible": True,
        "issues": [],
    }

    # Check seed
    if manifest["seed"] != current_seed:
        report["reproducible"] = False
        report["issues"].append(f"Seed mismatch: saved={manifest['seed']}, current={current_seed}")

    # Check config hash
    current_hash = compute_config_hash(current_config)
    if manifest["config_hash"] != current_hash:
        report["reproducible"] = False
        report["issues"].append(f"Config hash mismatch: saved={manifest['config_hash']}, current={current_hash}")

    # Check critical packages
    current_env = capture_environment()
    saved_env = manifest.get("environment", {})

    for key in ["torch_version", "numpy_version"]:
        if key in saved_env and key in current_env:
            if saved_env[key] != current_env[key]:
                report["issues"].append(f"{key} changed: {saved_env[key]} -> {current_env[key]}")

    # Git check
    if saved_env.get("git_commit") != current_env.get("git_commit"):
        report["issues"].append(
            f"Git commit changed: {saved_env.get('git_commit', 'unknown')[:8]} -> "
            f"{current_env.get('git_commit', 'unknown')[:8]}"
        )

    return report


def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    log.info(f"Set all random seeds to {seed}")


def create_run_id(config: dict[str, Any], seed: int) -> str:
    """Create unique but reproducible run ID."""
    config_hash = compute_config_hash(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{config_hash[:8]}_s{seed}"
