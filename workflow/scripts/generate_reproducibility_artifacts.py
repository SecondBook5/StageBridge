#!/usr/bin/env python3
"""Generate reproducibility artifacts for publication submission.

Creates:
- Environment snapshot (pip freeze, conda list)
- Config hash for exact reproducibility
- Data checksums
- Code version (git commit)
- Random seed audit
- Hardware info

Required for publication reproducibility standards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_info() -> dict:
    """Get git repository information."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0

        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True
        ).strip()

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "remote_url": remote_url,
        }
    except subprocess.CalledProcessError:
        return {"error": "not_a_git_repo"}


def get_environment_info() -> dict:
    """Get Python environment information."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["cudnn_version"] = str(torch.backends.cudnn.version())
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
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
        import scanpy
        env["scanpy_version"] = scanpy.__version__
    except ImportError:
        pass

    try:
        import scvi
        env["scvi_version"] = scvi.__version__
    except ImportError:
        pass

    return env


def get_pip_freeze() -> list[str]:
    """Get pip freeze output."""
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        return output.strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def get_conda_list() -> list[dict]:
    """Get conda list output."""
    try:
        output = subprocess.check_output(
            ["conda", "list", "--json"], text=True
        )
        return json.loads(output)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return []


def compute_file_checksum(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute file checksum."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_data_checksums(data_dir: Path) -> dict[str, str]:
    """Compute checksums for key data files."""
    checksums = {}
    key_files = [
        "cells.parquet",
        "neighborhoods.parquet",
        "split_manifest.json",
    ]

    for filename in key_files:
        filepath = data_dir / filename
        if filepath.exists():
            checksums[filename] = compute_file_checksum(filepath)

    return checksums


def compute_config_hash(config: dict) -> str:
    """Compute deterministic hash of config for exact reproducibility."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def audit_random_seeds(results_dir: Path) -> dict:
    """Audit that random seeds were properly set across runs."""
    seed_info = {
        "seeds_found": [],
        "configs_checked": 0,
        "seed_consistency": True,
    }

    for config_file in results_dir.glob("**/config.json"):
        with open(config_file) as f:
            config = json.load(f)
        if "seed" in config:
            seed_info["seeds_found"].append(config["seed"])
        seed_info["configs_checked"] += 1

    if seed_info["seeds_found"]:
        seed_info["unique_seeds"] = list(set(seed_info["seeds_found"]))

    return seed_info


def main():
    parser = argparse.ArgumentParser(description="Generate reproducibility artifacts")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, help="Optional config file to hash")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting reproducibility information...")

    artifacts = {
        "generated_at": datetime.now().isoformat(),
        "git": get_git_info(),
        "environment": get_environment_info(),
        "data_checksums": compute_data_checksums(Path(args.data_dir)),
        "seed_audit": audit_random_seeds(Path(args.results_dir)),
    }

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
        artifacts["config_hash"] = compute_config_hash(config)
        artifacts["config"] = config

    artifacts_path = output_dir / "reproducibility_artifacts.json"
    with open(artifacts_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)

    pip_packages = get_pip_freeze()
    pip_path = output_dir / "requirements_frozen.txt"
    pip_path.write_text("\n".join(pip_packages))

    conda_packages = get_conda_list()
    if conda_packages:
        conda_path = output_dir / "conda_environment.json"
        with open(conda_path, "w") as f:
            json.dump(conda_packages, f, indent=2)

    summary_lines = [
        "# Reproducibility Summary",
        "",
        f"Generated: {artifacts['generated_at']}",
        "",
        "## Git",
        f"- Commit: {artifacts['git'].get('commit', 'N/A')}",
        f"- Branch: {artifacts['git'].get('branch', 'N/A')}",
        f"- Clean: {not artifacts['git'].get('dirty', True)}",
        "",
        "## Environment",
        f"- Python: {artifacts['environment'].get('python_version', 'N/A').split()[0]}",
        f"- PyTorch: {artifacts['environment'].get('pytorch_version', 'N/A')}",
        f"- CUDA: {artifacts['environment'].get('cuda_version', 'N/A')}",
        f"- GPUs: {artifacts['environment'].get('gpu_count', 0)}",
        "",
        "## Data Integrity",
    ]

    for filename, checksum in artifacts["data_checksums"].items():
        summary_lines.append(f"- {filename}: {checksum[:16]}...")

    summary_lines.extend([
        "",
        "## Random Seeds",
        f"- Configs checked: {artifacts['seed_audit']['configs_checked']}",
        f"- Unique seeds: {artifacts['seed_audit'].get('unique_seeds', [])}",
    ])

    if "config_hash" in artifacts:
        summary_lines.extend([
            "",
            "## Config Hash",
            f"- Hash: {artifacts['config_hash']}",
        ])

    summary_path = output_dir / "reproducibility_summary.md"
    summary_path.write_text("\n".join(summary_lines))

    print(f"\nArtifacts saved to {output_dir}")
    print(f"  - reproducibility_artifacts.json")
    print(f"  - requirements_frozen.txt")
    print(f"  - reproducibility_summary.md")
    if conda_packages:
        print(f"  - conda_environment.json")


if __name__ == "__main__":
    main()
