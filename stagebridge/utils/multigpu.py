"""Multi-GPU training utilities for StageBridge.

Provides consistent GPU configuration across all training pipelines:
- scVI/scANVI reference model training
- StageBridge transformer training
- Any PyTorch Lightning-based training

Usage:
    from stagebridge.utils.multigpu import (
        setup_distributed_env,
        get_accelerator_config,
        get_lightning_trainer_kwargs,
        get_device,
    )

    # For scvi-tools
    model.train(**get_lightning_trainer_kwargs(max_epochs=400))

    # For PyTorch Lightning Trainer
    trainer = pl.Trainer(**get_lightning_trainer_kwargs())

    # For raw PyTorch
    device = get_device()
    model = model.to(device)
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch


def setup_distributed_env(
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> dict[str, str]:
    """Setup environment variables for distributed training.

    Call this at the start of your script before any GPU operations.

    Parameters
    ----------
    master_addr : str
        Master node address (default: localhost for single-node)
    master_port : str
        Master port for communication

    Returns
    -------
    dict
        Environment variables that were set
    """
    env_vars = {}

    # CUDA device visibility
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            cuda_devices = ",".join(str(i) for i in range(n_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            env_vars["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Distributed training coordination
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    env_vars["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
    env_vars["MASTER_PORT"] = os.environ["MASTER_PORT"]

    # Disable tokenizer parallelism (avoids warnings)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # For better NCCL performance on some systems
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    return env_vars


def get_gpu_info() -> dict[str, Any]:
    """Get information about available GPUs.

    Returns
    -------
    dict with:
        - available: bool
        - count: int
        - names: list[str]
        - memory_gb: list[float]
        - cuda_version: str
    """
    info = {
        "available": torch.cuda.is_available(),
        "count": 0,
        "names": [],
        "memory_gb": [],
        "cuda_version": None,
    }

    if not info["available"]:
        return info

    info["count"] = torch.cuda.device_count()
    info["cuda_version"] = torch.version.cuda

    for i in range(info["count"]):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / 1e9)

    return info


def get_device(prefer_gpu: bool = True, gpu_index: int = 0) -> torch.device:
    """Get the appropriate torch device.

    Parameters
    ----------
    prefer_gpu : bool
        Whether to prefer GPU if available
    gpu_index : int
        Which GPU to use if multiple available

    Returns
    -------
    torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        if gpu_index < torch.cuda.device_count():
            return torch.device(f"cuda:{gpu_index}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_accelerator_config(
    n_gpus: int | None = None,
    strategy: Literal["auto", "ddp", "ddp_spawn", "dp", "deepspeed"] = "auto",
) -> dict[str, Any]:
    """Get PyTorch Lightning accelerator configuration.

    Handles the common SLURM + Lightning DDP configuration issues.

    Parameters
    ----------
    n_gpus : int, optional
        Number of GPUs to use. None = auto-detect.
    strategy : str
        DDP strategy. "auto" picks the best for your environment:
        - Single GPU: no strategy needed
        - Multi-GPU + SLURM ntasks=1: ddp_spawn (Lightning spawns processes)
        - Multi-GPU + SLURM ntasks>1: ddp (SLURM manages processes)

    Returns
    -------
    dict
        Config to pass to Lightning Trainer or scvi model.train()

    Examples
    --------
    >>> # For scvi-tools
    >>> config = get_accelerator_config()
    >>> model.train(max_epochs=100, **config)

    >>> # For Lightning Trainer
    >>> config = get_accelerator_config(n_gpus=4, strategy="ddp_spawn")
    >>> trainer = pl.Trainer(**config)
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    # CPU training
    if n_gpus == 0 or not torch.cuda.is_available():
        return {"accelerator": "cpu"}

    # Single GPU - simple case
    if n_gpus == 1:
        return {
            "accelerator": "gpu",
            "devices": 1,
        }

    # Multi-GPU - need to pick strategy
    if strategy == "auto":
        # Check SLURM environment
        slurm_ntasks = os.environ.get("SLURM_NTASKS_PER_NODE", "1")
        try:
            ntasks = int(slurm_ntasks)
        except ValueError:
            ntasks = 1

        if ntasks > 1:
            # SLURM is managing processes - use ddp
            # Each task should see devices=1
            return {
                "accelerator": "gpu",
                "devices": 1,
                "strategy": "ddp",
            }
        else:
            # Single SLURM task - Lightning must spawn processes
            # ddp_spawn works better than ddp in this case
            strategy = "ddp_spawn"

    return {
        "accelerator": "gpu",
        "devices": n_gpus,
        "strategy": strategy,
    }


def get_lightning_trainer_kwargs(
    max_epochs: int = 100,
    n_gpus: int | None = None,
    strategy: str = "auto",
    early_stopping: bool = True,
    early_stopping_patience: int = 15,
    early_stopping_monitor: str = "val_loss",
    gradient_clip_val: float | None = 1.0,
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32",
    **extra_kwargs,
) -> dict[str, Any]:
    """Get complete kwargs for Lightning Trainer or scvi model.train().

    This is the main function to use - it combines accelerator config
    with common training settings.

    Parameters
    ----------
    max_epochs : int
        Maximum training epochs
    n_gpus : int, optional
        Number of GPUs (auto-detect if None)
    strategy : str
        DDP strategy ("auto", "ddp", "ddp_spawn", etc.)
    early_stopping : bool
        Whether to use early stopping
    early_stopping_patience : int
        Patience for early stopping
    early_stopping_monitor : str
        Metric to monitor for early stopping
    gradient_clip_val : float, optional
        Gradient clipping value (None to disable)
    precision : str
        Training precision ("32", "16-mixed", "bf16-mixed")
    **extra_kwargs
        Additional kwargs passed through

    Returns
    -------
    dict
        Complete kwargs for Trainer or model.train()

    Examples
    --------
    >>> # scvi-tools
    >>> from stagebridge.utils.multigpu import get_lightning_trainer_kwargs
    >>> model.train(**get_lightning_trainer_kwargs(max_epochs=400))

    >>> # PyTorch Lightning
    >>> import pytorch_lightning as pl
    >>> trainer = pl.Trainer(**get_lightning_trainer_kwargs(max_epochs=100))
    """
    # Get accelerator config
    config = get_accelerator_config(n_gpus=n_gpus, strategy=strategy)

    # Add training settings
    config["max_epochs"] = max_epochs

    if early_stopping:
        config["early_stopping"] = True
        config["early_stopping_patience"] = early_stopping_patience
        config["early_stopping_monitor"] = early_stopping_monitor

    if gradient_clip_val is not None:
        config["gradient_clip_val"] = gradient_clip_val

    # Precision (for mixed precision training)
    if precision != "32":
        config["precision"] = precision

    # Common useful settings
    config.setdefault("enable_progress_bar", True)
    config.setdefault("check_val_every_n_epoch", 1)

    # Merge extra kwargs
    config.update(extra_kwargs)

    return config


def get_scvi_train_kwargs(
    max_epochs: int = 400,
    n_gpus: int | None = None,
    train_size: float = 0.9,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **extra_kwargs,
) -> dict[str, Any]:
    """Get kwargs specifically for scvi-tools model.train().

    Includes scvi-specific settings like plan_kwargs.

    Parameters
    ----------
    max_epochs : int
        Maximum epochs
    n_gpus : int, optional
        Number of GPUs
    train_size : float
        Fraction for training (rest is validation)
    lr : float
        Learning rate
    weight_decay : float
        Weight decay

    Returns
    -------
    dict
        Kwargs for scvi model.train()

    Examples
    --------
    >>> from stagebridge.utils.multigpu import get_scvi_train_kwargs
    >>> model.train(**get_scvi_train_kwargs(max_epochs=400))
    """
    config = get_accelerator_config(n_gpus=n_gpus, strategy="auto")

    config.update({
        "max_epochs": max_epochs,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_validation",
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.5,
        "check_val_every_n_epoch": 1,
        "train_size": train_size,
        "enable_progress_bar": True,
        "plan_kwargs": {
            "lr": lr,
            "weight_decay": weight_decay,
        },
    })

    config.update(extra_kwargs)
    return config


def print_gpu_status():
    """Print GPU status summary."""
    info = get_gpu_info()

    print("=" * 50)
    print("GPU Status")
    print("=" * 50)

    if not info["available"]:
        print("No CUDA GPUs available")
        return

    print(f"CUDA Version: {info['cuda_version']}")
    print(f"GPU Count: {info['count']}")
    print()

    for i, (name, mem) in enumerate(zip(info["names"], info["memory_gb"])):
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")

    print()
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 50)


# Convenience exports
__all__ = [
    "setup_distributed_env",
    "get_gpu_info",
    "get_device",
    "get_accelerator_config",
    "get_lightning_trainer_kwargs",
    "get_scvi_train_kwargs",
    "print_gpu_status",
]
