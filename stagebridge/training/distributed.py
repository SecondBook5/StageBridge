"""Distributed training utilities for DDP."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed process group.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: "mean" or "sum"

    Returns:
        Reduced tensor (only valid on rank 0)
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if op == "mean":
        rt = rt / get_world_size()

    return rt
