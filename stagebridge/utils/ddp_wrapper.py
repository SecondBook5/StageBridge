"""DDP wrapper for multi-GPU training without refactoring existing code."""

from __future__ import annotations

import os
import functools
from typing import Callable, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup_ddp(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def wrap_model_ddp(model: torch.nn.Module, device_id: int) -> DDP:
    """Wrap model with DDP."""
    model = model.to(device_id)
    return DDP(model, device_ids=[device_id], output_device=device_id)


def wrap_dataloader_ddp(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    **kwargs,
) -> tuple[DataLoader, DistributedSampler]:
    """Create DDP-aware DataLoader with DistributedSampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        **kwargs,
    )

    return loader, sampler


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if op == "mean":
        rt /= get_world_size()

    return rt


class DDPContext:
    """Context manager for DDP training."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def __enter__(self):
        setup_ddp(self.rank, self.world_size)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_ddp()
        return False


def run_ddp(
    fn: Callable,
    world_size: int = None,
    args: tuple = (),
    kwargs: dict = None,
):
    """
    Run function with DDP across multiple GPUs.

    Usage:
        def train(rank, world_size, data_dir, ...):
            with DDPContext(rank, world_size):
                model = MyModel().to(rank)
                model = wrap_model_ddp(model, rank)
                ...

        run_ddp(train, world_size=4, kwargs={"data_dir": "/path"})
    """
    import torch.multiprocessing as mp

    if world_size is None:
        world_size = torch.cuda.device_count()

    if kwargs is None:
        kwargs = {}

    mp.spawn(
        _ddp_worker,
        args=(world_size, fn, args, kwargs),
        nprocs=world_size,
        join=True,
    )


def _ddp_worker(rank: int, world_size: int, fn: Callable, args: tuple, kwargs: dict):
    """Worker function for mp.spawn."""
    fn(rank, world_size, *args, **kwargs)
