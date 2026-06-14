# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Distributed training utilities."""

import os
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize distributed training.

    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi').
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and compute the mean across processes.

    Args:
        tensor: Tensor to reduce.

    Returns:
        Reduced tensor (mean across processes).
    """
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    return tensor


def all_gather_object(obj: Any) -> list[Any]:
    """Gather an object from all processes.

    Args:
        obj: Object to gather.

    Returns:
        List of objects from all processes.
    """
    if not is_distributed():
        return [obj]

    world_size = get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()
