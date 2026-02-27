"""
Distributed training utilities for DDP multi-GPU support.

All functions return sensible defaults when not running in distributed mode,
so callers don't need to check is_distributed() before calling.
"""
import os
import random
import logging
import warnings

import numpy as np
import torch
import torch.distributed as dist
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Check if we're running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get global rank. Returns 0 if not distributed."""
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get('RANK', 0))


def get_world_size() -> int:
    """Get world size. Returns 1 if not distributed."""
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank() -> int:
    """Get local rank (GPU index on this node). Returns 0 if not distributed."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> bool:
    """
    Initialize distributed training if launched via torchrun.

    Detects RANK env var (set by torchrun), calls dist.init_process_group('nccl'),
    and sets torch.cuda.set_device(local_rank).

    Returns:
        True if distributed training was initialized, False for single-GPU.
    """
    if 'RANK' not in os.environ:
        return False

    if not torch.cuda.is_available():
        warnings.warn(
            "RANK env var set but CUDA not available. "
            "Falling back to single-process training."
        )
        return False

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl')

    if is_main_process():
        logger.info(
            f"Distributed training initialized: "
            f"world_size={get_world_size()}, backend=nccl"
        )

    return True


def cleanup_distributed():
    """Destroy the process group if distributed training is active."""
    if is_distributed():
        dist.destroy_process_group()


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast a tensor from src rank to all ranks.

    No-op if not distributed.
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def all_reduce_dict(
    metrics: Dict[str, float],
    device: torch.device,
    op: str = 'avg',
) -> Dict[str, float]:
    """
    Reduce a Dict[str, float] across all DDP ranks.

    Args:
        metrics: Dictionary of metric name -> value
        device: Device to create tensors on
        op: 'avg' to average across ranks, 'sum' to sum

    Returns:
        Reduced metrics dictionary. No-op if not distributed.
    """
    if not is_distributed():
        return metrics

    world_size = get_world_size()
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)

    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    if op == 'avg':
        values /= world_size

    return {k: v.item() for k, v in zip(keys, values)}


def barrier():
    """Synchronize all processes. No-op if not distributed."""
    if is_distributed():
        dist.barrier()


def seed_everything(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
