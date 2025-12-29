"""
Training Script for Hierarchical LNN

Trains the 3-layer hierarchical Liquid Neural Network on 1-min data.

Usage:
    python train_hierarchical.py --epochs 100 --batch_size 64 --device cuda

Features:
- Trains on 1-min data (2015-2022)
- Validates on held-out test set (2023+)
- Early stopping based on validation loss
- Saves best model checkpoint
- Supports both lazy and preload modes
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import socket

import functools
from datetime import datetime
import json
import platform
import time
import threading
from typing import Dict, Tuple
from tqdm import tqdm

# DDP (DistributedDataParallel) imports for multi-GPU training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp  # For spawning DDP processes without torchrun

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from src.ml.hierarchical_model import HierarchicalLNN
from src.ml.hierarchical_dataset import create_hierarchical_dataset
from src.ml.features import TradingFeatureExtractor, FEATURE_VERSION, load_vix_data
from src.ml.data_feed import CSVDataFeed
# v6.0: Import duration-primary loss functions
from src.ml.loss_v6 import (
    compute_v6_loss, V6LossConfig, get_warmup_weight, get_temperature, format_loss_log
)
import yaml
import config as project_config


def get_hardware_info():
    """Detect available compute devices and hardware specs (including multi-GPU)."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'cpu_count': torch.get_num_threads(),
        'platform': platform.system()
    }

    if info['cuda_available']:
        # Multi-GPU detection
        num_gpus = torch.cuda.device_count()
        info['cuda_device_count'] = num_gpus
        info['cuda_devices'] = []
        total_vram = 0

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            info['cuda_devices'].append({
                'index': i,
                'name': gpu_name,
                'vram_gb': gpu_vram
            })
            total_vram += gpu_vram

        # Summary fields for backward compatibility
        info['cuda_device'] = info['cuda_devices'][0]['name'] if info['cuda_devices'] else 'Unknown'
        info['cuda_memory_gb'] = info['cuda_devices'][0]['vram_gb'] if info['cuda_devices'] else 0
        info['cuda_total_memory_gb'] = total_vram

    if info['mps_available']:
        info['mac_chip'] = platform.processor() or "Apple Silicon"
        # Estimate RAM (rough approximation)
        import psutil
        info['total_ram_gb'] = psutil.virtual_memory().total / 1e9

    return info


def get_best_device():
    """Auto-detect best available compute device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # Apple Silicon
    else:
        return 'cpu'


def get_recommended_batch_size(device: str, total_ram_gb: float = 16):
    """Get recommended batch size for device (v3.20: Multi-GPU support)."""
    recommendations = {
        'cuda_multi': 512,   # Multi-GPU (2+ GPUs with 40GB+ each) - split across GPUs
        'cuda_high': 256,    # Single GPU with 64GB+ VRAM (e.g., A100 80GB, H100)
        'cuda': 64,          # NVIDIA GPU (standard, reduced for 14K features)
        'mps_high': 8,       # M2 Max/Ultra with 64+ GB (very conservative, num_workers=0 + from_numpy fix)
        'mps_mid': 4,        # M2 Pro/M1 Max with 32-64 GB (4 is safer with memory efficiency fixes)
        'mps_low': 2,        # M1/M1 Pro with 16-32 GB (very safe, only 2 samples per batch)
        'cpu': 16
    }

    if device == 'cuda':
        # Check for multi-GPU and VRAM
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                # Get minimum VRAM across all GPUs (limiting factor for DataParallel)
                min_vram = min(
                    torch.cuda.get_device_properties(i).total_memory / 1e9
                    for i in range(num_gpus)
                )

                if num_gpus > 1 and min_vram >= 40:
                    # Multi-GPU with good VRAM per GPU (e.g., 2x A40 45GB)
                    return recommendations['cuda_multi']
                elif min_vram >= 64:
                    # Single high-VRAM GPU
                    return recommendations['cuda_high']
        except Exception:
            pass  # Fall through to standard cuda
        return recommendations['cuda']

    if device == 'mps':
        if total_ram_gb >= 64:
            return recommendations['mps_high']
        elif total_ram_gb >= 32:
            return recommendations['mps_mid']
        else:
            return recommendations['mps_low']

    return recommendations.get(device, 16)


# NOTE: fix_ncps_buffers() was removed - it was a workaround for DataParallel
# which is no longer used. Single-GPU mode and DDP don't need this fix.


class ShuffleBufferSampler(torch.utils.data.Sampler):
    """
    Shuffle buffer sampler for memory-efficient training with mmap data.

    Instead of full random shuffle (which causes random mmap access = page faults),
    this sampler:
    1. Divides indices into sequential chunks (buffer_size)
    2. Shuffles indices WITHIN each chunk only
    3. Yields indices in chunk order

    Result: Access pattern is sequential-ish (good for mmap), but with local
    randomization (good for training quality).

    Example with buffer_size=1000:
        Chunk 1: indices 0-999, shuffled → yields e.g. [234, 12, 891, ...]
        Chunk 2: indices 1000-1999, shuffled → yields e.g. [1456, 1023, ...]
        ...

    Performance vs full shuffle:
        - Full shuffle: O(random) mmap access → severe page faults
        - Shuffle buffer: O(sequential) mmap access → cache-friendly
        - Training quality: ~95% of full shuffle (local randomization)
    """

    def __init__(self, data_source, buffer_size: int = 10000, seed: int = None):
        """
        Args:
            data_source: Dataset to sample from (needs __len__)
            buffer_size: Number of samples per chunk (default 10000 ≈ 50MB overhead)
            seed: Random seed for reproducibility (None = random each epoch)
        """
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.seed = seed
        self._epoch = 0  # For distributed training compatibility

    def __iter__(self):
        import random
        n = len(self.data_source)
        indices = list(range(n))

        # Use epoch-based seed for reproducibility across epochs
        if self.seed is not None:
            rng = random.Random(self.seed + self._epoch)
        else:
            rng = random.Random()

        # Process in sequential chunks, shuffle within each chunk
        for chunk_start in range(0, n, self.buffer_size):
            chunk_end = min(chunk_start + self.buffer_size, n)
            chunk = indices[chunk_start:chunk_end]
            rng.shuffle(chunk)  # Local shuffle only
            yield from chunk

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling (DDP compatibility)."""
        self._epoch = epoch


class DistributedShuffleBufferSampler(torch.utils.data.Sampler):
    """
    Distributed version of ShuffleBufferSampler for multi-GPU DDP training.

    Combines the benefits of:
    - DistributedSampler: Splits data across GPUs so each processes different samples
    - ShuffleBufferSampler: Sequential chunk access with local shuffling (cache-friendly)

    When data is preloaded to RAM, random access is fast so the sequential access
    benefit is reduced. However, this sampler still provides:
    - Better CPU cache locality (accessing nearby indices)
    - Reproducible chunk-based shuffling across ranks

    v5.9.4: Added as user-selectable option when preload_tf_to_ram=True.
    """

    def __init__(
        self,
        data_source,
        num_replicas: int,
        rank: int,
        buffer_size: int = 10000,
        seed: int = 42,
        drop_last: bool = True
    ):
        """
        Args:
            data_source: Dataset to sample from (needs __len__)
            num_replicas: Number of distributed processes (world_size)
            rank: Rank of current process (0 to num_replicas-1)
            buffer_size: Number of samples per chunk for local shuffling
            seed: Random seed for reproducibility
            drop_last: Drop last incomplete batch to ensure equal samples per GPU
        """
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.buffer_size = buffer_size
        self.seed = seed
        self.drop_last = drop_last
        self._epoch = 0

        # Calculate samples per replica
        total_size = len(data_source)
        if drop_last:
            # Make total evenly divisible by num_replicas
            self.total_size = (total_size // num_replicas) * num_replicas
        else:
            self.total_size = total_size

        self.num_samples = self.total_size // num_replicas

    def __iter__(self):
        import random

        # Generate global indices with epoch-based seed (same across all ranks)
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)

        # Create indices and do a global shuffle (same order on all ranks)
        indices = list(range(len(self.data_source)))
        indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        # Truncate to total_size if drop_last
        if self.drop_last:
            indices = indices[:self.total_size]

        # Split indices across ranks - each rank gets every num_replicas-th sample
        # This ensures different ranks get different samples
        rank_indices = indices[self.rank::self.num_replicas]

        # Now apply chunk-based local shuffling (ShuffleBufferSampler logic)
        # Use a rank-specific seed for the local shuffle
        rng = random.Random(self.seed + self._epoch * 1000 + self.rank)

        for chunk_start in range(0, len(rank_indices), self.buffer_size):
            chunk_end = min(chunk_start + self.buffer_size, len(rank_indices))
            chunk = rank_indices[chunk_start:chunk_end]
            rng.shuffle(chunk)  # Local shuffle within chunk
            yield from chunk

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling (required for DDP)."""
        self._epoch = epoch


# =============================================================================
# DDP (DistributedDataParallel) Helper Functions
# =============================================================================

def setup_distributed():
    """
    Initialize DDP environment if launched with torchrun.
    Returns (rank, world_size, local_rank, is_distributed).

    Call this early in main() before any other setup.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True

    # Not launched with torchrun - single process mode
    return 0, 1, 0, False


def cleanup_distributed():
    """Clean up DDP resources at end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed_spawn(rank: int, world_size: int):
    """
    Initialize DDP for mp.spawn() launched processes.

    Unlike setup_distributed() which reads from env vars (torchrun),
    this takes rank/world_size directly from mp.spawn() arguments.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def is_main_process(rank: int) -> bool:
    """Check if this is the main process (rank 0) for printing/saving."""
    return rank == 0


def hierarchical_collate(batch, device: str = None, move_to_device: bool = False, torch_dtype=None, _debug_counter=[0], suppress_slow_warnings=False):
    """
    Fast collate: stack numpy arrays first, then single torch conversion.

    Supports two input formats:
    1. Legacy: ((main_channels, monthly_channels, non_channels), targets)
       - Returns: (Tensor [batch, seq, total_features], targets_dict)

    2. v4.1 Native timeframe: (Dict[str, np.ndarray], targets)
       - Returns: (Dict[str, Tensor], targets_dict)
       - Each timeframe has its own tensor with shape [batch, seq_len, features]
    """
    import os
    import sys
    import time
    _collate_start = time.perf_counter()

    # Debug: track collate calls (mutable default persists across calls)
    _debug_counter[0] += 1
    if _debug_counter[0] <= 3 and os.environ.get('TRAIN_DEBUG', '0') == '1':
        print(f"[DEBUG] collate worker PID={os.getpid()}, batch_size={len(batch)}, call #{_debug_counter[0]}", file=sys.stderr, flush=True)

    if torch_dtype is None:
        torch_dtype = torch.float32

    # v5.2: Handle both 2-element (legacy) and 4-element (features, targets, vix, events) tuples
    if len(batch[0]) == 4:
        # v5.2 format
        data_list = [sample[0] for sample in batch]
        targets_list = [sample[1] for sample in batch]
        vix_list = [sample[2] for sample in batch]
        events_list = [sample[3] for sample in batch]
    else:
        # Legacy format
        data_list = [d for d, _ in batch]
        targets_list = [t for _, t in batch]
        vix_list = None
        events_list = None

    # Detect format: dict (v4.1 native timeframe) or tuple (legacy)
    is_native_timeframe = isinstance(data_list[0], dict)

    if is_native_timeframe:
        # v4.1: Native timeframe mode - stack dicts into Dict[str, Tensor]
        # v5.9.7: Optimized - pre-allocate and direct copy (no intermediate list)
        batched_tf_data = {}
        batch_size = len(data_list)
        for tf in data_list[0].keys():
            # Get shape from first sample
            first_arr = data_list[0][tf]
            # Pre-allocate output array (C-contiguous by default)
            stacked = np.empty((batch_size, *first_arr.shape), dtype=first_arr.dtype)
            # Direct copy into pre-allocated array
            for i, d in enumerate(data_list):
                stacked[i] = d[tf]

            batched_tf_data[tf] = torch.from_numpy(stacked).to(dtype=torch_dtype)

        x = batched_tf_data
    else:
        # Legacy: tuple format (main_channels, monthly_channels, non_channels)
        first_ch_main, first_ch_monthly, first_nc = data_list[0]
        has_monthly = first_ch_monthly is not None
        has_nc = first_nc is not None

        # Stack numpy arrays (fast vectorized C operations)
        main_stack = np.stack([d[0] for d in data_list])  # [batch, seq, main_cols]

        if has_monthly:
            monthly_stack = np.stack([d[1] for d in data_list])  # [batch, seq, monthly_cols]

        if has_nc:
            nc_stack = np.stack([d[2] for d in data_list])  # [batch, seq, nc_cols]

        # Concatenate along feature axis (single operation)
        parts = [main_stack]
        if has_monthly:
            parts.append(monthly_stack)
        if has_nc:
            parts.append(nc_stack)

        combined = np.concatenate(parts, axis=2)  # [batch, seq, total_features]

        # Single torch conversion (fast)
        if not combined.flags['C_CONTIGUOUS']:
            combined = np.ascontiguousarray(combined)
        x = torch.from_numpy(combined).to(dtype=torch_dtype)

    # Build targets tensor dict with proper dtype
    # v5.9.5: Optimized - batch tensor creation per key (1013 calls vs 64,832)
    # Previously: 64 samples × 1013 keys = 64,832 torch.tensor() calls (~12s)
    # Now: 1013 keys × 1 torch.as_tensor() call (~0.2s)
    _targets_start = time.perf_counter()
    targets_batch = {}

    if targets_list and len(targets_list) > 0:
        first_target = targets_list[0]
        for k in first_target.keys():
            # v5.9.2: price_sequence stays as list (variable length, used in loss loop)
            # v6.0: window_r_squared and window_durations are also lists (14 values per sample)
            if '_price_sequence' in k:
                targets_batch[k] = [t[k] for t in targets_list]  # List of lists
            elif '_window_r_squared' in k or '_window_durations' in k:
                # v6.0: Stack into 2D tensor [batch, 14]
                values = [t[k] for t in targets_list]
                if isinstance(values[0], list):
                    targets_batch[k] = torch.tensor(values, dtype=torch_dtype)
                else:
                    targets_batch[k] = torch.as_tensor(values, dtype=torch_dtype)
            else:
                # Gather all values for this key across samples
                values = [t[k] for t in targets_list]
                # Check if already tensors
                if isinstance(values[0], torch.Tensor):
                    targets_batch[k] = torch.stack(values).to(dtype=torch_dtype)
                else:
                    # Single tensor creation for all samples (fast!)
                    targets_batch[k] = torch.as_tensor(values, dtype=torch_dtype)

    # v5.9.5: Profile targets tensor creation (first 5 batches)
    _targets_elapsed = time.perf_counter() - _targets_start
    if _debug_counter[0] <= 5:
        print(f"[PROFILE] targets tensor creation: {_targets_elapsed*1000:.1f}ms for {len(targets_list)} samples, {len(targets_batch)} keys", file=sys.stderr, flush=True)

    if move_to_device and device is not None:
        if is_native_timeframe:
            for tf in x:
                x[tf] = x[tf].to(device, non_blocking=True)
        else:
            x = x.to(device, non_blocking=True)
        for k, v in targets_batch.items():
            # v5.9.2: Skip price_sequence (list of lists, not tensor)
            if isinstance(v, torch.Tensor):
                targets_batch[k] = v.to(device, non_blocking=True)

    # v5.2: Collate VIX sequences
    vix_batch = None
    if vix_list is not None:
        # Filter out None values and stack
        valid_vix = [v for v in vix_list if v is not None]
        if len(valid_vix) > 0:
            vix_batch = torch.tensor(np.array(valid_vix), dtype=torch_dtype)
            if move_to_device and device is not None:
                vix_batch = vix_batch.to(device, non_blocking=True)

    # v5.2: Collate events (list of lists)
    events_batch = None
    if events_list is not None:
        # Events stay as list - EventEmbedding.forward_batch() handles lists
        events_batch = events_list

    # Log slow batch assembly (diagnose lazy loading bottlenecks)
    _collate_elapsed = time.perf_counter() - _collate_start
    if _collate_elapsed > 1.0 and not suppress_slow_warnings:  # Log if >1 second and not suppressed
        print(f"[SLOW_COLLATE] batch assembly took {_collate_elapsed:.1f}s for {len(batch)} samples ({_collate_elapsed/len(batch)*1000:.0f}ms/sample)", file=sys.stderr, flush=True)

    return x, targets_batch, vix_batch, events_batch


# =============================================================================
# v5.3.2: PRE-STACKED BATCH LOADER (Rolling Pre-Stack - Option B)
# =============================================================================

class PreStackedBatchLoader:
    """
    Pre-stacks batches before training for faster epoch iteration.

    Option B (Rolling Pre-Stack):
    - Pre-stack epoch 1 before training starts (blocking)
    - While training epoch N, background thread pre-stacks epoch N+1
    - Only keeps 2 epochs worth of batches in memory at once

    Benefits:
    - Eliminates ~1.3s/batch collate overhead during training
    - ~40% faster epochs

    Trade-offs:
    - Initial delay to pre-stack first epoch (~5-10 min)
    - Uses ~77-154GB RAM (2 epochs worth of batches)
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_epochs: int,
        collate_fn,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        drop_last: bool = True,
        use_pinned: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.collate_fn = collate_fn
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.drop_last = drop_last
        self.use_pinned = use_pinned

        # RAM storage for pre-stacked batches
        self.epoch_batches = {}  # {epoch: [batch0, batch1, ...]}

        # Pre-generate shuffled indices for all epochs upfront
        if self.verbose and rank == 0:
            print(f"\n📦 Pre-Stack: Generating shuffle indices for {num_epochs} epochs...")

        self.epoch_indices = {}
        dataset_size = len(dataset)
        for ep in range(num_epochs):
            # Use different seed per epoch for reproducibility
            g = torch.Generator()
            g.manual_seed(42 + ep)
            self.epoch_indices[ep] = torch.randperm(dataset_size, generator=g).tolist()

        # Threading for background pre-stacking
        self.prestack_thread = None
        self.prestack_lock = threading.Lock()
        self.prestack_error = None

        # Current epoch state
        self.current_epoch = -1
        self._num_batches_cache = {}

    def _get_rank_indices(self, epoch: int):
        """Get the indices assigned to this rank for an epoch."""
        all_indices = self.epoch_indices[epoch]

        if self.world_size == 1:
            return all_indices

        # Split indices across ranks (DDP)
        per_rank = len(all_indices) // self.world_size
        start = self.rank * per_rank
        end = start + per_rank
        return all_indices[start:end]

    def _pin_batch(self, batch):
        """
        Pin tensors in a batch for faster CPU→GPU transfer.

        Batch format: (features, targets, vix_batch, events_batch)
        - features: dict[str, Tensor] or Tensor
        - targets: dict[str, Tensor]
        - vix_batch: Tensor or None
        - events_batch: Tensor or None
        """
        def pin_tensor(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor) and not t.is_pinned():
                return t.pin_memory()
            return t

        def pin_dict(d):
            if d is None:
                return None
            return {k: pin_tensor(v) for k, v in d.items()}

        if len(batch) == 4:
            features, targets, vix_batch, events_batch = batch
        else:
            features, targets = batch
            vix_batch, events_batch = None, None

        # Pin features (dict or tensor)
        if isinstance(features, dict):
            features = pin_dict(features)
        else:
            features = pin_tensor(features)

        # Pin targets dict
        targets = pin_dict(targets)

        # Pin optional tensors
        vix_batch = pin_tensor(vix_batch)
        events_batch = pin_tensor(events_batch)

        if len(batch) == 4:
            return (features, targets, vix_batch, events_batch)
        else:
            return (features, targets)

    def prestack_epoch(self, epoch: int, show_progress: bool = True):
        """Pre-stack all batches for a given epoch."""
        rank_indices = self._get_rank_indices(epoch)
        num_full_batches = len(rank_indices) // self.batch_size

        batches = []

        # Initial message
        if show_progress and self.verbose and self.rank == 0:
            print(f"📦 Pre-stacking epoch {epoch + 1}: 0/{num_full_batches} batches...", flush=True)

        for batch_idx in range(num_full_batches):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_indices = rank_indices[start:end]

            # Fetch samples
            samples = [self.dataset[i] for i in batch_indices]

            # Collate into batch (suppress SLOW_COLLATE warnings during pre-stacking)
            batch = self.collate_fn(samples, suppress_slow_warnings=True)

            # v5.3.2: Pin memory if requested (faster CPU→GPU transfer)
            if self.use_pinned:
                batch = self._pin_batch(batch)

            batches.append(batch)

            # Progress update every 10 batches
            if show_progress and self.verbose and self.rank == 0 and (batch_idx + 1) % 10 == 0:
                progress_pct = (batch_idx + 1) / num_full_batches
                print(f"   Progress: {batch_idx + 1}/{num_full_batches} batches ({progress_pct:.0%})", flush=True)

        # Handle last partial batch if not drop_last
        if not self.drop_last:
            remaining = len(rank_indices) % self.batch_size
            if remaining > 0:
                start = num_full_batches * self.batch_size
                batch_indices = rank_indices[start:]
                samples = [self.dataset[i] for i in batch_indices]
                batch = self.collate_fn(samples, suppress_slow_warnings=True)
                if self.use_pinned:
                    batch = self._pin_batch(batch)
                batches.append(batch)
                num_full_batches += 1

        with self.prestack_lock:
            self.epoch_batches[epoch] = batches
            self._num_batches_cache[epoch] = len(batches)

        return len(batches)

    def _background_prestack(self, epoch: int):
        """Background worker for pre-stacking."""
        try:
            self.prestack_epoch(epoch, show_progress=False)
            if self.verbose and self.rank == 0:
                with self.prestack_lock:
                    num_batches = len(self.epoch_batches.get(epoch, []))
                print(f"   ✅ Background pre-stack complete: epoch {epoch + 1} ({num_batches} batches)", flush=True)
        except Exception as e:
            self.prestack_error = e
            if self.verbose and self.rank == 0:
                print(f"\n⚠️ Background pre-stack failed for epoch {epoch + 1}: {e}", flush=True)

    def start_background_prestack(self, epoch: int):
        """Start pre-stacking an epoch in a background thread."""
        # Wait for any existing background work
        self.wait_for_prestack()

        self.prestack_thread = threading.Thread(
            target=self._background_prestack,
            args=(epoch,),
            daemon=True
        )
        self.prestack_thread.start()

    def wait_for_prestack(self):
        """Wait for background pre-stacking to complete."""
        if self.prestack_thread is not None:
            self.prestack_thread.join()
            self.prestack_thread = None

    def set_epoch(self, epoch: int):
        """
        Prepare for a new epoch.

        - Ensures current epoch's batches are ready
        - Starts background pre-stacking of next epoch
        - Cleans up old epoch's batches to free memory
        """
        self.current_epoch = epoch

        # Check if current epoch is already pre-stacked
        with self.prestack_lock:
            has_epoch = epoch in self.epoch_batches

        if not has_epoch:
            # Wait for background thread in case it's working on this epoch
            self.wait_for_prestack()

            # v5.3.2: Check if background thread failed
            if self.prestack_error is not None:
                err = self.prestack_error
                self.prestack_error = None  # Clear for retry
                if self.verbose and self.rank == 0:
                    print(f"\n⚠️ Background pre-stack failed: {err}")
                    print(f"   Falling back to synchronous pre-stacking...")

            # Check again after waiting
            with self.prestack_lock:
                has_epoch = epoch in self.epoch_batches

            if not has_epoch:
                # Need to pre-stack now (blocking)
                if self.verbose and self.rank == 0:
                    print(f"\n📦 Pre-stacking epoch {epoch + 1} (this may take a few minutes)...")
                self.prestack_epoch(epoch, show_progress=True)
                if self.verbose and self.rank == 0:
                    with self.prestack_lock:
                        num_batches = len(self.epoch_batches.get(epoch, []))
                    print(f"   ✅ Pre-stacked {num_batches} batches for epoch {epoch + 1}")

        # Start pre-stacking next epoch in background
        if epoch + 1 < self.num_epochs:
            # Check if next epoch already pre-stacked
            with self.prestack_lock:
                next_ready = (epoch + 1) in self.epoch_batches

            if not next_ready:
                if self.verbose and self.rank == 0:
                    print(f"   🔄 Background: Pre-stacking epoch {epoch + 2}...")
                self.start_background_prestack(epoch + 1)

        # Clean up old epoch's data to free memory
        if epoch > 0:
            self._cleanup_epoch(epoch - 1)

    def _cleanup_epoch(self, epoch: int):
        """Remove an epoch's batches from memory."""
        with self.prestack_lock:
            if epoch in self.epoch_batches:
                del self.epoch_batches[epoch]
                if self.verbose and self.rank == 0:
                    pass  # Silent cleanup

    def __iter__(self):
        """Iterate over pre-stacked batches for current epoch."""
        epoch = self.current_epoch

        with self.prestack_lock:
            batches = self.epoch_batches.get(epoch, [])

        for batch in batches:
            yield batch

    def __len__(self):
        """Number of batches in current epoch."""
        epoch = self.current_epoch if self.current_epoch >= 0 else 0

        with self.prestack_lock:
            if epoch in self._num_batches_cache:
                return self._num_batches_cache[epoch]

        # Calculate from indices
        rank_indices = self._get_rank_indices(epoch)
        if self.drop_last:
            return len(rank_indices) // self.batch_size
        else:
            return (len(rank_indices) + self.batch_size - 1) // self.batch_size

    def cleanup(self):
        """Clean up all resources."""
        self.wait_for_prestack()
        self.epoch_batches.clear()
        self._num_batches_cache.clear()


class RollingBufferBatchLoader:
    """
    Rolling buffer batch loader (v5.3.2) - RAM-efficient pre-stacking.

    Instead of pre-stacking entire epochs (needs TB of RAM), maintains
    a rolling buffer of N batches (configurable based on available RAM).

    Benefits:
    - Eliminates collate overhead (~9% faster)
    - Fits in reasonable RAM (50-200 batches = 45-180 GB)
    - Collation happens in background (parallel with GPU training)

    Trade-offs:
    - Uses more RAM than standard DataLoader (but WAY less than full epoch)
    - Slightly more complex than standard DataLoader
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        buffer_size: int,  # Number of batches to keep in buffer
        collate_fn,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        drop_last: bool = True,
        use_pinned: bool = False,
        num_workers: int = 0,  # v5.9.3: Parallel sample fetching (0 = sequential)
    ):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        self.dataset = dataset
        self.num_workers = num_workers
        # Create thread pool for parallel fetching (if workers > 0)
        self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers)) if num_workers > 0 else None
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.collate_fn = collate_fn
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.drop_last = drop_last
        self.use_pinned = use_pinned

        # Rolling buffer (FIFO queue with max size)
        self.batch_buffer = deque(maxlen=buffer_size)

        # Shuffling state
        self.current_epoch = 0
        self.epoch_indices = {}

        # Background thread for collation
        self.collate_thread = None
        self.collate_lock = threading.Lock()
        self.collate_error = None
        self.stop_background = False

        # Track position
        self.current_batch_idx = 0
        self.total_batches = 0

    def _get_rank_indices(self, epoch: int):
        """Get shuffled indices for this epoch and rank."""
        if epoch not in self.epoch_indices:
            # Generate shuffle for this epoch
            dataset_size = len(self.dataset)
            g = torch.Generator()
            g.manual_seed(42 + epoch)
            all_indices = torch.randperm(dataset_size, generator=g).tolist()

            if self.world_size > 1:
                # Split across ranks (DDP)
                per_rank = len(all_indices) // self.world_size
                start = self.rank * per_rank
                end = start + per_rank
                self.epoch_indices[epoch] = all_indices[start:end]
            else:
                self.epoch_indices[epoch] = all_indices

        return self.epoch_indices[epoch]

    def _collate_batch(self, batch_idx: int, epoch_indices: list):
        """Collate a single batch."""
        start = batch_idx * self.batch_size
        end = start + self.batch_size

        if start >= len(epoch_indices):
            return None

        batch_indices = epoch_indices[start:end]
        if len(batch_indices) < self.batch_size and self.drop_last:
            return None

        # Fetch samples - v5.9.3: Use parallel fetching if workers > 0
        if self._executor is not None:
            # Parallel fetch using ThreadPoolExecutor
            futures = [self._executor.submit(self.dataset.__getitem__, i) for i in batch_indices]
            samples = [f.result() for f in futures]
        else:
            # Sequential fetch (original behavior)
            samples = [self.dataset[i] for i in batch_indices]

        # Collate
        batch = self.collate_fn(samples, suppress_slow_warnings=True)

        # Pin memory if requested
        if self.use_pinned:
            batch = self._pin_batch(batch)

        return batch

    def _pin_batch(self, batch):
        """Pin tensors for faster GPU transfer (same as PreStackedBatchLoader)."""
        def pin_tensor(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor) and not t.is_pinned():
                return t.pin_memory()
            return t

        def pin_dict(d):
            if d is None:
                return None
            return {k: pin_tensor(v) for k, v in d.items()}

        if len(batch) == 4:
            features, targets, vix_batch, events_batch = batch
            features = pin_dict(features) if isinstance(features, dict) else pin_tensor(features)
            targets = pin_dict(targets)
            vix_batch = pin_tensor(vix_batch)
            events_batch = pin_tensor(events_batch)
            return (features, targets, vix_batch, events_batch)
        else:
            features, targets = batch
            features = pin_dict(features) if isinstance(features, dict) else pin_tensor(features)
            targets = pin_dict(targets)
            return (features, targets)

    def _fill_initial_buffer(self, epoch_indices: list):
        """Pre-collate initial buffer before training starts."""
        if self.verbose and self.rank == 0:
            print(f"📦 Rolling buffer: Pre-collating first {self.buffer_size} batches...")

        for i in range(min(self.buffer_size, self.total_batches)):
            batch = self._collate_batch(i, epoch_indices)
            if batch is not None:
                self.batch_buffer.append(batch)

        if self.verbose and self.rank == 0:
            buffer_gb = len(self.batch_buffer) * (self.batch_size * 3.6e6) / 1e9
            print(f"   ✓ Buffer ready: {len(self.batch_buffer)} batches ({buffer_gb:.0f} GB)")

    def set_epoch(self, epoch: int):
        """Set current epoch and initialize buffer."""
        self.current_epoch = epoch
        self.current_batch_idx = 0

        # Get indices for this epoch
        epoch_indices = self._get_rank_indices(epoch)
        self.total_batches = len(epoch_indices) // self.batch_size
        if not self.drop_last and len(epoch_indices) % self.batch_size > 0:
            self.total_batches += 1

        # Clear old buffer
        self.batch_buffer.clear()

        # Fill initial buffer
        self._fill_initial_buffer(epoch_indices)

        # Store indices for background collation
        self._current_epoch_indices = epoch_indices

    def _background_collate_next(self, next_batch_idx: int):
        """Collate next batch in background."""
        try:
            batch = self._collate_batch(next_batch_idx, self._current_epoch_indices)
            if batch is not None:
                with self.collate_lock:
                    self.batch_buffer.append(batch)  # Deque auto-drops oldest
        except Exception as e:
            self.collate_error = e

    def __iter__(self):
        """Iterate over batches with rolling buffer."""
        epoch_indices = self._current_epoch_indices

        for batch_idx in range(self.total_batches):
            # Get batch from buffer (should be at front)
            with self.collate_lock:
                if len(self.batch_buffer) == 0:
                    # Buffer empty - shouldn't happen, but fallback to direct collate
                    batch = self._collate_batch(batch_idx, epoch_indices)
                else:
                    batch = self.batch_buffer.popleft()

            # Start background collation of future batch
            next_batch_idx = batch_idx + self.buffer_size
            if next_batch_idx < self.total_batches:
                thread = threading.Thread(
                    target=self._background_collate_next,
                    args=(next_batch_idx,),
                    daemon=True
                )
                thread.start()

            yield batch

    def __len__(self):
        """Number of batches in epoch."""
        return self.total_batches

    def cleanup(self):
        """Clean up resources."""
        self.stop_background = True
        self.batch_buffer.clear()
        self.epoch_indices.clear()
        # v5.9.3: Shutdown thread pool
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def load_cache_manifests(cache_dir: Path):
    """Load cache manifests from a directory."""
    manifests = []
    try:
        for path in Path(cache_dir).glob("cache_manifest_*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                    data["_manifest_path"] = str(path)
                    manifests.append(data)
            except Exception:
                continue
    except Exception:
        pass
    return manifests


def pick_manifest(manifests):
    """Choose a manifest via Inquirer if available."""
    if not manifests:
        return None
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        return None

    choices = []
    for m in manifests:
        ck = m.get("cache_key", "unknown")
        fr = m.get("date_range", {}).get("start", "?")
        to = m.get("date_range", {}).get("end", "?")
        mode = m.get("continuation_mode", "?")
        fv = m.get("feature_version", "?")
        choices.append(
            Choice(
                value=m,
                name=f"{ck} | {fr}→{to} | mode={mode} | feat={fv}"
            )
        )
    choices.append(Choice(value=None, name="Do not reuse cached settings"))

    return inquirer.select(
        message="Reuse existing cache (features/labels)?",
        choices=choices,
        default=choices[-1].value
    ).execute()


def validate_cache_layers(cache_dir: Path) -> dict:
    """
    v5.9.2: Validate cache layers independently.

    Returns status for each layer:
    - training_ready: tf_sequence_*.npy + tf_timestamps_*.npy + tf_meta_*.json exist
    - labels_ready: continuation_labels + transition_labels exist
    - features_ready: non_channel_features_*.pkl exists
    - can_regenerate_tf: mmap_chunks/ shards exist (can regenerate native TF)
    - can_regenerate_all: chunks + all extraction dependencies exist

    This allows users to delete 60GB of chunks while still training with
    existing native TF sequences (~7GB).
    """
    cache_dir = Path(cache_dir)

    result = {
        'training_ready': False,
        'labels_ready': False,
        'features_ready': False,
        'can_regenerate_tf': False,
        'can_regenerate_all': False,
        # Detailed info
        'tf_meta_path': None,
        'tf_sequence_count': 0,
        'tf_timestamps_count': 0,
        'continuation_labels_count': 0,
        'transition_labels_count': 0,
        'non_channel_path': None,
        'chunk_meta_path': None,
        'chunk_count': 0,
        'chunks_missing': [],
        'tf_size_gb': 0.0,
        'labels_size_gb': 0.0,
        'non_channel_size_gb': 0.0,
        'chunks_size_gb': 0.0,
    }

    if not cache_dir.exists():
        return result

    import json

    # Layer 1: Training Layer (Native TF sequences)
    tf_meta_files = list(cache_dir.glob("tf_meta_*.json"))
    tf_sequence_files = list(cache_dir.glob("tf_sequence_*.npy"))
    tf_timestamps_files = list(cache_dir.glob("tf_timestamps_*.npy"))

    if tf_meta_files:
        result['tf_meta_path'] = str(tf_meta_files[0])
        result['tf_sequence_count'] = len(tf_sequence_files)
        result['tf_timestamps_count'] = len(tf_timestamps_files)

        # Calculate TF layer size
        tf_size = sum(f.stat().st_size for f in tf_sequence_files if f.exists())
        tf_size += sum(f.stat().st_size for f in tf_timestamps_files if f.exists())
        result['tf_size_gb'] = tf_size / 1e9

        # v5.9.2: Validate tf_meta has all required timeframes
        try:
            with open(tf_meta_files[0]) as f:
                tf_meta = json.load(f)
            # Check that sequences referenced in meta actually exist
            # tf_meta uses 'sequence_lengths' dict, not 'timeframes' list
            sequence_lengths = tf_meta.get('sequence_lengths', {})
            tf_count = len(sequence_lengths)
            if tf_count >= 10:  # Allow 10/11 (3month may be missing)
                result['training_ready'] = True
        except Exception:
            pass

    # Layer 2: Labels Layer (continuation + transition labels)
    continuation_files = list(cache_dir.glob("continuation_labels_*.pkl"))
    transition_files = list(cache_dir.glob("transition_labels_*.pkl"))

    result['continuation_labels_count'] = len(continuation_files)
    result['transition_labels_count'] = len(transition_files)

    # Calculate labels size
    labels_size = sum(f.stat().st_size for f in continuation_files if f.exists())
    labels_size += sum(f.stat().st_size for f in transition_files if f.exists())
    result['labels_size_gb'] = labels_size / 1e9

    # Need at least 10/11 TFs for both continuation and transition
    if len(continuation_files) >= 10 and len(transition_files) >= 10:
        result['labels_ready'] = True

    # Layer 3: Features Layer (non-channel features)
    non_channel_files = list(cache_dir.glob("non_channel_features_*.pkl"))
    if non_channel_files:
        result['non_channel_path'] = str(non_channel_files[0])
        result['features_ready'] = True
        result['non_channel_size_gb'] = non_channel_files[0].stat().st_size / 1e9

    # Layer 4: Generation Layer (chunks for regenerating TF)
    chunk_meta_files = list(cache_dir.glob("features_mmap_meta_*.json"))
    if chunk_meta_files:
        result['chunk_meta_path'] = str(chunk_meta_files[0])

        try:
            with open(chunk_meta_files[0]) as f:
                chunk_meta = json.load(f)

            chunk_info = chunk_meta.get('chunk_info', [])
            result['chunk_count'] = len(chunk_info)

            # Check if all chunks exist
            cache_base_dir = chunk_meta_files[0].parent
            chunks_exist = []
            chunks_missing = []
            total_chunk_size = 0

            for c in chunk_info:
                chunk_path = Path(c['path'])
                if not chunk_path.is_absolute():
                    chunk_path = cache_base_dir / chunk_path
                if chunk_path.exists():
                    chunks_exist.append(str(chunk_path))
                    total_chunk_size += chunk_path.stat().st_size
                else:
                    chunks_missing.append(str(c['path']))

            result['chunks_missing'] = chunks_missing
            result['chunks_size_gb'] = total_chunk_size / 1e9

            # Can regenerate TF if all chunks exist
            if len(chunks_missing) == 0 and len(chunk_info) > 0:
                result['can_regenerate_tf'] = True
                result['can_regenerate_all'] = True

        except Exception as e:
            pass

    return result


def find_available_caches(cache_dir: Path):
    """Find available cache triplets (meta + continuation labels + non-channel) in a directory.

    Supports both:
    - Chunked/mmap mode: features_mmap_meta_*.json + shards
    - Non-chunked/pickle mode: rolling_channels_*.pkl (legacy)

    v5.2: Also tracks transition labels for multi-phase compositor
    v5.9.2: Includes layer status from validate_cache_layers()
    """
    cache_dir = Path(cache_dir)
    caches = []
    seen_keys = set()

    # v5.9.2: Get layer status for cache directory
    layer_status = validate_cache_layers(cache_dir)

    # v5.2: Count transition label files (per-TF pattern: transition_labels_{tf}_*.pkl)
    def count_transition_labels(cache_dir: Path) -> int:
        """Count how many TF transition label files exist."""
        return len(list(cache_dir.glob("transition_labels_*.pkl")))

    # 1. Find mmap caches (chunked mode)
    for meta_path in cache_dir.glob("features_mmap_meta_*.json"):
        cache_key = meta_path.name.replace("features_mmap_meta_", "").replace(".json", "")
        seen_keys.add(cache_key)

        mode_suffixes = ['adaptive']  # v5.2: Removed 'simple'
        cont_path = None
        for suffix in mode_suffixes:
            candidate = cache_dir / f"continuation_labels_{cache_key}_{suffix}.pkl"
            if candidate.exists():
                cont_path = candidate
                break

        # Check for non-channel features cache
        non_channel_path = cache_dir / f"non_channel_features_{cache_key}.pkl"
        has_non_channel = non_channel_path.exists()

        # v5.2: Check for transition labels
        transition_label_count = count_transition_labels(cache_dir)

        caches.append({
            "cache_key": cache_key,
            "cache_type": "mmap",
            "meta_path": str(meta_path),
            "cont_path": str(cont_path) if cont_path else None,
            "non_channel_path": str(non_channel_path) if has_non_channel else None,
            "transition_labels_count": transition_label_count,  # v5.2
            "complete": cont_path is not None and has_non_channel
        })

    # 2. Find pickle caches (non-chunked mode)
    for pickle_path in cache_dir.glob("rolling_channels_*.pkl"):
        # Extract cache key from pickle filename
        # Format: rolling_channels_{cache_key}.pkl or rolling_channels_{cache_key}_{suffix}.pkl
        name = pickle_path.name.replace("rolling_channels_", "").replace(".pkl", "")
        # Remove any suffix like _GPU_TEST or _CPU_TEST
        cache_key = name.split("_GPU_TEST")[0].split("_CPU_TEST")[0]

        if cache_key in seen_keys:
            continue  # Already found as mmap
        seen_keys.add(cache_key)

        mode_suffixes = ['adaptive']  # v5.2: Removed 'simple'
        cont_path = None
        for suffix in mode_suffixes:
            candidate = cache_dir / f"continuation_labels_{cache_key}_{suffix}.pkl"
            if candidate.exists():
                cont_path = candidate
                break

        # v5.2: Check for transition labels
        transition_label_count = count_transition_labels(cache_dir)

        caches.append({
            "cache_key": cache_key,
            "cache_type": "pickle",
            "meta_path": None,  # No mmap meta for pickle mode
            "pickle_path": str(pickle_path),
            "cont_path": str(cont_path) if cont_path else None,
            "non_channel_path": None,  # Pickle mode includes all features
            "transition_labels_count": transition_label_count,  # v5.2
            "complete": cont_path is not None  # Pickle has all features, just need continuation
        })

    # v5.9.2: Return both caches and layer_status
    return caches, layer_status


def pick_cache_pair(caches, layer_status=None):
    """Prompt user to select a cache triplet to reuse.

    v5.9.2: Added layer_status parameter for context-aware menu options.
    Shows training readiness even if chunks are deleted.
    """
    if not caches and layer_status is None:
        return None

    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        return None

    # v5.9.2: Show layer status header
    if layer_status:
        print("\n" + "=" * 60)
        print("  CACHE LAYER STATUS")
        print("=" * 60)

        # Training layer
        if layer_status['training_ready']:
            print(f"   ✓ Native TF sequences: Ready ({layer_status['tf_sequence_count']} files, {layer_status['tf_size_gb']:.1f} GB)")
        else:
            print(f"   ✗ Native TF sequences: Not found")

        # Labels layer
        if layer_status['labels_ready']:
            print(f"   ✓ Labels: Ready ({layer_status['continuation_labels_count']} cont + {layer_status['transition_labels_count']} trans, {layer_status['labels_size_gb']:.1f} GB)")
        else:
            print(f"   ✗ Labels: Incomplete ({layer_status['continuation_labels_count']} cont, {layer_status['transition_labels_count']} trans)")

        # Features layer
        if layer_status['features_ready']:
            print(f"   ✓ Non-channel features: Ready ({layer_status['non_channel_size_gb']:.1f} GB)")
        else:
            print(f"   ✗ Non-channel features: Not found")

        # Generation layer (chunks)
        if layer_status['can_regenerate_tf']:
            print(f"   ✓ Chunk shards: Available ({layer_status['chunk_count']} chunks, {layer_status['chunks_size_gb']:.1f} GB)")
        elif layer_status['chunk_meta_path']:
            print(f"   ⚠️  Chunk shards: Metadata found but {len(layer_status['chunks_missing'])} chunks DELETED")
            print(f"       (Cannot regenerate TF sequences - but training still possible!)")
        else:
            print(f"   ✗ Chunk shards: Not found")

        # Summary
        if layer_status['training_ready'] and layer_status['labels_ready'] and layer_status['features_ready']:
            total_gb = layer_status['tf_size_gb'] + layer_status['labels_size_gb'] + layer_status['non_channel_size_gb']
            print(f"\n   ✅ TRAINING READY ({total_gb:.1f} GB total)")
            if not layer_status['can_regenerate_tf']:
                print(f"   ℹ️  Chunks deleted - using ~{total_gb:.1f} GB instead of ~{layer_status['chunks_size_gb'] + total_gb:.1f} GB")

        print("=" * 60)

    # Handle case where no caches but layer_status indicates training is ready
    if not caches and layer_status and layer_status['training_ready'] and layer_status['labels_ready']:
        print(f"\n✅ Training ready via layer validation (no legacy cache pairs found)")
        # Return a synthetic cache entry
        return {
            'cache_key': 'native_tf_mode',
            'cache_type': 'native_tf',
            'training_ready': True,
            'skip_chunk_validation': True,
        }

    if not caches:
        return None

    # Auto-select when only one cache is available
    if len(caches) == 1:
        cache = caches[0]
        status = "COMPLETE" if cache.get("complete") else "partial"
        trans_count = cache.get("transition_labels_count", 0)
        trans_info = f" + {trans_count} transition labels" if trans_count > 0 else ""
        print(f"\n✅ Auto-selected cache: {cache['cache_key']} ({status}{trans_info})")
        print(f"   → Will skip feature extraction and use cached data\n")
        return cache

    choices = []
    for c in caches:
        # Build status string based on what's cached
        # v5.2: Include transition label count
        trans_count = c.get("transition_labels_count", 0)
        trans_info = f" +{trans_count}TL" if trans_count > 0 else ""

        if c.get("complete"):
            status = f"COMPLETE{trans_info} - skip extraction entirely"
        elif c.get("cont_path") and c.get("non_channel_path"):
            status = f"COMPLETE{trans_info} - skip extraction entirely"
        elif c.get("cont_path"):
            status = f"partial{trans_info}: channels + labels (will recompute non-channel ~10-30s)"
        elif c.get("non_channel_path"):
            status = "partial: channels + non-channel (no labels)"
        else:
            status = "channels only"
        choices.append(
            Choice(
                value=c,
                name=f"{c['cache_key']} ({status})"
            )
        )
    choices.append(Choice(value=None, name="Do not reuse cached features/labels"))

    return inquirer.select(
        message="Reuse existing cached features/labels?",
        choices=choices,
        default=choices[0].value  # Default to first cache
    ).execute()


def save_cache_manifest(
    cache_dir: Path,
    cache_key: str,
    continuation_path: str,
    args,
    df: pd.DataFrame,
    mmap_meta_path: str = None,  # Optional: only for chunked/mmap mode
    non_channel_path: str = None,  # Optional: only for chunked/mmap mode
    pickle_path: str = None,  # Optional: only for non-chunked/pickle mode
):
    """Persist a manifest alongside caches for reuse selection.

    Supports both:
    - Chunked/mmap mode: mmap_meta_path + non_channel_path
    - Non-chunked/pickle mode: pickle_path (all features in one file)
    """
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / f"cache_manifest_{cache_key}.json"

        # Determine cache type
        cache_type = "mmap" if mmap_meta_path else "pickle"

        # v4.4: Convert paths to filenames only (all files in cache_dir, portable)
        def extract_filename(path):
            """Extract filename from path (supports str and Path)."""
            if path is None:
                return None
            return Path(path).name  # Just the filename, no directory

        # v5.2: Auto-detect transition labels
        transition_label_files = list(cache_dir.glob("transition_labels_*.pkl"))
        transition_labels_count = len(transition_label_files)

        manifest = {
            "cache_key": cache_key,
            "cache_type": cache_type,
            "feature_version": FEATURE_VERSION,
            "date_range": {
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
                "rows": len(df)
            },
            "continuation_mode": project_config.CONTINUATION_MODE,
            "adaptive_horizon_range": [
                getattr(project_config, "ADAPTIVE_MIN_HORIZON", None),
                getattr(project_config, "ADAPTIVE_MAX_HORIZON", None)
            ],
            "prediction_horizon": args.prediction_horizon,
            "precision": project_config.TRAINING_PRECISION,
            "dtype": np.dtype(project_config.NUMPY_DTYPE).name,
            "files": {
                # v4.4: Filenames only (all files in cache_dir)
                "mmap_meta": extract_filename(mmap_meta_path),
                "continuation_labels": extract_filename(continuation_path),
                "non_channel_features": extract_filename(non_channel_path),
                "pickle_channels": extract_filename(pickle_path),
                # v5.2: Track transition labels for multi-phase compositor
                "transition_labels_count": transition_labels_count,
                "transition_labels": [f.name for f in transition_label_files] if transition_label_files else None,
            },
            "source_files": {
                # v4.4: Track VIX/Events files for staleness detection
                "vix_csv": "VIX_History.csv",
                "vix_path": str(Path(project_config.DATA_DIR) / "VIX_History.csv") if hasattr(project_config, 'DATA_DIR') else None,
                "vix_mtime": None,
                "vix_size_bytes": None,
                "events_csv": "tsla_events_REAL.csv",
                "events_path": str(project_config.TSLA_EVENTS_FILE) if hasattr(project_config, 'TSLA_EVENTS_FILE') else None,
                "events_mtime": None,
                "events_count": None,
            },
            "cache_dir_ref": str(cache_dir),  # Reference only (for human debugging)
            "shard_storage_path": getattr(args, "shard_path", None),
            "use_chunking": getattr(args, "use_chunking", False),
            "use_gpu_features": getattr(args, "use_gpu_features", False),
            "use_parallel": getattr(args, "use_parallel", False),
            "feature_workers": getattr(args, "feature_workers", None),
            "timestamp": datetime.now().isoformat(),
            "training_settings": {
                "device": args.device,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "sequence_length": args.sequence_length,
                "prediction_horizon": args.prediction_horizon,
                "train_start_year": args.train_start_year,
                "train_end_year": args.train_end_year,
                "multi_task": args.multi_task,
                "hidden_size": args.hidden_size,
                "internal_neurons_ratio": args.internal_neurons_ratio,
                "num_workers": args.num_workers,
                "preload": getattr(args, "preload", False),
                "output": getattr(args, "output", None),
                "model_version": "5.0",  # v5.0: Channel-based predictions
                "use_test_set": getattr(args, "use_test_set", False),
                "test_split": getattr(args, "test_split", None),
                "use_geometric_base": getattr(args, "use_geometric_base", True),
                "use_fusion_head": getattr(args, "use_fusion_head", True),
                "architecture_mode": (
                    "geometric_physics" if (getattr(args, "use_geometric_base", True) and not getattr(args, "use_fusion_head", True)) else
                    "geometric_fusion" if (getattr(args, "use_geometric_base", True) and getattr(args, "use_fusion_head", True)) else
                    "learned_fusion" if (not getattr(args, "use_geometric_base", True) and getattr(args, "use_fusion_head", True)) else
                    "learned_physics"
                ),
            }
        }

        # Populate VIX/Events metadata if files exist
        try:
            vix_csv_path = Path(project_config.DATA_DIR) / "VIX_History.csv" if hasattr(project_config, 'DATA_DIR') else None
            if vix_csv_path and vix_csv_path.exists():
                vix_stat = vix_csv_path.stat()
                manifest["source_files"]["vix_mtime"] = int(vix_stat.st_mtime)
                manifest["source_files"]["vix_size_bytes"] = vix_stat.st_size
        except Exception:
            pass

        try:
            events_csv_path = Path(project_config.TSLA_EVENTS_FILE) if hasattr(project_config, 'TSLA_EVENTS_FILE') else None
            if events_csv_path and events_csv_path.exists():
                events_stat = events_csv_path.stat()
                manifest["source_files"]["events_mtime"] = int(events_stat.st_mtime)
                manifest["source_files"]["events_size_bytes"] = events_stat.st_size
                # Try to count events
                try:
                    events_df = pd.read_csv(events_csv_path)
                    manifest["source_files"]["events_count"] = len(events_df)
                except Exception:
                    pass
        except Exception:
            pass

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"   🗂️  Saved cache manifest: {manifest_path.name}")
    except Exception as e:
        print(f"   ⚠️  Could not save cache manifest ({type(e).__name__}): {e}")


def validate_cache_from_manifest(manifest_path: Path, verbose: bool = True) -> dict:
    """
    Validate cache files using manifest (v4.4).

    Args:
        manifest_path: Path to cache_manifest_*.json file
        verbose: If True, print validation messages

    Returns:
        dict with validation results and resolved paths
    """
    cache_dir = manifest_path.parent

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        if verbose:
            print(f"   ❌ Failed to load manifest: {e}")
        return {"valid": False, "error": f"Manifest load failed: {e}"}

    result = {
        "valid": True,
        "manifest": manifest,
        "cache_dir": cache_dir,
        "paths": {},
        "missing_files": [],
        "warnings": []
    }

    # Resolve and validate all files
    files = manifest.get("files", {})

    for key, filename in files.items():
        if filename is None:
            continue

        file_path = cache_dir / filename
        result["paths"][key] = file_path

        if not file_path.exists():
            result["valid"] = False
            result["missing_files"].append(filename)

    # Check VIX/Events staleness
    source_files = manifest.get("source_files", {})

    # VIX staleness check
    if source_files.get("vix_mtime"):
        try:
            vix_csv_path = Path(source_files.get("vix_path", "data/VIX_History.csv"))
            if vix_csv_path.exists():
                current_vix_mtime = int(vix_csv_path.stat().st_mtime)
                cached_vix_mtime = source_files["vix_mtime"]
                if current_vix_mtime != cached_vix_mtime:
                    from datetime import datetime
                    result["warnings"].append({
                        "type": "vix_stale",
                        "message": f"VIX file updated: cached {datetime.fromtimestamp(cached_vix_mtime).date()}, current {datetime.fromtimestamp(current_vix_mtime).date()}"
                    })
        except Exception:
            pass

    # Events staleness check
    if source_files.get("events_mtime"):
        try:
            events_csv_path = Path(source_files.get("events_path", "data/tsla_events_REAL.csv"))
            if events_csv_path.exists():
                current_events_mtime = int(events_csv_path.stat().st_mtime)
                cached_events_mtime = source_files["events_mtime"]
                if current_events_mtime != cached_events_mtime:
                    from datetime import datetime
                    result["warnings"].append({
                        "type": "events_stale",
                        "message": f"Events file updated: cached {datetime.fromtimestamp(cached_events_mtime).date()}, current {datetime.fromtimestamp(current_events_mtime).date()}"
                    })
        except Exception:
            pass

    # Print results
    if verbose:
        if result["valid"]:
            print(f"   ✅ Manifest validation: All files found")
            if result["warnings"]:
                for warning in result["warnings"]:
                    print(f"   ⚠️  {warning['message']}")
        else:
            print(f"   ❌ Manifest validation failed: {len(result['missing_files'])} files missing")
            for filename in result["missing_files"]:
                print(f"      Missing: {filename}")

    return result



def interactive_setup(args, profiler=None):
    """
    Interactive menu for training setup.

    Args:
        args: Initial argparse namespace
        profiler: Optional MemoryProfiler for diagnostic logging

    Returns:
        Updated args with user selections
    """
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        print("⚠️ InquirerPy not installed. Install with: pip install InquirerPy")
        print("Falling back to command-line args...")
        return args

    # =========================================================================
    # RESUME FROM CHECKPOINT (first option)
    # =========================================================================
    print("\n" + "=" * 70)
    print("🔄 RESUME TRAINING CHECK")
    print("=" * 70)

    # Look for existing checkpoints
    default_checkpoint = Path(args.output)
    checkpoint_candidates = []

    if default_checkpoint.exists():
        checkpoint_candidates.append(str(default_checkpoint))

    # Also check models/ directory
    models_dir = Path('models')
    if models_dir.exists():
        for pth_file in models_dir.glob('*.pth'):
            if str(pth_file) not in checkpoint_candidates:
                checkpoint_candidates.append(str(pth_file))

    if checkpoint_candidates:
        print(f"\n📁 Found {len(checkpoint_candidates)} checkpoint(s):")
        for cp in checkpoint_candidates:
            try:
                ckpt = torch.load(cp, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', '?')
                val_loss = ckpt.get('val_loss', 0)
                print(f"   • {cp} (epoch {epoch}, val_loss: {val_loss:.4f})")
            except Exception as e:
                print(f"   • {cp} (could not read: {e})")

        resume_choice = inquirer.select(
            message="Resume from checkpoint?",
            choices=[
                Choice(value=None, name="No - Start fresh training"),
                *[Choice(value=cp, name=f"Resume from: {cp}") for cp in checkpoint_candidates]
            ],
            default=None
        ).execute()

        if resume_choice:
            args.resume_checkpoint = resume_choice
            print(f"\n✓ Will resume from: {resume_choice}")
            # Load checkpoint to get saved args
            ckpt = torch.load(resume_choice, map_location='cpu', weights_only=False)
            saved_args = ckpt.get('args', {})

            # Restore key settings from checkpoint
            for key in ['device', 'batch_size', 'lr', 'epochs',
                        'train_start_year', 'train_end_year', 'use_geometric_base',
                        'use_fusion_head', 'multi_task']:
                if key in saved_args:
                    setattr(args, key, saved_args[key])

            args.resume_epoch = ckpt.get('epoch', 0) + 1  # Start from next epoch
            print(f"   Resuming from epoch {args.resume_epoch}")
            print(f"   Settings restored from checkpoint")

            # =========================================================
            # VERIFY CACHE PATH (features/labels/native TFs)
            # Uses existing cache scanning functions
            # =========================================================
            saved_cache_path = saved_args.get('shard_path', 'data/feature_cache')
            cache_exists = Path(saved_cache_path).exists()

            print(f"\n📂 Cache Path Verification")
            print(f"   Saved path: {saved_cache_path}")
            print(f"   Status: {'✓ Exists' if cache_exists else '✗ Not found'}")

            if cache_exists:
                # Use existing functions to scan cache
                manifests = load_cache_manifests(saved_cache_path)
                cache_pairs, layer_status = find_available_caches(saved_cache_path)

                # v5.9.2: Use layer_status for detailed reporting
                print(f"   Manifests: {'✓ ' + str(len(manifests)) + ' found' if manifests else '✗ None'}")
                print(f"   Cache pairs: {'✓ ' + str(len(cache_pairs)) + ' found' if cache_pairs else '✗ None'}")
                print(f"   TF Meta: {'✓' if layer_status['tf_meta_path'] else '✗'}")
                print(f"   Native TF sequences: {'✓ ' + str(layer_status['tf_sequence_count']) + ' files' if layer_status['training_ready'] else '✗ None'}")
                if layer_status['training_ready'] and not layer_status['can_regenerate_tf']:
                    print(f"   ⚠️  Chunks deleted - cannot regenerate TF sequences")
                    print(f"   ✓ Training ready ({layer_status['tf_size_gb']:.1f} GB TF + {layer_status['labels_size_gb']:.1f} GB labels)")

            cache_choice = inquirer.select(
                message="Cache directory:",
                choices=[
                    Choice(value='use_saved', name=f"Use saved path: {saved_cache_path}" + (" ✓" if cache_exists else " (not found!)")),
                    Choice(value='specify', name="Specify different cache path"),
                ],
                default='use_saved' if cache_exists else 'specify'
            ).execute()

            if cache_choice == 'specify':
                args.shard_path = inquirer.text(
                    message="Cache directory for features/labels:",
                    default=saved_cache_path
                ).execute()
            else:
                args.shard_path = saved_cache_path

            # Verify the chosen path exists
            if not Path(args.shard_path).exists():
                print(f"\n⚠️ Warning: Cache path '{args.shard_path}' does not exist!")
                print("   Training will fail if features/labels cannot be loaded.")

            # Ask if they want to continue with same epochs or extend
            extend_choice = inquirer.select(
                message=f"Original training was {saved_args.get('epochs', 10)} epochs. You're at epoch {args.resume_epoch}.",
                choices=[
                    Choice(value='continue', name=f"Continue to epoch {saved_args.get('epochs', 10)}"),
                    Choice(value='extend', name="Extend training (set new total epochs)"),
                ],
                default='continue'
            ).execute()

            if extend_choice == 'extend':
                args.epochs = inquirer.number(
                    message="New total epochs:",
                    default=saved_args.get('epochs', 10) + 10,
                    min_allowed=args.resume_epoch
                ).execute()
                args.epochs = int(args.epochs)

            return args  # Skip rest of setup, use saved settings
    else:
        print("\n   No existing checkpoints found. Starting fresh.")
        args.resume_checkpoint = None

    # Initial cache directory selection
    print("\n📂 Cache Directory Selection")
    cache_dir_default = getattr(args, 'shard_path', 'data/feature_cache_v6') or 'data/feature_cache_v6'
    args.shard_path = inquirer.text(
        message="Cache directory for features/labels:",
        default=cache_dir_default
    ).execute()

    # Scan for manifests in the chosen cache dir
    manifests = load_cache_manifests(args.shard_path)
    manifest_defaults = {}
    selected_manifest = None
    if manifests:
        selected_manifest = pick_manifest(manifests)
        if selected_manifest:
            manifest_defaults = selected_manifest.get("training_settings", {})
            # Apply precision / continuation mode defaults immediately
            project_config.TRAINING_PRECISION = selected_manifest.get("precision", project_config.TRAINING_PRECISION)
            if project_config.TRAINING_PRECISION == 'float64':
                project_config.NUMPY_DTYPE = np.float64
                project_config._TORCH_DTYPE = torch.float64
            else:
                project_config.NUMPY_DTYPE = np.float32
                project_config._TORCH_DTYPE = torch.float32
            project_config.CONTINUATION_MODE = selected_manifest.get("continuation_mode", project_config.CONTINUATION_MODE)
            ah_range = selected_manifest.get("adaptive_horizon_range", [project_config.ADAPTIVE_MIN_HORIZON, project_config.ADAPTIVE_MAX_HORIZON])
            if ah_range and len(ah_range) == 2 and all(v is not None for v in ah_range):
                project_config.ADAPTIVE_MIN_HORIZON, project_config.ADAPTIVE_MAX_HORIZON = ah_range

    # Scan for cached feature/label pairs in selected directory
    cache_pairs, layer_status = find_available_caches(args.shard_path)
    selected_cache_pair = None
    if cache_pairs:
        selected_cache_pair = pick_cache_pair(cache_pairs, layer_status)

    # Default cache behavior: reuse if a cache pair was selected, regenerate otherwise
    # v6.0: Also don't regenerate if v6 cache is selected (set later in menu)
    args.regenerate_cache = False if selected_cache_pair else True

    # v5.9.2: Set skip_chunk_validation if training layer is ready but generation layer is missing
    if layer_status['training_ready'] and not layer_status['can_regenerate_tf']:
        args.skip_chunk_validation = True
    else:
        args.skip_chunk_validation = False

    # Store layer_status for later use
    args.layer_status = layer_status

    # =========================================================================
    # v6.0 CACHE VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("📦 v6.0 DURATION-PRIMARY CACHE")
    print("=" * 70)

    # Get default v6 cache path from config
    v6_default_path = str(getattr(project_config, 'V6_CACHE_DIR', 'data/feature_cache_v6'))
    v6_cache_exists = Path(v6_default_path).exists()

    print(f"\n   Default path: {v6_default_path}")
    print(f"   Status: {'✓ Found' if v6_cache_exists else '✗ Not found'}")

    if v6_cache_exists:
        # Ask whether to validate
        validate_choice = inquirer.select(
            message="Validate v6 cache?",
            choices=[
                Choice(value='skip', name='Skip validation (faster) ⚡'),
                Choice(value='validate', name='Validate cache integrity (~30s)'),
            ],
            default='skip'
        ).execute()

        if validate_choice == 'validate':
            try:
                from src.ml.cache_v6 import validate_v6_cache, load_v6_cache
                print(f"\n   Validating v6 cache...")
                v6_valid = validate_v6_cache(v6_default_path)
                if v6_valid:
                    print(f"   ✓ v6 cache is valid")
                else:
                    print(f"   ✗ v6 cache validation failed")
                    v6_cache_exists = False
            except Exception as e:
                print(f"   ✗ v6 cache validation error: {e}")
                v6_cache_exists = False
        else:
            print(f"   ⚡ Skipping validation")

    # Build choices based on current state
    v6_choices = []
    if v6_cache_exists:
        v6_choices.append(Choice(value='use', name=f'Use existing v6 cache ✓'))
    if layer_status.get('training_ready', False):
        v6_choices.append(Choice(value='generate', name='Generate v6 cache from v5.9 features (~30-60 min)'))
    else:
        # v5.9 doesn't exist - offer to generate everything
        v6_choices.append(Choice(value='generate_all', name='Generate everything: v5.9 features → v6 cache (~2-3 hours)'))
    v6_choices.append(Choice(value='specify', name='Specify different v6 cache path'))
    v6_choices.append(Choice(value='skip', name='Skip v6 (use v5 legacy mode)'))

    v6_action = inquirer.select(
        message="v6.0 cache action:",
        choices=v6_choices,
        default='use' if v6_cache_exists else ('generate' if layer_status.get('training_ready') else 'skip')
    ).execute()

    args.use_v6_cache = True  # Default
    args.v6_cache_dir = v6_default_path

    if v6_action == 'use':
        print(f"   ✓ Using v6 cache: {v6_default_path}")
        args.regenerate_cache = False  # v6 cache is ready - don't regenerate

    elif v6_action == 'generate':
        print(f"\n   Generating v6 cache from v5.9 features...")
        print(f"   This may take 30-60 minutes.\n")

        # Find tf_meta file
        tf_meta_files = list(Path(args.shard_path).glob("tf_meta_v5.9*.json"))
        if not tf_meta_files:
            print(f"   ✗ No tf_meta_v5.9*.json found in {args.shard_path}")
            print(f"   Cannot generate v6 cache without v5.9 features.")
            args.use_v6_cache = False
        else:
            tf_meta_path = str(tf_meta_files[0])
            print(f"   Using: {tf_meta_path}")

            proceed = inquirer.confirm(
                message="Generate v6 cache now?",
                default=True
            ).execute()

            if proceed:
                try:
                    from src.ml.cache_v6 import generate_v6_cache
                    import pandas as pd

                    # Load raw OHLC
                    ohlc_path = project_config.RAW_DATA_FILE
                    print(f"   Loading raw OHLC from {ohlc_path}...")
                    raw_ohlc_df = pd.read_csv(ohlc_path, parse_dates=['timestamp'])
                    raw_ohlc_df.set_index('timestamp', inplace=True)

                    # Generate
                    output_dir = Path(v6_default_path)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    metadata = generate_v6_cache(
                        features_df=pd.DataFrame(),  # Empty - will load from v5.9
                        raw_ohlc_df=raw_ohlc_df,
                        output_dir=str(output_dir),
                        v5_cache_dir=args.shard_path,
                        verbose=True
                    )

                    print(f"\n   ✓ v6 cache generated successfully!")
                    print(f"   ✓ {len(metadata.get('timeframes', {}))} timeframes")

                except Exception as e:
                    print(f"\n   ✗ v6 cache generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    args.use_v6_cache = False
            else:
                print(f"   Skipping v6 generation. Using v5 legacy mode.")
                args.use_v6_cache = False

    elif v6_action == 'generate_all':
        # Generate everything from scratch: v5.9 features → v6 cache
        print(f"\n   🔨 FULL CACHE GENERATION")
        print(f"   Step 1: Generate v5.9 features (~1-2 hours)")
        print(f"   Step 2: Generate v6 cache (~30-60 min)")
        print(f"\n   Total estimated time: 2-3 hours\n")

        proceed = inquirer.confirm(
            message="Generate everything now?",
            default=True
        ).execute()

        if proceed:
            try:
                import pandas as pd
                from src.ml.features import TradingFeatureExtractor
                from src.ml.cache_v6 import generate_v6_cache

                # Step 1: Load raw data
                ohlc_path = project_config.RAW_DATA_FILE
                print(f"\n   📂 Loading raw OHLC from {ohlc_path}...")
                raw_df = pd.read_csv(ohlc_path, parse_dates=['timestamp'])
                raw_df.set_index('timestamp', inplace=True)
                print(f"   ✓ Loaded {len(raw_df):,} bars")

                # Load VIX data
                vix_data = None
                vix_path = getattr(project_config, 'VIX_DATA_FILE', None)
                if vix_path and Path(vix_path).exists():
                    print(f"   📂 Loading VIX data from {vix_path}...")
                    vix_data = load_vix_data(str(vix_path))
                    print(f"   ✓ Loaded VIX data")

                # Step 2: Generate v5.9 features
                print(f"\n   ⚙️  STEP 1/2: Generating v5.9 features...")
                print(f"   This will take 1-2 hours...\n")

                extractor = TradingFeatureExtractor()
                shard_path = Path(args.shard_path)
                shard_path.mkdir(parents=True, exist_ok=True)

                result = extractor.extract_features(
                    raw_df,
                    use_cache=False,  # Force regeneration
                    continuation=True,
                    use_chunking=True,
                    use_gpu='auto',
                    shard_storage_path=str(shard_path),
                    vix_data=vix_data,
                )

                print(f"\n   ✓ v5.9 features generated!")

                # Step 3: Generate v6 cache
                print(f"\n   ⚙️  STEP 2/2: Generating v6 cache...")

                output_dir = Path(v6_default_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                metadata = generate_v6_cache(
                    features_df=pd.DataFrame(),  # Empty - loads from v5.9
                    raw_ohlc_df=raw_df,
                    output_dir=str(output_dir),
                    v5_cache_dir=str(shard_path),
                    verbose=True
                )

                print(f"\n   ✓ v6 cache generated successfully!")
                print(f"   ✓ {len(metadata.get('timeframes', {}))} timeframes")

                # Update layer_status since we just generated everything
                args.layer_status['training_ready'] = True

            except Exception as e:
                print(f"\n   ✗ Generation failed: {e}")
                import traceback
                traceback.print_exc()
                args.use_v6_cache = False
        else:
            print(f"   Skipping generation. Using v5 legacy mode.")
            args.use_v6_cache = False

    elif v6_action == 'specify':
        args.v6_cache_dir = inquirer.text(
            message="v6 cache directory path:",
            default=v6_default_path
        ).execute()

        if Path(args.v6_cache_dir).exists():
            try:
                from src.ml.cache_v6 import validate_v6_cache
                if validate_v6_cache(args.v6_cache_dir):
                    print(f"   ✓ v6 cache valid: {args.v6_cache_dir}")
                else:
                    print(f"   ✗ v6 cache invalid")
                    args.use_v6_cache = False
            except Exception as e:
                print(f"   ✗ v6 cache error: {e}")
                args.use_v6_cache = False
        else:
            print(f"   ✗ Path not found: {args.v6_cache_dir}")
            args.use_v6_cache = False

    elif v6_action == 'skip':
        print(f"   Using v5 legacy mode (no v6 cache)")
        args.use_v6_cache = False
        args.v6_cache_dir = None

    def dflt(key, fallback):
        return manifest_defaults.get(key, fallback)

    print("\n" + "=" * 70)
    print("🎯 HIERARCHICAL LNN - INTERACTIVE TRAINING SETUP")
    print("=" * 70)

    # Detect hardware
    hw_info = get_hardware_info()

    print("\n📱 Hardware Detection:")
    if hw_info['cuda_available']:
        num_gpus = hw_info.get('cuda_device_count', 1)
        if num_gpus > 1:
            # Multi-GPU display
            print(f"  ✓ NVIDIA GPUs: {num_gpus}x detected ({hw_info['cuda_total_memory_gb']:.0f} GB total)")
            for gpu in hw_info['cuda_devices']:
                print(f"      GPU {gpu['index']}: {gpu['name']} ({gpu['vram_gb']:.0f} GB)")
        else:
            print(f"  ✓ NVIDIA GPU: {hw_info['cuda_device']} ({hw_info['cuda_memory_gb']:.1f} GB)")
    if hw_info['mps_available']:
        print(f"  ✓ Apple Silicon: {hw_info['mac_chip']} ({hw_info['total_ram_gb']:.0f} GB RAM)")
    print(f"  ✓ CPU: {hw_info['cpu_count']} threads")

    # Device selection (always show all options, mark availability)
    device_choices = []

    # CUDA option (always show, mark if detected) - includes multi-GPU info
    if hw_info['cuda_available']:
        num_gpus = hw_info.get('cuda_device_count', 1)
        total_vram = hw_info.get('cuda_total_memory_gb', hw_info['cuda_memory_gb'])
        if num_gpus > 1:
            gpu_label = f'NVIDIA GPUs ({num_gpus}x, {total_vram:.0f}GB total) - Fastest ⚡ [Multi-GPU]'
        else:
            gpu_label = f'NVIDIA GPU ({hw_info["cuda_device"]}, {total_vram:.0f}GB) - Fastest ⚡ [Detected]'
        device_choices.append(Choice(value='cuda', name=gpu_label))
    else:
        device_choices.append(Choice(value='cuda', name='NVIDIA GPU (CUDA) - Fastest ⚡ [Not Detected]'))

    # MPS option (only show if available)
    if hw_info['mps_available']:
        device_choices.append(Choice(value='mps', name='Apple Silicon GPU (MPS) - Fast 🍎 [Detected]'))

    # CPU option (always available)
    device_choices.append(Choice(value='cpu', name='CPU - Slowest 🐢'))

    # Determine default (best available device)
    if hw_info['cuda_available']:
        default_device = 'cuda'
    elif hw_info['mps_available']:
        default_device = 'mps'
    else:
        default_device = 'cpu'

    print()
    args.device = inquirer.select(
        message="Select compute device:",
        choices=device_choices,
        default=dflt('device', default_device)
    ).execute()

    # Validate selection
    if args.device == 'cuda' and not hw_info['cuda_available']:
        print("\n⚠️  WARNING: CUDA selected but not detected on this system.")
        print("   Training will fail if CUDA is truly unavailable.")
        print("   This option is provided for external GPU scenarios.")
        proceed = inquirer.confirm(
            message="Continue with CUDA anyway?",
            default=False
        ).execute()

        if not proceed:
            args.device = default_device
            print(f"   Switched to {args.device.upper()}")

    # ==========================================================================
    # GPU Mode Selection (Single GPU vs Multi-GPU DDP)
    # ==========================================================================
    args.gpu_mode = 'single'
    args.use_ddp = False

    if args.device == 'cuda' and hw_info.get('cuda_device_count', 1) > 1:
        num_gpus = hw_info['cuda_device_count']
        gpu_devices = hw_info.get('cuda_devices', [])

        print(f"\n🎮 Multiple GPUs Detected ({num_gpus})")

        gpu_mode_choices = [
            Choice(value='single', name='Single GPU - All features available (adaptive, preload, etc.)'),
            Choice(value='multi_ddp', name='Multi-GPU (DDP) - Maximum throughput, preload only'),
        ]

        args.gpu_mode = inquirer.select(
            message="GPU configuration:",
            choices=gpu_mode_choices,
            default='single'
        ).execute()

        if args.gpu_mode == 'single':
            # Let user pick which GPU
            if gpu_devices:
                gpu_choices = [
                    Choice(value=f'cuda:{gpu["index"]}',
                           name=f'GPU {gpu["index"]}: {gpu["name"]} ({gpu["vram_gb"]:.0f} GB)')
                    for gpu in gpu_devices
                ]
            else:
                gpu_choices = [
                    Choice(value=f'cuda:{i}', name=f'GPU {i}')
                    for i in range(num_gpus)
                ]

            args.device = inquirer.select(
                message="Select GPU:",
                choices=gpu_choices,
                default='cuda:0'
            ).execute()
            args.use_ddp = False
            args.num_ddp_gpus = 1
            print(f"   ✓ Using {args.device} in single-GPU mode")
        else:
            # Multi-GPU DDP mode - let user choose how many GPUs
            args.use_ddp = True
            args.device = 'cuda'  # DDP will handle device assignment per process

            # Build GPU count choices
            gpu_count_choices = []
            for n in range(2, num_gpus + 1):
                if n == num_gpus:
                    gpu_count_choices.append(
                        Choice(value=n, name=f'{n} GPUs (all available)')
                    )
                else:
                    gpu_count_choices.append(
                        Choice(value=n, name=f'{n} GPUs')
                    )

            args.num_ddp_gpus = inquirer.select(
                message="How many GPUs to use?",
                choices=gpu_count_choices,
                default=num_gpus  # Default to all
            ).execute()

            print(f"\n   ✓ Multi-GPU DDP mode: {args.num_ddp_gpus} GPUs")
            print(f"   Will automatically spawn {args.num_ddp_gpus} training processes")
            print(f"   Using GPUs: cuda:0 through cuda:{args.num_ddp_gpus - 1}")
            print(f"   Effective batch size: {args.batch_size} x {args.num_ddp_gpus} = {args.batch_size * args.num_ddp_gpus}")
            print(f"   Note: DDP forces Full Pre-merge mode (adaptive not compatible)")

    # Consolidated precision menu for CUDA
    # v5.7.2: Removed AMP (FP16) - caused NaN issues and duplicated code
    args.precision_mode = 'fp32'  # Track the user's choice
    args.use_tf32 = False  # v5.3: TF32 Tensor Core acceleration

    if args.device.startswith('cuda'):
        print()
        precision_choice = inquirer.select(
            message="Training precision:",
            choices=[
                Choice(value='fp32_tf32', name="FP32 with TF32 Tensor Cores ⭐ Recommended (~2x faster)"),
                Choice(value='fp32', name="FP32 Standard (no acceleration)"),
                Choice(value='fp64', name="FP64 - Double precision (very slow)")
            ],
            default=dflt('precision_mode', 'fp32_tf32')
        ).execute()

        args.precision_mode = precision_choice

        if precision_choice == 'fp32_tf32':
            args.use_tf32 = True
            torch.set_float32_matmul_precision('medium')  # Enable TF32
            project_config.TRAINING_PRECISION = 'float32'
            project_config.NUMPY_DTYPE = np.float32
            project_config._TORCH_DTYPE = torch.float32
            print("   ⭐ FP32 with TF32 Tensor Cores")
            print("   → Ampere+ GPU acceleration (~2x matmul speedup)")
            print("   → Stable (no NaN, same range as FP32)")
        elif precision_choice == 'fp32':
            args.use_tf32 = False
            torch.set_float32_matmul_precision('highest')  # Disable TF32
            project_config.TRAINING_PRECISION = 'float32'
            project_config.NUMPY_DTYPE = np.float32
            project_config._TORCH_DTYPE = torch.float32
            print("   → FP32 Standard - Highest precision (slower)")
        else:  # fp64
            args.use_tf32 = False
            project_config.TRAINING_PRECISION = 'float64'
            project_config.NUMPY_DTYPE = np.float64
            project_config._TORCH_DTYPE = torch.float64
            print("   → FP64 - Maximum precision (very slow, high memory)")

        # torch.compile option (CUDA only, PyTorch 2.0+)
        print()
        if hasattr(torch, 'compile'):
            args.use_compile = inquirer.confirm(
                message="Enable torch.compile? (JIT compilation, first batch takes 5-15 min but subsequent batches faster)",
                default=True
            ).execute()
            if args.use_compile:
                print("   ⚡ torch.compile enabled - first forward pass will show progress updates")
            else:
                print("   → torch.compile disabled - standard eager mode")
        else:
            args.use_compile = False
            print("   ℹ️  torch.compile not available (requires PyTorch 2.0+)")

        # torch.compile verbose debug option (only if compile is enabled)
        if args.use_compile:
            args.compile_verbose = inquirer.confirm(
                message="Enable verbose torch.compile output? (shows dynamo/inductor logs during first forward pass)",
                default=False
            ).execute()
            if args.compile_verbose:
                print("   🔍 Verbose compile output enabled - will show graph breaks, kernel fusion, etc.")
            else:
                print("   → Standard compile output (minimal logging)")
        else:
            args.compile_verbose = False

        # RAM detection
        try:
            import psutil
            detected_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            detected_ram = 0

        args.container_ram_gb = 0  # Use psutil detection by default
        if profiler:
            profiler.log_info(f"RAM_DETECTED | psutil_ram={detected_ram:.0f}GB")

    elif args.device == 'mps':
        # MPS only supports float32
        print()
        print("   ℹ️  MPS uses FP32 precision (float64 not supported)")
        project_config.TRAINING_PRECISION = 'float32'
        project_config.NUMPY_DTYPE = np.float32
        project_config._TORCH_DTYPE = torch.float32

    else:  # CPU
        print()
        precision_choice = inquirer.select(
            message="Training precision:",
            choices=[
                Choice(value='fp32', name="FP32 - Standard precision (recommended)"),
                Choice(value='fp64', name="FP64 - Maximum precision (slower)")
            ],
            default='fp32'
        ).execute()

        args.precision_mode = precision_choice

        if precision_choice == 'fp64':
            project_config.TRAINING_PRECISION = 'float64'
            project_config.NUMPY_DTYPE = np.float64
            project_config._TORCH_DTYPE = torch.float64
            print("   → FP64 - Maximum precision")
        else:
            project_config.TRAINING_PRECISION = 'float32'
            project_config.NUMPY_DTYPE = np.float32
            project_config._TORCH_DTYPE = torch.float32
            print("   → FP32 - Standard precision")

    # Data loading workers
    print()

    # Detect RAM for guidance
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        container_ram = getattr(args, 'container_ram_gb', 0)
        if container_ram > 0:
            total_ram_gb = container_ram
    except:
        total_ram_gb = 0

    default_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}.get(args.device, 2)

    if total_ram_gb > 0:
        guidance = f"\n   ℹ️  RAM: {total_ram_gb:.0f}GB available. With large datasets, each worker uses extra RAM."
    else:
        guidance = ""

    args.num_workers = int(inquirer.number(
        message=f"Data loading workers (CPU threads for batch prep, recommended: {default_workers}):{guidance}",
        default=dflt('num_workers', default_workers),
        min_allowed=0
        # No max cap - user can set based on CPU cores available
    ).execute())

    if args.num_workers != default_workers:
        if args.device == 'mps' and args.num_workers > 2:
            print(f"   ⚠️  Using {args.num_workers} workers on MPS (default: 2)")
            print(f"      More workers = faster but more RAM usage (unified memory)")
        elif args.device == 'cuda' and args.num_workers != 4:
            print(f"   → Using {args.num_workers} workers for CUDA (default: 4)")

    print(f"   ℹ️  Training uses ALL {args.device.upper()} cores, workers are for data loading only")

    # Memory profiling option (for debugging RAM usage)
    args.memory_profile = inquirer.confirm(
        message="Enable memory profiling? (logs to logs/memory_debug.log)",
        default=True
    ).execute()

    # v5.9.3: Source data loading option (preload TF sequences to RAM)
    print()
    total_ram = hw_info.get('total_ram_gb', 16)
    tf_sequences_gb = 3.2  # Known size: 11 TF sequence files

    source_choices = [
        "Memory-mapped (default, ~0 GB extra RAM)",
        f"Preload to RAM (~{tf_sequences_gb:.1f} GB extra RAM, faster batching)"
    ]

    # Auto-select preload if system has enough RAM (12+ GB)
    default_source = source_choices[1] if total_ram >= 12 else source_choices[0]

    source_choice = inquirer.select(
        message="Source data loading:",
        choices=source_choices,
        default=default_source
    ).execute()

    if "Preload" in source_choice:
        args.preload_tf_to_ram = True
        print(f"   📥 Will preload {tf_sequences_gb:.1f} GB TF sequences to RAM at startup")
        print(f"      → Eliminates disk I/O during training")
    else:
        args.preload_tf_to_ram = False
        print("   → Memory-mapped (OS page cache handles caching)")

    # v5.9.4: Sampler choice - only ask when data is preloaded to RAM
    # When mmap, always use chunk-based sampler (ShuffleBufferSampler) for I/O optimization
    # When preloaded, user can choose since random access is fast in RAM
    if args.preload_tf_to_ram:
        print()
        sampler_choices = [
            "Chunk-based (ShuffleBufferSampler) - Better cache locality (Recommended)",
            "Random (DistributedSampler) - True global shuffle"
        ]

        sampler_choice = inquirer.select(
            message="Sampler strategy (data is in RAM, both are fast):",
            choices=sampler_choices,
            default=sampler_choices[0]
        ).execute()

        if "Chunk-based" in sampler_choice:
            args.use_chunk_sampler = True
            print("   → Chunk-based sampler (sequential chunks, local shuffle)")
            print("      Works for both single-GPU and multi-GPU DDP")
        else:
            args.use_chunk_sampler = False
            print("   → Random sampler (DistributedSampler for DDP, shuffle for single-GPU)")
    else:
        # mmap mode: always use chunk-based sampler for I/O optimization
        args.use_chunk_sampler = True
        print("   → Using chunk-based sampler (optimized for mmap access)")

    # v5.9.6: Sample selection strategy (boundary sampling)
    print()
    boundary_sampling_choice = inquirer.select(
        message="Sample selection strategy:",
        choices=[
            "All samples - Train on every bar (standard, ~417K samples)",
            "Boundary samples - Train only near channel breaks (10-20x fewer samples, faster epochs) ⚡"
        ],
        default="All samples - Train on every bar (standard, ~417K samples)"
    ).execute()

    if "Boundary" in boundary_sampling_choice:
        args.use_boundary_sampling = True

        # Choose boundary sampling mode
        mode_choice = inquirer.select(
            message="Boundary sampling mode:",
            choices=[
                "Approaching breaks - Train near channel endings (duration ≤ threshold)",
                "New starts - Train on fresh channels (duration ≥ threshold) ⭐ Your original idea",
                "Both - Train on breaks AND starts (most variety)"
            ],
            default="New starts - Train on fresh channels (duration ≥ threshold) ⭐ Your original idea"
        ).execute()

        if "Approaching" in mode_choice:
            args.boundary_mode = "breaks"
        elif "New starts" in mode_choice:
            args.boundary_mode = "starts"
        else:
            args.boundary_mode = "both"

        # Threshold selection (meaning depends on mode)
        if args.boundary_mode in ["breaks", "both"]:
            threshold_message = "Break threshold (bars until break):" if args.boundary_mode == "breaks" else "Threshold (bars from break/start):"
            threshold_choice = inquirer.select(
                message=threshold_message,
                choices=[
                    "2 bars - Very strict",
                    "5 bars - Strict ⭐ Recommended",
                    "10 bars - Moderate",
                    "20 bars - Loose"
                ],
                default="5 bars - Strict ⭐ Recommended"
            ).execute()
            args.boundary_threshold = int(threshold_choice.split()[0])
        else:
            # New starts mode: use threshold for minimum duration
            threshold_choice = inquirer.select(
                message="Minimum duration for 'fresh' channel:",
                choices=[
                    "10 bars - Very fresh channels",
                    "20 bars - Fresh channels ⭐ Recommended",
                    "30 bars - Established channels",
                    "50 bars - Strong channels only"
                ],
                default="20 bars - Fresh channels ⭐ Recommended"
            ).execute()
            args.boundary_threshold = int(threshold_choice.split()[0])

        # Print selected configuration
        if args.boundary_mode == "breaks":
            print(f"   ⚡ Boundary sampling: Approaching breaks (≤{args.boundary_threshold} bars)")
            print(f"      → Train on samples near channel endings")
        elif args.boundary_mode == "starts":
            print(f"   ⚡ Boundary sampling: New channel starts (≥{args.boundary_threshold} bars)")
            print(f"      → Train on fresh channels with high duration")
        else:
            print(f"   ⚡ Boundary sampling: Both breaks and starts (≤{args.boundary_threshold} or ≥{args.boundary_threshold} bars)")
            print(f"      → Train on transitions (endings + beginnings)")
        print(f"      → Faster epochs, focuses on high-information samples")
    else:
        args.use_boundary_sampling = False
        args.boundary_mode = None
        args.boundary_threshold = 5
        print("   → Standard sampling (all bars)")

    # v5.3.2: Pre-stacking option for faster epochs
    print()

    # v5.3.2: Calculate realistic RAM requirements per mode
    # Native TF mode: ~3.6 MB per sample
    bytes_per_sample = 3.6e6
    estimated_train_samples = 1_400_000  # Approximate
    batch_size_estimate = 256  # Typical
    batches_per_epoch = estimated_train_samples // batch_size_estimate
    bytes_per_batch = batch_size_estimate * bytes_per_sample

    # Full epoch RAM (2 epochs in memory)
    full_epoch_ram_gb = (2 * batches_per_epoch * bytes_per_batch) / 1e9

    # Rolling buffer RAM (calculate optimal buffer size)
    safety_margin = 0.8  # Use 80% of available RAM
    usable_ram_gb = total_ram * safety_margin
    max_buffer_batches = int((usable_ram_gb * 1e9) / bytes_per_batch)
    rolling_buffer_ram_gb = (max_buffer_batches * bytes_per_batch) / 1e9

    # Pre-stacking mode selection
    prestack_choices = [
        "Disabled - Standard DataLoader (safe, ~9% slower) ⭐ Recommended",
        f"Rolling Buffer - Pre-stack {max_buffer_batches} batches (~{rolling_buffer_ram_gb:.0f}GB RAM, ~9% faster)",
        f"Full Epoch - Pre-stack all batches (~{full_epoch_ram_gb:.0f}GB RAM, ~40% faster) ⚠️ Needs massive RAM"
    ]

    prestack_choice = inquirer.select(
        message="Batch pre-stacking mode:",
        choices=prestack_choices,
        default=prestack_choices[0]
    ).execute()

    # Parse selection
    if "Disabled" in prestack_choice:
        args.use_prestack = False
        args.prestack_mode = None
        print("   → Standard DataLoader (collate during training)")
    elif "Rolling Buffer" in prestack_choice:
        args.use_prestack = True
        args.prestack_mode = 'rolling'
        args.prestack_buffer_size = max_buffer_batches
        print(f"   📦 Rolling buffer enabled ({max_buffer_batches} batches, {rolling_buffer_ram_gb:.0f} GB RAM)")
        print(f"      → Pre-stacks next {max_buffer_batches} batches ahead")
        print(f"      → Background thread maintains rolling window")
        print(f"      → Eliminates collate wait (~9% faster)")
        print()
    else:  # Full Epoch
        args.use_prestack = True
        args.prestack_mode = 'full_epoch'
        args.prestack_buffer_size = None
        print(f"   📦 Full epoch pre-stacking enabled")
        print(f"      ⚠️ WARNING: Needs ~{full_epoch_ram_gb:.0f}GB RAM (you have {total_ram:.0f}GB)")
        print(f"      → Will likely OOM if RAM insufficient!")
        print()

    # v5.3.2: Pinned memory option (only if prestack enabled)
    args.use_pinned_prestack = False
    if args.use_prestack:
        # Pinned memory sub-option
        pinned_msg = "Use pinned memory for faster GPU transfer? (⚠️ uses locked RAM, may cause issues if RAM tight)"
        args.use_pinned_prestack = inquirer.confirm(
            message=pinned_msg,
            default=False  # Default OFF - safer
        ).execute()

        if args.use_pinned_prestack:
            print("   📌 Pinned memory enabled (faster CPU→GPU transfer)")
            print("      ⚠️ Warning: If you see CUDA OOM, disable this option")
        else:
            print("   → Standard memory (safe, slightly slower transfer)")

    # Get recommended batch size
    total_ram = hw_info.get('total_ram_gb', 16)
    recommended_batch = get_recommended_batch_size(args.device, total_ram)

    # Training data range
    print()
    # v6.0: If using self-contained v6 cache, skip date questions (data is in cache)
    v6_self_contained = getattr(args, 'use_v6_cache', False) and getattr(args, 'v6_cache_dir', None)
    if v6_self_contained:
        # v6.0: Extract date range from v6 cache metadata (fast, metadata only)
        try:
            from src.ml.cache_v6 import load_v6_metadata
            v6_meta = load_v6_metadata(args.v6_cache_dir)
            v6_date_range = v6_meta.get('data_range', {})
            cache_start = int(str(v6_date_range.get('start', '2015'))[:4])
            cache_end = int(str(v6_date_range.get('end', '2025'))[:4])
        except Exception:
            cache_start, cache_end = 2015, 2025

        print(f"\n   📅 v6 cache covers: {cache_start}-{cache_end}")

        # Let user choose full range or subset
        date_range_choice = inquirer.select(
            message="Training date range:",
            choices=[
                Choice(value='full', name=f'Full range ({cache_start}-{cache_end}) - Maximum data'),
                Choice(value='subset', name='Custom range - Select start/end within cache'),
            ],
            default='full'
        ).execute()

        if date_range_choice == 'full':
            args.train_start_year = cache_start
            args.train_end_year = cache_end
            print(f"   ✓ Using full range: {cache_start}-{cache_end}")
        else:
            args.train_start_year = int(inquirer.number(
                message="Training start year:",
                default=cache_start,
                min_allowed=cache_start,
                max_allowed=cache_end - 1
            ).execute())

            args.train_end_year = int(inquirer.number(
                message="Training end year:",
                default=cache_end,
                min_allowed=args.train_start_year + 1,
                max_allowed=cache_end
            ).execute())
            print(f"   ✓ Using custom range: {args.train_start_year}-{args.train_end_year}")
    elif selected_cache_pair:
        # Lock to cached date range to avoid regenerating
        m_start = manifest_defaults.get('train_start_year', None)
        m_end = manifest_defaults.get('train_end_year', None)
        if not m_start or not m_end:
            # Fallback to manifest date_range if training settings missing
            dr = manifest_defaults.get('date_range', {})
            try:
                m_start = int(str(dr.get('start', ''))[:4])
                m_end = int(str(dr.get('end', ''))[:4])
            except Exception:
                # v5.9.2: Extract dates from cache_key when no manifest exists
                # cache_key format: v5.9.0_..._20150102_20250927_1692233_...
                cache_key = selected_cache_pair.get('cache_key', '')
                try:
                    # Parse dates from cache_key (YYYYMMDD format after version prefix)
                    import re
                    date_match = re.search(r'_(\d{8})_(\d{8})_', cache_key)
                    if date_match:
                        m_start = int(date_match.group(1)[:4])
                        m_end = int(date_match.group(2)[:4])
                        print(f"   ℹ️  No manifest found, extracted dates from cache_key: {m_start}-{m_end}")
                    else:
                        m_start, m_end = 2015, 2025  # Updated default
                except Exception:
                    m_start, m_end = 2015, 2025  # Updated default
        args.train_start_year = int(m_start)
        args.train_end_year = int(m_end)
        print(f"\n   📅 Using cached date range: {args.train_start_year}-{args.train_end_year}")
    else:
        args.train_start_year = int(inquirer.number(
            message="Training data start year:",
            default=dflt('train_start_year', 2015),
            min_allowed=2010,
            max_allowed=2023
        ).execute())

        args.train_end_year = int(inquirer.number(
            message="Training data end year:",
            default=dflt('train_end_year', 2022),
            min_allowed=int(args.train_start_year),  # Explicit int conversion
            max_allowed=2100
        ).execute())

    # Validate and notify about warmup impact
    if not selected_cache_pair and not v6_self_contained:
        warmup_years = project_config.MIN_LOOKBACK_MONTHS / 12 if hasattr(project_config, 'MIN_LOOKBACK_MONTHS') else 2.5
        requested_years = args.train_end_year - args.train_start_year
        effective_start_year = args.train_start_year + warmup_years
        effective_years = args.train_end_year - effective_start_year

        print(f"\n   📅 Training Date Range Analysis:")
        print(f"   Requested: {args.train_start_year}-{args.train_end_year} ({requested_years} years)")
        print(f"   Warmup required: {warmup_years} years (257,400 bars for 14-window system)")
        print(f"   ")
        print(f"   ⚠️  IF your CSV starts at {args.train_start_year}:")
        print(f"       Effective training: {effective_start_year:.1f}-{args.train_end_year} ({effective_years:.1f} years)")
        print(f"       → First {warmup_years} years used for warmup (ensures complete feature history)")
        print(f"   ")
        print(f"   💡 To train from {args.train_start_year}, you need CSV data from {args.train_start_year - warmup_years:.1f}")
        print(f"   ")

        if effective_years < 2.0:
            print(f"   ⚠️  Warning: Only {effective_years:.1f} years of usable training data after warmup!")
            proceed = inquirer.confirm(
                message=f"Continue with {args.train_start_year}-{args.train_end_year} anyway?",
                default=False
            ).execute()
            if not proceed:
                print("   Exiting - please adjust dates or get more historical data")
                sys.exit(0)
            else:
                print(f"   ⚠️  Continuing with limited data ({effective_years:.1f} years)")
        else:
            print(f"   ✓ Good! {effective_years:.1f} years of quality training data after warmup")

    # GPU Acceleration option (skip if using cached features)
    print()
    # v6.0: Include v6_self_contained in will_use_cache check
    will_use_cache = not getattr(args, 'regenerate_cache', True) or v6_self_contained
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if v6_self_contained:
        # v6 cache is self-contained - no feature extraction needed
        args.use_gpu_features = False
        print("⚡ GPU Acceleration: Skipped (v6 cache is self-contained)")
    elif selected_cache_pair and will_use_cache:
        args.use_gpu_features = False
        print("⚡ GPU Acceleration: Skipped (using cached features/labels)")
    else:
        # Check if GPU is available
        if torch.cuda.is_available():
            gpu_type = 'CUDA'
            gpu_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            gpu_type = 'MPS'
            gpu_name = 'Apple Silicon'
        else:
            gpu_type = None
            gpu_name = 'Not Available'

        if gpu_available:
            print(f"⚡ GPU Acceleration Available: {gpu_name} ({gpu_type})")

            if will_use_cache:
                # Cache will be loaded - GPU won't be used this run
                message = "Use GPU acceleration for feature extraction? (Note: Cache will be loaded this run, GPU only applies if cache regenerates)"
            else:
                # Will calculate features - GPU will be used
                message = "Use GPU acceleration for feature extraction? (Speeds up calculation: ~45 mins → ~3 mins)"

            args.use_gpu_features = inquirer.select(
                message=message,
                choices=[
                    Choice(True, f"Yes - Use {gpu_type} GPU (10-20x faster for calculation) ⚡"),
                    Choice(False, "No - Use CPU (reliable, compatible) 💾")
                ],
                default=True  # Default to GPU if available
            ).execute()

            if args.use_gpu_features:
                if will_use_cache:
                    print(f"   ℹ️  GPU selected (will be used if cache needs regeneration)")
                else:
                    print(f"   ⚡ GPU will accelerate feature calculation (~3 minutes instead of ~45 minutes)")
            else:
                print(f"   💾 CPU will be used for feature calculation")
        else:
            # No GPU available
            args.use_gpu_features = False
            print(f"⚡ GPU Acceleration: Not Available (CPU will be used)")

    # Parallel Processing option (for CPU mode)
    print()
    import multiprocessing as std_mp
    n_cores = std_mp.cpu_count()

    # Only show parallel option if:
    # 1. Not using GPU (GPU and parallel are incompatible)
    # 2. Have more than 2 cores
    # 3. Will be calculating features (not just loading cache)
    show_parallel = not args.use_gpu_features and n_cores > 2 and not will_use_cache

    if show_parallel:
        print(f"🚀 Parallel Processing Available: {n_cores} CPU cores detected")

        args.use_parallel = inquirer.select(
            message="Use parallel processing for channel calculations? (CPU mode)",
            choices=[
                Choice(True, f"Yes - Use up to {min(n_cores-1, 8)} cores (5-8x faster) 🚀"),
                Choice(False, "No - Sequential processing (uses less memory) 💾")
            ],
            default=True
        ).execute()

        if args.use_parallel:
            cores_to_use = min(n_cores - 1, 8)
            print(f"   ✓ Will use {cores_to_use} CPU cores for parallel channel calculation")
            print(f"   ℹ️  Expected speedup: ~{cores_to_use-1}x to {cores_to_use}x faster")
        else:
            print(f"   💾 Will use sequential processing (single core)")
    else:
        args.use_parallel = False
        if args.use_gpu_features:
            # GPU mode - parallel not compatible
            pass  # GPU already selected, no message needed
        elif will_use_cache:
            # Will load from cache - no calculation needed
            pass  # Cache loading, no message needed
        else:
            # Less than 3 cores
            print(f"🚀 Parallel Processing: Not available (only {n_cores} cores detected)")

    # Feature extraction parallel workers (only if using parallel)
    if args.use_parallel:
        print()

        # Calculate RAM-based recommendation (each worker uses ~15GB)
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9
        ram_based_workers = max(1, int(total_ram_gb / 15))
        core_based_workers = min(n_cores - 1, 8)

        # Use the more conservative of RAM-based or core-based
        default_feature_workers = min(ram_based_workers, core_based_workers)

        print(f"   RAM: {total_ram_gb:.1f}GB (each worker uses ~15GB)")
        print(f"   Safe workers for your RAM: {ram_based_workers}")

        args.feature_workers = int(inquirer.number(
            message=f"Feature extraction workers (recommended: {default_feature_workers}, max safe: {ram_based_workers}):",
            default=default_feature_workers,
            min_allowed=1,
            max_allowed=128
        ).execute())

        # Warn if user picks more than RAM allows
        if args.feature_workers > ram_based_workers:
            print(f"   ⚠️  WARNING: {args.feature_workers} workers × 15GB = {args.feature_workers * 15}GB needed")
            print(f"   ⚠️  You only have {total_ram_gb:.1f}GB RAM - expect heavy swapping or OOM!")

        print(f"   → Using {args.feature_workers} workers for feature extraction")
        project_config.MAX_PARALLEL_WORKERS = args.feature_workers

    # Chunked Feature Extraction option
    print()

    # Only show if we'll be calculating features (not just loading cache)
    show_chunking = not will_use_cache

    if show_chunking:
        # Detect RAM
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9

        print(f"💾 Memory Management")
        print(f"   Total RAM: {total_ram_gb:.1f} GB")

        # Smart default based on RAM
        if total_ram_gb < 32:
            default_chunking = True
            recommendation = "⭐ Recommended for your system"
        elif total_ram_gb < 64:
            default_chunking = True
            recommendation = "Recommended (safer)"
        else:
            default_chunking = False
            recommendation = "Optional (you have plenty of RAM)"

        args.use_chunking = inquirer.select(
            message=f"Use chunked feature extraction? {recommendation}",
            choices=[
                Choice(True, "Yes - Process in 1-year chunks (2-5GB RAM, ~10% slower) 💾"),
                Choice(False, "No - Process all at once (20-40GB RAM, fastest) ⚡")
            ],
            default=default_chunking
        ).execute()

        if args.use_chunking:
            print(f"   ✓ Will process features in 1-year chunks")
            print(f"   ℹ️  Peak RAM during extraction: ~2-5GB")
            # Respect the initial cache directory selection for shard storage
            print(f"   ✓ Shard storage path: {args.shard_path} (from initial selection)")
        else:
            print(f"   ⚡ Will process all features at once")
            print(f"   ⚠️  Peak RAM during extraction: ~20-40GB")
    else:
        # Cache will be loaded - chunking doesn't apply
        args.use_chunking = False

    # v5.9.2: Native TF generation option - context-aware based on layer_status
    import glob
    chunk_meta_files = glob.glob(str(Path(args.shard_path or 'data/feature_cache') / 'features_mmap_meta_*.json'))
    has_existing_chunks = len(chunk_meta_files) > 0

    # Check layer_status for context-aware menu
    can_regenerate_tf = layer_status.get('can_regenerate_tf', False) if layer_status else has_existing_chunks
    training_ready = layer_status.get('training_ready', False) if layer_status else False

    # v6.0: Skip native TF generation menu when v6 cache is self-contained
    if v6_self_contained:
        print(f"\n   ✓ Native TF: Using v6 cache (self-contained)")
        args.generate_native_tf = None
        args.native_tf_streaming = True
    elif can_regenerate_tf or args.use_chunking:
        # Chunks exist - can regenerate TF sequences
        print()
        print("=" * 60)
        print("  NATIVE TIMEFRAME GENERATION")
        print("=" * 60)
        if has_existing_chunks:
            print(f"   Found {len(chunk_meta_files)} existing chunk metadata file(s)")
            if layer_status:
                print(f"   Chunk shards: {layer_status.get('chunk_count', 0)} chunks ({layer_status.get('chunks_size_gb', 0):.1f} GB)")

        args.generate_native_tf = inquirer.select(
            message="Generate native timeframe sequences from chunk shards?",
            choices=[
                Choice('skip', "Skip - Use existing tf_meta or generate during training"),
                Choice('streaming', "Yes (Streaming) - Low RAM (~5-8GB), processes one chunk at a time 💾"),
                Choice('full_load', "Yes (Full Load) - Fast but needs ~50GB RAM ⚡"),
            ],
            default='skip'
        ).execute()

        if args.generate_native_tf == 'streaming':
            print(f"   ✓ Will generate native TF sequences (streaming mode)")
            print(f"   💾 Peak RAM: ~5-8GB")
            args.native_tf_streaming = True
        elif args.generate_native_tf == 'full_load':
            print(f"   ✓ Will generate native TF sequences (full load mode)")
            print(f"   💾 Peak RAM: ~50GB - ensure sufficient memory!")
            args.native_tf_streaming = False
        else:
            print(f"   → Skipping native TF generation")
            args.generate_native_tf = None
            args.native_tf_streaming = True  # Default for CLI usage
    elif training_ready:
        # v5.9.2: Chunks deleted but native TF sequences exist - training is ready
        print()
        print("=" * 60)
        print("  NATIVE TIMEFRAME STATUS")
        print("=" * 60)
        print(f"   ✓ Native TF sequences: READY ({layer_status.get('tf_sequence_count', 0)} files)")
        print(f"   ⚠️  Chunk shards: DELETED (cannot regenerate TF sequences)")
        print(f"   ✓ Training will use existing native TF sequences")
        print(f"   ℹ️  To regenerate TF sequences, you would need to re-extract features (~5-8 hours)")
        args.generate_native_tf = None
        args.native_tf_streaming = True  # Default for CLI usage
    else:
        print(f"\n   ℹ️  No chunk shards found. Native TF will be generated during training if needed.")
        args.generate_native_tf = None
        args.native_tf_streaming = True  # Default for CLI usage

    # Note: Precision is now selected in the consolidated menu right after device selection
    # This section only handles continuation mode

    # v5.2/v5.3: Continuation mode hardcoded to adaptive_labels
    # Actual-duration targets use continuation_labels duration_bars anyway
    continuation_mode = 'adaptive_labels'
    project_config.CONTINUATION_MODE = continuation_mode

    # Use cached mode if available
    if selected_cache_pair and will_use_cache:
        continuation_mode = project_config.CONTINUATION_MODE

    print(f"\n   ✓ Continuation mode: {continuation_mode} (v5.2 uses actual duration from labels)")

    # Model parameters
    print()

    # Model capacity selection
    capacity_choices = [
        Choice(value=2.0, name='Standard (256 total, 128 output) - Recommended ⭐'),
        Choice(value=3.0, name='High (384 total, 128 output) - Better accuracy, slower'),
        Choice(value=4.0, name='Maximum (512 total, 128 output) - Best accuracy, much slower'),
        Choice(value=1.5, name='Minimum (192 total, 128 output) - Faster training')
    ]

    args.internal_neurons_ratio = inquirer.select(
        message="Model capacity (internal neurons):",
        choices=capacity_choices,
        default=2.0
    ).execute()

    total_neurons = int(128 * args.internal_neurons_ratio)
    print(f"   → Total neurons per layer: {total_neurons}, Output neurons: 128")

    # Sequence length presets (temporal context)
    print()
    SEQUENCE_LENGTH_PRESETS = {
        'low': {
            '5min': 75, '15min': 75, '30min': 75, '1h': 75, '2h': 75,
            '3h': 75, '4h': 75, 'daily': 75,
            'weekly': 20, 'monthly': 12, '3month': 8
        },
        'medium': {
            '5min': 200, '15min': 200, '30min': 200, '1h': 300, '2h': 300,
            '3h': 300, '4h': 300, 'daily': 600,
            'weekly': 20, 'monthly': 12, '3month': 8
        },
        'high': {
            '5min': 300, '15min': 300, '30min': 300, '1h': 500, '2h': 500,
            '3h': 500, '4h': 500, 'daily': 1200,
            'weekly': 20, 'monthly': 12, '3month': 8
        }
    }

    seq_preset = inquirer.select(
        message="Sequence length (temporal context):",
        choices=[
            Choice(value='low', name='Low (75 bars) - Fast training, ~3-4x faster ⚡'),
            Choice(value='medium', name='Medium (200-600 bars) - Balanced ⭐'),
            Choice(value='high', name='High (300-1200 bars) - Maximum context, slower'),
        ],
        default='medium'
    ).execute()

    # Override config with selected preset
    project_config.TIMEFRAME_SEQUENCE_LENGTHS = SEQUENCE_LENGTH_PRESETS[seq_preset]
    args.seq_preset = seq_preset  # Store for summary
    print(f"   → Sequence lengths: {seq_preset.upper()}")
    print(f"      5min: {SEQUENCE_LENGTH_PRESETS[seq_preset]['5min']}, "
          f"1h: {SEQUENCE_LENGTH_PRESETS[seq_preset]['1h']}, "
          f"daily: {SEQUENCE_LENGTH_PRESETS[seq_preset]['daily']}")

    args.epochs = int(inquirer.number(
        message="Number of epochs:",
        default=dflt('epochs', 100),
        min_allowed=1,
        max_allowed=1000
    ).execute())

    args.batch_size = int(inquirer.number(
        message=f"Batch size (recommended: {recommended_batch}):",
        default=dflt('batch_size', recommended_batch),
        min_allowed=1
    ).execute())

    args.lr = float(inquirer.number(
        message="Learning rate:",
        default=dflt('lr', 0.001),
        min_allowed=0.00001,
        max_allowed=0.01,
        float_allowed=True
    ).execute())

    # Data loading: mmap-only mode with OS page cache (no pre-merge)
    # Workers are safe now - mmap data is read-only, no COW explosion
    args.preload = False

    # Dataset split configuration
    print()
    args.use_test_set = inquirer.select(
        message="Dataset split configuration:",
        choices=[
            Choice(value=True, name='3-way split: 85% train, 10% validation, 5% test (recommended) ⭐'),
            Choice(value=False, name='2-way split: 90% train, 10% validation (classic)')
        ],
        default=dflt('use_test_set', True)
    ).execute()

    if args.use_test_set:
        args.train_split = 0.85
        args.val_split = 0.10
        args.test_split = 0.05
        print(f"   ✓ 3-way split: 85% train, 10% validation, 5% test")
        print(f"   ℹ️  Test set will ONLY be evaluated after training completes")

        # Split preview (estimated - actual dates shown during dataset creation)
        total_years = args.train_end_year - args.train_start_year
        train_years = total_years * 0.85
        val_years = total_years * 0.10
        test_years = total_years * 0.05

        est_train_end_year = args.train_start_year + train_years
        est_val_end_year = est_train_end_year + val_years

        print(f"\n   📅 Estimated Split Ranges:")
        print(f"      Train: {args.train_start_year} - ~{est_train_end_year:.1f} ({train_years:.1f} years)")
        print(f"      Val:   ~{est_train_end_year:.1f} - ~{est_val_end_year:.1f} ({val_years:.1f} years)")
        print(f"      Test:  ~{est_val_end_year:.1f} - {args.train_end_year} ({test_years:.1f} years)")
        print(f"      (Actual dates shown during dataset creation)")
    else:
        args.train_split = 0.90
        args.val_split = 0.10
        args.test_split = None
        print(f"   ✓ 2-way split: 90% train, 10% validation")

        # Split preview
        total_years = args.train_end_year - args.train_start_year
        train_years = total_years * 0.90
        val_years = total_years * 0.10
        est_val_start_year = args.train_start_year + train_years

        print(f"\n   📅 Estimated Split Ranges:")
        print(f"      Train: {args.train_start_year} - ~{est_val_start_year:.1f} ({train_years:.1f} years)")
        print(f"      Val:   ~{est_val_start_year:.1f} - {args.train_end_year} ({val_years:.1f} years)")
        print(f"      (Actual dates shown during dataset creation)")

    # Multi-task learning
    print()
    args.multi_task = inquirer.confirm(
        message="Enable multi-task learning (hit_band, hit_target, expected_return)?",
        default=dflt('multi_task', True)
    ).execute()

    # =========================================================================
    # v5.3: ARCHITECTURE LOCKED TO PRODUCTION
    # =========================================================================
    # Hardcoded to Geometric + Physics-Only (proven best)
    args.use_geometric_base = True
    args.use_fusion_head = False

    print()
    print("=" * 70)
    print("🏗️  ARCHITECTURE: v5.3 Production (Geometric + Physics-Only)")
    print("=" * 70)
    print()
    print("   ✅ Geometric Channel Projections")
    print("      - 21 windows per TF, selects best by quality")
    print("      - Pure geometry: slope × duration + bounds")
    print()
    print("   ✅ Physics-Only TF Selection")
    print("      - Selects best TF by validity (argmax)")
    print("      - Uses validity heads (forward-looking)")
    print()
    print("   ✅ v5.2-v5.3 Enhancements:")
    print("      - VIX CfC (90-day sequence)")
    print("      - Events (FOMC + earnings + macro)")
    print("      - Probabilistic duration (mean ± std)")
    print("      - Parent TF context (hierarchical learning)")
    print("      - Containment analysis")
    print("      - Confidence calibration")
    print()
    print("   Interpretability: 10/10")
    print("─" * 70)

    # v5.3.2: Information Flow Direction (added 'independent' mode)
    print()
    args.information_flow = inquirer.select(
        message="Information flow strategy:",
        choices=[
            Choice('independent', 'Independent - Each TF alone (no cross-TF hidden states) ⭐ Baseline'),
            Choice('bottom_up', 'Bottom-Up - Fast → Slow (details inform strategy)'),
            Choice('top_down', 'Top-Down - Slow → Fast (strategy guides details)'),
            Choice('bidirectional_bottom', 'Bidirectional (Bottom-First) - Micro foundation + macro overlay'),
            Choice('bidirectional_top', 'Bidirectional (Top-First) - Macro framework + micro refinement'),
        ],
        default='independent'
    ).execute()

    if args.information_flow == 'independent':
        print("   🔲 Independent: Each TF processes alone")
        print("      No cross-TF hidden state passing")
        print("      Good for: Baseline comparison, stability, debugging")
        print("      Simplest mode - each timeframe is self-contained")
    elif args.information_flow == 'bottom_up':
        print("   ✅ Bottom-Up: 5min → 3month (current v5.3)")
        print("      Each TF sees previous (faster) TF's understanding")
        print("      Good for: Detail aggregation, noise filtering")
    elif args.information_flow == 'top_down':
        print("   🔄 Top-Down: 3month → 5min (reversed)")
        print("      Each TF sees next (slower) TF's understanding")
        print("      Good for: Macro constraints, strategic context")
    elif args.information_flow == 'bidirectional_bottom':
        print("   ⇅ Bidirectional (Bottom-First): 5min→3month then 3month→5min")
        print("      Pass 1: Build micro understanding (bottom-up)")
        print("      Pass 2: Add macro overlay (top-down refinement)")
        print("      Good for: Micro-driven with macro validation")
        print("      +550K parameters (refinement networks)")
    else:  # bidirectional_top
        print("   ⇅ Bidirectional (Top-First): 3month→5min then 5min→3month")
        print("      Pass 1: Build macro framework (top-down)")
        print("      Pass 2: Add micro details (bottom-up refinement)")
        print("      Good for: Macro-driven with micro timing")
        print("      +550K parameters (refinement networks)")

    # v5.3: RSI cross-TF direction guidance
    print()
    args.rsi_direction_guidance = inquirer.select(
        message="3️⃣  RSI cross-TF direction validation:",
        choices=[
            Choice(value='soft_bias', name='Soft Bias - Model learns RSI patterns ⭐ Default'),
            Choice(value='validation', name='Validation Check - Verify with larger TF RSI, adjust confidence'),
            Choice(value='none', name='None - Disable explicit RSI guidance'),
        ],
        default='soft_bias'
    ).execute()

    if args.rsi_direction_guidance == 'soft_bias':
        print("   ✅ Soft Bias: RSI features in model input, learns importance")
    elif args.rsi_direction_guidance == 'validation':
        print("   🔍 Validation: Post-prediction check against parent TF RSI")
        print("      - Boosts confidence if RSI agrees")
        print("      - Reduces confidence if RSI conflicts")
    else:
        print("   ⚠️  None: RSI guidance disabled (for ablation testing)")

    # Debug logging
    print()
    args.debug = inquirer.confirm(
        message="Enable debug logging? (Shows collate workers, slow batches)",
        default=dflt('debug', False)
    ).execute()

    # Output path
    print()
    args.output = inquirer.text(
        message="Model output path:",
        default=dflt('output', 'models/hierarchical_lnn.pth')
    ).execute()

    # Summary
    # Compose human-readable extras
    parallel_workers = getattr(args, 'feature_workers', 'auto')
    parallel_str = "Yes" if getattr(args, 'use_parallel', False) else "No"
    if getattr(args, 'use_parallel', False):
        parallel_str += f" (workers={parallel_workers})"

    chunk_path = getattr(args, 'shard_path', None)
    chunk_str = "Yes" if getattr(args, 'use_chunking', False) else "No"
    if chunk_path:
        chunk_str += f" → {chunk_path}"

    # Format precision display
    # v5.7.2: Removed AMP option
    precision_mode = getattr(args, 'precision_mode', 'fp32')
    if precision_mode == 'fp64':
        precision_display = "FP64"
    elif precision_mode == 'fp32_tf32':
        precision_display = "FP32 (TF32) ⚡"
    else:
        precision_display = "FP32"

    print("\n" + "=" * 70)
    print("📋 TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Device: {args.device.upper()} (num_workers={args.num_workers})")
    print(f"  Precision: {precision_display}")
    print(f"  Training Period: {args.train_start_year}-{args.train_end_year}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  LR Scheduler: ReduceLROnPlateau (adaptive, v5.3.2)")
    # v5.9.3: Show actual data loading mode
    if getattr(args, 'preload_tf_to_ram', False):
        print(f"  Data Loading: Preload to RAM (3.2 GB TF sequences) ⚡")
    else:
        print(f"  Data Loading: mmap + OS page cache")
    # v5.3.2: Pre-stacking status
    if getattr(args, 'use_prestack', False):
        prestack_mode = getattr(args, 'prestack_mode', 'full_epoch')
        pinned_str = " + pinned" if getattr(args, 'use_pinned_prestack', False) else ""
        if prestack_mode == 'rolling':
            buffer_size = getattr(args, 'prestack_buffer_size', 100)
            print(f"  Batch Pre-Stack: Rolling buffer ({buffer_size} batches{pinned_str}, ~9% faster) ⭐")
        else:
            print(f"  Batch Pre-Stack: Full epoch (2 epochs{pinned_str}, ~40% faster) ⚠️ High RAM")
    else:
        print(f"  Batch Pre-Stack: Disabled (standard collate)")
    # v5.9.6: Sample selection display
    if getattr(args, 'use_boundary_sampling', False):
        mode = getattr(args, 'boundary_mode', 'breaks')
        threshold = getattr(args, 'boundary_threshold', 5)
        mode_desc = {'breaks': f'near breaks (≤{threshold} bars)',
                     'starts': f'fresh channels (≥{threshold} bars)',
                     'both': f'transitions (≤{threshold} or ≥{threshold*4} bars)'}
        print(f"  Sample Selection: Boundary sampling - {mode_desc.get(mode, mode)} ⚡")
    else:
        print(f"  Sample Selection: All samples (standard)")
    print(f"  Cache: {'Regenerate' if getattr(args, 'regenerate_cache', True) else 'Use existing'}")
    print(f"  Feature GPU: {'Yes' if getattr(args, 'use_gpu_features', False) else 'No'}")
    print(f"  Parallel CPU: {parallel_str}")
    print(f"  Chunking: {chunk_str}")
    if getattr(args, 'generate_native_tf', None):
        mode = 'streaming (~5-8GB)' if getattr(args, 'native_tf_streaming', True) else 'full load (~50GB)'
        print(f"  Native TF Gen: {mode}")
    print(f"  Continuation Mode: {project_config.CONTINUATION_MODE} "
          f"(horizon {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars)")
    print(f"  Model Capacity: internal_ratio={args.internal_neurons_ratio}, hidden_size={args.hidden_size}")
    seq_preset = getattr(args, 'seq_preset', 'high')  # Default to 'high' if not set
    print(f"  Sequence Lengths: {seq_preset.upper()} "
          f"(5min:{project_config.TIMEFRAME_SEQUENCE_LENGTHS['5min']}, "
          f"1h:{project_config.TIMEFRAME_SEQUENCE_LENGTHS['1h']}, "
          f"daily:{project_config.TIMEFRAME_SEQUENCE_LENGTHS['daily']})")
    print(f"  Multi-Task: {'Enabled' if args.multi_task else 'Disabled'}")

    # v5.3: Architecture (locked to production)
    print(f"  Architecture: Geometric + Physics-Only ⭐")
    print(f"    Base: Geometric (channel projections)")
    print(f"    Combine: Physics-based (select best TF)")

    # v5.3.1: Information flow
    info_flow = getattr(args, 'information_flow', 'bottom_up')
    flow_display = {
        'independent': 'Each TF alone (no cross-TF)',
        'bottom_up': '5min→3month (details first)',
        'top_down': '3month→5min (strategy first)',
        'bidirectional_bottom': '5min→3month→5min (micro+macro)',
        'bidirectional_top': '3month→5min→3month (macro+micro)',
    }
    print(f"  Information Flow: {flow_display.get(info_flow, info_flow)}")

    # v5.2/v5.3: Context systems
    print(f"  VIX CfC: 90-day sequence (regime awareness)")
    print(f"  Events: FOMC + Earnings + Deliveries + Macro")
    print(f"  Parent Context: Duration sees 2 larger TFs")

    # v5.3.2: Quality & break predictors (adaptive windows)
    print(f"  Channel Quality: Ping-pongs primary (v5.3.2)")
    print(f"  Break Predictors: ALL 11 TFs (adaptive windows) - duration_ratio + SPY-TSLA alignment")

    # v5.3: RSI validation
    rsi_mode = getattr(args, 'rsi_direction_guidance', 'soft_bias')
    rsi_display = {
        'soft_bias': 'Learned (model trains on RSI)',
        'validation': 'Post-check (adjusts confidence)',
        'none': 'Disabled'
    }
    print(f"  RSI Validation: {rsi_display.get(rsi_mode, rsi_mode)}")

    # v5.3: Additional features
    print(f"  Confidence Calibration: Enabled (accuracy-based)")
    print(f"  Hierarchical Containment: Analysis enabled")

    debug_mode = 'Enabled' if getattr(args, 'debug', False) else 'Disabled'
    print(f"  Debug Logging: {debug_mode}")
    print(f"  Output: {args.output}")
    if selected_cache_pair:
        print(f"  Cache reuse: {selected_cache_pair.get('cache_key')} from {args.shard_path}")
    print("=" * 70)

    # Confirmation
    print()
    proceed = inquirer.confirm(
        message="Start training with these settings?",
        default=True
    ).execute()

    if not proceed:
        print("❌ Training cancelled")
        sys.exit(0)

    return args


def run_training(rank: int, world_size: int, args_dict: dict):
    """
    Training worker function. Called either directly (single GPU) or via mp.spawn (multi-GPU).

    Args:
        rank: Process rank (0 for single GPU, 0 to world_size-1 for DDP)
        world_size: Total number of processes (1 for single GPU)
        args_dict: Arguments as dictionary (required for mp.spawn serialization)
    """
    # Reconstruct args namespace from dict
    args = argparse.Namespace(**args_dict)

    # v5.8: Apply sequence length preset in each spawned process
    # DDP spawns separate processes - each needs to override their own config
    if hasattr(args, 'seq_preset') and args.seq_preset:
        SEQUENCE_LENGTH_PRESETS = {
            'low': {
                '5min': 75, '15min': 75, '30min': 75, '1h': 75, '2h': 75,
                '3h': 75, '4h': 75, 'daily': 75,
                'weekly': 20, 'monthly': 12, '3month': 8
            },
            'medium': {
                '5min': 200, '15min': 200, '30min': 200, '1h': 300, '2h': 300,
                '3h': 300, '4h': 300, 'daily': 600,
                'weekly': 20, 'monthly': 12, '3month': 8
            },
            'high': {
                '5min': 300, '15min': 300, '30min': 300, '1h': 500, '2h': 500,
                '3h': 500, '4h': 500, 'daily': 1200,
                'weekly': 20, 'monthly': 12, '3month': 8
            }
        }
        project_config.TIMEFRAME_SEQUENCE_LENGTHS = SEQUENCE_LENGTH_PRESETS[args.seq_preset]

    # Set debug mode environment variable for child processes (collate workers, dataset)
    if getattr(args, 'debug', False):
        os.environ['TRAIN_DEBUG'] = '1'
    else:
        os.environ['TRAIN_DEBUG'] = '0'

    # DDP setup for multi-GPU
    is_distributed = world_size > 1
    local_rank = rank  # For mp.spawn, rank == local_rank (single node)

    if is_distributed:
        # Check if DDP is already initialized (torchrun case)
        if not dist.is_initialized():
            # Not initialized - we're being called from mp.spawn
            setup_distributed_spawn(rank, world_size)
            if is_main_process(rank):
                print(f"\n🚀 DDP Initialized via mp.spawn: {world_size} processes across {torch.cuda.device_count()} GPUs")
        else:
            # Already initialized - we're being called from torchrun
            if is_main_process(rank):
                print(f"\n🚀 DDP already initialized (torchrun): {world_size} processes")

        args.device = f'cuda:{rank}'
        args.use_ddp = True

        if is_main_process(rank):
            print(f"   Process rank {rank} using device {args.device}")

    # =========================================================================
    # TRAINING CODE STARTS HERE (moved from main())
    # =========================================================================

    # Setup memory profiler if enabled
    profiler = None
    if args.memory_profile:
        from src.ml.memory_profiler import MemoryProfiler
        profiler = MemoryProfiler(
            log_path="logs/memory_debug.log",
            device=args.device if args.device != 'auto' else 'unknown',
            log_every_n=10,
            spike_threshold_mb=500
        )
        profiler.log_info(f"CONTAINER_RAM_GB={os.environ.get('CONTAINER_RAM_GB', 'not_set')}")
        profiler.log_info(f"PREMERGE_LIMIT_GB={os.environ.get('PREMERGE_LIMIT_GB', 'not_set')}")
        profiler.log_info(f"PROFILER_RANK={rank}")

    # Auto-detect device if 'auto' (only for single GPU mode)
    if args.device == 'auto':
        args.device = get_best_device()
        if is_main_process(rank):
            print(f"🔍 Auto-detected device: {args.device}")

    # Validate device (skip for DDP since device is already set)
    if not is_distributed:
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU")
            args.device = 'cpu'
        elif args.device == 'mps' and not torch.backends.mps.is_available():
            print("⚠️ MPS not available, falling back to CPU")
            args.device = 'cpu'

    # MPS compatibility check: MPS doesn't support float64
    if args.device == 'mps' and project_config.TRAINING_PRECISION == 'float64':
        print("⚠️  MPS device doesn't support float64 precision")
        print("   Automatically using float32 instead")
        project_config.TRAINING_PRECISION = 'float32'
        project_config.NUMPY_DTYPE = np.float32
        project_config._TORCH_DTYPE = torch.float32
        print("   ✓ Switched to float32 for MPS compatibility")

    # v5.9.3: Re-apply TF32 setting in spawned processes (mp.spawn doesn't inherit PyTorch settings)
    if getattr(args, 'use_tf32', False):
        torch.set_float32_matmul_precision('medium')
        if is_main_process(rank):
            print(f"   ✓ TF32 Tensor Cores enabled (matmul precision: medium)")
    elif hasattr(args, 'precision_mode') and args.precision_mode == 'fp32':
        torch.set_float32_matmul_precision('highest')
    elif hasattr(args, 'precision_mode') and args.precision_mode == 'fp32_tf32':
        # Fallback if use_tf32 flag missing but precision_mode is set
        torch.set_float32_matmul_precision('medium')
        if is_main_process(rank):
            print(f"   ✓ TF32 Tensor Cores enabled (matmul precision: medium)")

    # Auto-set num_workers if not specified
    if args.num_workers is None:
        args.num_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}.get(args.device.split(':')[0], 2)

    # Override parallel worker count for feature extraction if specified
    if args.feature_workers is not None:
        project_config.MAX_PARALLEL_WORKERS = args.feature_workers
        if is_main_process(rank):
            print(f"✓ Feature extraction workers: {args.feature_workers} cores (via --feature_workers)")

    # Auto-detect chunking if not specified
    if args.use_chunking is None:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9
        args.use_chunking = (total_ram_gb < 64)
        if is_main_process(rank):
            print(f"✓ Auto-detected chunking: {'Enabled' if args.use_chunking else 'Disabled'} (RAM: {total_ram_gb:.1f}GB)")

    # Native timeframes + chunking: Allow chunking for extraction, native TF will be generated later
    # This enables the two-machine workflow:
    #   Machine A (low RAM): --use-chunking → creates chunks
    #   Machine B (high RAM): generate_native_tf_from_chunks() → then train with native TF
    if args.use_native_timeframes and args.use_chunking:
        if is_main_process(rank):
            print(f"   ℹ️  Chunked extraction mode: Native TF sequences will need to be generated after")
            print(f"       Run generate_native_tf_from_chunks() on high-RAM machine before training")
        # Don't disable chunking - let extraction proceed
        # Native TF mode will fail gracefully later if tf_meta doesn't exist

    # Set torch.compile verbose logging if requested (must be before model creation)
    if args.device.startswith('cuda') and getattr(args, 'use_compile', False) and getattr(args, 'compile_verbose', False):
        os.environ['TORCH_LOGS'] = 'dynamo,inductor,aot_autograd,output_code,graph_breaks,fusion'

        # Enable additional verbose output from dynamo and inductor
        try:
            torch._dynamo.config.verbose = True
            torch._dynamo.config.suppress_errors = False  # Show why graphs break
            torch._inductor.config.verbose = True
            torch._inductor.config.debug = True
        except (ImportError, AttributeError):
            pass  # Older PyTorch versions might not have these

        if is_main_process(rank):
            print(f"✓ torch.compile verbose logging enabled (comprehensive mode)")
            print(f"   TORCH_LOGS=dynamo,inductor,aot_autograd,output_code,graph_breaks,fusion")

    # Load configuration (currently unused - model uses hardcoded loss weights)
    # v5.3.2: Removed unused loss_weights variable (was loaded but never passed to model)
    config = None
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        if is_main_process(rank):
            print(f"✅ Loaded config from: {args.config} (currently unused)")
    else:
        if is_main_process(rank):
            print(f"⚠️ Config not found: {args.config}, using defaults")

    # Hardware info (rank 0 only prints)
    hw_info = get_hardware_info()

    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("🎯 HIERARCHICAL LNN TRAINING")
        print("=" * 70)
        print(f"📱 Device: {args.device.upper()}")
        if args.device.startswith('cuda'):
            print(f"   GPU: {hw_info.get('cuda_device', 'Unknown')}")
            print(f"   VRAM: {hw_info.get('cuda_memory_gb', 0):.1f} GB")
            if is_distributed:
                print(f"   DDP: {world_size} processes, effective batch = {args.batch_size * world_size}")
        elif args.device == 'mps':
            print(f"   Chip: {hw_info.get('mac_chip', 'Apple Silicon')}")
            print(f"   RAM: {hw_info.get('total_ram_gb', 0):.0f} GB")
        print(f"📅 Training: {args.train_start_year}-{args.train_end_year}")
        print(f"📊 Sequence: {args.sequence_length} bars")
        print(f"🎯 Horizon: Adaptive (base {args.prediction_horizon} bars)")
        print(f"🔢 Batch size: {args.batch_size}" + (f" x {world_size} GPUs = {args.batch_size * world_size}" if is_distributed else ""))
        print(f"🔄 Epochs: {args.epochs}")
        print(f"💾 Data mode: {'Preload to RAM ⚡' if getattr(args, 'preload_tf_to_ram', False) else 'mmap + OS page cache'}")
        print(f"🎭 Multi-task: {'Enabled' if args.multi_task else 'Disabled'}")
        print("=" * 70)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    if is_main_process(rank):
        print("\n1. Loading 1-min data...")
    if profiler:
        profiler.snapshot("pre_data_load", 0, force_log=True)

    data_feed = CSVDataFeed(timeframe=args.input_timeframe)

    historical_buffer_years = 2
    load_start_year = max(2010, args.train_start_year - historical_buffer_years)

    df = data_feed.load_aligned_data(
        start_date=f'{load_start_year}-01-01',
        end_date=f'{args.train_end_year}-12-31'
    )

    data_years = (df.index[-1] - df.index[0]).days / 365.25
    if is_main_process(rank):
        print(f"   Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    if profiler:
        profiler.log_info(f"DATA_LOADED | rows={len(df)} | cols={len(df.columns)}")
        profiler.snapshot("post_data_load", 0, force_log=True)

    if is_main_process(rank):
        print(f"   Data range: {data_years:.1f} years")
        print(f"   Historical buffer: {historical_buffer_years} years (for continuation analysis)")

    # Validate minimum data requirement
    min_required = project_config.MIN_DATA_YEARS if hasattr(project_config, 'MIN_DATA_YEARS') else 2.5
    if data_years < min_required:
        if is_main_process(rank):
            print(f"\n   ⚠️  WARNING: Insufficient data!")
            print(f"   You have: {data_years:.1f} years")
            print(f"   Recommended: {min_required}+ years")
    else:
        if is_main_process(rank):
            print(f"   ✓ Data requirement met ({data_years:.1f} years >= {min_required} required)")

    # Slice data to training range
    training_start = pd.to_datetime(f'{args.train_start_year}-01-01')
    training_end = pd.to_datetime(f'{args.train_end_year}-12-31')
    df_sliced = df[(df.index >= training_start) & (df.index <= training_end)].copy()

    if is_main_process(rank):
        print(f"   Training slice: {len(df_sliced)} bars ({df_sliced.index[0]} to {df_sliced.index[-1]})")

    # Smart lookback buffer system
    if project_config.SKIP_WARMUP_PERIOD:
        first_training_idx = df.index.get_loc(df_sliced.index[0])
        if first_training_idx < project_config.MIN_LOOKBACK_BARS:
            needed = project_config.MIN_LOOKBACK_BARS - first_training_idx
            old_start = df_sliced.index[0]
            df_sliced = df_sliced.iloc[needed:]
            if is_main_process(rank):
                print(f"   ⚠️  Skipped {needed} initial samples (warmup period)")
                print(f"   Adjusted start: {old_start} → {df_sliced.index[0]}")

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    if is_main_process(rank):
        print("\n2. Extracting features...")

        # v5.3.3: Deprecation warning for legacy mmap mode
        if not args.use_native_timeframes:
            print("\n" + "=" * 70)
            print("⚠️  WARNING: Legacy mmap mode is DEPRECATED as of v5.3.3")
            print("=" * 70)
            print("   Breakdown features will use OLD calculation (1-min resolution).")
            print("   This may cause train-test mismatch with live predictions.")
            print("")
            print("   RECOMMENDED: Use native timeframe mode instead:")
            print("   - Remove --no-native-timeframes flag")
            print("   - Or add --native-timeframes explicitly")
            print("")
            print("   Native TF mode benefits:")
            print("   - Faster training (optimized data loading)")
            print("   - Better live prediction consistency")
            print("   - Correct breakdown feature calculation")
            print("=" * 70)
            print("   Continuing with legacy mode in 5 seconds...")

            # Give user 5 seconds to cancel
            for i in range(5, 0, -1):
                print(f"   {i}...", end='\r', flush=True)
                time.sleep(1)
            print("   Proceeding with legacy mode.\n")

    if profiler:
        profiler.snapshot("pre_feature_extraction", 0, force_log=True)

    extractor = TradingFeatureExtractor()

    # Shard path for channel features (mmaps)
    shard_path = Path(args.shard_path) if args.shard_path else Path(project_config.SHARD_DIR)

    # Load VIX data for volatility regime features (v3.20)
    vix_data = None
    vix_csv_path = Path(project_config.DATA_DIR) / "VIX_History.csv"
    if vix_csv_path.exists():
        try:
            vix_data = load_vix_data(str(vix_csv_path))
            if is_main_process(rank):
                print(f"   ✓ VIX data loaded: {len(vix_data)} days ({vix_data.index[0].date()} to {vix_data.index[-1].date()})")
        except Exception as e:
            if is_main_process(rank):
                print(f"   ⚠️  VIX data load failed: {e}")
    else:
        if is_main_process(rank):
            print(f"   ⚠️  VIX data not found at {vix_csv_path}")

    # v4.0: Load events handler for earnings/FOMC features
    events_handler = None
    try:
        from src.ml.events import CombinedEventsHandler
        events_handler = CombinedEventsHandler(
            tsla_file=str(project_config.TSLA_EVENTS_FILE),
            macro_api_key=project_config.MACRO_EVENTS_API_KEY if hasattr(project_config, 'MACRO_EVENTS_API_KEY') else None
        )
        # Pre-load events to check coverage
        events_df = events_handler.load_events()
        if is_main_process(rank):
            print(f"   ✓ Events loaded: {len(events_df)} events from {events_df['date'].min().date()} to {events_df['date'].max().date()}")
    except Exception as e:
        if is_main_process(rank):
            print(f"   ⚠️  Events handler load failed: {e}")
            print(f"      Event features will be zeros. Check {project_config.TSLA_EVENTS_FILE}")

    # v6.0: Check if v6 cache is self-contained (skip feature extraction entirely)
    v6_self_contained = getattr(args, 'use_v6_cache', False) and getattr(args, 'v6_cache_dir', None)

    if v6_self_contained:
        # v6 cache has features embedded - skip all feature extraction
        if is_main_process(rank):
            print(f"   ✓ v6 cache is self-contained - skipping feature extraction")
            print(f"   ✓ Features, OHLC, and labels will load from v6 cache")
        features_df = None  # Dataset will load from v6 cache
        continuation_labels_dir = None  # v6 cache has labels
        mmap_meta_path = None
    elif getattr(args, 'preprocessed_cache_ready', False):
        # Features already extracted in main() - just load from cache
        if is_main_process(rank):
            print(f"   Loading features from cache (preprocessed before DDP)...")

        result = extractor.extract_features(
            df,
            use_cache=True,  # Always use cache - already extracted
            continuation=True,
            use_chunking=args.use_chunking,
            use_gpu=args.use_gpu_features,
            shard_storage_path=str(shard_path),
            vix_data=vix_data,
            events_handler=events_handler,
            skip_chunk_validation=getattr(args, 'skip_chunk_validation', False),  # v5.9.2
        )
        features_df = result[0]
        continuation_labels_dir = result[1]
        mmap_meta_path = result[2] if len(result) > 2 else None

        if is_main_process(rank):
            print(f"   ✓ Cache loaded instantly")
    else:
        # Single-process mode OR torchrun mode - extract here
        if is_main_process(rank):
            print(f"   Extracting features (use_chunking={args.use_chunking})...")

        result = extractor.extract_features(
            df,
            use_cache=True,
            continuation=True,
            use_chunking=args.use_chunking,
            use_gpu=args.use_gpu_features,
            shard_storage_path=str(shard_path),
            vix_data=vix_data,
            events_handler=events_handler,
            skip_chunk_validation=getattr(args, 'skip_chunk_validation', False),  # v5.9.2
        )
        features_df = result[0]
        continuation_labels_dir = result[1]
        mmap_meta_path = result[2] if len(result) > 2 else None

    # v5.3.2: Slice features_df to match df_sliced (apply warmup period)
    # Features were extracted on full df for rolling window context, but we only use post-warmup data
    if features_df is not None and len(df_sliced) != len(df):
        warmup_offset = df.index.get_loc(df_sliced.index[0])
        if is_main_process(rank):
            print(f"   Applying warmup offset: Slicing features from index {warmup_offset} ({df.index[warmup_offset].date()})")
        features_df = features_df.iloc[warmup_offset:]
        # Also slice raw OHLC to match
        df = df.iloc[warmup_offset:]
        if is_main_process(rank):
            print(f"   ✓ Features after warmup: {len(features_df):,} rows ({features_df.index[0].date()} to {features_df.index[-1].date()})")

    non_channel_cols = features_df.columns.tolist() if features_df is not None else None

    if profiler:
        nc_cols = len(non_channel_cols) if non_channel_cols is not None else 0
        cont_labels_info = str(continuation_labels_dir) if continuation_labels_dir else "None"
        profiler.log_info(f"FEATURES_EXTRACTED | non_channel_cols={nc_cols} | continuation_labels_dir={cont_labels_info}")
        profiler.snapshot("post_feature_extraction", 0, force_log=True)

    # v5.3.3: Validate that live data can support adaptive windows
    if is_main_process(rank) and args.use_native_timeframes:
        compatibility_warnings = project_config.validate_live_data_compatibility()
        if compatibility_warnings:
            print("\n⚠️  Live Prediction Data Warnings:")
            for warning in compatibility_warnings:
                print(f"   • {warning}")
            print("   (Training uses full historical data; live may have degraded features)\n")

    # Save cache manifest for future reuse (supports both mmap and pickle modes)
    if is_main_process(rank) and hasattr(extractor, '_cache_key'):
        cache_key = extractor._cache_key
        cache_dir = extractor._unified_cache_dir if hasattr(extractor, '_unified_cache_dir') else shard_path
        # v5.0 fix: Use continuation_labels_dir from return value (not non-existent _cont_cache_path)
        cont_path = str(continuation_labels_dir) if continuation_labels_dir else None
        nc_path = str(extractor._non_channel_cache_path) if hasattr(extractor, '_non_channel_cache_path') else None
        mmap_path = mmap_meta_path or (extractor._mmap_meta_path if hasattr(extractor, '_mmap_meta_path') else None)

        # For non-chunked mode, find the pickle cache path
        pickle_path = None
        if not mmap_path:
            potential_pickle = Path(cache_dir) / f"rolling_channels_{cache_key}.pkl"
            if potential_pickle.exists():
                pickle_path = str(potential_pickle)

        # Save manifest for either mode: mmap (chunked) or pickle (non-chunked)
        if cont_path and (mmap_path or pickle_path):
            save_cache_manifest(
                cache_dir=cache_dir,
                cache_key=cache_key,
                continuation_path=cont_path,
                args=args,
                df=df,
                mmap_meta_path=mmap_path,
                non_channel_path=nc_path,
                pickle_path=pickle_path,
            )
            cache_type = "mmap" if mmap_path else "pickle"
            print(f"   💾 Cache manifest saved ({cache_type}): cache_manifest_{cache_key}.json")

    # =========================================================================
    # GENERATE NATIVE TF SEQUENCES FROM CHUNKS (if requested via interactive menu)
    # =========================================================================
    if is_main_process(rank) and getattr(args, 'generate_native_tf', None) and args.generate_native_tf != 'skip':
        # Find the chunk metadata file
        chunk_meta_files = list(shard_path.glob('features_mmap_meta_*.json'))
        if chunk_meta_files:
            chunk_meta_path = chunk_meta_files[0]  # Use most recent
            print(f"\n   🔄 Generating native timeframe sequences from chunks...")
            print(f"   📄 Using: {chunk_meta_path.name}")

            # TradingFeatureExtractor is already imported at the top of this file
            gen_extractor = TradingFeatureExtractor()
            gen_extractor.generate_native_tf_from_chunks(
                chunks_meta_path=chunk_meta_path,
                output_cache_dir=shard_path,
                streaming=getattr(args, 'native_tf_streaming', True)
            )

            # Update tf_meta_path for native timeframe mode
            tf_meta_files = list(shard_path.glob('tf_meta_*.json'))
            if tf_meta_files:
                args.tf_meta_path = str(tf_meta_files[0])
                print(f"   ✓ Native TF sequences ready: {args.tf_meta_path}")
        else:
            print(f"   ⚠️  No chunk metadata found in {shard_path}")
            print(f"       Run with --use-chunking first to generate chunk shards")

    # =========================================================================
    # AUTO-DISCOVER TF_META PATH FOR NATIVE TIMEFRAME MODE
    # =========================================================================
    # v6.0: Skip if v6 cache is self-contained (doesn't need tf_meta)
    if v6_self_contained:
        if is_main_process(rank):
            print(f"   ✓ v6 cache provides tf_meta (self-contained)")
        args.tf_meta_path = None  # Not needed for v6
    elif args.use_native_timeframes and args.tf_meta_path is None:
        # Auto-discover tf_meta_*.json in cache directory
        cache_dir = extractor._unified_cache_dir if hasattr(extractor, '_unified_cache_dir') else shard_path
        cache_key = extractor._cache_key if hasattr(extractor, '_cache_key') else None

        if cache_key:
            potential_tf_meta = Path(cache_dir) / f"tf_meta_{cache_key}.json"
            if potential_tf_meta.exists():
                args.tf_meta_path = str(potential_tf_meta)
                if is_main_process(rank):
                    print(f"   🔄 Auto-discovered native timeframe meta: {potential_tf_meta.name}")
            else:
                # Try to find any tf_meta file in cache dir (v4.4: sort by mtime, pick newest)
                tf_meta_files = list(Path(cache_dir).glob("tf_meta_*.json"))
                if tf_meta_files:
                    # Sort by modification time (newest first) to pick most recent cache
                    tf_meta_files_sorted = sorted(tf_meta_files, key=lambda p: p.stat().st_mtime, reverse=True)
                    args.tf_meta_path = str(tf_meta_files_sorted[0])
                    if is_main_process(rank):
                        if len(tf_meta_files) > 1:
                            print(f"   🔄 Auto-discovered {len(tf_meta_files)} tf_meta files, using newest: {tf_meta_files_sorted[0].name}")
                        else:
                            print(f"   🔄 Auto-discovered native timeframe meta: {tf_meta_files_sorted[0].name}")
                else:
                    if is_main_process(rank):
                        print(f"   ⚠️  Native timeframes enabled but no tf_meta_*.json found in {cache_dir}")
                        print(f"       Option 1: Run with --no-chunking to generate native TF sequences")
                        print(f"       Option 2: Generate from existing chunks:")
                        print(f"                 from src.ml.features import TradingFeatureExtractor")
                        print(f"                 extractor = TradingFeatureExtractor()")
                        print(f"                 extractor.generate_native_tf_from_chunks(")
                        print(f"                     'data/feature_cache/features_mmap_meta_*.json',")
                        print(f"                     'data/feature_cache')")
                    # v5.2: Require native TF mode, no fallback to legacy
                    print(f"\n   ❌ ERROR: Native TF metadata required for v5.2")
                    print(f"   Please regenerate features with native TF generation enabled.")
                    import sys
                    sys.exit(1)

    # =========================================================================
    # DATASET CREATION
    # =========================================================================

    if is_main_process(rank):
        print("\n3. Creating dataset...")
    if profiler:
        profiler.snapshot("pre_dataset_create", 0, force_log=True)

    def check_dataloader_memory_safety(num_workers, container_ram_gb=0):
        """
        Check if DataLoader with num_workers will fit in available RAM.
        Uses ACTUAL current RAM (not estimates) since dataset is already created.

        Args:
            num_workers: Number of worker processes
            container_ram_gb: Container RAM limit (0 = use psutil)

        Returns:
            dict with safety info and recommendations
        """
        import psutil

        # Get ACTUAL current process RAM
        process = psutil.Process()
        current_ram_gb = process.memory_info().rss / (1024**3)

        # Get RAM limit
        if container_ram_gb > 0:
            total_ram_gb = container_ram_gb
        else:
            total_ram_gb = psutil.virtual_memory().total / (1024**3)

        # Calculate peak with workers (spawn gives each worker full dataset copy)
        if num_workers > 0:
            # Main process + (num_workers × dataset copy)
            estimated_peak_gb = current_ram_gb * (num_workers + 1)
        else:
            estimated_peak_gb = current_ram_gb

        # Safety check (80% threshold)
        is_safe = estimated_peak_gb < (total_ram_gb * 0.80)

        # Calculate max safe workers
        if current_ram_gb > 0:
            max_safe_workers = int((total_ram_gb * 0.80 / current_ram_gb) - 1)
            recommended_workers = max(0, min(max_safe_workers, 2))
        else:
            recommended_workers = 0

        return {
            'current_ram_gb': current_ram_gb,
            'estimated_peak_gb': estimated_peak_gb,
            'total_ram_gb': total_ram_gb,
            'is_safe': is_safe,
            'recommended_workers': recommended_workers,
            'num_workers': num_workers
        }

    # Create datasets
    # Note: raw_ohlc_df must match features_df length (both from full df)
    # validation_split handles train/val separation
    # Don't pass profiler to dataset if using workers (can't be pickled for spawn)
    dataset_profiler = profiler if args.num_workers == 0 else None

    # v6.0: Get v6 cache directory (from interactive menu or config)
    if not getattr(args, 'use_v6_cache', True):
        v6_cache_dir_path = None
        if is_main_process(rank):
            print(f"   ⚠️  v5 legacy mode: Not loading v6 cache")
    else:
        # Use path from interactive menu if available, otherwise fall back to config
        v6_cache_dir_path = getattr(args, 'v6_cache_dir', None)
        if v6_cache_dir_path is None:
            v6_cache_dir_path = str(getattr(project_config, 'V6_CACHE_DIR', 'data/feature_cache_v6'))
        if is_main_process(rank):
            print(f"   ✓ v6.0: Will load duration-primary labels from {v6_cache_dir_path}")

    train_dataset, val_dataset, test_dataset = create_hierarchical_dataset(
        features_df=features_df,
        raw_ohlc_df=df,  # Must match features_df length (both from full df)
        continuation_labels_df=None,  # v4.3: No legacy DataFrame, use per-TF labels
        continuation_labels_dir=str(continuation_labels_dir) if continuation_labels_dir else None,  # v4.3: Per-TF labels from shard_path
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        validation_split=args.val_split,
        test_split=args.test_split if hasattr(args, 'test_split') else None,  # v4.4: Optional test split
        include_continuation=True,
        mmap_meta_path=mmap_meta_path,
        profiler=dataset_profiler,
        preload_tf_to_ram=args.preload_tf_to_ram,  # v5.9.3: User-selectable
        use_native_timeframes=args.use_native_timeframes,
        tf_meta_path=args.tf_meta_path,
        use_boundary_sampling=getattr(args, 'use_boundary_sampling', False),  # v5.9.6
        boundary_threshold=getattr(args, 'boundary_threshold', 5),  # v5.9.6
        boundary_mode=getattr(args, 'boundary_mode', 'breaks'),  # v5.9.6
        v6_cache_dir=v6_cache_dir_path,  # v6.0: Duration-primary labels
    )

    # v5.8: Override sequence lengths if user selected a preset
    # Dataset reads from cached tf_meta.json, but user may have chosen different preset
    if hasattr(args, 'seq_preset') and args.seq_preset:
        from src.ml.features import HIERARCHICAL_TIMEFRAMES
        preset_lens = project_config.TIMEFRAME_SEQUENCE_LENGTHS

        # Override all three datasets
        for dataset in [train_dataset, val_dataset, test_dataset]:
            if dataset and hasattr(dataset, 'tf_sequence_lengths'):
                dataset.tf_sequence_lengths = preset_lens.copy()

        if is_main_process(rank):
            print(f"   ✓ Applied {args.seq_preset.upper()} sequence lengths: "
                  f"5min={preset_lens.get('5min', '?')}, 1h={preset_lens.get('1h', '?')}, "
                  f"daily={preset_lens.get('daily', '?')}")

    if profiler:
        profiler.log_info(f"DATASET_CREATED | train_samples={len(train_dataset)} | val_samples={len(val_dataset)}")
        profiler.snapshot("post_dataset_create", 0, force_log=True)

    if is_main_process(rank):
        print(f"   Training samples: {len(train_dataset):,}")
        print(f"   Validation samples: {len(val_dataset):,}")

    # =========================================================================
    # DATALOADER SETUP
    # =========================================================================

    if is_main_process(rank):
        print("\n4. Setting up data loaders...")

    # v5.2: Get feature dimensions from dataset (now 4-tuple)
    sample = train_dataset[0]
    if len(sample) == 4:
        sample_data, sample_target, sample_vix, sample_events = sample
    else:
        sample_data, sample_target = sample

    # Detect native timeframe mode (dict) vs legacy mode (tuple)
    if isinstance(sample_data, dict):
        # v4.1 Native timeframe mode - sample_data is Dict[str, np.ndarray]
        total_features = sample_data['5min'].shape[1]  # All TFs have same feature count
        if is_main_process(rank):
            print(f"   Native TF mode: {len(sample_data)} timeframes × {total_features} features each")
    else:
        # Legacy mode - sample_data is (ch_main, ch_monthly, nc) tuple
        ch_main, ch_monthly, nc = sample_data
        main_cols = ch_main.shape[1]
        monthly_cols = ch_monthly.shape[1] if ch_monthly is not None else 0
        nc_cols = nc.shape[1] if nc is not None else 0
        total_features = main_cols + monthly_cols + nc_cols
        if is_main_process(rank):
            print(f"   Features: {main_cols} main + {monthly_cols} monthly + {nc_cols} non-channel = {total_features} total")

    # Collate function
    torch_dtype = project_config.get_torch_dtype()
    collate_fn = functools.partial(
        hierarchical_collate,
        device=args.device,
        move_to_device=False,
        torch_dtype=torch_dtype
    )

    # DataLoader kwargs
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.device.startswith('cuda'),
        'collate_fn': collate_fn,
        'drop_last': True,
        'persistent_workers': args.num_workers > 0,
    }

    val_loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': args.device.startswith('cuda'),
        'collate_fn': collate_fn,
        'shuffle': False,
        'persistent_workers': args.num_workers > 0,
    }

    # Sampler setup
    # v5.9.4: User can choose between chunk-based (ShuffleBufferSampler) and random (DistributedSampler)
    # When preload_tf_to_ram=True, user gets a choice; when False (mmap), chunk-based is always used
    train_sampler = None
    use_chunk_sampler = getattr(args, 'use_chunk_sampler', True)  # Default to chunk-based

    if is_distributed:
        if use_chunk_sampler:
            # v5.9.4: Use DistributedShuffleBufferSampler - chunk-based with DDP support
            train_sampler = DistributedShuffleBufferSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                buffer_size=10000,  # 10K samples per chunk
                seed=42,
                drop_last=True
            )
            train_loader_kwargs['sampler'] = train_sampler
            train_loader_kwargs['shuffle'] = False
            if is_main_process(rank):
                print(f"   Using DistributedShuffleBufferSampler ({world_size} replicas, buffer=10000)")
        else:
            # Standard DistributedSampler - true global random shuffle
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True
            )
            train_loader_kwargs['sampler'] = train_sampler
            train_loader_kwargs['shuffle'] = False
            if is_main_process(rank):
                print(f"   Using DistributedSampler ({world_size} replicas, random shuffle)")
    else:
        if use_chunk_sampler:
            # v5.0: Use ShuffleBufferSampler for mmap data (8-10x faster than random shuffle)
            # Sequential chunks with local shuffling avoids random disk seeks
            train_sampler = ShuffleBufferSampler(
                train_dataset,
                buffer_size=10000,  # 10K samples per chunk (~50MB)
                seed=42  # Reproducible
            )
            train_loader_kwargs['sampler'] = train_sampler
            train_loader_kwargs['shuffle'] = False  # Sampler handles shuffling
            if is_main_process(rank):
                print(f"   Using ShuffleBufferSampler (buffer_size=10000)")
        else:
            # Standard DataLoader shuffle - true random
            train_loader_kwargs['shuffle'] = True
            if is_main_process(rank):
                print(f"   Using standard DataLoader shuffle (random)")

    # === MEMORY SAFETY CHECK: Warn before DataLoader creation ===
    # Check if num_workers with spawn will cause OOM
    if args.num_workers > 0 and is_main_process(rank):
        mem_check = check_dataloader_memory_safety(
            args.num_workers,
            container_ram_gb=getattr(args, 'container_ram_gb', 0)
        )

        if not mem_check['is_safe']:
            print(f"\n{'='*70}")
            print(f"⚠️  MEMORY WARNING: Potential OOM Risk")
            print(f"{'='*70}")
            print(f"   Current process RAM: {mem_check['current_ram_gb']:.1f} GB")
            print(f"   num_workers: {mem_check['num_workers']}")
            print(f"   Multiprocessing: spawn (CUDA-safe, but each worker gets dataset copy)")
            print(f"   ")
            print(f"   Expected peak RAM usage:")
            print(f"   • Main process: {mem_check['current_ram_gb']:.1f} GB (current usage)")
            print(f"   • {mem_check['num_workers']} workers: {mem_check['current_ram_gb'] * mem_check['num_workers']:.1f} GB ({mem_check['num_workers']} × {mem_check['current_ram_gb']:.1f} GB)")
            print(f"   • Model + training overhead: ~5 GB")
            print(f"   • TOTAL: ~{mem_check['estimated_peak_gb']:.1f} GB")
            print(f"   ")
            print(f"   Available RAM: {mem_check['total_ram_gb']:.1f} GB")
            print(f"   ")
            print(f"   ❌ This will likely cause OOM kill!")
            print(f"   ")
            print(f"   Recommended: num_workers={mem_check['recommended_workers']}")
            print(f"   • num_workers=0: {mem_check['current_ram_gb'] + 5:.1f} GB total (no worker copies)")
            print(f"   • num_workers=1: {mem_check['current_ram_gb'] * 2 + 5:.1f} GB total (1 worker copy)")
            if mem_check['recommended_workers'] >= 2:
                print(f"   • num_workers=2: {mem_check['current_ram_gb'] * 3 + 5:.1f} GB total (2 worker copies)")
            print(f"{'='*70}")

            # Pause and ask user
            response = input("\n   Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("\n   Aborting training. Please restart with lower num_workers.")
                print(f"   Suggested command: python train_hierarchical.py --num-workers {mem_check['recommended_workers']} [other args]")
                import sys
                sys.exit(1)
            else:
                print("\n   ⚠️  Continuing with high OOM risk...")
        else:
            # Safe - just log for confirmation
            print(f"\n   ✓ Memory check: {mem_check['estimated_peak_gb']:.1f}GB peak / {mem_check['total_ram_gb']:.1f}GB available (safe)")

    # Log memory before DataLoader creation (to catch OOM during worker spawn)
    if profiler:
        profiler.snapshot("pre_dataloader_creation", 0, force_log=True)
        profiler.log_info(f"CREATING_DATALOADERS | num_workers={args.num_workers}")

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # v4.4: Create test loader if test_dataset exists
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, **val_loader_kwargs)  # Use val kwargs (no shuffle, no drop_last)

    # Log memory after DataLoader creation (measure worker spawn impact)
    if profiler:
        profiler.snapshot("post_dataloader_creation", 0, force_log=True)
        test_info = f" | test_batches={len(test_loader)}" if test_loader else ""
        profiler.log_info(f"DATALOADERS_CREATED | train_batches={len(train_loader)} | val_batches={len(val_loader)}{test_info}")

    if is_main_process(rank):
        print(f"   Train batches: {len(train_loader):,}")
        print(f"   Val batches: {len(val_loader):,}")
        if test_loader:
            print(f"   Test batches: {len(test_loader):,} (held-out, evaluated after training)")

    # v5.3.2: Create PreStackedBatchLoader if enabled
    prestack_loader = None
    if getattr(args, 'use_prestack', False):
        prestack_mode = getattr(args, 'prestack_mode', 'full_epoch')

        if prestack_mode == 'rolling':
            # v5.3.2: Rolling buffer mode - RAM-efficient
            # v5.9.3: Now supports parallel sample fetching via num_workers
            if is_main_process(rank):
                print(f"\n   📦 Rolling buffer mode - creating RollingBufferBatchLoader...")
            prestack_loader = RollingBufferBatchLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                buffer_size=getattr(args, 'prestack_buffer_size', 100),
                collate_fn=hierarchical_collate,
                rank=rank if is_distributed else 0,
                world_size=world_size if is_distributed else 1,
                verbose=is_main_process(rank),
                drop_last=True,
                use_pinned=getattr(args, 'use_pinned_prestack', False),
                num_workers=args.num_workers,  # v5.9.3: Parallel fetching
            )
            if is_main_process(rank):
                buffer_size = getattr(args, 'prestack_buffer_size', 100)
                workers_msg = f", {args.num_workers} fetch workers" if args.num_workers > 0 else ""
                print(f"   ✓ RollingBufferBatchLoader ready ({buffer_size} batch buffer{workers_msg})")
        else:
            # Original full epoch mode
            if is_main_process(rank):
                print(f"\n   📦 Full epoch mode - creating PreStackedBatchLoader...")
            prestack_loader = PreStackedBatchLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                collate_fn=hierarchical_collate,
                rank=rank if is_distributed else 0,
                world_size=world_size if is_distributed else 1,
                verbose=is_main_process(rank),
                drop_last=True,
                use_pinned=getattr(args, 'use_pinned_prestack', False),
            )
            if is_main_process(rank):
                print(f"   ✓ PreStackedBatchLoader ready ({len(prestack_loader)} batches/epoch)")


    # =========================================================================
    # MODEL SETUP
    # =========================================================================

    if is_main_process(rank):
        print("\n5. Creating model...")
    if profiler:
        profiler.snapshot("pre_model_create", 0, force_log=True)

    # v4.0: HierarchicalLNN now has 11 CfC layers (one per timeframe)
    # v4.1: Native timeframe mode uses per-TF feature sizes
    if getattr(args, 'use_native_timeframes', False) and hasattr(train_dataset, 'timeframe_feature_counts'):
        # Native timeframe mode - each layer gets different feature count
        input_sizes = train_dataset.timeframe_feature_counts
        if is_main_process(rank):
            print(f"   Using native timeframe mode with per-layer input sizes:")
            for tf, count in input_sizes.items():
                seq_len = train_dataset.tf_sequence_lengths.get(tf, '?')
                print(f"      {tf}: {count} features × {seq_len} bars")

        model = HierarchicalLNN(
            input_sizes=input_sizes,  # v4.1: Dict[str, int] for native timeframe mode
            hidden_size=args.hidden_size,
            internal_neurons_ratio=args.internal_neurons_ratio,
            device=args.device,
            multi_task=args.multi_task,
            use_fusion_head=False,  # v5.3: Locked to Physics-Only
            use_geometric_base=True,  # v5.3: Locked to Geometric
            information_flow=getattr(args, 'information_flow', 'bottom_up'),  # v5.3.1
        )
    else:
        # Legacy mode - same size for all timeframes
        model = HierarchicalLNN(
            input_size=total_features,  # Backward compat: same size for all timeframes
            hidden_size=args.hidden_size,
            internal_neurons_ratio=args.internal_neurons_ratio,
            device=args.device,
            multi_task=args.multi_task,
            use_fusion_head=False,  # v5.3: Locked to Physics-Only
            use_geometric_base=True,  # v5.3: Locked to Geometric
            information_flow=getattr(args, 'information_flow', 'bottom_up'),  # v5.3.1
        )

    if profiler:
        total_params = sum(p.numel() for p in model.parameters())
        profiler.log_info(f"MODEL_CREATED | params={total_params:,} | device={args.device}")
        profiler.snapshot("post_model_create", 0, force_log=True)

    # v5.7: Set feature columns for geometric projection channel indexer
    if hasattr(train_dataset, 'tf_columns') and train_dataset.tf_columns:
        model._feature_columns = train_dataset.tf_columns
        if is_main_process(rank):
            print(f"   ✓ Channel indexer feature columns set ({len(train_dataset.tf_columns)} timeframes)")

    # Move model to device and wrap with DDP if distributed
    model = model.to(args.device)

    if is_distributed:
        # find_unused_parameters=True needed because:
        # 1. fusion_weights is only used for logging, not in loss
        # 2. CfC layers have sparse AutoNCP wiring - some neurons don't activate every forward pass
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process(rank):
            print(f"   ✓ Model wrapped with DDP (find_unused_parameters=True)")
    elif args.device.startswith('cuda') and getattr(args, 'use_compile', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            if is_main_process(rank):
                print(f"   ✓ Model compiled with torch.compile")
        except Exception as e:
            if is_main_process(rank):
                print(f"   ⚠️ torch.compile failed: {e}")

    if profiler:
        profiler.log_info(f"MODEL_READY | multi_gpu={is_distributed} | device={args.device}")
        profiler.snapshot("post_model_ready", 0, force_log=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(rank):
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    # =========================================================================
    # OPTIMIZER AND SCHEDULER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # v5.3.2: ReduceLROnPlateau for stable training (adapts to actual progress)
    # Cosine annealing was dropping LR too aggressively (to 0.000002) causing instability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Reduce when val_loss stops decreasing
        factor=0.5,           # Halve LR when stuck
        patience=5,           # Wait 5 epochs before reducing
        min_lr=1e-6           # Don't go below this
    )

    # v5.7.2: Removed AMP/GradScaler (caused NaN issues, use TF32 instead)

    # =========================================================================
    # RESUME FROM CHECKPOINT (if requested)
    # =========================================================================
    start_epoch = 0
    if hasattr(args, 'resume_checkpoint') and args.resume_checkpoint:
        if is_main_process(rank):
            print(f"\n📂 Loading checkpoint: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)

        # Load model weights
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Get starting epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))

        if is_main_process(rank):
            print(f"   ✓ Model weights loaded")
            print(f"   ✓ Optimizer state loaded")
            print(f"   ✓ Resuming from epoch {start_epoch + 1}")
            print(f"   ✓ Best val_loss so far: {best_val_loss:.4f}")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("STARTING TRAINING" + (f" (resuming from epoch {start_epoch + 1})" if start_epoch > 0 else ""))
        print("=" * 70)

    if profiler:
        profiler.log_info(f"TRAINING_START | device={args.device} | batch_size={args.batch_size} | num_workers={args.num_workers}")
        profiler.snapshot("pre_training_loop", 0, force_log=True)

    # Only reset best_val_loss if not resuming (it was set during checkpoint load)
    if start_epoch == 0:
        best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_errors = []

    # v5.3: Component-level tracking per epoch
    component_history = {
        'primary': [],
        'multi_task': [],
        'duration': [],
        'validity': [],
        'transition': [],
        'calibration': [],
        'containment': [],  # v5.9: channel containment loss (replaced geo_price)
        'hit_probability': [],  # v5.9: hit probability prediction loss
        'multi_tf': [],   # v5.7: multi-timeframe loss
        'entropy': [],    # v5.7: entropy regularization
    }

    # v5.3: Diagnostic tracking for transition loss issue
    transition_diagnostics = {
        'matches': 0,           # How many batches had matching selected_tf
        'total_batches': 0,     # Total batches
        'selected_tf_counts': {},  # Which TFs selected how often (hard argmax)
        # v5.8: Soft attention diagnostics (don't rely on selected_tf proxy alone)
        'tf_weights_sum': None,    # Accumulated tf_weights for mean calculation
        'tf_weights_count': 0,     # Number of samples accumulated
    }

    # v5.3.2: Enhanced diagnostics (what was missing from history!)
    learning_rates = []           # LR per epoch (scheduler changes it)
    epoch_times = []              # Minutes per epoch
    gradient_norms = []           # Avg gradient norm per epoch (before clipping)
    best_epoch = 0                # Which epoch had best val_loss
    early_stop_triggered = False  # Did early stopping fire?

    # v5.3.2: Duration/validity prediction statistics per epoch
    duration_stats = {
        'mean_predictions': [],   # Avg duration prediction per epoch
        'std_predictions': [],    # Std of duration predictions
    }
    validity_stats = {
        'mean_validity': [],      # Avg validity score per epoch
        'selected_tf_mode': [],   # Most common TF selected per epoch
    }

    # v5.3.2: Test set results (computed at end but wasn't saved!)
    test_results = None

    # =========================================================================
    # v5.7: LOSS WARMUP CONFIGURATION
    # =========================================================================
    # Secondary losses ramp up over warmup_epochs to prevent geo_price explosion
    WARMUP_EPOCHS = 5

    # =========================================================================
    # v6.0: DURATION-PRIMARY LOSS CONFIGURATION (DEFAULT)
    # =========================================================================
    USE_V5_LEGACY = getattr(args, 'use_v5_legacy', False)
    USE_V6_LOSS = not USE_V5_LEGACY  # v6 is default
    V6_WARMUP_EPOCHS = getattr(args, 'v6_warmup_epochs', None) or getattr(project_config, 'V6_WARMUP_EPOCHS', 10)

    if USE_V6_LOSS:
        # Create v6 loss config from project_config
        v6_weights = getattr(project_config, 'V6_LOSS_WEIGHTS', {})
        v6_loss_config = V6LossConfig(
            duration_weight=v6_weights.get('duration', 1.0),
            window_selection_weight=v6_weights.get('window_selection', 0.3),
            tf_selection_weight=v6_weights.get('tf_selection', 0.3),
            containment_weight_final=v6_weights.get('containment_final', 1.0),
            breakout_timing_weight=v6_weights.get('breakout_timing', 0.5),
            return_bonus_weight=v6_weights.get('return_bonus', 0.2),
            transition_weight_final=v6_weights.get('transition_final', 0.5),
            warmup_epochs=V6_WARMUP_EPOCHS,
        )
        if is_main_process(rank):
            print(f"\n   ✓ v6.0 Duration-Primary Architecture (DEFAULT)")
            print(f"     Warmup epochs: {V6_WARMUP_EPOCHS}")
            print(f"     Loss weights: duration={v6_loss_config.duration_weight}, "
                  f"window={v6_loss_config.window_selection_weight}, "
                  f"tf={v6_loss_config.tf_selection_weight}")
    else:
        v6_loss_config = None
        if is_main_process(rank):
            print(f"\n   ⚠️  Using v5.x Legacy Mode (--v5-legacy)")
            print(f"     This mode predicts high/low directly (old behavior)")

    def get_loss_warmup_weight(epoch: int, warmup_epochs: int, final_weight: float) -> float:
        """Quadratic warmup: 0 → final_weight over warmup_epochs."""
        if epoch >= warmup_epochs:
            return final_weight
        progress = (epoch + 1) / warmup_epochs  # +1 so epoch 0 has some weight
        return final_weight * (progress ** 2)

    # Progress bar (rank 0 only)
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", disable=not is_main_process(rank))

    for epoch in epoch_pbar:
        # v5.3.2: Track epoch timing
        epoch_start_time = time.perf_counter()

        # Set epoch for distributed sampler OR prestack loader
        if prestack_loader is not None:
            # v5.3.2: PreStackedBatchLoader handles its own shuffling
            prestack_loader.set_epoch(epoch)
        elif train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            # v5.9.6: Call set_epoch for any sampler (single or multi-GPU)
            train_sampler.set_epoch(epoch)

        # v5.3.2: DDP barrier - ensure all ranks finish pre-stacking before training
        # (Prevents rank 0 starting while rank 7 still pre-stacking)
        if is_distributed and prestack_loader is not None:
            dist.barrier()

        if profiler:
            profiler.log_phase(f"EPOCH_START | epoch={epoch + 1}")
            profiler.snapshot("pre_train_epoch", epoch + 1, force_log=True)

        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        # v5.8: Reset TF selector diagnostics for this epoch
        transition_diagnostics['selected_tf_counts'] = {}
        transition_diagnostics['total_batches'] = 0
        transition_diagnostics['matches'] = 0
        transition_diagnostics['tf_weights_sum'] = None
        transition_diagnostics['tf_weights_count'] = 0

        # v5.7: Calculate warmup weights for this epoch
        duration_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.3)
        geo_price_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.4)
        validity_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.2)
        transition_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.5)  # 0.3 + 0.2 for trans + direction

        # v5.7/v6.0: Anneal selection temperature
        # High temp = soft selection (gradients to all TFs), low temp = approaches hard selection
        model_core = model.module if hasattr(model, 'module') else model

        if USE_V6_LOSS:
            # v6.0: Use unified temperature annealing for both TF and window selection
            v6_temps = getattr(project_config, 'V6_TEMPERATURE', {})
            if hasattr(model_core, 'set_selection_temperatures'):
                model_core.set_selection_temperatures(
                    epoch=epoch,
                    warmup_epochs=V6_WARMUP_EPOCHS,
                    tf_start=v6_temps.get('tf_start', 2.0),
                    tf_end=v6_temps.get('tf_end', 0.5),
                    window_start=v6_temps.get('window_start', 2.0),
                    window_end=v6_temps.get('window_end', 0.5),
                )
                new_temp = model_core.selection_temperature
            else:
                new_temp = get_temperature(epoch, V6_WARMUP_EPOCHS)
        else:
            # v5.7: Legacy temperature annealing
            TEMP_ANNEAL_EPOCHS = 10
            if epoch < TEMP_ANNEAL_EPOCHS:
                new_temp = 2.0 - (epoch * 0.15)  # 2.0 -> 0.5
            else:
                new_temp = 0.5
            if hasattr(model_core, 'selection_temperature'):
                model_core.selection_temperature = new_temp

        if epoch < WARMUP_EPOCHS and is_main_process(rank):
            print(f"\n   Warmup epoch {epoch+1}/{WARMUP_EPOCHS}: duration={duration_weight:.3f}, geo={geo_price_weight:.3f}, validity={validity_weight:.3f}, transition={transition_weight:.3f}, temp={new_temp:.2f}")

        # v5.3: Accumulate components for this epoch
        epoch_components = {k: [] for k in component_history.keys()}

        # v5.3.2: Accumulate gradient norms and prediction stats per epoch
        epoch_grad_norms = []
        epoch_duration_preds = []
        epoch_validity_preds = []

        if profiler:
            profiler.snapshot("pre_dataloader_iter", epoch + 1, force_log=True)
            profiler.log_phase(f"dataloader_starting | epoch={epoch + 1} | batch=0 | num_workers={args.num_workers} | batch_size={args.batch_size}")

        # v5.3.2: Use prestack_loader if enabled, otherwise standard train_loader
        active_loader = prestack_loader if prestack_loader is not None else train_loader
        batch_pbar = tqdm(active_loader, desc=f"Epoch {epoch + 1}", leave=False, disable=not is_main_process(rank))

        # Track if this is the very first batch (for torch.compile feedback)
        _is_first_batch_ever = (epoch == 0)
        _first_forward_start = None

        # v5.9.5: Coarse timing to identify bottlenecks
        _iter_end_time = time.perf_counter()  # Track end of previous iteration
        _timing_samples = {'data': [], 'forward': [], 'loss': [], 'backward': [], 'optimizer': []}

        for batch_idx, batch_data in enumerate(batch_pbar):
            # Measure data loading time (from end of previous iteration)
            _data_time = time.perf_counter() - _iter_end_time

            if profiler and batch_idx == 0:
                profiler.log_info(f"FIRST_BATCH_COMPLETE | time_sec={0}")
                profiler.snapshot("first_batch_received", epoch + 1, force_log=True)

            # v5.2: Unpack batch (supports both 2-element and 4-element formats)
            if len(batch_data) == 4:
                features, targets, vix_batch, events_batch = batch_data
            else:
                features, targets = batch_data
                vix_batch = None
                events_batch = None

            # Move features to device - handle both native TF mode (dict) and legacy mode (tensor)
            if isinstance(features, dict):
                features = {tf: f.to(args.device, non_blocking=True) for tf, f in features.items()}
            else:
                features = features.to(args.device, non_blocking=True)

            # Move each tensor in targets dict to device
            # v5.9.2: Skip price_sequence (list of lists, not tensor)
            targets = {k: (v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

            # v5.2: Move VIX batch to device if present
            if vix_batch is not None:
                vix_batch = vix_batch.to(args.device, non_blocking=True)

            optimizer.zero_grad()

            # Track loss components (v5.7: init outside AMP/non-AMP paths)
            loss_components = {'primary': 0.0, 'multi_task': 0.0, 'duration': 0.0,
                              'validity': 0.0, 'transition': 0.0, 'calibration': 0.0,
                              'geo_price': 0.0, 'multi_tf': 0.0, 'entropy': 0.0}

            # Feedback for first batch (torch.compile takes time)
            # Use a background thread to print progress while compiling
            if _is_first_batch_ever and batch_idx == 0 and getattr(args, 'use_compile', False):
                print(f"   ⏳ First forward pass (torch.compile JIT compilation, may take 5-15 min)...", flush=True)
                _first_forward_start = time.perf_counter()

                # Start background thread to print progress dots with memory stats
                import threading
                _compile_done = threading.Event()
                def _print_compile_progress():
                    elapsed = 0
                    update_interval = 10  # More frequent updates (10s instead of 30s)

                    while not _compile_done.wait(timeout=update_interval):
                        elapsed += update_interval

                        # Show memory usage to indicate activity
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            cpu_mem_gb = process.memory_info().rss / (1024**3)

                            if args.device.startswith('cuda'):
                                gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3)
                                print(f"      ⏳ Compiling... {elapsed}s [CPU: {cpu_mem_gb:.1f}GB, GPU: {gpu_mem_gb:.1f}GB]", flush=True)
                            else:
                                print(f"      ⏳ Compiling... {elapsed}s [Memory: {cpu_mem_gb:.1f}GB]", flush=True)
                        except:
                            # Fallback if psutil not available
                            print(f"      ⏳ Compiling... {elapsed}s", flush=True)
                _progress_thread = threading.Thread(target=_print_compile_progress, daemon=True)
                _progress_thread.start()

            # Forward pass (v5.7.2: removed AMP, use TF32 instead)
            _forward_start = time.perf_counter()
            predictions, hidden_states = model(features, vix_sequence=vix_batch, events=events_batch)
            _forward_time = time.perf_counter() - _forward_start
            _loss_start = time.perf_counter()  # Start loss timing

            # 🛡️ NaN Check 1: Predictions (v6: check hidden states instead)
            if USE_V6_LOSS:
                # v6.0: Check duration outputs for NaN
                if 'duration' in hidden_states:
                    for tf, dur_data in hidden_states['duration'].items():
                        if not torch.isfinite(dur_data['mean']).all():
                            print(f"\n🚨 NaN/Inf in {tf} duration at batch {batch_idx}, epoch {epoch}!")
                            raise ValueError("Non-finite duration predictions - training aborted")
            else:
                # v5.x: Check predictions tensor
                if not torch.isfinite(predictions).all():
                    print(f"\n🚨 NaN/Inf in predictions at batch {batch_idx}, epoch {epoch}!")
                    print(f"   Min: {predictions.min().item()}, Max: {predictions.max().item()}")
                    raise ValueError("Non-finite predictions - training aborted")

            # =====================================================================
            # v6.0: DURATION-PRIMARY LOSS (replaces high/low MSE as primary)
            # =====================================================================
            if USE_V6_LOSS:
                # Extract v6 predictions from model output
                v6_predictions = model_core.get_v6_output_dict(hidden_states)

                # Prepare v6 targets from batch targets
                # v6 cache provides: {tf}_final_duration, {tf}_first_break_bar, {tf}_returned, etc.
                v6_targets = {}

                # v6.0: Define batch_size and device for fallback paths
                batch_size = next(iter(targets.values())).shape[0] if targets else 1
                device = args.device

                # Per-TF targets (from v6 cache labels)
                TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
                for tf in TIMEFRAMES:
                    # v6.0: Use actual v6 cache keys (not approximations)
                    dur_key = f'{tf}_final_duration'
                    break_key = f'{tf}_first_break_bar'
                    valid_key = f'{tf}_v6_valid'

                    if dur_key in targets:
                        # Duration and break bar (actual values from v6 cache)
                        v6_targets[f'{tf}_final_duration'] = targets[dur_key].unsqueeze(-1)
                        v6_targets[f'{tf}_first_break_bar'] = targets[break_key].unsqueeze(-1)
                        v6_targets[f'{tf}_valid_mask'] = targets.get(valid_key, torch.ones_like(targets[dur_key]))

                        # Break/return tracking (for return bonus)
                        if f'{tf}_returned' in targets:
                            v6_targets[f'{tf}_returned'] = targets[f'{tf}_returned']
                            v6_targets[f'{tf}_bars_outside'] = targets[f'{tf}_bars_outside']
                            v6_targets[f'{tf}_max_consecutive_outside'] = targets[f'{tf}_max_consecutive_outside']

                        # Window-level labels (for window selection loss)
                        if f'{tf}_window_r_squared' in targets:
                            # These are lists in targets, need to stack to tensors
                            r2_list = targets[f'{tf}_window_r_squared']
                            dur_list = targets[f'{tf}_window_durations']

                            # Stack lists to tensors [batch, 14]
                            if isinstance(r2_list, list) and len(r2_list) > 0:
                                if isinstance(r2_list[0], list):
                                    # List of lists -> stack
                                    v6_targets[f'{tf}_window_r_squared'] = torch.tensor(r2_list, device=device)
                                    v6_targets[f'{tf}_window_durations'] = torch.tensor(dur_list, device=device)
                                else:
                                    # Already a flat tensor
                                    v6_targets[f'{tf}_window_r_squared'] = torch.tensor(r2_list, device=device).unsqueeze(0)
                                    v6_targets[f'{tf}_window_durations'] = torch.tensor(dur_list, device=device).unsqueeze(0)

                        # Price sequences for containment loss
                        if f'{tf}_price_sequence' in targets:
                            v6_targets[f'{tf}_price_sequences'] = targets[f'{tf}_price_sequence']

                    else:
                        # Fallback to old format if v6 cache not loaded
                        old_dur_key = f'cont_{tf}_duration'
                        if old_dur_key in targets:
                            v6_targets[f'{tf}_final_duration'] = targets[old_dur_key].unsqueeze(-1)
                            v6_targets[f'{tf}_first_break_bar'] = targets[old_dur_key].unsqueeze(-1)  # Approximation
                            v6_targets[f'{tf}_valid_mask'] = targets.get(f'cont_{tf}_valid', torch.ones_like(targets[old_dur_key]))

                # Aggregated TF-level targets (for TF selection loss)
                tf_durations = []
                tf_broke_early = []
                for tf in TIMEFRAMES:
                    dur_key = f'{tf}_final_duration'
                    if dur_key in v6_targets:
                        tf_durations.append(v6_targets[dur_key])
                        # Broke early if duration < median (50 bars)
                        tf_broke_early.append((v6_targets[dur_key] < 50).float())
                    else:
                        tf_durations.append(torch.zeros(batch_size, 1, device=device))
                        tf_broke_early.append(torch.ones(batch_size, 1, device=device))

                if tf_durations:
                    v6_targets['tf_durations'] = torch.cat(tf_durations, dim=-1)  # [batch, 11]
                    v6_targets['tf_broke_early'] = torch.cat(tf_broke_early, dim=-1)  # [batch, 11]

                # Transition targets (global, from v6 cache)
                if 'v6_transition_type' in targets:
                    v6_targets['transition_type'] = targets['v6_transition_type'].long()
                    v6_targets['transition_direction'] = targets['v6_transition_direction'].long()
                    v6_targets['transition_next_tf'] = targets['v6_transition_next_tf'].long()
                elif 'transition_type' in targets:
                    # Fallback to old format
                    v6_targets['transition_type'] = targets['transition_type'].long()
                    v6_targets['transition_direction'] = targets.get('transition_direction', torch.ones_like(targets['transition_type'])).long()
                    v6_targets['transition_next_tf'] = targets.get('transition_next_tf', torch.zeros_like(targets['transition_type'])).long()

                # Compute v6 loss
                loss, loss_components = compute_v6_loss(
                    predictions=v6_predictions,
                    targets=v6_targets,
                    epoch=epoch,
                    config=v6_loss_config,
                    timeframes=TIMEFRAMES,
                )

                # Log v6 loss components on first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"[v6.0] {format_loss_log(loss_components, epoch)}", flush=True)

                    # v6.0: Debug v6 target stats to verify data pipeline is working
                    print(f"[v6.0] ===== V6 TARGET VERIFICATION =====", flush=True)
                    v6_debug_tfs = ['5min', '1h', 'daily']
                    for tf in v6_debug_tfs:
                        dur_key = f'{tf}_final_duration'
                        break_key = f'{tf}_first_break_bar'
                        valid_key = f'{tf}_valid_mask'
                        ret_key = f'{tf}_returned'
                        r2_key = f'{tf}_window_r_squared'

                        if dur_key in v6_targets:
                            dur_vals = v6_targets[dur_key]
                            break_vals = v6_targets.get(break_key, torch.zeros_like(dur_vals))
                            valid_vals = v6_targets.get(valid_key, torch.zeros(dur_vals.shape[0], device=dur_vals.device))

                            # Check for all zeros (bad sign)
                            is_all_zero = (dur_vals.abs().sum() == 0).item()
                            mean_dur = dur_vals.mean().item()
                            mean_break = break_vals.mean().item()
                            valid_pct = (valid_vals > 0).float().mean().item() * 100

                            status = "❌ ALL ZEROS" if is_all_zero else "✓"
                            print(f"  {tf}: {status} dur={mean_dur:.1f}, break={mean_break:.1f}, valid={valid_pct:.0f}%", flush=True)

                            # Check return/break tracking
                            if ret_key in v6_targets:
                                ret_vals = v6_targets[ret_key]
                                ret_pct = ret_vals.float().mean().item() * 100
                                print(f"       returned={ret_pct:.0f}%", flush=True)

                            # Check window-level data
                            if r2_key in v6_targets:
                                r2_tensor = v6_targets[r2_key]
                                mean_r2 = r2_tensor.mean().item()
                                print(f"       window R²={mean_r2:.3f}", flush=True)
                        else:
                            print(f"  {tf}: ❌ MISSING from v6_targets", flush=True)

                    # Check transition targets
                    if 'transition_type' in v6_targets:
                        trans_types = v6_targets['transition_type']
                        print(f"  transitions: types={trans_types[:5].tolist()}...", flush=True)
                    else:
                        print(f"  transitions: ❌ MISSING", flush=True)

                    print(f"[v6.0] =====================================", flush=True)

            else:
                # v5.x: Legacy loss computation
                # Primary loss: high/low predictions (raw % - data alignment fixed in dataset)
                target_tensor = torch.stack([targets['high'], targets['low']], dim=1)

                # 🛡️ NaN Check 2: Targets
                if not torch.isfinite(target_tensor).all():
                    print(f"\n🚨 NaN/Inf in targets at batch {batch_idx}!")
                    raise ValueError("Non-finite targets detected")

                # loss_components initialized before if/else block (v5.7)

                primary_loss = F.mse_loss(predictions[:, :2], target_tensor)
                loss = primary_loss
                loss_components['primary'] = primary_loss.item()

            # 🛡️ NaN Check 3: Loss
            if not torch.isfinite(loss):
                print(f"\n🚨 NaN/Inf loss at batch {batch_idx}!")
                print(f"   Predictions: mean={predictions[:,:2].mean().item()}, std={predictions[:,:2].std().item()}")
                if not USE_V6_LOSS:
                    print(f"   Targets: mean={target_tensor.mean().item()}, std={target_tensor.std().item()}")
                raise ValueError("Non-finite loss - check for exploding gradients or bad data")

            # Debug: Print target/prediction stats on first batch (v5.x legacy only)
            if batch_idx == 0 and epoch == 0 and not USE_V6_LOSS:
                print(f"[DEBUG] targets high: mean={target_tensor[:,0].mean():.2f}%, min={target_tensor[:,0].min():.2f}%, max={target_tensor[:,0].max():.2f}%", flush=True)
                print(f"[DEBUG] targets low: mean={target_tensor[:,1].mean():.2f}%, min={target_tensor[:,1].min():.2f}%, max={target_tensor[:,1].max():.2f}%", flush=True)
                print(f"[DEBUG] predictions: mean={predictions[:,:2].mean():.3f}, std={predictions[:,:2].std():.3f}", flush=True)
                print(f"[DEBUG] primary loss (high/low MSE): {loss.item():.4f}", flush=True)
                # v5.9.6: Debug duration values to verify fix
                duration_debug = []
                for tf in ['5min', '1h', 'daily']:
                    dur_key = f'cont_{tf}_duration'
                    valid_key = f'cont_{tf}_valid'
                    if dur_key in targets:
                        dur_vals = targets[dur_key]
                        valid_vals = targets.get(valid_key, torch.zeros_like(dur_vals))
                        valid_count = (valid_vals > 0).sum().item()
                        if valid_count > 0:
                            valid_durs = dur_vals[valid_vals > 0]
                            duration_debug.append(f"{tf}:{valid_durs.mean():.1f}±{valid_durs.std():.1f}({valid_count})")
                        else:
                            duration_debug.append(f"{tf}:no_valid")
                    else:
                        duration_debug.append(f"{tf}:missing")
                print(f"[DEBUG] durations (mean±std, count): {', '.join(duration_debug)}", flush=True)

            # =====================================================================
            # v5.7: MULTI-TF LOSS (all timeframes contribute, fixes mode collapse)
            # v6.0: Only needed for legacy v5 mode (uses target_tensor)
            # =====================================================================
            # Each TF's predictions are weighted by Gumbel-Softmax confidence
            if not USE_V6_LOSS and 'per_tf_predictions' in hidden_states and 'tf_weights' in hidden_states:
                per_tf_preds = hidden_states['per_tf_predictions']
                tf_weights = hidden_states['tf_weights']  # [batch, 11] from Gumbel-Softmax

                per_tf_highs = per_tf_preds['highs']  # [batch, 11]
                per_tf_lows = per_tf_preds['lows']    # [batch, 11]

                # v5.9.6: Vectorized multi-TF loss (replaces loop over 11 TFs)
                # Broadcast targets to [batch, 11]
                target_highs = target_tensor[:, 0:1].expand(-1, 11)  # [batch, 11]
                target_lows = target_tensor[:, 1:2].expand(-1, 11)   # [batch, 11]

                # Compute MSE for all TFs in single operation
                high_mse = (per_tf_highs - target_highs) ** 2  # [batch, 11]
                low_mse = (per_tf_lows - target_lows) ** 2     # [batch, 11]

                # Weighted average: weight by Gumbel-Softmax (detached), sum across TFs, mean across batch
                multi_tf_loss = ((high_mse + low_mse) / 2 * tf_weights.detach()).sum(dim=1).mean()

                # v5.7.2: Log weighted contribution so breakdown sums to total
                multi_tf_contribution = 0.1 * multi_tf_loss
                loss = loss + multi_tf_contribution
                loss_components['multi_tf'] = multi_tf_contribution.item()

                # =====================================================================
                # v5.7: ENTROPY REGULARIZATION (encourages diverse TF selection)
                # =====================================================================
                # Entropy = -sum(p * log(p)), higher = more uniform selection
                # We maximize entropy by subtracting from loss
                # v5.7.2: Log actual contribution (negative) + raw for monitoring collapse
                entropy_raw = -(tf_weights * torch.log(tf_weights + 1e-8)).sum(dim=-1).mean()
                entropy_weight = get_loss_warmup_weight(epoch, WARMUP_EPOCHS, 0.05)
                entropy_contribution = -entropy_weight * entropy_raw
                loss = loss + entropy_contribution
                loss_components['entropy'] = entropy_contribution.item()
                loss_components['entropy_raw'] = entropy_raw.item()

                # v5.7.2: Track TF weight distribution entropy (for diagnostics, not loss)
                mean_weights = tf_weights.detach().mean(dim=0)  # [11]
                loss_components['tf_weights_entropy'] = -(mean_weights * torch.log(mean_weights + 1e-8)).sum().item()

            # Multi-task losses (if enabled)
            # v6.0: Skip hit_band/hit_target/expected_return (removed heads)
            # v5.7.2: Accumulate ALL multi-task losses before logging (matches AMP path)
            if not USE_V6_LOSS and args.multi_task and 'multi_task' in hidden_states:
                mt = hidden_states['multi_task']
                # v6.0: hit_band, hit_target, expected_return removed
                # Only compute these losses in legacy mode
                if 'hit_band' in mt and 'hit_band' in targets:
                    mt_loss_total = (0.1 * F.binary_cross_entropy_with_logits(mt['hit_band'].squeeze(), targets['hit_band']) +
                              0.1 * F.binary_cross_entropy_with_logits(mt['hit_target'].squeeze(), targets['hit_target']) +
                              0.1 * F.mse_loss(mt['expected_return'].squeeze(), targets['expected_return']))
                else:
                    mt_loss_total = torch.tensor(0.0, device=args.device)

                # Adaptive projection losses (trains the adaptive_projection network)
                if 'price_change_pct' in mt and 'price_change_pct' in targets:
                    mt_loss_total = mt_loss_total + 0.4 * F.mse_loss(mt['price_change_pct'].squeeze(), targets['price_change_pct'])
                    mt_loss_total = mt_loss_total + 0.3 * F.mse_loss(mt['horizon_bars_log'].squeeze(), targets['horizon_bars_log'])
                    # adaptive_confidence uses BCE (sigmoid output vs 0/1 target)
                    mt_loss_total = mt_loss_total + 0.2 * F.binary_cross_entropy(
                        mt['adaptive_confidence'].float().squeeze(),
                        targets['adaptive_confidence'].float()
                    )

                loss = loss + mt_loss_total
                loss_components['multi_task'] = mt_loss_total.item()

            # v5.3: Confidence calibration (ALL samples in batch, every 10th batch)
            # v6.0: Only needed for legacy v5 mode (uses target_tensor)
            if not USE_V6_LOSS and 'layer_predictions' in hidden_states and batch_idx % 10 == 0:
                calib_loss_total = 0.0
                num_calibrated = 0

                for tf_name, preds in hidden_states['layer_predictions'].items():
                    # preds: [batch_size, 3]
                    for sample_idx in range(preds.shape[0]):
                        pred_high = preds[sample_idx, 0]
                        conf = preds[sample_idx, 2]
                        target_high = target_tensor[sample_idx, 0]

                        error = torch.abs(pred_high - target_high)
                        accuracy = (1.0 - error / (torch.abs(target_high) + 1e-6).clamp(0, 2)).clamp(0, 1)
                        calib_loss_total += (conf - accuracy.detach()) ** 2
                        num_calibrated += 1

                if num_calibrated > 0:
                    calib_component = 0.05 * (calib_loss_total / num_calibrated)
                    loss = loss + calib_component
                    loss_components['calibration'] = calib_component.item()

            # =====================================================================
            # v5.7.1: PER-SAMPLE VALIDITY MASKS AND LABELS (fixes batch-global bug)
            # v6.0: Only needed for legacy v5 mode
            # =====================================================================
            if not USE_V6_LOSS:
                TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
                # Get batch_size from targets instead of predictions (v6 doesn't use predictions tensor)
                batch_size = targets['high'].shape[0] if 'high' in targets else len(targets_list)

                # Transition labels as full tensors (not scalars from [0])
                transition_labels_tensors = {}
                for tf in TIMEFRAMES:
                    trans_type_key = f'trans_{tf}_type'
                    trans_valid_key = f'trans_{tf}_valid'
                    trans_dir_key = f'trans_{tf}_direction'

                    if trans_type_key in targets:
                        # Keep as [batch] tensors, not scalars!
                        trans_valid = targets.get(trans_valid_key, torch.zeros(batch_size, device=args.device))
                        trans_type = targets[trans_type_key]  # [batch]
                        trans_dir = targets.get(trans_dir_key, torch.ones(batch_size, device=args.device))  # [batch]

                    transition_labels_tensors[tf] = {
                        'valid_mask': trans_valid.float(),  # [batch] - 1.0 for valid, 0.0 for invalid
                        'transition_type': trans_type.long(),  # [batch]
                        'direction': trans_dir.long(),  # [batch]
                    }

            # Duration NLL loss - v5.7.1: per-sample validity
            # v6.0: Skip in v6 mode (duration loss handled by compute_v6_loss)
            duration_loss_total = 0.0
            if not USE_V6_LOSS and 'duration' in hidden_states:
                for tf, dur_data in hidden_states['duration'].items():
                    target_dur_key = f'cont_{tf}_duration'
                    cont_valid_key = f'cont_{tf}_valid'

                    if target_dur_key in targets:
                        # Per-sample validity mask
                        valid_mask = targets.get(cont_valid_key, torch.zeros(batch_size, device=args.device)).float()
                        num_valid = valid_mask.sum()

                        if num_valid > 0:
                            mean = dur_data['mean'].squeeze()
                            log_std = dur_data['log_std'].squeeze()
                            target_dur = targets[target_dur_key].squeeze()
                            variance = torch.exp(2 * log_std) + 1e-6

                            # Per-sample NLL, then apply mask
                            nll = 0.5 * ((target_dur - mean) ** 2 / variance + 2 * log_std)
                            masked_nll = (nll * valid_mask).sum() / (num_valid + 1e-8)
                            duration_loss_total += duration_weight * masked_nll

            if duration_loss_total > 0:
                loss = loss + duration_loss_total
                loss_components['duration'] = duration_loss_total.item()

            # =====================================================================
            # v5.9.6: VECTORIZED CONTAINMENT & HIT PROBABILITY LOSS
            # =====================================================================
            # Optimized from nested Python loops to batched tensor operations
            # Original: 128 samples × 11 TFs × 14 windows = ~20,000 loop iterations
            # Now: ~30 tensor operations per TF
            containment_loss_total = 0.0
            hit_probability_loss_total = 0.0

            # Cache window sizes for this batch
            _windows = project_config.CHANNEL_WINDOW_SIZES  # [100, 90, 80, ..., 10] (14 values)
            _num_windows = len(_windows)

            if 'geometric_predictions' in hidden_states:
                for tf, geo_data in hidden_states['geometric_predictions'].items():
                    if 'high' in geo_data and 'low' in geo_data and 'window_weights' in geo_data:
                        geo_high_pred = geo_data['high'].squeeze()  # [batch]
                        geo_low_pred = geo_data['low'].squeeze()    # [batch]
                        window_weights = geo_data['window_weights']  # [batch, 14]
                        batch_size = geo_high_pred.shape[0]

                        # =============================================================
                        # Phase 3: Pre-gather all targets into tensors (eliminates dict lookups in loops)
                        # =============================================================
                        # Build validity mask tensor [batch, num_windows]
                        validity_list = []
                        hit_upper_list = []
                        hit_midline_list = []
                        hit_lower_list = []

                        for win_idx, window in enumerate(_windows):
                            key_valid = f'cont_{tf}_w{window}_valid'
                            key_hit_upper = f'cont_{tf}_w{window}_hit_upper'
                            key_hit_midline = f'cont_{tf}_w{window}_hit_midline'
                            key_hit_lower = f'cont_{tf}_w{window}_hit_lower'

                            if key_valid in targets:
                                validity_list.append(targets[key_valid])  # [batch]
                            else:
                                validity_list.append(torch.zeros(batch_size, device=args.device))

                            if key_hit_upper in targets:
                                hit_upper_list.append(targets[key_hit_upper])
                                hit_midline_list.append(targets[key_hit_midline])
                                hit_lower_list.append(targets[key_hit_lower])
                            else:
                                hit_upper_list.append(torch.zeros(batch_size, device=args.device))
                                hit_midline_list.append(torch.zeros(batch_size, device=args.device))
                                hit_lower_list.append(torch.zeros(batch_size, device=args.device))

                        # Stack into [batch, num_windows] tensors
                        validity_mask = torch.stack(validity_list, dim=1) > 0  # [batch, 14] bool
                        hit_upper_targets = torch.stack(hit_upper_list, dim=1)  # [batch, 14]
                        hit_midline_targets = torch.stack(hit_midline_list, dim=1)  # [batch, 14]
                        hit_lower_targets = torch.stack(hit_lower_list, dim=1)  # [batch, 14]

                        # =============================================================
                        # Phase 2: Vectorized Containment Loss
                        # =============================================================
                        # For containment, we still need price sequences (variable length lists)
                        # But we can vectorize the inner containment check per sample
                        containment_scores = torch.full((batch_size,), 0.5, device=args.device)  # Default neutral

                        for sample_idx in range(batch_size):
                            # Get valid windows for this sample
                            sample_validity = validity_mask[sample_idx]  # [14]
                            if not sample_validity.any():
                                continue  # Keep default 0.5

                            # Find highest-weighted valid window
                            masked_weights = window_weights[sample_idx].clone()  # [14]
                            masked_weights[~sample_validity] = -float('inf')
                            best_win_idx = masked_weights.argmax().item()
                            best_window = _windows[best_win_idx]

                            # Get price sequence for best window
                            key = f'cont_{tf}_w{best_window}_price_sequence'
                            if key in targets and sample_idx < len(targets[key]):
                                price_seq = targets[key][sample_idx]
                                if price_seq is not None and len(price_seq) > 0:
                                    # Vectorized containment check
                                    price_tensor = torch.tensor(price_seq, device=args.device, dtype=geo_high_pred.dtype)
                                    low_bound = geo_low_pred[sample_idx]
                                    high_bound = geo_high_pred[sample_idx]
                                    contained = (price_tensor >= low_bound) & (price_tensor <= high_bound)
                                    containment_scores[sample_idx] = contained.float().mean()

                        # Loss = 1 - containment_rate (minimize when containment is high)
                        containment_loss = (1.0 - containment_scores).mean()
                        containment_loss_total += geo_price_weight * containment_loss

                        # =============================================================
                        # Phase 1: Vectorized Hit Probability Loss (biggest speedup)
                        # =============================================================
                        if 'hit_prob_upper' in geo_data:
                            hit_prob_upper_pred = geo_data['hit_prob_upper'].squeeze()  # [batch]
                            hit_prob_midline_pred = geo_data['hit_prob_midline'].squeeze()  # [batch]
                            hit_prob_lower_pred = geo_data['hit_prob_lower'].squeeze()  # [batch]

                            # Compute weighted targets using batched operations
                            # validity_mask: [batch, 14], window_weights: [batch, 14]
                            valid_weights = window_weights * validity_mask.float()  # [batch, 14]
                            total_weight = valid_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
                            normalized_weights = valid_weights / total_weight  # [batch, 14]

                            # Weighted average of targets: [batch]
                            blended_hit_upper = (normalized_weights * hit_upper_targets).sum(dim=1)
                            blended_hit_midline = (normalized_weights * hit_midline_targets).sum(dim=1)
                            blended_hit_lower = (normalized_weights * hit_lower_targets).sum(dim=1)

                            # Mask for samples with valid targets
                            has_valid = validity_mask.any(dim=1)  # [batch]
                            num_valid_samples = has_valid.sum()

                            if num_valid_samples > 0:
                                # Single batched BCE call (replaces 128 individual calls)
                                # Clamp targets to valid BCE range [0, 1]
                                blended_hit_upper = blended_hit_upper.clamp(0, 1)
                                blended_hit_midline = blended_hit_midline.clamp(0, 1)
                                blended_hit_lower = blended_hit_lower.clamp(0, 1)

                                # Compute BCE for all samples, then mask
                                bce_upper = F.binary_cross_entropy(hit_prob_upper_pred, blended_hit_upper, reduction='none')
                                bce_midline = F.binary_cross_entropy(hit_prob_midline_pred, blended_hit_midline, reduction='none')
                                bce_lower = F.binary_cross_entropy(hit_prob_lower_pred, blended_hit_lower, reduction='none')

                                # Average over 3 hit types, then apply validity mask
                                hit_loss_per_sample = (bce_upper + bce_midline + bce_lower) / 3.0  # [batch]
                                hit_probability_loss_total += (hit_loss_per_sample * has_valid.float()).sum()

            # Add containment loss (replaces geo_price)
            if containment_loss_total > 0:
                loss = loss + containment_loss_total
                loss_components['containment'] = containment_loss_total.item() if hasattr(containment_loss_total, 'item') else containment_loss_total

            # Add hit probability loss
            if hit_probability_loss_total > 0:
                loss = loss + hit_probability_loss_total * 0.1  # Lower weight than containment
                loss_components['hit_probability'] = (hit_probability_loss_total * 0.1).item() if hasattr(hit_probability_loss_total, 'item') else hit_probability_loss_total * 0.1

            # Validity loss - v5.7.1: per-sample
            # v6.0: Skip in v6 mode (validity loss handled by compute_v6_loss)
            validity_loss_total = 0.0
            if not USE_V6_LOSS and 'validity' in hidden_states:
                for tf, validity_pred in hidden_states['validity'].items():
                    if tf in transition_labels_tensors:
                        labels = transition_labels_tensors[tf]
                        valid_mask = labels['valid_mask']  # [batch]
                        num_valid = valid_mask.sum()

                        if num_valid > 0:
                            # Target: 1.0 if transition_type == 0 (CONTINUE), else 0.0
                            target_validity = (labels['transition_type'] == 0).float()  # [batch]

                            # Per-sample BCE, then apply mask
                            per_sample_loss = F.binary_cross_entropy(
                                validity_pred.squeeze(),
                                target_validity,
                                reduction='none'
                            )
                            masked_loss = (per_sample_loss * valid_mask).sum() / (num_valid + 1e-8)
                            validity_loss_total += validity_weight * masked_loss

            if validity_loss_total > 0:
                loss = loss + validity_loss_total
                loss_components['validity'] = validity_loss_total.item()

            # =====================================================================
            # v5.7.1: COMPOSITOR MULTI-TF TRAINING (per-sample labels)
            # v6.0: Skip in v6 mode (transition loss handled by compute_v6_loss)
            # =====================================================================
            transition_loss_total = 0.0

            # Track diagnostics (using selected TF for backwards compat stats) - always do this
            if 'selected_tf' in hidden_states:
                selected_tf = hidden_states['selected_tf']
                transition_diagnostics['total_batches'] += 1
                transition_diagnostics['selected_tf_counts'][selected_tf] = transition_diagnostics['selected_tf_counts'].get(selected_tf, 0) + 1

                # Check if any samples in batch have valid labels for selected TF
                # v6.0: transition_labels_tensors only exists in v5.x mode
                if not USE_V6_LOSS and selected_tf in transition_labels_tensors:
                    if transition_labels_tensors[selected_tf]['valid_mask'].sum() > 0:
                        transition_diagnostics['matches'] += 1

            # v5.8: Accumulate soft tf_weights for epoch-level mean (not just hard argmax)
            if 'tf_weights' in hidden_states:
                tf_weights = hidden_states['tf_weights']  # [batch, 11]
                batch_sum = tf_weights.sum(dim=0).detach().cpu()  # [11]
                if transition_diagnostics['tf_weights_sum'] is None:
                    transition_diagnostics['tf_weights_sum'] = batch_sum
                else:
                    transition_diagnostics['tf_weights_sum'] += batch_sum
                transition_diagnostics['tf_weights_count'] += tf_weights.size(0)

            # Train on ALL TFs with valid labels (v6.0: skip in v6 mode)
            if not USE_V6_LOSS:
              for tf, labels in transition_labels_tensors.items():
                compositor_key = f'compositor_{tf}'
                if compositor_key in hidden_states:
                    compositor = hidden_states[compositor_key]
                    valid_mask = labels['valid_mask']  # [batch]
                    num_valid = valid_mask.sum()

                    if num_valid > 0:
                        # Get TF weight from Gumbel-Softmax (detached)
                        if 'tf_weights' in hidden_states:
                            tf_idx = TIMEFRAMES.index(tf)
                            tf_weight = hidden_states['tf_weights'][:, tf_idx].mean().detach()
                        else:
                            tf_weight = 1.0 / len(transition_labels_tensors)

                        # Per-sample cross entropy with mask
                        trans_loss_per_sample = F.cross_entropy(
                            compositor['transition_logits'],
                            labels['transition_type'],  # [batch] - full tensor!
                            reduction='none'
                        )
                        dir_loss_per_sample = F.cross_entropy(
                            compositor['direction_logits'],
                            labels['direction'],  # [batch] - full tensor!
                            reduction='none'
                        )

                        # Apply validity mask
                        masked_trans = (trans_loss_per_sample * valid_mask).sum() / (num_valid + 1e-8)
                        masked_dir = (dir_loss_per_sample * valid_mask).sum() / (num_valid + 1e-8)

                        transition_loss_total += tf_weight * (masked_trans + masked_dir)

            # v5.7.2: Log weighted contribution so breakdown sums to total
            if transition_loss_total > 0:
                transition_contribution = transition_weight * transition_loss_total
                loss = loss + transition_contribution
                loss_components['transition'] = transition_contribution.item()
            else:
                loss_components['transition'] = 0.0

            _loss_time = time.perf_counter() - _loss_start  # End loss timing
            _backward_start = time.perf_counter()
            loss.backward()
            _backward_time = time.perf_counter() - _backward_start

            # v5.3.2: Track gradient norm BEFORE clipping (shows true gradient magnitude)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)

            # 🛡️ Gradient clipping (prevent explosion)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 🛡️ NaN Check 4: Gradients
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"\n🚨 NaN/Inf gradient in {name} at batch {batch_idx}!")
                    raise ValueError(f"Non-finite gradient in {name}")

            _optim_start = time.perf_counter()
            optimizer.step()
            _optim_time = time.perf_counter() - _optim_start

            # v5.9.5: Collect timing samples and print every 5 batches
            _timing_samples['data'].append(_data_time)
            _timing_samples['forward'].append(_forward_time)
            _timing_samples['loss'].append(_loss_time)
            _timing_samples['backward'].append(_backward_time)
            _timing_samples['optimizer'].append(_optim_time)

            if batch_idx < 10 or (batch_idx < 50 and batch_idx % 10 == 0):
                print(f"[TIMING] batch {batch_idx}: data={_data_time*1000:.0f}ms, fwd={_forward_time*1000:.0f}ms, "
                      f"loss={_loss_time*1000:.0f}ms, bwd={_backward_time*1000:.0f}ms, opt={_optim_time*1000:.0f}ms, "
                      f"total={(_data_time+_forward_time+_loss_time+_backward_time+_optim_time)*1000:.0f}ms", flush=True)

            _iter_end_time = time.perf_counter()  # Mark end of this iteration

            # Report first batch completion time and stop progress thread
            if _is_first_batch_ever and batch_idx == 0 and _first_forward_start is not None:
                _compile_done.set()  # Stop the progress thread
                _first_forward_elapsed = time.perf_counter() - _first_forward_start
                print(f"   ✓ First batch complete ({_first_forward_elapsed:.1f}s) - subsequent batches will be fast", flush=True)
                _is_first_batch_ever = False  # Don't print again

            train_loss += loss.item()
            num_batches += 1

            # v5.3: Accumulate component losses for epoch averaging
            # v5.7.2: Use setdefault to handle new keys (e.g. entropy_raw) without KeyError
            for component, value in loss_components.items():
                epoch_components.setdefault(component, []).append(value)

            # v5.3.2: Track duration/validity predictions for diagnostics
            if 'duration' in hidden_states and 'selected_tf' in hidden_states:
                sel_tf = hidden_states['selected_tf']
                if sel_tf in hidden_states['duration']:
                    dur_mean = hidden_states['duration'][sel_tf]['mean'].mean().item()
                    epoch_duration_preds.append(dur_mean)
            if 'validity' in hidden_states and 'selected_tf' in hidden_states:
                sel_tf = hidden_states['selected_tf']
                if sel_tf in hidden_states['validity']:
                    val_score = hidden_states['validity'][sel_tf].mean().item()
                    epoch_validity_preds.append(val_score)

            # Update loss display every batch (negligible overhead)
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Print component breakdown every 100 batches for diagnosis
            # v5.7.2: All values now reflect actual weighted contributions to loss
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"\n📊 [{batch_idx}] Loss components:")
                print(f"   Primary: {loss_components.get('primary', 0):.3f}")
                print(f"   Multi-task: {loss_components.get('multi_task', 0):.3f}")
                print(f"   Duration: {loss_components.get('duration', 0):.3f}")
                print(f"   Containment: {loss_components.get('containment', 0):.3f}")
                print(f"   Hit-Prob: {loss_components.get('hit_probability', 0):.3f}")
                print(f"   Validity: {loss_components.get('validity', 0):.3f}")
                print(f"   Transition: {loss_components.get('transition', 0):.3f}")
                print(f"   Calibration: {loss_components.get('calibration', 0):.3f}")
                print(f"   Entropy: {loss_components.get('entropy', 0):.3f}")
                print(f"   Multi-TF: {loss_components.get('multi_tf', 0):.3f}")
                print(f"   ─────────────────────")

                # v5.7.2: Verify tracked components sum to total (Fix 5)
                CONTRIBUTION_KEYS = ['primary', 'multi_task', 'duration', 'containment', 'hit_probability',
                                     'validity', 'transition', 'calibration', 'multi_tf', 'entropy']
                tracked_sum = sum(loss_components.get(k, 0) for k in CONTRIBUTION_KEYS)
                print(f"   Tracked: {tracked_sum:.3f}")
                print(f"   Total: {loss.item():.3f}")
                delta = loss.item() - tracked_sum
                if abs(delta) > 0.01:
                    print(f"   ⚠️  Untracked: {delta:.3f}")

                # 🔍 Multi-TF Training diagnostics (v5.7.2: replaces obsolete match rate)
                # Count TFs with at least one valid sample (not just keys existing)
                tfs_with_valid = sum(
                    (labels['valid_mask'].sum() > 0).item()
                    for labels in transition_labels_tensors.values()
                )
                print(f"\n🔍 Multi-TF Training:")
                print(f"   TFs with valid labels: {tfs_with_valid}/11")
                print(f"   TF weight entropy: {loss_components.get('tf_weights_entropy', 0):.3f}")
                print(f"   Entropy (raw): {loss_components.get('entropy_raw', 0):.3f}")

                # Most picked TF (attention mechanism preference)
                if transition_diagnostics['selected_tf_counts']:
                    mode_tf = max(transition_diagnostics['selected_tf_counts'],
                                  key=transition_diagnostics['selected_tf_counts'].get)
                    mode_count = transition_diagnostics['selected_tf_counts'][mode_tf]
                    total = transition_diagnostics['total_batches']
                    print(f"   Most picked TF: {mode_tf} ({mode_count}/{total} = {mode_count/total*100:.1f}%)")

            if profiler and batch_idx > 0 and batch_idx % profiler.log_every_n == 0:
                profiler.snapshot(f"batch_{batch_idx}", epoch + 1)

        avg_train_loss = train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # v5.8: Epoch-level TF selector diagnostics (don't rely on selected_tf proxy alone)
        if is_main_process(rank) and transition_diagnostics['total_batches'] > 0:
            total = transition_diagnostics['total_batches']
            print(f"\n   📊 TF Selector Diagnostics (epoch {epoch + 1}):")
            # (a) Argmax distribution - how often each TF wins
            print(f"      Argmax selection distribution:")
            for tf in sorted(transition_diagnostics['selected_tf_counts'].keys()):
                count = transition_diagnostics['selected_tf_counts'][tf]
                print(f"         {tf}: {count}/{total} = {count/total*100:.1f}%")
            # (b) Mean soft weights - average attention per TF
            if transition_diagnostics['tf_weights_sum'] is not None and transition_diagnostics['tf_weights_count'] > 0:
                mean_weights = transition_diagnostics['tf_weights_sum'] / transition_diagnostics['tf_weights_count']
                print(f"      Mean tf_weights (soft attention):")
                from src.ml.features import HIERARCHICAL_TIMEFRAMES
                for i, tf in enumerate(HIERARCHICAL_TIMEFRAMES):
                    if i < len(mean_weights):
                        print(f"         {tf}: {mean_weights[i].item():.3f}")

        # v5.3: Average component losses for this epoch
        for component, values in epoch_components.items():
            # v5.7.2: Use setdefault to handle new keys without KeyError
            if len(values) > 0:
                avg_component = sum(values) / len(values)
                component_history.setdefault(component, []).append(avg_component)
            else:
                component_history.setdefault(component, []).append(0.0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_error = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # v5.2: Unpack batch
                if len(batch_data) == 4:
                    features, targets, vix_batch_val, events_batch_val = batch_data
                else:
                    features, targets = batch_data
                    vix_batch_val = None
                    events_batch_val = None

                # Move features to device
                if isinstance(features, dict):
                    features = {tf: f.to(args.device, non_blocking=True) for tf, f in features.items()}
                else:
                    features = features.to(args.device, non_blocking=True)

                # Move targets to device
                # v5.9.2: Skip price_sequence (list of lists, not tensor)
                targets = {k: (v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

                # v5.2: Move VIX batch to device if present
                if vix_batch_val is not None:
                    vix_batch_val = vix_batch_val.to(args.device, non_blocking=True)

                # v5.7.2: Forward pass (removed AMP)
                predictions, hidden_states = model(features, vix_sequence=vix_batch_val, events=events_batch_val)

                if USE_V6_LOSS:
                    # v6.0: Use v6 loss for validation
                    v6_predictions = model_core.get_v6_output_dict(hidden_states)

                    # Prepare v6 targets from batch targets
                    v6_targets = {}
                    TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
                    for tf in TIMEFRAMES:
                        dur_key = f'cont_{tf}_duration'
                        if dur_key in targets:
                            v6_targets[f'{tf}_final_duration'] = targets[dur_key].unsqueeze(-1)
                            v6_targets[f'{tf}_first_break_bar'] = targets[dur_key].unsqueeze(-1)
                            v6_targets[f'{tf}_valid_mask'] = targets.get(f'cont_{tf}_valid', torch.ones_like(targets[dur_key]))

                    if 'transition_type' in targets:
                        v6_targets['transition_type'] = targets['transition_type']
                        v6_targets['transition_direction'] = targets.get('transition_direction', torch.ones_like(targets['transition_type']))
                        v6_targets['transition_next_tf'] = targets.get('transition_next_tf', torch.zeros_like(targets['transition_type']))
                        v6_targets['transition_valid_mask'] = targets.get('transition_valid', torch.ones_like(targets['transition_type']))

                    # Add window arrays if available
                    for tf in TIMEFRAMES:
                        r2_key = f'{tf}_window_r_squared'
                        dur_key = f'{tf}_window_durations'
                        if r2_key in targets:
                            v6_targets[r2_key] = targets[r2_key]
                        if dur_key in targets:
                            v6_targets[dur_key] = targets[dur_key]

                    loss, _ = compute_v6_loss(
                        predictions=v6_predictions,
                        targets=v6_targets,
                        epoch=epoch,
                        config=v6_loss_config,
                        timeframes=TIMEFRAMES,
                    )

                    # For error tracking, use duration prediction error
                    val_error_batch = 0.0
                    if 'duration' in hidden_states:
                        for tf in TIMEFRAMES:
                            dur_key = f'cont_{tf}_duration'
                            if tf in hidden_states['duration'] and dur_key in targets:
                                pred_dur = hidden_states['duration'][tf]['mean'].squeeze()
                                target_dur = targets[dur_key].squeeze()
                                val_error_batch += torch.abs(pred_dur - target_dur).mean().item()
                    val_error += val_error_batch / len(TIMEFRAMES)
                else:
                    # v5.x: Legacy validation
                    target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                    loss = F.mse_loss(predictions[:, :2], target_tensor)
                    val_error += torch.abs(predictions[:, :2] - target_tensor).mean().item()

                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)
        avg_val_error = val_error / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)
        val_errors.append(avg_val_error)

        # v5.3.2: ReduceLROnPlateau needs the metric to monitor
        scheduler.step(avg_val_loss)

        # v5.3.2: Track epoch-level diagnostics
        epoch_end_time = time.perf_counter()
        epoch_minutes = (epoch_end_time - epoch_start_time) / 60.0
        epoch_times.append(epoch_minutes)

        # Track LR (scheduler may have changed it)
        # v5.3.2: ReduceLROnPlateau doesn't have get_last_lr(), get from optimizer instead
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Track gradient norm stats
        if epoch_grad_norms:
            gradient_norms.append(sum(epoch_grad_norms) / len(epoch_grad_norms))
        else:
            gradient_norms.append(0.0)

        # Track duration/validity prediction stats
        if epoch_duration_preds:
            duration_stats['mean_predictions'].append(sum(epoch_duration_preds) / len(epoch_duration_preds))
            duration_stats['std_predictions'].append(
                (sum((x - duration_stats['mean_predictions'][-1])**2 for x in epoch_duration_preds) / len(epoch_duration_preds)) ** 0.5
            )
        else:
            duration_stats['mean_predictions'].append(0.0)
            duration_stats['std_predictions'].append(0.0)

        if epoch_validity_preds:
            validity_stats['mean_validity'].append(sum(epoch_validity_preds) / len(epoch_validity_preds))
        else:
            validity_stats['mean_validity'].append(0.0)

        # Track most common TF this epoch
        if transition_diagnostics['selected_tf_counts']:
            mode_tf = max(transition_diagnostics['selected_tf_counts'], key=transition_diagnostics['selected_tf_counts'].get)
            validity_stats['selected_tf_mode'].append(mode_tf)
        else:
            validity_stats['selected_tf_mode'].append('unknown')

        # Update progress bar
        epoch_pbar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{avg_val_loss:.4f}',
            'best': f'{best_val_loss:.4f}'
        })

        # Check for improvement (rank 0 saves model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1  # v5.3.2: Track which epoch was best
            patience_counter = 0

            if is_main_process(rank):
                # Save best model
                model_to_save = model.module if hasattr(model, 'module') else model
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'args': vars(args),
                    # CRITICAL: Save model architecture params for future compatibility
                    'input_sizes': model_to_save.input_sizes,  # v4.x: Dict[str, int] for native TF mode
                    'hidden_size': model_to_save.hidden_size,
                    'internal_neurons_ratio': model_to_save.internal_neurons_ratio,
                    # v5.1: Save architecture flags at top level for easier access
                    'use_fusion_head': getattr(args, 'use_fusion_head', False),
                    'use_geometric_base': getattr(args, 'use_geometric_base', True),
                    'multi_task': getattr(args, 'multi_task', True),
                    # v5.3.1: Save information flow (critical!)
                    'information_flow': getattr(args, 'information_flow', 'bottom_up'),
                    # v5.7: Dual prediction mode (direct + geometric)
                    'model_version': '5.7',
                    'has_geometric_projection': True,
                    'feature_columns': getattr(model_to_save, '_feature_columns', None),
                }, output_path)

                tqdm.write(f"   ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            early_stop_triggered = True  # v5.3.2: Track that early stopping fired
            if is_main_process(rank):
                tqdm.write(f"\n   Early stopping at epoch {epoch + 1} (patience={args.patience})")
            break

        if profiler:
            profiler.snapshot("post_epoch", epoch + 1, force_log=True)

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================

    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best val loss: {best_val_loss:.4f}")

        # v4.4: Held-out test set evaluation (if test_loader exists)
        if test_loader is not None:
            print("\n" + "=" * 70)
            print("🧪 HELD-OUT TEST SET EVALUATION")
            print("=" * 70)
            print("Evaluating on truly unseen data (not used during training)...")
            print()

            model.eval()
            test_loss = 0.0
            test_error = 0.0
            test_predictions_list = []
            test_targets_list = []
            num_test_batches = 0

            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc="Test batches", leave=False):
                    # v5.2: Unpack 4-tuple
                    if len(batch_data) == 4:
                        features, targets, vix_batch_test, events_batch_test = batch_data
                    else:
                        features, targets = batch_data
                        vix_batch_test = None
                        events_batch_test = None

                    # Move to device
                    if isinstance(features, dict):
                        features = {tf: f.to(args.device, non_blocking=True) for tf, f in features.items()}
                    else:
                        features = features.to(args.device, non_blocking=True)

                    # v5.9.2: Skip price_sequence (list of lists, not tensor)
                    targets = {k: (v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}

                    if vix_batch_test is not None:
                        vix_batch_test = vix_batch_test.to(args.device, non_blocking=True)

                    # Forward pass (v5.7.2: removed AMP)
                    predictions, hidden_states = model(features, vix_sequence=vix_batch_test, events=events_batch_test)

                    if USE_V6_LOSS:
                        # v6.0: Use v6 loss for test evaluation
                        v6_predictions = model_core.get_v6_output_dict(hidden_states)

                        # Prepare v6 targets
                        v6_targets = {}
                        TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
                        for tf in TIMEFRAMES:
                            dur_key = f'cont_{tf}_duration'
                            if dur_key in targets:
                                v6_targets[f'{tf}_final_duration'] = targets[dur_key].unsqueeze(-1)
                                v6_targets[f'{tf}_first_break_bar'] = targets[dur_key].unsqueeze(-1)
                                v6_targets[f'{tf}_valid_mask'] = targets.get(f'cont_{tf}_valid', torch.ones_like(targets[dur_key]))

                        if 'transition_type' in targets:
                            v6_targets['transition_type'] = targets['transition_type']
                            v6_targets['transition_direction'] = targets.get('transition_direction', torch.ones_like(targets['transition_type']))
                            v6_targets['transition_next_tf'] = targets.get('transition_next_tf', torch.zeros_like(targets['transition_type']))
                            v6_targets['transition_valid_mask'] = targets.get('transition_valid', torch.ones_like(targets['transition_type']))

                        # Add window arrays if available
                        for tf in TIMEFRAMES:
                            r2_key = f'{tf}_window_r_squared'
                            dur_key = f'{tf}_window_durations'
                            if r2_key in targets:
                                v6_targets[r2_key] = targets[r2_key]
                            if dur_key in targets:
                                v6_targets[dur_key] = targets[dur_key]

                        loss, _ = compute_v6_loss(
                            predictions=v6_predictions,
                            targets=v6_targets,
                            epoch=epoch,
                            config=v6_loss_config,
                            timeframes=TIMEFRAMES,
                        )

                        # For error tracking, use duration prediction error
                        test_error_batch = 0.0
                        if 'duration' in hidden_states:
                            for tf in TIMEFRAMES:
                                dur_key = f'cont_{tf}_duration'
                                if tf in hidden_states['duration'] and dur_key in targets:
                                    pred_dur = hidden_states['duration'][tf]['mean'].squeeze()
                                    target_dur = targets[dur_key].squeeze()
                                    test_error_batch += torch.abs(pred_dur - target_dur).mean().item()
                        test_error += test_error_batch / len(TIMEFRAMES)

                        # For detailed analysis, collect duration predictions
                        if 'duration' in hidden_states:
                            duration_preds = []
                            duration_targets = []
                            for tf in TIMEFRAMES:
                                dur_key = f'cont_{tf}_duration'
                                if tf in hidden_states['duration'] and dur_key in targets:
                                    duration_preds.append(hidden_states['duration'][tf]['mean'].cpu())
                                    duration_targets.append(targets[dur_key].cpu())
                            if duration_preds:
                                test_predictions_list.append(torch.stack(duration_preds, dim=-1))
                                test_targets_list.append(torch.stack(duration_targets, dim=-1))
                    else:
                        # v5.x: Legacy test
                        target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                        loss = F.mse_loss(predictions[:, :2], target_tensor)
                        test_error += torch.abs(predictions[:, :2] - target_tensor).mean().item()

                        # Collect for detailed analysis
                        test_predictions_list.append(predictions[:, :2].cpu())
                        test_targets_list.append(target_tensor.cpu())

                    test_loss += loss.item()
                    num_test_batches += 1

            avg_test_loss = test_loss / max(num_test_batches, 1)
            avg_test_error = test_error / max(num_test_batches, 1)

            # Combine all predictions for analysis
            if test_predictions_list:
                all_test_preds = torch.cat(test_predictions_list, dim=0).numpy()
                all_test_targets = torch.cat(test_targets_list, dim=0).numpy()

                print(f"\n📊 Test Set Results:")
                print(f"   Test Loss: {avg_test_loss:.4f}")
                print(f"   Test Error: {avg_test_error:.4f}")

                if USE_V6_LOSS:
                    # v6.0: Duration prediction metrics
                    print(f"\n   Duration Predictions (per TF):")
                    print(f"     MAE: {avg_test_error:.2f} bars")
                    if all_test_preds.shape[1] >= 11:
                        # Show per-TF metrics
                        TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
                        for i, tf in enumerate(TIMEFRAMES[:min(11, all_test_preds.shape[1])]):
                            mae = np.abs(all_test_preds[:, i] - all_test_targets[:, i]).mean()
                            print(f"       {tf}: {mae:.2f} bars MAE")
                else:
                    # v5.x: High/low prediction metrics
                    high_mae = np.abs(all_test_preds[:, 0] - all_test_targets[:, 0]).mean()
                    low_mae = np.abs(all_test_preds[:, 1] - all_test_targets[:, 1]).mean()
                    high_rmse = np.sqrt(np.mean((all_test_preds[:, 0] - all_test_targets[:, 0])**2))
                    low_rmse = np.sqrt(np.mean((all_test_preds[:, 1] - all_test_targets[:, 1])**2))

                    print(f"\n   High Predictions:")
                    print(f"     MAE:  {high_mae:.4f}%")
                    print(f"     RMSE: {high_rmse:.4f}%")
                    print(f"   Low Predictions:")
                    print(f"     MAE:  {low_mae:.4f}%")
                    print(f"     RMSE: {low_rmse:.4f}%")

                print(f"\n   Samples evaluated: {len(all_test_preds):,}")
                print(f"   ⚠️  This is your TRUE performance on unseen data")
                print("=" * 70)
            else:
                print(f"\n📊 Test Set Results:")
                print(f"   Test Loss: {avg_test_loss:.4f}")
                print(f"   No predictions collected")
                print("=" * 70)

            # v5.3.2: Save test results (previously computed but not saved!)
            test_results = {
                'test_loss': avg_test_loss,
                'test_mae': avg_test_error,
                'high_mae': float(high_mae),
                'high_rmse': float(high_rmse),
                'low_mae': float(low_mae),
                'low_rmse': float(low_rmse),
                'num_samples': len(all_test_preds)
            }

    # Print memory profile summary
    if profiler and is_main_process(rank):
        profiler.print_summary()
        profiler.close()

    # Save training history (rank 0 only)
    if is_main_process(rank):
        history_path = Path(args.output).parent / 'hierarchical_training_history.json'

        # Calculate transition match rate
        match_rate = transition_diagnostics['matches'] / max(transition_diagnostics['total_batches'], 1)

        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_errors': val_errors,
            'loss_components': component_history,  # v5.3: Per-component breakdown
            'transition_diagnostics': {  # v5.3: Why is transition loss so low?
                'match_rate': match_rate,
                'matches': transition_diagnostics['matches'],
                'total_batches': transition_diagnostics['total_batches'],
                'selected_tf_distribution': transition_diagnostics['selected_tf_counts'],
                # v5.8: Soft attention diagnostics (final epoch only)
                'mean_tf_weights': (
                    (transition_diagnostics['tf_weights_sum'] / transition_diagnostics['tf_weights_count']).tolist()
                    if transition_diagnostics['tf_weights_sum'] is not None and transition_diagnostics['tf_weights_count'] > 0
                    else None
                ),
            },
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'args': vars(args),

            # v5.3.2: Enhanced diagnostics (what was missing!)
            'learning_rates': learning_rates,           # LR per epoch (scheduler changes)
            'epoch_times_minutes': epoch_times,         # Minutes per epoch
            'gradient_norms': gradient_norms,           # Avg gradient norm per epoch (before clipping)
            'best_epoch': best_epoch,                   # Which epoch had best val_loss
            'early_stop_triggered': early_stop_triggered,  # Did early stopping fire?
            'duration_stats': duration_stats,           # Mean/std of duration predictions
            'validity_stats': validity_stats,           # Mean validity, most common TF
            'test_results': test_results,               # Test set evaluation (was computed but not saved!)

            # v5.3.3: Breakdown calculation and train-test consistency metadata
            'v533_metadata': {
                'breakdown_method': 'native_tf' if args.use_native_timeframes else 'legacy_1min',
                'train_test_consistent': args.use_native_timeframes,  # Native TF = consistent with live
                'feature_version': extractor._cache_key if hasattr(extractor, '_cache_key') else 'unknown',
                'cache_structure': 'native_tf_sequences' if args.use_native_timeframes else 'legacy_mmap',
                'adaptive_window_method': 'native_tf_bars' if args.use_native_timeframes else 'legacy_1min_bars',
                'yfinance_limits': {
                    '1min': project_config.YFINANCE_MAX_DAYS.get('1min') if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                    'intraday': project_config.YFINANCE_MAX_DAYS.get('intraday') if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                    '1h': project_config.YFINANCE_MAX_DAYS.get('1h') if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                    'daily': project_config.YFINANCE_MAX_DAYS.get('daily') if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                    'weekly_monthly': project_config.YFINANCE_MAX_DAYS.get('weekly_monthly') if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                } if hasattr(project_config, 'YFINANCE_MAX_DAYS') else None,
                'cross_tf_features': args.use_native_timeframes,  # Native TF includes cross-TF breakdown
                'duplicate_breakdown_bug_fixed': True,  # v5.3.3 fix: no more duplicates in non-chunked mode
            },
        }

        if profiler:
            history['memory_profile'] = profiler.get_summary()

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Training history saved to: {history_path}")

    # Clean up DDP resources
    cleanup_distributed()


def main():
    # Fix for Unix + torch + multiprocessing: use forkserver to avoid torch cleanup deadlock
    # spawn causes workers to hang on exit when torch is imported
    # forkserver is safe and faster (no torch init in workers with lazy loading)
    # NOTE: We use torch.multiprocessing (imported as mp at top) which supports mp.spawn()
    # Standard multiprocessing doesn't have spawn(), so we import it with different name
    import multiprocessing as std_mp
    # Set multiprocessing start method for CUDA compatibility
    # spawn = CUDA-safe (fresh processes), fork = fast but CUDA-unsafe
    try:
        if torch.cuda.is_available():
            # CUDA requires spawn to avoid fork() deadlocks with CUDA contexts
            std_mp.set_start_method('spawn', force=True)
            print(f"✓ Using spawn multiprocessing (CUDA-safe for DataLoader workers)")
        else:
            # CPU/MPS can use forkserver (faster startup than spawn)
            std_mp.set_start_method('forkserver', force=True)
            platform_note = "macOS" if sys.platform == "darwin" else "Linux"
            print(f"✓ Using forkserver multiprocessing (optimized for {platform_note})")
    except RuntimeError:
        # Already set (e.g., from previous run in interactive Python)
        pass

    parser = argparse.ArgumentParser(description='Train Hierarchical LNN')

    # Data parameters
    parser.add_argument('--input_timeframe', type=str, default='1min',
                        help='Input timeframe (always 1min for hierarchical)')
    parser.add_argument('--sequence_length', type=int, default=200,
                        help='Input sequence length (number of 1-min bars)')
    parser.add_argument('--prediction_horizon', type=int, default=24,
                        help='Base prediction horizon in bars (24 = 24 minutes) - model adapts dynamically')
    parser.add_argument('--train_start_year', type=int, default=2015,
                        help='Training data start year')
    parser.add_argument('--train_end_year', type=int, default=2022,
                        help='Training data end year')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for CfC layers (output neurons)')
    parser.add_argument('--internal_neurons_ratio', type=float, default=2.0,
                        help='Total neurons = hidden_size × ratio (default: 2.0 → 256 total)')
    parser.add_argument('--downsample_fast_to_medium', type=int, default=5,
                        help='Downsampling ratio fast→medium (1min→5min)')
    parser.add_argument('--downsample_medium_to_slow', type=int, default=12,
                        help='Downsampling ratio medium→slow (5min→1hour)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--sequence-preset', dest='sequence_preset', type=str,
                        choices=['low', 'medium', 'high'], default='medium',
                        help='Sequence length preset: low (75 bars, fast), medium (200-600 bars), high (300-1200 bars)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    # v5.2: Preload removed - native TF mode makes it obsolete
    parser.add_argument('--use-gpu-features', dest='use_gpu_features', action='store_true', default=False,
                        help='Use GPU acceleration for feature extraction (CUDA only)')

    # v4.1: Native timeframe mode (default=True for performance)
    parser.add_argument('--native-timeframes', dest='use_native_timeframes', action='store_true', default=True,
                        help='Use native timeframe mode: each CfC layer receives features at its native resolution (5min layer sees 5-min bars)')
    parser.add_argument('--no-native-timeframes', dest='use_native_timeframes', action='store_false',
                        help='Disable native timeframe mode (use legacy mmap mode - SLOW)')
    parser.add_argument('--tf-meta', dest='tf_meta_path', type=str, default=None,
                        help='Path to tf_meta_*.json file for native timeframe mode')
    parser.add_argument('--generate-native-tf', dest='generate_native_tf',
                        choices=['skip', 'streaming', 'full_load'], default=None,
                        help='Generate native TF from chunks: skip, streaming (~5-8GB RAM), or full_load (~50GB RAM)')
    parser.add_argument('--native-tf-streaming', dest='native_tf_streaming',
                        action='store_true', default=True,
                        help='Use streaming mode for native TF generation (default: True, ~5-8GB RAM)')
    parser.add_argument('--no-native-tf-streaming', dest='native_tf_streaming',
                        action='store_false',
                        help='Use full load mode for native TF generation (~50GB RAM, faster)')

    # v4.3: Hierarchical continuation labels
    parser.add_argument('--continuation-labels-dir', dest='continuation_labels_dir', type=str, default=None,
                        help='Directory with per-TF continuation label files (v4.3). If not provided, uses legacy single-TF labels.')

    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'mps', 'cpu'],
                        help='Device: auto, cuda, cuda:N (specific GPU), mps, cpu')
    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel for multi-GPU (auto-spawns processes)')
    parser.add_argument('--num-gpus', dest='num_ddp_gpus', type=int, default=None,
                        help='Number of GPUs to use for DDP (default: all available)')
    parser.add_argument('--gpu_mode', type=str, default='single',
                        choices=['single', 'multi_ddp'],
                        help='GPU mode: single (one GPU) or multi_ddp (DDP across all GPUs)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader num_workers (auto-set based on device if None)')
    parser.add_argument('--feature_workers', type=int, default=None,
                        help='Number of CPU cores for parallel feature extraction (default: config.MAX_PARALLEL_WORKERS, 0=all cores)')
    parser.add_argument('--use-chunking', dest='use_chunking', action='store_true', default=None,
                        help='Use chunked feature extraction to save memory (auto-detect if not specified)')
    parser.add_argument('--no-chunking', dest='use_chunking', action='store_false',
                        help='Disable chunked extraction (faster if you have enough RAM)')
    parser.add_argument('--shard-path', type=str, default=None,
                        help='Custom path for shard storage (default: data/feature_cache)')
    parser.add_argument('--output', type=str, default='models/hierarchical_lnn.pth',
                        help='Output model path')

    # Configuration
    parser.add_argument('--config', type=str, default='config/hierarchical_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--multi_task', action='store_true', default=True,
                        help='Enable multi-task learning (default: True)')
    parser.add_argument('--use-fusion-head', dest='use_fusion_head', action='store_true', default=True,
                        help='Use fusion head for final predictions (default: True)')
    parser.add_argument('--no-fusion-head', dest='use_fusion_head', action='store_false',
                        help='Disable fusion head, use physics-based aggregation instead')
    # v5.7.2: Removed --amp flag (AMP caused NaN issues, use TF32 instead)
    parser.add_argument('--use-compile', dest='use_compile', action='store_true', default=False,
                        help='Enable torch.compile JIT compilation (CUDA only, first batch slow but faster afterward)')
    parser.add_argument('--compile-verbose', dest='compile_verbose', action='store_true', default=False,
                        help='Enable verbose torch.compile output (sets TORCH_LOGS=dynamo,inductor for debugging graph breaks and kernel fusion)')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode with menus')

    # Cache verification
    parser.add_argument('--verify-cache', dest='verify_cache', action='store_true',
                        help='Verify cache integrity and exit (checks all cache files, VIX/Events staleness)')

    # Memory profiling
    parser.add_argument('--memory-profile', dest='memory_profile', action='store_true', default=False,
                        help='Enable memory profiling to logs/memory_debug.log')

    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable verbose debug logging (collate workers, slow batches, etc.)')

    # v6.0 Duration-Primary Architecture (DEFAULT)
    parser.add_argument('--v5-legacy', dest='use_v5_legacy', action='store_true', default=False,
                        help='Use legacy v5.x architecture (predicts high/low directly). Default is v6.0 duration-primary.')
    parser.add_argument('--v6-warmup-epochs', dest='v6_warmup_epochs', type=int, default=None,
                        help='Warmup epochs for v6 containment/transition losses (default: config.V6_WARMUP_EPOCHS)')

    args = parser.parse_args()

    # ==========================================================================
    # APPLY SEQUENCE LENGTH PRESET (from CLI or will be set by interactive)
    # ==========================================================================
    if not args.interactive and hasattr(args, 'sequence_preset'):
        # Apply preset when not in interactive mode (interactive mode will set this)
        SEQUENCE_LENGTH_PRESETS = {
            'low': {
                '5min': 75, '15min': 75, '30min': 75, '1h': 75, '2h': 75,
                '3h': 75, '4h': 75, 'daily': 75,
                'weekly': 20, 'monthly': 12, '3month': 8
            },
            'medium': {
                '5min': 200, '15min': 200, '30min': 200, '1h': 300, '2h': 300,
                '3h': 300, '4h': 300, 'daily': 600,
                'weekly': 20, 'monthly': 12, '3month': 8
            },
            'high': {
                '5min': 300, '15min': 300, '30min': 300, '1h': 500, '2h': 500,
                '3h': 500, '4h': 500, 'daily': 1200,
                'weekly': 20, 'monthly': 12, '3month': 8
            }
        }
        project_config.TIMEFRAME_SEQUENCE_LENGTHS = SEQUENCE_LENGTH_PRESETS[args.sequence_preset]
        args.seq_preset = args.sequence_preset  # Store for summary

    # ==========================================================================
    # CACHE VERIFICATION MODE (v4.4)
    # ==========================================================================
    if args.verify_cache:
        cache_dir = Path(args.shard_path if args.shard_path else 'data/feature_cache')

        print("\n" + "=" * 70)
        print("🔍 CACHE INTEGRITY CHECK")
        print("=" * 70)
        print(f"Cache directory: {cache_dir}")
        print()

        # Find all manifests
        manifest_files = sorted(cache_dir.glob("cache_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not manifest_files:
            print("❌ No cache manifests found")
            print(f"   Expected: cache_manifest_*.json in {cache_dir}")
            sys.exit(1)

        print(f"Found {len(manifest_files)} cache manifest(s):\n")

        for i, manifest_path in enumerate(manifest_files, 1):
            print(f"[{i}] {manifest_path.name}")
            validation = validate_cache_from_manifest(manifest_path, verbose=True)

            if validation["valid"]:
                manifest = validation["manifest"]
                print(f"   Cache type: {manifest.get('cache_type', 'unknown')}")
                print(f"   Date range: {manifest['date_range']['start']} to {manifest['date_range']['end']}")
                print(f"   Feature version: {manifest.get('feature_version', 'unknown')}")
                print(f"   Precision: {manifest.get('precision', 'unknown')}")

                # Show source file status
                source_files = manifest.get("source_files", {})
                if source_files.get("vix_mtime"):
                    from datetime import datetime
                    vix_date = datetime.fromtimestamp(source_files["vix_mtime"]).date()
                    print(f"   VIX data: {vix_date}")
                if source_files.get("events_count"):
                    events_date = datetime.fromtimestamp(source_files["events_mtime"]).date() if source_files.get("events_mtime") else "unknown"
                    print(f"   Events: {source_files['events_count']} events ({events_date})")

            print()

        print("=" * 70)
        print(f"✅ Cache verification complete")
        print("=" * 70)
        sys.exit(0)

    # ==========================================================================
    # DDP Initialization (must be VERY early, before any other setup)
    # ==========================================================================
    rank, world_size, local_rank, is_distributed = setup_distributed()

    if is_distributed:
        # Override args for DDP mode
        args.use_ddp = True
        args.gpu_mode = 'multi_ddp'
        args.device = f'cuda:{local_rank}'

        if is_main_process(rank):
            print(f"\n🚀 DDP Initialized: {world_size} processes across {torch.cuda.device_count()} GPUs")
            print(f"   This process: rank {rank}, local_rank {local_rank}, device {args.device}")
    else:
        # Not DDP - ensure defaults are set
        if not hasattr(args, 'use_ddp'):
            args.use_ddp = False
        if not hasattr(args, 'gpu_mode'):
            args.gpu_mode = 'single'

    # Setup memory profiler early if enabled (so we can log diagnostics throughout)
    profiler = None
    if args.memory_profile:
        from src.ml.memory_profiler import MemoryProfiler
        profiler = MemoryProfiler(
            log_path="logs/memory_debug.log",
            device=args.device if args.device != 'auto' else 'unknown',
            log_every_n=10,
            spike_threshold_mb=500
        )
        # Log environment info
        import os
        profiler.log_info(f"CONTAINER_RAM_GB={os.environ.get('CONTAINER_RAM_GB', 'not_set')}")

    # Interactive mode overrides command-line args
    # For DDP: only rank 0 runs interactive, then broadcasts args to other ranks
    if args.interactive:
        if is_distributed:
            if is_main_process(rank):
                # Only rank 0 runs interactive setup
                print(f"\n📋 Interactive setup (rank 0 of {world_size})")
                args = interactive_setup(args, profiler=profiler)

            # Broadcast args from rank 0 to all other ranks
            # Convert args namespace to dict for broadcasting
            args_dict = [vars(args) if rank == 0 else None]
            dist.broadcast_object_list(args_dict, src=0)

            if not is_main_process(rank):
                # Reconstruct args namespace on non-rank-0 processes
                args = argparse.Namespace(**args_dict[0])

            # Sync all ranks before continuing
            dist.barrier()

            if is_main_process(rank):
                print(f"\n✓ Configuration broadcast to all {world_size} processes")
        else:
            # Not torchrun - single process mode
            # Run interactive setup, then decide dispatch method
            args = interactive_setup(args, profiler=profiler)

    # ==========================================================================
    # v5.9: FEATURE EXTRACTION (Before DDP/Training)
    # ==========================================================================
    # Do feature extraction ONCE in main process before launching DDP
    # This prevents timeout issues and race conditions

    if not is_distributed:  # Only in mp.spawn mode (not torchrun - that handles differently)
        print("\n" + "=" * 70)
        print("🔧 PREPROCESSING (CPU - Before GPU Training)")
        print("=" * 70)

        # Load data
        print("\n1. Loading data...")
        data_feed = CSVDataFeed(timeframe=args.input_timeframe)
        historical_buffer_years = 2
        load_start_year = max(2010, args.train_start_year - historical_buffer_years)

        df = data_feed.load_aligned_data(
            start_date=f'{load_start_year}-01-01',
            end_date=f'{args.train_end_year}-12-31'
        )
        print(f"   ✓ Loaded {len(df):,} bars")

        # Load VIX and events
        print("\n2. Loading VIX and events...")
        vix_data = None
        vix_csv_path = Path(project_config.DATA_DIR) / "VIX_History.csv"
        if vix_csv_path.exists():
            vix_data = load_vix_data(str(vix_csv_path))
            print(f"   ✓ VIX: {len(vix_data)} days")

        events_handler = None
        try:
            from src.ml.events import CombinedEventsHandler
            events_handler = CombinedEventsHandler(
                tsla_file=str(project_config.TSLA_EVENTS_FILE),
                macro_api_key=project_config.MACRO_EVENTS_API_KEY if hasattr(project_config, 'MACRO_EVENTS_API_KEY') else None
            )
            events_df = events_handler.load_events()
            print(f"   ✓ Events: {len(events_df)} events")
        except Exception as e:
            print(f"   ⚠️  Events load failed: {e}")

        # Extract features (CPU work)
        # v6.0: Skip if v6 cache is self-contained
        if getattr(args, 'use_v6_cache', False) and getattr(args, 'v6_cache_dir', None):
            print("\n3. Loading from v6 cache...")
            print("   ✓ v6 cache is self-contained (features embedded)")
            print("   ✓ Skipping feature extraction")
            args.preprocessed_cache_ready = False  # Will load from v6 directly
        else:
            print("\n3. Extracting features (CPU, 6 workers)...")
            print("   This happens ONCE before GPUs start")

            extractor = TradingFeatureExtractor()
            shard_path = Path(args.shard_path) if args.shard_path else Path(project_config.SHARD_DIR)

            result = extractor.extract_features(
                df,
                use_cache=not getattr(args, 'regenerate_cache', True),
                continuation=True,
                use_chunking=args.use_chunking,
                use_gpu=args.use_gpu_features,
                shard_storage_path=str(shard_path),
                vix_data=vix_data,
                events_handler=events_handler,
                skip_chunk_validation=getattr(args, 'skip_chunk_validation', False),  # v5.9.2
            )

            print(f"   ✓ Features extracted and cached to {shard_path}")
            print(f"   ✓ GPU processes will load from cache (instant)")

            # Store cache info in args for DDP processes
            args.preprocessed_cache_ready = True
            args.cache_shard_path = str(shard_path)

    # ==========================================================================
    # DISPATCH TO run_training()
    # ==========================================================================

    if is_distributed:
        # Torchrun mode: DDP already initialized via setup_distributed()
        # run_training will detect this and skip its own DDP setup
        run_training(rank, world_size, vars(args))
    elif getattr(args, 'use_ddp', False):
        # User selected multi-GPU in interactive menu, but not launched with torchrun
        # Use mp.spawn to launch DDP processes automatically
        num_gpus = getattr(args, 'num_ddp_gpus', torch.cuda.device_count())
        print(f"\n🚀 Launching {num_gpus} DDP processes via mp.spawn...")
        print(f"   Each process will use one GPU (cuda:0 to cuda:{num_gpus-1})")
        print(f"   Features already cached - GPUs will load instantly")
        mp.spawn(run_training, nprocs=num_gpus, args=(num_gpus, vars(args)))
    else:
        # Single GPU/device mode - call directly
        run_training(0, 1, vars(args))



if __name__ == '__main__':
    main()
