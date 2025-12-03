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
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import sys
import os
import functools
from datetime import datetime
import json
import platform
import time
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


def hierarchical_collate(batch, device: str = None, move_to_device: bool = False, torch_dtype=None, _debug_counter=[0]):
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
    if _debug_counter[0] <= 3:
        print(f"[DEBUG] collate worker PID={os.getpid()}, batch_size={len(batch)}, call #{_debug_counter[0]}", file=sys.stderr, flush=True)

    if torch_dtype is None:
        torch_dtype = torch.float32

    # Separate data and targets
    data_list = [d for d, _ in batch]
    targets_list = [t for _, t in batch]

    # Detect format: dict (v4.1 native timeframe) or tuple (legacy)
    is_native_timeframe = isinstance(data_list[0], dict)

    if is_native_timeframe:
        # v4.1: Native timeframe mode - stack dicts into Dict[str, Tensor]
        batched_tf_data = {}
        for tf in data_list[0].keys():
            # Stack all samples for this timeframe
            tf_arrays = [d[tf] for d in data_list]
            stacked = np.stack(tf_arrays)  # [batch, seq_len, features]

            if not stacked.flags['C_CONTIGUOUS']:
                stacked = np.ascontiguousarray(stacked)

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
    converted_targets = []
    for tgt in targets_list:
        ct = {}
        for k, v in tgt.items():
            if isinstance(v, torch.Tensor):
                ct[k] = v.to(dtype=torch_dtype)
            else:
                ct[k] = torch.tensor(v, dtype=torch_dtype)
        converted_targets.append(ct)

    targets_batch = default_collate(converted_targets)

    if move_to_device and device is not None:
        if is_native_timeframe:
            for tf in x:
                x[tf] = x[tf].to(device, non_blocking=True)
        else:
            x = x.to(device, non_blocking=True)
        for k, v in targets_batch.items():
            targets_batch[k] = v.to(device, non_blocking=True)

    # Log slow batch assembly (diagnose lazy loading bottlenecks)
    _collate_elapsed = time.perf_counter() - _collate_start
    if _collate_elapsed > 1.0:  # Log if >1 second
        print(f"[SLOW_COLLATE] batch assembly took {_collate_elapsed:.1f}s for {len(batch)} samples ({_collate_elapsed/len(batch)*1000:.0f}ms/sample)", file=sys.stderr, flush=True)

    return x, targets_batch


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


def find_available_caches(cache_dir: Path):
    """Find available cache triplets (meta + continuation labels + non-channel) in a directory.

    Supports both:
    - Chunked/mmap mode: features_mmap_meta_*.json + shards
    - Non-chunked/pickle mode: rolling_channels_*.pkl (legacy)
    """
    cache_dir = Path(cache_dir)
    caches = []
    seen_keys = set()

    # 1. Find mmap caches (chunked mode)
    for meta_path in cache_dir.glob("features_mmap_meta_*.json"):
        cache_key = meta_path.name.replace("features_mmap_meta_", "").replace(".json", "")
        seen_keys.add(cache_key)

        mode_suffixes = ['adaptive', 'simple']
        cont_path = None
        for suffix in mode_suffixes:
            candidate = cache_dir / f"continuation_labels_{cache_key}_{suffix}.pkl"
            if candidate.exists():
                cont_path = candidate
                break

        # Check for non-channel features cache
        non_channel_path = cache_dir / f"non_channel_features_{cache_key}.pkl"
        has_non_channel = non_channel_path.exists()

        caches.append({
            "cache_key": cache_key,
            "cache_type": "mmap",
            "meta_path": str(meta_path),
            "cont_path": str(cont_path) if cont_path else None,
            "non_channel_path": str(non_channel_path) if has_non_channel else None,
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

        mode_suffixes = ['adaptive', 'simple']
        cont_path = None
        for suffix in mode_suffixes:
            candidate = cache_dir / f"continuation_labels_{cache_key}_{suffix}.pkl"
            if candidate.exists():
                cont_path = candidate
                break

        caches.append({
            "cache_key": cache_key,
            "cache_type": "pickle",
            "meta_path": None,  # No mmap meta for pickle mode
            "pickle_path": str(pickle_path),
            "cont_path": str(cont_path) if cont_path else None,
            "non_channel_path": None,  # Pickle mode includes all features
            "complete": cont_path is not None  # Pickle has all features, just need continuation
        })

    return caches


def pick_cache_pair(caches):
    """Prompt user to select a cache triplet to reuse."""
    if not caches:
        return None
    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        return None

    # Auto-select when only one cache is available
    if len(caches) == 1:
        cache = caches[0]
        status = "COMPLETE" if cache.get("complete") else "partial"
        print(f"\n✅ Auto-selected cache: {cache['cache_key']} ({status})")
        print(f"   → Will skip feature extraction and use cached data\n")
        return cache

    choices = []
    for c in caches:
        # Build status string based on what's cached
        if c.get("complete"):
            status = "COMPLETE - skip extraction entirely"
        elif c.get("cont_path") and c.get("non_channel_path"):
            status = "COMPLETE - skip extraction entirely"
        elif c.get("cont_path"):
            status = "partial: channels + labels (will recompute non-channel ~10-30s)"
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
            "dtype": str(project_config.NUMPY_DTYPE),
            "paths": {
                "mmap_meta": mmap_meta_path,
                "continuation_labels": continuation_path,
                "non_channel_features": non_channel_path,
                "pickle_channels": pickle_path,  # For non-chunked mode
            },
            "cache_dir": str(cache_dir),
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
                # v4.0: downsample params removed (11-layer architecture uses native TF data)
                "num_workers": args.num_workers,
                "preload": getattr(args, "preload", False),
                "output": getattr(args, "output", None),
                "model_version": "4.0",  # v4.0: 11-layer architecture
            }
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"   🗂️  Saved cache manifest: {manifest_path.name}")
    except Exception as e:
        print(f"   ⚠️  Could not save cache manifest ({type(e).__name__}): {e}")


def train_epoch(
    model: HierarchicalLNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    loss_weights: Dict = None,
    scaler=None,
    use_multi_gpu: bool = False,
    profiler=None
) -> float:
    """
    Train for one epoch with multi-task loss.

    Args:
        model: HierarchicalLNN model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (MSE)
        device: 'cuda' or 'cpu'
        epoch: Current epoch number
        loss_weights: Dict with task weights (from config)
        scaler: Optional GradScaler for AMP (CUDA only)
        profiler: Optional MemoryProfiler for debugging

    Returns:
        avg_loss: Average training loss
    """
    from tqdm import tqdm

    if loss_weights is None:
        loss_weights = {
            'high_prediction': 1.0,
            'low_prediction': 1.0,
            'hit_band': 0.5,
            'hit_target': 0.5,
            'expected_return': 0.3,
            'overshoot': 0.3,
            'continuation_duration': 0.5,
            'continuation_gain': 0.5,
            'continuation_confidence': 0.3,
            'price_change_pct': 0.4,
            'horizon_bars_log': 0.3,
            'adaptive_confidence': 0.2,
            # Breakout prediction weights (v3.21)
            'breakout_occurred': 0.4,
            'breakout_direction': 0.3,
            'breakout_bars_log': 0.2,
            'breakout_magnitude': 0.2
        }

    # For DistributedSampler: update epoch for reproducible shuffling
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0

    # Handle DataParallel: access underlying model for attribute checks
    base_model = model.module if hasattr(model, 'module') else model

    # Debug: log before DataLoader iteration
    import time as _time
    _epoch_start = _time.perf_counter()
    if profiler:
        profiler.snapshot("pre_dataloader_iter", 0, force_log=True)
        profiler.log_phase("dataloader_starting", {"num_workers": dataloader.num_workers, "batch_size": dataloader.batch_size})

    # Progress bar for batches
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1} [Train]", leave=False, ncols=80)

    for batch_idx, (x, targets_dict) in enumerate(pbar):
        # Debug: log after first batch received (proves DataLoader worked)
        if batch_idx == 0:
            _first_batch_time = _time.perf_counter() - _epoch_start
            print(f"[TIMING] First batch loaded in {_first_batch_time:.1f}s", file=__import__('sys').stderr, flush=True)
            if profiler:
                profiler.log_info(f"FIRST_BATCH_COMPLETE | time_sec={_first_batch_time:.1f}")
                profiler.snapshot("first_batch_received", 0, force_log=True)
        # Move to device
        # If collate already moved to device, this is a no-op; otherwise it moves now
        x = x.to(device, non_blocking=True)

        # Move all targets to device
        target_high = targets_dict['high'].to(device, non_blocking=True)
        target_low = targets_dict['low'].to(device, non_blocking=True)

        if base_model.multi_task:
            target_hit_band = targets_dict['hit_band'].to(device, non_blocking=True)
            target_hit_target = targets_dict['hit_target'].to(device, non_blocking=True)
            target_expected_return = targets_dict['expected_return'].to(device, non_blocking=True)
            target_overshoot = targets_dict['overshoot'].to(device, non_blocking=True)
            target_continuation_duration = targets_dict['continuation_duration'].to(device, non_blocking=True)
            target_continuation_gain = targets_dict['continuation_gain'].to(device, non_blocking=True)
            target_continuation_confidence = targets_dict['continuation_confidence'].to(device, non_blocking=True)
            target_price_change_pct = targets_dict['price_change_pct'].to(device, non_blocking=True)
            target_horizon_bars_log = targets_dict['horizon_bars_log'].to(device, non_blocking=True)
            target_adaptive_confidence = targets_dict['adaptive_confidence'].to(device, non_blocking=True)

            # Adaptive mode targets (only exist when using adaptive continuation mode)
            if 'adaptive_horizon' in targets_dict:
                target_adaptive_horizon = targets_dict['adaptive_horizon'].to(device, non_blocking=True)
            if 'conf_score' in targets_dict:
                target_conf_score = targets_dict['conf_score'].to(device, non_blocking=True)

            # Breakout prediction targets (v3.21)
            if 'breakout_occurred' in targets_dict:
                target_breakout_occurred = targets_dict['breakout_occurred'].to(device, non_blocking=True)
                target_breakout_direction = targets_dict['breakout_direction'].to(device, non_blocking=True)
                target_breakout_bars_log = targets_dict['breakout_bars_log'].to(device, non_blocking=True)
                target_breakout_magnitude = targets_dict['breakout_magnitude'].to(device, non_blocking=True)

        # Forward pass with optional AMP
        use_amp = scaler is not None

        # Use autocast for AMP, otherwise normal forward
        if use_amp:
            with torch.amp.autocast('cuda'):
                predictions, hidden_states = model.forward(x)

                # Primary loss (high/low regression)
                pred_high = predictions[:, 0]
                pred_low = predictions[:, 1]

                loss_high = criterion(pred_high, target_high)
                loss_low = criterion(pred_low, target_low)

                # Weighted primary loss
                loss = (loss_weights['high_prediction'] * loss_high +
                        loss_weights['low_prediction'] * loss_low)

                # Multi-task losses
                if base_model.multi_task and 'multi_task' in hidden_states:
                    mt = hidden_states['multi_task']

                    # Hit band (binary classification)
                    # Disable autocast for BCE - unsafe with FP16
                    with torch.amp.autocast('cuda', enabled=False):
                        loss_hit_band = F.binary_cross_entropy(
                            mt['hit_band'].float().squeeze(),
                            target_hit_band.float()
                        )
                    loss += loss_weights['hit_band'] * loss_hit_band

                    # Hit target (binary classification)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss_hit_target = F.binary_cross_entropy(
                            mt['hit_target'].float().squeeze(),
                            target_hit_target.float()
                        )
                    loss += loss_weights['hit_target'] * loss_hit_target

                    # Expected return (regression)
                    loss_expected_return = criterion(
                        mt['expected_return'].squeeze(),
                        target_expected_return
                    )
                    loss += loss_weights['expected_return'] * loss_expected_return

                    # Overshoot (regression)
                    loss_overshoot = criterion(
                        mt['overshoot'].squeeze(),
                        target_overshoot
                    )
                    loss += loss_weights['overshoot'] * loss_overshoot

                    # Continuation duration (regression)
                    loss_continuation_duration = criterion(
                        mt['continuation_duration'].squeeze(),
                        target_continuation_duration
                    )
                    loss += loss_weights['continuation_duration'] * loss_continuation_duration

                    # Continuation gain (regression)
                    loss_continuation_gain = criterion(
                        mt['continuation_gain'].squeeze(),
                        target_continuation_gain
                    )
                    loss += loss_weights['continuation_gain'] * loss_continuation_gain

                    # Continuation confidence (binary classification)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss_continuation_confidence = F.binary_cross_entropy(
                            mt['continuation_confidence'].float().squeeze(),
                            target_continuation_confidence.float()
                        )
                    loss += loss_weights['continuation_confidence'] * loss_continuation_confidence

                    # Adaptive horizon losses (only when using adaptive mode)
                    if 'adaptive_horizon' in targets_dict and 'adaptive_horizon' in mt:
                        loss_adaptive_horizon = criterion(
                            mt['adaptive_horizon'].squeeze(),
                            target_adaptive_horizon
                        )
                        loss += loss_weights.get('adaptive_horizon', 0.3) * loss_adaptive_horizon

                    if 'conf_score' in targets_dict and 'adaptive_conf_score' in mt:
                        with torch.amp.autocast('cuda', enabled=False):
                            loss_adaptive_conf = F.binary_cross_entropy(
                                mt['adaptive_conf_score'].float().squeeze(),
                                target_conf_score.float()
                            )
                        loss += loss_weights.get('adaptive_conf_score', 0.3) * loss_adaptive_conf

                    # Adaptive projection losses
                    loss_price_change = criterion(
                        mt['price_change_pct'].squeeze(),
                        target_price_change_pct
                    )
                    loss += loss_weights['price_change_pct'] * loss_price_change

                    loss_horizon_log = criterion(
                        mt['horizon_bars_log'].squeeze(),
                        target_horizon_bars_log
                    )
                    loss += loss_weights['horizon_bars_log'] * loss_horizon_log

                    with torch.amp.autocast('cuda', enabled=False):
                        loss_adaptive_confidence = F.binary_cross_entropy(
                            mt['adaptive_confidence'].float().squeeze(),
                            target_adaptive_confidence.float()
                        )
                    loss += loss_weights['adaptive_confidence'] * loss_adaptive_confidence

                # Breakout prediction losses (v3.21)
                if 'breakout' in hidden_states and 'breakout_occurred' in targets_dict:
                    bo = hidden_states['breakout']

                    # Breakout occurred (binary classification)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss_breakout_occurred = F.binary_cross_entropy(
                            bo['probability'].float().squeeze(),
                            target_breakout_occurred.float()
                        )
                    loss += loss_weights['breakout_occurred'] * loss_breakout_occurred

                    # Breakout direction (binary classification)
                    with torch.amp.autocast('cuda', enabled=False):
                        loss_breakout_direction = F.binary_cross_entropy(
                            bo['direction'].float().squeeze(),
                            target_breakout_direction.float()
                        )
                    loss += loss_weights['breakout_direction'] * loss_breakout_direction

                    # Breakout timing (regression - log-scaled bars)
                    loss_breakout_bars = criterion(
                        bo['bars_until'].squeeze().log().clamp(-10, 10),
                        target_breakout_bars_log
                    )
                    loss += loss_weights['breakout_bars_log'] * loss_breakout_bars

                    # Breakout magnitude (regression)
                    loss_breakout_magnitude = criterion(
                        bo['confidence'].squeeze(),  # Use confidence head for magnitude
                        target_breakout_magnitude.clamp(0, 5)  # Cap at 5x channel width
                    )
                    loss += loss_weights['breakout_magnitude'] * loss_breakout_magnitude

            # AMP backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Handle DataParallel: get underlying model for gradient clipping
            model_for_grad = model.module if use_multi_gpu else model
            torch.nn.utils.clip_grad_norm_(model_for_grad.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            # Standard FP32 forward pass
            predictions, hidden_states = model.forward(x)

            # Primary loss (high/low regression)
            pred_high = predictions[:, 0]
            pred_low = predictions[:, 1]

            loss_high = criterion(pred_high, target_high)
            loss_low = criterion(pred_low, target_low)

            # Weighted primary loss
            loss = (loss_weights['high_prediction'] * loss_high +
                    loss_weights['low_prediction'] * loss_low)

            # Multi-task losses
            if base_model.multi_task and 'multi_task' in hidden_states:
                mt = hidden_states['multi_task']

                # Hit band (binary classification)
                loss_hit_band = F.binary_cross_entropy(
                    mt['hit_band'].squeeze(),
                    target_hit_band
                )
                loss += loss_weights['hit_band'] * loss_hit_band

                # Hit target (binary classification)
                loss_hit_target = F.binary_cross_entropy(
                    mt['hit_target'].squeeze(),
                    target_hit_target
                )
                loss += loss_weights['hit_target'] * loss_hit_target

                # Expected return (regression)
                loss_expected_return = criterion(
                    mt['expected_return'].squeeze(),
                    target_expected_return
                )
                loss += loss_weights['expected_return'] * loss_expected_return

                # Overshoot (regression)
                loss_overshoot = criterion(
                    mt['overshoot'].squeeze(),
                    target_overshoot
                )
                loss += loss_weights['overshoot'] * loss_overshoot

                # Continuation duration (regression)
                loss_continuation_duration = criterion(
                    mt['continuation_duration'].squeeze(),
                    target_continuation_duration
                )
                loss += loss_weights['continuation_duration'] * loss_continuation_duration

                # Continuation gain (regression)
                loss_continuation_gain = criterion(
                    mt['continuation_gain'].squeeze(),
                    target_continuation_gain
                )
                loss += loss_weights['continuation_gain'] * loss_continuation_gain

                # Continuation confidence (binary classification)
                loss_continuation_confidence = F.binary_cross_entropy(
                    mt['continuation_confidence'].squeeze(),
                    target_continuation_confidence
                )
                loss += loss_weights['continuation_confidence'] * loss_continuation_confidence

                # Adaptive horizon losses (only when using adaptive mode)
                if 'adaptive_horizon' in targets_dict and 'adaptive_horizon' in mt:
                    loss_adaptive_horizon = criterion(
                        mt['adaptive_horizon'].squeeze(),
                        target_adaptive_horizon
                    )
                    loss += loss_weights.get('adaptive_horizon', 0.3) * loss_adaptive_horizon

                if 'conf_score' in targets_dict and 'adaptive_conf_score' in mt:
                    loss_adaptive_conf = F.binary_cross_entropy(
                        mt['adaptive_conf_score'].squeeze(),
                        target_conf_score
                    )
                    loss += loss_weights.get('adaptive_conf_score', 0.3) * loss_adaptive_conf

                # Adaptive projection losses
                loss_price_change = criterion(
                    mt['price_change_pct'].squeeze(),
                    target_price_change_pct
                )
                loss += loss_weights['price_change_pct'] * loss_price_change

                loss_horizon_log = criterion(
                    mt['horizon_bars_log'].squeeze(),
                    target_horizon_bars_log
                )
                loss += loss_weights['horizon_bars_log'] * loss_horizon_log

                loss_adaptive_confidence = F.binary_cross_entropy(
                    mt['adaptive_confidence'].squeeze(),
                    target_adaptive_confidence
                )
                loss += loss_weights['adaptive_confidence'] * loss_adaptive_confidence

            # Breakout prediction losses (v3.21)
            if 'breakout' in hidden_states and 'breakout_occurred' in targets_dict:
                bo = hidden_states['breakout']

                # Breakout occurred (binary classification)
                loss_breakout_occurred = F.binary_cross_entropy(
                    bo['probability'].squeeze(),
                    target_breakout_occurred
                )
                loss += loss_weights['breakout_occurred'] * loss_breakout_occurred

                # Breakout direction (binary classification)
                loss_breakout_direction = F.binary_cross_entropy(
                    bo['direction'].squeeze(),
                    target_breakout_direction
                )
                loss += loss_weights['breakout_direction'] * loss_breakout_direction

                # Breakout timing (regression - log-scaled bars)
                loss_breakout_bars = criterion(
                    bo['bars_until'].squeeze().log().clamp(-10, 10),
                    target_breakout_bars_log
                )
                loss += loss_weights['breakout_bars_log'] * loss_breakout_bars

                # Breakout magnitude (regression)
                loss_breakout_magnitude = criterion(
                    bo['confidence'].squeeze(),  # Use confidence head for magnitude
                    target_breakout_magnitude.clamp(0, 5)  # Cap at 5x channel width
                )
                loss += loss_weights['breakout_magnitude'] * loss_breakout_magnitude

            # Standard backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Handle DataParallel: get underlying model for gradient clipping
            model_for_grad = model.module if use_multi_gpu else model
            torch.nn.utils.clip_grad_norm_(model_for_grad.parameters(), max_norm=1.0)
            optimizer.step()

        # Immediate memory cleanup after gradient update (throttled for MPS)
        # Every 100 batches instead of every batch - much less overhead
        if device == 'mps' and batch_idx % 100 == 0:
            torch.mps.empty_cache()

        total_loss += loss.item()

        # Update progress bar with current loss (and memory if profiling)
        if profiler:
            mem = profiler.snapshot("batch_end", batch_idx)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mem': f'{mem.get("gpu_mb", 0):.0f}MB'})
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Memory cleanup to prevent accumulation
        # CUDA: every 500 batches (empty_cache is expensive, ~5-10ms per call)
        # MPS: every 50 batches (more memory constrained, needs more frequent cleanup)
        cleanup_frequency = 500 if device == 'cuda' else 50
        if batch_idx % cleanup_frequency == 0 and batch_idx > 0:
            # Track memory before cleanup if profiling
            if profiler:
                mem_before = profiler.snapshot("cleanup_start", batch_idx, force_log=True)

            # Clear model's cached hidden states
            base_model.clear_cached_states()

            # Explicitly delete intermediate tensors
            del predictions, hidden_states
            if base_model.multi_task:
                del mt

            # Explicitly delete batch data
            del x, targets_dict

            # Clear GPU/MPS cache (expensive operations - do sparingly)
            if device == 'cuda':
                torch.cuda.empty_cache()
            elif device == 'mps':
                torch.mps.empty_cache()

            # Run garbage collection
            import gc
            gc.collect()

            # Log memory freed if profiling
            if profiler:
                mem_after = profiler.snapshot("cleanup_end", batch_idx, force_log=True)
                freed_gpu = mem_before.get('gpu_mb', 0) - mem_after.get('gpu_mb', 0)
                freed_ram = mem_before.get('ram_mb', 0) - mem_after.get('ram_mb', 0)
                profiler.log_cleanup(freed_gpu, freed_ram)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model: HierarchicalLNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """
    Validate model (handles both dict and tensor targets).

    Args:
        model: HierarchicalLNN model
        dataloader: Validation data loader
        criterion: Loss function
        device: 'cuda', 'mps', or 'cpu'

    Returns:
        avg_loss: Average validation loss
        avg_error: Average prediction error (%)
    """
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_error = 0.0

    # Progress bar for validation
    pbar = tqdm(dataloader, desc="  Validating", leave=False, ncols=100)

    with torch.no_grad():
        for x, targets in pbar:
            # Move to device
            x = x.to(device)

            # Handle dict targets (multi-task) or tensor targets (legacy)
            if isinstance(targets, dict):
                target_high = targets['high'].to(device)
                target_low = targets['low'].to(device)
            else:
                # Legacy tensor format
                targets = targets.to(device)
                target_high = targets[:, 0]
                target_low = targets[:, 1]

            # Forward pass
            predictions, _ = model.forward(x)

            # Extract predictions
            pred_high = predictions[:, 0]
            pred_low = predictions[:, 1]

            # Calculate loss (primary targets only for validation)
            loss = criterion(pred_high, target_high) + criterion(pred_low, target_low)
            total_loss += loss.item()

            # Calculate error (MAE)
            error = (torch.abs(pred_high - target_high) + torch.abs(pred_low - target_low)) / 2
            total_error += error.mean().item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'error': f'{error.mean().item():.4f}%'})

    avg_loss = total_loss / len(dataloader)
    avg_error = total_error / len(dataloader)

    return avg_loss, avg_error


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

    # Initial cache directory selection
    print("\n📂 Cache Directory Selection")
    cache_dir_default = getattr(args, 'shard_path', 'data/feature_cache') or 'data/feature_cache'
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
    cache_pairs = find_available_caches(args.shard_path)
    selected_cache_pair = None
    if cache_pairs:
        selected_cache_pair = pick_cache_pair(cache_pairs)

    # Default cache behavior: reuse if a cache pair was selected, regenerate otherwise
    args.regenerate_cache = False if selected_cache_pair else True

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

    # Consolidated precision menu for CUDA (combines AMP + base precision)
    args.amp = False  # Default to disabled
    args.precision_mode = 'fp32'  # Track the user's choice

    if args.device == 'cuda':
        print()
        precision_choice = inquirer.select(
            message="Training precision:",
            choices=[
                Choice(value='fp16_amp', name="FP16 (AMP) - 2-3x faster, uses tensor cores ⚡"),
                Choice(value='fp32', name="FP32 - Standard precision"),
                Choice(value='fp64', name="FP64 - Maximum precision (slowest)")
            ],
            default='fp16_amp'
        ).execute()

        args.precision_mode = precision_choice

        if precision_choice == 'fp16_amp':
            args.amp = True
            project_config.TRAINING_PRECISION = 'float32'
            project_config.NUMPY_DTYPE = np.float32
            project_config._TORCH_DTYPE = torch.float32
            print("   ⚡ FP16 (AMP) - Using FP16 tensor cores with FP32 base weights")
        elif precision_choice == 'fp32':
            args.amp = False
            project_config.TRAINING_PRECISION = 'float32'
            project_config.NUMPY_DTYPE = np.float32
            project_config._TORCH_DTYPE = torch.float32
            print("   → FP32 - Standard precision training")
        else:  # fp64
            args.amp = False
            project_config.TRAINING_PRECISION = 'float64'
            project_config.NUMPY_DTYPE = np.float64
            project_config._TORCH_DTYPE = torch.float64
            print("   → FP64 - Maximum precision (slower, more memory)")

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
        args.amp = False
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

    # Preload to RAM option (for CUDA with high-RAM systems)
    # NOTE: Only relevant for legacy mmap mode - native TF mode uses smaller files (~5GB vs ~90GB)
    args.preload_to_ram = False  # Default

    # v4.1: Skip preload option when native TF mode is enabled (default)
    # Native TF files are already small and efficient, preloading provides no benefit
    if getattr(args, 'use_native_timeframes', True):
        # Native TF mode is ON by default - no need for preload option
        pass  # preload_to_ram stays False, no question asked
    elif args.device.startswith('cuda'):
        print()
        # Detect RAM for guidance
        try:
            import psutil
            detected_ram = psutil.virtual_memory().total / (1024**3)
            container_ram = getattr(args, 'container_ram_gb', 0)
            if container_ram > 0:
                detected_ram = container_ram
        except:
            detected_ram = 0

        # Only offer preload if system has enough RAM (90GB for data + overhead)
        if detected_ram >= 100:
            args.preload_to_ram = inquirer.confirm(
                message=f"Preload all data to RAM? (~90GB, requires {int(detected_ram)}GB available - MUCH faster training)",
                default=True  # Default to yes for high-RAM systems
            ).execute()

            if args.preload_to_ram:
                print("   📦 Will load all channel data into RAM at startup")
                print("   ⚡ Data access will be instant (RAM speed vs 400ms/sample disk I/O)")
                print("   → Forcing num_workers=0 (data loading is instant, no prefetch needed)")
        elif detected_ram >= 50:
            print(f"   ℹ️  Detected {detected_ram:.0f}GB RAM - preload requires ~100GB (using mmap mode)")
            args.preload_to_ram = False

    # Data loading workers (RIGHT after device selection)
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

    if args.preload_to_ram:
        # With preload, each worker duplicates ~93GB of numpy arrays (spawn multiprocessing)
        # Calculate max safe workers based on available RAM
        preload_per_process = 93  # GB per process (main or worker)
        base_overhead = 5  # GB for model, optimizer, etc.
        max_safe_workers = max(0, int((total_ram_gb - base_overhead) / preload_per_process) - 1)

        print(f"   ⚠️  PRELOAD MODE: Each worker duplicates ~93GB (spawn multiprocessing)")
        print(f"      RAM: {total_ram_gb:.0f}GB available")
        print(f"      0 workers: ~{base_overhead + preload_per_process}GB")
        if max_safe_workers >= 1:
            print(f"      1 worker:  ~{base_overhead + 2*preload_per_process}GB")
        if max_safe_workers >= 2:
            print(f"      2 workers: ~{base_overhead + 3*preload_per_process}GB")
        if max_safe_workers >= 3:
            print(f"      3 workers: ~{base_overhead + 4*preload_per_process}GB")
        print(f"      Max safe: {max_safe_workers} workers")
        print(f"      Note: With data in RAM, __getitem__ is <1ms (workers add minimal benefit)")

        default_workers = 0  # Recommend 0 since data loading is instant
        args.num_workers = int(inquirer.number(
            message=f"Data loading workers (recommended: 0, max safe: {max_safe_workers}):",
            default=dflt('num_workers', default_workers),
            min_allowed=0,
            max_allowed=max(max_safe_workers, 0)
        ).execute())

        if args.num_workers > 0:
            estimated_ram = base_overhead + (args.num_workers + 1) * preload_per_process
            print(f"   → Using {args.num_workers} workers (~{estimated_ram}GB estimated RAM)")
    else:
        default_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}.get(args.device, 2)

        if total_ram_gb > 0:
            guidance = f"\n   ℹ️  RAM: {total_ram_gb:.0f}GB available. With large datasets, each worker uses extra RAM."
        else:
            guidance = ""

        args.num_workers = int(inquirer.number(
            message=f"Data loading workers (CPU threads for batch prep, recommended: {default_workers}):{guidance}",
            default=dflt('num_workers', default_workers),
            min_allowed=0,
            max_allowed=128  # High limit for systems with many cores (e.g., 32-64 core servers)
        ).execute())

    if not args.preload_to_ram and args.num_workers != default_workers:
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

    # Get recommended batch size
    total_ram = hw_info.get('total_ram_gb', 16)
    recommended_batch = get_recommended_batch_size(args.device, total_ram)

    # Training data range
    print()
    if selected_cache_pair:
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
                m_start, m_end = 2015, 2022
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
    if not selected_cache_pair:
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
    will_use_cache = not getattr(args, 'regenerate_cache', True)
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if selected_cache_pair and will_use_cache:
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
        default_feature_workers = min(n_cores - 1, 8)

        args.feature_workers = int(inquirer.number(
            message=f"Feature extraction cores (0 = use all {n_cores}, default: {default_feature_workers}):",
            default=default_feature_workers,
            min_allowed=0,
            max_allowed=128
        ).execute())

        if args.feature_workers == 0:
            actual_cores = os.cpu_count()
            print(f"   → Using ALL {actual_cores} CPU cores for feature extraction")
            project_config.MAX_PARALLEL_WORKERS = actual_cores
        else:
            print(f"   → Using {args.feature_workers} cores for feature extraction")
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

    # Note: Precision is now selected in the consolidated menu right after device selection
    # This section only handles continuation mode

    # Continuation label mode selection (skip if using cached features/labels)
    if selected_cache_pair and will_use_cache:
        continuation_mode = project_config.CONTINUATION_MODE
        print(f"\n   ⚙️  Using cached continuation mode: {continuation_mode}")
    else:
        print()
        continuation_mode = inquirer.select(
            message="Continuation prediction mode:",
            choices=[
                Choice(value='simple', name='Simple - Fixed 24-bar for all targets ⭐ Baseline'),
                Choice(value='adaptive_labels', name='Adaptive Labels - Adaptive continuation, fixed high/low 🎯 Default'),
                Choice(value='adaptive_full', name='Fully Adaptive - All targets use adaptive horizon 🔬 Experimental'),
            ],
            default=dflt('continuation_mode', 'adaptive_labels')
        ).execute()

    # Update config with continuation mode
    project_config.CONTINUATION_MODE = continuation_mode

    if continuation_mode == 'simple':
        print(f"   → Simple mode: Fixed 24-bar horizon (24 minutes)")
        print(f"      All targets calculated over same fixed window")

    elif continuation_mode == 'adaptive_labels':
        print(f"   → Adaptive Labels mode:")
        print(f"      Primary targets (high/low): Fixed 24-bar window")
        print(f"      Continuation labels: Adaptive horizon based on confidence")
        print()
        print(f"   Current adaptive horizon range: {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars")

        # Ask if user wants to customize the horizon range (only if not locked to cache)
        customize_horizon = False
        if not (selected_cache_pair and will_use_cache):
            customize_horizon = inquirer.confirm(
                message="Customize adaptive horizon range?",
                default=False
            ).execute()

        if customize_horizon:
            # Get minimum horizon
            min_horizon = int(inquirer.number(
                message="Minimum horizon (bars):",
                default=20,
                min_allowed=10,
                max_allowed=60
            ).execute())

            # Get maximum horizon (must be >= min_horizon)
            max_horizon = int(inquirer.number(
                message="Maximum horizon (bars):",
                default=40,
                min_allowed=min_horizon,
                max_allowed=100
            ).execute())

            # Update config values in memory
            project_config.ADAPTIVE_MIN_HORIZON = min_horizon
            project_config.ADAPTIVE_MAX_HORIZON = max_horizon

            print(f"   ✓ Updated adaptive horizons: {min_horizon}-{max_horizon} bars ({min_horizon}-{max_horizon} minutes)")

            # Warn about cache invalidation if values differ from defaults
            if min_horizon != 20 or max_horizon != 40:
                print(f"   ⚠️  Non-default horizons will invalidate continuation label cache")
        else:
            print(f"   → Using default horizons: {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars")

        print(f"   ℹ️  Horizon adjusts based on RSI/slope confidence")
        print(f"   ℹ️  High/low targets: Fixed 24-bar window (multi-task learning)")

    elif continuation_mode == 'adaptive_full':
        print(f"   → Fully Adaptive mode (EXPERIMENTAL):")
        print(f"      ALL targets use adaptive horizon (high/low + continuation)")
        print()
        print(f"   Current adaptive horizon range: {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars")

        # Ask if user wants to customize the horizon range (only if not locked to cache)
        customize_horizon = False
        if not (selected_cache_pair and will_use_cache):
            customize_horizon = inquirer.confirm(
                message="Customize adaptive horizon range?",
                default=False
            ).execute()

        if customize_horizon:
            # Get minimum horizon
            min_horizon = int(inquirer.number(
                message="Minimum horizon (bars):",
                default=20,
                min_allowed=10,
                max_allowed=60
            ).execute())

            # Get maximum horizon (must be >= min_horizon)
            max_horizon = int(inquirer.number(
                message="Maximum horizon (bars):",
                default=40,
                min_allowed=min_horizon,
                max_allowed=100
            ).execute())

            # Update config values in memory
            project_config.ADAPTIVE_MIN_HORIZON = min_horizon
            project_config.ADAPTIVE_MAX_HORIZON = max_horizon

            print(f"   ✓ Updated adaptive horizons: {min_horizon}-{max_horizon} bars ({min_horizon}-{max_horizon} minutes)")

            # Warn about cache invalidation if values differ from defaults
            if min_horizon != 20 or max_horizon != 40:
                print(f"   ⚠️  Non-default horizons will invalidate continuation label cache")
        else:
            print(f"   → Using default horizons: {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars")

        print(f"   ℹ️  All targets (high/low/continuation) calculated over SAME adaptive horizon")
        print(f"   ⚠️  Experimental - targets vary per sample based on confidence")

    else:  # simple mode
        print(f"   → Simple mode: Fixed 24-bar horizon (24 minutes at 1-min resolution)")
        print(f"      All targets over same fixed window")

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

    # Multi-task learning
    print()
    args.multi_task = inquirer.confirm(
        message="Enable multi-task learning (hit_band, hit_target, expected_return)?",
        default=dflt('multi_task', True)
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
    precision_mode = getattr(args, 'precision_mode', 'fp32')
    if precision_mode == 'fp16_amp':
        precision_display = "FP16 (AMP) ⚡"
    elif precision_mode == 'fp64':
        precision_display = "FP64"
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
    print(f"  Data Loading: mmap + OS page cache (no pre-merge)")
    print(f"  Cache: {'Regenerate' if getattr(args, 'regenerate_cache', True) else 'Use existing'}")
    print(f"  Feature GPU: {'Yes' if getattr(args, 'use_gpu_features', False) else 'No'}")
    print(f"  Parallel CPU: {parallel_str}")
    print(f"  Chunking: {chunk_str}")
    print(f"  Continuation Mode: {project_config.CONTINUATION_MODE} "
          f"(horizon {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars)")
    print(f"  Model Capacity: internal_ratio={args.internal_neurons_ratio}, hidden_size={args.hidden_size}")
    print(f"  Multi-Task: {'Enabled' if args.multi_task else 'Disabled'}")
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

    # Native timeframes requires non-chunked mode for pre-computed sequences
    if args.use_native_timeframes and args.use_chunking:
        if is_main_process(rank):
            print(f"⚠️  Native timeframes requires --no-chunking (disabling chunking)")
        args.use_chunking = False

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

    # Load configuration
    config = None
    loss_weights = None
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            loss_weights = config.get('loss_weights', None)
        if is_main_process(rank):
            print(f"✅ Loaded config from: {args.config}")
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
        print(f"💾 Data mode: mmap + OS page cache")
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

    # Extract features (handles caching internally)
    if is_main_process(rank):
        print(f"   Extracting features (use_chunking={args.use_chunking})...")

    result = extractor.extract_features(
        df,
        use_cache=True,
        continuation=True,
        use_chunking=args.use_chunking,
        use_gpu=args.use_gpu_features,  # Pass user's GPU selection
        shard_storage_path=str(shard_path),
        vix_data=vix_data,  # v3.20: VIX features for volatility regime
        events_handler=events_handler,  # v4.0: Event features for earnings/FOMC patterns
    )
    # Handle variable return: (features_df, continuation_df) or (features_df, continuation_df, mmap_meta_path)
    features_df = result[0]
    continuation_df = result[1]
    mmap_meta_path = result[2] if len(result) > 2 else None
    non_channel_cols = features_df.columns.tolist() if features_df is not None else None

    if profiler:
        nc_cols = len(non_channel_cols) if non_channel_cols is not None else 0
        cont_rows = len(continuation_df) if continuation_df is not None else 0
        profiler.log_info(f"FEATURES_EXTRACTED | non_channel_cols={nc_cols} | continuation_rows={cont_rows}")
        profiler.snapshot("post_feature_extraction", 0, force_log=True)

    # Save cache manifest for future reuse (supports both mmap and pickle modes)
    if is_main_process(rank) and hasattr(extractor, '_cache_key'):
        cache_key = extractor._cache_key
        cache_dir = extractor._unified_cache_dir if hasattr(extractor, '_unified_cache_dir') else shard_path
        cont_path = str(extractor._cont_cache_path) if hasattr(extractor, '_cont_cache_path') and extractor._cont_cache_path else None
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
    # AUTO-DISCOVER TF_META PATH FOR NATIVE TIMEFRAME MODE
    # =========================================================================
    if args.use_native_timeframes and args.tf_meta_path is None:
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
                # Try to find any tf_meta file in cache dir
                tf_meta_files = list(Path(cache_dir).glob("tf_meta_*.json"))
                if tf_meta_files:
                    args.tf_meta_path = str(tf_meta_files[0])
                    if is_main_process(rank):
                        print(f"   🔄 Auto-discovered native timeframe meta: {tf_meta_files[0].name}")
                else:
                    if is_main_process(rank):
                        print(f"   ⚠️  Native timeframes enabled but no tf_meta_*.json found in {cache_dir}")
                        print(f"       Run with --no-chunking to generate native timeframe sequences")
                        print(f"       Or pass --tf-meta /path/to/tf_meta_*.json explicitly")

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

    train_dataset, val_dataset = create_hierarchical_dataset(
        features_df=features_df,
        raw_ohlc_df=df,  # Must match features_df length (both from full df)
        continuation_labels_df=continuation_df,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        validation_split=args.val_split,
        include_continuation=True,
        mmap_meta_path=mmap_meta_path,
        profiler=dataset_profiler,
        preload_to_ram=args.preload_to_ram,
        use_native_timeframes=args.use_native_timeframes,
        tf_meta_path=args.tf_meta_path
    )

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

    # Get feature dimensions from dataset (always 3-tuple: main, monthly, non_channel)
    sample_data, sample_target = train_dataset[0]
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
    train_sampler = None
    if is_distributed:
        # DDP: use DistributedSampler
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
            print(f"   Using DistributedSampler ({world_size} replicas)")
    else:
        # Standard shuffle (mmap-only mode with OS page cache)
        train_loader_kwargs['shuffle'] = True
        if is_main_process(rank):
            print(f"   Using standard shuffle")

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

    # Log memory after DataLoader creation (measure worker spawn impact)
    if profiler:
        profiler.snapshot("post_dataloader_creation", 0, force_log=True)
        profiler.log_info(f"DATALOADERS_CREATED | train_batches={len(train_loader)} | val_batches={len(val_loader)}")

    if is_main_process(rank):
        print(f"   Train batches: {len(train_loader):,}")
        print(f"   Val batches: {len(val_loader):,}")

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
        )
    else:
        # Legacy mode - same size for all timeframes
        model = HierarchicalLNN(
            input_size=total_features,  # Backward compat: same size for all timeframes
            hidden_size=args.hidden_size,
            internal_neurons_ratio=args.internal_neurons_ratio,
            device=args.device,
            multi_task=args.multi_task,
        )

    if profiler:
        total_params = sum(p.numel() for p in model.parameters())
        profiler.log_info(f"MODEL_CREATED | params={total_params:,} | device={args.device}")
        profiler.snapshot("post_model_create", 0, force_log=True)

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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )

    # AMP scaler for mixed precision
    scaler = None
    if args.amp and args.device.startswith('cuda'):
        scaler = torch.amp.GradScaler('cuda')
        if is_main_process(rank):
            print("   ✓ AMP enabled (FP16 training)")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

    if profiler:
        profiler.log_info(f"TRAINING_START | device={args.device} | batch_size={args.batch_size} | num_workers={args.num_workers}")
        profiler.snapshot("pre_training_loop", 0, force_log=True)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_errors = []

    # Progress bar (rank 0 only)
    epoch_pbar = tqdm(range(args.epochs), desc="Training", disable=not is_main_process(rank))

    for epoch in epoch_pbar:
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if profiler:
            profiler.log_phase(f"EPOCH_START | epoch={epoch + 1}")
            profiler.snapshot("pre_train_epoch", epoch + 1, force_log=True)

        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        if profiler:
            profiler.snapshot("pre_dataloader_iter", epoch + 1, force_log=True)
            profiler.log_phase(f"dataloader_starting | epoch={epoch + 1} | batch=0 | num_workers={args.num_workers} | batch_size={args.batch_size}")

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False, disable=not is_main_process(rank))

        # Track if this is the very first batch (for torch.compile feedback)
        _is_first_batch_ever = (epoch == 0)
        _first_forward_start = None

        for batch_idx, (features, targets) in enumerate(batch_pbar):
            if profiler and batch_idx == 0:
                profiler.log_info(f"FIRST_BATCH_COMPLETE | time_sec={0}")
                profiler.snapshot("first_batch_received", epoch + 1, force_log=True)

            features = features.to(args.device, non_blocking=True)
            # Move each tensor in targets dict to device
            targets = {k: v.to(args.device, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad()

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

            # Forward pass with optional AMP
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    predictions, hidden_states = model(features)
                    # Primary loss: high/low predictions (raw % - data alignment fixed in dataset)
                    target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                    loss = F.mse_loss(predictions[:, :2], target_tensor)

                    # Debug: Print target/prediction stats on first batch
                    if batch_idx == 0 and epoch == 0:
                        print(f"[DEBUG] targets high: mean={target_tensor[:,0].mean():.2f}%, min={target_tensor[:,0].min():.2f}%, max={target_tensor[:,0].max():.2f}%", flush=True)
                        print(f"[DEBUG] targets low: mean={target_tensor[:,1].mean():.2f}%, min={target_tensor[:,1].min():.2f}%, max={target_tensor[:,1].max():.2f}%", flush=True)
                        print(f"[DEBUG] predictions: mean={predictions[:,:2].mean():.3f}, std={predictions[:,:2].std():.3f}", flush=True)
                        print(f"[DEBUG] primary loss (high/low MSE): {loss.item():.4f}", flush=True)

                    # Multi-task losses (if enabled)
                    if args.multi_task and 'multi_task' in hidden_states:
                        mt = hidden_states['multi_task']
                        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(mt['hit_band'].squeeze(), targets['hit_band'])
                        loss = loss + 0.1 * F.binary_cross_entropy_with_logits(mt['hit_target'].squeeze(), targets['hit_target'])
                        loss = loss + 0.1 * F.mse_loss(mt['expected_return'].squeeze(), targets['expected_return'])

                        # Adaptive projection losses (trains the adaptive_projection network)
                        if 'price_change_pct' in mt and 'price_change_pct' in targets:
                            loss = loss + 0.4 * F.mse_loss(mt['price_change_pct'].squeeze(), targets['price_change_pct'])
                            loss = loss + 0.3 * F.mse_loss(mt['horizon_bars_log'].squeeze(), targets['horizon_bars_log'])
                            # adaptive_confidence uses BCE (sigmoid output vs 0/1 target)
                            loss = loss + 0.2 * F.binary_cross_entropy(
                                mt['adaptive_confidence'].float().squeeze(),
                                targets['adaptive_confidence'].float()
                            )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions, hidden_states = model(features)
                # Primary loss: high/low predictions (raw % - data alignment fixed in dataset)
                target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                loss = F.mse_loss(predictions[:, :2], target_tensor)

                # Debug: Print target/prediction stats on first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"[DEBUG] targets high: mean={target_tensor[:,0].mean():.2f}%, min={target_tensor[:,0].min():.2f}%, max={target_tensor[:,0].max():.2f}%", flush=True)
                    print(f"[DEBUG] targets low: mean={target_tensor[:,1].mean():.2f}%, min={target_tensor[:,1].min():.2f}%, max={target_tensor[:,1].max():.2f}%", flush=True)
                    print(f"[DEBUG] predictions: mean={predictions[:,:2].mean():.3f}, std={predictions[:,:2].std():.3f}", flush=True)
                    print(f"[DEBUG] primary loss (high/low MSE): {loss.item():.4f}", flush=True)

                # Multi-task losses (if enabled)
                if args.multi_task and 'multi_task' in hidden_states:
                    mt = hidden_states['multi_task']
                    loss = loss + 0.1 * F.binary_cross_entropy_with_logits(mt['hit_band'].squeeze(), targets['hit_band'])
                    loss = loss + 0.1 * F.binary_cross_entropy_with_logits(mt['hit_target'].squeeze(), targets['hit_target'])
                    loss = loss + 0.1 * F.mse_loss(mt['expected_return'].squeeze(), targets['expected_return'])

                    # Adaptive projection losses (trains the adaptive_projection network)
                    if 'price_change_pct' in mt and 'price_change_pct' in targets:
                        loss = loss + 0.4 * F.mse_loss(mt['price_change_pct'].squeeze(), targets['price_change_pct'])
                        loss = loss + 0.3 * F.mse_loss(mt['horizon_bars_log'].squeeze(), targets['horizon_bars_log'])
                        # adaptive_confidence uses BCE (sigmoid output vs 0/1 target)
                        loss = loss + 0.2 * F.binary_cross_entropy(
                            mt['adaptive_confidence'].float().squeeze(),
                            targets['adaptive_confidence'].float()
                        )
                loss.backward()
                optimizer.step()

            # Report first batch completion time and stop progress thread
            if _is_first_batch_ever and batch_idx == 0 and _first_forward_start is not None:
                _compile_done.set()  # Stop the progress thread
                _first_forward_elapsed = time.perf_counter() - _first_forward_start
                print(f"   ✓ First batch complete ({_first_forward_elapsed:.1f}s) - subsequent batches will be fast", flush=True)
                _is_first_batch_ever = False  # Don't print again

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if profiler and batch_idx > 0 and batch_idx % profiler.log_every_n == 0:
                profiler.snapshot(f"batch_{batch_idx}", epoch + 1)

        avg_train_loss = train_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_error = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(args.device, non_blocking=True)
                # Move each tensor in targets dict to device
                targets = {k: v.to(args.device, non_blocking=True) for k, v in targets.items()}

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        predictions, hidden_states = model(features)
                        # Raw % (data alignment fixed in dataset)
                        target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                        loss = F.mse_loss(predictions[:, :2], target_tensor)
                else:
                    predictions, hidden_states = model(features)
                    # Raw % (data alignment fixed in dataset)
                    target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                    loss = F.mse_loss(predictions[:, :2], target_tensor)

                val_loss += loss.item()
                val_error += torch.abs(predictions[:, :2] - target_tensor).mean().item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)
        avg_val_error = val_error / max(num_val_batches, 1)
        val_losses.append(avg_val_loss)
        val_errors.append(avg_val_error)

        scheduler.step()

        # Update progress bar
        epoch_pbar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{avg_val_loss:.4f}',
            'best': f'{best_val_loss:.4f}'
        })

        # Check for improvement (rank 0 saves model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
                    'input_size': model_to_save.input_size,
                    'hidden_size': model_to_save.hidden_size,
                    'internal_neurons_ratio': model_to_save.internal_neurons_ratio
                }, output_path)

                tqdm.write(f"   ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
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

    # Print memory profile summary
    if profiler and is_main_process(rank):
        profiler.print_summary()
        profiler.close()

    # Save training history (rank 0 only)
    if is_main_process(rank):
        history_path = Path(args.output).parent / 'hierarchical_training_history.json'
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_errors': val_errors,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'args': vars(args)
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
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--preload', action='store_true',
                        help='Preload all data into memory')
    parser.add_argument('--use-gpu-features', dest='use_gpu_features', action='store_true', default=False,
                        help='Use GPU acceleration for feature extraction (CUDA only)')

    # v4.1: Native timeframe mode (default=True for performance)
    parser.add_argument('--native-timeframes', dest='use_native_timeframes', action='store_true', default=True,
                        help='Use native timeframe mode: each CfC layer receives features at its native resolution (5min layer sees 5-min bars)')
    parser.add_argument('--no-native-timeframes', dest='use_native_timeframes', action='store_false',
                        help='Disable native timeframe mode (use legacy mmap mode - SLOW)')
    parser.add_argument('--tf-meta', dest='tf_meta_path', type=str, default=None,
                        help='Path to tf_meta_*.json file for native timeframe mode')

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
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (CUDA only, 2-3x faster)')
    parser.add_argument('--use-compile', dest='use_compile', action='store_true', default=False,
                        help='Enable torch.compile JIT compilation (CUDA only, first batch slow but faster afterward)')
    parser.add_argument('--compile-verbose', dest='compile_verbose', action='store_true', default=False,
                        help='Enable verbose torch.compile output (sets TORCH_LOGS=dynamo,inductor for debugging graph breaks and kernel fusion)')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode with menus')

    # Memory profiling
    parser.add_argument('--memory-profile', dest='memory_profile', action='store_true', default=False,
                        help='Enable memory profiling to logs/memory_debug.log')

    args = parser.parse_args()

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
        mp.spawn(run_training, nprocs=num_gpus, args=(num_gpus, vars(args)))
    else:
        # Single GPU/device mode - call directly
        run_training(0, 1, vars(args))



if __name__ == '__main__':
    main()
