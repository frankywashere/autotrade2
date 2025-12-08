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
    if _collate_elapsed > 1.0:  # Log if >1 second
        print(f"[SLOW_COLLATE] batch assembly took {_collate_elapsed:.1f}s for {len(batch)} samples ({_collate_elapsed/len(batch)*1000:.0f}ms/sample)", file=sys.stderr, flush=True)

    return x, targets_batch, vix_batch, events_batch


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

    v5.2: Also tracks transition labels for multi-phase compositor
    """
    cache_dir = Path(cache_dir)
    caches = []
    seen_keys = set()

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
                cache_pairs = find_available_caches(saved_cache_path)

                # Check for native TF files
                cache_path = Path(saved_cache_path)
                tf_meta_files = list(cache_path.glob('tf_meta*.json'))
                native_tf_files = list(cache_path.glob('tf_sequence*.npy'))

                print(f"   Manifests: {'✓ ' + str(len(manifests)) + ' found' if manifests else '✗ None'}")
                print(f"   Cache pairs: {'✓ ' + str(len(cache_pairs)) + ' found' if cache_pairs else '✗ None'}")
                print(f"   TF Meta: {'✓' if tf_meta_files else '✗'}")
                print(f"   Native TF sequences: {'✓ ' + str(len(native_tf_files)) + ' files' if native_tf_files else '✗ None'}")

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

    # v5.2: Native TF mode is default - preload option removed (obsolete)
    args.preload_to_ram = False  # Always false

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
        min_allowed=0,
        max_allowed=128  # High limit for systems with many cores
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

    # Native TF generation option (only show if chunking is enabled OR existing chunks found)
    import glob
    chunk_meta_files = glob.glob(str(Path(args.shard_path or 'data/feature_cache') / 'features_mmap_meta_*.json'))
    has_existing_chunks = len(chunk_meta_files) > 0

    if has_existing_chunks or args.use_chunking:
        print()
        print("=" * 60)
        print("  NATIVE TIMEFRAME GENERATION")
        print("=" * 60)
        if has_existing_chunks:
            print(f"   Found {len(chunk_meta_files)} existing chunk metadata file(s)")

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

    # v5.3.1: Information Flow Direction
    print()
    args.information_flow = inquirer.select(
        message="Information flow strategy:",
        choices=[
            Choice('bottom_up', 'Bottom-Up - Fast → Slow (details inform strategy) ⭐ Default'),
            Choice('top_down', 'Top-Down - Slow → Fast (strategy guides details)'),
            Choice('bidirectional_bottom', 'Bidirectional (Bottom-First) - Micro foundation + macro overlay'),
            Choice('bidirectional_top', 'Bidirectional (Top-First) - Macro framework + micro refinement'),
        ],
        default='bottom_up'
    ).execute()

    if args.information_flow == 'bottom_up':
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
    if getattr(args, 'generate_native_tf', None):
        mode = 'streaming (~5-8GB)' if getattr(args, 'native_tf_streaming', True) else 'full load (~50GB)'
        print(f"  Native TF Gen: {mode}")
    print(f"  Continuation Mode: {project_config.CONTINUATION_MODE} "
          f"(horizon {project_config.ADAPTIVE_MIN_HORIZON}-{project_config.ADAPTIVE_MAX_HORIZON} bars)")
    print(f"  Model Capacity: internal_ratio={args.internal_neurons_ratio}, hidden_size={args.hidden_size}")
    print(f"  Multi-Task: {'Enabled' if args.multi_task else 'Disabled'}")

    # v5.0: Show architecture mode
    use_geo = getattr(args, 'use_geometric_base', True)
    use_fusion = getattr(args, 'use_fusion_head', True)

    if use_geo and not use_fusion:
        arch_mode = "Geometric + Physics-Only ⭐"
    elif use_geo and use_fusion:
        arch_mode = "Geometric + Fusion Head 🧪"
    elif not use_geo and use_fusion:
        arch_mode = "Learned + Fusion Head 📊"
    else:
        arch_mode = "Learned + Physics-Only"

    print(f"  Architecture: {arch_mode}")
    print(f"    Base: {'Geometric (channel projections)' if use_geo else 'Learned (neural approximation)'}")
    print(f"    Combine: {'Physics-based (weighted avg)' if not use_fusion else 'Fusion Head (neural net)'}")

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
    # Handle variable return: (features_df, continuation_labels_dir) or (features_df, continuation_labels_dir, mmap_meta_path)
    # v4.3: continuation_labels_dir is a Path to per-TF label files, NOT a DataFrame
    features_df = result[0]
    continuation_labels_dir = result[1]  # Path to directory with per-TF label files
    mmap_meta_path = result[2] if len(result) > 2 else None
    non_channel_cols = features_df.columns.tolist() if features_df is not None else None

    if profiler:
        nc_cols = len(non_channel_cols) if non_channel_cols is not None else 0
        cont_labels_info = str(continuation_labels_dir) if continuation_labels_dir else "None"
        profiler.log_info(f"FEATURES_EXTRACTED | non_channel_cols={nc_cols} | continuation_labels_dir={cont_labels_info}")
        profiler.snapshot("post_feature_extraction", 0, force_log=True)

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
        preload_to_ram=False,  # v5.2: Always false
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
            print(f"   Using ShuffleBufferSampler (buffer_size=10000, mmap-optimized)")

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

    # Progress bar (rank 0 only)
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", disable=not is_main_process(rank))

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

        for batch_idx, batch_data in enumerate(batch_pbar):
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
            targets = {k: v.to(args.device, non_blocking=True) for k, v in targets.items()}

            # v5.2: Move VIX batch to device if present
            if vix_batch is not None:
                vix_batch = vix_batch.to(args.device, non_blocking=True)

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
                    # v5.2: Pass VIX sequence and events to model
                    predictions, hidden_states = model(features, vix_sequence=vix_batch, events=events_batch)
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

                # v5.3: Confidence calibration (ALL samples in batch, every 10th batch)
                if 'layer_predictions' in hidden_states and batch_idx % 10 == 0:
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
                        loss = loss + 0.05 * (calib_loss_total / num_calibrated)

                # v5.2/v5.3 LOSSES: Duration, Validity, Transition (same as non-AMP path)
                transition_labels_dict = {}
                for tf in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
                    trans_type_key = f'trans_{tf}_type'
                    if trans_type_key in targets:
                        transition_labels_dict[tf] = {
                            'transition_type': int(targets[trans_type_key][0].item()),
                            'new_direction': int(targets.get(f'trans_{tf}_direction', torch.tensor([1]))[0].item()),
                        }

                # Duration NLL loss (check key exists)
                if 'duration' in hidden_states:
                    for tf, dur_data in hidden_states['duration'].items():
                        target_dur_key = f'cont_{tf}_duration'
                        if target_dur_key in targets:  # Only train if target exists
                            mean = dur_data['mean'].squeeze()
                            log_std = dur_data['log_std'].squeeze()
                            target_dur = targets[target_dur_key].squeeze()
                            variance = torch.exp(2 * log_std) + 1e-6
                            nll = 0.5 * ((target_dur - mean) ** 2 / variance + 2 * log_std)
                            loss = loss + 0.3 * nll.mean()

                if 'validity' in hidden_states:
                    for tf, validity_pred in hidden_states['validity'].items():
                        if tf in transition_labels_dict:
                            target_val = 1.0 if transition_labels_dict[tf]['transition_type'] == 0 else 0.0
                            loss = loss + 0.2 * F.binary_cross_entropy(
                                validity_pred.squeeze(),
                                torch.tensor([target_val], device=args.device).expand(validity_pred.shape[0])
                            )

                if 'compositor' in hidden_states and 'selected_tf' in hidden_states:
                    sel_tf = hidden_states['selected_tf']
                    if sel_tf in transition_labels_dict:
                        comp = hidden_states['compositor']
                        trans = torch.tensor([transition_labels_dict[sel_tf]['transition_type']], device=args.device, dtype=torch.long).expand(predictions.shape[0])
                        direc = torch.tensor([transition_labels_dict[sel_tf]['new_direction']], device=args.device, dtype=torch.long).expand(predictions.shape[0])
                        loss = loss + 0.3 * F.cross_entropy(comp['transition_logits'], trans)
                        loss = loss + 0.2 * F.cross_entropy(comp['direction_logits'], direc)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # v5.2: Pass VIX sequence and events to model
                predictions, hidden_states = model(features, vix_sequence=vix_batch, events=events_batch)
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

                # v5.3: Confidence calibration (ALL samples in batch, every 10th batch)
                if 'layer_predictions' in hidden_states and batch_idx % 10 == 0:
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
                        loss = loss + 0.05 * (calib_loss_total / num_calibrated)

                # =====================================================================
                # v5.2/v5.3 LOSSES: Duration, Validity, Transition, Direction
                # =====================================================================
                # Extract transition labels from targets (if available)
                transition_labels_dict = {}
                for i, tf in enumerate(['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']):
                    trans_type_key = f'trans_{tf}_type'
                    if trans_type_key in targets:
                        transition_labels_dict[tf] = {
                            'transition_type': int(targets[trans_type_key][0].item()),
                            'new_direction': int(targets.get(f'trans_{tf}_direction', torch.tensor([1]))[0].item()),
                        }

                # Duration NLL loss (check key exists before using)
                if 'duration' in hidden_states:
                    for tf, dur_data in hidden_states['duration'].items():
                        target_dur_key = f'cont_{tf}_duration'
                        if target_dur_key in targets:  # Only train if target exists
                            mean = dur_data['mean'].squeeze()
                            log_std = dur_data['log_std'].squeeze()
                            target_dur = targets[target_dur_key].squeeze()
                            variance = torch.exp(2 * log_std) + 1e-6
                            nll = 0.5 * ((target_dur - mean) ** 2 / variance + 2 * log_std)
                            loss = loss + 0.3 * nll.mean()

                # Validity loss
                if 'validity' in hidden_states and len(transition_labels_dict) > 0:
                    for tf, validity_pred in hidden_states['validity'].items():
                        if tf in transition_labels_dict:
                            trans_type = transition_labels_dict[tf]['transition_type']
                            target_validity = 1.0 if trans_type == 0 else 0.0
                            loss_validity = F.binary_cross_entropy(
                                validity_pred.squeeze(),
                                torch.tensor([target_validity], device=args.device).expand(validity_pred.shape[0])
                            )
                            loss = loss + 0.2 * loss_validity

                # Transition/Direction losses
                if 'compositor' in hidden_states and 'selected_tf' in hidden_states:
                    selected_tf = hidden_states['selected_tf']
                    if selected_tf in transition_labels_dict:
                        compositor = hidden_states['compositor']
                        trans_type = transition_labels_dict[selected_tf]['transition_type']
                        new_dir = transition_labels_dict[selected_tf]['new_direction']

                        target_trans = torch.tensor([trans_type], device=args.device, dtype=torch.long).expand(predictions.shape[0])
                        loss = loss + 0.3 * F.cross_entropy(compositor['transition_logits'], target_trans)

                        target_dir = torch.tensor([new_dir], device=args.device, dtype=torch.long).expand(predictions.shape[0])
                        loss = loss + 0.2 * F.cross_entropy(compositor['direction_logits'], target_dir)

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
                targets = {k: v.to(args.device, non_blocking=True) for k, v in targets.items()}

                # v5.2: Move VIX batch to device if present
                if vix_batch_val is not None:
                    vix_batch_val = vix_batch_val.to(args.device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        # v5.2: Pass VIX and events
                        predictions, hidden_states = model(features, vix_sequence=vix_batch_val, events=events_batch_val)
                        # Raw % (data alignment fixed in dataset)
                        target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                        loss = F.mse_loss(predictions[:, :2], target_tensor)
                else:
                    # v5.2: Pass VIX and events
                    predictions, hidden_states = model(features, vix_sequence=vix_batch_val, events=events_batch_val)
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
                    'input_sizes': model_to_save.input_sizes,  # v4.x: Dict[str, int] for native TF mode
                    'hidden_size': model_to_save.hidden_size,
                    'internal_neurons_ratio': model_to_save.internal_neurons_ratio,
                    # v5.1: Save architecture flags at top level for easier access
                    'use_fusion_head': getattr(args, 'use_fusion_head', False),
                    'use_geometric_base': getattr(args, 'use_geometric_base', True),
                    'multi_task': getattr(args, 'multi_task', True),
                    # v5.3.1: Save information flow (critical!)
                    'information_flow': getattr(args, 'information_flow', 'bottom_up'),
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
                for features, targets in tqdm(test_loader, desc="Test batches", leave=False):
                    # Move to device
                    if isinstance(features, dict):
                        features = {tf: f.to(args.device, non_blocking=True) for tf, f in features.items()}
                    else:
                        features = features.to(args.device, non_blocking=True)
                    targets = {k: v.to(args.device, non_blocking=True) for k, v in targets.items()}

                    # Forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            predictions, _ = model(features)
                            target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                            loss = F.mse_loss(predictions[:, :2], target_tensor)
                    else:
                        predictions, _ = model(features)
                        target_tensor = torch.stack([targets['high'], targets['low']], dim=1)
                        loss = F.mse_loss(predictions[:, :2], target_tensor)

                    test_loss += loss.item()
                    test_error += torch.abs(predictions[:, :2] - target_tensor).mean().item()
                    num_test_batches += 1

                    # Collect for detailed analysis
                    test_predictions_list.append(predictions[:, :2].cpu())
                    test_targets_list.append(target_tensor.cpu())

            avg_test_loss = test_loss / max(num_test_batches, 1)
            avg_test_error = test_error / max(num_test_batches, 1)

            # Combine all predictions for analysis
            all_test_preds = torch.cat(test_predictions_list, dim=0).numpy()
            all_test_targets = torch.cat(test_targets_list, dim=0).numpy()

            # Calculate metrics
            high_mae = np.abs(all_test_preds[:, 0] - all_test_targets[:, 0]).mean()
            low_mae = np.abs(all_test_preds[:, 1] - all_test_targets[:, 1]).mean()
            high_rmse = np.sqrt(np.mean((all_test_preds[:, 0] - all_test_targets[:, 0])**2))
            low_rmse = np.sqrt(np.mean((all_test_preds[:, 1] - all_test_targets[:, 1])**2))

            print(f"\n📊 Test Set Results:")
            print(f"   Test Loss (MSE): {avg_test_loss:.4f}")
            print(f"   Test MAE: {avg_test_error:.4f}%")
            print(f"   ")
            print(f"   High Predictions:")
            print(f"     MAE:  {high_mae:.4f}%")
            print(f"     RMSE: {high_rmse:.4f}%")
            print(f"   Low Predictions:")
            print(f"     MAE:  {low_mae:.4f}%")
            print(f"     RMSE: {low_rmse:.4f}%")
            print(f"   ")
            print(f"   Samples evaluated: {len(all_test_preds):,}")
            print(f"   ⚠️  This is your TRUE performance on unseen data")
            print("=" * 70)

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
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (CUDA only, 2-3x faster)')
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

    args = parser.parse_args()

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
