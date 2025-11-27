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
from typing import Dict, Tuple
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from src.ml.hierarchical_model import HierarchicalLNN
from src.ml.hierarchical_dataset import create_hierarchical_dataset
from src.ml.features import TradingFeatureExtractor, FEATURE_VERSION
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


def fix_ncps_buffers(model):
    """
    Register ncps sparsity_mask tensors as buffers for DataParallel compatibility.

    The ncps library's CfcCell has a sparsity_mask tensor that isn't registered as a
    proper PyTorch buffer. When DataParallel replicates the model to other GPUs,
    the sparsity_mask stays on GPU 0, causing device mismatch errors.

    This function iterates through all modules and registers any sparsity_mask
    tensors as buffers so they properly move with the model.
    """
    fixed_count = 0
    for name, module in model.named_modules():
        # Check for ncps CfC cells that have sparsity_mask
        if hasattr(module, 'sparsity_mask') and isinstance(module.sparsity_mask, torch.Tensor):
            mask_tensor = module.sparsity_mask

            # If it's a Parameter (as in ncps library), extract the underlying tensor
            if isinstance(mask_tensor, nn.Parameter):
                mask_tensor = mask_tensor.data

            # Delete the old attribute (Parameter or tensor) to avoid conflicts
            delattr(module, 'sparsity_mask')

            # Register as a buffer (properly replicated by DataParallel)
            module.register_buffer('sparsity_mask', mask_tensor)
            fixed_count += 1
    if fixed_count > 0:
        print(f"   🔧 Fixed {fixed_count} ncps sparsity_mask tensors for multi-GPU compatibility")
    return fixed_count


def hierarchical_collate(batch, device: str = None, move_to_device: bool = False, torch_dtype=None, _debug_counter=[0]):
    """
    Memory-efficient collate: pre-allocate final tensor and fill directly.
    Eliminates redundant intermediate tensors from stack+cat pattern.
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

    batch_size = len(batch)
    targets = []

    # Parse first sample to determine dimensions
    first_data, first_tgt = batch[0]
    if isinstance(first_data, tuple) and len(first_data) == 3:
        first_ch_main, first_ch_monthly, first_nc = first_data
    elif isinstance(first_data, tuple) and len(first_data) == 2:
        first_ch_main, first_nc = first_data
        first_ch_monthly = None
    else:
        first_ch_main, first_ch_monthly, first_nc = first_data, None, None

    seq_len = first_ch_main.shape[0]
    ch_main_cols = first_ch_main.shape[1]
    ch_monthly_cols = first_ch_monthly.shape[1] if first_ch_monthly is not None else 0
    nc_cols = first_nc.shape[1] if first_nc is not None else 0
    total_features = ch_main_cols + ch_monthly_cols + nc_cols

    # Pre-allocate final tensor (single allocation instead of stack+cat)
    x = torch.zeros((batch_size, seq_len, total_features), dtype=torch_dtype)

    # Fill tensor directly without intermediate allocations
    # Track contiguous copies for debugging memory duplication
    copies_made = 0
    copy_bytes = 0

    for i, (data, tgt) in enumerate(batch):
        # Parse data tuple
        if isinstance(data, tuple) and len(data) == 3:
            ch_main, ch_monthly, nc = data
        elif isinstance(data, tuple) and len(data) == 2:
            ch_main, nc = data
            ch_monthly = None
        else:
            ch_main, ch_monthly, nc = data, None, None

        # Copy channel features directly into pre-allocated tensor
        col_start = 0
        if not ch_main.flags['C_CONTIGUOUS']:
            copies_made += 1
            copy_bytes += ch_main.nbytes
            ch_main = np.ascontiguousarray(ch_main)
        x[i, :, col_start:col_start + ch_main_cols] = torch.from_numpy(ch_main)
        col_start += ch_main_cols

        # Copy monthly features if present
        if ch_monthly is not None:
            if not ch_monthly.flags['C_CONTIGUOUS']:
                copies_made += 1
                copy_bytes += ch_monthly.nbytes
                ch_monthly = np.ascontiguousarray(ch_monthly)
            x[i, :, col_start:col_start + ch_monthly_cols] = torch.from_numpy(ch_monthly)
        col_start += ch_monthly_cols

        # Copy non-channel features if present
        if nc is not None:
            if not nc.flags['C_CONTIGUOUS']:
                copies_made += 1
                copy_bytes += nc.nbytes
                nc = np.ascontiguousarray(nc)
            x[i, :, col_start:col_start + nc_cols] = torch.from_numpy(nc)

        targets.append(tgt)

    # Log copy detection for first few batches
    if _debug_counter[0] <= 5 and copies_made > 0:
        print(f"[DEBUG] collate made {copies_made} contiguous copies ({copy_bytes/1e6:.1f} MB) in batch #{_debug_counter[0]}", file=sys.stderr, flush=True)

    # Build targets tensor dict with proper dtype
    converted_targets = []
    for tgt in targets:
        ct = {}
        for k, v in tgt.items():
            if isinstance(v, torch.Tensor):
                ct[k] = v.to(dtype=torch_dtype)
            else:
                ct[k] = torch.tensor(v, dtype=torch_dtype)
        converted_targets.append(ct)

    targets_batch = default_collate(converted_targets)

    if move_to_device and device is not None:
        x = x.to(device, non_blocking=True)
        for k, v in targets_batch.items():
            targets_batch[k] = v.to(device, non_blocking=True)

    # Log slow batch assembly (diagnose lazy loading bottlenecks)
    _collate_elapsed = time.perf_counter() - _collate_start
    if _collate_elapsed > 1.0:  # Log if >1 second
        print(f"[SLOW_COLLATE] batch assembly took {_collate_elapsed:.1f}s for {batch_size} samples ({_collate_elapsed/batch_size*1000:.0f}ms/sample)", file=sys.stderr, flush=True)

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
    """Find available cache triplets (meta + continuation labels + non-channel) in a directory."""
    cache_dir = Path(cache_dir)
    caches = []
    for meta_path in cache_dir.glob("features_mmap_meta_*.json"):
        cache_key = meta_path.name.replace("features_mmap_meta_", "").replace(".json", "")
        mode_suffixes = ['adaptive', 'simple']
        cont_path = None
        for suffix in mode_suffixes:
            candidate = cache_dir / f"continuation_labels_{cache_key}_{suffix}.pkl"
            if candidate.exists():
                cont_path = candidate
                break

        # Check for non-channel features cache (new!)
        non_channel_path = cache_dir / f"non_channel_features_{cache_key}.pkl"
        has_non_channel = non_channel_path.exists()

        caches.append({
            "cache_key": cache_key,
            "meta_path": str(meta_path),
            "cont_path": str(cont_path) if cont_path else None,
            "non_channel_path": str(non_channel_path) if has_non_channel else None,
            "complete": cont_path is not None and has_non_channel  # True = skip extraction entirely
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
        return caches[0]

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
    mmap_meta_path: str,
    continuation_path: str,
    non_channel_path: str,  # NEW: path to non-channel features cache
    args,
    df: pd.DataFrame
):
    """Persist a manifest alongside caches for reuse selection."""
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / f"cache_manifest_{cache_key}.json"

        manifest = {
            "cache_key": cache_key,
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
                "non_channel_features": non_channel_path  # NEW
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
                "downsample_fast_to_medium": args.downsample_fast_to_medium,
                "downsample_medium_to_slow": args.downsample_medium_to_slow,
                "num_workers": args.num_workers,
                "preload": getattr(args, "preload", False),
                "output": getattr(args, "output", None),
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
            'adaptive_confidence': 0.2
        }

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

        # Container RAM detection for cloud GPU instances (RunPod, Lambda, etc.)
        print()
        try:
            import psutil
            detected_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            detected_ram = 0

        if detected_ram > 200:  # Likely seeing host RAM, not container
            print(f"   ⚠️  Detected {detected_ram:.0f}GB RAM - likely cloud container (seeing host RAM)")
            if profiler:
                profiler.log_info(f"CONTAINER_DETECTED | psutil_ram={detected_ram:.0f}GB")
            args.container_ram_gb = int(inquirer.number(
                message="Actual container RAM (GB):",
                default=46,
                min_allowed=8,
                max_allowed=256
            ).execute())
            print(f"   ℹ️  Using {args.container_ram_gb}GB for memory calculations")

            # Set environment variables for other modules to use
            import os
            os.environ['CONTAINER_RAM_GB'] = str(args.container_ram_gb)
            # Only set PREMERGE_LIMIT_GB if not already specified by user
            if not os.environ.get('PREMERGE_LIMIT_GB'):
                os.environ['PREMERGE_LIMIT_GB'] = str(max(6, args.container_ram_gb // 3))  # ~1/3 of RAM for pre-merge
            if profiler:
                actual_premerge = os.environ.get('PREMERGE_LIMIT_GB', str(max(6, args.container_ram_gb // 3)))
                profiler.log_info(f"CONTAINER_RAM_SET | user_specified={args.container_ram_gb}GB | premerge_limit={actual_premerge}GB")
        else:
            args.container_ram_gb = 0  # Use psutil detection
            if profiler:
                profiler.log_info(f"NATIVE_RAM | detected={detected_ram:.0f}GB | container_mode=False")

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

    # Data loading workers (RIGHT after device selection)
    print()
    default_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}.get(args.device, 2)

    args.num_workers = int(inquirer.number(
        message=f"Data loading workers (CPU threads for batch prep, recommended: {default_workers}):",
        default=dflt('num_workers', default_workers),
        min_allowed=0,
        max_allowed=128  # High limit for systems with many cores (e.g., 32-64 core servers)
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
        print(f"   Warmup required: {warmup_years} years (257,400 bars for 21-window system)")
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
    import multiprocessing as mp
    n_cores = mp.cpu_count()

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

    # Device-specific max batch sizes
    # Note: 512 may cause OOM on systems with <16GB RAM/VRAM
    max_batch_sizes = {
        'cuda': 512,    # Increased max for modern GPUs
        'mps': 512,     # Increased max for M2 Max/Ultra with high RAM
        'cpu': 512      # Increased max (requires 32GB+ RAM)
    }
    max_batch_size = max_batch_sizes.get(args.device, 512)

    args.batch_size = int(inquirer.number(
        message=f"Batch size (recommended: {recommended_batch}, max for {args.device.upper()}: {max_batch_size}):",
        default=dflt('batch_size', recommended_batch),
        min_allowed=1,  # Allow very small batches for MPS (1-4 for memory constrained)
        max_allowed=max_batch_size
    ).execute())

    args.lr = float(inquirer.number(
        message="Learning rate:",
        default=dflt('lr', 0.001),
        min_allowed=0.00001,
        max_allowed=0.01,
        float_allowed=True
    ).execute())

    # Data loading
    print()
    preload_choice = inquirer.select(
        message="Data loading mode:",
        choices=[
            Choice(value=False, name=f'Lazy loading (2-3 GB RAM) - Recommended'),
            Choice(value=True, name=f'Preload (requires ~40 GB RAM) - 20% faster')
        ],
        default=dflt('preload', False)
    ).execute()
    args.preload = preload_choice

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
    print(f"  Data Loading: {'Preload' if args.preload else 'Lazy'}")
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


def main():
    # Fix for Unix + torch + multiprocessing: use forkserver to avoid torch cleanup deadlock
    # spawn causes workers to hang on exit when torch is imported
    # forkserver is safe and faster (no torch init in workers with lazy loading)
    import multiprocessing as mp
    import sys
    try:
        mp.set_start_method('forkserver', force=True)
        platform_note = "macOS" if sys.platform == "darwin" else "Linux"
        print(f"✓ Using forkserver multiprocessing (safer for torch on {platform_note})")
    except ValueError:
        pass  # Already set, or not available on this platform

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

    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device: auto (detect), cuda (NVIDIA), mps (Apple Silicon), cpu')
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

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode with menus')

    # Memory profiling
    parser.add_argument('--memory-profile', dest='memory_profile', action='store_true', default=False,
                        help='Enable memory profiling to logs/memory_debug.log')

    args = parser.parse_args()

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
        profiler.log_info(f"PREMERGE_LIMIT_GB={os.environ.get('PREMERGE_LIMIT_GB', 'not_set')}")

    # Interactive mode overrides command-line args
    if args.interactive:
        args = interactive_setup(args, profiler=profiler)

    # FIX: Create profiler if user enabled it in interactive mode (wasn't created earlier)
    if args.memory_profile and profiler is None:
        import os
        from src.ml.memory_profiler import MemoryProfiler
        profiler = MemoryProfiler(
            log_path="logs/memory_debug.log",
            device=args.device if args.device != 'auto' else 'unknown',
            log_every_n=10,
            spike_threshold_mb=500
        )
        profiler.log_info(f"CONTAINER_RAM_GB={os.environ.get('CONTAINER_RAM_GB', 'not_set')}")
        profiler.log_info(f"PREMERGE_LIMIT_GB={os.environ.get('PREMERGE_LIMIT_GB', 'not_set')}")
        profiler.log_info("PROFILER_CREATED_POST_INTERACTIVE")

    # Auto-detect device if 'auto'
    if args.device == 'auto':
        args.device = get_best_device()
        print(f"🔍 Auto-detected device: {args.device}")

    # Validate device
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
        project_config._TORCH_DTYPE = torch.float32  # Reset cached value
        print("   ✓ Switched to float32 for MPS compatibility")

    # Auto-set num_workers if not specified (MPS needs 0 for memory efficiency)
    if args.num_workers is None:
        args.num_workers = {'cuda': 4, 'mps': 0, 'cpu': 2}.get(args.device, 2)

    # Override parallel worker count for feature extraction if specified
    if args.feature_workers is not None:
        # project_config already imported at module level
        project_config.MAX_PARALLEL_WORKERS = args.feature_workers
        print(f"✓ Feature extraction workers: {args.feature_workers} cores (via --feature_workers)")

    # Auto-detect chunking if not specified
    if args.use_chunking is None:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9
        args.use_chunking = (total_ram_gb < 64)  # Enable if <64GB RAM
        print(f"✓ Auto-detected chunking: {'Enabled' if args.use_chunking else 'Disabled'} (RAM: {total_ram_gb:.1f}GB)")

    # Load configuration
    config = None
    loss_weights = None
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            loss_weights = config.get('loss_weights', None)
        print(f"✅ Loaded config from: {args.config}")
    else:
        print(f"⚠️ Config not found: {args.config}, using defaults")

    # Hardware info
    hw_info = get_hardware_info()

    print("\n" + "=" * 70)
    print("🎯 HIERARCHICAL LNN TRAINING")
    print("=" * 70)
    print(f"📱 Device: {args.device.upper()}")
    if args.device == 'cuda':
        print(f"   GPU: {hw_info.get('cuda_device', 'Unknown')}")
        print(f"   VRAM: {hw_info.get('cuda_memory_gb', 0):.1f} GB")
    elif args.device == 'mps':
        print(f"   Chip: {hw_info.get('mac_chip', 'Apple Silicon')}")
        print(f"   RAM: {hw_info.get('total_ram_gb', 0):.0f} GB")
    print(f"📅 Training: {args.train_start_year}-{args.train_end_year}")
    print(f"📊 Sequence: {args.sequence_length} bars ({args.sequence_length} minutes)")
    print(f"🎯 Horizon: Adaptive (base {args.prediction_horizon} bars, model adjusts dynamically)")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"💾 Data mode: {'Preload' if args.preload else 'Lazy'}")
    print(f"🎭 Multi-task: {'Enabled' if args.multi_task else 'Disabled'}")
    print("=" * 70)

    # Load data with historical buffer for continuation analysis
    print("\n1. Loading 1-min data...")
    if profiler:
        profiler.snapshot("pre_data_load", 0, force_log=True)

    data_feed = CSVDataFeed(timeframe=args.input_timeframe)

    # Load with 2-year historical buffer for continuation analysis
    # This ensures timestamps have sufficient lookback history
    historical_buffer_years = 2
    load_start_year = max(2010, args.train_start_year - historical_buffer_years)  # Don't go before 2010

    df = data_feed.load_aligned_data(
        start_date=f'{load_start_year}-01-01',
        end_date=f'{args.train_end_year}-12-31'
    )

    data_years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"   Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    if profiler:
        profiler.log_info(f"DATA_LOADED | rows={len(df)} | cols={len(df.columns)}")
        profiler.snapshot("post_data_load", 0, force_log=True)
    print(f"   Data range: {data_years:.1f} years")
    print(f"   Historical buffer: {historical_buffer_years} years (for continuation analysis)")

    # Validate minimum data requirement (for 21-window system with 3-month TF)
    min_required = project_config.MIN_DATA_YEARS if hasattr(project_config, 'MIN_DATA_YEARS') else 2.5
    if data_years < min_required:
        print(f"\n   ⚠️  WARNING: Insufficient data!")
        print(f"   You have: {data_years:.1f} years")
        print(f"   Recommended: {min_required}+ years (for 3-month TF with 10-bar lookback)")
        print(f"   Long timeframes will have features with insufficient_data=1.0")

        if args.interactive:
            response = input("\n   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("   Exiting...")
                sys.exit(0)
        else:
            print("   Continuing in non-interactive mode...")
    else:
        print(f"   ✓ Data requirement met ({data_years:.1f} years >= {min_required} required)")

    # Slice data to user's selected training range (after loading historical buffer)
    training_start = pd.to_datetime(f'{args.train_start_year}-01-01')
    training_end = pd.to_datetime(f'{args.train_end_year}-12-31')

    # Keep full dataset for feature extraction (needs historical context)
    # But create sliced version for timestamps used in continuation analysis
    df_sliced = df[(df.index >= training_start) & (df.index <= training_end)].copy()
    print(f"   Training slice: {len(df_sliced)} bars ({df_sliced.index[0]} to {df_sliced.index[-1]})")

    # Smart lookback buffer system: Check if warmup period adjustment is needed
    if project_config.SKIP_WARMUP_PERIOD:
        # Calculate where first training timestamp sits in full dataset
        first_training_idx = df.index.get_loc(df_sliced.index[0])

        # Check if we have enough historical data before first training timestamp
        if first_training_idx < project_config.MIN_LOOKBACK_BARS:
            # Need to skip warmup period
            warmup_end_idx = project_config.MIN_LOOKBACK_BARS
            effective_start = df.index[warmup_end_idx]

            print(f"\n   ⚠️  Adjusting training range for data quality:")
            print(f"      Requested start: {df_sliced.index[0]}")
            print(f"      Data file starts: {df.index[0]}")
            print(f"      Minimum lookback required: {project_config.MIN_LOOKBACK_BARS} bars (~{project_config.MIN_LOOKBACK_MONTHS} months)")
            print(f"      Effective training start: {effective_start}")

            # Skip the warmup period
            original_length = len(df_sliced)
            df_sliced = df_sliced[df_sliced.index >= effective_start].copy()
            warmup_skipped = original_length - len(df_sliced)

            print(f"      Skipped {warmup_skipped} warmup bars ({warmup_skipped/390:.1f} trading days)")
            print(f"      Remaining training data: {len(df_sliced)} bars ({df_sliced.index[0]} to {df_sliced.index[-1]})")
            print(f"      ✓ All training samples will have complete {project_config.MIN_LOOKBACK_MONTHS}-month feature history")
        else:
            print(f"   ✓ Sufficient historical data available ({first_training_idx} bars before training start)")

    # Extract features and continuation labels
    print("\n2. Extracting features and continuation labels...")
    if profiler:
        profiler.snapshot("pre_feature_extraction", 0, force_log=True)
    extractor = TradingFeatureExtractor()

    # Validate data availability first
    print("  Pre-validating continuation data...")
    timestamps = df_sliced.index.tolist()
    validation = extractor.validate_continuation_data_availability(df, timestamps)

    if validation['sufficient_raw_data'] == 0:
        print("  ⚠️  WARNING: No timestamps have sufficient data for continuation analysis!")
        print("     This will result in 0 continuation labels.")
        print("     Consider using a dataset with more historical data.")
        print("     Enabling debug mode for detailed analysis...")
        debug_mode = True
    else:
        sufficient_pct = (validation['sufficient_raw_data'] / validation['total_timestamps']) * 100
        print(f"  📊 {sufficient_pct:.1f}% of timestamps have sufficient continuation data")
        debug_mode = False

    # Use cache unless regenerate_cache flag is set (from interactive menu)
    use_cache = not getattr(args, 'regenerate_cache', False)

    # Use GPU if enabled (from interactive menu or auto-detect)
    # If --device cpu was specified, force CPU for features too
    if hasattr(args, 'device') and args.device == 'cpu':
        use_gpu = False
    else:
        use_gpu = args.use_gpu_features if hasattr(args, 'use_gpu_features') else 'auto'


    # Set parallel processing option in config if specified
    if hasattr(args, 'use_parallel'):
        import config
        config.PARALLEL_CHANNEL_CALC = args.use_parallel
    else:
        # If not in interactive mode and parallel isn't set, enable it by default for CPU
        if use_gpu == False:  # Only enable parallel if we're definitely using CPU
            import multiprocessing as mp
            if mp.cpu_count() > 2:
                import config
                args.use_parallel = True
                config.PARALLEL_CHANNEL_CALC = True
                print("   🚀 Auto-enabling parallel processing for CPU mode")

    result = extractor.extract_features(
        df,
        use_cache=use_cache,
        use_gpu=use_gpu,
        continuation=True,
        continuation_mode=project_config.CONTINUATION_MODE,
        use_chunking=getattr(args, 'use_chunking', False),
        chunk_size_years=project_config.CHUNK_SIZE_YEARS,
        shard_storage_path=getattr(args, 'shard_path', None)
    )

    # Handle both normal (2-tuple) and mmap (3-tuple) return formats
    if len(result) == 3:
        features_df, continuation_df, mmap_meta_path = result
        print(f"   ℹ️  Using memory-mapped channel features from: {Path(mmap_meta_path).name}")
    else:
        features_df, continuation_df = result
        mmap_meta_path = None

    # If we got 0 labels and didn't enable debug, enable it now for diagnosis
    if continuation_df is not None and len(continuation_df) == 0 and not debug_mode:
        print("  ⚠️  Got 0 continuation labels. Re-running with debug mode enabled...")
        debug_mode = True
        continuation_df = extractor.generate_continuation_labels(df, timestamps, prediction_horizon=24, mode=project_config.CONTINUATION_MODE, debug=debug_mode)

    if profiler:
        profiler.log_info(f"FEATURES_EXTRACTED | non_channel_cols={len(features_df.columns)} | continuation_rows={len(continuation_df) if continuation_df is not None else 0}")
        profiler.snapshot("post_feature_extraction", 0, force_log=True)

    # Display feature counts (clarify mmap vs non-mmap)
    total_feature_dim = extractor.get_feature_dim()
    print(f"   Total feature dimension: {total_feature_dim} (model input size)")
    if mmap_meta_path:
        channel_features = total_feature_dim - len(features_df.columns)
        print(f"     ├─ Channel features (mmaps): {channel_features}")
        print(f"     └─ Non-channel features (df): {len(features_df.columns)}")
    else:
        print(f"     └─ All features in dataframe: {len(features_df.columns)}")
    print(f"   Generated {len(continuation_df) if continuation_df is not None else 0} continuation labels")
    print(f"   Feature names (first 5): {extractor.get_feature_names()[:5]}...")

    # Save cache manifest for future reuse selection
    cache_dir = getattr(extractor, "_unified_cache_dir", getattr(args, "shard_path", "data/feature_cache"))
    cache_key = getattr(extractor, "_cache_key", None)
    cont_path = getattr(extractor, "_cont_cache_path", None)
    non_channel_path = getattr(extractor, "_non_channel_cache_path", None)  # NEW
    if cache_key:
        save_cache_manifest(
            cache_dir=Path(cache_dir),
            cache_key=cache_key,
            mmap_meta_path=mmap_meta_path,
            continuation_path=str(cont_path) if cont_path else None,
            non_channel_path=str(non_channel_path) if non_channel_path else None,  # NEW
            args=args,
            df=df
        )

    # Create datasets
    print("\n3. Creating datasets...")
    if profiler:
        profiler.snapshot("pre_dataset_create", 0, force_log=True)

    train_dataset, val_dataset = create_hierarchical_dataset(
        features_df,
        raw_ohlc_df=df,
        continuation_labels_df=continuation_df,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        mode='uniform_bars',
        preload=args.preload,
        validation_split=args.val_split,
        include_continuation=True,
        mmap_meta_path=mmap_meta_path,  # Pass mmap metadata if using sharded storage
        profiler=profiler  # Pass profiler for granular RAM logging
    )

    if profiler:
        profiler.log_info(f"DATASET_CREATED | train_samples={len(train_dataset)} | val_samples={len(val_dataset) if val_dataset else 0}")
        profiler.snapshot("post_dataset_create", 0, force_log=True)

    print(f"   Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"   Val samples: {len(val_dataset)}")

    # Free large DataFrames - dataset has extracted what it needs
    # This is CRITICAL for memory on systems where psutil misreads container RAM
    import gc
    df_mem_mb = df.memory_usage(deep=True).sum() / 1e6 if hasattr(df, 'memory_usage') else 0
    features_mem_mb = features_df.memory_usage(deep=True).sum() / 1e6 if hasattr(features_df, 'memory_usage') else 0
    cont_mem_mb = continuation_df.memory_usage(deep=True).sum() / 1e6 if continuation_df is not None and hasattr(continuation_df, 'memory_usage') else 0
    total_freed_mb = df_mem_mb + features_mem_mb + cont_mem_mb
    print(f"   🧹 Freeing DataFrames: df={df_mem_mb:.0f}MB, features={features_mem_mb:.0f}MB, continuation={cont_mem_mb:.0f}MB")
    if profiler:
        profiler.log_info(f"DATAFRAME_CLEANUP | df={df_mem_mb:.0f}MB | features={features_mem_mb:.0f}MB | continuation={cont_mem_mb:.0f}MB | total={total_freed_mb:.0f}MB")
        profiler.snapshot("pre_dataframe_cleanup", 0, force_log=True)
    del df, features_df, continuation_df
    gc.collect()
    if profiler:
        profiler.snapshot("post_dataframe_cleanup", 0, force_log=True)

    # Verify feature dimension matches between extractor and dataset
    print("\n   🔍 Verifying feature dimensions...")
    sample_x, sample_y = train_dataset[0]
    if isinstance(sample_x, tuple):
        dims = 0
        for part in sample_x:
            if part is not None and hasattr(part, 'shape'):
                dims += part.shape[-1]
        actual_input_dim = dims if dims > 0 else sample_x.shape[-1]
    else:
        actual_input_dim = sample_x.shape[-1]
    expected_dim = extractor.get_feature_dim()

    if actual_input_dim != expected_dim:
        # Enhanced error with feature name comparison
        expected_names = set(extractor.get_feature_names())

        # Try to get actual feature names from dataset
        if mmap_meta_path:
            # With mmaps, combine channel + non-channel names
            actual_names = set(features_df.columns.tolist())  # Non-channel only
            print(f"\n   ⚠️  Note: With mmaps, only showing non-channel feature names for comparison")
        else:
            actual_names = set(features_df.columns.tolist())

        missing = expected_names - actual_names
        extra = actual_names - expected_names

        error_msg = f"\n❌ FATAL: Feature dimension mismatch!\n"
        error_msg += f"   Model expects: {expected_dim} features\n"
        error_msg += f"   Dataset returns: {actual_input_dim} features\n"
        error_msg += f"   Difference: {abs(actual_input_dim - expected_dim)}\n\n"

        if missing and len(missing) < 100:
            error_msg += f"   Missing from dataset ({len(missing)} features):\n"
            error_msg += f"      {sorted(list(missing))[:10]}...\n\n"
        elif missing:
            error_msg += f"   Missing from dataset: {len(missing)} features (too many to list)\n\n"

        if extra and len(extra) < 100:
            error_msg += f"   Extra in dataset ({len(extra)} features):\n"
            error_msg += f"      {sorted(list(extra))[:10]}...\n\n"
        elif extra:
            error_msg += f"   Extra in dataset: {len(extra)} features (likely old cache)\n\n"

        error_msg += f"   Likely cause: Using old v3.13 shards (12,474 features) vs new v3.17 code (14,322 features)\n"
        error_msg += f"   Fix: Delete old shards OR bump FEATURE_VERSION caused regeneration"

        raise RuntimeError(error_msg)
    print(f"   ✅ Dimension check passed: {actual_input_dim} features match expected {expected_dim}")

    # DataLoader tuning (MPS-friendly defaults)
    prefetch_factor = 1 if args.num_workers and args.num_workers > 0 else None  # Reduced from 2 to prevent memory buildup
    mp_context = 'forkserver' if platform.system() == 'Darwin' and args.num_workers and args.num_workers > 0 else None

    # Create dataloaders
    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        persistent_workers=(args.num_workers > 0),  # Prevent worker respawn overhead
        collate_fn=functools.partial(
            hierarchical_collate,
            device=args.device,
            move_to_device=False,
            torch_dtype=project_config._TORCH_DTYPE
        )
    )
    if prefetch_factor is not None:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor
    if mp_context is not None:
        train_loader_kwargs['multiprocessing_context'] = mp_context

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = None
    if val_dataset:
        val_loader_kwargs = dict(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == 'cuda'),
            persistent_workers=(args.num_workers > 0),  # Prevent worker respawn overhead
            collate_fn=functools.partial(
                hierarchical_collate,
                device=args.device,
                move_to_device=False,
                torch_dtype=project_config._TORCH_DTYPE
            )
        )
        if prefetch_factor is not None:
            val_loader_kwargs['prefetch_factor'] = prefetch_factor
        if mp_context is not None:
            val_loader_kwargs['multiprocessing_context'] = mp_context

        val_loader = DataLoader(**val_loader_kwargs)

    # Create model
    print("\n4. Creating HierarchicalLNN model...")
    if profiler:
        profiler.snapshot("pre_model_create", 0, force_log=True)

    total_neurons = int(args.hidden_size * args.internal_neurons_ratio)
    print(f"   Capacity: {total_neurons} total neurons, {args.hidden_size} output neurons")
    print(f"   Internal processing neurons: {total_neurons - args.hidden_size}")

    model = HierarchicalLNN(
        input_size=extractor.get_feature_dim(),
        hidden_size=args.hidden_size,
        internal_neurons_ratio=args.internal_neurons_ratio,
        device=args.device,
        downsample_fast_to_medium=args.downsample_fast_to_medium,
        downsample_medium_to_slow=args.downsample_medium_to_slow,
        multi_task=args.multi_task
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    if profiler:
        profiler.log_info(f"MODEL_CREATED | params={num_params:,} | device={args.device}")
        profiler.snapshot("post_model_create", 0, force_log=True)
    print(f"   Multi-task heads: {'Enabled' if args.multi_task else 'Disabled'}")
    print(f"   Input features: {extractor.get_feature_dim()}")

    # Multi-GPU DataParallel wrapping (before torch.compile and optimizer)
    use_multi_gpu = False
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"\n   🚀 Multi-GPU Training: Using {num_gpus} GPUs with DataParallel")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"      GPU {i}: {name} ({vram:.0f} GB)")

        # Fix ncps library buffers for multi-GPU compatibility
        # The ncps CfcCell has sparsity_mask tensors that aren't registered as buffers
        fix_ncps_buffers(model)

        model = nn.DataParallel(model)
        use_multi_gpu = True
        print(f"   📦 Effective batch size: {args.batch_size} (split across {num_gpus} GPUs, ~{args.batch_size // num_gpus} per GPU)")
    else:
        print(f"\n   📱 Single-Device Training: {args.device.upper()}")

    # Apply torch.compile() for CUDA (PyTorch 2.0+ optimization, 10-40% speedup)
    # Note: First epoch is slower due to compilation warmup, benefits show from epoch 2+
    # Note: torch.compile works with DataParallel but may need mode adjustment
    if args.device == 'cuda' and hasattr(torch, 'compile') and not use_multi_gpu:
        # Skip torch.compile for multi-GPU (can cause issues with DataParallel)
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("   🔥 torch.compile() enabled - JIT compilation for faster training")
        except Exception as e:
            print(f"   ⚠️  torch.compile() failed ({e}), continuing without compilation")
    elif use_multi_gpu:
        print("   ℹ️  torch.compile() skipped for multi-GPU (DataParallel handles optimization)")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if profiler:
        gpu_mem_mb = 0
        if args.device == 'cuda':
            gpu_mem_mb = torch.cuda.memory_allocated() / 1e6
        profiler.log_info(f"MODEL_READY | multi_gpu={use_multi_gpu} | gpu_mem_mb={gpu_mem_mb:.0f}")
        profiler.snapshot("post_model_ready", 0, force_log=True)

    # Setup AMP (Automatic Mixed Precision) if enabled
    scaler = None
    if getattr(args, 'amp', False) and args.device == 'cuda':
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')
        print("   ⚡ Mixed Precision (AMP) enabled - using FP16 tensor cores")
    elif getattr(args, 'amp', False) and args.device != 'cuda':
        print("   ⚠️  AMP requested but only supported on CUDA - using FP32")

    # Note: Memory profiler was created early (after arg parsing) to capture diagnostics
    if profiler:
        profiler.log_info(f"TRAINING_START | device={args.device} | batch_size={args.batch_size} | num_workers={args.num_workers}")
        profiler.snapshot("pre_training_loop", 0, force_log=True)

    # Training loop
    print("\n5. Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_errors = []

    # Outer progress bar for overall training
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", ncols=80, position=0, ascii=True)

    for epoch in epoch_pbar:
        tqdm.write(f"\nEpoch {epoch + 1}/{args.epochs}")
        tqdm.write("-" * 70)

        # Set epoch in profiler
        if profiler:
            profiler.set_epoch(epoch + 1)
            profiler.snapshot("pre_train_epoch", 0, force_log=True)

        # Train (with loss_weights for multi-task, and optional AMP scaler)
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch, loss_weights, scaler, use_multi_gpu, profiler
        )
        train_losses.append(train_loss)

        tqdm.write(f"  Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader:
            val_loss, val_error = validate(model, val_loader, criterion, args.device)
            val_losses.append(val_loss)
            val_errors.append(val_error)

            tqdm.write(f"  Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                tqdm.write(f"  ✓ New best model (val_loss: {val_loss:.4f})")

                metadata = {
                    'model_type': 'HierarchicalLNN',
                    'input_size': extractor.get_feature_dim(),
                    'hidden_size': args.hidden_size,
                    'input_timeframe': args.input_timeframe,
                    'sequence_length': args.sequence_length,
                    'prediction_horizon': args.prediction_horizon,
                    'prediction_mode': 'uniform_bars',
                    'train_start_year': args.train_start_year,
                    'train_end_year': args.train_end_year,
                    'feature_names': extractor.get_feature_names(),
                    'device_type': args.device,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_error': val_error,
                    'downsample_fast_to_medium': args.downsample_fast_to_medium,
                    'downsample_medium_to_slow': args.downsample_medium_to_slow,
                    'timestamp': datetime.now().isoformat()
                }

                # Handle DataParallel: save underlying model (without module. prefix)
                model_to_save = model.module if use_multi_gpu else model
                model_to_save.save_checkpoint(args.output, metadata)
            else:
                patience_counter += 1
                tqdm.write(f"  Patience: {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    tqdm.write(f"\n  Early stopping triggered!")
                    break

    # Training complete
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")

    # Print memory profile summary if enabled
    if profiler:
        profiler.print_summary()
        profiler.close()

    # Save training history
    history_path = Path(args.output).parent / 'hierarchical_training_history.json'
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_errors': val_errors,
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'args': vars(args)
    }

    # Add memory profile to history if profiling was enabled
    if profiler:
        history['memory_profile'] = profiler.get_summary()

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
