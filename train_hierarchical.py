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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
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
from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed
import yaml
import config as project_config


def get_hardware_info():
    """Detect available compute devices and hardware specs."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'cpu_count': torch.get_num_threads(),
        'platform': platform.system()
    }

    if info['cuda_available']:
        info['cuda_device'] = torch.cuda.get_device_name()
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

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
    """Get recommended batch size for device."""
    recommendations = {
        'cuda': 128,  # NVIDIA GPU
        'mps_high': 96,  # M2 Max/Ultra with 64+ GB
        'mps_mid': 64,   # M2 Pro/M1 Max with 32-64 GB
        'mps_low': 32,   # M1/M1 Pro with 16-32 GB
        'cpu': 32
    }

    if device == 'mps':
        if total_ram_gb >= 64:
            return recommendations['mps_high']
        elif total_ram_gb >= 32:
            return recommendations['mps_mid']
        else:
            return recommendations['mps_low']

    return recommendations.get(device, 32)


def train_epoch(
    model: HierarchicalLNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    loss_weights: Dict = None
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

    # Progress bar for batches
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1} [Train]", leave=False, ncols=80)

    for batch_idx, (x, targets_dict) in enumerate(pbar):
        # Move to device
        x = x.to(device)

        # Move all targets to device
        target_high = targets_dict['high'].to(device)
        target_low = targets_dict['low'].to(device)

        if model.multi_task:
            target_hit_band = targets_dict['hit_band'].to(device)
            target_hit_target = targets_dict['hit_target'].to(device)
            target_expected_return = targets_dict['expected_return'].to(device)
            target_overshoot = targets_dict['overshoot'].to(device)
            target_continuation_duration = targets_dict['continuation_duration'].to(device)
            target_continuation_gain = targets_dict['continuation_gain'].to(device)
            target_continuation_confidence = targets_dict['continuation_confidence'].to(device)
            target_price_change_pct = targets_dict['price_change_pct'].to(device)
            target_horizon_bars_log = targets_dict['horizon_bars_log'].to(device)
            target_adaptive_confidence = targets_dict['adaptive_confidence'].to(device)

        # Forward pass
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
        if model.multi_task and 'multi_task' in hidden_states:
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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


def interactive_setup(args):
    """
    Interactive menu for training setup.

    Args:
        args: Initial argparse namespace

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

    print("\n" + "=" * 70)
    print("🎯 HIERARCHICAL LNN - INTERACTIVE TRAINING SETUP")
    print("=" * 70)

    # Detect hardware
    hw_info = get_hardware_info()

    print("\n📱 Hardware Detection:")
    if hw_info['cuda_available']:
        print(f"  ✓ NVIDIA GPU: {hw_info['cuda_device']} ({hw_info['cuda_memory_gb']:.1f} GB)")
    if hw_info['mps_available']:
        print(f"  ✓ Apple Silicon: {hw_info['mac_chip']} ({hw_info['total_ram_gb']:.0f} GB RAM)")
    print(f"  ✓ CPU: {hw_info['cpu_count']} threads")

    # Device selection (always show all options, mark availability)
    device_choices = []

    # CUDA option (always show, mark if detected)
    if hw_info['cuda_available']:
        device_choices.append(Choice(value='cuda', name='NVIDIA GPU (CUDA) - Fastest ⚡ [Detected]'))
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
        default=default_device
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

    # Get recommended batch size
    total_ram = hw_info.get('total_ram_gb', 16)
    recommended_batch = get_recommended_batch_size(args.device, total_ram)

    # Training data range
    print()
    args.train_start_year = int(inquirer.number(
        message="Training data start year:",
        default=2015,
        min_allowed=2010,
        max_allowed=2023
    ).execute())

    args.train_end_year = int(inquirer.number(
        message="Training data end year:",
        default=2022,
        min_allowed=int(args.train_start_year),  # Explicit int conversion
        max_allowed=2024
    ).execute())

    # Check for feature cache
    print()
    from src.ml.features import FEATURE_VERSION
    from datetime import datetime as dt

    cache_dir = Path('data/feature_cache')
    cache_dir.mkdir(exist_ok=True)

    # Look for cache files matching this date range (don't know exact length yet)
    cache_pattern = f"rolling_channels_{FEATURE_VERSION}_{args.train_start_year}0101_{args.train_end_year}1231_*.pkl"
    cache_files = list(cache_dir.glob(cache_pattern))

    if cache_files:
        # Found existing cache
        cache_file = cache_files[0]
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        created = dt.fromtimestamp(cache_file.stat().st_mtime)

        print("📂 Feature Cache Found:")
        print(f"   💾 Size: {size_mb:.1f} MB")
        print(f"   📅 Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📊 Version: {FEATURE_VERSION}")
        print()

        args.regenerate_cache = inquirer.select(
            message="Use existing cache or regenerate features?",
            choices=[
                Choice(value=False, name="Use cache (fast - loads in ~5 seconds) ⭐"),
                Choice(value=True, name="Regenerate cache (slow - takes ~45 minutes)")
            ],
            default=False
        ).execute()

        if args.regenerate_cache:
            print(f"   ⚠️  Cache will be regenerated (~45 minutes)")
        else:
            print(f"   ✓ Will use existing cache")
    else:
        # No cache found
        args.regenerate_cache = True
        print("📂 No Feature Cache Found:")
        print(f"   ⚠️  First run will take ~45 minutes to generate rolling channels")
        print(f"   💡 Subsequent runs will load instantly from cache")

    # GPU Acceleration option (ALWAYS shown)
    print()

    # Check if GPU is available
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if torch.cuda.is_available():
        gpu_type = 'CUDA'
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_type = 'MPS'
        gpu_name = 'Apple Silicon'
    else:
        gpu_type = None
        gpu_name = 'Not Available'

    # Determine if cache will be used
    will_use_cache = not getattr(args, 'regenerate_cache', True)

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
        default=100,
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
        default=recommended_batch,
        min_allowed=8,
        max_allowed=max_batch_size
    ).execute())

    args.lr = float(inquirer.number(
        message="Learning rate:",
        default=0.001,
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
        default=False
    ).execute()
    args.preload = preload_choice

    # Multi-task learning
    print()
    args.multi_task = inquirer.confirm(
        message="Enable multi-task learning (hit_band, hit_target, expected_return)?",
        default=True
    ).execute()

    # Output path
    print()
    args.output = inquirer.text(
        message="Model output path:",
        default='models/hierarchical_lnn.pth'
    ).execute()

    # Summary
    print("\n" + "=" * 70)
    print("📋 TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Device: {args.device.upper()}")
    print(f"  Training Period: {args.train_start_year}-{args.train_end_year}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Data Loading: {'Preload' if args.preload else 'Lazy'}")
    print(f"  Multi-Task: {'Enabled' if args.multi_task else 'Disabled'}")
    print(f"  Output: {args.output}")
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
    parser.add_argument('--output', type=str, default='models/hierarchical_lnn.pth',
                        help='Output model path')

    # Configuration
    parser.add_argument('--config', type=str, default='config/hierarchical_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--multi_task', action='store_true', default=True,
                        help='Enable multi-task learning (default: True)')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode with menus')

    args = parser.parse_args()

    # Interactive mode overrides command-line args
    if args.interactive:
        args = interactive_setup(args)

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

    # Auto-set num_workers if not specified
    if args.num_workers is None:
        args.num_workers = {'cuda': 4, 'mps': 2, 'cpu': 2}.get(args.device, 2)

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

    # Clear terminal for clean progress bar display
    os.system('clear')

    # Load data with historical buffer for continuation analysis
    print("\n1. Loading 1-min data...")
    data_feed = CSVDataFeed(timeframe=args.input_timeframe)

    # Load with 2-year historical buffer for continuation analysis
    # This ensures timestamps have sufficient lookback history
    historical_buffer_years = 2
    load_start_year = max(2010, args.train_start_year - historical_buffer_years)  # Don't go before 2010

    df = data_feed.load_aligned_data(
        start_date=f'{load_start_year}-01-01',
        end_date=f'{args.train_end_year}-12-31'
    )

    print(f"   Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    print(f"   Historical buffer: {historical_buffer_years} years (for continuation analysis)")

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

    features_df, continuation_df = extractor.extract_features(
        df,
        use_cache=use_cache,
        use_gpu=use_gpu,
        continuation=True
    )

    # If we got 0 labels and didn't enable debug, enable it now for diagnosis
    if len(continuation_df) == 0 and not debug_mode:
        print("  ⚠️  Got 0 continuation labels. Re-running with debug mode enabled...")
        debug_mode = True
        continuation_df = extractor.generate_continuation_labels(df, timestamps, prediction_horizon=24, debug=debug_mode)

    print(f"   Extracted {len(features_df.columns)} features")
    print(f"   Generated {len(continuation_df) if continuation_df is not None else 0} continuation labels")
    print(f"   Feature names: {extractor.get_feature_names()[:5]}... (showing first 5)")

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset, val_dataset = create_hierarchical_dataset(
        features_df,
        raw_ohlc_df=df,
        continuation_labels_df=continuation_df,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        mode='uniform_bars',
        preload=args.preload,
        validation_split=args.val_split,
        include_continuation=True
    )

    print(f"   Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"   Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda')
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == 'cuda')
        )

    # Create model
    print("\n4. Creating HierarchicalLNN model...")

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
    print(f"   Multi-task heads: {'Enabled' if args.multi_task else 'Disabled'}")
    print(f"   Input features: {extractor.get_feature_dim()}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

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

        # Train (with loss_weights for multi-task)
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch, loss_weights
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

                model.save_checkpoint(args.output, metadata)
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

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
