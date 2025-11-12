"""
train_model_lazy.py - Memory-efficient training with lazy sequence loading

This version creates sequences on-demand during training instead of
pre-creating all 1.35M sequences upfront, reducing memory from 30GB to ~2GB.

Usage:
    python train_model_lazy.py --tsla_events data/tsla_events_REAL.csv --epochs 50
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import time
from tqdm import tqdm
import psutil
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.ml.features_lazy import TradingFeatureExtractorWithProgress
from src.ml.events import CombinedEventsHandler
from src.ml.model import LNNTradingModel, LSTMTradingModel, SelfSupervisedPretrainer
from src.ml.device_manager import DeviceManager
from src.ml.interactive_params import InteractiveParameterSelector, create_argparse_from_params
from src.ml.gpu_monitor import GPUMonitor


def validate_required_files(input_timeframe, events_file=None, verbose=True):
    """
    Validate that required CSV files exist before training.

    Args:
        input_timeframe: Timeframe string (e.g., '15min', '1hour', '4hour', 'daily')
        events_file: Optional path to events file
        verbose: If True, print file information

    Returns:
        bool: True if all required files exist, False otherwise
    """
    missing_files = []
    found_files = []

    # Build file paths
    spy_file = Path(config.DATA_DIR) / f"SPY_{input_timeframe}.csv"
    tsla_file = Path(config.DATA_DIR) / f"TSLA_{input_timeframe}.csv"

    # Check SPY file
    if spy_file.exists():
        size_mb = spy_file.stat().st_size / (1024 * 1024)
        found_files.append(f"  ✓ Found {spy_file} ({size_mb:.1f} MB)")
    else:
        missing_files.append(f"  ✗ SPY data: {spy_file}")

    # Check TSLA file
    if tsla_file.exists():
        size_mb = tsla_file.stat().st_size / (1024 * 1024)
        found_files.append(f"  ✓ Found {tsla_file} ({size_mb:.1f} MB)")
    else:
        missing_files.append(f"  ✗ TSLA data: {tsla_file}")

    # Check events file if provided
    if events_file:
        events_path = Path(events_file)
        if events_path.exists():
            # Count lines to estimate events
            with open(events_path, 'r') as f:
                event_count = sum(1 for line in f) - 1  # Subtract header
            found_files.append(f"  ✓ Found {events_file} ({event_count} events)")
        else:
            missing_files.append(f"  ✗ Events file: {events_file}")

    # Report results
    if missing_files:
        if verbose:
            print("\n" + "=" * 70)
            print("❌ MISSING REQUIRED FILES")
            print("=" * 70)
            print("\nThe following data files are required but not found:\n")
            for file in missing_files:
                print(file)
            print("\nTo create timeframe CSV files, run:")
            print("  python scripts/create_multiscale_csvs.py")
            print("\nFor events file, check:")
            print("  data/tsla_events_REAL.csv")
            print("=" * 70 + "\n")
        return False

    # All files found
    if verbose and found_files:
        print("\n📁 Validated data files:")
        for file in found_files:
            print(file)
        print()

    return True


def validate_all_timeframes(events_file=None):
    """
    Validate CSV files exist for all 4 timeframes used in multi-model training.

    Returns:
        bool: True if all files exist, False if any are missing
    """
    timeframes = ['15min', '1hour', '4hour', 'daily']
    all_valid = True

    print("\n" + "=" * 70)
    print("📁 VALIDATING MULTI-TIMEFRAME DATA FILES")
    print("=" * 70)

    for tf in timeframes:
        print(f"\nChecking {tf} timeframe:")
        if not validate_required_files(tf, events_file=None, verbose=False):
            spy_file = Path(config.DATA_DIR) / f"SPY_{tf}.csv"
            tsla_file = Path(config.DATA_DIR) / f"TSLA_{tf}.csv"
            if not spy_file.exists():
                print(f"  ✗ Missing: {spy_file}")
            if not tsla_file.exists():
                print(f"  ✗ Missing: {tsla_file}")
            all_valid = False
        else:
            spy_file = Path(config.DATA_DIR) / f"SPY_{tf}.csv"
            tsla_file = Path(config.DATA_DIR) / f"TSLA_{tf}.csv"
            spy_size = spy_file.stat().st_size / (1024 * 1024)
            tsla_size = tsla_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ SPY_{tf}.csv ({spy_size:.1f} MB)")
            print(f"  ✓ TSLA_{tf}.csv ({tsla_size:.1f} MB)")

    # Check events file
    if events_file:
        events_path = Path(events_file)
        if events_path.exists():
            with open(events_path, 'r') as f:
                event_count = sum(1 for line in f) - 1
            print(f"\n✓ Events file: {events_file} ({event_count} events)")
        else:
            print(f"\n✗ Missing events file: {events_file}")
            all_valid = False

    if not all_valid:
        print("\n" + "=" * 70)
        print("❌ Some required files are missing!")
        print("Run: python scripts/create_multiscale_csvs.py")
        print("=" * 70 + "\n")
        return False

    print("\n✅ All required files validated successfully!")
    print("=" * 70 + "\n")
    return True


class LazyTradingDataset(Dataset):
    """
    Memory-efficient dataset that creates sequences on-demand.
    Stores only the raw features DataFrame, not pre-computed sequences.
    """

    def __init__(self, features_df, sequence_length=168, target_horizon=24,
                 events_handler=None, validation_split=0.0, is_validation=False):
        """
        Args:
            features_df: DataFrame with extracted features
            sequence_length: Number of timesteps per sequence
            target_horizon: Number of hours to predict ahead
            events_handler: Optional events handler for embeddings
            validation_split: Fraction for validation (0.0-1.0)
            is_validation: Whether this is the validation set
        """
        self.features_df = features_df
        self.features_array = features_df.values  # Convert once for speed
        self.feature_names = features_df.columns.tolist()
        self.timestamps = features_df.index

        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.events_handler = events_handler

        # Calculate valid indices for sequences
        total_samples = len(features_df) - sequence_length - target_horizon

        if total_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {sequence_length + target_horizon} bars")

        # Split train/validation
        if validation_split > 0:
            val_size = int(total_samples * validation_split)
            train_size = total_samples - val_size

            if is_validation:
                self.start_idx = train_size
                self.num_samples = val_size
            else:
                self.start_idx = 0
                self.num_samples = train_size
        else:
            self.start_idx = 0
            self.num_samples = total_samples

        print(f"  Dataset: {self.num_samples:,} sequences available (lazy loading)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Create a single sequence on-demand.
        This is called by DataLoader for each sample in the batch.
        """
        # Adjust index to account for train/val split
        actual_idx = self.start_idx + idx

        # Extract input sequence (X)
        seq_start = actual_idx
        seq_end = actual_idx + self.sequence_length

        # Get sequence features
        X_seq = self.features_array[seq_start:seq_end]

        # Extract targets (y) - high and low in next target_horizon
        target_start = seq_end
        target_end = seq_end + self.target_horizon

        # Find tsla_close column index
        try:
            close_idx = self.feature_names.index('tsla_close')
        except ValueError:
            # Fallback if tsla_close not found
            close_idx = 0

        # Get future prices
        future_prices = self.features_array[target_start:target_end, close_idx]

        # Calculate high and low
        if len(future_prices) > 0:
            target_high = np.max(future_prices)
            target_low = np.min(future_prices)
        else:
            # Edge case: use last available price
            target_high = self.features_array[target_start - 1, close_idx]
            target_low = target_high

        # Convert to tensors
        X = torch.tensor(X_seq, dtype=torch.float32)
        y = torch.tensor([target_high, target_low], dtype=torch.float32)

        # Get event embedding if handler provided
        if self.events_handler is not None:
            seq_timestamp = self.timestamps[seq_end]
            events = self.events_handler.get_events_for_date(
                str(seq_timestamp.date()),
                lookback_days=config.EVENT_LOOKBACK_DAYS
            )
            event_embed = self.events_handler.embed_events(events).squeeze(0)
        else:
            event_embed = torch.zeros(21, dtype=torch.float32)

        return X, y, event_embed


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_and_prepare_data_lazy(spy_file, tsla_file, start_year, end_year,
                               tsla_events_file=None, macro_api_key=None,
                               timeframe='1min'):
    """
    Load and prepare data WITHOUT pre-creating sequences.
    Returns features DataFrame and events handler for lazy loading.
    """
    print("\n" + "=" * 70)
    print("📊 LOADING AND PREPARING DATA (Lazy Mode)")
    print("=" * 70)
    print("  Note: Only 3 steps (no sequence pre-creation - that's the memory fix!)")
    start_time = time.time()

    # 1. Load aligned SPY-TSLA data
    print(f"\n▶ Step 1/3: Loading SPY and TSLA data ({start_year} to {end_year})...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    data_feed = CSVDataFeed(timeframe=timeframe)
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    with tqdm(total=2, desc="  Loading data files", unit="file") as pbar:
        aligned_df = data_feed.load_aligned_data(start_date, end_date)
        pbar.update(2)

    print(f"  ✓ Loaded {len(aligned_df):,} aligned {timeframe} bars")
    print(f"  ✓ Date range: {aligned_df.index[0]} to {aligned_df.index[-1]}")

    # 2. Extract features
    print(f"\n▶ Step 2/3: Extracting features...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    # Use feature extractor with progress feedback
    feature_extractor = TradingFeatureExtractorWithProgress()

    # Extract features with progress bar
    extract_start = time.time()
    features_df = feature_extractor.extract_features(aligned_df)
    extract_time = time.time() - extract_start

    print(f"  ✓ Extracted {feature_extractor.get_feature_dim()} features in {extract_time:.1f}s")
    print(f"  ✓ Features DataFrame: {features_df.shape} ({features_df.memory_usage().sum() / 1024**2:.1f} MB)")

    # 3. Load events handler (but don't create embeddings yet)
    print(f"\n▶ Step 3/3: Loading events handler...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    events_handler = CombinedEventsHandler(tsla_events_file, macro_api_key)
    all_events = events_handler.load_events(start_date, end_date)
    print(f"  ✓ Loaded {len(all_events)} events (will embed on-demand)")

    # Clean up original dataframe to free memory
    del aligned_df

    elapsed = time.time() - start_time
    print(f"\n  ✓ Data preparation complete in {elapsed:.1f}s")
    print(f"  ✓ Memory usage: {get_memory_usage():.1f} MB (vs ~30GB with pre-created sequences)")

    return features_df, events_handler, feature_extractor


def train_supervised_lazy(model, features_df, events_handler, epochs=50, batch_size=32,
                         lr=0.001, validation_split=0.1, sequence_length=168,
                         target_horizon=24, device=torch.device('cpu'),
                         num_workers=0, pin_memory=False, gpu_monitor=False):
    """
    Supervised training using lazy sequence loading.

    Args:
        gpu_monitor: If True, display real-time GPU utilization metrics
    """
    print("\n" + "=" * 70)
    print("🎯 SUPERVISED TRAINING (Memory-Efficient)")
    print("=" * 70)
    print(f"  Device: {device}")
    start_time = time.time()

    print(f"  Total data points: {len(features_df):,}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Target horizon: {target_horizon}")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial learning rate: {lr}")
    print(f"  Memory usage: {get_memory_usage():.1f} MB")

    # Initialize GPU monitor if requested
    monitor = GPUMonitor(device) if gpu_monitor and device.type == 'cuda' else None
    if monitor:
        print(f"  GPU monitoring: enabled")

    # Create lazy datasets
    train_dataset = LazyTradingDataset(
        features_df, sequence_length, target_horizon,
        events_handler, validation_split, is_validation=False
    )

    val_dataset = LazyTradingDataset(
        features_df, sequence_length, target_horizon,
        events_handler, validation_split, is_validation=True
    )

    print(f"\n  Train size: {len(train_dataset):,} sequences")
    print(f"  Validation size: {len(val_dataset):,} sequences")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),  # Reuse workers to prevent file descriptor leaks
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),  # Reuse workers to prevent file descriptor leaks
        pin_memory=pin_memory
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    print(f"\n  Starting training...")
    print(f"  " + "-" * 65)

    # Main epoch loop
    epoch_pbar = tqdm(range(epochs), desc="  Training progress", unit="epoch")

    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"    Training", unit="batch", leave=False)

        for batch_x, batch_y, batch_events in train_pbar:
            # Move batches to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions, _ = model.forward(batch_x)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Update progress bar with loss, memory, and optional GPU metrics
            postfix = {'loss': f'{loss.item():.4f}', 'mem': f'{get_memory_usage():.0f}MB'}
            if monitor:
                postfix.update(monitor.get_compact_metrics())
            train_pbar.set_postfix(postfix)

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        val_pbar = tqdm(val_loader, desc=f"    Validating", unit="batch", leave=False)

        with torch.no_grad():
            for batch_x, batch_y, batch_events in val_pbar:
                # Move batches to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                predictions, _ = model.forward(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                val_batches += 1

                # Update validation progress bar with optional GPU metrics
                postfix = {'loss': f'{loss.item():.4f}'}
                if monitor:
                    postfix.update(monitor.get_compact_metrics())
                val_pbar.set_postfix(postfix)

        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # Update progress bar
        epoch_pbar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{avg_val_loss:.4f}',
            'lr': f'{new_lr:.1e}',
            'mem': f'{get_memory_usage():.0f}MB'
        })

        # Periodic cleanup to prevent memory/file descriptor accumulation
        if (epoch + 1) % 10 == 0:
            import gc
            gc.collect()  # Force garbage collection
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear CUDA cache

        # Print epoch summary
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            status = " 🎉 NEW BEST!"

        if new_lr < old_lr:
            status += " ⚡ LR reduced"

        # Print epoch summary with optional GPU details
        print(f"  Epoch {epoch + 1:3d}/{epochs}: "
              f"Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | "
              f"LR={new_lr:.1e} | Mem={get_memory_usage():.0f}MB | "
              f"Time={epoch_time:.1f}s{status}")

        # Show detailed GPU metrics if monitoring enabled
        if monitor:
            gpu_details = monitor.format_detailed()
            if gpu_details:
                print(f"    {gpu_details}")

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n  " + "-" * 65)
    print(f"  ✓ Training completed in {elapsed/60:.1f} minutes")
    print(f"  ✓ Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  ✓ Final train/val loss: {train_losses[-1]:.4f}/{val_losses[-1]:.4f}")
    print(f"  ✓ Peak memory usage: {get_memory_usage():.1f} MB")

    return train_losses, val_losses


def self_supervised_pretrain_lazy(model, features_df, events_handler, epochs=10,
                                  batch_size=32, lr=0.001, sequence_length=168,
                                  device=torch.device('cpu'), num_workers=0, pin_memory=False,
                                  gpu_monitor=False):
    """
    Self-supervised pretraining using lazy loading.

    Args:
        gpu_monitor: If True, display real-time GPU utilization metrics
    """
    print("\n" + "=" * 70)
    print("🔧 SELF-SUPERVISED PRETRAINING (Memory-Efficient)")
    print("=" * 70)
    print(f"  Learning to understand patterns via masked reconstruction")
    print(f"  Device: {device}")
    print(f"  Mask ratio: 15% | Learning rate: {lr}")

    # Initialize GPU monitor if requested
    monitor = GPUMonitor(device) if gpu_monitor and device.type == 'cuda' else None
    if monitor:
        print(f"  GPU monitoring: enabled")

    pretrainer = SelfSupervisedPretrainer(model, mask_ratio=0.15)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pretrainer.reconstruction_head.parameters()),
        lr=lr
    )

    # Create lazy dataset for pretraining
    pretrain_dataset = LazyTradingDataset(
        features_df, sequence_length, target_horizon=24,
        events_handler=events_handler
    )

    dataloader = DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),  # Reuse workers to prevent file descriptor leaks
        pin_memory=pin_memory
    )

    num_batches_per_epoch = len(dataloader)
    print(f"  Dataset: {len(pretrain_dataset):,} sequences")
    print(f"  Batches per epoch: {num_batches_per_epoch:,} (batch size: {batch_size})")
    print(f"  Memory usage: {get_memory_usage():.1f} MB")
    print(f"\n  Starting pretraining...")
    print(f"  " + "-" * 65)

    pretrain_start = time.time()
    pretrain_losses = []

    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="  Pretraining progress", unit="epoch",
                      bar_format="{l_bar}{bar:25}{r_bar}")

    for epoch in epoch_pbar:
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0

        # Batch progress bar (nested)
        batch_pbar = tqdm(dataloader, desc=f"    Epoch {epoch + 1}/{epochs}",
                         unit="batch", leave=False,
                         bar_format="{l_bar}{bar:20}{r_bar}")

        for batch_x, _, _ in batch_pbar:
            # Move batch to device
            batch_x = batch_x.to(device)

            loss = pretrainer.pretrain_step(batch_x, optimizer)
            total_loss += loss
            batch_count += 1

            # Update batch progress bar every 100 batches
            if batch_count % 100 == 0:
                current_avg = total_loss / batch_count
                postfix = {
                    'loss': f'{loss:.4f}',
                    'avg': f'{current_avg:.4f}',
                    'mem': f'{get_memory_usage():.0f}MB'
                }
                if monitor:
                    postfix.update(monitor.get_compact_metrics())
                batch_pbar.set_postfix(postfix)

        avg_loss = total_loss / batch_count
        pretrain_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{epoch_time:.1f}s',
            'mem': f'{get_memory_usage():.0f}MB'
        })

        # Periodic cleanup to prevent memory/file descriptor accumulation
        if (epoch + 1) % 5 == 0:  # More frequent for pretraining since it's shorter
            import gc
            gc.collect()  # Force garbage collection
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear CUDA cache

        print(f"    Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f} | "
              f"Time={epoch_time:.1f}s | Mem={get_memory_usage():.0f}MB")

        # Show detailed GPU metrics if monitoring enabled
        if monitor:
            gpu_details = monitor.format_detailed()
            if gpu_details:
                print(f"      {gpu_details}")

    elapsed = time.time() - pretrain_start
    print(f"\n  " + "-" * 65)
    print(f"  ✓ Pretraining completed in {elapsed/60:.1f} minutes")
    if pretrain_losses:
        print(f"  ✓ Final loss: {pretrain_losses[-1]:.4f}")
        if len(pretrain_losses) > 1:
            improvement = (pretrain_losses[0] - pretrain_losses[-1]) / pretrain_losses[0] * 100
            print(f"  ✓ Loss reduction: {improvement:.1f}%")


def run_training_pipeline(args):
    """
    Run complete training pipeline with given arguments.

    This function contains all the training logic extracted from main()
    so it can be called multiple times (e.g., for multi-model training).

    Args:
        args: Argparse namespace with all training configuration

    Returns:
        dict with training results (losses, model_path, etc.)
    """
    # Validate required files exist before any setup
    if not validate_required_files(args.input_timeframe, args.tsla_events):
        sys.exit(1)

    # Print header
    print("\n" + "=" * 70)
    print("🚀 MEMORY-EFFICIENT TRAINING (Lazy Loading)")
    print("=" * 70)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model: {args.model_type}")
    print(f"⏱️  Input Timeframe: {args.input_timeframe}")
    print(f"📏 Sequence Length: {args.sequence_length} bars")
    print(f"📊 Training period: {args.start_year}-{args.end_year}")
    print(f"🔄 Epochs: {args.epochs} supervised + {args.pretrain_epochs} pretrain")
    print(f"💾 Output: {args.output}")
    print(f"🖥️  System memory: {psutil.virtual_memory().available / 1024**3:.1f} GB available")
    print(f"📉 Expected memory usage: ~2-3 GB (vs 30+ GB with pre-created sequences)")
    print("=" * 70)

    # Device selection
    device_manager = DeviceManager()

    if args.device:
        # Force specific device
        device = torch.device(args.device)
        print(f"\n🖥️  Using forced device: {device}")
        if device.type == 'mps':
            device_manager.setup_mps_environment()
    elif args.auto_device:
        # Auto-select best device
        device = device_manager.select_device_auto(verbose=True)
        if device.type == 'mps':
            device_manager.setup_mps_environment()
    else:
        # Interactive selection
        device = device_manager.select_device_interactive()
        if device.type == 'mps':
            device_manager.setup_mps_environment()

    # Print device summary
    device_manager.print_device_summary(device)

    total_start = time.time()

    # 1. Load and prepare data (lazy mode)
    features_df, events_handler, feature_extractor = load_and_prepare_data_lazy(
        args.spy_data, args.tsla_data,
        args.start_year, args.end_year,
        args.tsla_events, args.macro_api_key,
        args.input_timeframe
    )

    # 2. Create model
    print("\n" + "=" * 70)
    print("🧠 CREATING MODEL")
    print("=" * 70)

    input_size = feature_extractor.get_feature_dim()

    if args.model_type == "LNN":
        print("  Creating Liquid Neural Network (memory-efficient)")
        model = LNNTradingModel(input_size, args.hidden_size)
    else:
        model = LSTMTradingModel(input_size, args.hidden_size)

    # Move model to device
    model = device_manager.move_to_device(model, device, verbose=True)
    print(f"  ✓ Model moved to {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model type: {args.model_type}")
    print(f"  Input features: {input_size}")
    print(f"  Hidden units: {args.hidden_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB")
    print(f"  Current memory: {get_memory_usage():.1f} MB")

    # GPU optimization: Auto-detect num_workers and pin_memory based on device
    # Detect system capabilities for smart defaults
    vram_gb = None
    cpu_count = os.cpu_count() or 4

    if device.type == 'cuda':
        try:
            vram_bytes = torch.cuda.get_device_properties(device).total_memory
            vram_gb = vram_bytes / (1024**3)
        except:
            vram_gb = None

    if args.num_workers is None:
        # Smart auto-detection based on device and system capabilities
        if device.type == 'cuda':
            # For CUDA: Use more workers for powerful CPUs
            # Balance: ~2-4 CPU cores per worker, max 8 workers
            import platform
            if platform.system() == 'Darwin':
                # macOS: Conservative due to file descriptor limits
                num_workers = 2
            else:
                # Linux/Windows: Scale with CPU count
                if cpu_count >= 16:
                    num_workers = 6  # Powerful CPU (16+ cores)
                elif cpu_count >= 8:
                    num_workers = 4  # Mid-range CPU (8-15 cores)
                else:
                    num_workers = 2  # Lower-end CPU (<8 cores)
        else:
            # CPU/MPS: Use main thread only
            num_workers = 0
        workers_source = "auto-detected"
    else:
        num_workers = args.num_workers
        workers_source = "user-specified"

    if args.pin_memory is None:
        # Auto-detect: Enable for CUDA, disable for CPU/MPS
        pin_memory = (device.type == 'cuda')
        pin_source = "auto-detected"
    else:
        pin_memory = args.pin_memory
        pin_source = "user-specified"

    # macOS warning for high num_workers
    import platform
    if platform.system() == 'Darwin' and num_workers > 4:
        print(f"\n  ⚠️  macOS File Limit Warning:")
        print(f"     - You selected num_workers={num_workers}")
        print(f"     - macOS has a default limit of 256 open files")
        print(f"     - This may cause 'Too many open files' errors after 15-20 epochs")
        print(f"     - Solutions:")
        print(f"       1. Run: ulimit -n 4096 (before training)")
        print(f"       2. Or use num_workers=0-4")
        print(f"     - persistent_workers=True is enabled to help mitigate this")

    # Display GPU optimization settings
    if device.type == 'cuda':
        print(f"\n  🚀 GPU optimizations enabled:")
        print(f"     - Device: {torch.cuda.get_device_name(device)}")
        if vram_gb:
            print(f"     - VRAM: {vram_gb:.1f} GB")
        print(f"     - CPU cores: {cpu_count}")
        print(f"     - num_workers: {num_workers} ({workers_source}, parallel data loading)")
        print(f"     - pin_memory: {pin_memory} ({pin_source}, faster GPU transfers)")

        # Smart performance tips based on VRAM
        if vram_gb and vram_gb >= 40:
            # A40/A100-class GPU
            optimal_batch = "256-512"
            optimal_workers = "6-8"
        elif vram_gb and vram_gb >= 16:
            # RTX 3090/4090-class GPU
            optimal_batch = "128-256"
            optimal_workers = "4-6"
        else:
            # Lower-end GPU
            optimal_batch = "64-128"
            optimal_workers = "2-4"

        print(f"     💡 Optimal settings for your GPU:")
        print(f"        --batch_size {optimal_batch}")
        print(f"        --num_workers {optimal_workers}")
    else:
        print(f"\n  ℹ️  CPU/MPS mode:")
        print(f"     - num_workers: {num_workers} ({workers_source})")
        print(f"     - pin_memory: {pin_memory} ({pin_source})")
        if num_workers > 0 and device.type == 'mps':
            print(f"     ⚠️  Warning: num_workers>0 may cause issues on MPS, consider using 0")

    # 3. Self-supervised pretraining (optional)
    if args.pretrain_epochs > 0:
        self_supervised_pretrain_lazy(
            model, features_df, events_handler,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sequence_length=args.sequence_length,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            gpu_monitor=args.gpu_monitor
        )

    # 4. Supervised training
    train_losses, val_losses = train_supervised_lazy(
        model, features_df, events_handler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        validation_split=config.ML_VALIDATION_SPLIT,
        sequence_length=args.sequence_length,
        target_horizon=args.prediction_horizon,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        gpu_monitor=args.gpu_monitor
    )

    # 5. Save model
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL")
    print("=" * 70)

    metadata = {
        'model_type': args.model_type,
        'input_size': input_size,
        'hidden_size': args.hidden_size,
        'input_timeframe': args.input_timeframe,
        'sequence_length': args.sequence_length,
        'prediction_horizon': args.prediction_horizon,
        'prediction_mode': args.prediction_mode,
        'train_start_year': args.start_year,
        'train_end_year': args.end_year,
        'epochs': args.epochs,
        'pretrain_epochs': args.pretrain_epochs,
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0,
        'training_date': datetime.now().isoformat(),
        'feature_names': feature_extractor.get_feature_names(),
        'training_mode': 'lazy_loading',
        'peak_memory_mb': get_memory_usage(),
        'device': str(device),
        'device_type': device.type
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(args.output, metadata)

    # Print final summary
    total_time = time.time() - total_start
    print(f"  ✓ Model saved to: {args.output}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"💾 Peak memory: {get_memory_usage():.1f} MB (vs ~30GB with pre-created sequences)")
    print(f"\n📈 Next steps:")
    print(f"   1. Run backtesting: python backtest.py --model_path {args.output}")
    print(f"   2. Validate results: python validate_results.py --model_path {args.output}")
    print("=" * 70)

    return {
        'success': True,
        'model_path': args.output,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_time
    }


def train_all_models_interactive(base_args):
    """
    Train all 4 timeframe models sequentially with shared configuration.

    User configures parameters once, then all 4 models (15min, 1hour, 4hour, daily)
    are trained with the same settings but different input timeframes.
    """
    import copy

    print("\n" + "=" * 70)
    print("TRAINING ALL 4 MODELS")
    print("=" * 70)
    print("\nYou'll configure parameters once, then all 4 models will train:")
    print("  1. LNN_15min (200 bars = 50 hours)")
    print("  2. LNN_1hour (200 bars = 8 days)")
    print("  3. LNN_4hour (200 bars = 33 days)")
    print("  4. LNN_daily (200 bars = 9 months)")
    print("\nModels will train SEQUENTIALLY (not parallel).")
    print("Estimated total time: ~60-100 minutes on T4 GPU")
    print()

    # Validate all timeframe files exist before configuration
    if not validate_all_timeframes(base_args.tsla_events):
        print("\n⚠️  Cannot proceed with multi-model training until all files exist.")
        sys.exit(1)

    # Get configuration from user (ONE TIME)
    selector = InteractiveParameterSelector(mode='lazy')
    base_params = selector.run()

    # Timeframes to train
    timeframes = ['15min', '1hour', '4hour', 'daily']

    # Track results
    results = []

    # Train each model
    for i, tf in enumerate(timeframes, 1):
        print("\n\n" + "=" * 70)
        print(f"TRAINING MODEL {i}/4: {tf.upper()}")
        print("=" * 70)

        # Create params for this timeframe
        params = copy.deepcopy(base_params)

        # Override timeframe-specific settings
        params['input_timeframe'] = tf
        params['spy_data'] = f'data/SPY_{tf}.csv'
        params['tsla_data'] = f'data/TSLA_{tf}.csv'

        # Handle prediction mode
        if base_params.get('prediction_horizon_mode') == 'uniform_time':
            # For uniform time mode (24 hours), adjust bars per timeframe
            bars_for_24h = {
                '15min': 96,  # 96 * 15min = 24 hours
                '1hour': 24,  # 24 * 1h = 24 hours
                '4hour': 6,   # 6 * 4h = 24 hours
                'daily': 1    # 1 * 24h = 24 hours
            }
            params['prediction_horizon_hours'] = bars_for_24h.get(tf, 24)
            print(f"  📊 Uniform time mode: Using {params['prediction_horizon_hours']} bars for 24-hour prediction")
        # else: uniform_bars mode - use the same value for all models (already set)

        # Create args for this model
        model_args = copy.deepcopy(base_args)
        model_args = create_argparse_from_params(params, model_args)
        model_args.output = f'models/lnn_{tf}.pth'

        # Ensure data paths are set
        model_args.spy_data = f'data/SPY_{tf}.csv'
        model_args.tsla_data = f'data/TSLA_{tf}.csv'
        model_args.input_timeframe = tf

        print(f"\n📁 Training on: {model_args.tsla_data}")
        print(f"💾 Output: {model_args.output}")
        print()

        # Run full training pipeline for this model
        try:
            result = run_training_pipeline(model_args)
            results.append(result)
            print(f"\n✓ Model {i}/4 complete: {model_args.output}")

        except Exception as e:
            print(f"\n❌ Error training {tf} model: {e}")
            import traceback
            traceback.print_exc()

            cont = input("\nContinue with next model? [y/N]: ").strip().lower()
            if cont != 'y':
                print("Stopped.")
                break

    print("\n\n" + "=" * 70)
    print("✅ MULTI-MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTrained {len(results)}/{len(timeframes)} models:")
    for i, tf in enumerate(timeframes):
        model_path = Path(f'models/lnn_{tf}.pth')
        if model_path.exists():
            if i < len(results) and results[i]['success']:
                time_min = results[i]['total_time'] / 60
                print(f"  ✓ {model_path} ({time_min:.1f} min)")
            else:
                print(f"  ✓ {model_path}")
        else:
            print(f"  ✗ {model_path} (not found)")

    print("\n📈 Next steps:")
    print("   1. Backtest all models: python backtest_all_models.py --test_year 2023 --num_simulations 500")
    print("   2. Train Meta-LNN: python train_meta_lnn.py --mode backtest_no_news")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient ML training with lazy loading')

    # Data arguments
    parser.add_argument('--input_timeframe', type=str, default='1min',
                       choices=['1min', '5min', '15min', '30min', '1hour', '2hour', '3hour', '4hour', 'daily', 'weekly', 'monthly', '3month'],
                       help='Timeframe for input data (default: 1min)')
    parser.add_argument('--spy_data', type=str, default=None,
                       help='SPY data file (default: data/SPY_{timeframe}.csv)')
    parser.add_argument('--tsla_data', type=str, default=None,
                       help='TSLA data file (default: data/TSLA_{timeframe}.csv)')
    parser.add_argument('--tsla_events', type=str, default='data/tsla_events_REAL.csv')
    parser.add_argument('--macro_api_key', type=str, default=None)

    # Training arguments
    parser.add_argument('--start_year', type=int, default=config.ML_TRAIN_START_YEAR)
    parser.add_argument('--end_year', type=int, default=config.ML_TRAIN_END_YEAR)
    parser.add_argument('--epochs', type=int, default=config.ML_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.ML_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LNN_LEARNING_RATE)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--sequence_length', type=int, default=config.ML_SEQUENCE_LENGTH,
                       help='Number of bars to look back (default: 200, uniform across all timeframes)')
    parser.add_argument('--prediction_horizon', type=int, default=config.PREDICTION_HORIZON_HOURS,
                       help='Number of BARS ahead to predict (default: 24 bars, NOT hours!)')
    parser.add_argument('--prediction_mode', type=str, default='uniform_bars',
                       choices=['uniform_bars', 'uniform_time'],
                       help='Prediction mode: uniform_bars (same bar count) or uniform_time (24h for all)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default=config.ML_MODEL_TYPE)
    parser.add_argument('--hidden_size', type=int, default=config.LNN_HIDDEN_SIZE)

    # Output
    parser.add_argument('--output', type=str, default='models/lnn_lazy.pth')

    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Force specific device (default: interactive selection)')
    parser.add_argument('--auto_device', action='store_true',
                       help='Auto-select best device without prompting')

    # GPU optimization arguments
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers (default: auto-detect based on device)')
    parser.add_argument('--pin_memory', type=lambda x: x.lower() == 'true', default=None,
                       help='Pin memory for faster GPU transfers (default: auto-detect based on device)')
    parser.add_argument('--gpu_monitor', action='store_true',
                       help='Enable real-time GPU utilization monitoring (requires nvidia-ml-py)')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive parameter selection mode')

    args = parser.parse_args()

    # Set default data paths based on timeframe if not explicitly provided
    if args.spy_data is None:
        args.spy_data = f'data/SPY_{args.input_timeframe}.csv'
    if args.tsla_data is None:
        args.tsla_data = f'data/TSLA_{args.input_timeframe}.csv'

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 70)
        print("🎛️  LAUNCHING INTERACTIVE MODE")
        print("=" * 70)

        # Ask: Single model or all 4?
        print("\nTraining mode:")
        print("  1. Single model (choose one timeframe)")
        print("  2. All 4 models (15min, 1hour, 4hour, daily) - runs sequentially")
        mode_choice = input("\nChoice [1-2]: ").strip()

        if mode_choice == '2':
            # Train all 4 models with shared config
            train_all_models_interactive(args)
            return  # Exit after training all models
        else:
            # Single model: Ask for timeframe first
            print("\n" + "=" * 70)
            print("SELECT TIMEFRAME")
            print("=" * 70)
            print("\nAvailable timeframes:")
            timeframes = ['15min', '1hour', '4hour', 'daily']
            for i, tf in enumerate(timeframes, 1):
                descriptions = {
                    '15min': '50 hours lookback, 6 hours prediction',
                    '1hour': '8 days lookback, 24 hours prediction',
                    '4hour': '33 days lookback, 4 days prediction',
                    'daily': '9 months lookback, 24 days prediction'
                }
                print(f"  {i}. {tf:<10} - {descriptions[tf]}")

            tf_choice = input("\nSelect timeframe [1-4]: ").strip()
            timeframe_map = {'1': '15min', '2': '1hour', '3': '4hour', '4': 'daily'}
            selected_timeframe = timeframe_map.get(tf_choice, '15min')

            print(f"\n✓ Selected: {selected_timeframe}")
            print(f"  Data files will be set to: data/SPY_{selected_timeframe}.csv, data/TSLA_{selected_timeframe}.csv")

            # Normal single-model interactive with pre-selected timeframe
            selector = InteractiveParameterSelector(mode='lazy')
            params = selector.run()

            # Override timeframe with user's choice
            params['input_timeframe'] = selected_timeframe
            params['spy_data'] = f'data/SPY_{selected_timeframe}.csv'
            params['tsla_data'] = f'data/TSLA_{selected_timeframe}.csv'

            args = create_argparse_from_params(params, args)
            print("\n✓ Configuration complete! Starting training...\n")

    # Run training pipeline
    run_training_pipeline(args)


if __name__ == '__main__':
    main()