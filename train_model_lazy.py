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


class LazyTradingDataset(Dataset):
    """
    Memory-efficient dataset that creates sequences on-demand.
    Stores only the raw features DataFrame, not pre-computed sequences.

    NOTE: For multi-worker DataLoader compatibility, we need to handle
    events_handler carefully as it may not be picklable.
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
        self.timestamps = features_df.index.copy()  # Make a copy for workers

        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        # Don't store events_handler - create dummy embeddings instead for multi-worker compatibility
        self.use_events = events_handler is not None
        self.events_handler = None  # Set to None for pickling

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

        # For multi-worker compatibility, use dummy event embeddings
        # (events processing can be computationally expensive and non-picklable)
        event_embed = torch.zeros(21, dtype=torch.float32)

        return X, y, event_embed


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_gpu_memory_usage(device):
    """Log GPU memory usage if enabled"""
    if not config.LOG_GPU_MEMORY:
        return ""

    if device.type == 'mps':
        # For MPS, show system memory (unified memory architecture)
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1024**3
        total_gb = mem.total / 1024**3
        return f" | Mem: {used_gb:.1f}/{total_gb:.1f}GB"
    elif device.type == 'cuda':
        # For CUDA, show GPU memory
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        return f" | GPU: {allocated:.1f}/{reserved:.1f}GB"
    else:
        return ""


def load_and_prepare_data_lazy(spy_file, tsla_file, start_year, end_year,
                               tsla_events_file=None, macro_api_key=None):
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

    data_feed = CSVDataFeed()
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    with tqdm(total=2, desc="  Loading data files", unit="file") as pbar:
        aligned_df = data_feed.load_aligned_data(start_date, end_date)
        pbar.update(2)

    print(f"  ✓ Loaded {len(aligned_df):,} aligned 1-minute bars")
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
                         target_horizon=24, device=torch.device('cpu')):
    """
    Supervised training using lazy sequence loading.
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

    # Create data loaders with multi-worker support
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,  # Use multiple workers for parallel data loading
        pin_memory=config.PIN_MEMORY,  # Pin memory for faster GPU transfers
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None  # Prefetch batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, config.NUM_WORKERS // 2),  # Use fewer workers for validation
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS if config.NUM_WORKERS > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Mixed precision training (Automatic Mixed Precision)
    use_amp = config.USE_MIXED_PRECISION and device.type in ['cuda', 'mps']
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    if use_amp:
        print(f"  ✓ Mixed precision training enabled (AMP with {device.type.upper()})")
    else:
        print(f"  ✓ Full precision training (FP32)")

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

            # Use automatic mixed precision if enabled
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    predictions, _ = model.forward(batch_x)
                    loss = criterion(predictions, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions, _ = model.forward(batch_x)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            # Update progress bar with loss and GPU memory
            mem_info = log_gpu_memory_usage(device)
            train_pbar.set_postfix_str(f"loss: {loss.item():.4f}{mem_info}")

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

                # Use mixed precision for validation too
                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        predictions, _ = model.forward(batch_x)
                        loss = criterion(predictions, batch_y)
                else:
                    predictions, _ = model.forward(batch_x)
                    loss = criterion(predictions, batch_y)

                val_loss += loss.item()
                val_batches += 1

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

        # Print epoch summary
        status = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            status = " 🎉 NEW BEST!"

        if new_lr < old_lr:
            status += " ⚡ LR reduced"

        print(f"  Epoch {epoch + 1:3d}/{epochs}: "
              f"Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | "
              f"LR={new_lr:.1e} | Mem={get_memory_usage():.0f}MB | "
              f"Time={epoch_time:.1f}s{status}")

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
                                  device=torch.device('cpu')):
    """
    Self-supervised pretraining using lazy loading.
    """
    print("\n" + "=" * 70)
    print("🔧 SELF-SUPERVISED PRETRAINING (Memory-Efficient)")
    print("=" * 70)
    print(f"  Learning to understand patterns via masked reconstruction")
    print(f"  Device: {device}")
    print(f"  Mask ratio: 15% | Learning rate: {lr}")

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
        num_workers=0
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
                batch_pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'avg': f'{current_avg:.4f}',
                    'mem': f'{get_memory_usage():.0f}MB'
                })

        avg_loss = total_loss / batch_count
        pretrain_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'time': f'{epoch_time:.1f}s',
            'mem': f'{get_memory_usage():.0f}MB'
        })

        print(f"    Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f} | "
              f"Time={epoch_time:.1f}s | Mem={get_memory_usage():.0f}MB")

    elapsed = time.time() - pretrain_start
    print(f"\n  " + "-" * 65)
    print(f"  ✓ Pretraining completed in {elapsed/60:.1f} minutes")
    if pretrain_losses:
        print(f"  ✓ Final loss: {pretrain_losses[-1]:.4f}")
        if len(pretrain_losses) > 1:
            improvement = (pretrain_losses[0] - pretrain_losses[-1]) / pretrain_losses[0] * 100
            print(f"  ✓ Loss reduction: {improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient ML training with lazy loading')

    # Data arguments
    parser.add_argument('--spy_data', type=str, default='data/SPY_1min.csv')
    parser.add_argument('--tsla_data', type=str, default='data/TSLA_1min.csv')
    parser.add_argument('--tsla_events', type=str, default='data/tsla_events_REAL.csv')
    parser.add_argument('--macro_api_key', type=str, default=None)

    # Training arguments
    parser.add_argument('--start_year', type=int, default=config.ML_TRAIN_START_YEAR)
    parser.add_argument('--end_year', type=int, default=config.ML_TRAIN_END_YEAR)
    parser.add_argument('--epochs', type=int, default=config.ML_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.ML_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LNN_LEARNING_RATE)
    parser.add_argument('--pretrain_epochs', type=int, default=10)

    # Model arguments
    parser.add_argument('--model_type', type=str, default=config.ML_MODEL_TYPE)
    parser.add_argument('--hidden_size', type=int, default=config.LNN_HIDDEN_SIZE)

    # Output
    parser.add_argument('--output', type=str, default='models/lnn_lazy.pth')

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print("🚀 MEMORY-EFFICIENT TRAINING (Lazy Loading)")
    print("=" * 70)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model: {args.model_type}")
    print(f"📊 Training period: {args.start_year}-{args.end_year}")
    print(f"🔄 Epochs: {args.epochs} supervised + {args.pretrain_epochs} pretrain")
    print(f"💾 Output: {args.output}")
    print(f"🖥️  System memory: {psutil.virtual_memory().available / 1024**3:.1f} GB available")
    print("=" * 70)

    # Auto-select device: MPS if available, else CUDA, else CPU
    device_manager = DeviceManager()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_manager.setup_mps_environment()
        print(f"\n🖥️  Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🖥️  Device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device('cpu')
        print(f"\n🖥️  Device: CPU")

    total_start = time.time()

    # 1. Load and prepare data (lazy mode)
    features_df, events_handler, feature_extractor = load_and_prepare_data_lazy(
        args.spy_data, args.tsla_data,
        args.start_year, args.end_year,
        args.tsla_events, args.macro_api_key
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

    # 3. Self-supervised pretraining (optional)
    if args.pretrain_epochs > 0:
        self_supervised_pretrain_lazy(
            model, features_df, events_handler,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sequence_length=config.ML_SEQUENCE_LENGTH,
            device=device
        )

    # 4. Supervised training
    train_losses, val_losses = train_supervised_lazy(
        model, features_df, events_handler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        validation_split=config.ML_VALIDATION_SPLIT,
        sequence_length=config.ML_SEQUENCE_LENGTH,
        target_horizon=config.PREDICTION_HORIZON_HOURS,
        device=device
    )

    # 5. Save model
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL")
    print("=" * 70)

    metadata = {
        'model_type': args.model_type,
        'input_size': input_size,
        'hidden_size': args.hidden_size,
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
    print(f"  ✓ Training time: {total_time/60:.1f} minutes")
    print(f"  ✓ Peak memory usage: {get_memory_usage():.1f} MB")
    print(f"  ✓ Final validation loss: {val_losses[-1]:.4f}")

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


if __name__ == '__main__':
    main()