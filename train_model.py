"""
train_model.py - Main training script for Stage 2 ML model

Trains Liquid Neural Network on 10 years of SPY/TSLA data
with event integration and self-supervised pretraining

Usage:
    python train_model.py --spy_data data/SPY_1min.csv --tsla_data data/TSLA_1min.csv \\
                          --tsla_events data/tsla_events.csv --epochs 50 --output models/lnn_model.pth
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
from src.ml.events import CombinedEventsHandler
from src.ml.model import LNNTradingModel, LSTMTradingModel, SelfSupervisedPretrainer
from src.ml.device_manager import DeviceManager
from src.ml.interactive_params import InteractiveParameterSelector, create_argparse_from_params


class TradingDataset(Dataset):
    """PyTorch Dataset for trading sequences"""

    def __init__(self, X, y, events_embeddings=None):
        self.X = X
        self.y = y
        self.events_embeddings = events_embeddings

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.events_embeddings is not None:
            return self.X[idx], self.y[idx], self.events_embeddings[idx]
        return self.X[idx], self.y[idx]


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_and_prepare_data(spy_file, tsla_file, start_year, end_year,
                          tsla_events_file=None, macro_api_key=None):
    """
    Load and prepare training data with alignment and feature extraction
    """
    print("\n" + "=" * 70)
    print("📊 LOADING AND PREPARING DATA")
    print("=" * 70)
    start_time = time.time()

    # 1. Load aligned SPY-TSLA data
    print(f"\n▶ Step 1/4: Loading SPY and TSLA data ({start_year} to {end_year})...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    data_feed = CSVDataFeed()
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    with tqdm(total=2, desc="  Loading data files", unit="file", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        pbar.set_description("  Loading SPY data")
        time.sleep(0.5)  # Give visual feedback
        pbar.update(1)
        pbar.set_description("  Loading TSLA data")
        aligned_df = data_feed.load_aligned_data(start_date, end_date)
        pbar.update(1)

    print(f"  ✓ Loaded {len(aligned_df):,} aligned 1-minute bars")
    print(f"  ✓ Date range: {aligned_df.index[0]} to {aligned_df.index[-1]}")

    # 2. Extract features
    print(f"\n▶ Step 2/4: Extracting features...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    feature_extractor = TradingFeatureExtractor()

    # Show progress for feature extraction
    feature_types = [
        ("Price features", 10),
        ("Channel features (multi-timeframe)", 25),
        ("RSI indicators", 15),
        ("Correlation features", 10),
        ("Cycle features", 10),
        ("Volume features", 5),
        ("Time features", 5)
    ]

    with tqdm(total=80, desc="  Extracting features", unit="%", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for feature_name, progress in feature_types:
            pbar.set_postfix_str(feature_name)
            time.sleep(0.2)  # Visual feedback
            pbar.update(progress)

    features_df = feature_extractor.extract_features(aligned_df)
    print(f"  ✓ Extracted {feature_extractor.get_feature_dim()} features")
    print(f"  ✓ Features: channels, RSI, correlations, cycles, volume, time")

    # 3. Load events
    print(f"\n▶ Step 3/4: Loading events data...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    events_handler = CombinedEventsHandler(tsla_events_file, macro_api_key)

    with tqdm(total=2, desc="  Loading events", unit="source", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        pbar.set_description("  Loading TSLA events")
        time.sleep(0.2)
        pbar.update(1)
        pbar.set_description("  Loading macro events")
        all_events = events_handler.load_events(start_date, end_date)
        pbar.update(1)

    print(f"  ✓ Loaded {len(all_events)} events (TSLA + macro)")

    # 4. Create sequences with event embeddings
    print(f"\n▶ Step 4/4: Creating sequences for training...")
    print(f"  Memory: {get_memory_usage():.1f} MB")

    sequence_length = config.ML_SEQUENCE_LENGTH  # 168 bars (1 week)
    target_horizon = config.PREDICTION_HORIZON_HOURS  # 24 hours

    X, y = feature_extractor.create_sequences(features_df, sequence_length, target_horizon)
    num_sequences = len(X)

    # Create event embeddings for each sequence with progress bar
    events_embeddings = []

    with tqdm(total=num_sequences, desc="  Creating event embeddings", unit="seq",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for i in range(num_sequences):
            # Get timestamp for this sequence
            seq_end_idx = i + sequence_length
            if seq_end_idx < len(features_df):
                seq_timestamp = features_df.index[seq_end_idx]

                # Get events around this timestamp
                events = events_handler.get_events_for_date(
                    str(seq_timestamp.date()),
                    lookback_days=config.EVENT_LOOKBACK_DAYS
                )

                # Embed events
                event_embed = events_handler.embed_events(events)
                events_embeddings.append(event_embed)
            else:
                # Default zero embedding
                events_embeddings.append(torch.zeros(1, 21))  # 10 + 11 dims

            # Update progress every 100 sequences
            if i % 100 == 0:
                pbar.update(min(100, num_sequences - i))

    events_embeddings = torch.cat(events_embeddings, dim=0)

    elapsed = time.time() - start_time
    print(f"\n  ✓ Created {len(X):,} sequences in {elapsed:.1f}s")
    print(f"  ✓ X shape: {X.shape} (batch, sequence, features)")
    print(f"  ✓ y shape: {y.shape} (batch, targets)")
    print(f"  ✓ Events shape: {events_embeddings.shape}")
    print(f"  ✓ Memory usage: {get_memory_usage():.1f} MB")

    return X, y, events_embeddings, feature_extractor


def self_supervised_pretrain(model, X, epochs=10, batch_size=32, lr=0.001, device=torch.device('cpu'),
                            num_workers=0, pin_memory=False):
    """
    Self-supervised pretraining using masking and reconstruction
    """
    print("\n" + "=" * 70)
    print("🔧 SELF-SUPERVISED PRETRAINING")
    print("=" * 70)
    print(f"  Learning to understand patterns via masked reconstruction")
    print(f"  Mask ratio: 15% | Learning rate: {lr}")
    print(f"  Device: {device}")
    start_time = time.time()

    pretrainer = SelfSupervisedPretrainer(model, mask_ratio=0.15)
    optimizer = torch.optim.Adam(list(model.parameters()) +
                                 list(pretrainer.reconstruction_head.parameters()), lr=lr)

    dataset = TradingDataset(X, torch.zeros(len(X), 2))  # Dummy y for pretraining
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=pin_memory)
    num_batches = len(dataloader)

    pretrain_losses = []

    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="  Pretraining", unit="epoch",
                      bar_format="{l_bar}{bar:20}{r_bar}")

    for epoch in epoch_pbar:
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0

        # Progress bar for batches
        batch_pbar = tqdm(dataloader, desc=f"    Epoch {epoch + 1}/{epochs}",
                         unit="batch", leave=False,
                         bar_format="{l_bar}{bar:20}{r_bar}")

        for batch_x, _ in batch_pbar:
            # Move batch to device
            batch_x = batch_x.to(device)

            loss = pretrainer.pretrain_step(batch_x, optimizer)
            total_loss += loss
            batch_count += 1

            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = total_loss / batch_count
        pretrain_losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # Update epoch progress bar
        epoch_pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'time': f'{epoch_time:.1f}s'})

        # Print epoch summary
        print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s")

    elapsed = time.time() - start_time
    print(f"\n  ✓ Pretraining completed in {elapsed:.1f}s")
    print(f"  ✓ Final loss: {pretrain_losses[-1]:.4f}")
    print(f"  ✓ Loss reduction: {(pretrain_losses[0] - pretrain_losses[-1]) / pretrain_losses[0] * 100:.1f}%")


def train_supervised(model, X, y, events_embeddings, epochs=50, batch_size=32,
                     lr=0.001, validation_split=0.1, device=torch.device('cpu'),
                     num_workers=0, pin_memory=False):
    """
    Supervised training on high/low predictions
    """
    print("\n" + "=" * 70)
    print("🎯 SUPERVISED TRAINING")
    print("=" * 70)
    print(f"  Device: {device}")
    start_time = time.time()

    # Split train/validation
    val_size = int(len(X) * validation_split)
    train_size = len(X) - val_size

    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    events_train, events_val = events_embeddings[:train_size], events_embeddings[train_size:]

    print(f"  Train size: {train_size:,} sequences")
    print(f"  Validation size: {val_size:,} sequences")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial learning rate: {lr}")

    # Create datasets
    train_dataset = TradingDataset(X_train, y_train, events_train)
    val_dataset = TradingDataset(X_val, y_val, events_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    lr_history = []

    print(f"\n  Starting training...")
    print(f"  " + "-" * 65)

    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="  Training progress", unit="epoch",
                      bar_format="{l_bar}{bar:25}{r_bar}")

    for epoch in epoch_pbar:
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # ========== Training Phase ==========
        model.train()
        train_loss = 0
        train_batches = 0

        # Training batch progress bar
        train_pbar = tqdm(train_loader, desc=f"    Training", unit="batch",
                         leave=False, bar_format="{l_bar}{bar:20}{r_bar}")

        for batch_x, batch_y, _ in train_pbar:
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

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_batches

        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0
        val_batches = 0

        val_pbar = tqdm(val_loader, desc=f"    Validating", unit="batch",
                       leave=False, bar_format="{l_bar}{bar:20}{r_bar}")

        with torch.no_grad():
            for batch_x, batch_y, _ in val_pbar:
                # Move batches to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

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

        # Update main progress bar
        epoch_pbar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{avg_val_loss:.4f}',
            'lr': f'{new_lr:.1e}',
            'time': f'{epoch_time:.1f}s'
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
              f"LR={new_lr:.1e} | Time={epoch_time:.1f}s{status}")

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n  " + "-" * 65)
    print(f"  ✓ Training completed in {elapsed/60:.1f} minutes")
    print(f"  ✓ Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  ✓ Final train/val loss: {train_losses[-1]:.4f}/{val_losses[-1]:.4f}")
    print(f"  ✓ Learning rate decay: {lr_history[0]:.1e} → {lr_history[-1]:.1e}")

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train Stage 2 ML model')

    # Data arguments
    parser.add_argument('--spy_data', type=str, default='data/SPY_1min.csv',
                       help='Path to SPY 1-minute data CSV')
    parser.add_argument('--tsla_data', type=str, default='data/TSLA_1min.csv',
                       help='Path to TSLA 1-minute data CSV')
    parser.add_argument('--tsla_events', type=str, default=None,
                       help='Path to TSLA events CSV (earnings, deliveries)')
    parser.add_argument('--macro_api_key', type=str, default=None,
                       help='API key for macro events (optional)')

    # Training arguments
    parser.add_argument('--start_year', type=int, default=config.ML_TRAIN_START_YEAR,
                       help='Training start year (default: 2015)')
    parser.add_argument('--end_year', type=int, default=config.ML_TRAIN_END_YEAR,
                       help='Training end year (default: 2023)')
    parser.add_argument('--epochs', type=int, default=config.ML_EPOCHS,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=config.ML_BATCH_SIZE,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=config.LNN_LEARNING_RATE,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                       help='Self-supervised pretraining epochs (default: 10)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default=config.ML_MODEL_TYPE,
                       choices=['LNN', 'LSTM'],
                       help='Model type (default: LNN)')
    parser.add_argument('--hidden_size', type=int, default=config.LNN_HIDDEN_SIZE,
                       help='Hidden size (default: 128)')

    # Output
    parser.add_argument('--output', type=str, default='models/lnn_model.pth',
                       help='Output model path (default: models/lnn_model.pth)')

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

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive parameter selection mode')

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 70)
        print("🎛️  LAUNCHING INTERACTIVE MODE")
        print("=" * 70)
        selector = InteractiveParameterSelector(mode='standard')
        params = selector.run()
        args = create_argparse_from_params(params, args)
        print("\n✓ Configuration complete! Starting training...\n")

    # Print header with system info
    print("\n" + "=" * 70)
    print("🚀 STAGE 2: ML MODEL TRAINING")
    print("=" * 70)
    print(f"📅 Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Model: {args.model_type} (Liquid Neural Network)" if args.model_type == "LNN" else f"🤖 Model: {args.model_type}")
    print(f"📊 Training period: {args.start_year}-{args.end_year}")
    print(f"🔄 Epochs: {args.epochs} supervised + {args.pretrain_epochs} pretrain")
    print(f"💾 Output: {args.output}")
    print(f"🖥️  System memory: {psutil.virtual_memory().available / 1024**3:.1f} GB available")
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

    # 1. Load and prepare data
    X, y, events_embeddings, feature_extractor = load_and_prepare_data(
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
        print("  Creating Liquid Neural Network with Closed-form Continuous-time (CfC)")
        model = LNNTradingModel(input_size, args.hidden_size)
    else:
        print("  Creating LSTM model with attention mechanism")
        model = LSTMTradingModel(input_size, args.hidden_size)

    # Move model to device
    model = device_manager.move_to_device(model, device, verbose=True)
    print(f"  ✓ Model moved to {device}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model type: {args.model_type}")
    print(f"  Input features: {input_size}")
    print(f"  Hidden units: {args.hidden_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB")

    # GPU optimization: Auto-detect num_workers and pin_memory based on device
    if args.num_workers is None:
        num_workers = 2 if device.type == 'cuda' else 0
        workers_source = "auto-detected"
    else:
        num_workers = args.num_workers
        workers_source = "user-specified"

    if args.pin_memory is None:
        pin_memory = (device.type == 'cuda')
        pin_source = "auto-detected"
    else:
        pin_memory = args.pin_memory
        pin_source = "user-specified"

    # Display GPU optimization settings
    if device.type == 'cuda':
        print(f"\n  🚀 GPU optimizations enabled:")
        print(f"     - num_workers: {num_workers} ({workers_source}, parallel data loading)")
        print(f"     - pin_memory: {pin_memory} ({pin_source}, faster GPU transfers)")
        if num_workers != 2 and workers_source == "user-specified":
            print(f"     ⚠️  Note: Default for CUDA is 2 workers, you selected {num_workers}")
        print(f"     💡 Tip: Use --batch_size 128 for maximum GPU performance!")
    else:
        print(f"\n  ℹ️  CPU/MPS mode:")
        print(f"     - num_workers: {num_workers} ({workers_source})")
        print(f"     - pin_memory: {pin_memory} ({pin_source})")
        if num_workers > 0 and device.type == 'mps':
            print(f"     ⚠️  Warning: num_workers>0 may cause issues on MPS, consider using 0")

    # 3. Self-supervised pretraining
    if args.pretrain_epochs > 0:
        self_supervised_pretrain(model, X, epochs=args.pretrain_epochs,
                                batch_size=args.batch_size, lr=args.lr, device=device,
                                num_workers=num_workers, pin_memory=pin_memory)

    # 4. Supervised training
    train_losses, val_losses = train_supervised(
        model, X, y, events_embeddings,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, validation_split=config.ML_VALIDATION_SPLIT, device=device,
        num_workers=num_workers, pin_memory=pin_memory
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
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'training_date': datetime.now().isoformat(),
        'feature_names': feature_extractor.get_feature_names(),
        'total_sequences': len(X),
        'training_time_minutes': (time.time() - total_start) / 60,
        'device': str(device),
        'device_type': device.type
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(args.output, metadata)

    # Print final summary
    total_time = time.time() - total_start
    print(f"  ✓ Model saved to: {args.output}")
    print(f"  ✓ File size: {Path(args.output).stat().st_size / 1024**2:.1f} MB")
    print(f"  ✓ Training time: {total_time/60:.1f} minutes")
    print(f"  ✓ Final validation loss: {val_losses[-1]:.4f}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"\n📈 Next steps:")
    print(f"   1. Run backtesting: python backtest.py --model_path {args.output}")
    print(f"   2. Validate results: python validate_results.py --model_path {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
