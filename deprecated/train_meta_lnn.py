#!/usr/bin/env python3
"""
Train Meta-LNN coach model on historical sub-model predictions.

This script loads predictions from all timeframe-specific sub-models
and trains a meta-learner to combine them adaptively based on
market conditions.

Usage:
    # Train on backtest predictions (news disabled)
    python train_meta_lnn.py --mode backtest_no_news

    # Later: Fine-tune with news (after collecting news data)
    python train_meta_lnn.py --mode live_with_news --resume models/meta_lnn.pth
"""

import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ml.meta_models import (
    MetaLNN,
    MetaLNNWithModalityDropout,
    MetaFTTransformer,
    calculate_market_state,
    meta_loss
)
from src.ml.events import CombinedEventsHandler
import config


class MetaTrainingDataset(Dataset):
    """
    Dataset for training meta-model from database predictions.

    Loads historical predictions from all sub-models and prepares
    inputs for meta-learner training.
    """

    def __init__(self,
                 predictions_df: pd.DataFrame,
                 features_df: pd.DataFrame,
                 events_handler: CombinedEventsHandler,
                 mode='backtest_no_news',
                 news_vec_dim=768):
        """
        Initialize dataset.

        Args:
            predictions_df: Pivoted predictions (one row per timestamp, columns per model)
            features_df: Market data features for calculating market state
            events_handler: For event-based features
            mode: 'backtest_no_news' or 'live_with_news'
            news_vec_dim: Dimension of news embeddings
        """
        self.predictions_df = predictions_df
        self.features_df = features_df
        self.events_handler = events_handler
        self.mode = mode
        self.news_vec_dim = news_vec_dim

        # Validate data
        assert len(predictions_df) > 0, "No predictions in dataset"
        assert 'actual_high' in predictions_df.columns, "Missing actuals in dataset"

    def __len__(self):
        return len(self.predictions_df)

    def __getitem__(self, idx):
        """
        Get training sample.

        Returns:
            subpreds: [num_submodels, 3] - Sub-model predictions
            market_state: [12] - Market regime features
            news_vec: [news_vec_dim] - News embeddings (zeros for backtest mode)
            news_mask: [1] - News availability flag
            y_high: Actual high
            y_low: Actual low
        """
        row = self.predictions_df.iloc[idx]

        # Extract sub-model predictions
        # Expected columns: {timeframe}_{metric} e.g., 15min_high, 15min_low, 15min_conf
        subpreds_list = []
        for tf in ['15min', '1hour', '4hour', 'daily']:
            high = row.get(f'{tf}_high', 0.0)
            low = row.get(f'{tf}_low', 0.0)
            conf = row.get(f'{tf}_conf', 0.5)
            subpreds_list.append([high, low, conf])

        subpreds = torch.tensor(subpreds_list, dtype=torch.float32)

        # Calculate market state
        timestamp = row.name  # Index should be timestamp
        try:
            # Find corresponding index in features_df
            feat_idx = self.features_df.index.get_loc(timestamp, method='nearest')
            market_state = calculate_market_state(
                self.features_df,
                feat_idx,
                self.events_handler
            )
        except:
            # Fallback: zeros if timestamp not found
            market_state = torch.zeros(12, dtype=torch.float32)

        # News features (disabled for backtest mode)
        if self.mode == 'backtest_no_news':
            news_vec = torch.zeros(self.news_vec_dim, dtype=torch.float32)
            news_mask = torch.tensor([0.0], dtype=torch.float32)
        else:
            # TODO: Load news embeddings from news.db
            news_vec = torch.zeros(self.news_vec_dim, dtype=torch.float32)
            news_mask = torch.tensor([0.0], dtype=torch.float32)

        # Targets (convert actual prices to percentages)
        # Predictions are already in percentage terms, so targets must be too
        actual_high = row['actual_high']
        actual_low = row['actual_low']
        current_price = row.get('current_price', row.get('tsla_close', 250.0))  # Fallback to reasonable default

        # Convert to percentage changes from current price
        actual_high_pct = (actual_high - current_price) / current_price * 100
        actual_low_pct = (actual_low - current_price) / current_price * 100

        y_high = torch.tensor([actual_high_pct], dtype=torch.float32)
        y_low = torch.tensor([actual_low_pct], dtype=torch.float32)

        return subpreds, market_state, news_vec, news_mask, y_high, y_low


def purged_kfold_cv(timestamps: pd.DatetimeIndex,
                    n_splits=5,
                    embargo_days=7) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Time-series cross-validation with purging and embargo.

    Based on López de Prado's approach to prevent leakage:
    - Purging: Remove samples close to validation set
    - Embargo: Add gap between train and validation

    Args:
        timestamps: Sorted datetime index
        n_splits: Number of CV folds
        embargo_days: Days of embargo between train and val

    Returns:
        List of (train_indices, val_indices) tuples
    """
    n_samples = len(timestamps)
    fold_size = n_samples // n_splits
    embargo_td = timedelta(days=embargo_days)

    splits = []

    for i in range(n_splits):
        # Validation set: fold i
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples

        val_idx = np.arange(val_start, val_end)
        val_timestamps = timestamps[val_idx]

        # Training set: all data before validation (with embargo)
        train_end_time = val_timestamps[0] - embargo_td
        train_idx = np.where(timestamps < train_end_time)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def load_predictions_from_db(db_path: str,
                             timeframes: List[str]) -> pd.DataFrame:
    """
    Load predictions from database and pivot to wide format.

    Args:
        db_path: Path to predictions.db
        timeframes: List of timeframes to load (e.g., ['15min', '1hour', '4hour', 'daily'])

    Returns:
        DataFrame with one row per timestamp, columns for each model's predictions
    """
    conn = sqlite3.connect(db_path)

    # Load all predictions with actuals
    query = """
        SELECT
            timestamp,
            target_timestamp,
            simulation_date,
            model_timeframe,
            predicted_high,
            predicted_low,
            confidence,
            actual_high,
            actual_low
        FROM predictions
        WHERE actual_high IS NOT NULL
          AND actual_low IS NOT NULL
          AND model_timeframe IN ({})
          AND simulation_date IS NOT NULL
        ORDER BY simulation_date
    """.format(','.join(['?'] * len(timeframes)))

    df = pd.read_sql(query, conn, params=timeframes, parse_dates=['timestamp', 'target_timestamp', 'simulation_date'])
    conn.close()

    if len(df) == 0:
        raise ValueError(f"No predictions found in database for timeframes: {timeframes}")

    print(f"Loaded {len(df)} prediction records from database")
    print(f"  Timeframes: {df['model_timeframe'].unique()}")
    print(f"  Simulation date range: {df['simulation_date'].min()} to {df['simulation_date'].max()}")

    # Use simulation_date for alignment (all models tested same historical date)
    df['sim_date'] = df['simulation_date'].dt.normalize()

    # Pivot to wide format by sim_date (groups all models that tested the same historical date)
    pivot_df = df.pivot_table(
        index='sim_date',
        columns='model_timeframe',
        values=['predicted_high', 'predicted_low', 'confidence'],
        aggfunc='first'  # Use first if duplicates (shouldn't happen)
    )

    # Flatten column names: (metric, timeframe) → timeframe_metric
    pivot_df.columns = [f'{col[1]}_{col[0].replace("predicted_", "")}' for col in pivot_df.columns]

    # Add actuals (same across all timeframes, use any one)
    actuals_df = df[df['model_timeframe'] == timeframes[0]][['sim_date', 'actual_high', 'actual_low']]
    actuals_df = actuals_df.set_index('sim_date')
    actuals_df = actuals_df[~actuals_df.index.duplicated(keep='first')]  # Keep first if multiple on same date

    pivot_df = pivot_df.join(actuals_df, how='inner')

    # Drop rows with missing predictions
    pivot_df = pivot_df.dropna()

    print(f"Pivoted to {len(pivot_df)} complete samples")
    print(f"  Columns: {list(pivot_df.columns)}")

    return pivot_df


def train_meta_model(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     epochs: int,
                     lr: float,
                     device: torch.device):
    """
    Train meta-model.

    Args:
        model: MetaLNN or MetaFTTransformer
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device (cpu/cuda/mps)

    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for subpreds, market_state, news_vec, news_mask, y_high, y_low in train_loader:
            # Move to device
            subpreds = subpreds.to(device)
            market_state = market_state.to(device)
            news_vec = news_vec.to(device)
            news_mask = news_mask.to(device)
            y_high = y_high.to(device)
            y_low = y_low.to(device)

            # Forward pass
            pred_high, pred_low, pred_conf, _ = model(subpreds, market_state, news_vec, news_mask)

            # Calculate loss
            loss = meta_loss(pred_high.squeeze(), pred_low.squeeze(), pred_conf.squeeze(),
                           y_high.squeeze(), y_low.squeeze())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for subpreds, market_state, news_vec, news_mask, y_high, y_low in val_loader:
                subpreds = subpreds.to(device)
                market_state = market_state.to(device)
                news_vec = news_vec.to(device)
                news_mask = news_mask.to(device)
                y_high = y_high.to(device)
                y_low = y_low.to(device)

                pred_high, pred_low, pred_conf, _ = model(subpreds, market_state, news_vec, news_mask)

                loss = meta_loss(pred_high.squeeze(), pred_low.squeeze(), pred_conf.squeeze(),
                               y_high.squeeze(), y_low.squeeze())

                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train Meta-LNN coach model')

    # Data arguments
    parser.add_argument('--db_path', type=str, default='data/predictions.db',
                       help='Path to predictions database')
    parser.add_argument('--features_csv', type=str, default='data/TSLA_1min.csv',
                       help='Path to features CSV for market state calculation')
    parser.add_argument('--events_csv', type=str, default='data/tsla_events_REAL.csv',
                       help='Path to events CSV')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='meta_lnn',
                       choices=['meta_lnn', 'meta_lnn_dropout', 'meta_transformer'],
                       help='Meta-model architecture')
    parser.add_argument('--mode', type=str, default='backtest_no_news',
                       choices=['backtest_no_news', 'live_with_news'],
                       help='Training mode')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of CV splits')
    parser.add_argument('--embargo_days', type=int, default=7,
                       help='Embargo days for CV')

    # Output
    parser.add_argument('--output', type=str, default='models/meta_lnn.pth',
                       help='Output path for trained model')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device for training')

    args = parser.parse_args()

    print("=" * 70)
    print("META-LNN COACH TRAINING")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_type}")
    print(f"Database: {args.db_path}")
    print(f"Device: {args.device}")
    print()

    # Load predictions
    print("Loading predictions from database...")
    timeframes = ['15min', '1hour', '4hour', 'daily']
    predictions_df = load_predictions_from_db(args.db_path, timeframes)

    # Load features for market state calculation
    print("\nLoading features for market state calculation...")
    features_df = pd.read_csv(args.features_csv, parse_dates=['timestamp'], index_col='timestamp')
    print(f"  Loaded {len(features_df)} feature bars")

    # Load events
    print("\nLoading events...")
    events_handler = CombinedEventsHandler(args.events_csv)

    # Create dataset
    print("\nCreating dataset...")
    full_dataset = MetaTrainingDataset(
        predictions_df,
        features_df,
        events_handler,
        mode=args.mode
    )

    # Purged K-Fold CV
    print(f"\nSetting up {args.n_splits}-fold CV with {args.embargo_days} days embargo...")
    cv_splits = purged_kfold_cv(
        predictions_df.index,
        n_splits=args.n_splits,
        embargo_days=args.embargo_days
    )

    print(f"  Generated {len(cv_splits)} valid splits")

    # Device
    device = torch.device(args.device)

    # Train on each fold
    all_val_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_idx + 1}/{len(cv_splits)}")
        print("=" * 70)
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Val samples: {len(val_idx)}")

        # Create data loaders
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        # Create model
        if args.model_type == 'meta_lnn':
            model = MetaLNN(num_submodels=4, market_state_dim=12, news_vec_dim=768, hidden_size=64)
        elif args.model_type == 'meta_lnn_dropout':
            model = MetaLNNWithModalityDropout(num_submodels=4, market_state_dim=12, news_vec_dim=768, hidden_size=64, dropout_prob=0.4)
        else:  # meta_transformer
            model = MetaFTTransformer(num_submodels=4, market_state_dim=12, news_vec_dim=768)

        # Train
        model, train_losses, val_losses = train_meta_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, device=device
        )

        all_val_losses.append(val_losses[-1])
        print(f"\nFold {fold_idx + 1} final val loss: {val_losses[-1]:.6f}")

    # Train final model on all data
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 70)

    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == 'meta_lnn':
        final_model = MetaLNN(num_submodels=4, market_state_dim=12, news_vec_dim=768, hidden_size=64)
    elif args.model_type == 'meta_lnn_dropout':
        final_model = MetaLNNWithModalityDropout(num_submodels=4, market_state_dim=12, news_vec_dim=768, hidden_size=64, dropout_prob=0.4)
    else:
        final_model = MetaFTTransformer(num_submodels=4, market_state_dim=12, news_vec_dim=768)

    final_model, _, _ = train_meta_model(
        final_model, full_loader, full_loader,  # Use full data for both train and val
        epochs=args.epochs, lr=args.lr, device=device
    )

    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': final_model.state_dict(),
        'model_type': args.model_type,
        'mode': args.mode,
        'num_submodels': 4,
        'market_state_dim': 12,
        'news_vec_dim': 768,
        'timeframes': timeframes,
        'cv_val_losses': all_val_losses,
        'avg_cv_val_loss': np.mean(all_val_losses),
        'training_date': datetime.now().isoformat(),
        'n_training_samples': len(full_dataset),
        'embargo_days': args.embargo_days
    }

    torch.save(checkpoint, args.output)

    print(f"  ✓ Model saved to: {args.output}")
    print(f"  Average CV validation loss: {np.mean(all_val_losses):.6f}")
    print(f"  Training samples: {len(full_dataset)}")

    print("\n" + "=" * 70)
    print("✅ META-LNN TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
