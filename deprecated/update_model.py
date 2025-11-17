"""
update_model.py - Online learning update script

Incrementally updates model weights based on recent prediction errors
Uses logged predictions from database to identify flopped patterns

Usage:
    python update_model.py --model_path models/lnn_model.pth \\
                           --new_data data/recent_data.csv --db_path data/predictions.db \\
                           --output models/updated_model.pth
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.ml.data_feed import CSVDataFeed
from src.ml.features import TradingFeatureExtractor
from src.ml.model import LNNTradingModel, LSTMTradingModel
from src.ml.database import SQLitePredictionDB


def load_model(model_path):
    """Load model from checkpoint"""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    metadata = checkpoint.get('metadata', {})

    model_type = metadata.get('model_type', 'LNN')
    input_size = metadata['input_size']
    hidden_size = metadata.get('hidden_size', config.LNN_HIDDEN_SIZE)

    if model_type == 'LNN':
        model = LNNTradingModel(input_size, hidden_size)
    else:
        model = LSTMTradingModel(input_size, hidden_size)

    model.load_checkpoint(model_path)

    print(f"Model loaded: {model_type}")
    return model, metadata


def get_error_samples(db, error_threshold=10.0, limit=100):
    """
    Get samples with high errors for focused learning
    Returns DataFrame of worst predictions
    """
    print(f"\nFetching high-error predictions (error > {error_threshold}%)...")

    error_df = db.get_error_patterns(limit=limit)

    if error_df.empty:
        print("No error samples found in database")
        return None

    # Filter by error threshold
    high_error_df = error_df[error_df['absolute_error'] > error_threshold]

    print(f"Found {len(high_error_df)} high-error samples (>{error_threshold}% error)")

    if not high_error_df.empty:
        print(f"  Mean error: {high_error_df['absolute_error'].mean():.2f}%")
        print(f"  Max error: {high_error_df['absolute_error'].max():.2f}%")

    return high_error_df


def prepare_update_data(error_df, data_feed, feature_extractor):
    """
    Prepare training data from high-error samples
    Re-extracts features and targets for those specific dates
    """
    print("\nPreparing update data from error samples...")

    X_update = []
    y_update = []

    for idx, row in error_df.iterrows():
        try:
            # Get context data for this prediction
            target_date = pd.to_datetime(row['timestamp'])
            context_start = target_date - timedelta(days=7)

            aligned_df = data_feed.load_aligned_data(
                context_start.strftime('%Y-%m-%d'),
                target_date.strftime('%Y-%m-%d')
            )

            if len(aligned_df) < config.ML_SEQUENCE_LENGTH:
                continue

            # Extract features
            features_df = feature_extractor.extract_features(aligned_df)

            # Get sequence
            sequence = features_df.tail(config.ML_SEQUENCE_LENGTH).values

            # Target: convert actual prices to percentage changes
            # Get current price (last price in sequence)
            current_price = aligned_df['tsla_close'].iloc[-1]

            # Convert actual prices to percentage changes from current price
            actual_high_pct = (row['actual_high'] - current_price) / current_price * 100
            actual_low_pct = (row['actual_low'] - current_price) / current_price * 100

            target = [actual_high_pct, actual_low_pct]

            X_update.append(sequence)
            y_update.append(target)

        except Exception as e:
            print(f"  ⚠ Error processing sample {idx}: {e}")
            continue

    if X_update:
        X_update = torch.tensor(np.array(X_update), dtype=torch.float32)
        y_update = torch.tensor(np.array(y_update), dtype=torch.float32)

        print(f"Prepared {len(X_update)} update samples")
        return X_update, y_update
    else:
        print("No valid update samples prepared")
        return None, None


def online_update(model, X, y, lr=0.0001, epochs=5):
    """
    Perform online learning updates on error samples
    Uses lower learning rate to avoid catastrophic forgetting
    """
    print(f"\nPerforming online learning updates...")
    print(f"  Learning rate: {lr}")
    print(f"  Update epochs: {epochs}")

    initial_loss = None
    final_loss = None

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            x_sample = X[i:i+1]  # Single sample
            y_sample = y[i:i+1]

            loss = model.update_online(x_sample, y_sample, lr=lr)
            total_loss += loss

        avg_loss = total_loss / len(X)

        if epoch == 0:
            initial_loss = avg_loss
        if epoch == epochs - 1:
            final_loss = avg_loss

        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    print(f"\nUpdate completed:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Online learning update for Stage 2 ML model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint to update')
    parser.add_argument('--db_path', type=str, default=str(config.ML_DB_PATH),
                       help='Path to prediction database')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save updated model')

    parser.add_argument('--error_threshold', type=float, default=10.0,
                       help='Error threshold for selecting samples (default: 10.0%)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of error samples to use (default: 100)')
    parser.add_argument('--lr', type=float, default=config.ONLINE_LEARNING_LR,
                       help='Learning rate for updates (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of update epochs (default: 5)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("STAGE 2: ONLINE LEARNING UPDATE")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # 1. Load model
    model, metadata = load_model(args.model_path)

    # 2. Initialize components
    print("\nInitializing components...")
    db = SQLitePredictionDB(args.db_path)
    data_feed = CSVDataFeed()
    feature_extractor = TradingFeatureExtractor()

    # 3. Get high-error samples
    error_df = get_error_samples(db, args.error_threshold, args.max_samples)

    if error_df is None or error_df.empty:
        print("\n⚠ No error samples available for update. Exiting.")
        return

    # 4. Prepare update data
    X_update, y_update = prepare_update_data(error_df, data_feed, feature_extractor)

    if X_update is None:
        print("\n⚠ Failed to prepare update data. Exiting.")
        return

    # 5. Perform online learning
    online_update(model, X_update, y_update, lr=args.lr, epochs=args.epochs)

    # 6. Save updated model
    print("\n" + "=" * 70)
    print("SAVING UPDATED MODEL")
    print("=" * 70)

    # Update metadata
    metadata['last_update'] = datetime.now().isoformat()
    metadata['update_samples'] = len(X_update)
    metadata['update_error_threshold'] = args.error_threshold
    metadata['update_lr'] = args.lr
    metadata['update_epochs'] = args.epochs

    if 'update_count' in metadata:
        metadata['update_count'] += 1
    else:
        metadata['update_count'] = 1

    model.save_checkpoint(args.output, metadata)

    print(f"\nUpdated model saved to: {args.output}")
    print(f"Update count: {metadata['update_count']}")
    print(f"Samples used: {len(X_update)}")
    print(f"\nUpdate completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
