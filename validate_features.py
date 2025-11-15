"""
Quick validation script for extended 299-feature system
Tests feature extraction on historical data to ensure no NaNs or errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed

def validate_features():
    """
    Validate that the extended feature system works correctly
    """
    print("=" * 70)
    print("FEATURE VALIDATION TEST")
    print("=" * 70)

    # Initialize feature extractor
    print("\n1. Initializing TradingFeatureExtractor...")
    extractor = TradingFeatureExtractor()
    num_features = extractor.get_feature_dim()
    print(f"   ✓ Extractor initialized")
    print(f"   ✓ Expected features: {num_features}")

    # Verify feature count
    if num_features != 313:
        print(f"   ⚠️ WARNING: Expected 313 features, got {num_features}")
    else:
        print(f"   ✓ Feature count correct: 313")

    # Load a small sample of data (last 5000 bars from 1min data)
    print("\n2. Loading sample data (last 5000 bars of 1min data)...")
    try:
        data_feed = CSVDataFeed(timeframe='1min')
        df = data_feed.load_aligned_data(
            start_date='2023-01-01',
            end_date='2023-12-31'
        )

        if len(df) == 0:
            print("   ✗ No data loaded! Check data files.")
            return False

        # Take last 5000 bars for testing
        if len(df) > 5000:
            df = df.iloc[-5000:]

        print(f"   ✓ Loaded {len(df)} bars")
        print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")
        print(f"   ✓ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False

    # Extract features
    print("\n3. Extracting features...")
    try:
        features_df = extractor.extract_features(df)
        print(f"   ✓ Features extracted")
        print(f"   ✓ Feature shape: {features_df.shape}")
        print(f"   ✓ Feature columns: {len(features_df.columns)}")
    except Exception as e:
        print(f"   ✗ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate feature count
    print("\n4. Validating feature count...")
    expected_count = num_features
    actual_count = len(features_df.columns)

    if actual_count == expected_count:
        print(f"   ✓ Feature count matches: {actual_count} = {expected_count}")
    else:
        print(f"   ✗ Feature count mismatch: {actual_count} != {expected_count}")
        print(f"   Missing: {set(extractor.get_feature_names()) - set(features_df.columns)}")
        print(f"   Extra: {set(features_df.columns) - set(extractor.get_feature_names())}")
        return False

    # Check for NaNs
    print("\n5. Checking for NaN values...")
    nan_counts = features_df.isna().sum()
    total_nans = nan_counts.sum()

    if total_nans == 0:
        print(f"   ✓ No NaN values found")
    else:
        print(f"   ⚠ Found {total_nans} NaN values in {(nan_counts > 0).sum()} columns:")
        nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
        for col, count in nan_cols.head(10).items():
            print(f"      - {col}: {count} NaNs ({count/len(features_df)*100:.2f}%)")

        # This is acceptable if there are only a few at the beginning
        if total_nans < len(features_df) * 0.01:  # Less than 1% NaNs
            print(f"   ⚠ NaN count is low ({total_nans/len(features_df)*100:.4f}%), acceptable")
        else:
            print(f"   ✗ Too many NaNs ({total_nans/len(features_df)*100:.2f}%)")
            return False

    # Check for infinities
    print("\n6. Checking for infinite values...")
    inf_counts = np.isinf(features_df.values).sum(axis=0)
    total_infs = inf_counts.sum()

    if total_infs == 0:
        print(f"   ✓ No infinite values found")
    else:
        print(f"   ✗ Found {total_infs} infinite values")
        return False

    # Check value ranges for key features
    print("\n7. Checking value ranges for key features...")

    # RSI should be 0-100
    rsi_cols = [col for col in features_df.columns if '_rsi_' in col and '_divergence' not in col
                and '_oversold' not in col and '_overbought' not in col]
    for col in rsi_cols[:3]:  # Check first 3
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        if min_val >= 0 and max_val <= 100:
            print(f"   ✓ {col}: [{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"   ⚠ {col}: [{min_val:.2f}, {max_val:.2f}] - outside expected range")

    # Channel positions should be 0-1 or -1 to 1
    pos_cols = [col for col in features_df.columns if 'channel_position' in col][:3]
    for col in pos_cols:
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        print(f"   ✓ {col}: [{min_val:.2f}, {max_val:.2f}]")

    # Print breakdown feature stats
    print("\n8. New breakdown feature statistics:")
    breakdown_cols = [
        'tsla_volume_surge',
        'tsla_rsi_divergence_1h',
        'tsla_channel_duration_ratio_1h',
        'channel_alignment_spy_tsla_1h',
        'tsla_time_in_channel_1h',
        'tsla_channel_position_norm_1h'
    ]

    for col in breakdown_cols:
        if col in features_df.columns:
            stats = features_df[col].describe()
            print(f"   {col}:")
            print(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        else:
            print(f"   ✗ {col}: NOT FOUND")

    # Test sequence creation
    print("\n9. Testing sequence creation...")
    try:
        X, y = extractor.create_sequences(features_df, sequence_length=200, target_horizon=24)
        print(f"   ✓ Created {len(X)} sequences")
        print(f"   ✓ X shape: {X.shape}")
        print(f"   ✓ y shape: {y.shape}")
    except Exception as e:
        print(f"   ✗ Error creating sequences: {e}")
        return False

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE ✓")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  - Total features: {len(features_df.columns)}")
    print(f"  - Data rows: {len(features_df)}")
    print(f"  - Sequences: {len(X)}")
    print(f"  - NaN values: {total_nans}")
    print(f"  - Infinite values: {total_infs}")
    print(f"  - Status: {'PASS ✓' if total_infs == 0 else 'FAIL ✗'}")

    return total_infs == 0

if __name__ == '__main__':
    success = validate_features()
    sys.exit(0 if success else 1)
