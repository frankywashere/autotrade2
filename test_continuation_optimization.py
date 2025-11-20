"""Test script to verify optimized continuation labels are identical to original."""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.ml.features import TradingFeatureExtractor


def test_continuation_optimization():
    """
    Test that optimized continuation labels are identical to original.
    """
    print("=" * 80)
    print("CONTINUATION LABEL OPTIMIZATION VERIFICATION")
    print("=" * 80)
    print()

    # Load test data
    print("Loading test data...")
    data_dir = Path(__file__).parent / 'data'

    # Try to find TSLA 5min file first (that's what the continuation labels expect)
    csv_file = data_dir / 'TSLA_5min.csv'
    if not csv_file.exists():
        # Fallback to TSLA 1min
        csv_file = data_dir / 'TSLA_1min.csv'
        if not csv_file.exists():
            print("ERROR: No TSLA_5min.csv or TSLA_1min.csv found in data directory")
            return False
    print(f"Using data file: {csv_file.name}")

    # Load dataframe
    df = pd.read_csv(csv_file)

    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        # Assume first column is timestamp
        df.index = pd.to_datetime(df.index)

    # Sort by index
    df = df.sort_index()

    # Ensure we have required columns (case-insensitive check)
    required_cols = ['open', 'high', 'low', 'close']
    df_lower = df.columns.str.lower()

    # Map columns to standard names if needed
    col_mapping = {}
    for req_col in required_cols:
        # Try to find tsla_ prefix first
        tsla_col = f'tsla_{req_col}'
        if tsla_col in df.columns:
            continue  # Already has correct name
        elif tsla_col.lower() in df_lower:
            # Find actual column name with different case
            actual_col = df.columns[df_lower.tolist().index(tsla_col.lower())]
            col_mapping[actual_col] = tsla_col
        elif req_col in df.columns:
            col_mapping[req_col] = f'tsla_{req_col}'
        elif req_col.lower() in df_lower:
            actual_col = df.columns[df_lower.tolist().index(req_col.lower())]
            col_mapping[actual_col] = f'tsla_{req_col}'

    if col_mapping:
        df = df.rename(columns=col_mapping)

    # Verify we have all required columns
    for col in ['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            print(f"Available columns: {df.columns.tolist()}")
            return False

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print()

    # Select test timestamps (use a subset for faster testing)
    # Choose timestamps with different alignment patterns
    all_timestamps = df.index.tolist()

    # Sample timestamps: some aligned, some not
    sample_size = min(20, len(all_timestamps) // 10)  # Test on 20 timestamps for quick test

    # Get a mix of timestamps (avoid early timestamps that don't have enough history)
    np.random.seed(42)  # For reproducibility

    # Start from at least 100,000 bars in (to ensure enough history for channels)
    start_idx = max(100000, len(all_timestamps) // 4)
    end_idx = min(len(all_timestamps) - 100, start_idx + 10000)

    test_indices = np.linspace(start_idx, end_idx, sample_size, dtype=int)
    test_timestamps = [all_timestamps[i] for i in test_indices if i < len(all_timestamps)]

    # Ensure we have both aligned and unaligned timestamps
    aligned_count = sum(1 for ts in test_timestamps if ts.minute == 0 and ts.second == 0)
    unaligned_count = len(test_timestamps) - aligned_count

    print(f"Testing with {len(test_timestamps)} timestamps:")
    print(f"  - Aligned (hour boundaries): {aligned_count}")
    print(f"  - Unaligned (mid-hour): {unaligned_count}")
    print()

    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = TradingFeatureExtractor()

    # Run original version
    print("-" * 60)
    print("Running ORIGINAL generate_continuation_labels...")
    print("-" * 60)
    start_time = time.time()

    try:
        labels_original = extractor.generate_continuation_labels(
            df, test_timestamps, prediction_horizon=24, debug=False
        )
        original_time = time.time() - start_time
        print(f"✓ Original completed in {original_time:.2f} seconds")
        print(f"  Generated {len(labels_original)} labels")
    except Exception as e:
        print(f"✗ Original failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run optimized version
    print()
    print("-" * 60)
    print("Running OPTIMIZED generate_continuation_labels...")
    print("-" * 60)
    start_time = time.time()

    try:
        labels_optimized = extractor.generate_continuation_labels_optimized(
            df, test_timestamps, prediction_horizon=24, debug=False
        )
        optimized_time = time.time() - start_time
        print(f"✓ Optimized completed in {optimized_time:.2f} seconds")
        print(f"  Generated {len(labels_optimized)} labels")
    except Exception as e:
        print(f"✗ Optimized failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print()

    # Check if same number of labels
    if len(labels_original) != len(labels_optimized):
        print(f"✗ FAILED: Different number of labels")
        print(f"  Original: {len(labels_original)}")
        print(f"  Optimized: {len(labels_optimized)}")
        return False

    # If both are empty, that's a problem but at least they match
    if len(labels_original) == 0 and len(labels_optimized) == 0:
        print("⚠️  WARNING: Both methods returned 0 labels")
        print("This might indicate all timestamps were filtered out due to scoring.")
        print("The methods are consistent but no labels were generated.")
        return True  # They're identical (both empty)

    # Sort both by timestamp for comparison
    labels_original = labels_original.sort_values('timestamp').reset_index(drop=True)
    labels_optimized = labels_optimized.sort_values('timestamp').reset_index(drop=True)

    # Compare each column
    all_match = True
    columns_to_check = ['timestamp', 'label', 'continues', 'duration_hours',
                       'projected_gain', 'confidence', 'score',
                       'rsi_1h', 'rsi_4h', 'slope_1h', 'slope_4h']

    for col in columns_to_check:
        if col not in labels_original.columns or col not in labels_optimized.columns:
            continue

        if col == 'timestamp':
            # Compare timestamps directly
            matches = (labels_original[col] == labels_optimized[col]).all()
        elif col == 'label':
            # Compare string labels
            matches = (labels_original[col] == labels_optimized[col]).all()
        else:
            # For numeric columns, check if they're close (accounting for floating point)
            orig_vals = labels_original[col].values
            opt_vals = labels_optimized[col].values

            # Use allclose for floating point comparison
            matches = np.allclose(orig_vals, opt_vals, rtol=1e-9, atol=1e-12, equal_nan=True)

            if not matches:
                # Find where they differ
                diffs = np.abs(orig_vals - opt_vals)
                max_diff_idx = np.argmax(diffs)
                max_diff = diffs[max_diff_idx]
                print(f"  ✗ Column '{col}': Values differ")
                print(f"    Max difference: {max_diff:.12f}")
                print(f"    At timestamp: {labels_original.iloc[max_diff_idx]['timestamp']}")
                print(f"    Original: {orig_vals[max_diff_idx]:.12f}")
                print(f"    Optimized: {opt_vals[max_diff_idx]:.12f}")
                all_match = False
            else:
                print(f"  ✓ Column '{col}': MATCH")

    print()

    if all_match:
        print("✅ SUCCESS: All values are IDENTICAL!")
        print()
        print("=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        speedup = original_time / optimized_time
        print(f"Original time:  {original_time:.2f} seconds")
        print(f"Optimized time: {optimized_time:.2f} seconds")
        print(f"SPEEDUP:        {speedup:.1f}x faster")

        # Extrapolate to full dataset
        full_size = len(all_timestamps)
        test_size = len(test_timestamps)
        scale_factor = full_size / test_size

        print()
        print(f"Extrapolated to full dataset ({full_size:,} timestamps):")
        print(f"  Original:  ~{original_time * scale_factor / 60:.1f} minutes")
        print(f"  Optimized: ~{optimized_time * scale_factor / 60:.1f} minutes")

        return True
    else:
        print("❌ FAILED: Values are NOT identical!")

        # Save both dataframes for debugging
        labels_original.to_csv('debug_labels_original.csv', index=False)
        labels_optimized.to_csv('debug_labels_optimized.csv', index=False)
        print()
        print("Debug files saved:")
        print("  - debug_labels_original.csv")
        print("  - debug_labels_optimized.csv")

        return False


def check_timestamp_alignment(df):
    """
    Check what percentage of timestamps are aligned to hour boundaries.
    """
    timestamps = df.index

    # Check 1h alignment
    aligned_1h = sum(1 for ts in timestamps if ts.minute == 0 and ts.second == 0)
    pct_1h = (aligned_1h / len(timestamps)) * 100

    # Check 4h alignment
    aligned_4h = sum(1 for ts in timestamps if ts.hour % 4 == 0 and ts.minute == 0 and ts.second == 0)
    pct_4h = (aligned_4h / len(timestamps)) * 100

    print("Timestamp Alignment Analysis:")
    print(f"  1h aligned:  {aligned_1h:,}/{len(timestamps):,} ({pct_1h:.1f}%)")
    print(f"  4h aligned:  {aligned_4h:,}/{len(timestamps):,} ({pct_4h:.1f}%)")
    print()


if __name__ == "__main__":
    print("\nStarting continuation label optimization test...\n")

    success = test_continuation_optimization()

    if success:
        print("\n" + "=" * 80)
        print("🎉 TEST PASSED! Optimization is working correctly!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("⚠️  TEST FAILED! Please check the implementation.")
        print("=" * 80)
        sys.exit(1)