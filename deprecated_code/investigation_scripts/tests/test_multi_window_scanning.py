#!/usr/bin/env python3
"""
Test script for multi-window feature extraction in v7/training/scanning.py

This script verifies that:
1. Features are extracted for all valid windows
2. The per_window_features dict is populated
3. Backward compatibility is maintained (features field still works)
4. The implementation works in both sequential and parallel modes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from v7.training.scanning import scan_valid_channels
from v7.core.channel import STANDARD_WINDOWS


def create_test_data(num_bars=5000):
    """Create synthetic test data for scanning."""
    print("Creating synthetic test data...")

    # Create timestamps (5-min bars)
    start_date = pd.Timestamp('2024-01-01 09:30:00')
    timestamps = pd.date_range(start=start_date, periods=num_bars, freq='5min')

    # Create synthetic TSLA data with some trends and channels
    np.random.seed(42)
    price = 200.0
    prices = []

    for i in range(num_bars):
        # Add some trend and noise
        trend = 0.001 * (i % 200 - 100)
        noise = np.random.randn() * 2
        price += trend + noise
        prices.append(price)

    prices = np.array(prices)

    # Create OHLC data
    tsla_df = pd.DataFrame({
        'open': prices + np.random.randn(num_bars) * 0.5,
        'high': prices + np.abs(np.random.randn(num_bars)) * 1.5,
        'low': prices - np.abs(np.random.randn(num_bars)) * 1.5,
        'close': prices + np.random.randn(num_bars) * 0.5,
        'volume': np.random.randint(1000000, 10000000, num_bars)
    }, index=timestamps)

    # Ensure high >= close, low <= close
    tsla_df['high'] = tsla_df[['high', 'close', 'open']].max(axis=1)
    tsla_df['low'] = tsla_df[['low', 'close', 'open']].min(axis=1)

    # Create SPY data (similar but not identical)
    spy_df = pd.DataFrame({
        'open': prices * 2 + np.random.randn(num_bars) * 1,
        'high': prices * 2 + np.abs(np.random.randn(num_bars)) * 2,
        'low': prices * 2 - np.abs(np.random.randn(num_bars)) * 2,
        'close': prices * 2 + np.random.randn(num_bars) * 1,
        'volume': np.random.randint(5000000, 50000000, num_bars)
    }, index=timestamps)

    spy_df['high'] = spy_df[['high', 'close', 'open']].max(axis=1)
    spy_df['low'] = spy_df[['low', 'close', 'open']].min(axis=1)

    # Create VIX data (daily, forward-filled to 5-min)
    vix_daily = pd.DataFrame({
        'open': 15 + np.random.randn(num_bars // 78) * 2,
        'high': 16 + np.random.randn(num_bars // 78) * 2,
        'low': 14 + np.random.randn(num_bars // 78) * 2,
        'close': 15 + np.random.randn(num_bars // 78) * 2,
    })

    # Forward fill to match 5-min timestamps
    vix_df = pd.DataFrame(index=timestamps)
    for col in ['open', 'high', 'low', 'close']:
        vix_df[col] = 15.0  # Simple constant for testing

    print(f"Created {len(tsla_df)} bars of test data")
    return tsla_df, spy_df, vix_df


def test_sequential_scanning():
    """Test multi-window feature extraction in sequential mode."""
    print("\n" + "="*80)
    print("TEST 1: Sequential Scanning with Multi-Window Features")
    print("="*80)

    # Create test data (need ~33k bars for warmup + forward data)
    tsla_df, spy_df, vix_df = create_test_data(num_bars=50000)

    # Run scanning in sequential mode (easier to debug)
    print("\nRunning sequential scan...")
    samples, min_warmup = scan_valid_channels(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        window=20,
        step=100,  # Large step for faster testing
        min_cycles=1,
        max_scan=500,
        return_threshold=20,
        include_history=False,  # Faster without history
        lookforward_bars=200,
        progress=True,
        parallel=False,  # Sequential mode
        custom_return_thresholds=None
    )

    print(f"\nScan complete. Found {len(samples)} valid samples")

    if len(samples) == 0:
        print("WARNING: No samples found. Test data may not have valid channels.")
        return False

    # Verify multi-window features
    print("\n" + "-"*80)
    print("Verifying Multi-Window Features")
    print("-"*80)

    sample = samples[0]
    print(f"\nSample 0 details:")
    print(f"  Timestamp: {sample.timestamp}")
    print(f"  Best window: {sample.best_window}")
    print(f"  Available channels: {list(sample.channels.keys())}")

    # Check per_window_features
    if sample.per_window_features is None:
        print("\n❌ FAIL: per_window_features is None")
        return False

    print(f"  Available feature windows: {list(sample.per_window_features.keys())}")
    print(f"  Number of windows: {len(sample.per_window_features)}")

    # Verify we have features for multiple windows
    if len(sample.per_window_features) < 2:
        print(f"\n❌ FAIL: Expected features for multiple windows, got {len(sample.per_window_features)}")
        return False

    print(f"\n✓ Multi-window features extracted for {len(sample.per_window_features)} windows")

    # Verify backward compatibility
    print("\n" + "-"*80)
    print("Verifying Backward Compatibility")
    print("-"*80)

    if sample.features is None:
        print("\n❌ FAIL: Backward compatibility broken - features field is None")
        return False

    best_win_features = sample.per_window_features.get(sample.best_window)
    if best_win_features is None:
        print(f"\n❌ FAIL: No features for best_window ({sample.best_window})")
        return False

    print(f"✓ Backward compatible 'features' field present")
    print(f"✓ Features available for best_window ({sample.best_window})")

    # Verify feature structure
    print("\n" + "-"*80)
    print("Verifying Feature Structure")
    print("-"*80)

    for window, features in sample.per_window_features.items():
        print(f"\nWindow {window}:")
        print(f"  Timestamp: {features.timestamp}")
        print(f"  TSLA timeframes: {list(features.tsla.keys())}")
        print(f"  SPY timeframes: {list(features.spy.keys())}")

        # Check that features have expected structure
        if '5min' not in features.tsla:
            print(f"  ❌ FAIL: Missing '5min' timeframe in TSLA features")
            return False

        tsla_5min = features.tsla['5min']
        print(f"  TSLA 5min position: {tsla_5min.position:.3f}")
        print(f"  TSLA 5min direction: {tsla_5min.direction}")

    print("\n✓ All feature windows have correct structure")

    # Test helper methods
    print("\n" + "-"*80)
    print("Testing Helper Methods")
    print("-"*80)

    if not sample.has_multi_window_features():
        print("\n❌ FAIL: has_multi_window_features() returned False")
        return False

    print(f"✓ has_multi_window_features() = True")
    print(f"✓ get_window_count() = {sample.get_window_count()}")

    # Test get_features_for_window
    for window in STANDARD_WINDOWS[:3]:  # Test first 3 windows
        features = sample.get_features_for_window(window)
        if window in sample.per_window_features:
            if features is None:
                print(f"❌ FAIL: get_features_for_window({window}) returned None")
                return False
            print(f"✓ get_features_for_window({window}) returned features")
        else:
            if features is not None:
                print(f"❌ FAIL: get_features_for_window({window}) should return None")
                return False
            print(f"✓ get_features_for_window({window}) correctly returned None")

    print("\n" + "="*80)
    print("✓ SEQUENTIAL SCANNING TEST PASSED")
    print("="*80)
    return True


def test_parallel_scanning():
    """Test multi-window feature extraction in parallel mode."""
    print("\n" + "="*80)
    print("TEST 2: Parallel Scanning with Multi-Window Features")
    print("="*80)

    # Create test data (need ~33k bars for warmup + forward data)
    tsla_df, spy_df, vix_df = create_test_data(num_bars=50000)

    # Run scanning in parallel mode
    print("\nRunning parallel scan...")
    samples, min_warmup = scan_valid_channels(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        window=20,
        step=100,  # Large step for faster testing
        min_cycles=1,
        max_scan=500,
        return_threshold=20,
        include_history=False,  # Faster without history
        lookforward_bars=200,
        progress=True,
        parallel=True,  # Parallel mode
        max_workers=4,
        custom_return_thresholds=None
    )

    print(f"\nScan complete. Found {len(samples)} valid samples")

    if len(samples) == 0:
        print("WARNING: No samples found. Test data may not have valid channels.")
        return False

    # Quick verification
    sample = samples[0]
    print(f"\nSample 0 details:")
    print(f"  Best window: {sample.best_window}")
    print(f"  Feature windows: {list(sample.per_window_features.keys())}")
    print(f"  Number of windows: {len(sample.per_window_features)}")

    if len(sample.per_window_features) < 2:
        print(f"\n❌ FAIL: Expected features for multiple windows, got {len(sample.per_window_features)}")
        return False

    print("\n" + "="*80)
    print("✓ PARALLEL SCANNING TEST PASSED")
    print("="*80)
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Multi-Window Feature Extraction Test Suite")
    print("="*80)

    # Run tests
    test1_passed = test_sequential_scanning()
    test2_passed = test_parallel_scanning()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Sequential Scanning: {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Parallel Scanning:   {'✓ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
