"""
Test RSI Optimization: Verify that extracting current RSI from series produces identical results

This test ensures that the optimization in full_features.py:
    rsi_series = calculate_rsi_series(prices, period=14)
    rsi = float(rsi_series[-1])

Produces IDENTICAL results to the original approach:
    rsi_series = calculate_rsi_series(prices, period=14)
    rsi = calculate_rsi(prices, period=14)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.rsi import calculate_rsi, calculate_rsi_series


def test_rsi_extraction_equivalence():
    """Test that extracting RSI from series is identical to calculating it directly."""

    # Test with various price scenarios
    test_cases = [
        # Case 1: Trending up
        np.array([100, 102, 105, 108, 110, 112, 115, 118, 120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148]),

        # Case 2: Trending down
        np.array([150, 148, 145, 142, 140, 138, 135, 132, 130, 128, 125, 122, 120, 118, 115, 112, 110, 108, 105, 102]),

        # Case 3: Sideways
        np.array([100, 101, 100, 99, 100, 101, 100, 99, 100, 101, 100, 99, 100, 101, 100, 99, 100, 101, 100, 99]),

        # Case 4: Volatile
        np.array([100, 110, 95, 105, 90, 115, 85, 120, 80, 125, 75, 130, 70, 135, 65, 140, 60, 145, 55, 150]),

        # Case 5: Real-ish TSLA prices
        np.array([250.5, 251.2, 252.3, 251.8, 253.1, 254.5, 253.9, 255.2, 256.1, 255.8,
                  257.3, 258.9, 257.5, 259.2, 260.5, 261.2, 260.8, 262.3, 263.5, 264.1]),

        # Case 6: Large dataset (100 bars)
        np.cumsum(np.random.randn(100)) + 200,
    ]

    all_passed = True

    for i, prices in enumerate(test_cases):
        # Original approach (what we're replacing)
        rsi_series = calculate_rsi_series(prices, period=14)
        rsi_original = calculate_rsi(prices, period=14)

        # Optimized approach (what we're using now)
        rsi_optimized = float(rsi_series[-1])

        # Check if they're identical
        difference = abs(rsi_original - rsi_optimized)

        if difference > 1e-10:  # Allow for tiny floating point errors
            print(f"FAIL: Test case {i+1}")
            print(f"  Original RSI: {rsi_original}")
            print(f"  Optimized RSI: {rsi_optimized}")
            print(f"  Difference: {difference}")
            all_passed = False
        else:
            print(f"PASS: Test case {i+1} - RSI values identical (diff={difference:.2e})")

    return all_passed


def test_rsi_edge_cases():
    """Test edge cases for RSI calculation."""

    print("\n=== Testing Edge Cases ===")
    all_passed = True

    # Edge case 1: Minimum data (period + 1 bars)
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
    rsi_series = calculate_rsi_series(prices, period=14)
    rsi_original = calculate_rsi(prices, period=14)
    rsi_optimized = float(rsi_series[-1])

    diff = abs(rsi_original - rsi_optimized)
    if diff > 1e-10:
        print(f"FAIL: Minimum data edge case")
        print(f"  Difference: {diff}")
        all_passed = False
    else:
        print(f"PASS: Minimum data edge case (diff={diff:.2e})")

    # Edge case 2: All gains (RSI should be ~100)
    prices = np.linspace(100, 200, 50)
    rsi_series = calculate_rsi_series(prices, period=14)
    rsi_original = calculate_rsi(prices, period=14)
    rsi_optimized = float(rsi_series[-1])

    diff = abs(rsi_original - rsi_optimized)
    if diff > 1e-10:
        print(f"FAIL: All gains edge case")
        print(f"  Difference: {diff}")
        all_passed = False
    else:
        print(f"PASS: All gains edge case - RSI={rsi_optimized:.2f} (diff={diff:.2e})")

    # Edge case 3: All losses (RSI should be ~0)
    prices = np.linspace(200, 100, 50)
    rsi_series = calculate_rsi_series(prices, period=14)
    rsi_original = calculate_rsi(prices, period=14)
    rsi_optimized = float(rsi_series[-1])

    diff = abs(rsi_original - rsi_optimized)
    if diff > 1e-10:
        print(f"FAIL: All losses edge case")
        print(f"  Difference: {diff}")
        all_passed = False
    else:
        print(f"PASS: All losses edge case - RSI={rsi_optimized:.2f} (diff={diff:.2e})")

    return all_passed


def test_full_integration():
    """Test the full feature extraction pipeline to ensure identical outputs."""

    print("\n=== Testing Full Integration ===")

    from features.full_features import extract_tsla_channel_features
    from core.channel import detect_channel

    # Create synthetic TSLA data
    dates = pd.date_range('2024-01-01 09:30', periods=500, freq='5min')
    np.random.seed(42)

    # Create realistic OHLCV data
    base_price = 250.0
    prices = base_price + np.cumsum(np.random.randn(500) * 0.5)

    tsla_df = pd.DataFrame({
        'open': prices + np.random.randn(500) * 0.2,
        'high': prices + np.abs(np.random.randn(500) * 0.3),
        'low': prices - np.abs(np.random.randn(500) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)

    # Pre-compute channel and RSI (required by extract_tsla_channel_features)
    channel = detect_channel(tsla_df, window=50)
    rsi_series = calculate_rsi_series(tsla_df['close'].values, period=14)

    # Extract features with pre-computed channel and RSI
    features = extract_tsla_channel_features(tsla_df, '5min', channel, rsi_series, window=50)

    # Manually verify RSI calculation
    rsi_manual = calculate_rsi(tsla_df['close'].values, period=14)
    rsi_from_series = float(rsi_series[-1])

    print(f"Feature extraction RSI: {features.rsi:.6f}")
    print(f"Manual calculate_rsi(): {rsi_manual:.6f}")
    print(f"From series[-1]: {rsi_from_series:.6f}")

    diff1 = abs(features.rsi - rsi_manual)
    diff2 = abs(features.rsi - rsi_from_series)

    if diff1 > 1e-10 or diff2 > 1e-10:
        print(f"FAIL: Integration test")
        print(f"  Diff (feature vs manual): {diff1}")
        print(f"  Diff (feature vs series): {diff2}")
        return False
    else:
        print(f"PASS: Integration test - All RSI values identical")
        return True


if __name__ == '__main__':
    print("=" * 60)
    print("RSI OPTIMIZATION VERIFICATION TEST")
    print("=" * 60)
    print("\nThis test verifies that extracting current RSI from the")
    print("series produces IDENTICAL results to calculating it directly.")
    print("=" * 60)

    print("\n=== Testing RSI Extraction Equivalence ===")
    test1_passed = test_rsi_extraction_equivalence()

    test2_passed = test_rsi_edge_cases()

    test3_passed = test_full_integration()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed and test3_passed:
        print("ALL TESTS PASSED ✓")
        print("\nOptimization verified:")
        print("  - Calculating RSI once as series and extracting current value")
        print("  - Produces IDENTICAL results to separate calculations")
        print("  - Eliminates redundant RSI computation")
        print("=" * 60)
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        sys.exit(1)
