#!/usr/bin/env python3
"""
Correctness verification for detect_new_channel optimization.

This verifies that the optimized version produces EXACTLY the same results
as a reference implementation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from v7.training.labels import detect_new_channel
from v7.core.channel import detect_channel


def detect_new_channel_reference(df, start_idx, window=50, max_scan=100):
    """
    Reference implementation using the old sequential approach.
    This is used to verify correctness of the optimized version.
    """
    end_idx = min(start_idx + max_scan, len(df) - window + 1)
    if start_idx >= end_idx:
        return None

    for i in range(end_idx - start_idx):
        slice_end = start_idx + i + window
        if slice_end > len(df):
            break

        df_slice = df.iloc[start_idx + i:slice_end]
        channel = detect_channel(df_slice, window=window)
        if channel.valid:
            return channel

    return None


def generate_test_data(n_bars=1000, seed=42):
    """Generate realistic OHLCV test data."""
    np.random.seed(seed)
    trend = np.linspace(100, 110, n_bars)
    noise = np.random.randn(n_bars) * 2
    close = trend + noise
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_price = close + np.random.randn(n_bars) * 0.3
    volume = np.random.randint(1000, 10000, n_bars)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def compare_channels(c1, c2, test_name):
    """Compare two channels for equality."""
    if c1 is None and c2 is None:
        print(f"  {test_name}: PASS (both None)")
        return True

    if (c1 is None) != (c2 is None):
        print(f"  {test_name}: FAIL (one is None, other is not)")
        print(f"    Reference: {c1}")
        print(f"    Optimized: {c2}")
        return False

    # Both are not None, compare properties
    checks = [
        ('valid', c1.valid == c2.valid),
        ('direction', c1.direction == c2.direction),
        ('slope', np.isclose(c1.slope, c2.slope, rtol=1e-10)),
        ('intercept', np.isclose(c1.intercept, c2.intercept, rtol=1e-10)),
        ('r_squared', np.isclose(c1.r_squared, c2.r_squared, rtol=1e-10)),
        ('std_dev', np.isclose(c1.std_dev, c2.std_dev, rtol=1e-10)),
        ('window', c1.window == c2.window),
        ('bounce_count', c1.bounce_count == c2.bounce_count),
        ('alternations', c1.alternations == c2.alternations),
    ]

    all_pass = all(check[1] for check in checks)

    if all_pass:
        print(f"  {test_name}: PASS (all properties match)")
    else:
        print(f"  {test_name}: FAIL")
        for name, passed in checks:
            if not passed:
                print(f"    {name}: MISMATCH")
                print(f"      Reference: {getattr(c1, name)}")
                print(f"      Optimized: {getattr(c2, name)}")

    return all_pass


def verify_correctness():
    """Verify that optimized version produces identical results."""
    print("=" * 80)
    print("CORRECTNESS VERIFICATION: detect_new_channel()")
    print("=" * 80)

    # Generate multiple test datasets
    test_datasets = [
        ("Dataset 1 (seed=42)", generate_test_data(n_bars=2000, seed=42)),
        ("Dataset 2 (seed=123)", generate_test_data(n_bars=2000, seed=123)),
        ("Dataset 3 (seed=999)", generate_test_data(n_bars=3000, seed=999)),
    ]

    test_cases = [
        {"window": 50, "max_scan": 100, "start_idx": 100},
        {"window": 50, "max_scan": 200, "start_idx": 50},
        {"window": 30, "max_scan": 100, "start_idx": 200},
        {"window": 70, "max_scan": 150, "start_idx": 100},
    ]

    all_passed = True
    total_tests = 0

    for dataset_name, df in test_datasets:
        print(f"\n{dataset_name} ({len(df)} bars)")
        print("-" * 80)

        for i, params in enumerate(test_cases, 1):
            window = params["window"]
            max_scan = params["max_scan"]
            start_idx = params["start_idx"]

            test_name = f"Test {i} (w={window}, scan={max_scan}, start={start_idx})"

            # Run both implementations
            ref_result = detect_new_channel_reference(df, start_idx, window, max_scan)
            opt_result = detect_new_channel(df, start_idx, window, max_scan)

            # Compare
            passed = compare_channels(ref_result, opt_result, test_name)
            all_passed = all_passed and passed
            total_tests += 1

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests: {total_tests}")
    print(f"Result: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")

    if all_passed:
        print("\nThe optimized implementation produces EXACTLY the same results")
        print("as the reference implementation. The optimization is correct.")
    else:
        print("\nWARNING: Some tests failed. The optimization may have introduced bugs.")

    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = verify_correctness()
    sys.exit(0 if success else 1)
