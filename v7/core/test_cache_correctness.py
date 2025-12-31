"""
Correctness Verification Tests for Cache Layer

CRITICAL: Ensures cached values are BITWISE IDENTICAL to non-cached computations.
Run this before deploying to ensure no calculation corruption.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List
import hashlib

from core.cache import FeatureCache
from core.timeframe import resample_ohlc, TIMEFRAMES
from core.channel import detect_channel, Channel
from features.rsi import calculate_rsi_series, calculate_rsi


def create_test_dataframe(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic test dataframe."""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')
    prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

    return pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 0.1,
        'high': prices + np.abs(np.random.randn(n_bars) * 0.3),
        'low': prices - np.abs(np.random.randn(n_bars) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)


def verify_dataframe_identical(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Verify two dataframes are bitwise identical."""
    # Check shape
    if df1.shape != df2.shape:
        print(f"Shape mismatch: {df1.shape} vs {df2.shape}")
        return False

    # Check index
    if not df1.index.equals(df2.index):
        print("Index mismatch")
        return False

    # Check columns
    if not df1.columns.equals(df2.columns):
        print("Column mismatch")
        return False

    # Check values (bitwise)
    for col in df1.columns:
        if not np.array_equal(df1[col].values, df2[col].values):
            print(f"Column {col} mismatch")
            return False

    return True


def verify_channel_identical(ch1: Channel, ch2: Channel) -> bool:
    """Verify two channels are bitwise identical."""
    # Check scalar attributes
    if ch1.valid != ch2.valid:
        print(f"valid mismatch: {ch1.valid} vs {ch2.valid}")
        return False

    if ch1.direction != ch2.direction:
        print(f"direction mismatch: {ch1.direction} vs {ch2.direction}")
        return False

    if ch1.slope != ch2.slope:
        print(f"slope mismatch: {ch1.slope} vs {ch2.slope}")
        return False

    if ch1.intercept != ch2.intercept:
        print(f"intercept mismatch: {ch1.intercept} vs {ch2.intercept}")
        return False

    if ch1.r_squared != ch2.r_squared:
        print(f"r_squared mismatch: {ch1.r_squared} vs {ch2.r_squared}")
        return False

    if ch1.std_dev != ch2.std_dev:
        print(f"std_dev mismatch: {ch1.std_dev} vs {ch2.std_dev}")
        return False

    if ch1.width_pct != ch2.width_pct:
        print(f"width_pct mismatch: {ch1.width_pct} vs {ch2.width_pct}")
        return False

    if ch1.bounce_count != ch2.bounce_count:
        print(f"bounce_count mismatch: {ch1.bounce_count} vs {ch2.bounce_count}")
        return False

    if ch1.complete_cycles != ch2.complete_cycles:
        print(f"complete_cycles mismatch: {ch1.complete_cycles} vs {ch2.complete_cycles}")
        return False

    if ch1.window != ch2.window:
        print(f"window mismatch: {ch1.window} vs {ch2.window}")
        return False

    # Check array attributes (bitwise)
    if not np.array_equal(ch1.upper_line, ch2.upper_line):
        print("upper_line mismatch")
        return False

    if not np.array_equal(ch1.lower_line, ch2.lower_line):
        print("lower_line mismatch")
        return False

    if not np.array_equal(ch1.center_line, ch2.center_line):
        print("center_line mismatch")
        return False

    # Check touches
    if len(ch1.touches) != len(ch2.touches):
        print(f"touches count mismatch: {len(ch1.touches)} vs {len(ch2.touches)}")
        return False

    for i, (t1, t2) in enumerate(zip(ch1.touches, ch2.touches)):
        if t1.bar_index != t2.bar_index or t1.touch_type != t2.touch_type or t1.price != t2.price:
            print(f"Touch {i} mismatch")
            return False

    return True


def verify_array_identical(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """Verify two arrays are bitwise identical."""
    return np.array_equal(arr1, arr2, equal_nan=True)


# ============================================================================
# Test 1: Resampling Cache Correctness
# ============================================================================

def test_resampling_correctness():
    """Verify resampling cache produces identical results."""
    print("\n" + "="*70)
    print("TEST 1: Resampling Cache Correctness")
    print("="*70)

    df = create_test_dataframe(1000, seed=42)
    cache = FeatureCache()

    passed = 0
    failed = 0

    for tf in ['15min', '30min', '1h', '4h', 'daily']:
        # Compute without cache
        cache.disable()
        result_no_cache = resample_ohlc(df, tf)

        # Compute with cache (cold)
        cache.enable()
        cache.clear()
        result_cache_cold = cache.resampling.get_or_resample(df, tf, resample_ohlc)

        # Compute with cache (warm)
        result_cache_warm = cache.resampling.get_or_resample(df, tf, resample_ohlc)

        # Verify all three are identical
        if verify_dataframe_identical(result_no_cache, result_cache_cold) and \
           verify_dataframe_identical(result_no_cache, result_cache_warm):
            print(f"✓ {tf:8s} - PASS (bitwise identical)")
            passed += 1
        else:
            print(f"✗ {tf:8s} - FAIL (corrupted data)")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 2: Channel Cache Correctness
# ============================================================================

def test_channel_correctness():
    """Verify channel cache produces identical results."""
    print("\n" + "="*70)
    print("TEST 2: Channel Cache Correctness")
    print("="*70)

    df = create_test_dataframe(1000, seed=42)
    cache = FeatureCache()

    passed = 0
    failed = 0

    windows = [20, 30, 40, 50, 60, 80, 100]

    for window in windows:
        # Compute without cache
        cache.disable()
        result_no_cache = detect_channel(df, window=window)

        # Compute with cache (cold)
        cache.enable()
        cache.clear()
        result_cache_cold = cache.channel.get_or_detect(df, '5min', window, detect_channel)

        # Compute with cache (warm)
        result_cache_warm = cache.channel.get_or_detect(df, '5min', window, detect_channel)

        # Verify all three are identical
        if verify_channel_identical(result_no_cache, result_cache_cold) and \
           verify_channel_identical(result_no_cache, result_cache_warm):
            print(f"✓ window={window:3d} - PASS (bitwise identical)")
            passed += 1
        else:
            print(f"✗ window={window:3d} - FAIL (corrupted channel)")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 3: RSI Cache Correctness
# ============================================================================

def test_rsi_correctness():
    """Verify RSI cache produces identical results."""
    print("\n" + "="*70)
    print("TEST 3: RSI Cache Correctness")
    print("="*70)

    df = create_test_dataframe(1000, seed=42)
    prices = df['close'].values
    cache = FeatureCache()

    passed = 0
    failed = 0

    periods = [7, 14, 21, 28, 50]

    for period in periods:
        # Compute without cache
        cache.disable()
        result_no_cache = calculate_rsi_series(prices, period)

        # Compute with cache (cold)
        cache.enable()
        cache.clear()
        result_cache_cold = cache.rsi.get_or_calculate(
            prices, period, calculate_rsi_series, 'series'
        )

        # Compute with cache (warm)
        result_cache_warm = cache.rsi.get_or_calculate(
            prices, period, calculate_rsi_series, 'series'
        )

        # Verify all three are identical
        if verify_array_identical(result_no_cache, result_cache_cold) and \
           verify_array_identical(result_no_cache, result_cache_warm):
            print(f"✓ period={period:2d} - PASS (bitwise identical)")
            passed += 1
        else:
            print(f"✗ period={period:2d} - FAIL (corrupted RSI)")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================================
# Test 4: Multi-Parameter Variations
# ============================================================================

def test_parameter_variations():
    """Verify cache correctly distinguishes different parameters."""
    print("\n" + "="*70)
    print("TEST 4: Parameter Variation Correctness")
    print("="*70)

    df = create_test_dataframe(1000, seed=42)
    cache = FeatureCache()

    print("\nTesting channel with different std_multipliers...")

    # Detect channels with different parameters
    ch1 = cache.channel.get_or_detect(df, '5min', 50, detect_channel, std_multiplier=1.5)
    ch2 = cache.channel.get_or_detect(df, '5min', 50, detect_channel, std_multiplier=2.0)
    ch3 = cache.channel.get_or_detect(df, '5min', 50, detect_channel, std_multiplier=2.5)

    # These should be DIFFERENT (different parameters)
    # Check width_pct or upper_line (std_dev itself is the same - it's the residual std)
    if not np.array_equal(ch1.upper_line, ch2.upper_line) and \
       not np.array_equal(ch2.upper_line, ch3.upper_line):
        print("✓ Different std_multipliers produce different results")
    else:
        print("✗ Cache incorrectly returning same result for different parameters")
        return False

    # Get same parameters again - should be identical
    ch1_again = cache.channel.get_or_detect(df, '5min', 50, detect_channel, std_multiplier=1.5)

    if verify_channel_identical(ch1, ch1_again):
        print("✓ Same parameters produce identical results (cache hit)")
    else:
        print("✗ Cache corrupted result on second call")
        return False

    stats = cache.stats()
    print(f"\nCache stats: {stats['channel'].hits} hits, {stats['channel'].misses} misses")

    return True


# ============================================================================
# Test 5: Data Variation Correctness
# ============================================================================

def test_data_variations():
    """Verify cache correctly distinguishes different data."""
    print("\n" + "="*70)
    print("TEST 5: Data Variation Correctness")
    print("="*70)

    cache = FeatureCache()

    # Create two different dataframes
    df1 = create_test_dataframe(1000, seed=42)
    df2 = create_test_dataframe(1000, seed=123)

    # Resample both
    result1 = cache.resampling.get_or_resample(df1, '15min', resample_ohlc)
    result2 = cache.resampling.get_or_resample(df2, '15min', resample_ohlc)

    # Should be DIFFERENT
    if not verify_dataframe_identical(result1, result2):
        print("✓ Different input data produces different results")
    else:
        print("✗ Cache incorrectly returning same result for different data")
        return False

    # Get df1 again - should be identical to result1
    result1_again = cache.resampling.get_or_resample(df1, '15min', resample_ohlc)

    if verify_dataframe_identical(result1, result1_again):
        print("✓ Same input data produces identical results (cache hit)")
    else:
        print("✗ Cache corrupted result on second call")
        return False

    stats = cache.stats()
    print(f"\nCache stats: {stats['resampling'].hits} hits, {stats['resampling'].misses} misses")

    return True


# ============================================================================
# Test 6: Thread Safety (Basic)
# ============================================================================

def test_thread_safety():
    """Basic thread safety test."""
    print("\n" + "="*70)
    print("TEST 6: Thread Safety (Basic)")
    print("="*70)

    import threading

    df = create_test_dataframe(1000, seed=42)
    cache = FeatureCache()
    results = {}
    errors = []

    def worker(worker_id: int):
        try:
            # Each worker extracts features
            resampled = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
            channel = cache.channel.get_or_detect(df, '5min', 50, detect_channel)
            results[worker_id] = (resampled, channel)
        except Exception as e:
            errors.append(f"Worker {worker_id}: {e}")

    # Create 10 threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # Wait for all
    for t in threads:
        t.join()

    if errors:
        print(f"✗ Thread safety errors: {errors}")
        return False

    # Verify all results are identical
    first_resampled, first_channel = results[0]
    for i in range(1, 10):
        resampled, channel = results[i]
        if not verify_dataframe_identical(first_resampled, resampled):
            print(f"✗ Worker {i} got different resampled result")
            return False
        if not verify_channel_identical(first_channel, channel):
            print(f"✗ Worker {i} got different channel result")
            return False

    print("✓ All 10 threads produced identical results")
    cache.print_stats()

    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all correctness tests."""
    print("\n" + "="*70)
    print("CACHE CORRECTNESS VERIFICATION")
    print("="*70)
    print("\nThis test suite verifies cached values are BITWISE IDENTICAL")
    print("to non-cached computations.")

    tests = [
        ("Resampling Cache", test_resampling_correctness),
        ("Channel Cache", test_channel_correctness),
        ("RSI Cache", test_rsi_correctness),
        ("Parameter Variations", test_parameter_variations),
        ("Data Variations", test_data_variations),
        ("Thread Safety", test_thread_safety),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} - EXCEPTION: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("Cache layer is verified to preserve exact calculations!")
        print("="*70 + "\n")
        return True
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED ✗")
        print("DO NOT USE CACHE until issues are resolved!")
        print("="*70 + "\n")
        return False


if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
