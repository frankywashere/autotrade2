#!/usr/bin/env python3
"""
Simple test runner for thread-safe cache tests.
Doesn't require pytest - runs tests directly.
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

from v7.training.labels import (
    cached_resample_ohlc,
    clear_resample_cache,
    _get_resample_cache,
    get_cache_stats,
    reset_cache_stats,
    ENABLE_CACHE_STATS,
)
from v7.core.timeframe import resample_ohlc, TIMEFRAMES


def create_sample_data(n_bars=1000, seed=42):
    """Create sample OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)

    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)

    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)

    return df


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name, func):
        """Run a single test."""
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)
        try:
            func()
            print(f"✓ PASSED")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))

    def summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")

        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error[:100]}")

        return self.failed == 0


# =============================================================================
# Test Functions
# =============================================================================

def test_basic_cache_hit():
    """Test that cache returns same object on second call."""
    df = create_sample_data()
    clear_resample_cache()

    result1 = cached_resample_ohlc(df, '15min')
    result2 = cached_resample_ohlc(df, '15min')

    assert result1 is result2, "Cache should return same object"
    print(f"  Cache returned same object: {id(result1) == id(result2)}")


def test_different_timeframes():
    """Test cache stores different timeframes separately."""
    df = create_sample_data()
    clear_resample_cache()

    r15 = cached_resample_ohlc(df, '15min')
    r1h = cached_resample_ohlc(df, '1h')
    rdaily = cached_resample_ohlc(df, 'daily')

    assert r15 is not r1h, "Different timeframes should be different objects"
    assert len(r15) > len(r1h) > len(rdaily), "Expected decreasing lengths"
    print(f"  Lengths: 15min={len(r15)}, 1h={len(r1h)}, daily={len(rdaily)}")


def test_cache_clear():
    """Test that clear_resample_cache empties the cache."""
    df = create_sample_data()
    clear_resample_cache()

    cached_resample_ohlc(df, '15min')
    cached_resample_ohlc(df, '1h')
    cache = _get_resample_cache()
    assert len(cache) == 2, f"Expected 2 entries, got {len(cache)}"

    clear_resample_cache()
    cache = _get_resample_cache()
    assert len(cache) == 0, f"Cache should be empty, has {len(cache)}"
    print(f"  Cache cleared successfully")


def test_cached_matches_direct():
    """Test cached results match direct resample_ohlc calls."""
    df = create_sample_data()
    clear_resample_cache()

    for tf in ['15min', '30min', '1h', '4h', 'daily']:
        cached = cached_resample_ohlc(df, tf)
        direct = resample_ohlc(df, tf)

        pd.testing.assert_frame_equal(cached, direct, check_exact=True)

    print(f"  All timeframes match direct calls")


def test_concurrent_same_input():
    """Test concurrent access with same input."""
    df = create_sample_data()
    clear_resample_cache()

    def task(df, tf, task_id):
        return task_id, cached_resample_ohlc(df, tf)

    num_tasks = 20
    num_workers = 4

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(task, df, '15min', i) for i in range(num_tasks)]
        results = [f.result() for f in as_completed(futures)]

    # All results should be identical
    first_result = results[0][1]
    for task_id, result in results[1:]:
        pd.testing.assert_frame_equal(result, first_result)

    print(f"  {num_tasks} concurrent calls produced identical results")


def test_concurrent_different_timeframes():
    """Test concurrent access with different timeframes."""
    df = create_sample_data()
    clear_resample_cache()

    def task(df, tf):
        return tf, cached_resample_ohlc(df, tf)

    num_workers = 8
    timeframes = ['15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly']
    tasks = timeframes * 5  # 40 total tasks

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(task, df, tf) for tf in tasks]
        results = [f.result() for f in as_completed(futures)]

    # Group by timeframe and verify consistency
    by_tf = {}
    for tf, result_df in results:
        if tf not in by_tf:
            by_tf[tf] = []
        by_tf[tf].append(result_df)

    for tf, dfs in by_tf.items():
        first = dfs[0]
        for df_result in dfs[1:]:
            pd.testing.assert_frame_equal(first, df_result)

    print(f"  {len(results)} concurrent calls across {len(timeframes)} timeframes - all consistent")


def test_concurrent_with_clears():
    """Test concurrent reads and clears."""
    df = create_sample_data()
    clear_resample_cache()

    errors = []
    lock = Lock()

    def read_task(df, tf, task_id):
        try:
            result = cached_resample_ohlc(df, tf)
            assert len(result) > 0
            return 'read', task_id, True
        except Exception as e:
            with lock:
                errors.append(('read', task_id, str(e)))
            raise

    def clear_task(task_id):
        try:
            clear_resample_cache()
            return 'clear', task_id, True
        except Exception as e:
            with lock:
                errors.append(('clear', task_id, str(e)))
            raise

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []

        # 30 read tasks
        for i in range(30):
            tf = ['15min', '1h', 'daily'][i % 3]
            futures.append(executor.submit(read_task, df, tf, i))

        # 5 clear tasks
        for i in range(5):
            futures.append(executor.submit(clear_task, i))

        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception:
                pass

    assert len(errors) == 0, f"Errors occurred: {errors}"
    print(f"  {len(results)} mixed operations completed without errors")


def test_stress_100_concurrent():
    """Stress test with 100 concurrent calls."""
    df = create_sample_data()
    clear_resample_cache()

    errors = []
    lock = Lock()

    def task(df, tf, task_id):
        try:
            result = cached_resample_ohlc(df, tf)
            assert len(result) > 0
            return task_id, len(result), True
        except Exception as e:
            with lock:
                errors.append((task_id, str(e)))
            return task_id, 0, False

    num_tasks = 100
    num_workers = 8
    timeframes = ['15min', '30min', '1h', '4h', 'daily']

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(task, df, timeframes[i % len(timeframes)], i)
            for i in range(num_tasks)
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(errors) == 0, f"Errors: {errors}"
    assert all(r[2] for r in results), "All tasks should succeed"

    print(f"  {num_tasks} concurrent calls completed successfully")


def test_stress_200_rapid_fire():
    """Stress test with 200 rapid-fire calls."""
    df = create_sample_data()
    clear_resample_cache()

    errors = []
    keyerrors = []
    lock = Lock()

    def task(df, tf, task_id):
        try:
            result = cached_resample_ohlc(df, tf)
            return task_id, len(result), None
        except KeyError as e:
            with lock:
                keyerrors.append((task_id, str(e)))
            return task_id, 0, 'KeyError'
        except Exception as e:
            with lock:
                errors.append((task_id, type(e).__name__, str(e)))
            return task_id, 0, type(e).__name__

    num_tasks = 200
    num_workers = 10
    timeframes = ['15min', '1h', 'daily']

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(task, df, timeframes[i % len(timeframes)], i)
            for i in range(num_tasks)
        ]
        results = [f.result() for f in as_completed(futures)]

    assert len(keyerrors) == 0, f"KeyError exceptions: {keyerrors}"
    assert len(errors) == 0, f"Errors: {errors[:10]}"

    print(f"  {num_tasks} rapid-fire calls - no KeyErrors or race conditions")


def test_no_keyerror_race_condition():
    """Test for KeyError race conditions."""
    df = create_sample_data()
    clear_resample_cache()

    keyerrors = []
    lock = Lock()

    def task(df, tf, task_id):
        try:
            for _ in range(5):
                cached_resample_ohlc(df, tf)
            return True
        except KeyError as e:
            with lock:
                keyerrors.append((task_id, str(e)))
            raise

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(task, df, '15min', i) for i in range(50)]
        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except KeyError:
                pass

    assert len(keyerrors) == 0, f"KeyErrors: {keyerrors}"
    print(f"  No KeyError race conditions detected")


def test_determinism_sequential_vs_parallel():
    """Test sequential and parallel produce same results."""
    df = create_sample_data()
    timeframes = ['15min', '30min', '1h', '2h', '4h', 'daily']

    # Sequential
    clear_resample_cache()
    seq_results = {}
    for tf in timeframes:
        seq_results[tf] = cached_resample_ohlc(df, tf).copy()

    # Parallel
    clear_resample_cache()

    def task(df, tf):
        return tf, cached_resample_ohlc(df, tf).copy()

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(task, df, tf) for tf in timeframes]
        par_results = dict([f.result() for f in as_completed(futures)])

    for tf in timeframes:
        pd.testing.assert_frame_equal(seq_results[tf], par_results[tf], check_exact=True)

    print(f"  Sequential and parallel results are identical")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("THREAD-SAFE CACHE TEST SUITE")
    print("="*60)

    runner = TestRunner()

    # Basic functionality tests
    runner.run_test("Basic cache hit", test_basic_cache_hit)
    runner.run_test("Different timeframes", test_different_timeframes)
    runner.run_test("Cache clear", test_cache_clear)
    runner.run_test("Cached matches direct", test_cached_matches_direct)

    # Concurrent access tests
    runner.run_test("Concurrent same input", test_concurrent_same_input)
    runner.run_test("Concurrent different timeframes", test_concurrent_different_timeframes)
    runner.run_test("Concurrent with clears", test_concurrent_with_clears)

    # Stress tests
    runner.run_test("Stress: 100 concurrent calls", test_stress_100_concurrent)
    runner.run_test("Stress: 200 rapid-fire calls", test_stress_200_rapid_fire)

    # Race condition tests
    runner.run_test("No KeyError race conditions", test_no_keyerror_race_condition)

    # Determinism tests
    runner.run_test("Determinism: sequential vs parallel", test_determinism_sequential_vs_parallel)

    # Summary
    success = runner.summary()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
