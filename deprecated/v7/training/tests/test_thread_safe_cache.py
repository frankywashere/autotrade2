"""
Comprehensive unit tests for the thread-safe cache implementation in labels.py.

The cache uses Python's threading.local() for per-thread storage, which provides
inherent thread safety without requiring locks. Each thread has its own independent
cache dictionary, eliminating race conditions and ensuring thread isolation.

These tests verify:
1. Basic cache functionality (single-threaded)
2. Concurrent access with ThreadPoolExecutor
3. Cache clearing in multi-threaded context
4. Determinism - same inputs produce same outputs across threads
5. Cached results match direct resample_ohlc() calls
6. Stress test with 100+ concurrent calls
7. No KeyError exceptions or race conditions
8. Thread isolation - each thread has independent cache
"""

import pytest
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import sys
from pathlib import Path
import time
from typing import List, Tuple, Dict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.labels import (
    cached_resample_ohlc,
    clear_resample_cache,
    _get_resample_cache,
    get_cache_stats,
    reset_cache_stats,
)
from core.timeframe import resample_ohlc, TIMEFRAMES


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """
    Create sample OHLCV data for testing.

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    # Create 1000 bars of 5min data (about 3.5 days)
    np.random.seed(42)
    n_bars = 1000

    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')

    # Generate realistic-looking price data
    close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.1)

    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)

    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['high', 'close', 'open']].max(axis=1)
    df['low'] = df[['low', 'close', 'open']].min(axis=1)

    return df


@pytest.fixture
def multiple_dataframes():
    """
    Create multiple different DataFrames for testing cache isolation.

    Returns:
        List of 3 DataFrames with different data
    """
    dfs = []
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        n_bars = 500
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
        dfs.append(df)

    return dfs


# =============================================================================
# Test 1: Basic Cache Functionality (Single-threaded)
# =============================================================================

class TestBasicCacheFunctionality:
    """Test basic cache operations in single-threaded context."""

    def test_cache_hit_returns_same_object(self, sample_ohlcv_data):
        """Test that cache returns the exact same object on subsequent calls."""
        clear_resample_cache()

        # First call - should cache the result
        result1 = cached_resample_ohlc(sample_ohlcv_data, '15min')

        # Second call - should return cached result
        result2 = cached_resample_ohlc(sample_ohlcv_data, '15min')

        # Should be the exact same object in memory
        assert result1 is result2, "Cache should return same object"

    def test_cache_stores_different_timeframes(self, sample_ohlcv_data):
        """Test that cache stores results separately for different timeframes."""
        clear_resample_cache()

        result_15min = cached_resample_ohlc(sample_ohlcv_data, '15min')
        result_1h = cached_resample_ohlc(sample_ohlcv_data, '1h')
        result_daily = cached_resample_ohlc(sample_ohlcv_data, 'daily')

        # All should be different objects
        assert result_15min is not result_1h
        assert result_15min is not result_daily
        assert result_1h is not result_daily

        # Verify they have different lengths (as expected)
        assert len(result_15min) > len(result_1h) > len(result_daily)

    def test_cache_key_uses_id_and_length(self, sample_ohlcv_data):
        """Test that cache key correctly uses df id and length."""
        clear_resample_cache()

        # First call
        result1 = cached_resample_ohlc(sample_ohlcv_data, '15min')

        # Check that cache has exactly one entry
        cache = _get_resample_cache()
        assert len(cache) == 1

        # The key should be (id, len, timeframe)
        expected_key = (id(sample_ohlcv_data), len(sample_ohlcv_data), '15min')
        assert expected_key in cache

    def test_clear_cache_empties_cache(self, sample_ohlcv_data):
        """Test that clear_resample_cache() properly clears the cache."""
        clear_resample_cache()

        # Add multiple entries
        cached_resample_ohlc(sample_ohlcv_data, '15min')
        cached_resample_ohlc(sample_ohlcv_data, '1h')
        cached_resample_ohlc(sample_ohlcv_data, 'daily')

        cache = _get_resample_cache()
        assert len(cache) == 3

        # Clear cache
        clear_resample_cache()

        cache = _get_resample_cache()
        assert len(cache) == 0

    def test_cached_results_match_direct_calls(self, sample_ohlcv_data):
        """Test that cached results are identical to direct resample_ohlc() calls."""
        clear_resample_cache()

        for tf in ['15min', '30min', '1h', '4h', 'daily']:
            # Get cached result
            cached_result = cached_resample_ohlc(sample_ohlcv_data, tf)

            # Get direct result
            direct_result = resample_ohlc(sample_ohlcv_data, tf)

            # Should be identical
            pd.testing.assert_frame_equal(cached_result, direct_result)


# =============================================================================
# Test 2: Concurrent Access with ThreadPoolExecutor
# =============================================================================

class TestConcurrentAccess:
    """Test cache behavior under concurrent access."""

    def test_concurrent_same_input_4_workers(self, sample_ohlcv_data):
        """Test 4 workers accessing cache with same input simultaneously."""
        clear_resample_cache()

        def resample_task(df, tf, task_id):
            """Task to be run by worker thread."""
            result = cached_resample_ohlc(df, tf)
            return task_id, result

        # Run 20 tasks with 4 workers, all using same input
        num_tasks = 20
        num_workers = 4
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(resample_task, sample_ohlcv_data, '15min', i)
                for i in range(num_tasks)
            ]

            for future in as_completed(futures):
                results.append(future.result())

        # All results should be identical DataFrames
        first_result = results[0][1]
        for task_id, result in results[1:]:
            pd.testing.assert_frame_equal(result, first_result)

    def test_concurrent_different_timeframes_8_workers(self, sample_ohlcv_data):
        """Test 8 workers accessing cache with different timeframes."""
        clear_resample_cache()

        def resample_task(df, tf):
            """Task to resample to a specific timeframe."""
            return tf, cached_resample_ohlc(df, tf)

        # Test with 8 workers, multiple timeframes
        num_workers = 8
        timeframes = ['15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly']

        # Run each timeframe 5 times (40 total tasks)
        tasks = timeframes * 5

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(resample_task, sample_ohlcv_data, tf) for tf in tasks]
            results = [future.result() for future in as_completed(futures)]

        # Group results by timeframe
        results_by_tf: Dict[str, List[pd.DataFrame]] = {}
        for tf, df in results:
            if tf not in results_by_tf:
                results_by_tf[tf] = []
            results_by_tf[tf].append(df)

        # Verify all results for each timeframe are identical
        for tf, dfs in results_by_tf.items():
            first_df = dfs[0]
            for df in dfs[1:]:
                pd.testing.assert_frame_equal(df, first_df)

    def test_concurrent_mixed_operations(self, sample_ohlcv_data):
        """Test concurrent reads, writes, and clears."""
        clear_resample_cache()

        lock = Lock()
        errors = []

        def read_task(df, tf, task_id):
            """Read from cache."""
            try:
                result = cached_resample_ohlc(df, tf)
                return ('read', task_id, len(result))
            except Exception as e:
                with lock:
                    errors.append(('read', task_id, str(e)))
                raise

        def clear_task(task_id):
            """Clear cache."""
            try:
                clear_resample_cache()
                return ('clear', task_id, None)
            except Exception as e:
                with lock:
                    errors.append(('clear', task_id, str(e)))
                raise

        # Mix of operations
        num_workers = 6

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # 30 read tasks
            for i in range(30):
                tf = ['15min', '1h', 'daily'][i % 3]
                futures.append(executor.submit(read_task, sample_ohlcv_data, tf, i))

            # 5 clear tasks interspersed
            for i in range(5):
                futures.append(executor.submit(clear_task, i))

            # Wait for all to complete
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass  # Errors already captured

        # Should have no errors
        assert len(errors) == 0, f"Encountered errors: {errors}"

        # All read tasks should have succeeded
        read_results = [r for r in results if r[0] == 'read']
        assert len(read_results) == 30, "All read tasks should complete"


# =============================================================================
# Test 3: Cache Clearing in Multi-threaded Context
# =============================================================================

class TestCacheClearingMultithreaded:
    """Test cache clearing behavior with concurrent access."""

    def test_clear_during_concurrent_reads(self, sample_ohlcv_data):
        """Test that clearing cache during reads doesn't cause errors."""
        clear_resample_cache()

        errors = []
        lock = Lock()

        def read_worker(df, worker_id):
            """Worker that continuously reads from cache."""
            try:
                for i in range(10):
                    tf = ['15min', '1h', 'daily'][i % 3]
                    result = cached_resample_ohlc(df, tf)
                    # Verify result is valid
                    assert len(result) > 0
                    assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
                return worker_id, 'success'
            except Exception as e:
                with lock:
                    errors.append((worker_id, str(e)))
                return worker_id, 'error'

        def clear_worker(worker_id):
            """Worker that clears cache."""
            try:
                for _ in range(5):
                    time.sleep(0.001)  # Small delay
                    clear_resample_cache()
                return worker_id, 'success'
            except Exception as e:
                with lock:
                    errors.append((worker_id, str(e)))
                return worker_id, 'error'

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            # 6 read workers
            for i in range(6):
                futures.append(executor.submit(read_worker, sample_ohlcv_data, i))

            # 2 clear workers
            for i in range(2):
                futures.append(executor.submit(clear_worker, i + 100))

            results = [future.result() for future in as_completed(futures)]

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All workers should succeed
        assert all(result[1] == 'success' for result in results)

    def test_cache_state_after_concurrent_clears(self, sample_ohlcv_data):
        """Test cache state is consistent after multiple concurrent clears."""
        clear_resample_cache()

        # Pre-populate cache
        cached_resample_ohlc(sample_ohlcv_data, '15min')
        cached_resample_ohlc(sample_ohlcv_data, '1h')

        assert len(_resample_cache) == 2

        # Clear from multiple threads
        def clear_task():
            clear_resample_cache()
            return True

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(clear_task) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]

        # Cache should be empty
        assert len(_resample_cache) == 0
        assert all(results)  # All clears succeeded


# =============================================================================
# Test 4: Determinism - Same Inputs Produce Same Outputs
# =============================================================================

class TestDeterminism:
    """Test that cache produces deterministic results across threads."""

    def test_sequential_vs_parallel_determinism(self, sample_ohlcv_data):
        """Test that sequential and parallel execution produce identical results."""
        timeframes = ['15min', '30min', '1h', '2h', '4h', 'daily']

        # Sequential execution
        clear_resample_cache()
        sequential_results = {}
        for tf in timeframes:
            sequential_results[tf] = cached_resample_ohlc(sample_ohlcv_data, tf).copy()

        # Parallel execution
        clear_resample_cache()

        def resample_task(df, tf):
            return tf, cached_resample_ohlc(df, tf).copy()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(resample_task, sample_ohlcv_data, tf) for tf in timeframes]
            parallel_results = dict([future.result() for future in as_completed(futures)])

        # Compare results
        for tf in timeframes:
            pd.testing.assert_frame_equal(
                sequential_results[tf],
                parallel_results[tf],
                check_exact=True
            )

    def test_repeated_parallel_runs_identical(self, sample_ohlcv_data):
        """Test that multiple parallel runs produce identical results."""
        timeframes = ['15min', '1h', 'daily']

        def run_parallel_resampling(df, tfs):
            """Run parallel resampling and return results."""
            clear_resample_cache()

            def task(df, tf):
                return tf, cached_resample_ohlc(df, tf).copy()

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(task, df, tf) for tf in tfs]
                return dict([future.result() for future in as_completed(futures)])

        # Run 5 times
        runs = [run_parallel_resampling(sample_ohlcv_data, timeframes) for _ in range(5)]

        # All runs should produce identical results
        first_run = runs[0]
        for run in runs[1:]:
            for tf in timeframes:
                pd.testing.assert_frame_equal(first_run[tf], run[tf], check_exact=True)

    def test_same_data_different_threads_same_result(self, sample_ohlcv_data):
        """Test that same data accessed from different threads yields identical results."""
        clear_resample_cache()

        results = []
        lock = Lock()

        def worker_task(df, tf, worker_id):
            """Each worker gets its own copy of the result."""
            result = cached_resample_ohlc(df, tf).copy()
            with lock:
                results.append((worker_id, result))
            return worker_id

        # 10 workers all access same data/timeframe
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(worker_task, sample_ohlcv_data, '15min', i)
                for i in range(10)
            ]
            [future.result() for future in as_completed(futures)]

        # All results should be identical
        first_result = results[0][1]
        for worker_id, result in results[1:]:
            pd.testing.assert_frame_equal(result, first_result, check_exact=True)


# =============================================================================
# Test 5: Cached Results Match Direct Calls
# =============================================================================

class TestCachedVsDirectResults:
    """Test that cached results match direct resample_ohlc calls."""

    def test_all_timeframes_match_direct(self, sample_ohlcv_data):
        """Test all timeframes produce identical results."""
        clear_resample_cache()

        # Test subset of timeframes (some are too long for small dataset)
        testable_tfs = ['15min', '30min', '1h', '2h', '3h', '4h', 'daily']

        for tf in testable_tfs:
            cached = cached_resample_ohlc(sample_ohlcv_data, tf)
            direct = resample_ohlc(sample_ohlcv_data, tf)

            pd.testing.assert_frame_equal(
                cached, direct,
                check_exact=True,
                obj=f"Timeframe: {tf}"
            )

    def test_parallel_cached_vs_sequential_direct(self, sample_ohlcv_data):
        """Test parallel cached calls match sequential direct calls."""
        timeframes = ['15min', '30min', '1h', '4h', 'daily']

        # Sequential direct calls
        direct_results = {}
        for tf in timeframes:
            direct_results[tf] = resample_ohlc(sample_ohlcv_data, tf)

        # Parallel cached calls
        clear_resample_cache()

        def task(df, tf):
            return tf, cached_resample_ohlc(df, tf)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(task, sample_ohlcv_data, tf) for tf in timeframes]
            cached_results = dict([future.result() for future in as_completed(futures)])

        # Compare
        for tf in timeframes:
            pd.testing.assert_frame_equal(
                cached_results[tf],
                direct_results[tf],
                check_exact=True
            )

    def test_cache_preserves_data_types(self, sample_ohlcv_data):
        """Test that cache preserves DataFrame dtypes."""
        clear_resample_cache()

        direct = resample_ohlc(sample_ohlcv_data, '15min')
        cached = cached_resample_ohlc(sample_ohlcv_data, '15min')

        # Check dtypes match
        assert direct.dtypes.equals(cached.dtypes)

        # Check index type matches
        assert type(direct.index) == type(cached.index)

    def test_cache_preserves_index_properties(self, sample_ohlcv_data):
        """Test that cache preserves DatetimeIndex properties."""
        clear_resample_cache()

        tf = '1h'
        direct = resample_ohlc(sample_ohlcv_data, tf)
        cached = cached_resample_ohlc(sample_ohlcv_data, tf)

        # Index should be identical
        assert direct.index.equals(cached.index)

        # Check timezone info (if any)
        assert direct.index.tz == cached.index.tz

        # Check frequency info (if any)
        assert direct.index.freq == cached.index.freq


# =============================================================================
# Test 6: Stress Test - 100+ Concurrent Calls
# =============================================================================

class TestStressTest:
    """Stress test with high concurrency."""

    def test_100_concurrent_calls_same_input(self, sample_ohlcv_data):
        """Stress test with 100 concurrent calls on same input."""
        clear_resample_cache()

        errors = []
        lock = Lock()

        def task(df, tf, task_id):
            """Resample task."""
            try:
                result = cached_resample_ohlc(df, tf)
                # Verify result validity
                assert len(result) > 0
                assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
                return task_id, 'success', len(result)
            except Exception as e:
                with lock:
                    errors.append((task_id, str(e)))
                return task_id, 'error', 0

        # 100 tasks across 8 workers
        num_tasks = 100
        num_workers = 8
        timeframes = ['15min', '30min', '1h', '4h', 'daily']

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(task, sample_ohlcv_data, timeframes[i % len(timeframes)], i)
                for i in range(num_tasks)
            ]
            results = [future.result() for future in as_completed(futures)]

        # No errors
        assert len(errors) == 0, f"Errors: {errors}"

        # All tasks succeeded
        assert all(r[1] == 'success' for r in results)

        # All results for same TF should have same length
        results_by_tf: Dict[str, List[int]] = {}
        for task_id, status, length in results:
            tf = timeframes[task_id % len(timeframes)]
            if tf not in results_by_tf:
                results_by_tf[tf] = []
            results_by_tf[tf].append(length)

        for tf, lengths in results_by_tf.items():
            assert len(set(lengths)) == 1, f"TF {tf} has inconsistent lengths: {lengths}"

    def test_150_concurrent_calls_mixed_inputs(self, multiple_dataframes):
        """Stress test with 150 concurrent calls on multiple different inputs."""
        clear_resample_cache()

        errors = []
        lock = Lock()

        def task(df, tf, task_id):
            """Resample task."""
            try:
                result = cached_resample_ohlc(df, tf)
                assert len(result) > 0
                return task_id, 'success', id(df), tf, len(result)
            except Exception as e:
                with lock:
                    errors.append((task_id, str(e)))
                return task_id, 'error', None, None, 0

        num_tasks = 150
        num_workers = 8
        timeframes = ['15min', '30min', '1h', '2h', 'daily']

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    task,
                    multiple_dataframes[i % len(multiple_dataframes)],
                    timeframes[i % len(timeframes)],
                    i
                )
                for i in range(num_tasks)
            ]
            results = [future.result() for future in as_completed(futures)]

        # No errors
        assert len(errors) == 0, f"Errors: {errors}"

        # All succeeded
        assert all(r[1] == 'success' for r in results)

        # Group by (df_id, tf) and verify consistent lengths
        results_by_key: Dict[Tuple[int, str], List[int]] = {}
        for task_id, status, df_id, tf, length in results:
            if status == 'success':
                key = (df_id, tf)
                if key not in results_by_key:
                    results_by_key[key] = []
                results_by_key[key].append(length)

        for key, lengths in results_by_key.items():
            assert len(set(lengths)) == 1, f"Key {key} has inconsistent lengths: {lengths}"

    def test_200_rapid_fire_calls(self, sample_ohlcv_data):
        """Stress test with 200 rapid-fire calls (minimal delay between submissions)."""
        clear_resample_cache()

        errors = []
        lock = Lock()

        def task(df, tf, task_id):
            try:
                result = cached_resample_ohlc(df, tf)
                return task_id, len(result), None
            except Exception as e:
                with lock:
                    errors.append((task_id, type(e).__name__, str(e)))
                return task_id, 0, str(e)

        num_tasks = 200
        num_workers = 10
        timeframes = ['15min', '1h', 'daily']

        # Submit all at once for maximum concurrency pressure
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(task, sample_ohlcv_data, timeframes[i % len(timeframes)], i)
                for i in range(num_tasks)
            ]

            results = [future.result() for future in as_completed(futures)]

        # Check for specific error types
        keyerrors = [e for e in errors if e[1] == 'KeyError']
        assert len(keyerrors) == 0, f"KeyError exceptions occurred: {keyerrors}"

        # No errors at all
        assert len(errors) == 0, f"Total errors: {len(errors)}, Details: {errors[:10]}"

        # All tasks completed
        assert len(results) == num_tasks


# =============================================================================
# Test 7: Race Condition Detection
# =============================================================================

class TestRaceConditions:
    """Specific tests to detect race conditions."""

    def test_no_keyerror_under_concurrent_load(self, sample_ohlcv_data):
        """Test that no KeyError occurs under heavy concurrent load."""
        clear_resample_cache()

        keyerrors = []
        lock = Lock()

        def task(df, tf, task_id):
            try:
                # Access cache multiple times
                for _ in range(5):
                    cached_resample_ohlc(df, tf)
                return True
            except KeyError as e:
                with lock:
                    keyerrors.append((task_id, str(e)))
                raise

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(task, sample_ohlcv_data, '15min', i)
                for i in range(50)
            ]

            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except KeyError:
                    pass  # Already logged

        assert len(keyerrors) == 0, f"KeyError exceptions: {keyerrors}"

    def test_cache_consistency_during_modifications(self, sample_ohlcv_data):
        """Test cache remains consistent when modified from multiple threads."""
        clear_resample_cache()

        inconsistencies = []
        lock = Lock()

        def reader_task(df, worker_id):
            """Read from cache repeatedly."""
            try:
                for i in range(20):
                    tf = ['15min', '1h'][i % 2]
                    result = cached_resample_ohlc(df, tf)
                    # Verify result is valid
                    if len(result) == 0:
                        with lock:
                            inconsistencies.append(f"Reader {worker_id}: Empty result for {tf}")
                return True
            except Exception as e:
                with lock:
                    inconsistencies.append(f"Reader {worker_id}: {type(e).__name__}: {e}")
                return False

        def clearer_task(worker_id):
            """Clear cache repeatedly."""
            try:
                for _ in range(10):
                    time.sleep(0.001)
                    clear_resample_cache()
                return True
            except Exception as e:
                with lock:
                    inconsistencies.append(f"Clearer {worker_id}: {type(e).__name__}: {e}")
                return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []

            # 5 readers
            for i in range(5):
                futures.append(executor.submit(reader_task, sample_ohlcv_data, i))

            # 3 clearers
            for i in range(3):
                futures.append(executor.submit(clearer_task, i + 100))

            results = [future.result() for future in as_completed(futures)]

        assert len(inconsistencies) == 0, f"Inconsistencies: {inconsistencies}"
        assert all(results), "All tasks should succeed"


# =============================================================================
# Test 8: Cache Isolation Between Different DataFrames
# =============================================================================

class TestCacheIsolation:
    """Test that cache properly isolates different DataFrames."""

    def test_different_dataframes_different_cache_entries(self, multiple_dataframes):
        """Test that different DataFrames create separate cache entries."""
        clear_resample_cache()

        # Resample all three DataFrames to same timeframe
        results = []
        for df in multiple_dataframes:
            results.append(cached_resample_ohlc(df, '15min'))

        # Should have 3 cache entries
        assert len(_resample_cache) == 3

        # Results should be different
        pd.testing.assert_frame_equal(results[0], results[1], check_exact=False)
        # The above should raise, so we use try/except
        try:
            pd.testing.assert_frame_equal(results[0], results[1], check_exact=True)
            same_01 = True
        except AssertionError:
            same_01 = False

        try:
            pd.testing.assert_frame_equal(results[0], results[2], check_exact=True)
            same_02 = True
        except AssertionError:
            same_02 = False

        # At least one should be different (different random seeds)
        assert not (same_01 and same_02), "All results should not be identical"

    def test_concurrent_access_different_dataframes(self, multiple_dataframes):
        """Test concurrent access to different DataFrames."""
        clear_resample_cache()

        def task(df, tf, df_index):
            result = cached_resample_ohlc(df, tf)
            return df_index, tf, result.copy()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []

            # Each DataFrame gets accessed multiple times
            for _ in range(10):
                for i, df in enumerate(multiple_dataframes):
                    futures.append(executor.submit(task, df, '15min', i))

            results = [future.result() for future in as_completed(futures)]

        # Group by DataFrame index
        results_by_df: Dict[int, List[pd.DataFrame]] = {}
        for df_idx, tf, result in results:
            if df_idx not in results_by_df:
                results_by_df[df_idx] = []
            results_by_df[df_idx].append(result)

        # All results for same DataFrame should be identical
        for df_idx, dfs in results_by_df.items():
            first = dfs[0]
            for df in dfs[1:]:
                pd.testing.assert_frame_equal(first, df, check_exact=True)


# =============================================================================
# Test 9: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_cache_concurrent_access(self, sample_ohlcv_data):
        """Test concurrent access to empty cache (all miss initially)."""
        clear_resample_cache()

        def task(df, tf):
            return cached_resample_ohlc(df, tf).copy()

        # All access simultaneously (cache starts empty)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(task, sample_ohlcv_data, '15min')
                for _ in range(10)
            ]
            results = [future.result() for future in as_completed(futures)]

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            pd.testing.assert_frame_equal(first, result, check_exact=True)

    def test_very_small_dataframe(self):
        """Test cache with very small DataFrame."""
        clear_resample_cache()

        # Create minimal DataFrame
        dates = pd.date_range(start='2024-01-01', periods=10, freq='5min')
        df = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10
        }, index=dates)

        # Should work fine
        result = cached_resample_ohlc(df, '15min')
        assert len(result) > 0

        # Second call should return cached
        result2 = cached_resample_ohlc(df, '15min')
        assert result is result2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
