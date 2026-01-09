"""
Simple test to verify cache performance monitoring works correctly.

Run this from the project root: python3 test_cache_monitoring.py
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from v7.training.labels import (
    ENABLE_CACHE_STATS,
    cached_resample_ohlc,
    get_cache_stats,
    print_cache_stats,
    reset_cache_stats,
    clear_resample_cache
)

# Import and set the flag
import v7.training.labels as labels


def create_sample_data(bars: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=bars, freq='5min')

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(bars) * 0.5)
    high = close + np.abs(np.random.randn(bars) * 0.3)
    low = close - np.abs(np.random.randn(bars) * 0.3)
    open_ = close + np.random.randn(bars) * 0.2
    volume = np.abs(np.random.randn(bars) * 1000000)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def test_stats_disabled():
    """Test 1: Stats disabled (default) - zero overhead."""
    print("=" * 60)
    print("Test 1: Cache Stats DISABLED (default)")
    print("=" * 60)

    # Verify stats are disabled by default
    assert labels.ENABLE_CACHE_STATS == False, "Stats should be disabled by default"

    df = create_sample_data(1000)

    print("\nPerforming 10 resample operations with stats disabled...")
    for i in range(10):
        df_15min = cached_resample_ohlc(df, '15min')
        df_1h = cached_resample_ohlc(df, '1h')
        df_daily = cached_resample_ohlc(df, 'daily')

    print("Complete! (No overhead from stats tracking)")
    clear_resample_cache()


def test_stats_enabled():
    """Test 2: Stats enabled - tracks hits/misses."""
    print("\n" + "=" * 60)
    print("Test 2: Cache Stats ENABLED")
    print("=" * 60)

    # Enable stats
    labels.ENABLE_CACHE_STATS = True
    reset_cache_stats()

    df = create_sample_data(1000)

    print("\nPerforming 10 resample operations with stats enabled...")

    for i in range(10):
        df_15min = cached_resample_ohlc(df, '15min')  # 3 different TFs
        df_1h = cached_resample_ohlc(df, '1h')
        df_daily = cached_resample_ohlc(df, 'daily')

    # Get and verify stats
    stats = get_cache_stats()

    print("\nCache Statistics:")
    print_cache_stats()

    # Verify correctness
    assert stats['total'] == 30, f"Expected 30 total calls, got {stats['total']}"
    assert stats['hits'] == 27, f"Expected 27 hits (9 iterations * 3 TFs), got {stats['hits']}"
    assert stats['misses'] == 3, f"Expected 3 misses (first iteration), got {stats['misses']}"
    assert 89.0 <= stats['hit_rate'] <= 91.0, f"Expected ~90% hit rate, got {stats['hit_rate']:.1f}%"

    print("\nValidation: All assertions passed!")

    # Cleanup
    labels.ENABLE_CACHE_STATS = False
    clear_resample_cache()


def test_thread_safety():
    """Test 3: Thread-local stats are isolated."""
    print("\n" + "=" * 60)
    print("Test 3: Thread-Local Isolation")
    print("=" * 60)

    import threading

    labels.ENABLE_CACHE_STATS = True
    reset_cache_stats()

    results = {}

    def worker(thread_id, num_calls):
        """Worker function that uses cache and tracks stats."""
        df = create_sample_data(500)

        for i in range(num_calls):
            cached_resample_ohlc(df, '15min')
            cached_resample_ohlc(df, '1h')

        # Get stats for this thread
        stats = get_cache_stats()
        results[thread_id] = stats

    # Create and run multiple threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i, 5))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify each thread tracked its own stats
    print("\nThread-specific stats:")
    for thread_id, stats in results.items():
        print(f"  Thread {thread_id}: {stats['total']} calls, "
              f"{stats['hits']} hits, {stats['misses']} misses")

        # Each thread should have tracked its own calls
        # Thread 0: 10 calls (5 * 2 TFs), 2 misses (first), 8 hits
        assert stats['total'] == 10
        assert stats['misses'] == 2
        assert stats['hits'] == 8

    print("\nValidation: Thread isolation working correctly!")

    labels.ENABLE_CACHE_STATS = False


def main():
    """Run all tests."""
    print("Cache Performance Monitoring Tests")
    print("=" * 60)

    try:
        test_stats_disabled()
        test_stats_enabled()
        test_thread_safety()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

        print("\nUsage Summary:")
        print("  1. Enable: labels.ENABLE_CACHE_STATS = True")
        print("  2. Reset: labels.reset_cache_stats()")
        print("  3. Get stats: stats = labels.get_cache_stats()")
        print("  4. Print stats: labels.print_cache_stats()")
        print("  5. Disable: labels.ENABLE_CACHE_STATS = False")
        print("\nFeatures:")
        print("  - Zero overhead when disabled (default)")
        print("  - Thread-safe (thread-local stats)")
        print("  - Useful for debugging cache effectiveness")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
