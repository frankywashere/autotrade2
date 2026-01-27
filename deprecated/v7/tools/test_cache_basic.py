"""
Basic test to verify cache system works correctly.

This is a minimal test using synthetic data to verify:
1. Channel detection works
2. Cache saving works
3. Cache loading works
4. Cached channels match original channels

Run this before running full pre-computation to ensure everything works.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Direction
from tools.precompute_channels import ChannelCache, ChannelCacheEntry


def create_synthetic_ohlcv(n_bars: int = 200, trend: float = 0.5) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate trending price with noise
    time = np.arange(n_bars)
    trend_line = 100 + trend * time
    noise = np.random.randn(n_bars) * 2

    close = trend_line + noise

    # Generate OHLC from close
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_ = close - np.random.randn(n_bars) * 0.3
    volume = np.random.randint(1000, 10000, size=n_bars)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

    return df


def test_channel_detection():
    """Test basic channel detection."""
    print("\nTest 1: Channel Detection")
    print("-" * 50)

    df = create_synthetic_ohlcv(200, trend=0.5)

    channel = detect_channel(df, window=50)

    print(f"  Valid: {channel.valid}")
    print(f"  Direction: {channel.direction.name}")
    print(f"  Slope: {channel.slope:.4f}")
    print(f"  R²: {channel.r_squared:.4f}")
    print(f"  Cycles: {channel.complete_cycles}")
    print(f"  Bounces: {channel.bounce_count}")
    print(f"  Width: {channel.width_pct:.2f}%")

    assert channel.window == 50, "Window mismatch"
    assert len(channel.close) == 50, "Data length mismatch"
    assert len(channel.upper_line) == 50, "Upper line length mismatch"

    print("  ✓ Channel detection works!")
    return channel


def test_cache_entry_conversion(channel):
    """Test conversion to/from cache entry."""
    print("\nTest 2: Cache Entry Conversion")
    print("-" * 50)

    # Convert to cache entry
    entry = ChannelCacheEntry.from_channel(channel)

    print(f"  Entry valid: {entry.valid}")
    print(f"  Entry direction: {entry.direction}")
    print(f"  Entry slope: {entry.slope:.4f}")
    print(f"  Entry r²: {entry.r_squared:.4f}")

    # Verify conversion
    assert entry.valid == channel.valid, "Valid mismatch"
    assert entry.direction == int(channel.direction), "Direction mismatch"
    assert abs(entry.slope - channel.slope) < 1e-10, "Slope mismatch"
    assert abs(entry.r_squared - channel.r_squared) < 1e-10, "R² mismatch"
    assert len(entry.upper_line) == channel.window, "Upper line length mismatch"

    print("  ✓ Cache entry conversion works!")
    return entry


def test_cache_save_load():
    """Test cache saving and loading."""
    print("\nTest 3: Cache Save/Load")
    print("-" * 50)

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"  Using temp dir: {temp_dir}")

    try:
        # Create cache
        cache = ChannelCache()

        # Generate some test channels
        df = create_synthetic_ohlcv(200, trend=0.5)

        for window in [20, 50, 100]:
            for bar_idx in range(window, min(window + 20, 200)):
                df_slice = df.iloc[:bar_idx]
                channel = detect_channel(df_slice, window=window)
                cache.add_channel('5min', window, bar_idx - 1, channel)

        print(f"  Added channels: {len(cache.cache.get('5min', {}))} windows")

        # Save cache
        for compression in ['lz4', 'gzip', 'pickle']:
            cache_path = temp_dir / f"test_cache.{compression}"
            print(f"\n  Testing {compression} compression...")

            file_size = cache.save(cache_path, compression=compression)
            print(f"    Saved: {file_size:,} bytes")

            # Load cache
            loaded_cache = ChannelCache.load(cache_path, compression=compression)
            print(f"    Loaded successfully")

            # Verify contents
            original_windows = set(cache.cache.get('5min', {}).keys())
            loaded_windows = set(loaded_cache.cache.get('5min', {}).keys())

            assert original_windows == loaded_windows, "Window sets don't match"

            # Check one channel
            window = 50
            bar_idx = 60
            original_entry = cache.get_channel('5min', window, bar_idx)
            loaded_entry = loaded_cache.get_channel('5min', window, bar_idx)

            assert loaded_entry is not None, "Failed to load channel"
            assert abs(loaded_entry.slope - original_entry.slope) < 1e-10, "Slope mismatch after reload"
            assert loaded_entry.valid == original_entry.valid, "Valid mismatch after reload"

            print(f"    ✓ {compression} compression works!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n  Cleaned up temp dir")

    print("  ✓ Cache save/load works!")


def test_cache_manager():
    """Test cache manager functionality."""
    print("\nTest 4: Cache Manager")
    print("-" * 50)

    from tools.channel_cache_loader import ChannelCacheManager

    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"  Using temp dir: {temp_dir}")

    try:
        # Create and save cache
        cache = ChannelCache()
        df = create_synthetic_ohlcv(200, trend=0.5)

        for window in [20, 50, 100]:
            for bar_idx in range(window, min(window + 20, 200)):
                df_slice = df.iloc[:bar_idx]
                channel = detect_channel(df_slice, window=window)
                cache.add_channel('5min', window, bar_idx - 1, channel)

        cache_path = temp_dir / "channel_cache_5min.lz4"
        cache.save(cache_path, compression='lz4')
        print(f"  Created test cache: {cache_path.name}")

        # Test manager
        manager = ChannelCacheManager(temp_dir, compression='lz4')
        manager.load_timeframe('5min')
        print(f"  Loaded timeframe: 5min")

        # Query channel
        channel = manager.get_channel('5min', window=50, bar_idx=60)
        print(f"  Queried channel: valid={channel.valid}, slope={channel.slope:.4f}")

        # Multi-window query
        channels = manager.get_channels_multi_window('5min', bar_idx=60)
        print(f"  Multi-window query: {len(channels)} windows")

        # Best channel
        best = manager.get_best_channel('5min', bar_idx=60)
        if best:
            print(f"  Best channel: window={best.window}, cycles={best.complete_cycles}")
        else:
            print(f"  Best channel: None found (no valid channels with cycles)")

        # Statistics
        stats = manager.get_statistics()
        print(f"  Statistics: {stats['timeframe_details']['5min']['total_channels']} total channels")

        print("  ✓ Cache manager works!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\n  Cleaned up temp dir")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("CHANNEL CACHE BASIC TESTS")
    print("="*70)

    try:
        # Run tests
        channel = test_channel_detection()
        entry = test_cache_entry_conversion(channel)
        test_cache_save_load()
        test_cache_manager()

        # Success
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nThe cache system is working correctly.")
        print("You can now run the full pre-computation:")
        print("  python3 v7/tools/precompute_channels.py")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
