"""
Example: Using Pre-computed Channel Cache During Training

This script demonstrates how to integrate the pre-computed channel cache
into your training pipeline for massive speedup.

Before/After comparison:
  - WITHOUT cache: Detect channels on-the-fly (slow, 10-100ms per sample)
  - WITH cache: Load from disk (fast, ~1μs per sample)

Expected speedup: 10-50x faster feature extraction
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.channel_cache_loader import ChannelCacheManager
from core.channel import detect_channel
from core.timeframe import resample_ohlc, TIMEFRAMES
from features.full_features import extract_full_features


def compare_cache_vs_live_detection(
    data_dir: Path,
    cache_dir: Path,
    num_samples: int = 100
):
    """
    Compare performance of cached vs live channel detection.

    Args:
        data_dir: Directory with CSV files
        cache_dir: Directory with cache files
        num_samples: Number of samples to test
    """
    print("\n" + "="*80)
    print("CACHE vs LIVE DETECTION COMPARISON")
    print("="*80)

    # Load data
    print("\nLoading TSLA data...")
    tsla_path = data_dir / "TSLA_1min.csv"
    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]

    # Resample to 5min
    tsla_5min = resample_ohlc(tsla_df, '5min')
    print(f"Loaded {len(tsla_5min):,} bars (5min)")

    # Load cache
    print("\nLoading channel cache...")
    cache_manager = ChannelCacheManager(cache_dir)
    cache_manager.load_all()

    # Generate random sample positions
    window = 50
    start_idx = window
    end_idx = min(len(tsla_5min), 5000)  # Limit to avoid long waits

    sample_indices = np.random.randint(start_idx, end_idx, size=num_samples)

    print(f"\nTesting {num_samples} samples...")
    print(f"Sample indices: {start_idx} to {end_idx}")

    # Method 1: LIVE DETECTION (slow)
    print("\n" + "-"*80)
    print("Method 1: LIVE CHANNEL DETECTION (traditional approach)")
    print("-"*80)

    live_channels = []
    start_time = time.time()

    for i, bar_idx in enumerate(sample_indices):
        df_slice = tsla_5min.iloc[:bar_idx]
        channel = detect_channel(df_slice, window=window)
        live_channels.append(channel)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  Progress: {i+1}/{num_samples} ({rate:.1f} samples/sec, ETA: {eta:.1f}s)")

    live_time = time.time() - start_time
    live_rate = num_samples / live_time

    print(f"\n  Total time: {live_time:.2f}s")
    print(f"  Rate: {live_rate:.1f} samples/sec")
    print(f"  Avg per sample: {live_time/num_samples*1000:.2f}ms")

    # Method 2: CACHED CHANNELS (fast)
    print("\n" + "-"*80)
    print("Method 2: CACHED CHANNELS (new approach)")
    print("-"*80)

    cached_channels = []
    cache_hits = 0
    start_time = time.time()

    for bar_idx in sample_indices:
        channel = cache_manager.get_channel('5min', window=window, bar_idx=bar_idx - 1)
        if channel is not None:
            cache_hits += 1
        cached_channels.append(channel)

    cache_time = time.time() - start_time
    cache_rate = num_samples / cache_time

    print(f"\n  Total time: {cache_time:.4f}s")
    print(f"  Rate: {cache_rate:,.0f} samples/sec")
    print(f"  Avg per sample: {cache_time/num_samples*1e6:.2f}μs")
    print(f"  Cache hits: {cache_hits}/{num_samples} ({cache_hits/num_samples*100:.1f}%)")

    # Comparison
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS")
    print("="*80)

    speedup = live_time / cache_time if cache_time > 0 else float('inf')

    print(f"\nSpeedup: {speedup:.1f}x faster")
    print(f"Time saved per sample: {(live_time - cache_time)/num_samples*1000:.2f}ms")
    print(f"Time saved per 1000 samples: {(live_time - cache_time)*1000/num_samples:.1f}s")

    # Verify correctness (compare a few samples)
    print("\n" + "="*80)
    print("CORRECTNESS VERIFICATION")
    print("="*80)

    mismatches = 0
    for i in range(min(10, num_samples)):
        if cached_channels[i] is None:
            continue

        live = live_channels[i]
        cached = cached_channels[i]

        # Compare key metrics
        if (abs(live.slope - cached.slope) > 1e-6 or
            abs(live.r_squared - cached.r_squared) > 1e-6 or
            live.valid != cached.valid):
            mismatches += 1
            print(f"  Sample {i}: MISMATCH")
            print(f"    Live:   valid={live.valid}, slope={live.slope:.6f}, r²={live.r_squared:.6f}")
            print(f"    Cached: valid={cached.valid}, slope={cached.slope:.6f}, r²={cached.r_squared:.6f}")

    if mismatches == 0:
        print(f"\n✓ All {min(10, num_samples)} verified samples match perfectly!")
    else:
        print(f"\n✗ Found {mismatches} mismatches (this may indicate cache corruption)")

    print("\n" + "="*80)


def demonstrate_multi_timeframe_loading(cache_dir: Path):
    """
    Demonstrate loading channels from multiple timeframes simultaneously.

    This shows how the cache enables instant access to multi-timeframe data.
    """
    print("\n" + "="*80)
    print("MULTI-TIMEFRAME CACHE DEMONSTRATION")
    print("="*80)

    # Load cache
    print("\nLoading all timeframe caches...")
    cache_manager = ChannelCacheManager(cache_dir)
    cache_manager.load_all()

    # Print statistics
    cache_manager.print_statistics()

    # Example: Get channels at the same timestamp across multiple timeframes
    print("\n" + "-"*80)
    print("Example: Querying channels across timeframes at bar 1000")
    print("-"*80)

    bar_idx = 1000
    window = 50

    for tf in cache_manager.loaded_timeframes[:5]:  # Show first 5 timeframes
        channel = cache_manager.get_channel(tf, window=window, bar_idx=bar_idx)
        if channel:
            print(f"\n{tf:10s}: valid={channel.valid}, dir={channel.direction.name:8s}, "
                  f"cycles={channel.complete_cycles}, r²={channel.r_squared:.3f}, "
                  f"width={channel.width_pct:.2f}%")
        else:
            print(f"\n{tf:10s}: No channel found")

    # Example: Get best channel across multiple windows
    print("\n" + "-"*80)
    print("Example: Finding best channel across all window sizes")
    print("-"*80)

    best_channel = cache_manager.get_best_channel('5min', bar_idx=bar_idx)
    if best_channel:
        print(f"\nBest channel for 5min @ bar {bar_idx}:")
        print(f"  Window: {best_channel.window}")
        print(f"  Valid: {best_channel.valid}")
        print(f"  Direction: {best_channel.direction.name}")
        print(f"  Complete cycles: {best_channel.complete_cycles}")
        print(f"  R²: {best_channel.r_squared:.3f}")
        print(f"  Width: {best_channel.width_pct:.2f}%")
    else:
        print(f"\nNo valid channels found")

    print("\n" + "="*80)


def estimate_training_speedup(cache_dir: Path, num_samples: int = 10000):
    """
    Estimate total training speedup from using cached channels.

    Args:
        cache_dir: Directory with cache files
        num_samples: Typical number of training samples
    """
    print("\n" + "="*80)
    print("TRAINING SPEEDUP ESTIMATION")
    print("="*80)

    # Load cache
    cache_manager = ChannelCacheManager(cache_dir)
    cache_manager.load_all()

    # Estimate times
    avg_live_detection_ms = 15  # Typical time to detect one channel
    avg_cache_lookup_us = 1     # Typical time to look up from cache

    live_total_s = (avg_live_detection_ms * num_samples) / 1000
    cache_total_s = (avg_cache_lookup_us * num_samples) / 1e6
    speedup = live_total_s / cache_total_s if cache_total_s > 0 else float('inf')

    print(f"\nAssumptions:")
    print(f"  Training samples: {num_samples:,}")
    print(f"  Live detection time: {avg_live_detection_ms}ms per sample")
    print(f"  Cache lookup time: {avg_cache_lookup_us}μs per sample")

    print(f"\nEstimated training time per epoch:")
    print(f"  WITHOUT cache: {live_total_s:.1f}s ({live_total_s/60:.1f}min)")
    print(f"  WITH cache: {cache_total_s:.4f}s ({cache_total_s*1000:.1f}ms)")
    print(f"  Speedup: {speedup:.0f}x faster")

    print(f"\nFor 100 epochs:")
    print(f"  WITHOUT cache: {live_total_s*100/60:.1f}min ({live_total_s*100/3600:.1f}h)")
    print(f"  WITH cache: {cache_total_s*100:.2f}s ({cache_total_s*100/60:.2f}min)")
    print(f"  Time saved: {(live_total_s-cache_total_s)*100/60:.1f}min ({(live_total_s-cache_total_s)*100/3600:.1f}h)")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Channel cache usage examples")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data',
        help='Directory with CSV files'
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'channel_cache',
        help='Directory with cache files'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison benchmark'
    )
    parser.add_argument(
        '--multi-tf',
        action='store_true',
        help='Demonstrate multi-timeframe loading'
    )
    parser.add_argument(
        '--estimate',
        action='store_true',
        help='Estimate training speedup'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples for comparison'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all demonstrations'
    )

    args = parser.parse_args()

    if args.all:
        args.compare = True
        args.multi_tf = True
        args.estimate = True

    if args.compare:
        compare_cache_vs_live_detection(args.data_dir, args.cache_dir, args.num_samples)

    if args.multi_tf:
        demonstrate_multi_timeframe_loading(args.cache_dir)

    if args.estimate:
        estimate_training_speedup(args.cache_dir, num_samples=10000)

    if not (args.compare or args.multi_tf or args.estimate):
        print("No action specified. Use --help for options.")
        print("Quick start: python example_cache_usage.py --all")


if __name__ == '__main__':
    main()
