"""
Channel Pre-computation Tool

This script pre-computes ALL channels for the entire dataset across all timeframes
and window sizes, then saves them to disk in an efficient compressed format.

This eliminates the need to detect channels during training - just load from cache.

Usage:
    python precompute_channels.py [--data-dir PATH] [--output-dir PATH] [--workers N]

Output:
    - Compressed channel cache files organized by timeframe
    - Statistics on detection rates and cache sizes
    - Expected speedup estimates for training
"""

import numpy as np
import pandas as pd
import pickle
import gzip
import lz4.frame
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import time
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm
import psutil
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel
from core.timeframe import resample_ohlc, TIMEFRAMES, RESAMPLE_RULES


# Configuration
WINDOW_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
COMPRESSION = 'lz4'  # 'lz4', 'gzip', or 'pickle'


@dataclass
class CacheStats:
    """Statistics about the cache generation process."""
    timeframe: str
    total_bars: int
    total_channels_detected: int
    valid_channels: int
    invalid_channels: int
    valid_percentage: float
    cache_file_size_mb: float
    processing_time_sec: float
    bars_per_second: float
    window_stats: Dict[int, int]  # window -> count of valid channels


@dataclass
class ChannelCacheEntry:
    """Lightweight channel data for caching."""
    # Core metrics
    valid: bool
    direction: int
    slope: float
    intercept: float
    r_squared: float
    std_dev: float

    # Channel lines (as lists for serialization)
    upper_line: List[float]
    lower_line: List[float]
    center_line: List[float]

    # Touch data (simplified)
    touches: List[Tuple[int, int, float]]  # (bar_index, touch_type, price)
    complete_cycles: int
    bounce_count: int
    width_pct: float
    window: int

    # Optional price data (stored as lists)
    close: Optional[List[float]] = None
    high: Optional[List[float]] = None
    low: Optional[List[float]] = None

    @classmethod
    def from_channel(cls, channel: Channel, store_prices: bool = True):
        """Convert a Channel object to a cacheable entry."""
        return cls(
            valid=channel.valid,
            direction=int(channel.direction),
            slope=float(channel.slope),
            intercept=float(channel.intercept),
            r_squared=float(channel.r_squared),
            std_dev=float(channel.std_dev),
            upper_line=channel.upper_line.tolist(),
            lower_line=channel.lower_line.tolist(),
            center_line=channel.center_line.tolist(),
            touches=[(t.bar_index, int(t.touch_type), float(t.price)) for t in channel.touches],
            complete_cycles=channel.complete_cycles,
            bounce_count=channel.bounce_count,
            width_pct=float(channel.width_pct),
            window=channel.window,
            close=channel.close.tolist() if store_prices and channel.close is not None else None,
            high=channel.high.tolist() if store_prices and channel.high is not None else None,
            low=channel.low.tolist() if store_prices and channel.low is not None else None,
        )


class ChannelCache:
    """
    Manages the channel cache structure.

    Organization:
        {timeframe: {window: {bar_idx: ChannelCacheEntry}}}
    """

    def __init__(self):
        self.cache: Dict[str, Dict[int, Dict[int, ChannelCacheEntry]]] = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'compression': COMPRESSION,
            'timeframes': TIMEFRAMES,
            'window_sizes': WINDOW_SIZES,
        }

    def add_channel(self, timeframe: str, window: int, bar_idx: int, channel: Channel):
        """Add a channel to the cache."""
        if timeframe not in self.cache:
            self.cache[timeframe] = {}
        if window not in self.cache[timeframe]:
            self.cache[timeframe][window] = {}

        self.cache[timeframe][window][bar_idx] = ChannelCacheEntry.from_channel(channel)

    def get_channel(self, timeframe: str, window: int, bar_idx: int) -> Optional[ChannelCacheEntry]:
        """Retrieve a channel from cache."""
        return self.cache.get(timeframe, {}).get(window, {}).get(bar_idx)

    def save(self, output_path: Path, compression: str = COMPRESSION):
        """Save cache to disk with compression."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            'metadata': self.metadata,
            'cache': self.cache
        }

        # Pickle the data
        pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        # Compress and save
        if compression == 'lz4':
            compressed = lz4.frame.compress(pickled_data, compression_level=9)
            with open(output_path, 'wb') as f:
                f.write(compressed)
        elif compression == 'gzip':
            with gzip.open(output_path, 'wb', compresslevel=9) as f:
                f.write(pickled_data)
        else:  # pickle only
            with open(output_path, 'wb') as f:
                f.write(pickled_data)

        return output_path.stat().st_size

    @classmethod
    def load(cls, cache_path: Path, compression: str = COMPRESSION):
        """Load cache from disk."""
        # Read compressed data
        if compression == 'lz4':
            with open(cache_path, 'rb') as f:
                compressed = f.read()
            pickled_data = lz4.frame.decompress(compressed)
        elif compression == 'gzip':
            with gzip.open(cache_path, 'rb') as f:
                pickled_data = f.read()
        else:  # pickle only
            with open(cache_path, 'rb') as f:
                pickled_data = f.read()

        # Unpickle
        data = pickle.loads(pickled_data)

        # Create cache object
        cache = cls()
        cache.metadata = data['metadata']
        cache.cache = data['cache']

        return cache


def load_and_resample_data(
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load raw data and resample to ALL timeframes.

    Returns:
        Tuple of (tsla_resampled, spy_resampled) where each is a dict of {timeframe: df}
    """
    print("\n" + "="*80)
    print("LOADING AND RESAMPLING DATA")
    print("="*80)

    # Load TSLA 1min data
    print("\nLoading TSLA 1min data...")
    tsla_path = data_dir / "TSLA_1min.csv"
    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]

    # Load SPY 1min data
    print("Loading SPY 1min data...")
    spy_path = data_dir / "SPY_1min.csv"
    spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df.columns = [c.lower() for c in spy_df.columns]

    # Filter by date range if provided
    if start_date:
        tsla_df = tsla_df[tsla_df.index >= start_date]
        spy_df = spy_df[spy_df.index >= start_date]

    if end_date:
        tsla_df = tsla_df[tsla_df.index <= end_date]
        spy_df = spy_df[spy_df.index <= end_date]

    print(f"TSLA: {len(tsla_df):,} bars ({tsla_df.index[0]} to {tsla_df.index[-1]})")
    print(f"SPY: {len(spy_df):,} bars ({spy_df.index[0]} to {spy_df.index[-1]})")

    # Resample to all timeframes
    print("\nResampling to all timeframes...")
    tsla_resampled = {}
    spy_resampled = {}

    for tf in tqdm(TIMEFRAMES, desc="Resampling"):
        tsla_resampled[tf] = resample_ohlc(tsla_df, tf)
        spy_resampled[tf] = resample_ohlc(spy_df, tf)
        print(f"  {tf:10s}: {len(tsla_resampled[tf]):6,} bars")

    return tsla_resampled, spy_resampled


def detect_channels_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    window_sizes: List[int]
) -> Tuple[Dict[int, Dict[int, Channel]], CacheStats]:
    """
    Detect channels at all window sizes for a single timeframe.

    Args:
        df: Resampled OHLCV data for this timeframe
        timeframe: Timeframe name (e.g., '5min', '1h')
        window_sizes: List of window sizes to compute

    Returns:
        Tuple of (channels_dict, stats) where channels_dict is {window: {bar_idx: Channel}}
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING TIMEFRAME: {timeframe.upper()}")
    print(f"{'='*80}")
    print(f"Total bars: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    start_time = time.time()

    channels = {w: {} for w in window_sizes}
    total_channels = 0
    valid_channels = 0
    window_valid_counts = {w: 0 for w in window_sizes}

    # For each window size
    for window in window_sizes:
        if len(df) < window:
            print(f"  Skipping window={window} (insufficient data)")
            continue

        # Detect channels at every bar position from window onwards
        bar_indices = range(window, len(df) + 1)

        print(f"\n  Window {window:3d}: Processing {len(bar_indices):,} positions...")

        for bar_idx in tqdm(bar_indices, desc=f"    Window {window}", leave=False):
            # Get data up to this bar
            df_slice = df.iloc[:bar_idx]

            # Detect channel
            try:
                channel = detect_channel(df_slice, window=window)
                channels[window][bar_idx - 1] = channel  # Store at ending index

                total_channels += 1
                if channel.valid:
                    valid_channels += 1
                    window_valid_counts[window] += 1

            except Exception as e:
                # Skip on error (e.g., insufficient data)
                continue

        valid_pct = (window_valid_counts[window] / len(channels[window]) * 100) if channels[window] else 0
        print(f"    Valid channels: {window_valid_counts[window]:,} / {len(channels[window]):,} ({valid_pct:.1f}%)")

    end_time = time.time()
    processing_time = end_time - start_time
    bars_per_sec = total_channels / processing_time if processing_time > 0 else 0

    # Calculate statistics
    valid_pct = (valid_channels / total_channels * 100) if total_channels > 0 else 0

    stats = CacheStats(
        timeframe=timeframe,
        total_bars=len(df),
        total_channels_detected=total_channels,
        valid_channels=valid_channels,
        invalid_channels=total_channels - valid_channels,
        valid_percentage=valid_pct,
        cache_file_size_mb=0.0,  # Will be updated after saving
        processing_time_sec=processing_time,
        bars_per_second=bars_per_sec,
        window_stats=window_valid_counts
    )

    print(f"\n  Summary:")
    print(f"    Total channels detected: {total_channels:,}")
    print(f"    Valid channels: {valid_channels:,} ({valid_pct:.1f}%)")
    print(f"    Processing time: {processing_time:.1f}s")
    print(f"    Speed: {bars_per_sec:.1f} channels/sec")

    return channels, stats


def precompute_all_channels(
    data_dir: Path,
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframes: Optional[List[str]] = None,
    window_sizes: Optional[List[int]] = None
) -> Dict[str, CacheStats]:
    """
    Main function to pre-compute all channels.

    Args:
        data_dir: Directory containing input CSV files
        output_dir: Directory to save cache files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        timeframes: List of timeframes to process (default: all)
        window_sizes: List of window sizes (default: WINDOW_SIZES)

    Returns:
        Dict of {timeframe: CacheStats}
    """
    if timeframes is None:
        timeframes = TIMEFRAMES
    if window_sizes is None:
        window_sizes = WINDOW_SIZES

    print("\n" + "="*80)
    print("CHANNEL PRE-COMPUTATION PIPELINE")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Date range: {start_date or 'All'} to {end_date or 'All'}")
    print(f"Timeframes: {len(timeframes)}")
    print(f"Window sizes: {window_sizes}")
    print(f"Compression: {COMPRESSION}")

    # Get memory info
    mem = psutil.virtual_memory()
    print(f"\nSystem Memory: {mem.total / 1e9:.1f} GB total, {mem.available / 1e9:.1f} GB available")

    overall_start = time.time()

    # Step 1: Load and resample data
    tsla_resampled, spy_resampled = load_and_resample_data(data_dir, start_date, end_date)

    # Step 2: Process each timeframe
    all_stats = {}

    for tf in timeframes:
        df = tsla_resampled[tf]

        # Detect channels for this timeframe
        channels_dict, stats = detect_channels_for_timeframe(df, tf, window_sizes)

        # Create cache for this timeframe
        cache = ChannelCache()
        cache.metadata['timeframe'] = tf

        for window, window_channels in channels_dict.items():
            for bar_idx, channel in window_channels.items():
                cache.add_channel(tf, window, bar_idx, channel)

        # Save cache
        cache_path = output_dir / f"channel_cache_{tf}.{COMPRESSION}"
        print(f"\n  Saving cache to {cache_path}...")
        file_size = cache.save(cache_path, compression=COMPRESSION)

        stats.cache_file_size_mb = file_size / 1e6
        print(f"  Cache size: {stats.cache_file_size_mb:.2f} MB")

        all_stats[tf] = stats

        # Memory check
        mem = psutil.virtual_memory()
        print(f"  Memory used: {(mem.total - mem.available) / 1e9:.1f} GB / {mem.total / 1e9:.1f} GB ({mem.percent:.1f}%)")

    overall_time = time.time() - overall_start

    # Step 3: Save summary statistics
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)

    summary = {
        'total_processing_time_sec': overall_time,
        'total_processing_time_human': str(timedelta(seconds=int(overall_time))),
        'compression': COMPRESSION,
        'window_sizes': window_sizes,
        'timeframe_stats': {tf: asdict(stats) for tf, stats in all_stats.items()}
    }

    summary_path = output_dir / "cache_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")

    # Print final report
    print_final_report(all_stats, overall_time, output_dir)

    return all_stats


def print_final_report(stats_dict: Dict[str, CacheStats], total_time: float, output_dir: Path):
    """Print a comprehensive final report."""
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)

    # Aggregate statistics
    total_channels = sum(s.total_channels_detected for s in stats_dict.values())
    total_valid = sum(s.valid_channels for s in stats_dict.values())
    total_cache_size_mb = sum(s.cache_file_size_mb for s in stats_dict.values())

    print(f"\n{'Timeframe':<12} {'Total Bars':<12} {'Channels':<12} {'Valid':<12} {'Valid %':<10} {'Cache (MB)':<12} {'Time (s)':<10} {'Speed':<12}")
    print("-" * 100)

    for tf, stats in stats_dict.items():
        print(f"{tf:<12} {stats.total_bars:<12,} {stats.total_channels_detected:<12,} {stats.valid_channels:<12,} "
              f"{stats.valid_percentage:<10.1f} {stats.cache_file_size_mb:<12.2f} {stats.processing_time_sec:<10.1f} "
              f"{stats.bars_per_second:<12.1f}")

    print("-" * 100)
    print(f"{'TOTAL':<12} {'':<12} {total_channels:<12,} {total_valid:<12,} "
          f"{(total_valid/total_channels*100):<10.1f} {total_cache_size_mb:<12.2f} {total_time:<10.1f}")

    print(f"\n{'='*80}")
    print("CACHE DETAILS")
    print("="*80)
    print(f"Total cache size: {total_cache_size_mb:.2f} MB ({total_cache_size_mb/1024:.2f} GB)")
    print(f"Compression: {COMPRESSION}")
    print(f"Output directory: {output_dir}")

    # Estimate speedup
    print(f"\n{'='*80}")
    print("EXPECTED SPEEDUP")
    print("="*80)
    print("\nDuring training, loading pre-computed channels from cache will:")
    print(f"  - Eliminate {total_channels:,} channel detections")
    print(f"  - Save ~{total_time:.1f}s of computation time per epoch")
    print(f"  - Reduce memory overhead (no need to store raw OHLCV for detection)")
    print(f"  - Enable instant access to multi-timeframe channel data")
    print(f"\nExpected speedup: 10-50x faster feature extraction")

    # Window size breakdown
    print(f"\n{'='*80}")
    print("VALID CHANNELS BY WINDOW SIZE")
    print("="*80)

    # Aggregate window stats across all timeframes
    window_totals = {}
    for stats in stats_dict.values():
        for window, count in stats.window_stats.items():
            window_totals[window] = window_totals.get(window, 0) + count

    print(f"\n{'Window':<10} {'Valid Channels':<15} {'Percentage':<10}")
    print("-" * 40)
    for window in sorted(window_totals.keys()):
        count = window_totals[window]
        pct = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"{window:<10} {count:<15,} {pct:<10.1f}")

    print("\n" + "="*80)
    print("PRE-COMPUTATION COMPLETE!")
    print("="*80)
    print(f"\nCache files saved to: {output_dir}")
    print(f"Load with: ChannelCache.load('{output_dir}/channel_cache_<timeframe>.{COMPRESSION}')")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-compute channel cache for entire dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all data with default settings
  python precompute_channels.py

  # Process specific date range
  python precompute_channels.py --start-date 2020-01-01 --end-date 2023-12-31

  # Use custom output directory
  python precompute_channels.py --output-dir /path/to/cache

  # Process only specific timeframes
  python precompute_channels.py --timeframes 5min 15min 1h daily
        """
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data',
        help='Directory containing input CSV files (default: ../data)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'channel_cache',
        help='Directory to save cache files (default: ../data/channel_cache)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=None,
        help='Timeframes to process (default: all)'
    )

    parser.add_argument(
        '--windows',
        type=int,
        nargs='+',
        default=None,
        help='Window sizes to process (default: [10, 15, ..., 100])'
    )

    args = parser.parse_args()

    # Run pre-computation
    try:
        stats = precompute_all_channels(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframes=args.timeframes,
            window_sizes=args.windows
        )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during pre-computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
