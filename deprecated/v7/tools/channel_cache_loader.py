"""
Channel Cache Loader

Efficient loading and querying of pre-computed channel caches.
This module provides utilities to load cached channels during training.

Usage:
    from tools.channel_cache_loader import ChannelCacheManager

    # Initialize manager
    manager = ChannelCacheManager('/path/to/cache/dir')

    # Load all caches
    manager.load_all()

    # Query a channel
    channel = manager.get_channel('5min', window=50, bar_idx=1000)
"""

import numpy as np
import pandas as pd
import pickle
import gzip
import lz4.frame
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import Channel, Direction, TouchType, Touch


def convert_cache_entry_to_channel(entry) -> Channel:
    """
    Convert a cache entry (from any source) back to a full Channel object.

    This function works with ChannelCacheEntry objects from both
    precompute_channels.py and channel_cache_loader.py to avoid
    pickle deserialization issues.
    """
    # Reconstruct Touch objects
    touch_objects = [
        Touch(bar_index=t[0], touch_type=TouchType(t[1]), price=t[2])
        for t in entry.touches
    ]

    return Channel(
        valid=entry.valid,
        direction=Direction(entry.direction),
        slope=entry.slope,
        intercept=entry.intercept,
        r_squared=entry.r_squared,
        std_dev=entry.std_dev,
        upper_line=np.array(entry.upper_line),
        lower_line=np.array(entry.lower_line),
        center_line=np.array(entry.center_line),
        touches=touch_objects,
        complete_cycles=entry.complete_cycles,
        bounce_count=entry.bounce_count,
        width_pct=entry.width_pct,
        window=entry.window,
        close=np.array(entry.close) if entry.close is not None else None,
        high=np.array(entry.high) if entry.high is not None else None,
        low=np.array(entry.low) if entry.low is not None else None,
    )


@dataclass
class ChannelCacheEntry:
    """Lightweight channel data (matches precompute_channels.py)."""
    valid: bool
    direction: int
    slope: float
    intercept: float
    r_squared: float
    std_dev: float
    upper_line: List[float]
    lower_line: List[float]
    center_line: List[float]
    touches: List[Tuple[int, int, float]]
    complete_cycles: int
    bounce_count: int
    width_pct: float
    window: int
    close: Optional[List[float]] = None
    high: Optional[List[float]] = None
    low: Optional[List[float]] = None

    def to_channel(self) -> Channel:
        """Convert cache entry back to a full Channel object."""
        return convert_cache_entry_to_channel(self)


class ChannelCache:
    """Channel cache container (matches precompute_channels.py)."""

    def __init__(self):
        self.cache: Dict[str, Dict[int, Dict[int, ChannelCacheEntry]]] = {}
        self.metadata = {}

    def get_channel(self, timeframe: str, window: int, bar_idx: int) -> Optional[ChannelCacheEntry]:
        """Retrieve a channel from cache."""
        return self.cache.get(timeframe, {}).get(window, {}).get(bar_idx)

    @classmethod
    def load(cls, cache_path: Path, compression: str = 'lz4'):
        """Load cache from disk."""
        # Detect compression from extension if not specified
        if cache_path.suffix == '.lz4':
            compression = 'lz4'
        elif cache_path.suffix == '.gz':
            compression = 'gzip'
        elif cache_path.suffix == '.pkl':
            compression = 'pickle'

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


class ChannelCacheManager:
    """
    Manages multiple channel caches for efficient querying.

    This class loads all timeframe caches and provides a unified interface
    for querying channels during training.
    """

    def __init__(self, cache_dir: Path, compression: str = 'lz4'):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory containing cache files
            compression: Compression format ('lz4', 'gzip', or 'pickle')
        """
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        self.caches: Dict[str, ChannelCache] = {}
        self.loaded_timeframes: List[str] = []

    def load_timeframe(self, timeframe: str) -> bool:
        """
        Load cache for a specific timeframe.

        Args:
            timeframe: Timeframe to load (e.g., '5min', '1h')

        Returns:
            True if loaded successfully, False otherwise
        """
        cache_path = self.cache_dir / f"channel_cache_{timeframe}.{self.compression}"

        if not cache_path.exists():
            print(f"Warning: Cache file not found: {cache_path}")
            return False

        try:
            print(f"Loading {timeframe} cache from {cache_path}...")
            cache = ChannelCache.load(cache_path, compression=self.compression)
            self.caches[timeframe] = cache
            self.loaded_timeframes.append(timeframe)
            print(f"  Loaded {timeframe} cache successfully")
            return True
        except Exception as e:
            print(f"Error loading {timeframe} cache: {e}")
            return False

    def load_all(self, timeframes: Optional[List[str]] = None):
        """
        Load all available timeframe caches.

        Args:
            timeframes: List of timeframes to load (default: auto-detect from files)
        """
        if timeframes is None:
            # Auto-detect cache files
            pattern = f"channel_cache_*.{self.compression}"
            cache_files = list(self.cache_dir.glob(pattern))
            timeframes = [
                f.stem.replace('channel_cache_', '')
                for f in cache_files
            ]

        print(f"\nLoading {len(timeframes)} timeframe caches...")
        for tf in timeframes:
            self.load_timeframe(tf)

        print(f"\nLoaded {len(self.loaded_timeframes)} timeframes: {self.loaded_timeframes}")

    def get_channel(
        self,
        timeframe: str,
        window: int,
        bar_idx: int,
        as_object: bool = True
    ) -> Optional[Channel]:
        """
        Get a channel from the cache.

        Args:
            timeframe: Timeframe (e.g., '5min')
            window: Window size
            bar_idx: Bar index (position where channel ends)
            as_object: If True, return Channel object; if False, return ChannelCacheEntry

        Returns:
            Channel object or ChannelCacheEntry or None if not found
        """
        if timeframe not in self.caches:
            return None

        entry = self.caches[timeframe].get_channel(timeframe, window, bar_idx)

        if entry is None:
            return None

        if as_object:
            return convert_cache_entry_to_channel(entry)
        else:
            return entry

    def get_channels_multi_window(
        self,
        timeframe: str,
        bar_idx: int,
        windows: Optional[List[int]] = None,
        only_valid: bool = False
    ) -> Dict[int, Channel]:
        """
        Get channels for multiple window sizes at a specific bar.

        Args:
            timeframe: Timeframe to query
            bar_idx: Bar index
            windows: List of window sizes (default: all available)
            only_valid: If True, only return valid channels

        Returns:
            Dict mapping window size to Channel object
        """
        if timeframe not in self.caches:
            return {}

        cache = self.caches[timeframe]

        # Get available windows if not specified
        if windows is None:
            windows = list(cache.cache.get(timeframe, {}).keys())

        channels = {}
        for window in windows:
            entry = cache.get_channel(timeframe, window, bar_idx)
            if entry is not None:
                if only_valid and not entry.valid:
                    continue
                channels[window] = convert_cache_entry_to_channel(entry)

        return channels

    def get_best_channel(
        self,
        timeframe: str,
        bar_idx: int,
        windows: Optional[List[int]] = None
    ) -> Optional[Channel]:
        """
        Get the best valid channel (most cycles) for a given bar.

        Args:
            timeframe: Timeframe to query
            bar_idx: Bar index
            windows: List of window sizes to consider

        Returns:
            Best Channel or None if no valid channels found
        """
        channels = self.get_channels_multi_window(timeframe, bar_idx, windows, only_valid=True)

        if not channels:
            return None

        # Sort by complete_cycles (descending), then r_squared (descending)
        best = max(
            channels.values(),
            key=lambda c: (c.complete_cycles, c.r_squared)
        )

        return best

    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded caches.

        Returns:
            Dict with cache statistics
        """
        stats = {
            'loaded_timeframes': self.loaded_timeframes,
            'cache_dir': str(self.cache_dir),
            'compression': self.compression,
            'timeframe_details': {}
        }

        for tf in self.loaded_timeframes:
            cache = self.caches[tf]
            tf_cache = cache.cache.get(tf, {})

            # Count channels per window
            window_counts = {}
            total_channels = 0
            valid_channels = 0

            for window, window_data in tf_cache.items():
                count = len(window_data)
                valid_count = sum(1 for entry in window_data.values() if entry.valid)
                window_counts[window] = {'total': count, 'valid': valid_count}
                total_channels += count
                valid_channels += valid_count

            stats['timeframe_details'][tf] = {
                'total_channels': total_channels,
                'valid_channels': valid_channels,
                'valid_percentage': (valid_channels / total_channels * 100) if total_channels > 0 else 0,
                'window_sizes': list(tf_cache.keys()),
                'window_counts': window_counts,
                'metadata': cache.metadata
            }

        return stats

    def print_statistics(self):
        """Print cache statistics in a readable format."""
        stats = self.get_statistics()

        print("\n" + "="*80)
        print("CHANNEL CACHE STATISTICS")
        print("="*80)
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Compression: {stats['compression']}")
        print(f"Loaded timeframes: {len(stats['loaded_timeframes'])}")

        print(f"\n{'Timeframe':<12} {'Windows':<10} {'Total':<12} {'Valid':<12} {'Valid %':<10}")
        print("-" * 60)

        for tf, details in stats['timeframe_details'].items():
            print(f"{tf:<12} {len(details['window_sizes']):<10} "
                  f"{details['total_channels']:<12,} {details['valid_channels']:<12,} "
                  f"{details['valid_percentage']:<10.1f}")

        print("\n" + "="*80)


def benchmark_cache_loading(cache_dir: Path, num_queries: int = 10000):
    """
    Benchmark cache loading and query performance.

    Args:
        cache_dir: Directory containing cache files
        num_queries: Number of random queries to perform
    """
    import time
    import random

    print("\n" + "="*80)
    print("CACHE PERFORMANCE BENCHMARK")
    print("="*80)

    # Load caches
    start = time.time()
    manager = ChannelCacheManager(cache_dir)
    manager.load_all()
    load_time = time.time() - start

    print(f"\nCache loading time: {load_time:.2f}s")

    # Get cache statistics
    stats = manager.get_statistics()

    # Random queries
    print(f"\nPerforming {num_queries:,} random queries...")

    # Generate random query parameters
    queries = []
    for _ in range(num_queries):
        tf = random.choice(manager.loaded_timeframes)
        details = stats['timeframe_details'][tf]
        window = random.choice(details['window_sizes'])
        # Random bar index (we don't know valid indices, so some will miss)
        bar_idx = random.randint(window, 10000)
        queries.append((tf, window, bar_idx))

    # Time queries
    start = time.time()
    hits = 0
    for tf, window, bar_idx in queries:
        channel = manager.get_channel(tf, window, bar_idx)
        if channel is not None:
            hits += 1
    query_time = time.time() - start

    queries_per_sec = num_queries / query_time
    avg_query_time_us = (query_time / num_queries) * 1e6

    print(f"\nQuery performance:")
    print(f"  Total queries: {num_queries:,}")
    print(f"  Cache hits: {hits:,} ({hits/num_queries*100:.1f}%)")
    print(f"  Query time: {query_time:.3f}s")
    print(f"  Queries/sec: {queries_per_sec:,.0f}")
    print(f"  Avg query time: {avg_query_time_us:.2f} μs")

    print("\n" + "="*80)


if __name__ == '__main__':
    """Example usage and benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Channel cache loader and benchmark")
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'channel_cache',
        help='Directory containing cache files'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=10000,
        help='Number of queries for benchmark'
    )

    args = parser.parse_args()

    if args.benchmark:
        benchmark_cache_loading(args.cache_dir, args.num_queries)
    else:
        # Simple usage example
        manager = ChannelCacheManager(args.cache_dir)
        manager.load_all()
        manager.print_statistics()

        # Example query
        print("\nExample query:")
        channel = manager.get_channel('5min', window=50, bar_idx=1000)
        if channel:
            print(f"  Found channel: valid={channel.valid}, direction={channel.direction}, "
                  f"cycles={channel.complete_cycles}, r²={channel.r_squared:.3f}")
        else:
            print("  No channel found at this location")
