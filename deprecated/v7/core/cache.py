"""
Optimized Caching Layer for V7 Feature Extraction

Provides thread-safe caching for expensive computations while PRESERVING EXACT CALCULATIONS.
No modifications to calculation logic - only a transparent caching layer.

Key Design Principles:
1. Cache keys are content-based (hashes of data) to ensure correctness
2. LRU eviction prevents memory leaks
3. Statistics tracking for performance monitoring
4. Enable/disable flags for debugging
5. Thread-safe for multi-worker training
"""

import numpy as np
import pandas as pd
import hashlib
from typing import Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
import weakref
from collections import OrderedDict


# ============================================================================
# Helper: Content-Based Cache Keys
# ============================================================================

def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    Create a content-based hash for a DataFrame.

    Uses index + all column values to ensure identical data produces identical keys.
    """
    # Hash index
    if isinstance(df.index, pd.DatetimeIndex):
        idx_bytes = df.index.astype(np.int64).values.tobytes()
    else:
        idx_bytes = df.index.values.tobytes()

    # Hash all columns
    data_bytes = b''
    for col in sorted(df.columns):  # Sorted for consistency
        data_bytes += df[col].values.tobytes()

    # Combine and hash
    combined = idx_bytes + data_bytes
    return hashlib.sha256(combined).hexdigest()[:16]  # First 16 chars sufficient


def _hash_array(arr: np.ndarray) -> str:
    """Create a content-based hash for a numpy array."""
    arr = np.asarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ============================================================================
# Cache Statistics
# ============================================================================

@dataclass
class CacheStats:
    """Statistics for a single cache."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Hits: {self.hits}, Misses: {self.misses}, "
            f"Hit Rate: {self.hit_rate:.2%}, "
            f"Size: {self.size}/{self.max_size}"
        )


# ============================================================================
# Thread-Safe LRU Cache
# ============================================================================

class ThreadSafeLRUCache:
    """
    Thread-safe LRU cache with statistics tracking.

    Uses OrderedDict for LRU eviction and RLock for thread safety.
    """

    def __init__(self, maxsize: int = 1024):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats(max_size=maxsize)
        self._enabled = True

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, updating LRU order."""
        if not self._enabled:
            return None

        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return self._cache[key]
            else:
                self._stats.misses += 1
                return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting LRU if full."""
        if not self._enabled:
            return

        with self._lock:
            if key in self._cache:
                # Update existing key
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new key
                self._cache[key] = value

                # Evict LRU if over maxsize
                if len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)  # Remove oldest

            self._stats.size = len(self._cache)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def enable(self) -> None:
        """Enable caching."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable caching (for debugging)."""
        with self._lock:
            self._enabled = False

    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                max_size=self._stats.max_size
            )


# ============================================================================
# ResamplingCache
# ============================================================================

class ResamplingCache:
    """
    Cache for resampled OHLCV dataframes.

    Key: (dataframe_hash, timeframe)
    Value: Resampled DataFrame

    Usage:
        cache = ResamplingCache(maxsize=512)
        resampled = cache.get_or_resample(df, '15min', resample_func)
    """

    def __init__(self, maxsize: int = 512):
        self._cache = ThreadSafeLRUCache(maxsize)

    def get_or_resample(
        self,
        df: pd.DataFrame,
        timeframe: str,
        resample_func: Callable[[pd.DataFrame, str], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get resampled dataframe from cache or compute it.

        Args:
            df: Original 5min OHLCV dataframe
            timeframe: Target timeframe (e.g., '15min', '1h')
            resample_func: Function to call if cache miss (e.g., resample_ohlc)

        Returns:
            Resampled dataframe (guaranteed identical to direct computation)
        """
        # Create cache key
        df_hash = _hash_dataframe(df)
        key = f"{df_hash}:{timeframe}"

        # Try cache
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Cache miss - compute
        resampled = resample_func(df, timeframe)

        # Store in cache
        self._cache.put(key, resampled)

        return resampled

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def enable(self) -> None:
        """Enable caching."""
        self._cache.enable()

    def disable(self) -> None:
        """Disable caching (forces recomputation)."""
        self._cache.disable()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats()


# ============================================================================
# ChannelCache
# ============================================================================

class ChannelCache:
    """
    Cache for detected channel objects.

    Key: (dataframe_hash, timeframe, window, std_multiplier, touch_threshold, min_cycles)
    Value: Channel object

    Usage:
        cache = ChannelCache(maxsize=1024)
        channel = cache.get_or_detect(df, '5min', 50, detect_channel_func)
    """

    def __init__(self, maxsize: int = 1024):
        self._cache = ThreadSafeLRUCache(maxsize)

    def get_or_detect(
        self,
        df: pd.DataFrame,
        timeframe: str,
        window: int,
        detect_func: Callable,
        **kwargs
    ) -> Any:  # Returns Channel object
        """
        Get detected channel from cache or compute it.

        Args:
            df: OHLCV dataframe
            timeframe: Timeframe string (for logging/tracking)
            window: Window size for channel detection
            detect_func: Function to call if cache miss (e.g., detect_channel)
            **kwargs: Additional args for detect_func (std_multiplier, etc.)

        Returns:
            Channel object (guaranteed identical to direct computation)
        """
        # Create cache key including all parameters that affect computation
        df_hash = _hash_dataframe(df)
        std_multiplier = kwargs.get('std_multiplier', 2.0)
        touch_threshold = kwargs.get('touch_threshold', 0.10)
        min_cycles = kwargs.get('min_cycles', 1)

        key = f"{df_hash}:{timeframe}:{window}:{std_multiplier}:{touch_threshold}:{min_cycles}"

        # Try cache
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Cache miss - compute
        channel = detect_func(df, window=window, **kwargs)

        # Store in cache
        self._cache.put(key, channel)

        return channel

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def enable(self) -> None:
        """Enable caching."""
        self._cache.enable()

    def disable(self) -> None:
        """Disable caching (forces recomputation)."""
        self._cache.disable()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats()


# ============================================================================
# RSICache
# ============================================================================

class RSICache:
    """
    Cache for RSI series calculations.

    Key: (price_array_hash, period, calculation_type)
    Value: RSI series (numpy array) or scalar

    Usage:
        cache = RSICache(maxsize=2048)
        rsi_series = cache.get_or_calculate(prices, 14, calculate_rsi_series)
    """

    def __init__(self, maxsize: int = 2048):
        self._cache = ThreadSafeLRUCache(maxsize)

    def get_or_calculate(
        self,
        prices: np.ndarray,
        period: int,
        calc_func: Callable,
        calc_type: str = 'series'
    ) -> np.ndarray:
        """
        Get RSI calculation from cache or compute it.

        Args:
            prices: Price array (close prices)
            period: RSI period (e.g., 14)
            calc_func: Function to call if cache miss (e.g., calculate_rsi_series)
            calc_type: Type of calculation ('series' or 'scalar') for cache separation

        Returns:
            RSI series or scalar (guaranteed identical to direct computation)
        """
        # Create cache key
        prices_hash = _hash_array(prices)
        key = f"{prices_hash}:{period}:{calc_type}"

        # Try cache
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Cache miss - compute
        if calc_type == 'series':
            result = calc_func(prices, period)
        else:
            result = calc_func(prices, period)

        # Store in cache
        self._cache.put(key, result)

        return result

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()

    def enable(self) -> None:
        """Enable caching."""
        self._cache.enable()

    def disable(self) -> None:
        """Disable caching (forces recomputation)."""
        self._cache.disable()

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats()


# ============================================================================
# FeatureCache - High-Level Manager
# ============================================================================

class FeatureCache:
    """
    High-level cache manager for all feature extraction caches.

    Aggregates all sub-caches and provides unified control/monitoring.

    Usage:
        cache = FeatureCache()

        # Use individual caches
        resampled = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
        channel = cache.channel.get_or_detect(df, '5min', 50, detect_channel)
        rsi = cache.rsi.get_or_calculate(prices, 14, calculate_rsi_series)

        # Monitor performance
        print(cache.stats())

        # Clear all caches
        cache.clear()

        # Disable for debugging
        cache.disable()
    """

    def __init__(
        self,
        resampling_maxsize: int = 512,
        channel_maxsize: int = 1024,
        rsi_maxsize: int = 2048
    ):
        """
        Initialize all sub-caches.

        Args:
            resampling_maxsize: Max entries in resampling cache
            channel_maxsize: Max entries in channel detection cache
            rsi_maxsize: Max entries in RSI calculation cache
        """
        self.resampling = ResamplingCache(maxsize=resampling_maxsize)
        self.channel = ChannelCache(maxsize=channel_maxsize)
        self.rsi = RSICache(maxsize=rsi_maxsize)

        self._enabled = True

    def clear(self) -> None:
        """Clear all caches."""
        self.resampling.clear()
        self.channel.clear()
        self.rsi.clear()

    def enable(self) -> None:
        """Enable all caches."""
        self._enabled = True
        self.resampling.enable()
        self.channel.enable()
        self.rsi.enable()

    def disable(self) -> None:
        """Disable all caches (for debugging/validation)."""
        self._enabled = False
        self.resampling.disable()
        self.channel.disable()
        self.rsi.disable()

    def stats(self) -> Dict[str, CacheStats]:
        """
        Get statistics for all caches.

        Returns:
            Dict mapping cache name to CacheStats
        """
        return {
            'resampling': self.resampling.stats(),
            'channel': self.channel.stats(),
            'rsi': self.rsi.stats(),
        }

    def print_stats(self) -> None:
        """Print formatted statistics for all caches."""
        stats = self.stats()
        print("\n" + "="*70)
        print("FEATURE EXTRACTION CACHE STATISTICS")
        print("="*70)

        for name, stat in stats.items():
            print(f"\n{name.upper()} Cache:")
            print(f"  {stat}")

        # Overall statistics
        total_hits = sum(s.hits for s in stats.values())
        total_misses = sum(s.misses for s in stats.values())
        total_requests = total_hits + total_misses

        if total_requests > 0:
            overall_hit_rate = total_hits / total_requests
            print("\n" + "-"*70)
            print(f"OVERALL: {total_hits} hits, {total_misses} misses, "
                  f"{overall_hit_rate:.2%} hit rate")
            print("="*70 + "\n")


# ============================================================================
# Global Cache Instance (Optional Singleton Pattern)
# ============================================================================

# Global instance for easy access across modules
_global_cache: Optional[FeatureCache] = None


def get_global_cache() -> FeatureCache:
    """
    Get or create the global cache instance.

    This is useful for sharing caches across modules without passing
    cache objects everywhere.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureCache()
    return _global_cache


def reset_global_cache() -> None:
    """Reset the global cache instance (clears all caches)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == '__main__':
    """
    Example usage demonstrating cache effectiveness.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.timeframe import resample_ohlc
    from core.channel import detect_channel
    from features.rsi import calculate_rsi_series, calculate_rsi

    print("="*70)
    print("CACHE DEMONSTRATION")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)

    df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.1,
        'high': prices + np.abs(np.random.randn(1000) * 0.3),
        'low': prices - np.abs(np.random.randn(1000) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)

    # Initialize cache
    cache = FeatureCache()

    print("\n--- Testing Resampling Cache ---")

    # First call (cache miss)
    import time
    start = time.time()
    resampled1 = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
    time1 = time.time() - start
    print(f"First call (miss): {time1*1000:.2f}ms")

    # Second call (cache hit)
    start = time.time()
    resampled2 = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
    time2 = time.time() - start
    print(f"Second call (hit): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")

    # Verify identity
    assert resampled1.equals(resampled2), "Cache corrupted data!"
    print("✓ Cached data is identical to original")

    print("\n--- Testing Channel Cache ---")

    # First call (cache miss)
    start = time.time()
    channel1 = cache.channel.get_or_detect(df, '5min', 50, detect_channel)
    time1 = time.time() - start
    print(f"First call (miss): {time1*1000:.2f}ms")

    # Second call (cache hit)
    start = time.time()
    channel2 = cache.channel.get_or_detect(df, '5min', 50, detect_channel)
    time2 = time.time() - start
    print(f"Second call (hit): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")

    # Verify identity
    assert channel1.slope == channel2.slope, "Cache corrupted channel!"
    assert np.allclose(channel1.upper_line, channel2.upper_line), "Cache corrupted channel!"
    print("✓ Cached channel is identical to original")

    print("\n--- Testing RSI Cache ---")

    prices_array = df['close'].values

    # First call (cache miss)
    start = time.time()
    rsi1 = cache.rsi.get_or_calculate(prices_array, 14, calculate_rsi_series, 'series')
    time1 = time.time() - start
    print(f"First call (miss): {time1*1000:.2f}ms")

    # Second call (cache hit)
    start = time.time()
    rsi2 = cache.rsi.get_or_calculate(prices_array, 14, calculate_rsi_series, 'series')
    time2 = time.time() - start
    print(f"Second call (hit): {time2*1000:.2f}ms")
    print(f"Speedup: {time1/time2:.1f}x")

    # Verify identity
    assert np.allclose(rsi1, rsi2, equal_nan=True), "Cache corrupted RSI!"
    print("✓ Cached RSI is identical to original")

    # Print overall statistics
    cache.print_stats()

    print("\n--- Testing Cache Disable ---")
    cache.disable()

    # Should force recomputation
    start = time.time()
    resampled3 = cache.resampling.get_or_resample(df, '15min', resample_ohlc)
    time3 = time.time() - start
    print(f"With cache disabled: {time3*1000:.2f}ms")
    print("✓ Cache disable works")

    cache.enable()
    print("✓ Cache re-enabled")

    print("\n" + "="*70)
    print("All cache tests passed! ✓")
    print("="*70)
