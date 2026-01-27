"""
Live Data Fetcher for Real-Time Channel Prediction

Provides real-time data fetching with time-based caching to prevent excessive API calls.
Designed for live trading scenarios where data freshness is critical but rate limiting is necessary.

Key Features:
1. Simple TTL-based cache (5 minutes default)
2. Thread-safe operations
3. Cache key generation from symbol+interval
4. Automatic expiration checking
5. Statistics tracking for monitoring
6. Multi-symbol, multi-timeframe support
7. Data validation and quality checks
8. Configurable thresholds and error handling
"""

import time
import hashlib
from threading import RLock
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path
import sys

# Import timeframe utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from v7.core.timeframe import TIMEFRAMES, RESAMPLE_RULES


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LiveDataConfig:
    """
    Configuration for live data fetching.

    This dataclass centralizes all configuration parameters for fetching,
    caching, validation, and processing live market data for real-time
    channel prediction.

    Categories:
    - Symbol and timeframe configuration
    - Cache settings (TTL, directory, size limits)
    - Fetch settings (lookback, retries, timeouts)
    - Validation thresholds (data quality, freshness)
    - Quality checks (volume, outliers, market hours)
    - Rate limiting (API call throttling)
    - Error handling (partial data, fallbacks)
    - Logging configuration
    """

    # =========================================================================
    # Symbols Configuration
    # =========================================================================
    symbols: List[str] = field(default_factory=lambda: ['TSLA', 'SPY', '^VIX'])

    # =========================================================================
    # Timeframes Configuration
    # =========================================================================
    timeframes: List[str] = field(default_factory=lambda: [
        '5min', '15min', '30min', '1h', '2h', '3h', '4h',
        'daily', 'weekly', 'monthly', '3month'
    ])

    # =========================================================================
    # Cache Settings
    # =========================================================================
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60  # 1 minute default
    cache_dir: Path = field(default_factory=lambda: Path.home() / '.x6' / 'live_cache')
    cache_max_size: int = 1000  # Maximum number of cache entries

    # =========================================================================
    # Fetch Settings
    # =========================================================================
    lookback_days: int = 180  # 6 months for longest timeframes
    interval: str = '5m'  # Base interval for yfinance
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    request_timeout_seconds: int = 30

    # =========================================================================
    # Validation Thresholds
    # =========================================================================
    min_bars_required: Dict[str, int] = field(default_factory=lambda: {
        '5min': 100,
        '15min': 80,
        '30min': 60,
        '1h': 50,
        '2h': 40,
        '3h': 35,
        '4h': 30,
        'daily': 60,
        'weekly': 30,
        'monthly': 12,
        '3month': 4
    })

    max_price_change_pct: float = 20.0  # Flag changes >20% as suspicious
    max_missing_data_pct: float = 5.0   # Allow up to 5% missing bars

    require_recent_data: bool = True
    max_data_age_minutes: int = 15  # Data must be within last 15 minutes

    # =========================================================================
    # Quality Thresholds
    # =========================================================================
    min_volume_threshold: int = 100  # Minimum volume per bar
    outlier_std_threshold: float = 5.0  # Std devs for outlier detection
    require_market_hours: bool = True  # Only accept data during market hours

    # =========================================================================
    # Rate Limiting
    # =========================================================================
    rate_limit_calls_per_minute: int = 60
    rate_limit_enabled: bool = True

    # =========================================================================
    # Error Handling
    # =========================================================================
    allow_partial_data: bool = True  # Continue with some symbols missing
    fallback_on_error: bool = True   # Use cache on API failure
    raise_on_validation_error: bool = False  # Don't raise on validation issues

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = 'INFO'
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure cache directory is a Path object
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate symbols
        if not self.symbols:
            raise ValueError("Must specify at least one symbol")

        # Validate timeframes
        invalid_tfs = set(self.timeframes) - set(TIMEFRAMES)
        if invalid_tfs:
            raise ValueError(f"Invalid timeframes: {invalid_tfs}")

        # Validate intervals
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m',
                          '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if self.interval not in valid_intervals:
            raise ValueError(f"Invalid interval: {self.interval}. "
                           f"Must be one of {valid_intervals}")

        # Validate lookback period
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")

        if self.lookback_days > 730:  # 2 years max
            raise ValueError("lookback_days exceeds maximum (730 days)")

        # Validate thresholds
        if self.max_price_change_pct <= 0:
            raise ValueError("max_price_change_pct must be > 0")

        if not 0 <= self.max_missing_data_pct <= 100:
            raise ValueError("max_missing_data_pct must be between 0 and 100")

        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")

        # Validate rate limiting
        if self.rate_limit_calls_per_minute < 1:
            raise ValueError("rate_limit_calls_per_minute must be >= 1")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

    def get_cache_path(self, symbol: str) -> Path:
        """
        Get cache file path for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Path to cache file
        """
        # Clean symbol (remove ^ for file names)
        clean_symbol = symbol.replace('^', '').replace('/', '_')
        return self.cache_dir / f"{clean_symbol}_{self.interval}.pkl"

    def is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache file exists and is within TTL.

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.cache_enabled:
            return False

        if not cache_path.exists():
            return False

        # Check age
        mtime = cache_path.stat().st_mtime
        age_seconds = time.time() - mtime

        return age_seconds < self.cache_ttl_seconds

    def should_fetch_symbol(self, symbol: str) -> bool:
        """
        Determine if a symbol should be fetched (cache miss or expired).

        Args:
            symbol: Ticker symbol

        Returns:
            True if fetch is needed, False if cache is valid
        """
        if not self.cache_enabled:
            return True

        cache_path = self.get_cache_path(symbol)
        return not self.is_cache_valid(cache_path)

    def get_market_hours(self) -> Tuple[int, int]:
        """
        Get market hours for validation.

        Returns:
            Tuple of (market_open_hour, market_close_hour) in EST
        """
        # US market: 9:30 AM - 4:00 PM EST
        return (9, 16)

    def get_required_bars(self, timeframe: str) -> int:
        """
        Get minimum required bars for a timeframe.

        Args:
            timeframe: Timeframe string

        Returns:
            Minimum number of bars required
        """
        return self.min_bars_required.get(timeframe, 50)

    def to_dict(self) -> Dict:
        """
        Convert config to dictionary for serialization.

        Returns:
            Dictionary representation of config
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'LiveDataConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            LiveDataConfig instance
        """
        # Convert string paths back to Path objects
        if 'cache_dir' in config_dict and isinstance(config_dict['cache_dir'], str):
            config_dict['cache_dir'] = Path(config_dict['cache_dir'])

        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"LiveDataConfig(\n"
            f"  symbols={self.symbols},\n"
            f"  timeframes={len(self.timeframes)} timeframes,\n"
            f"  cache_ttl={self.cache_ttl_seconds}s,\n"
            f"  lookback_days={self.lookback_days},\n"
            f"  interval='{self.interval}'\n"
            f")"
        )


# ============================================================================
# Cache Entry
# ============================================================================

@dataclass
class CacheEntry:
    """
    Single cache entry with data and expiration timestamp.

    Attributes:
        data: The cached data (typically a DataFrame)
        timestamp: When the data was cached (seconds since epoch)
        ttl: Time-to-live in seconds
    """
    data: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return (time.time() - self.timestamp) > self.ttl

    def age(self) -> float:
        """Get age of this entry in seconds."""
        return time.time() - self.timestamp

    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds (negative if expired)."""
        return self.ttl - self.age()


# ============================================================================
# Cache Statistics
# ============================================================================

@dataclass
class LiveCacheStats:
    """Statistics for the live data cache."""
    hits: int = 0
    misses: int = 0
    expired: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1), excluding expired hits."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    def __str__(self) -> str:
        return (
            f"Requests: {self.total_requests}, "
            f"Hits: {self.hits}, Misses: {self.misses}, "
            f"Expired: {self.expired}, "
            f"Hit Rate: {self.hit_rate:.2%}, "
            f"Size: {self.size}"
        )


# ============================================================================
# Live Data Cache
# ============================================================================

class LiveDataCache:
    """
    Thread-safe TTL-based cache for live market data.

    Designed specifically for real-time data fetching where:
    - Data expires after a fixed time (default 5 minutes)
    - Cache keys are based on symbol + interval
    - Thread safety is critical for concurrent requests
    - Statistics help monitor cache performance

    Usage:
        cache = LiveDataCache(ttl=300)  # 5 minute TTL

        # Try to get cached data
        data = cache.get('BTCUSDT', '5m')
        if data is None:
            # Cache miss - fetch from API
            data = fetch_from_api('BTCUSDT', '5m')
            cache.set('BTCUSDT', '5m', data)

        # Monitor performance
        print(cache.stats())
    """

    def __init__(self, ttl: float = 300.0, max_size: int = 1000):
        """
        Initialize the live data cache.

        Args:
            ttl: Time-to-live in seconds (default 300 = 5 minutes)
            max_size: Maximum number of entries to store (default 1000)
        """
        self.ttl = ttl
        self.max_size = max_size

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._stats = LiveCacheStats()
        self._enabled = True

    def _generate_key(self, symbol: str, interval: str) -> str:
        """
        Generate cache key from symbol and interval.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '5m', '1h')

        Returns:
            Cache key string
        """
        # Simple key format: symbol:interval
        # Could add hashing if needed, but simple keys are easier to debug
        return f"{symbol.upper()}:{interval.lower()}"

    def get(self, symbol: str, interval: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '5m', '1h')

        Returns:
            Cached data if available and fresh, None otherwise
        """
        if not self._enabled:
            return None

        key = self._generate_key(symbol, interval)

        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._stats.expired += 1
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Valid cache hit
            self._stats.hits += 1
            return entry.data

    def set(self, symbol: str, interval: str, data: Any, ttl: Optional[float] = None) -> None:
        """
        Store data in cache with TTL.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '5m', '1h')
            data: Data to cache (typically a DataFrame)
            ttl: Optional custom TTL for this entry (uses default if None)
        """
        if not self._enabled:
            return

        key = self._generate_key(symbol, interval)
        entry_ttl = ttl if ttl is not None else self.ttl

        with self._lock:
            # Create new entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=entry_ttl
            )

            # Store entry
            self._cache[key] = entry

            # Check if we need to evict old entries
            if len(self._cache) > self.max_size:
                self._evict_oldest()

            self._stats.size = len(self._cache)

    def _evict_oldest(self) -> None:
        """
        Evict the oldest cache entry (lowest timestamp).

        Note: This is called when the cache is full. We evict by age
        rather than LRU since data staleness is the primary concern.
        """
        if not self._cache:
            return

        # Find entry with oldest timestamp
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].timestamp
        )

        del self._cache[oldest_key]
        self._stats.evictions += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0

    def remove(self, symbol: str, interval: str) -> bool:
        """
        Remove a specific entry from cache.

        Args:
            symbol: Trading symbol
            interval: Time interval

        Returns:
            True if entry was removed, False if not found
        """
        key = self._generate_key(symbol, interval)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of expired entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                self._stats.size = len(self._cache)
                self._stats.expired += len(expired_keys)

            return len(expired_keys)

    def get_entry_info(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cache entry without returning the data.

        Args:
            symbol: Trading symbol
            interval: Time interval

        Returns:
            Dict with entry info (age, ttl, etc.) or None if not found
        """
        key = self._generate_key(symbol, interval)

        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            return {
                'symbol': symbol,
                'interval': interval,
                'age_seconds': entry.age(),
                'remaining_ttl': entry.remaining_ttl(),
                'is_expired': entry.is_expired(),
                'cached_at': datetime.fromtimestamp(entry.timestamp),
                'expires_at': datetime.fromtimestamp(entry.timestamp + entry.ttl)
            }

    def list_entries(self) -> list[Dict[str, Any]]:
        """
        List all cache entries with their info.

        Returns:
            List of dicts containing entry information
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                # Parse key back to symbol and interval
                symbol, interval = key.split(':', 1)
                entries.append({
                    'symbol': symbol,
                    'interval': interval,
                    'age_seconds': entry.age(),
                    'remaining_ttl': entry.remaining_ttl(),
                    'is_expired': entry.is_expired(),
                    'cached_at': datetime.fromtimestamp(entry.timestamp),
                    'expires_at': datetime.fromtimestamp(entry.timestamp + entry.ttl)
                })

            # Sort by age (newest first)
            entries.sort(key=lambda x: x['age_seconds'])
            return entries

    def enable(self) -> None:
        """Enable caching."""
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        """Disable caching (for debugging)."""
        with self._lock:
            self._enabled = False

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        with self._lock:
            return self._enabled

    def stats(self) -> LiveCacheStats:
        """
        Get current cache statistics.

        Returns:
            LiveCacheStats object with current statistics
        """
        with self._lock:
            return LiveCacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                expired=self._stats.expired,
                evictions=self._stats.evictions,
                size=self._stats.size
            )

    def reset_stats(self) -> None:
        """Reset statistics counters (but keep cache data)."""
        with self._lock:
            self._stats.hits = 0
            self._stats.misses = 0
            self._stats.expired = 0
            self._stats.evictions = 0
            # Keep size as-is

    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.stats()
        print("\n" + "="*70)
        print("LIVE DATA CACHE STATISTICS")
        print("="*70)
        print(f"  {stats}")
        print(f"  TTL: {self.ttl}s ({self.ttl/60:.1f} minutes)")
        print(f"  Max Size: {self.max_size}")
        print(f"  Enabled: {self._enabled}")
        print("="*70 + "\n")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == '__main__':
    """
    Example usage demonstrating the cache functionality.
    """
    print("="*70)
    print("LIVE DATA CACHE DEMONSTRATION")
    print("="*70)

    # Create cache with 5 minute TTL
    cache = LiveDataCache(ttl=300)

    print("\n--- Testing Basic Operations ---")

    # Simulate fetching data
    print("Fetching BTCUSDT 5m data...")
    data = cache.get('BTCUSDT', '5m')
    print(f"Cache result: {data}")  # Should be None (miss)

    # Store some data
    print("\nStoring data in cache...")
    sample_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    })
    cache.set('BTCUSDT', '5m', sample_data)

    # Fetch again (should be hit)
    print("\nFetching BTCUSDT 5m data again...")
    data = cache.get('BTCUSDT', '5m')
    print(f"Cache result: {'HIT' if data is not None else 'MISS'}")

    print("\n--- Cache Entry Info ---")
    info = cache.get_entry_info('BTCUSDT', '5m')
    if info:
        print(f"Symbol: {info['symbol']}")
        print(f"Interval: {info['interval']}")
        print(f"Age: {info['age_seconds']:.2f}s")
        print(f"Remaining TTL: {info['remaining_ttl']:.2f}s")
        print(f"Cached at: {info['cached_at']}")
        print(f"Expires at: {info['expires_at']}")

    # Add more entries
    print("\n--- Adding More Entries ---")
    cache.set('ETHUSDT', '5m', sample_data)
    cache.set('BTCUSDT', '1h', sample_data)
    cache.set('ETHUSDT', '1h', sample_data)

    print("\nAll cache entries:")
    for entry in cache.list_entries():
        print(f"  {entry['symbol']:10} {entry['interval']:5} - "
              f"age: {entry['age_seconds']:6.2f}s, "
              f"remaining: {entry['remaining_ttl']:6.2f}s")

    print("\n--- Testing Expiration ---")

    # Create a cache with very short TTL for testing
    short_cache = LiveDataCache(ttl=2.0)  # 2 seconds
    short_cache.set('TEST', '5m', sample_data)

    print("Fetching immediately...")
    data = short_cache.get('TEST', '5m')
    print(f"Result: {'HIT' if data is not None else 'MISS'}")

    print("Waiting 3 seconds...")
    time.sleep(3)

    print("Fetching after expiration...")
    data = short_cache.get('TEST', '5m')
    print(f"Result: {'HIT' if data is not None else 'MISS'}")

    # Print statistics
    cache.print_stats()

    print("\n--- Testing Thread Safety ---")
    import threading

    # Test concurrent access
    def worker(cache, symbol, interval, iterations):
        for i in range(iterations):
            data = cache.get(symbol, interval)
            if data is None:
                cache.set(symbol, interval, sample_data)

    threads = []
    for i in range(5):
        t = threading.Thread(
            target=worker,
            args=(cache, f'SYM{i}', '5m', 10)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Concurrent access completed successfully")
    cache.print_stats()

    print("\n--- Testing Cache Cleanup ---")
    print(f"Entries before cleanup: {cache.stats().size}")
    removed = cache.cleanup_expired()
    print(f"Expired entries removed: {removed}")
    print(f"Entries after cleanup: {cache.stats().size}")

    print("\n" + "="*70)
    print("All cache tests passed!")
    print("="*70)


# ============================================================================
# Data Merger for Multi-Resolution Alignment
# ============================================================================

class DataMerger:
    """
    Merges TSLA, SPY, and VIX data at multiple resolutions.

    This class handles:
    1. Inner join alignment of TSLA+SPY at each interval
    2. Creating base 1min DataFrame with multi_resolution attribute
    3. Forward-filling daily VIX to 1min timestamps
    4. Validation for alignment gaps
    5. Graceful handling of empty DataFrames
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize DataMerger.

        Args:
            verbose: Print alignment statistics and warnings
        """
        self.verbose = verbose

    def align_at_resolution(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        resolution: str
    ) -> pd.DataFrame:
        """
        Inner join TSLA and SPY at one interval.

        Performs zero-tolerance timestamp alignment - only keeps timestamps
        that exist in both TSLA and SPY data at this resolution.

        Args:
            tsla_df: TSLA OHLCV data with DatetimeIndex
            spy_df: SPY OHLCV data with DatetimeIndex
            resolution: Timeframe string (e.g., '1min', '5min', '1h', 'daily')

        Returns:
            Aligned DataFrame with tsla_* and spy_* columns
            Empty DataFrame if either input is empty

        Example:
            >>> merger = DataMerger()
            >>> aligned = merger.align_at_resolution(tsla_1min, spy_1min, '5min')
            >>> print(aligned.columns)
            Index(['tsla_open', 'tsla_high', ..., 'spy_close', 'spy_volume'])
        """
        import warnings as warn

        # Handle empty DataFrames
        if tsla_df.empty or spy_df.empty:
            if self.verbose:
                warn.warn(
                    f"Cannot align {resolution}: "
                    f"TSLA={'empty' if tsla_df.empty else 'OK'}, "
                    f"SPY={'empty' if spy_df.empty else 'OK'}"
                )
            return pd.DataFrame()

        # Make copies to avoid modifying originals
        tsla_data = tsla_df.copy()
        spy_data = spy_df.copy()

        # Ensure lowercase column names
        tsla_data.columns = [c.lower() for c in tsla_data.columns]
        spy_data.columns = [c.lower() for c in spy_data.columns]

        # Add symbol prefix to avoid column name collisions
        tsla_data = tsla_data.add_prefix('tsla_')
        spy_data = spy_data.add_prefix('spy_')

        # Inner join - only timestamps that exist in both datasets
        aligned = tsla_data.join(spy_data, how='inner')

        # Validation: Check for alignment gaps
        if len(aligned) == 0:
            if self.verbose:
                warn.warn(
                    f"Zero-length alignment at {resolution}: "
                    f"No overlapping timestamps between TSLA and SPY"
                )
            return pd.DataFrame()

        # Validation: Check for significant data loss
        tsla_len = len(tsla_data)
        spy_len = len(spy_data)
        aligned_len = len(aligned)

        tsla_loss_pct = (1 - aligned_len / tsla_len) * 100 if tsla_len > 0 else 0
        spy_loss_pct = (1 - aligned_len / spy_len) * 100 if spy_len > 0 else 0

        if self.verbose:
            print(f"   {resolution:7s}: {aligned_len:,} bars aligned "
                  f"(TSLA loss: {tsla_loss_pct:.1f}%, SPY loss: {spy_loss_pct:.1f}%)")

        # Warning for excessive data loss (>10% lost)
        if max(tsla_loss_pct, spy_loss_pct) > 10.0:
            warn.warn(
                f"High data loss at {resolution}: "
                f"TSLA lost {tsla_loss_pct:.1f}%, SPY lost {spy_loss_pct:.1f}%. "
                f"Check for timestamp misalignment or data quality issues."
            )

        return aligned

    def forward_fill_vix(
        self,
        base_timestamps: pd.DatetimeIndex,
        vix_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align daily VIX to 1min timestamps using forward fill.

        VIX is daily data, so we forward-fill it to match the higher
        frequency base timestamps (typically 1min or 5min).

        Args:
            base_timestamps: Target DatetimeIndex (e.g., 1min bars)
            vix_df: VIX daily OHLCV data with DatetimeIndex

        Returns:
            VIX DataFrame reindexed to base_timestamps with forward fill
            Empty DataFrame if vix_df is empty or base_timestamps is empty

        Example:
            >>> vix_aligned = merger.forward_fill_vix(tsla_1min.index, vix_daily)
            >>> # Each 1min bar now has VIX value from its corresponding day
        """
        import warnings as warn

        # Handle empty inputs
        if vix_df.empty:
            if self.verbose:
                warn.warn("VIX DataFrame is empty - cannot forward fill")
            return pd.DataFrame()

        if len(base_timestamps) == 0:
            if self.verbose:
                warn.warn("Base timestamps are empty - cannot forward fill VIX")
            return pd.DataFrame()

        # Make copy and ensure lowercase columns
        vix_data = vix_df.copy()
        vix_data.columns = [c.lower() for c in vix_data.columns]

        # Add vix_ prefix if not already present
        if not all(col.startswith('vix_') for col in vix_data.columns):
            vix_data = vix_data.add_prefix('vix_')

        # Reindex to base timestamps using forward fill
        # This propagates each daily VIX value to all intraday bars
        vix_aligned = vix_data.reindex(base_timestamps, method='ffill')

        # Count how many NaN values remain (at start before first VIX date)
        nan_count = vix_aligned.isnull().any(axis=1).sum()

        if self.verbose and nan_count > 0:
            warn.warn(
                f"VIX forward fill resulted in {nan_count:,} rows with NaN "
                f"(likely at start before VIX data begins)"
            )

        return vix_aligned

    def merge_all_resolutions(
        self,
        tsla_1min: pd.DataFrame,
        spy_1min: pd.DataFrame,
        vix_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create base 1min DataFrame with multi_resolution attribute.

        This is the main entry point that:
        1. Aligns TSLA+SPY at 1min base resolution
        2. Resamples to all higher timeframes and aligns each
        3. Forward-fills VIX to 1min timestamps
        4. Stores all resolutions in base_1min.attrs['multi_resolution']

        Args:
            tsla_1min: TSLA 1-minute OHLCV data
            spy_1min: SPY 1-minute OHLCV data
            vix_daily: VIX daily OHLCV data

        Returns:
            Base 1min aligned DataFrame with multi_resolution attribute

        Structure:
            base_1min.columns = ['tsla_open', 'tsla_high', ..., 'spy_close',
                                 'spy_volume', 'vix_open', 'vix_close', ...]
            base_1min.attrs['multi_resolution'] = {
                '1min': <aligned 1min data>,
                '5min': <aligned 5min data>,
                '15min': <aligned 15min data>,
                ...
                '3month': <aligned 3month data>
            }

        Example:
            >>> merger = DataMerger()
            >>> base = merger.merge_all_resolutions(tsla_1m, spy_1m, vix_d)
            >>> # Access 5min data: base.attrs['multi_resolution']['5min']
        """
        import warnings as warn
        from v7.core.timeframe import resample_ohlc

        if self.verbose:
            print("\n=== Multi-Resolution Data Merge ===")
            print(f"Input data:")
            if not tsla_1min.empty:
                print(f"  TSLA 1min: {len(tsla_1min):,} bars "
                      f"({tsla_1min.index[0]} to {tsla_1min.index[-1]})")
            else:
                print(f"  TSLA 1min: EMPTY")

            if not spy_1min.empty:
                print(f"  SPY 1min:  {len(spy_1min):,} bars "
                      f"({spy_1min.index[0]} to {spy_1min.index[-1]})")
            else:
                print(f"  SPY 1min:  EMPTY")

            if not vix_daily.empty:
                print(f"  VIX daily: {len(vix_daily):,} bars "
                      f"({vix_daily.index[0]} to {vix_daily.index[-1]})")
            else:
                print(f"  VIX daily: EMPTY")

        # Handle case where base inputs are empty
        if tsla_1min.empty or spy_1min.empty:
            if self.verbose:
                warn.warn("Cannot merge: TSLA or SPY 1min data is empty")
            return pd.DataFrame()

        # Step 1: Align at base 1min resolution
        if self.verbose:
            print(f"\nAligning at all resolutions:")

        base_1min = self.align_at_resolution(tsla_1min, spy_1min, '1min')

        if base_1min.empty:
            if self.verbose:
                warn.warn("Base 1min alignment failed - cannot proceed")
            return pd.DataFrame()

        # Step 2: Resample and align at all higher timeframes
        # Store 1min as base
        multi_res = {'1min': base_1min}

        # Get resampling source for each timeframe
        # We resample from 1min to avoid cumulative errors
        # TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']
        for tf in TIMEFRAMES:
            # Resample from 1min for all timeframes
            tsla_resampled = resample_ohlc(tsla_1min, tf)
            spy_resampled = resample_ohlc(spy_1min, tf)

            # Align at this resolution
            aligned_tf = self.align_at_resolution(tsla_resampled, spy_resampled, tf)

            if not aligned_tf.empty:
                multi_res[tf] = aligned_tf
            else:
                if self.verbose:
                    warn.warn(f"Skipping {tf} - alignment resulted in empty DataFrame")

        # Step 3: Forward-fill VIX to 1min timestamps
        if not vix_daily.empty:
            vix_aligned = self.forward_fill_vix(base_1min.index, vix_daily)

            if not vix_aligned.empty:
                # Add VIX to base 1min DataFrame
                base_1min = pd.concat([base_1min, vix_aligned], axis=1)

                if self.verbose:
                    print(f"\n   VIX forward-filled: {len(vix_aligned):,} bars aligned to 1min")
            else:
                if self.verbose:
                    warn.warn("VIX forward fill failed - VIX data will not be available")
        else:
            if self.verbose:
                warn.warn("VIX data is empty - skipping VIX alignment")

        if self.verbose:
            print(f"\n=== Merge Complete ===")
            print(f"Base 1min shape: {base_1min.shape}")
            print(f"Available resolutions: {list(multi_res.keys())}")
            print(f"Columns: {list(base_1min.columns)}")

        # Final validation: Check for NaN values
        # Note: Must do this BEFORE adding multi_resolution to avoid recursion issues with pandas.attrs
        nan_cols = base_1min.columns[base_1min.isnull().any()].tolist()
        if nan_cols and self.verbose:
            warn.warn(
                f"NaN values detected in columns: {nan_cols}. "
                f"Consider using dropna() or fillna() before feature extraction."
            )

        # Step 4: Store multi-resolution data in attrs (AFTER NaN check to avoid recursion)
        base_1min.attrs['multi_resolution'] = multi_res

        return base_1min

    def validate_alignment(
        self,
        base_df: pd.DataFrame,
        max_gap_minutes: int = 10
    ) -> Tuple[bool, List[str]]:
        """
        Validate alignment quality and detect gaps.

        Checks for:
        - Large timestamp gaps (potential missing data)
        - NaN values in critical columns
        - Multi-resolution data availability

        Args:
            base_df: Base DataFrame returned from merge_all_resolutions()
            max_gap_minutes: Maximum allowed gap between timestamps (minutes)

        Returns:
            Tuple of (is_valid, list_of_issues)
            - is_valid: True if validation passes
            - list_of_issues: Human-readable list of detected issues

        Example:
            >>> is_valid, issues = merger.validate_alignment(base_1min)
            >>> if not is_valid:
            ...     for issue in issues:
            ...         print(f"WARNING: {issue}")
        """
        issues = []

        # Check 1: DataFrame not empty
        if base_df.empty:
            issues.append("Base DataFrame is empty")
            return False, issues

        # Check 2: Multi-resolution data exists
        if 'multi_resolution' not in base_df.attrs:
            issues.append("No multi_resolution attribute found")
        else:
            multi_res = base_df.attrs['multi_resolution']
            if len(multi_res) == 0:
                issues.append("multi_resolution dict is empty")

            # Check that expected timeframes are present
            expected_tfs = ['1min', '5min', '15min', '30min', '1h']
            missing_tfs = [tf for tf in expected_tfs if tf not in multi_res]
            if missing_tfs:
                issues.append(f"Missing expected timeframes: {missing_tfs}")

        # Check 3: Timestamp gaps
        time_diffs = base_df.index.to_series().diff()
        large_gaps = time_diffs > pd.Timedelta(minutes=max_gap_minutes)

        if large_gaps.any():
            num_gaps = large_gaps.sum()
            max_gap = time_diffs.max()
            issues.append(
                f"Found {num_gaps} large timestamp gaps (max: {max_gap}, "
                f"threshold: {max_gap_minutes} min)"
            )

        # Check 4: NaN values in TSLA/SPY columns
        tsla_cols = [c for c in base_df.columns if c.startswith('tsla_')]
        spy_cols = [c for c in base_df.columns if c.startswith('spy_')]

        tsla_nans = base_df[tsla_cols].isnull().any().sum()
        spy_nans = base_df[spy_cols].isnull().any().sum()

        if tsla_nans > 0:
            issues.append(f"NaN values in {tsla_nans} TSLA columns")
        if spy_nans > 0:
            issues.append(f"NaN values in {spy_nans} SPY columns")

        # Check 5: VIX data availability
        vix_cols = [c for c in base_df.columns if c.startswith('vix_')]
        if len(vix_cols) == 0:
            issues.append("No VIX columns found (expected vix_open, vix_close, etc.)")

        # Determine overall validity
        is_valid = len(issues) == 0

        return is_valid, issues


# ============================================================================
# Helper Functions
# ============================================================================

def load_csv_data(
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, and VIX CSV files.

    Helper function to load raw CSV data for testing DataMerger.

    Args:
        data_dir: Directory containing CSV files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Tuple of (tsla_1min, spy_1min, vix_daily)
    """
    # Load TSLA 1min
    tsla_path = data_dir / "TSLA_1min.csv"
    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]

    # Load SPY 1min
    spy_path = data_dir / "SPY_1min.csv"
    spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df.columns = [c.lower() for c in spy_df.columns]

    # Load VIX daily
    vix_path = data_dir / "VIX_History.csv"
    vix_df = pd.read_csv(vix_path, parse_dates=['DATE'])
    vix_df.set_index('DATE', inplace=True)
    vix_df.columns = [c.lower() for c in vix_df.columns]

    # Apply date filters if provided
    if start_date:
        tsla_df = tsla_df[tsla_df.index >= start_date]
        spy_df = spy_df[spy_df.index >= start_date]
        vix_df = vix_df[vix_df.index >= start_date]

    if end_date:
        tsla_df = tsla_df[tsla_df.index <= end_date]
        spy_df = spy_df[spy_df.index <= end_date]
        vix_df = vix_df[vix_df.index <= end_date]

    return tsla_df, spy_df, vix_df


def test_data_merger():
    """
    Test suite for DataMerger class.
    """
    print("\n" + "="*70)
    print("DATAMERGER TEST SUITE")
    print("="*70)

    # Setup paths
    data_dir = Path(__file__).parent.parent.parent / "data"

    # Load CSV data
    print("\n1. Loading CSV data...")
    tsla_1min, spy_1min, vix_daily = load_csv_data(
        data_dir,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    print(f"\nLoaded:")
    print(f"  TSLA 1min: {len(tsla_1min):,} bars")
    print(f"  SPY 1min:  {len(spy_1min):,} bars")
    print(f"  VIX daily: {len(vix_daily):,} bars")

    # Test DataMerger
    print("\n2. Testing DataMerger...")
    merger = DataMerger(verbose=True)

    # Test individual alignment
    from v7.core.timeframe import resample_ohlc

    print("\n   Testing align_at_resolution():")
    aligned_5min_tsla = resample_ohlc(tsla_1min, '5min')
    aligned_5min_spy = resample_ohlc(spy_1min, '5min')
    test_5min = merger.align_at_resolution(aligned_5min_tsla, aligned_5min_spy, '5min')
    print(f"   5min alignment result: {test_5min.shape}")

    # Test VIX forward fill
    print("\n   Testing forward_fill_vix():")
    test_vix = merger.forward_fill_vix(tsla_1min.index[:1000], vix_daily)
    print(f"   VIX forward fill result: {test_vix.shape}")

    # Test full merge
    print("\n3. Testing merge_all_resolutions()...")
    base_1min = merger.merge_all_resolutions(tsla_1min, spy_1min, vix_daily)

    # Validate results
    print("\n4. Validating alignment...")
    is_valid, issues = merger.validate_alignment(base_1min)

    if is_valid:
        print("   ✓ Validation PASSED")
    else:
        print("   ✗ Validation FAILED:")
        for issue in issues:
            print(f"      - {issue}")

    # Display summary
    print("\n5. Summary:")
    print(f"   Base 1min shape: {base_1min.shape}")
    print(f"   Columns: {list(base_1min.columns)[:5]}...")
    print(f"   Multi-resolution timeframes: {list(base_1min.attrs.get('multi_resolution', {}).keys())}")

    # Test empty DataFrame handling
    print("\n6. Testing empty DataFrame handling...")
    empty_df = pd.DataFrame()
    result = merger.merge_all_resolutions(empty_df, spy_1min, vix_daily)
    print(f"   Empty TSLA result: {len(result)} bars (expected 0)")

    result = merger.merge_all_resolutions(tsla_1min, empty_df, vix_daily)
    print(f"   Empty SPY result: {len(result)} bars (expected 0)")

    result = merger.merge_all_resolutions(tsla_1min, spy_1min, empty_df)
    print(f"   Empty VIX result: {len(result)} bars (has data, no VIX cols)")

    print("\n" + "="*70)
    print("DataMerger test complete!")
    print("="*70)
