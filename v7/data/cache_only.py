"""
LiveDataCache - Standalone Caching Layer

This is the pure caching implementation without any external dependencies.
Designed for live data fetching with TTL-based expiration.

Features:
1. Simple memory cache with TTL (5 minutes default)
2. Cache key generation from symbol+interval
3. get() and set() methods
4. Cache expiration checking
5. Thread-safe operations with RLock
"""

import time
from threading import RLock
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


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
# Live Data Cache - Main Implementation
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

    def list_entries(self) -> list:
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
# Simple Usage Example
# ============================================================================

if __name__ == '__main__':
    """
    Simple demonstration of the cache functionality.
    """
    print("="*70)
    print("LIVE DATA CACHE - BASIC TEST")
    print("="*70)

    # Create cache with 5 minute TTL
    cache = LiveDataCache(ttl=300)

    # Test 1: Cache miss
    print("\n1. Testing cache miss...")
    data = cache.get('BTCUSDT', '5m')
    print(f"   Result: {data}")
    assert data is None, "Should be cache miss"
    print("   PASS: Cache miss works correctly")

    # Test 2: Set data
    print("\n2. Setting data in cache...")
    sample_data = {"price": 50000, "volume": 1000}
    cache.set('BTCUSDT', '5m', sample_data)
    print("   PASS: Data stored")

    # Test 3: Cache hit
    print("\n3. Testing cache hit...")
    data = cache.get('BTCUSDT', '5m')
    print(f"   Result: {data}")
    assert data == sample_data, "Should be cache hit"
    print("   PASS: Cache hit works correctly")

    # Test 4: Different keys
    print("\n4. Testing different keys...")
    cache.set('ETHUSDT', '5m', {"price": 3000})
    cache.set('BTCUSDT', '1h', {"price": 50100})

    btc_5m = cache.get('BTCUSDT', '5m')
    eth_5m = cache.get('ETHUSDT', '5m')
    btc_1h = cache.get('BTCUSDT', '1h')

    assert btc_5m["price"] == 50000
    assert eth_5m["price"] == 3000
    assert btc_1h["price"] == 50100
    print("   PASS: Multiple keys work correctly")

    # Test 5: Statistics
    print("\n5. Testing statistics...")
    stats = cache.stats()
    print(f"   Hits: {stats.hits}")
    print(f"   Misses: {stats.misses}")
    print(f"   Size: {stats.size}")
    print(f"   Hit rate: {stats.hit_rate:.2%}")
    assert stats.hits == 4  # BTCUSDT 5m + ETHUSDT 5m + BTCUSDT 1h + BTCUSDT 5m
    assert stats.misses == 1  # Initial BTCUSDT 5m
    assert stats.size == 3  # BTCUSDT-5m, ETHUSDT-5m, BTCUSDT-1h
    print("   PASS: Statistics tracking works correctly")

    # Test 6: Expiration
    print("\n6. Testing expiration...")
    short_cache = LiveDataCache(ttl=1.0)  # 1 second TTL
    short_cache.set('TEST', '5m', {"test": True})

    data = short_cache.get('TEST', '5m')
    assert data is not None, "Should be cached"
    print("   Immediate fetch: HIT")

    time.sleep(1.5)  # Wait for expiration

    data = short_cache.get('TEST', '5m')
    assert data is None, "Should be expired"
    print("   After expiration: MISS")
    print("   PASS: TTL expiration works correctly")

    # Test 7: Thread safety (basic test)
    print("\n7. Testing thread safety...")
    import threading

    def worker(cache_obj, thread_id):
        for i in range(10):
            cache_obj.set(f'SYM{thread_id}', '5m', {'thread': thread_id, 'iter': i})
            data = cache_obj.get(f'SYM{thread_id}', '5m')
            assert data is not None

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(cache, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("   PASS: Thread safety test completed")

    # Final statistics
    print("\n" + "="*70)
    cache.print_stats()
    print("All tests passed!")
    print("="*70)
