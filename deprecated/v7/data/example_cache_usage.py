"""
Example usage of the LiveDataCache for real-time data fetching.

This demonstrates how to integrate the cache with a live data fetching system.
"""

import time
import pandas as pd
import numpy as np
from live_fetcher import LiveDataCache


def simulate_api_fetch(symbol: str, interval: str) -> pd.DataFrame:
    """
    Simulate fetching data from an API (e.g., Binance, Yahoo Finance).

    In a real implementation, this would make an actual API call.
    """
    print(f"  [API CALL] Fetching {symbol} {interval} from external API...")
    time.sleep(0.1)  # Simulate network delay

    # Generate sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
    np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol

    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 0.1,
        'high': prices + np.abs(np.random.randn(100) * 0.3),
        'low': prices - np.abs(np.random.randn(100) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })

    return df


def get_live_data(cache: LiveDataCache, symbol: str, interval: str) -> pd.DataFrame:
    """
    Get live data with caching.

    This is the main function you'd use in a live trading system.
    It automatically handles cache hits/misses.
    """
    # Try cache first
    data = cache.get(symbol, interval)

    if data is not None:
        print(f"  [CACHE HIT] {symbol} {interval} - using cached data")
        return data

    # Cache miss - fetch from API
    print(f"  [CACHE MISS] {symbol} {interval} - fetching fresh data")
    data = simulate_api_fetch(symbol, interval)

    # Store in cache
    cache.set(symbol, interval, data)

    return data


def main():
    """Demonstrate cache usage patterns."""

    print("="*70)
    print("LIVE DATA CACHE USAGE EXAMPLE")
    print("="*70)

    # Create cache with 5 minute TTL
    cache = LiveDataCache(ttl=300)

    print("\n--- Scenario 1: Multiple requests for same data ---")
    print("First request:")
    df1 = get_live_data(cache, 'BTCUSDT', '5m')
    print(f"Received {len(df1)} rows")

    print("\nSecond request (should be cached):")
    df2 = get_live_data(cache, 'BTCUSDT', '5m')
    print(f"Received {len(df2)} rows")

    print("\nThird request (should still be cached):")
    df3 = get_live_data(cache, 'BTCUSDT', '5m')
    print(f"Received {len(df3)} rows")

    print("\n--- Scenario 2: Different symbols ---")
    print("Request for ETHUSDT:")
    eth_df = get_live_data(cache, 'ETHUSDT', '5m')
    print(f"Received {len(eth_df)} rows")

    print("\nRequest for BTCUSDT again (should be cached):")
    btc_df = get_live_data(cache, 'BTCUSDT', '5m')
    print(f"Received {len(btc_df)} rows")

    print("\n--- Scenario 3: Different intervals ---")
    print("Request for BTCUSDT 1h:")
    df_1h = get_live_data(cache, 'BTCUSDT', '1h')
    print(f"Received {len(df_1h)} rows")

    print("\nRequest for BTCUSDT 5m (should still be cached):")
    df_5m = get_live_data(cache, 'BTCUSDT', '5m')
    print(f"Received {len(df_5m)} rows")

    # Show cache statistics
    print("\n--- Cache Statistics ---")
    cache.print_stats()

    # Show cache entries
    print("\n--- Cache Entries ---")
    for entry in cache.list_entries():
        print(f"  {entry['symbol']:10} {entry['interval']:5} - "
              f"age: {entry['age_seconds']:6.2f}s, "
              f"expires in: {entry['remaining_ttl']:6.2f}s")

    print("\n--- Scenario 4: Cache expiration ---")
    # Create a cache with short TTL for demo
    short_cache = LiveDataCache(ttl=2.0)

    print("Fetching with 2-second TTL:")
    df = get_live_data(short_cache, 'TEST', '5m')

    print("\nImmediate re-fetch (should be cached):")
    df = get_live_data(short_cache, 'TEST', '5m')

    print("\nWaiting 3 seconds for expiration...")
    time.sleep(3)

    print("Fetching after expiration (should fetch fresh):")
    df = get_live_data(short_cache, 'TEST', '5m')

    print("\n--- Scenario 5: Manual cache operations ---")

    print("\nGetting entry info:")
    info = cache.get_entry_info('BTCUSDT', '5m')
    if info:
        print(f"  Symbol: {info['symbol']}")
        print(f"  Interval: {info['interval']}")
        print(f"  Age: {info['age_seconds']:.2f}s")
        print(f"  Expires at: {info['expires_at']}")

    print("\nRemoving specific entry:")
    removed = cache.remove('ETHUSDT', '5m')
    print(f"  Removed: {removed}")

    print("\nCleaning up expired entries:")
    expired_count = cache.cleanup_expired()
    print(f"  Removed {expired_count} expired entries")

    # Final statistics
    print("\n--- Final Statistics ---")
    cache.print_stats()

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
