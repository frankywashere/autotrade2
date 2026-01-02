"""
v7 Data Module - Market Data Loading and Processing

This module provides utilities for loading and processing market data for channel detection
and model training. It serves as the central interface for all data-related operations.

Main Components:
1. Market data loaders (TSLA, SPY, VIX)
2. Live data fetching utilities with TTL caching
3. Data validation and alignment
4. Cache management for processed data

Usage:
    from v7.data import load_market_data, validate_date_range, LiveDataCache

    # Historical data loading
    tsla_df, spy_df, vix_df = load_market_data(
        data_dir=Path("data"),
        start_date="2020-01-01",
        end_date="2023-12-31"
    )

    # Live data caching
    cache = LiveDataCache(ttl=300)  # 5 minute TTL
    data = cache.get('BTCUSDT', '5m')
    if data is None:
        data = fetch_from_api('BTCUSDT', '5m')
        cache.set('BTCUSDT', '5m', data)
"""

# Version information
__version__ = "7.3.0"

# Import data loading utilities from training.dataset
# NOTE: These are currently in training/dataset.py but logically belong here
# Future refactoring may move them to this module
from ..training.dataset import (
    load_market_data,
    validate_date_range,
    get_data_date_range,
)

# Import live data cache
from .live_fetcher import LiveDataCache, CacheEntry, LiveCacheStats

# Import VIX fetcher
from .vix_fetcher import FREDVixFetcher, fetch_vix_data

# Import live data integration for dashboard (NEW)
from .live import (
    fetch_live_data,
    load_live_data_tuple,
    LiveDataResult,
    is_market_open
)


# Public API - explicitly define what should be exported
__all__ = [
    # Version
    '__version__',

    # Market data loaders
    'load_market_data',

    # Validation utilities
    'validate_date_range',
    'get_data_date_range',

    # Live data caching
    'LiveDataCache',
    'CacheEntry',
    'LiveCacheStats',

    # VIX data fetching
    'FREDVixFetcher',
    'fetch_vix_data',

    # Live data integration (dashboard support)
    'fetch_live_data',
    'load_live_data_tuple',
    'LiveDataResult',
    'is_market_open',
]
