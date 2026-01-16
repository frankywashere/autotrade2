"""
Data loading and resampling utilities for v15.

This module provides market data loading and resampling with proper
handling of partial (incomplete) bars - a critical requirement for
live trading where the current bar is always incomplete.

Also provides native timeframe data loading from yfinance for more
accurate OHLC values at higher timeframes.
"""

from .loader import load_market_data, validate_ohlcv
from .resampler import resample_with_partial, get_bar_metadata
from .native_tf import (
    fetch_native_tf,
    load_native_tf_data,
    get_native_tf_slice,
    align_native_tf_timestamps,
    validate_native_data,
    clear_cache as clear_native_tf_cache,
    TF_TO_YF_INTERVAL,
    ALL_TIMEFRAMES,
    DEFAULT_SYMBOLS,
)

# Backward compatibility alias
resample_ohlc = resample_with_partial

__all__ = [
    # Loader
    'load_market_data',
    'validate_ohlcv',
    # Resampler
    'resample_with_partial',
    'resample_ohlc',  # Alias for backward compatibility
    'get_bar_metadata',
    # Native TF
    'fetch_native_tf',
    'load_native_tf_data',
    'get_native_tf_slice',
    'align_native_tf_timestamps',
    'validate_native_data',
    'clear_native_tf_cache',
    'TF_TO_YF_INTERVAL',
    'ALL_TIMEFRAMES',
    'DEFAULT_SYMBOLS',
]
