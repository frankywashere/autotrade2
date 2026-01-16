"""
Data loading and resampling utilities for v15.

This module provides market data loading and resampling with proper
handling of partial (incomplete) bars - a critical requirement for
live trading where the current bar is always incomplete.
"""

from .loader import load_market_data, validate_ohlcv
from .resampler import resample_with_partial, get_bar_metadata

# Backward compatibility alias
resample_ohlc = resample_with_partial

__all__ = [
    'load_market_data',
    'validate_ohlcv',
    'resample_with_partial',
    'resample_ohlc',  # Alias for backward compatibility
    'get_bar_metadata',
]
