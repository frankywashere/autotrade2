"""
Timeframe utilities for resampling OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Dict

# All timeframes in hierarchical order (fastest to slowest)
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly', '3month'
]

# Pandas resample rules
RESAMPLE_RULES = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1ME',
    '3month': '3ME',
}

# How many base bars (5min) per timeframe bar
BARS_PER_TF = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,     # ~6.5 hours of trading
    'weekly': 390,   # 5 days × 78
    'monthly': 1638, # ~21 trading days × 78
    '3month': 4914,  # ~63 trading days × 78
}


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
        timeframe: Target timeframe (e.g., '15min', '1h', 'daily')

    Returns:
        Resampled DataFrame with same columns
    """
    if timeframe not in RESAMPLE_RULES:
        raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(RESAMPLE_RULES.keys())}")

    rule = RESAMPLE_RULES[timeframe]

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


def get_longer_timeframes(tf: str) -> list:
    """
    Get all timeframes longer than the given one.

    Args:
        tf: Current timeframe

    Returns:
        List of longer timeframes in order
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {tf}")

    idx = TIMEFRAMES.index(tf)
    return TIMEFRAMES[idx + 1:]


def get_shorter_timeframes(tf: str) -> list:
    """
    Get all timeframes shorter than the given one.

    Args:
        tf: Current timeframe

    Returns:
        List of shorter timeframes in order (fastest first)
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe: {tf}")

    idx = TIMEFRAMES.index(tf)
    return TIMEFRAMES[:idx]
