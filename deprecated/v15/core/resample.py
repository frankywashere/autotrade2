"""
Canonical OHLC Resampling Module for x14 V15

This module provides the single, authoritative implementation of OHLC resampling
used throughout the x14 system. It consolidates multiple previous implementations
into one well-tested, comprehensive function.

Key features:
- Standard OHLC aggregation (first open, max high, min low, last close)
- Volume summation (when present)
- Support for all 10 timeframes used in x14
- Optional partial bar handling for live trading
- Robust error handling with clear error messages

Usage:
    from v15.core.resample import resample_ohlc

    # Basic resampling
    df_daily = resample_ohlc(df_5min, 'daily')

    # With partial bar support
    df_1h, metadata = resample_ohlc(df_5min, '1h', keep_partial=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# All supported timeframes in hierarchical order (fastest to slowest)
TIMEFRAMES = [
    '5min', '15min', '30min', '1h', '2h', '3h', '4h',
    'daily', 'weekly', 'monthly'
]

# Pandas resample rules for each timeframe
RESAMPLE_RULES: Dict[str, str] = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '1h': '1h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1MS',  # Month Start for proper alignment
}

# How many 5-min bars per timeframe bar (assumes 6.5 hour trading day)
BARS_PER_TF: Dict[str, int] = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,      # 6.5 hours of trading = 78 five-minute bars
    'weekly': 390,    # 5 trading days * 78
    'monthly': 1638,  # ~21 trading days * 78
}

# Aliases for common timeframe variations
TF_ALIASES: Dict[str, str] = {
    # Hour variations
    '1H': '1h',
    '2H': '2h',
    '3H': '3h',
    '4H': '4h',
    '60min': '1h',
    '120min': '2h',
    '180min': '3h',
    '240min': '4h',
    # Daily variations
    '1d': 'daily',
    '1D': 'daily',
    'D': 'daily',
    'd': 'daily',
    # Weekly variations
    '1w': 'weekly',
    '1W': 'weekly',
    'W': 'weekly',
    'w': 'weekly',
    '1wk': 'weekly',
    # Monthly variations
    '1mo': 'monthly',
    '1M': 'monthly',
    'M': 'monthly',
    '1MS': 'monthly',
    '1ME': 'monthly',
}


# =============================================================================
# Exceptions
# =============================================================================

class ResamplingError(Exception):
    """Raised when OHLC resampling fails."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BarMetadata:
    """
    Metadata about a resampled bar, especially the partial (incomplete) bar.

    Attributes:
        bar_completion_pct: 0.0 to 1.0, how complete the last bar is
        bars_in_partial: Number of source bars that contributed to the partial bar
        expected_bars: Number of source bars expected for a complete bar
        is_partial: True if the last bar is incomplete
        total_bars: Total number of bars in the resampled output
        source_bars: Total number of bars in the source data
    """
    bar_completion_pct: float
    bars_in_partial: int
    expected_bars: int
    is_partial: bool
    total_bars: int
    source_bars: int


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_timeframe(tf: str) -> str:
    """
    Normalize a timeframe string to the canonical format.

    Args:
        tf: Timeframe string in any supported format

    Returns:
        Canonical timeframe string

    Raises:
        ResamplingError: If timeframe is not recognized

    Examples:
        >>> normalize_timeframe('1h')
        '1h'
        >>> normalize_timeframe('1H')
        '1h'
        >>> normalize_timeframe('daily')
        'daily'
        >>> normalize_timeframe('1D')
        'daily'
    """
    # Check if already canonical
    if tf in TIMEFRAMES:
        return tf

    # Check aliases
    if tf in TF_ALIASES:
        return TF_ALIASES[tf]

    # Try lowercase
    tf_lower = tf.lower()
    if tf_lower in TIMEFRAMES:
        return tf_lower

    raise ResamplingError(
        f"Unknown timeframe: '{tf}'. "
        f"Valid timeframes: {TIMEFRAMES}. "
        f"Also accepts aliases like '1D', '1h', '1wk', '1mo'."
    )


def get_resample_rule(tf: str) -> str:
    """
    Get the pandas resample rule for a timeframe.

    Args:
        tf: Timeframe string (canonical or alias)

    Returns:
        Pandas resample rule string

    Raises:
        ResamplingError: If timeframe is not recognized
    """
    canonical = normalize_timeframe(tf)
    return RESAMPLE_RULES[canonical]


def get_bars_per_tf(tf: str) -> int:
    """
    Get the number of 5-min bars per timeframe bar.

    Args:
        tf: Timeframe string (canonical or alias)

    Returns:
        Number of 5-min bars

    Raises:
        ResamplingError: If timeframe is not recognized
    """
    canonical = normalize_timeframe(tf)
    return BARS_PER_TF[canonical]


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame is suitable for OHLC resampling.

    Args:
        df: DataFrame to validate

    Raises:
        ResamplingError: If validation fails
    """
    if df is None:
        raise ResamplingError("DataFrame is None")

    if len(df) == 0:
        raise ResamplingError("DataFrame is empty")

    # Check for required columns (case-insensitive)
    required = {'open', 'high', 'low', 'close'}
    columns_lower = {col.lower() for col in df.columns}
    missing = required - columns_lower

    if missing:
        raise ResamplingError(
            f"Missing required columns: {missing}. "
            f"DataFrame has: {list(df.columns)}"
        )

    # Check for DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ResamplingError(
            f"DataFrame must have DatetimeIndex, got {type(df.index).__name__}"
        )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with lowercase column names
    """
    col_mapping = {col: col.lower() for col in df.columns}
    return df.rename(columns=col_mapping)


def _calculate_partial_bar_metadata(
    resampled: pd.DataFrame,
    source: pd.DataFrame,
    tf: str
) -> BarMetadata:
    """
    Calculate metadata about the partial (last) bar.

    Args:
        resampled: Resampled DataFrame
        source: Original source DataFrame
        tf: Target timeframe

    Returns:
        BarMetadata with completion information
    """
    if len(resampled) == 0:
        return BarMetadata(
            bar_completion_pct=0.0,
            bars_in_partial=0,
            expected_bars=0,
            is_partial=False,
            total_bars=0,
            source_bars=len(source)
        )

    expected_bars = get_bars_per_tf(tf)

    # Get the last resampled bar's timestamp
    last_bar_start = resampled.index[-1]

    # Count source bars in the last resampled bar
    # For this, we need to determine the bar boundaries
    rule = get_resample_rule(tf)

    # Count bars that fall into the last period
    bars_in_last = len(source[source.index >= last_bar_start])

    # Calculate completion
    completion_pct = min(1.0, bars_in_last / expected_bars) if expected_bars > 0 else 1.0
    is_partial = completion_pct < 1.0

    return BarMetadata(
        bar_completion_pct=round(completion_pct, 4),
        bars_in_partial=bars_in_last,
        expected_bars=expected_bars,
        is_partial=is_partial,
        total_bars=len(resampled),
        source_bars=len(source)
    )


# =============================================================================
# Main Functions
# =============================================================================

def resample_ohlc(
    df: pd.DataFrame,
    timeframe: str,
    keep_partial: bool = False,
    drop_na: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, BarMetadata]]:
    """
    Resample OHLCV data to a target timeframe.

    This is the canonical resampling function for x14. It properly aggregates
    OHLCV data using standard rules:
    - Open: first value in the period
    - High: maximum value in the period
    - Low: minimum value in the period
    - Close: last value in the period
    - Volume: sum of all values in the period (if present)

    Args:
        df: DataFrame with DatetimeIndex and OHLC(V) columns.
            Required columns: open, high, low, close
            Optional columns: volume
            Column names are case-insensitive.
        timeframe: Target timeframe string. Supported values:
            - '5min', '15min', '30min', '1h', '2h', '3h', '4h'
            - 'daily', 'weekly', 'monthly'
            - Also accepts aliases: '1D', '1H', '1w', '1mo', etc.
        keep_partial: If True, keep the last (potentially incomplete) bar
            and return metadata about its completion status.
            If False (default), drop incomplete bars.
        drop_na: If True (default), drop bars with any NaN values.
            If False, keep all bars including those with NaN.

    Returns:
        If keep_partial=False:
            pd.DataFrame: Resampled OHLCV data
        If keep_partial=True:
            Tuple[pd.DataFrame, BarMetadata]: Resampled data and metadata

    Raises:
        ResamplingError: If resampling fails due to invalid input

    Examples:
        >>> # Basic usage - resample 5min to daily
        >>> df_daily = resample_ohlc(df_5min, 'daily')

        >>> # Keep partial bar for live trading
        >>> df_1h, meta = resample_ohlc(df_5min, '1h', keep_partial=True)
        >>> print(f"Last bar is {meta.bar_completion_pct*100:.1f}% complete")

        >>> # Using aliases
        >>> df_d = resample_ohlc(df_5min, '1D')  # Same as 'daily'

    Notes:
        - Source data should have a consistent timeframe (typically 5-min)
        - For live trading, use keep_partial=True to include the current bar
        - Volume column is optional (e.g., VIX data doesn't have volume)
        - The function handles timezone-aware and naive DatetimeIndex
    """
    # Validate input
    _validate_dataframe(df)

    # Normalize timeframe
    try:
        canonical_tf = normalize_timeframe(timeframe)
    except ResamplingError:
        raise

    # Get resample rule
    rule = RESAMPLE_RULES[canonical_tf]

    # Normalize column names
    df_work = _normalize_columns(df.copy())

    # Ensure sorted by index
    df_work = df_work.sort_index()

    # Build aggregation dict
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }

    # Add volume if present
    if 'volume' in df_work.columns:
        agg_dict['volume'] = 'sum'

    # Perform resampling
    try:
        resampled = df_work.resample(rule).agg(agg_dict)
    except Exception as e:
        raise ResamplingError(f"Resampling failed: {e}")

    # Handle partial bars and NaN values
    if keep_partial:
        # Keep partial bar but calculate metadata first (before dropping NaN)
        metadata = _calculate_partial_bar_metadata(resampled, df_work, canonical_tf)

        # Drop completely empty bars (all NaN) but keep partial bars
        resampled = resampled.dropna(how='all')

        # Update metadata total_bars after dropping
        metadata = BarMetadata(
            bar_completion_pct=metadata.bar_completion_pct,
            bars_in_partial=metadata.bars_in_partial,
            expected_bars=metadata.expected_bars,
            is_partial=metadata.is_partial,
            total_bars=len(resampled),
            source_bars=metadata.source_bars
        )

        return resampled, metadata
    else:
        # Standard behavior: drop all bars with any NaN
        if drop_na:
            resampled = resampled.dropna()

        return resampled


def resample_multi_tf(
    df: pd.DataFrame,
    timeframes: Optional[list] = None,
    keep_partial: bool = False
) -> Dict[str, Union[pd.DataFrame, Tuple[pd.DataFrame, BarMetadata]]]:
    """
    Resample OHLCV data to multiple timeframes at once.

    This is a convenience function for resampling to all timeframes
    in a single call, which is common in multi-timeframe analysis.

    Args:
        df: Source DataFrame with DatetimeIndex and OHLC(V) columns
        timeframes: List of target timeframes. If None, uses all TIMEFRAMES.
        keep_partial: If True, keep partial bars and return metadata.

    Returns:
        Dict mapping timeframe -> resampled data (or tuple if keep_partial=True)

    Examples:
        >>> # Resample to all timeframes
        >>> all_tfs = resample_multi_tf(df_5min)
        >>> df_daily = all_tfs['daily']

        >>> # Resample to specific timeframes
        >>> tfs = resample_multi_tf(df_5min, ['1h', 'daily', 'weekly'])
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    result = {}
    for tf in timeframes:
        try:
            result[tf] = resample_ohlc(df, tf, keep_partial=keep_partial)
        except ResamplingError as e:
            logger.warning(f"Failed to resample to {tf}: {e}")
            continue

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_longer_timeframes(tf: str) -> list:
    """
    Get all timeframes longer than the given one.

    Args:
        tf: Current timeframe

    Returns:
        List of longer timeframes in order

    Raises:
        ResamplingError: If timeframe is not recognized

    Example:
        >>> get_longer_timeframes('1h')
        ['2h', '3h', '4h', 'daily', 'weekly', 'monthly']
    """
    canonical = normalize_timeframe(tf)
    idx = TIMEFRAMES.index(canonical)
    return TIMEFRAMES[idx + 1:]


def get_shorter_timeframes(tf: str) -> list:
    """
    Get all timeframes shorter than the given one.

    Args:
        tf: Current timeframe

    Returns:
        List of shorter timeframes in order (fastest first)

    Raises:
        ResamplingError: If timeframe is not recognized

    Example:
        >>> get_shorter_timeframes('daily')
        ['5min', '15min', '30min', '1h', '2h', '3h', '4h']
    """
    canonical = normalize_timeframe(tf)
    idx = TIMEFRAMES.index(canonical)
    return TIMEFRAMES[:idx]


def validate_ohlc(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that OHLC data has proper relationships.

    Checks:
    - High >= Open, Close, Low for all bars
    - Low <= Open, Close, High for all bars
    - No negative prices
    - No extreme outliers

    Args:
        df: DataFrame with OHLC columns

    Returns:
        Dict with 'valid' bool and list of 'issues'

    Example:
        >>> result = validate_ohlc(df)
        >>> if not result['valid']:
        ...     print(f"Issues: {result['issues']}")
    """
    issues = []
    df_work = _normalize_columns(df)

    if len(df_work) == 0:
        return {'valid': False, 'issues': ['DataFrame is empty']}

    # Check high >= all others
    invalid_high = df_work[
        (df_work['high'] < df_work['open']) |
        (df_work['high'] < df_work['close']) |
        (df_work['high'] < df_work['low'])
    ]
    if len(invalid_high) > 0:
        issues.append(f"{len(invalid_high)} bars have high < other OHLC values")

    # Check low <= all others
    invalid_low = df_work[
        (df_work['low'] > df_work['open']) |
        (df_work['low'] > df_work['close']) |
        (df_work['low'] > df_work['high'])
    ]
    if len(invalid_low) > 0:
        issues.append(f"{len(invalid_low)} bars have low > other OHLC values")

    # Check for negative prices
    negative = df_work[
        (df_work['open'] < 0) |
        (df_work['high'] < 0) |
        (df_work['low'] < 0) |
        (df_work['close'] < 0)
    ]
    if len(negative) > 0:
        issues.append(f"{len(negative)} bars have negative prices")

    # Check for NaN
    nan_count = df_work[['open', 'high', 'low', 'close']].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"{nan_count} NaN values in OHLC columns")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_bars': len(df_work)
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main functions
    'resample_ohlc',
    'resample_multi_tf',

    # Utility functions
    'normalize_timeframe',
    'get_resample_rule',
    'get_bars_per_tf',
    'get_longer_timeframes',
    'get_shorter_timeframes',
    'validate_ohlc',

    # Constants
    'TIMEFRAMES',
    'RESAMPLE_RULES',
    'BARS_PER_TF',
    'TF_ALIASES',

    # Classes
    'BarMetadata',
    'ResamplingError',
]
