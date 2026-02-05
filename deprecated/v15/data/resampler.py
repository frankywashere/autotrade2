"""
Resampling utilities with partial bar support.

Critical design principle: KEEP PARTIAL BARS.

In live trading, the current bar is always incomplete. Traditional
resampling with dropna() discards this bar, which means:
- Features computed on the "current" bar are actually stale
- The model sees data that's up to one full timeframe old
- Predictions lag behind reality

This module keeps partial bars and provides metadata about their
completion status, allowing downstream code to make informed decisions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


class ResamplingError(Exception):
    """Raised when resampling fails."""
    pass


@dataclass
class BarMetadata:
    """Metadata about a resampled bar, especially the partial bar."""
    bar_completion_pct: float  # 0.0 to 1.0, how complete is the last bar
    bars_in_partial: int  # How many source bars contributed to partial
    expected_bars: int  # How many source bars expected for a complete bar
    is_partial: bool  # True if the last bar is incomplete
    partial_start: Optional[pd.Timestamp]  # When the partial bar started
    partial_end: Optional[pd.Timestamp]  # Last data point in partial bar
    total_bars: int  # Total bars in resampled output
    source_bars: int  # Total bars in source data


# Timeframe definitions in minutes
# Supports both short format (15m) and pandas format (15min)
TIMEFRAME_MINUTES = {
    '1m': 1,
    '2m': 2,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '3h': 180,
    '4h': 240,
    '1d': 1440,  # 24 * 60
    '1wk': 10080,  # 7 * 24 * 60
    '1mo': 43200,  # ~30 days (monthly only, no 3month)
    # Pandas-style formats
    '1min': 1,
    '2min': 2,
    '5min': 5,
    '15min': 15,
    '30min': 30,
}

# Market hours (for completion calculation during market hours)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
MARKET_MINUTES_PER_DAY = 390  # 6.5 hours


def _parse_timeframe(tf: str) -> int:
    """
    Parse timeframe string to minutes.

    Args:
        tf: Timeframe string like '1m', '5m', '1h', '4h', '1d'
            Also supports pandas formats: '15min', '1D', '1W', '1MS'
            NOTE: 3month (quarterly) is not supported - only monthly

    Returns:
        Number of minutes in the timeframe

    Raises:
        ResamplingError: If timeframe is invalid
    """
    tf_lower = tf.lower()

    if tf_lower in TIMEFRAME_MINUTES:
        return TIMEFRAME_MINUTES[tf_lower]

    # Try parsing custom formats
    try:
        # Pandas-style 'min' suffix (e.g., '15min', '30min')
        if tf_lower.endswith('min'):
            return int(tf_lower[:-3])
        # Short 'm' suffix (e.g., '15m', '30m')
        elif tf_lower.endswith('m') and not tf_lower.endswith('mo'):
            return int(tf_lower[:-1])
        # Hour suffix
        elif tf_lower.endswith('h'):
            return int(tf_lower[:-1]) * 60
        # Day suffix (both 'd' and 'D')
        elif tf_lower.endswith('d'):
            return int(tf_lower[:-1]) * 1440
        # Week suffix
        elif tf_lower.endswith('wk') or tf_lower.endswith('w'):
            num = tf_lower.rstrip('wk').rstrip('w')
            return int(num) * 10080
        # Month suffix (only 1mo supported, not 3mo)
        elif tf_lower.endswith('mo') or tf_lower.endswith('ms'):
            # Handle '1mo', '1MS' (monthly only, no quarterly)
            num_str = tf_lower.rstrip('mos')
            return int(num_str) * 43200  # ~30 days
    except ValueError:
        pass

    raise ResamplingError(f"Invalid timeframe: {tf}")


def _get_resample_rule(tf: str) -> str:
    """
    Convert timeframe to pandas resample rule.

    Args:
        tf: Timeframe string

    Returns:
        Pandas resample rule string
    """
    tf_lower = tf.lower()

    # Direct mappings - include both short and pandas formats
    # NOTE: Only monthly supported, no 3month (quarterly) timeframe
    mappings = {
        '1m': '1min',
        '2m': '2min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '3h': '3h',
        '4h': '4h',
        '1d': '1D',
        '1wk': '1W',
        '1w': '1W',
        '1mo': '1MS',  # Month start
        # Pandas-style formats (pass through)
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1ms': '1MS',
    }

    if tf_lower in mappings:
        return mappings[tf_lower]

    # Parse custom formats
    if tf_lower.endswith('min'):
        return tf_lower  # Already in pandas format
    elif tf_lower.endswith('m') and not tf_lower.endswith('mo'):
        return f"{tf_lower[:-1]}min"
    elif tf_lower.endswith('h'):
        return tf_lower
    elif tf_lower.endswith('d'):
        return tf_lower[:-1] + 'D'

    return tf  # Return as-is, let pandas handle it


def _calculate_bar_completion(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    tf: str,
    now: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculate completion percentage and metadata for the last bar.

    Args:
        df: Resampled DataFrame
        source_df: Original source DataFrame
        tf: Target timeframe
        now: Current time (defaults to last timestamp in source)

    Returns:
        Dictionary with completion metadata
    """
    if len(df) == 0:
        return {
            'bar_completion_pct': 0.0,
            'bars_in_partial': 0,
            'expected_bars': 0,
            'is_partial': False,
            'partial_start': None,
            'partial_end': None,
            'total_bars': 0,
            'source_bars': len(source_df),
        }

    tf_minutes = _parse_timeframe(tf)
    source_tf_minutes = 1  # Assume 1-minute source data by default

    # Detect source timeframe from data
    if len(source_df) >= 2:
        time_diffs = source_df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            source_tf_minutes = max(1, int(median_diff.total_seconds() / 60))

    # Expected bars in a complete target timeframe bar
    expected_bars = max(1, tf_minutes // source_tf_minutes)

    # Get the last resampled bar's timestamp
    last_bar_start = df.index[-1]

    # Count source bars that fall into the last resampled bar
    # The bar covers [last_bar_start, last_bar_start + tf_minutes)
    bar_end = last_bar_start + timedelta(minutes=tf_minutes)

    bars_in_last = len(source_df[
        (source_df.index >= last_bar_start) &
        (source_df.index < bar_end)
    ])

    # Calculate completion percentage
    completion_pct = min(1.0, bars_in_last / expected_bars) if expected_bars > 0 else 1.0

    # Determine if the bar is partial
    is_partial = completion_pct < 1.0

    # Get actual timestamps
    partial_start = last_bar_start if is_partial else None
    partial_end = source_df.index[-1] if is_partial and len(source_df) > 0 else None

    return {
        'bar_completion_pct': round(completion_pct, 4),
        'bars_in_partial': bars_in_last,
        'expected_bars': expected_bars,
        'is_partial': is_partial,
        'partial_start': partial_start,
        'partial_end': partial_end,
        'total_bars': len(df),
        'source_bars': len(source_df),
    }


def resample_with_partial(
    df: pd.DataFrame,
    tf: str,
    source_tf: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resample to target timeframe, keeping the partial (incomplete) bar.

    Unlike dropna(), this keeps the last bar even if incomplete,
    and returns metadata about its completion status.

    This is critical for live trading where the current bar is always
    incomplete, and we need to make predictions on the latest data.

    Args:
        df: Source DataFrame with OHLCV columns and DatetimeIndex
        tf: Target timeframe (e.g., '5m', '15m', '1h', '4h', '1d')
        source_tf: Source timeframe if known (for better completion calc)

    Returns:
        resampled_df: DataFrame with all bars including partial
        metadata: Dict with bar_completion_pct, bars_in_partial, etc.

    Raises:
        ResamplingError: If resampling fails

    Example:
        >>> df_1m = load_1min_data()  # 1-minute bars
        >>> df_15m, meta = resample_with_partial(df_1m, '15m')
        >>> print(f"Last bar is {meta['bar_completion_pct']*100:.1f}% complete")
        >>> print(f"Using {meta['bars_in_partial']}/{meta['expected_bars']} bars")
    """
    if df is None or len(df) == 0:
        raise ResamplingError("Cannot resample empty DataFrame")

    # Validate DataFrame has required columns
    required_cols = ['open', 'high', 'low', 'close']
    # Check case-insensitively
    df_cols_lower = {c.lower(): c for c in df.columns}

    missing = [c for c in required_cols if c not in df_cols_lower]
    if missing:
        raise ResamplingError(f"Missing required columns: {missing}")

    # Normalize column names to lowercase
    col_mapping = {df_cols_lower[c]: c for c in required_cols if c in df_cols_lower}
    if 'volume' in df_cols_lower:
        col_mapping[df_cols_lower['volume']] = 'volume'

    df_normalized = df.rename(columns=col_mapping)

    # Ensure DatetimeIndex
    if not isinstance(df_normalized.index, pd.DatetimeIndex):
        try:
            df_normalized.index = pd.to_datetime(df_normalized.index)
        except Exception as e:
            raise ResamplingError(f"Cannot convert index to DatetimeIndex: {e}")

    # Sort by index
    df_normalized = df_normalized.sort_index()

    # Get resample rule
    rule = _get_resample_rule(tf)

    try:
        # Define aggregation rules
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }

        if 'volume' in df_normalized.columns:
            agg_dict['volume'] = 'sum'

        # Resample WITHOUT dropping NaN - this keeps partial bars
        resampled = df_normalized.resample(rule).agg(agg_dict)

        # Remove completely empty bars (no data at all)
        # But keep partial bars (bars with some data)
        resampled = resampled.dropna(how='all')

        # For partial bars, forward-fill missing OHLC from close
        # This handles the case where we have some but not all data
        if len(resampled) > 0:
            # If open is NaN but close is not, use close for open
            # This can happen with sparse data
            for col in ['open', 'high', 'low']:
                mask = resampled[col].isna() & resampled['close'].notna()
                resampled.loc[mask, col] = resampled.loc[mask, 'close']

    except Exception as e:
        raise ResamplingError(f"Resampling failed: {e}")

    # Calculate metadata about the partial bar
    metadata = _calculate_bar_completion(resampled, df_normalized, tf)

    return resampled, metadata


def get_bar_metadata(
    df: pd.DataFrame,
    tf: str,
    current_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Get metadata about bar completion without resampling.

    Useful when you've already resampled but need to recalculate
    completion status (e.g., when time has passed).

    Args:
        df: Already-resampled DataFrame
        tf: The timeframe the data is in
        current_time: Current time for completion calculation

    Returns:
        Dict with:
            - bar_completion_pct: 0.0 to 1.0
            - time_in_bar: How long since bar started
            - time_remaining: Estimated time until bar completes
            - is_partial: Whether last bar is incomplete
            - bar_start: When the current bar started
            - bar_end: When the current bar will end

    Example:
        >>> meta = get_bar_metadata(df_15m, '15m')
        >>> if meta['bar_completion_pct'] < 0.5:
        ...     print("Warning: Less than half the bar data available")
    """
    if df is None or len(df) == 0:
        return {
            'bar_completion_pct': 0.0,
            'time_in_bar': timedelta(0),
            'time_remaining': timedelta(0),
            'is_partial': False,
            'bar_start': None,
            'bar_end': None,
        }

    tf_minutes = _parse_timeframe(tf)
    tf_delta = timedelta(minutes=tf_minutes)

    # Get the last bar's timestamp (bar start)
    bar_start = df.index[-1]
    bar_end = bar_start + tf_delta

    # Use provided time or current time
    if current_time is None:
        current_time = datetime.now()

    # Make timezone-aware if needed
    if bar_start.tzinfo is not None and current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=bar_start.tzinfo)
    elif bar_start.tzinfo is None and hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
        current_time = current_time.replace(tzinfo=None)

    # Calculate time in bar
    time_in_bar = current_time - bar_start
    time_remaining = bar_end - current_time

    # Clamp values
    if time_in_bar < timedelta(0):
        time_in_bar = timedelta(0)
    if time_remaining < timedelta(0):
        time_remaining = timedelta(0)

    # Calculate completion percentage based on time
    completion_pct = min(1.0, time_in_bar / tf_delta) if tf_delta.total_seconds() > 0 else 1.0

    # Is it partial?
    is_partial = completion_pct < 1.0

    return {
        'bar_completion_pct': round(completion_pct, 4),
        'time_in_bar': time_in_bar,
        'time_remaining': time_remaining,
        'is_partial': is_partial,
        'bar_start': bar_start,
        'bar_end': bar_end,
    }


def validate_resampling(
    source_df: pd.DataFrame,
    resampled_df: pd.DataFrame,
    tf: str
) -> Dict[str, Any]:
    """
    Validate that resampling was done correctly.

    Checks:
    - OHLC relationships (high >= low, etc.)
    - Volume preservation (sum matches)
    - No unexpected data loss
    - Proper bar alignment

    Args:
        source_df: Original source DataFrame
        resampled_df: Resampled DataFrame to validate
        tf: Target timeframe used for resampling

    Returns:
        Dict with validation results and any issues found
    """
    issues = []

    # Check OHLC relationships
    if len(resampled_df) > 0:
        # High should be >= Open, Close, Low
        invalid_high = resampled_df[
            (resampled_df['high'] < resampled_df['open']) |
            (resampled_df['high'] < resampled_df['close']) |
            (resampled_df['high'] < resampled_df['low'])
        ]
        if len(invalid_high) > 0:
            issues.append(f"Found {len(invalid_high)} bars with invalid high values")

        # Low should be <= Open, Close, High
        invalid_low = resampled_df[
            (resampled_df['low'] > resampled_df['open']) |
            (resampled_df['low'] > resampled_df['close']) |
            (resampled_df['low'] > resampled_df['high'])
        ]
        if len(invalid_low) > 0:
            issues.append(f"Found {len(invalid_low)} bars with invalid low values")

    # Check volume preservation if both have volume
    if 'volume' in source_df.columns and 'volume' in resampled_df.columns:
        source_vol = source_df['volume'].sum()
        resampled_vol = resampled_df['volume'].sum()

        if source_vol > 0:
            vol_diff_pct = abs(resampled_vol - source_vol) / source_vol
            if vol_diff_pct > 0.01:  # More than 1% difference
                issues.append(
                    f"Volume mismatch: source={source_vol:.0f}, "
                    f"resampled={resampled_vol:.0f} ({vol_diff_pct*100:.1f}% diff)"
                )

    # Check for reasonable bar count
    tf_minutes = _parse_timeframe(tf)
    if len(source_df) >= 2:
        source_minutes = (source_df.index[-1] - source_df.index[0]).total_seconds() / 60
        expected_bars = max(1, int(source_minutes / tf_minutes))
        actual_bars = len(resampled_df)

        if actual_bars > expected_bars * 1.5:
            issues.append(
                f"Too many bars: expected ~{expected_bars}, got {actual_bars}"
            )
        elif actual_bars < expected_bars * 0.5:
            issues.append(
                f"Too few bars: expected ~{expected_bars}, got {actual_bars}"
            )

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'source_bars': len(source_df),
        'resampled_bars': len(resampled_df),
    }
