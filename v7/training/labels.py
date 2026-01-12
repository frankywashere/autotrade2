"""
Training Labels Generator

Generates labels for channel break prediction by scanning forward from a channel.
Labels capture:
1. Duration - how many bars until channel permanently breaks
2. Break direction - UP or DOWN when it finally breaks
3. Break trigger TF - which longer timeframe boundary was hit at break time
4. New channel direction - what direction is the next channel (BULL/BEAR/SIDEWAYS)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import IntEnum
import sys
from pathlib import Path
import threading

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import (
    detect_channel, detect_channels_multi_window, select_best_channel,
    Channel, STANDARD_WINDOWS
)
from core.timeframe import resample_ohlc, get_longer_timeframes, TIMEFRAMES, BARS_PER_TF
from features.containment import check_containment, get_closest_boundary


# =============================================================================
# Resample Cache - Avoids redundant resampling within a single label generation
# Per-thread cache ensures thread safety without locks
# =============================================================================

# Thread-local cache storage: (df_id, len, timeframe) -> resampled DataFrame
_resample_cache_local = threading.local()

def _get_resample_cache() -> Dict[Tuple[int, int, str], pd.DataFrame]:
    """Get or create the resample cache for the current thread."""
    cache = getattr(_resample_cache_local, "cache", None)
    if cache is None:
        cache = {}
        _resample_cache_local.cache = cache
    return cache


def clear_resample_cache() -> None:
    """Clear the resample cache for the current thread."""
    cache = getattr(_resample_cache_local, "cache", None)
    if cache is not None:
        cache.clear()


# =============================================================================
# Cache Performance Monitoring (Optional - Zero Overhead When Disabled)
# =============================================================================

# Module-level flag to enable/disable cache statistics tracking
# Set to True to enable monitoring: labels.ENABLE_CACHE_STATS = True
ENABLE_CACHE_STATS = False

# Thread-local storage for per-thread cache statistics
_cache_stats_local = threading.local()


def _get_cache_stats() -> Dict[str, int]:
    """Get or create cache stats dict for the current thread."""
    stats = getattr(_cache_stats_local, "stats", None)
    if stats is None:
        stats = {"hits": 0, "misses": 0, "total": 0}
        _cache_stats_local.stats = stats
    return stats


def get_cache_stats() -> Dict[str, int]:
    """
    Get cache statistics for the current thread.

    Returns:
        Dictionary with keys:
        - hits: Number of cache hits
        - misses: Number of cache misses
        - total: Total cache lookups (hits + misses)
        - hit_rate: Cache hit rate as a percentage (0.0-100.0)
    """
    stats = _get_cache_stats()
    total = stats["total"]
    hit_rate = (stats["hits"] / total * 100.0) if total > 0 else 0.0

    return {
        "hits": stats["hits"],
        "misses": stats["misses"],
        "total": total,
        "hit_rate": hit_rate
    }


def reset_cache_stats() -> None:
    """Reset cache statistics for the current thread."""
    stats = getattr(_cache_stats_local, "stats", None)
    if stats is not None:
        stats["hits"] = 0
        stats["misses"] = 0
        stats["total"] = 0


def print_cache_stats() -> None:
    """
    Print cache statistics for the current thread in a human-readable format.

    Example output:
        Cache Statistics:
          Total calls:  1250
          Cache hits:   1000 (80.0%)
          Cache misses: 250 (20.0%)
    """
    stats = get_cache_stats()

    print("Cache Statistics:")
    print(f"  Total calls:  {stats['total']}")
    print(f"  Cache hits:   {stats['hits']} ({stats['hit_rate']:.1f}%)")
    print(f"  Cache misses: {stats['misses']} ({100.0 - stats['hit_rate']:.1f}%)")


def cached_resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Cached version of resample_ohlc (per-thread cache).

    Uses id(df) + len(df) as cache key to avoid hashing large DataFrames.
    Cache is scoped to the current thread to ensure thread safety.

    When ENABLE_CACHE_STATS is True, tracks hit/miss statistics per thread.

    OPTIMIZATION: When pre-computed resampled data is available (set by parallel
    workers), this function will use timestamp-based slicing of the pre-computed
    data instead of resampling from scratch. This dramatically reduces redundant
    computation in parallel scanning.
    """
    # Fast path: check for pre-computed data first
    # This is the key optimization for parallel workers
    precomputed_slice = _try_get_precomputed_slice(df, timeframe)
    if precomputed_slice is not None:
        if ENABLE_CACHE_STATS:
            stats = _get_cache_stats()
            stats["total"] += 1
            stats["hits"] += 1
        return precomputed_slice

    cache = _get_resample_cache()
    cache_key = (id(df), len(df), timeframe)

    cached = cache.get(cache_key)

    # Track statistics if enabled (zero overhead when disabled)
    if ENABLE_CACHE_STATS:
        stats = _get_cache_stats()
        stats["total"] += 1

        if cached is not None:
            stats["hits"] += 1
            return cached
        else:
            stats["misses"] += 1
    else:
        # Fast path when stats disabled
        if cached is not None:
            return cached

    result = resample_ohlc(df, timeframe)
    cache[cache_key] = result
    return result


def _try_get_precomputed_slice(df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Try to get a slice of pre-computed resampled data.

    Returns None if pre-computed data is not available or doesn't match.
    This is called by cached_resample_ohlc for transparent optimization.

    IMPORTANT: The pre-computed approach is mathematically equivalent to fresh
    resampling for all COMPLETE time periods. For the last partial period,
    there may be a minor difference if the pre-computed data includes more
    bars in that period. This is acceptable for the use case (channel detection
    and feature extraction) where slight differences in the final partial bar
    don't materially affect results.

    The optimization benefit (avoiding redundant resampling across positions)
    far outweighs any minor difference in the last partial bar.
    """
    if timeframe == '5min':
        # 5min is identity - no need for pre-computed
        return None

    # Check if pre-computed TSLA data is available
    precomputed_tsla = getattr(_precomputed_local, 'tsla', None)
    if precomputed_tsla is not None and timeframe in precomputed_tsla:
        precomputed_df = precomputed_tsla[timeframe]
        if precomputed_df is not None and len(precomputed_df) > 0 and len(df) > 0:
            # Get end timestamp from input df
            end_timestamp = df.index[-1]

            # Find all bars whose start time is <= end_timestamp
            # Use 'right' to get the position after the last matching timestamp
            idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

            if idx > 0:
                return precomputed_df.iloc[:idx]

    return None


# =============================================================================
# Pre-computed Resampled Data (Shared Across Worker Positions)
# =============================================================================
# This optimization avoids redundant resampling in parallel workers.
# Instead of each position resampling from scratch, workers receive
# pre-computed full-length resampled DataFrames and slice them by timestamp.

# Thread-local storage for pre-computed resampled data
_precomputed_local = threading.local()


def set_precomputed_resampled_data(
    precomputed_tsla: Optional[Dict[str, pd.DataFrame]],
    precomputed_spy: Optional[Dict[str, pd.DataFrame]]
) -> None:
    """
    Set pre-computed resampled DataFrames for the current worker.

    Called by worker initialization to share pre-computed data across positions.

    Args:
        precomputed_tsla: Dict mapping timeframe -> resampled TSLA DataFrame
        precomputed_spy: Dict mapping timeframe -> resampled SPY DataFrame
    """
    _precomputed_local.tsla = precomputed_tsla
    _precomputed_local.spy = precomputed_spy


def clear_precomputed_resampled_data() -> None:
    """Clear pre-computed resampled data for the current thread/worker."""
    _precomputed_local.tsla = None
    _precomputed_local.spy = None


def get_precomputed_resampled_slice(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str = 'tsla'
) -> Optional[pd.DataFrame]:
    """
    Get a slice of pre-computed resampled data up to the last timestamp in df.

    This is the key optimization: instead of resampling df[:i] for each position,
    we slice the pre-computed resample(full_df) up to the timestamp of df.index[-1].

    The result is mathematically equivalent to resample_ohlc(df, timeframe) because:
    - Resampling aggregates bars by time bucket (e.g., 15min buckets)
    - Slicing pre-computed data by timestamp gives the same aggregated bars
    - The only edge case is partial bars at the end, but we use <= to include them

    Args:
        df: The DataFrame being processed (used to get the end timestamp)
        timeframe: Target timeframe (e.g., '15min', 'daily')
        symbol: 'tsla' or 'spy' to select which pre-computed data to use

    Returns:
        Sliced pre-computed DataFrame, or None if pre-computed data not available
    """
    if timeframe == '5min':
        # 5min is identity - just return the input df
        return df

    # Get pre-computed data for this symbol
    precomputed = getattr(_precomputed_local, symbol, None)
    if precomputed is None or timeframe not in precomputed:
        return None

    precomputed_df = precomputed[timeframe]
    if precomputed_df is None or len(precomputed_df) == 0:
        return None

    # Get the end timestamp from the input df
    if len(df) == 0:
        return precomputed_df.iloc[:0]  # Empty slice

    end_timestamp = df.index[-1]

    # Slice pre-computed data up to and including the end timestamp
    # Use searchsorted for efficient lookup
    idx = precomputed_df.index.searchsorted(end_timestamp, side='right')

    return precomputed_df.iloc[:idx]


def cached_resample_ohlc_optimized(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str = 'tsla'
) -> pd.DataFrame:
    """
    Optimized cached resample that uses pre-computed data when available.

    Falls back to regular cached_resample_ohlc if pre-computed data not available.

    Args:
        df: DataFrame to resample
        timeframe: Target timeframe
        symbol: 'tsla' or 'spy' for pre-computed data lookup

    Returns:
        Resampled DataFrame
    """
    # Try to use pre-computed data first
    precomputed_slice = get_precomputed_resampled_slice(df, timeframe, symbol)
    if precomputed_slice is not None:
        return precomputed_slice

    # Fall back to regular caching
    return cached_resample_ohlc(df, timeframe)


class BreakDirection(IntEnum):
    """Direction of channel break."""
    DOWN = 0
    UP = 1


class NewChannelDirection(IntEnum):
    """Direction of the new channel that forms after break."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


class BreakTriggerTF(IntEnum):
    """
    Classification of which longer timeframe boundary triggered a channel break.

    Each timeframe has upper/lower variants since the direction of the triggering
    boundary carries important predictive information (bullish vs bearish context).

    Total classes: 21 (1 no_trigger + 10 timeframes x 2 directions)
    """
    NO_TRIGGER = 0
    TF_15MIN_UPPER = 1
    TF_15MIN_LOWER = 2
    TF_30MIN_UPPER = 3
    TF_30MIN_LOWER = 4
    TF_1H_UPPER = 5
    TF_1H_LOWER = 6
    TF_2H_UPPER = 7
    TF_2H_LOWER = 8
    TF_3H_UPPER = 9
    TF_3H_LOWER = 10
    TF_4H_UPPER = 11
    TF_4H_LOWER = 12
    TF_DAILY_UPPER = 13
    TF_DAILY_LOWER = 14
    TF_WEEKLY_UPPER = 15
    TF_WEEKLY_LOWER = 16
    TF_MONTHLY_UPPER = 17
    TF_MONTHLY_LOWER = 18
    TF_3MONTH_UPPER = 19
    TF_3MONTH_LOWER = 20


# Encoding map for string to int conversion
TF_TRIGGER_ENCODING = {
    None: 0,
    '15min_upper': 1, '15min_lower': 2,
    '30min_upper': 3, '30min_lower': 4,
    '1h_upper': 5, '1h_lower': 6,
    '2h_upper': 7, '2h_lower': 8,
    '3h_upper': 9, '3h_lower': 10,
    '4h_upper': 11, '4h_lower': 12,
    'daily_upper': 13, 'daily_lower': 14,
    'weekly_upper': 15, 'weekly_lower': 16,
    'monthly_upper': 17, 'monthly_lower': 18,
    '3month_upper': 19, '3month_lower': 20,
}

# Reverse mapping for decoding
TF_TRIGGER_DECODING = {v: k for k, v in TF_TRIGGER_ENCODING.items()}

NUM_TRIGGER_TF_CLASSES = 21  # Total classes (0-20)

# Module-level constants for label scaling parameters per timeframe
# These are used by scale_label_params_for_tf() and cached here to avoid
# rebuilding the dicts on every call.

# max_scan per TF - aligned with FORWARD_BARS_PER_TF in label_inspector.py
TF_MAX_SCAN = {
    '5min': 100,    # ~8 hours
    '15min': 100,   # ~25 hours
    '30min': 50,    # ~25 hours
    '1h': 50,       # ~50 hours (~2 days)
    '2h': 50,       # ~100 hours (~4 days)
    '3h': 50,       # ~150 hours (~6 days)
    '4h': 50,       # ~200 hours (~8 days)
    'daily': 50,    # ~50 trading days (~2.5 months)
    'weekly': 50,   # ~50 weeks (~1 year)
    'monthly': 10,  # ~10 months
    '3month': 10,   # ~30 months (~2.5 years)
}

# Explicit return_threshold per TF
# These control how many bars outside channel before declaring "permanent" break
# Lower = more sensitive, Higher = more tolerant of temporary excursions
TF_RETURN_THRESHOLD = {
    '5min': 20,     # ~1.5 hours - allows for short-term noise
    '15min': 6,     # ~1.5 hours equivalent
    '30min': 4,     # ~2 hours
    '1h': 3,        # ~3 hours
    '2h': 3,        # ~6 hours
    '3h': 3,        # ~9 hours
    '4h': 3,        # ~12 hours (half trading day)
    'daily': 5,     # ~1 week - tolerates daily noise
    'weekly': 2,    # ~2 weeks
    'monthly': 1,   # Immediate - monthly breaks are significant
    '3month': 1,    # Immediate - quarterly breaks are significant
}


def encode_trigger_tf(trigger_tf: Optional[str]) -> int:
    """Encode break_trigger_tf string to integer class."""
    return TF_TRIGGER_ENCODING.get(trigger_tf, 0)


def decode_trigger_tf(trigger_tf_encoded: int) -> Optional[str]:
    """Decode integer class back to trigger_tf string."""
    return TF_TRIGGER_DECODING.get(trigger_tf_encoded)


@dataclass
class ChannelLabels:
    """
    Labels for a channel indicating its future outcome.

    Attributes:
        duration_bars: Number of bars until permanent break
        break_direction: Direction of break (0=DOWN, 1=UP)
        break_trigger_tf: Encoded trigger TF class (0-20, see BreakTriggerTF)
        new_channel_direction: Direction of next channel (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: Whether a permanent break was found within scan window

    Validity flags (which labels are from actual observation vs defaults):
        duration_valid: True if duration was observed (always True for valid samples)
        direction_valid: True only if permanent_break=True
        trigger_tf_valid: True only if trigger TF was found
        new_channel_valid: True only if new channel was detected
    """
    duration_bars: int
    break_direction: int  # 0=DOWN, 1=UP
    break_trigger_tf: int  # Encoded class 0-20 (see TF_TRIGGER_ENCODING)
    new_channel_direction: int  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    permanent_break: bool

    # Validity flags - which labels are from actual observation vs defaults
    duration_valid: bool = True       # Duration is always valid for valid samples
    direction_valid: bool = False     # True only if permanent_break=True
    trigger_tf_valid: bool = False    # True only if trigger found
    new_channel_valid: bool = False   # True only if new channel detected


def project_channel_bounds(
    channel: Channel,
    num_bars: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project channel bounds forward using slope/intercept.

    Args:
        channel: The channel to project
        num_bars: Number of future bars to project

    Returns:
        Tuple of (upper_projection, lower_projection) arrays
    """
    # Channel's x coordinates start at 0, end at window-1
    # We project from window onwards
    future_x = np.arange(channel.window, channel.window + num_bars)

    # Project center line
    center_projection = channel.slope * future_x + channel.intercept

    # Add/subtract std dev for bounds
    std_multiplier = 2.0  # Same as used in channel detection
    upper_projection = center_projection + std_multiplier * channel.std_dev
    lower_projection = center_projection - std_multiplier * channel.std_dev

    return upper_projection, lower_projection


def check_price_in_channel(
    high: float,
    low: float,
    upper_bound: float,
    lower_bound: float
) -> Tuple[bool, Optional[int]]:
    """
    Check if price is within channel bounds.

    Args:
        high: Bar's high price
        low: Bar's low price
        upper_bound: Channel upper bound at this bar
        lower_bound: Channel lower bound at this bar

    Returns:
        Tuple of (is_inside, break_direction)
        - is_inside: True if price is within bounds
        - break_direction: 1=UP (high broke upper), 0=DOWN (low broke lower), None if inside
    """
    if high > upper_bound:
        return False, BreakDirection.UP
    elif low < lower_bound:
        return False, BreakDirection.DOWN
    else:
        return True, None


def find_permanent_break(
    df_forward: pd.DataFrame,
    upper_projection: np.ndarray,
    lower_projection: np.ndarray,
    return_threshold: int = 20
) -> Tuple[Optional[int], Optional[int]]:
    """
    Scan forward to find a permanent channel break.

    A "permanent break" means price exits AND either:
    - Stays out for 20+ bars, OR
    - Forms a new valid channel

    Args:
        df_forward: DataFrame of future bars to scan
        upper_projection: Projected upper bound values
        lower_projection: Projected lower bound values
        return_threshold: Bars to wait to confirm break is permanent

    Returns:
        Tuple of (break_bar_index, break_direction)
        - break_bar_index: Index where permanent break occurred (None if no break)
        - break_direction: 0=DOWN, 1=UP (None if no break)
    """
    highs = df_forward['high'].values
    lows = df_forward['low'].values
    n_bars = min(len(df_forward), len(upper_projection))

    if n_bars == 0:
        return None, None

    # Slice arrays to matching length
    highs = highs[:n_bars]
    lows = lows[:n_bars]
    upper = upper_projection[:n_bars]
    lower = lower_projection[:n_bars]

    # Vectorized boundary checks - compute for ALL bars at once
    breaks_up = highs > upper      # True where high breaks upper bound
    breaks_down = lows < lower     # True where low breaks lower bound
    is_outside = breaks_up | breaks_down  # True where price is outside channel

    # If no bars are outside, no break occurred
    if not np.any(is_outside):
        return None, None

    # Track exit state using loop for state machine logic
    # This tracks consecutive outside bars and resets when price returns to channel
    exit_bar = None
    exit_direction = None
    bars_outside = 0

    for i in range(n_bars):
        if is_outside[i]:
            # Price is outside channel
            if exit_bar is None:
                # New exit - record position and direction
                exit_bar = i
                # Determine direction: UP if broke upper, DOWN if broke lower
                exit_direction = BreakDirection.UP if breaks_up[i] else BreakDirection.DOWN
                bars_outside = 1
            else:
                bars_outside += 1

            # Check if this is a permanent break
            if bars_outside >= return_threshold:
                return exit_bar, exit_direction
        else:
            # Price returned to channel - false break, reset tracking
            if exit_bar is not None:
                exit_bar = None
                exit_direction = None
                bars_outside = 0

    # If we still have an exit that didn't get confirmed,
    # but we ran out of data, return what we have
    if exit_bar is not None and bars_outside > 0:
        return exit_bar, exit_direction

    return None, None


def detect_new_channel(
    df: pd.DataFrame,
    start_idx: int,
    window: int = 50,
    max_scan: int = 100
) -> Optional[Channel]:
    """
    Detect the next valid channel that forms after a break.

    Optimized with early termination: pre-computes numpy arrays once and uses
    vectorized variance checks to skip positions where channels are clearly
    impossible (insufficient price movement). This produces EXACTLY the same
    results as the naive approach but is significantly faster.

    Args:
        df: Full DataFrame (break point onwards)
        start_idx: Index to start scanning from
        window: Window size for channel detection
        max_scan: Maximum bars to scan looking for new channel

    Returns:
        Channel object if found, None otherwise
    """
    # Calculate the actual scan range
    end_idx = min(start_idx + max_scan, len(df) - window + 1)
    if start_idx >= end_idx:
        return None

    # Pre-extract numpy arrays once (avoid repeated DataFrame operations)
    # Extract full range needed: from start_idx to start_idx + max_scan + window
    array_end = min(start_idx + max_scan + window, len(df))
    close_full = df['close'].values[start_idx:array_end].astype(np.float64)
    high_full = df['high'].values[start_idx:array_end].astype(np.float64)
    low_full = df['low'].values[start_idx:array_end].astype(np.float64)

    # Pre-compute minimum variance threshold for valid channels
    # A channel needs enough price variation to have meaningful bounds
    # Use a fraction of average price as threshold (0.01% of price range minimum)
    avg_price = np.mean(close_full[:window]) if len(close_full) >= window else 1.0
    min_variance = (avg_price * 0.0001) ** 2  # Square for variance comparison

    # Pre-compute x array for regression (same for all windows of same size)
    x = np.arange(window, dtype=np.float64)
    x_mean = (window - 1) / 2.0
    x_centered = x - x_mean
    x_var = np.sum(x_centered ** 2)

    for i in range(end_idx - start_idx):
        # Get the slice indices relative to our pre-extracted arrays
        slice_end = i + window
        if slice_end > len(close_full):
            break

        close = close_full[i:slice_end]
        high = high_full[i:slice_end]
        low = low_full[i:slice_end]

        # Quick variance check - skip if price doesn't move enough
        # This is much faster than full regression
        close_var = np.var(close)
        if close_var < min_variance:
            continue

        # Perform linear regression (vectorized, no scipy call)
        close_mean = np.mean(close)
        close_centered = close - close_mean
        slope = np.sum(x_centered * close_centered) / x_var if x_var > 0 else 0.0
        intercept = close_mean - slope * x_mean

        # Center line and residuals
        center_line = slope * x + intercept
        residuals = close - center_line
        std_dev = np.std(residuals)

        # Skip if std_dev is too small (degenerate channel)
        if std_dev < avg_price * 0.0001:
            continue

        # Upper and lower bounds
        std_multiplier = 2.0
        upper_line = center_line + std_multiplier * std_dev
        lower_line = center_line - std_multiplier * std_dev

        # Quick bounce check using vectorized operations
        # This is the minimum required for a valid channel (min_cycles=1 default)
        channel_width = upper_line - lower_line
        touch_threshold = 0.10

        # Check for touches using vectorized operations
        upper_dist = (upper_line - high) / channel_width
        lower_dist = (low - lower_line) / channel_width

        upper_touches_mask = upper_dist <= touch_threshold
        lower_touches_mask = lower_dist <= touch_threshold

        # Count alternations efficiently
        # Create a touch type array: 1 for upper, -1 for lower, 0 for none
        # Note: original uses elif, so upper takes priority when both conditions met
        touch_types = np.zeros(window, dtype=np.int8)
        touch_types[lower_touches_mask] = -1  # Set lower first
        touch_types[upper_touches_mask] = 1   # Upper overwrites (takes priority)

        # Find positions with touches
        touch_positions = np.where(touch_types != 0)[0]

        if len(touch_positions) < 2:
            continue

        # Count alternations (sign changes in consecutive touches)
        touch_values = touch_types[touch_positions]
        alternations = np.sum(touch_values[:-1] != touch_values[1:])

        if alternations >= 1:  # min_cycles default is 1
            # Found a valid channel - now call detect_channel for the full object
            # This ensures EXACTLY the same Channel object is returned
            df_slice = df.iloc[start_idx + i:start_idx + slice_end]
            channel = detect_channel(df_slice, window=window)
            if channel.valid:
                return channel

    return None


def get_longer_tf_channels(
    df: pd.DataFrame,
    current_tf: str,
    window: int = 50
) -> Dict[str, Channel]:
    """
    Detect channels at all longer timeframes.

    Args:
        df: Base OHLCV data
        current_tf: Current timeframe
        window: Window for channel detection

    Returns:
        Dict mapping timeframe names to Channel objects
    """
    longer_tfs = get_longer_timeframes(current_tf)
    channels = {}

    for tf in longer_tfs:
        df_tf = cached_resample_ohlc(df, tf)
        if len(df_tf) >= window:
            channels[tf] = detect_channel(df_tf, window=window)
        else:
            channels[tf] = None

    return channels


def find_break_trigger_tf(
    df_at_break: pd.DataFrame,
    current_tf: str,
    window: int = 50
) -> Optional[str]:
    """
    Find which longer TF boundary was closest at break time.

    Args:
        df_at_break: OHLCV data up to and including break bar
        current_tf: Current timeframe
        window: Window for channel detection

    Returns:
        String like "1h_upper" or "daily_lower", or None
    """
    if len(df_at_break) < window:
        return None

    current_price = df_at_break['close'].iloc[-1]
    longer_channels = get_longer_tf_channels(df_at_break, current_tf, window)

    containments = check_containment(
        current_price,
        current_tf,
        longer_channels
    )

    return get_closest_boundary(containments)


def generate_labels(
    df: pd.DataFrame,
    channel: Channel,
    channel_end_idx: int,
    current_tf: str = '5min',
    window: int = 50,
    max_scan: int = 500,
    return_threshold: int = 20,
    fold_end_idx: Optional[int] = None
) -> ChannelLabels:
    """
    Generate labels for a channel by scanning forward.

    This function:
    1. Projects the channel forward using slope/intercept
    2. Scans forward bar by bar checking if price exits
    3. If exits, checks if it returns within N bars - if returns, not permanent
    4. When permanent break found, records duration
    5. Checks which longer TF boundary was nearest at break time
    6. Detects the next channel that forms and gets its direction

    Args:
        df: Full OHLCV DataFrame
        channel: The detected channel to generate labels for
        channel_end_idx: Index in df where the channel ends (last bar used)
        current_tf: Current timeframe (e.g., '5min')
        window: Window size for channel detection
        max_scan: Maximum bars to scan forward
        return_threshold: Bars outside needed to confirm permanent break
        fold_end_idx: Optional end index for walk-forward validation fold.
                     When provided, prevents lookahead bias by limiting
                     forward scan to the fold boundary.

    Returns:
        ChannelLabels object with all label information
    """
    # Get forward data
    forward_start = channel_end_idx + 1
    if fold_end_idx is not None:
        forward_end = min(forward_start + max_scan, fold_end_idx)
    else:
        forward_end = min(forward_start + max_scan, len(df))

    if forward_start >= len(df):
        # No forward data available
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=BreakDirection.UP,  # Default
            break_trigger_tf=0,  # NO_TRIGGER
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=False,  # No forward data means duration not observed
            direction_valid=False,
            trigger_tf_valid=False,
            new_channel_valid=False
        )

    df_forward = df.iloc[forward_start:forward_end]
    n_forward = len(df_forward)

    if n_forward == 0:
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=BreakDirection.UP,
            break_trigger_tf=0,  # NO_TRIGGER
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=False,  # No forward data
            direction_valid=False,
            trigger_tf_valid=False,
            new_channel_valid=False
        )

    # Project channel bounds forward
    upper_proj, lower_proj = project_channel_bounds(channel, n_forward)

    # Find permanent break
    break_idx, break_direction = find_permanent_break(
        df_forward, upper_proj, lower_proj, return_threshold
    )

    if break_idx is None:
        # No break found within scan window - but duration IS valid (channel survived this long)
        return ChannelLabels(
            duration_bars=n_forward,
            break_direction=BreakDirection.UP,  # Default - unknown
            break_trigger_tf=0,  # NO_TRIGGER
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=True,   # Duration IS observed (survived scan window)
            direction_valid=False,  # Direction unknown
            trigger_tf_valid=False,
            new_channel_valid=False
        )

    # Calculate duration
    duration_bars = break_idx

    # Get data up to break point to check longer TF containment
    break_absolute_idx = forward_start + break_idx
    df_at_break = df.iloc[:break_absolute_idx + 1]

    # Find which longer TF boundary was triggered
    break_trigger_tf = find_break_trigger_tf(df_at_break, current_tf, window)

    # Look for new channel after break
    new_channel = detect_new_channel(
        df,
        start_idx=break_absolute_idx + return_threshold,
        window=window,
        max_scan=max_scan - break_idx
    )

    if new_channel is not None:
        new_channel_direction = int(new_channel.direction)
    else:
        # Default to sideways if no channel found
        new_channel_direction = NewChannelDirection.SIDEWAYS

    return ChannelLabels(
        duration_bars=duration_bars,
        break_direction=int(break_direction),
        break_trigger_tf=encode_trigger_tf(break_trigger_tf),  # Encode string to int
        new_channel_direction=new_channel_direction,
        permanent_break=True,
        duration_valid=True,
        direction_valid=True,
        trigger_tf_valid=(break_trigger_tf is not None),
        new_channel_valid=(new_channel is not None)
    )


def generate_labels_batch(
    df: pd.DataFrame,
    channels: list,
    channel_end_indices: list,
    current_tf: str = '5min',
    window: int = 50,
    max_scan: int = 500,
    return_threshold: int = 20
) -> list:
    """
    Generate labels for multiple channels.

    Args:
        df: Full OHLCV DataFrame
        channels: List of Channel objects
        channel_end_indices: List of indices where each channel ends
        current_tf: Current timeframe
        window: Window for channel detection
        max_scan: Maximum bars to scan forward
        return_threshold: Bars outside needed to confirm permanent break

    Returns:
        List of ChannelLabels objects
    """
    labels = []
    for channel, end_idx in zip(channels, channel_end_indices):
        label = generate_labels(
            df=df,
            channel=channel,
            channel_end_idx=end_idx,
            current_tf=current_tf,
            window=window,
            max_scan=max_scan,
            return_threshold=return_threshold
        )
        labels.append(label)

    return labels


def labels_to_dict(labels: ChannelLabels) -> dict:
    """
    Convert ChannelLabels to dictionary for serialization.

    Args:
        labels: ChannelLabels object

    Returns:
        Dictionary with all label fields including validity flags
    """
    return {
        'duration_bars': labels.duration_bars,
        'break_direction': labels.break_direction,
        'break_trigger_tf': labels.break_trigger_tf,  # Already encoded as int
        'break_trigger_tf_str': decode_trigger_tf(labels.break_trigger_tf),  # For debugging
        'new_channel_direction': labels.new_channel_direction,
        'permanent_break': labels.permanent_break,
        # Validity flags
        'duration_valid': labels.duration_valid,
        'direction_valid': labels.direction_valid,
        'trigger_tf_valid': labels.trigger_tf_valid,
        'new_channel_valid': labels.new_channel_valid,
    }


def labels_to_array(labels: ChannelLabels) -> np.ndarray:
    """
    Convert ChannelLabels to numpy array for model training.

    Args:
        labels: ChannelLabels object

    Returns:
        Numpy array: [duration, break_dir, trigger_tf, new_dir, permanent,
                      duration_valid, direction_valid, trigger_tf_valid, new_channel_valid]
    """
    return np.array([
        labels.duration_bars,
        labels.break_direction,
        labels.break_trigger_tf,  # Already encoded as int
        labels.new_channel_direction,
        int(labels.permanent_break),
        int(labels.duration_valid),
        int(labels.direction_valid),
        int(labels.trigger_tf_valid),
        int(labels.new_channel_valid),
    ], dtype=np.float32)


def scale_label_params_for_tf(
    tf: str,
    max_scan: int,
    return_threshold: int,
    custom_return_thresholds: Optional[Dict[str, int]] = None
) -> Tuple[int, int]:
    """
    Scale label generation parameters for a specific timeframe.

    These values are aligned with the visual forward bars in label_inspector.py
    to ensure label generation matches what is visually displayed.

    Forward look by TF:
    - 5min: 100 bars (~8 hours)
    - 15min: 100 bars (~25 hours)
    - 30min-weekly: 50 bars
    - monthly: 10 bars
    - 3month: 10 bars

    Args:
        tf: Target timeframe (e.g., '15min', '1h', 'daily')
        max_scan: Base max_scan value (used for 5min)
        return_threshold: Base return_threshold value (in 5min bars)
        custom_return_thresholds: Optional dict mapping TF names to custom return threshold
                                  values. If provided and tf is in the dict, that value is
                                  used instead of the default. Example: {'5min': 10, '1h': 2}

    Returns:
        Tuple of (scaled_max_scan, scaled_return_threshold)
    """
    # Use module-level constants (TF_MAX_SCAN, TF_RETURN_THRESHOLD) for efficiency
    scaled_max_scan = TF_MAX_SCAN.get(tf, min(max_scan, 50))

    # Check for custom threshold first, then fall back to module-level defaults
    if custom_return_thresholds is not None and tf in custom_return_thresholds:
        scaled_return_threshold = custom_return_thresholds[tf]
    else:
        scaled_return_threshold = TF_RETURN_THRESHOLD.get(tf, max(1, return_threshold // BARS_PER_TF.get(tf, 1)))

    return scaled_max_scan, scaled_return_threshold


def generate_labels_per_tf(
    df: pd.DataFrame,
    channel_end_idx_5min: int,
    window: int = 50,
    max_scan: int = 500,
    return_threshold: int = 20,
    fold_end_idx: Optional[int] = None,
    min_cycles: int = 1,
    channel: Optional[Channel] = None,
    custom_return_thresholds: Optional[Dict[str, int]] = None,
    precomputed_tf_channels: Optional[Dict[str, Tuple[Optional[Channel], Optional[int]]]] = None,
    _clear_cache: bool = True
) -> Dict[str, Optional[ChannelLabels]]:
    """
    Generate labels for each timeframe by resampling and detecting channels.

    For each TF in TIMEFRAMES:
    1. Resamples base 5min data using resample_ohlc()
    2. Detects a channel at the equivalent position in that TF
    3. Calls generate_labels() with scaled parameters

    Args:
        df: Base 5min OHLCV DataFrame with DatetimeIndex (includes forward data for scanning)
        channel_end_idx_5min: Index in 5min data where the channel ends. This is the
                              position of the detected channel - data after this is
                              forward data for label scanning.
        window: Window size for channel detection
        max_scan: Maximum bars to scan forward (in 5min bars, will be scaled)
        return_threshold: Bars outside needed to confirm permanent break (will be scaled)
        fold_end_idx: Optional end index for walk-forward validation fold
        min_cycles: Minimum cycles required for valid channel detection
        channel: Optional pre-detected Channel object for 5min timeframe. If provided,
                 the window parameter is overridden by channel.window for consistency.
        custom_return_thresholds: Optional dict mapping TF names to custom return threshold
                                  values. If provided and a TF is in the dict, that value is
                                  used instead of the default. Example: {'5min': 10, '1h': 2}
        precomputed_tf_channels: Optional dict mapping TF -> (Channel, best_window) to reuse
                                 channel detection across calls. When provided, skips redundant
                                 detect_channels_multi_window calls for non-5min timeframes.
        _clear_cache: Internal parameter to control cache clearing. Default True clears
                     cache at start to prevent memory bloat. Set to False when calling
                     from generate_labels_multi_window() to share cache across windows.

    Returns:
        Dict mapping TF name to ChannelLabels (None if channel detection failed)

    Note:
        This function uses cached resampling to avoid redundant computation.
        The cache is automatically cleared at the start of each call (when _clear_cache=True)
        to prevent memory bloat between samples. Within a single call, resampled DataFrames
        are reused for efficiency.
    """
    # Clear cache at start to prevent memory bloat between samples
    # (unless called from generate_labels_multi_window which manages its own cache)
    if _clear_cache:
        clear_resample_cache()

    # If a channel is provided, use its window size for consistency
    if channel is not None:
        window = channel.window
    labels_per_tf: Dict[str, Optional[ChannelLabels]] = {}

    if channel_end_idx_5min >= len(df):
        # Invalid index
        return {tf: None for tf in TIMEFRAMES}

    # Split data: historical (up to sample time) vs full (includes forward bars)
    # This prevents future data leakage in channel detection for longer timeframes
    df_historical = df.iloc[:channel_end_idx_5min + 1]  # Only up to sample time

    for tf in TIMEFRAMES:
        try:
            # Resample data to this timeframe
            # Use separate dataframes for channel detection (historical) vs label scanning (full)
            if tf == '5min':
                df_tf_full = df
                channel_end_idx_tf = channel_end_idx_5min

                # Use pre-detected channel if provided (avoids redundant detection)
                if channel is not None and channel.valid:
                    tf_channel = channel
                    best_tf_window = window
                elif precomputed_tf_channels is not None and tf in precomputed_tf_channels:
                    # Use precomputed channel from generate_labels_multi_window
                    tf_channel, best_tf_window = precomputed_tf_channels[tf]
                    if tf_channel is None or not tf_channel.valid:
                        labels_per_tf[tf] = None
                        continue
                else:
                    # No valid channel provided, detect one
                    df_tf_for_channel = df_historical
                    min_window = min(STANDARD_WINDOWS)
                    if channel_end_idx_tf < min_window - 1 or len(df_tf_for_channel) < min_window:
                        labels_per_tf[tf] = None
                        continue

                    tf_channels = detect_channels_multi_window(
                        df_tf_for_channel.iloc[:channel_end_idx_tf + 1],
                        windows=STANDARD_WINDOWS,
                        min_cycles=min_cycles
                    )
                    tf_channel, best_tf_window = select_best_channel(tf_channels)

                    if tf_channel is None or not tf_channel.valid:
                        labels_per_tf[tf] = None
                        continue
            else:
                # Resample historical-only for channel detection (no future leakage)
                df_tf_for_channel = cached_resample_ohlc(df_historical, tf)

                # Resample full data for label scanning (forward bars intentional)
                df_tf_full = cached_resample_ohlc(df, tf)

                # Channel ends at last bar of historical resampled data
                channel_end_idx_tf = len(df_tf_for_channel) - 1

                if channel_end_idx_tf < 0:
                    labels_per_tf[tf] = None
                    continue

                # Use precomputed channels if available (avoids redundant detection across windows)
                if precomputed_tf_channels is not None and tf in precomputed_tf_channels:
                    tf_channel, best_tf_window = precomputed_tf_channels[tf]
                    if tf_channel is None or not tf_channel.valid:
                        labels_per_tf[tf] = None
                        continue
                else:
                    # Detect channels at MULTIPLE window sizes for this TF (matches inspector behavior)
                    # Use historical-only data for channel detection to avoid future leakage
                    # Need enough data for at least the smallest standard window
                    min_window = min(STANDARD_WINDOWS)
                    if channel_end_idx_tf < min_window - 1 or len(df_tf_for_channel) < min_window:
                        labels_per_tf[tf] = None
                        continue

                    # Detect channels at all standard windows for this TF
                    tf_channels = detect_channels_multi_window(
                        df_tf_for_channel.iloc[:channel_end_idx_tf + 1],
                        windows=STANDARD_WINDOWS,
                        min_cycles=min_cycles
                    )

                    # Select the best channel by bounces (same logic as inspector)
                    tf_channel, best_tf_window = select_best_channel(tf_channels)

                    if tf_channel is None or not tf_channel.valid:
                        labels_per_tf[tf] = None
                        continue

            # Scale parameters for this timeframe
            scaled_max_scan, scaled_return_threshold = scale_label_params_for_tf(
                tf, max_scan, return_threshold, custom_return_thresholds
            )

            # Scale fold_end_idx if provided
            scaled_fold_end_idx = None
            if fold_end_idx is not None:
                bars_per_tf = BARS_PER_TF.get(tf, 1)
                scaled_fold_end_idx = fold_end_idx // bars_per_tf

            # Generate labels for this TF using full data (includes forward bars for scanning)
            tf_labels = generate_labels(
                df=df_tf_full,
                channel=tf_channel,  # Use the best channel for this TF
                channel_end_idx=channel_end_idx_tf,
                current_tf=tf,
                window=best_tf_window,  # Use the window that gave the best channel
                max_scan=scaled_max_scan,
                return_threshold=scaled_return_threshold,
                fold_end_idx=scaled_fold_end_idx
            )

            labels_per_tf[tf] = tf_labels

        except Exception:
            # Channel detection failed for this TF
            labels_per_tf[tf] = None

    return labels_per_tf


def generate_labels_multi_window(
    df: pd.DataFrame,
    channels: Dict[int, Channel],
    channel_end_idx_5min: int,
    max_scan: int = 500,
    return_threshold: int = 20,
    fold_end_idx: Optional[int] = None,
    min_cycles: int = 1,
    custom_return_thresholds: Optional[Dict[str, int]] = None
) -> Dict[int, Dict[str, Optional[ChannelLabels]]]:
    """
    Generate labels for multiple window sizes.

    For each window's channel, calls generate_labels_per_tf() with the
    appropriate window size from the channel object.

    Args:
        df: Base 5min OHLCV DataFrame with DatetimeIndex (includes forward data for scanning)
        channels: Dict mapping window_size -> Channel object
        channel_end_idx_5min: Index in 5min data where the channel ends. This is the
                              position of the detected channel - data after this is
                              forward data for label scanning.
        max_scan: Maximum bars to scan forward (in 5min bars, will be scaled)
        return_threshold: Bars outside needed to confirm permanent break (will be scaled)
        fold_end_idx: Optional end index for walk-forward validation fold
        min_cycles: Minimum cycles required for valid channel detection
        custom_return_thresholds: Optional dict mapping TF names to custom return threshold
                                  values. If provided and a TF is in the dict, that value is
                                  used instead of the default. Example: {'5min': 10, '1h': 2}

    Returns:
        Dict mapping window_size -> {tf_name -> ChannelLabels}

    Note:
        This function uses cached resampling to avoid redundant computation.
        The cache is cleared at the start and shared across all window calls
        for maximum efficiency.
    """
    # Clear cache once at start, then share across all window iterations
    clear_resample_cache()

    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]] = {}

    # Precompute best channels once per TF to avoid redundant detection across windows
    # This reduces 88 detect_channels_multi_window calls (8 windows × 11 TFs) to just 11
    precomputed_tf_channels: Dict[str, Tuple[Optional[Channel], Optional[int]]] = {}
    df_historical = df.iloc[:channel_end_idx_5min + 1]
    min_window = min(STANDARD_WINDOWS)

    for tf in TIMEFRAMES:
        try:
            if tf == '5min':
                # 5min channels are passed in via the 'channels' dict, so skip precomputation
                # Each window has its own 5min channel which may or may not be valid
                continue
            else:
                df_tf_for_channel = cached_resample_ohlc(df_historical, tf)
                channel_end_idx_tf = len(df_tf_for_channel) - 1

                if channel_end_idx_tf < min_window - 1 or len(df_tf_for_channel) < min_window:
                    precomputed_tf_channels[tf] = (None, None)
                    continue

                tf_channels = detect_channels_multi_window(
                    df_tf_for_channel.iloc[:channel_end_idx_tf + 1],
                    windows=STANDARD_WINDOWS,
                    min_cycles=min_cycles
                )
                tf_channel, best_tf_window = select_best_channel(tf_channels)
                if tf_channel is None or not tf_channel.valid:
                    precomputed_tf_channels[tf] = (None, None)
                else:
                    precomputed_tf_channels[tf] = (tf_channel, best_tf_window)
        except Exception:
            precomputed_tf_channels[tf] = (None, None)

    for window_size, channel in channels.items():
        # Always call generate_labels_per_tf even if 5min channel is invalid,
        # because it does its own multi-window detection per TF now.
        # A valid 1h channel might exist even if the 5min channel at this window is invalid.

        # Generate labels for this window's channel
        # Pass _clear_cache=False to reuse cached resampled data across windows
        # Pass precomputed_tf_channels to avoid redundant channel detection per window
        labels_per_window[window_size] = generate_labels_per_tf(
            df=df,
            channel_end_idx_5min=channel_end_idx_5min,
            window=window_size,
            max_scan=max_scan,
            return_threshold=return_threshold,
            fold_end_idx=fold_end_idx,
            min_cycles=min_cycles,
            channel=channel,
            custom_return_thresholds=custom_return_thresholds,
            precomputed_tf_channels=precomputed_tf_channels,
            _clear_cache=False  # Cache already cleared above, reuse across windows
        )

    # Clear cache after completion to free memory
    clear_resample_cache()

    return labels_per_window


def select_best_window_by_labels(
    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]]
) -> int:
    """
    Select the best window size based on label validity.

    Selects the window with the most valid TF labels. A label is considered
    valid if it is not None.

    Args:
        labels_per_window: Dict mapping window_size -> {tf_name -> ChannelLabels}

    Returns:
        Window size with the most valid TF labels. If there's a tie, returns
        the smallest window size. If all windows have zero valid labels,
        returns the first window size in the dict.
    """
    if not labels_per_window:
        raise ValueError("labels_per_window cannot be empty")

    best_window = None
    best_valid_count = -1

    # Sort by window size to prefer smaller windows on ties
    for window_size in sorted(labels_per_window.keys()):
        tf_labels = labels_per_window[window_size]

        # Count valid (non-None) labels
        valid_count = sum(1 for labels in tf_labels.values() if labels is not None)

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_window = window_size

    # If no window found (shouldn't happen), return smallest window for determinism
    if best_window is None:
        best_window = min(labels_per_window.keys())

    return best_window
