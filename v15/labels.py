"""
Label Generation for Channel Break Prediction (v15)

Simplified label generation that:
1. Projects channel bounds forward using slope/intercept
2. Scans for permanent breaks (price stays outside for return_threshold bars)
3. Returns structured ChannelLabels with duration, direction, etc.

Reuses detect_channel from v7.core.channel.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import IntEnum

# Import from existing v7 modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.core.channel import detect_channel, Channel
from v7.core.timeframe import resample_ohlc, TIMEFRAMES, BARS_PER_TF


# =============================================================================
# Enums and Constants
# =============================================================================

class BreakDirection(IntEnum):
    """Direction of channel break."""
    DOWN = 0
    UP = 1


class NewChannelDirection(IntEnum):
    """Direction of the new channel that forms after break."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


# TF-specific max_scan values (forward bars to scan) - matched to v14
TF_MAX_SCAN = {
    '5min': 500,    # ~40 hours (was 100 in initial v15)
    '15min': 400,   # ~100 hours
    '30min': 350,   # ~175 hours
    '1h': 300,      # ~300 hours
    '2h': 250,      # ~500 hours
    '3h': 200,      # ~600 hours
    '4h': 150,      # ~600 hours
    'daily': 100,   # ~100 trading days
    'weekly': 52,   # ~1 year
    'monthly': 24,  # ~2 years
    '3month': 12,   # ~3 years
}

# TF-specific return_threshold values (bars outside to confirm permanent break)
# Scaled to ~5-10% of max_scan for better detection
TF_RETURN_THRESHOLD = {
    '5min': 25,     # 5% of 500 = 25 bars (~2 hours)
    '15min': 20,    # 5% of 400 = 20 bars (~5 hours)
    '30min': 18,    # 5% of 350 = 18 bars (~9 hours)
    '1h': 15,       # 5% of 300 = 15 bars (~15 hours)
    '2h': 12,       # 5% of 250 = 12 bars (~24 hours)
    '3h': 10,       # 5% of 200 = 10 bars (~30 hours)
    '4h': 8,        # 5% of 150 = 8 bars (~32 hours)
    'daily': 5,     # 5% of 100 = 5 bars (~1 week)
    'weekly': 3,    # ~6% of 52 = 3 bars (~3 weeks)
    'monthly': 2,   # ~8% of 24 = 2 bars (~2 months)
    '3month': 1,    # ~8% of 12 = 1 bar (~3 months)
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChannelLabels:
    """
    Labels for a channel indicating its future outcome.

    Attributes:
        duration_bars: Number of bars until permanent break
        break_direction: Direction of break (0=DOWN, 1=UP)
        new_channel_direction: Direction of next channel (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: Whether a permanent break was found within scan window
        duration_valid: True if duration was observed
        direction_valid: True only if permanent_break=True
    """
    duration_bars: int
    break_direction: int  # 0=DOWN, 1=UP
    new_channel_direction: int  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    permanent_break: bool
    duration_valid: bool = True
    direction_valid: bool = False


# =============================================================================
# Channel Projection
# =============================================================================

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
    # Project from window onwards
    future_x = np.arange(channel.window, channel.window + num_bars)

    # Project center line
    center_projection = channel.slope * future_x + channel.intercept

    # Add/subtract std dev for bounds (2 sigma)
    std_multiplier = 2.0
    upper_projection = center_projection + std_multiplier * channel.std_dev
    lower_projection = center_projection - std_multiplier * channel.std_dev

    return upper_projection, lower_projection


# =============================================================================
# Break Detection
# =============================================================================

def find_permanent_break(
    df_forward: pd.DataFrame,
    upper_projection: np.ndarray,
    lower_projection: np.ndarray,
    return_threshold: int = 20
) -> Tuple[Optional[int], Optional[int]]:
    """
    Scan forward to find a permanent channel break.

    A "permanent break" means price exits AND stays out for return_threshold bars.

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

    # Vectorized boundary checks
    breaks_up = highs > upper
    breaks_down = lows < lower
    is_outside = breaks_up | breaks_down

    if not np.any(is_outside):
        return None, None

    # Track exit state
    exit_bar = None
    exit_direction = None
    bars_outside = 0

    for i in range(n_bars):
        if is_outside[i]:
            if exit_bar is None:
                # New exit - record position and direction
                exit_bar = i
                exit_direction = BreakDirection.UP if breaks_up[i] else BreakDirection.DOWN
                bars_outside = 1
            else:
                bars_outside += 1

            # Check if this is a permanent break
            if bars_outside >= return_threshold:
                return exit_bar, exit_direction
        else:
            # Price returned to channel - false break, reset
            if exit_bar is not None:
                exit_bar = None
                exit_direction = None
                bars_outside = 0

    # Return what we have if exit still pending
    if exit_bar is not None and bars_outside > 0:
        return exit_bar, exit_direction

    return None, None


# =============================================================================
# Main Label Generation Functions
# =============================================================================

def generate_labels_for_tf(
    df: pd.DataFrame,
    channel: Channel,
    channel_end_idx: int,
    tf: str,
    window: int,
    max_scan: int,
    return_threshold: int
) -> ChannelLabels:
    """
    Generate labels for a single channel at a specific timeframe.

    Args:
        df: OHLCV DataFrame at the target timeframe
        channel: Detected channel object
        channel_end_idx: Index where channel ends in df
        tf: Timeframe name (for reference)
        window: Window size used for channel detection
        max_scan: Maximum bars to scan forward
        return_threshold: Bars outside needed to confirm permanent break

    Returns:
        ChannelLabels with duration, direction, etc.
    """
    # Get forward data
    forward_start = channel_end_idx + 1
    forward_end = min(forward_start + max_scan, len(df))

    if forward_start >= len(df):
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=BreakDirection.UP,
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=False,
            direction_valid=False
        )

    df_forward = df.iloc[forward_start:forward_end]
    n_forward = len(df_forward)

    if n_forward == 0:
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=BreakDirection.UP,
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=False,
            direction_valid=False
        )

    # Project channel bounds forward
    upper_proj, lower_proj = project_channel_bounds(channel, n_forward)

    # Find permanent break
    break_idx, break_direction = find_permanent_break(
        df_forward, upper_proj, lower_proj, return_threshold
    )

    if break_idx is None:
        # No break found - channel survived
        return ChannelLabels(
            duration_bars=n_forward,
            break_direction=BreakDirection.UP,  # Default
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=True,
            direction_valid=False
        )

    # Break found - determine new channel direction
    break_absolute_idx = forward_start + break_idx
    new_channel_direction = _detect_new_channel_direction(
        df, break_absolute_idx, return_threshold, window
    )

    return ChannelLabels(
        duration_bars=break_idx,
        break_direction=int(break_direction),
        new_channel_direction=new_channel_direction,
        permanent_break=True,
        duration_valid=True,
        direction_valid=True
    )


def _detect_new_channel_direction(
    df: pd.DataFrame,
    break_idx: int,
    return_threshold: int,
    window: int
) -> int:
    """
    Detect direction of new channel forming after break.

    Args:
        df: Full OHLCV DataFrame
        break_idx: Index where break occurred
        return_threshold: Bars to skip after break
        window: Window for channel detection

    Returns:
        NewChannelDirection value (0=BEAR, 1=SIDEWAYS, 2=BULL)
    """
    start_idx = break_idx + return_threshold
    end_idx = start_idx + window

    if end_idx > len(df):
        return NewChannelDirection.SIDEWAYS

    try:
        df_slice = df.iloc[start_idx:end_idx]
        new_channel = detect_channel(df_slice, window=window)
        if new_channel.valid:
            return int(new_channel.direction)
    except Exception:
        pass

    return NewChannelDirection.SIDEWAYS


def generate_labels_multi_window(
    df: pd.DataFrame,
    channels: Dict[int, Channel],
    channel_end_idx_5min: int,
    config: Optional[Dict] = None
) -> Dict[int, Dict[str, ChannelLabels]]:
    """
    Generate labels for multiple windows across all timeframes.

    For each window in channels.keys():
        For each timeframe in TIMEFRAMES:
            - If tf != '5min': resample and detect channel at that TF
            - Generate labels using TF-specific max_scan and return_threshold

    Args:
        df: Base 5min OHLCV DataFrame with DatetimeIndex
        channels: Dict mapping window_size -> Channel object (5min channels)
        channel_end_idx_5min: Index in 5min data where channels end
        config: Optional config dict with custom max_scan/return_threshold

    Returns:
        Nested dict: {window: {tf: ChannelLabels}}
    """
    # Extract config values if provided
    base_max_scan = config.get('max_scan', 100) if config else 100
    base_return_threshold = config.get('return_threshold', 20) if config else 20
    min_cycles = config.get('min_cycles', 1) if config else 1

    # Pre-compute resampled dataframes for efficiency
    resampled_dfs: Dict[str, pd.DataFrame] = {'5min': df}
    for tf in TIMEFRAMES[1:]:  # Skip 5min
        try:
            resampled_dfs[tf] = resample_ohlc(df, tf)
        except Exception:
            resampled_dfs[tf] = None

    result: Dict[int, Dict[str, ChannelLabels]] = {}

    for window_size, channel_5min in channels.items():
        result[window_size] = {}

        for tf in TIMEFRAMES:
            # Get TF-specific parameters
            max_scan = TF_MAX_SCAN.get(tf, base_max_scan)
            return_threshold = TF_RETURN_THRESHOLD.get(tf, base_return_threshold)

            try:
                if tf == '5min':
                    # Use the provided 5min channel directly
                    if channel_5min is None or not channel_5min.valid:
                        result[window_size][tf] = None
                        continue

                    labels = generate_labels_for_tf(
                        df=df,
                        channel=channel_5min,
                        channel_end_idx=channel_end_idx_5min,
                        tf=tf,
                        window=window_size,
                        max_scan=max_scan,
                        return_threshold=return_threshold
                    )
                    result[window_size][tf] = labels
                else:
                    # Resample and detect channel at this TF
                    df_tf = resampled_dfs.get(tf)
                    if df_tf is None or len(df_tf) < window_size:
                        result[window_size][tf] = None
                        continue

                    # Find equivalent end index in resampled data
                    # Use timestamp from 5min data to find position in TF data
                    if channel_end_idx_5min >= len(df):
                        result[window_size][tf] = None
                        continue

                    end_timestamp = df.index[channel_end_idx_5min]

                    # Find the last TF bar at or before end_timestamp
                    tf_idx = df_tf.index.searchsorted(end_timestamp, side='right') - 1
                    if tf_idx < window_size - 1:
                        result[window_size][tf] = None
                        continue

                    # Detect channel at this TF with the same window
                    df_tf_slice = df_tf.iloc[:tf_idx + 1]
                    tf_channel = detect_channel(
                        df_tf_slice,
                        window=window_size,
                        min_cycles=min_cycles
                    )

                    if not tf_channel.valid:
                        result[window_size][tf] = None
                        continue

                    # Generate labels
                    labels = generate_labels_for_tf(
                        df=df_tf,
                        channel=tf_channel,
                        channel_end_idx=tf_idx,
                        tf=tf,
                        window=window_size,
                        max_scan=max_scan,
                        return_threshold=return_threshold
                    )
                    result[window_size][tf] = labels

            except Exception:
                result[window_size][tf] = None

    return result


# =============================================================================
# Utility Functions
# =============================================================================

def labels_to_dict(labels: ChannelLabels) -> dict:
    """Convert ChannelLabels to dictionary for serialization."""
    return {
        'duration_bars': labels.duration_bars,
        'break_direction': labels.break_direction,
        'new_channel_direction': labels.new_channel_direction,
        'permanent_break': labels.permanent_break,
        'duration_valid': labels.duration_valid,
        'direction_valid': labels.direction_valid,
    }


def labels_to_array(labels: ChannelLabels) -> np.ndarray:
    """Convert ChannelLabels to numpy array for model training."""
    return np.array([
        labels.duration_bars,
        labels.break_direction,
        labels.new_channel_direction,
        int(labels.permanent_break),
        int(labels.duration_valid),
        int(labels.direction_valid),
    ], dtype=np.float32)
