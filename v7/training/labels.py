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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel, Direction
from core.timeframe import resample_ohlc, get_longer_timeframes, TIMEFRAMES, BARS_PER_TF
from features.containment import check_containment, get_closest_boundary, ContainmentInfo


class BreakDirection(IntEnum):
    """Direction of channel break."""
    DOWN = 0
    UP = 1


class NewChannelDirection(IntEnum):
    """Direction of the new channel that forms after break."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


@dataclass
class ChannelLabels:
    """
    Labels for a channel indicating its future outcome.

    Attributes:
        duration_bars: Number of bars until permanent break
        break_direction: Direction of break (0=DOWN, 1=UP)
        break_trigger_tf: Which longer TF boundary was nearest at break time
        new_channel_direction: Direction of next channel (0=BEAR, 1=SIDEWAYS, 2=BULL)
        permanent_break: Whether a permanent break was found within scan window
    """
    duration_bars: int
    break_direction: int  # 0=DOWN, 1=UP
    break_trigger_tf: Optional[str]  # e.g., "1h_upper", "daily_lower"
    new_channel_direction: int  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    permanent_break: bool


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

    # Track exit state
    exit_bar = None
    exit_direction = None
    bars_outside = 0

    for i in range(n_bars):
        is_inside, direction = check_price_in_channel(
            highs[i], lows[i],
            upper_projection[i], lower_projection[i]
        )

        if not is_inside:
            # Price exited channel
            if exit_bar is None:
                exit_bar = i
                exit_direction = direction
                bars_outside = 1
            else:
                bars_outside += 1

            # Check if this is a permanent break
            if bars_outside >= return_threshold:
                return exit_bar, exit_direction
        else:
            # Price returned to channel
            if exit_bar is not None:
                # False break - reset tracking
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

    Args:
        df: Full DataFrame (break point onwards)
        start_idx: Index to start scanning from
        window: Window size for channel detection
        max_scan: Maximum bars to scan looking for new channel

    Returns:
        Channel object if found, None otherwise
    """
    for i in range(start_idx, min(start_idx + max_scan, len(df))):
        if i + window > len(df):
            break

        # Try to detect channel at this point
        df_slice = df.iloc[i:i + window]
        if len(df_slice) < window:
            continue

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
        df_tf = resample_ohlc(df, tf)
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
            break_trigger_tf=None,
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False
        )

    df_forward = df.iloc[forward_start:forward_end].copy()
    n_forward = len(df_forward)

    if n_forward == 0:
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=BreakDirection.UP,
            break_trigger_tf=None,
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False
        )

    # Project channel bounds forward
    upper_proj, lower_proj = project_channel_bounds(channel, n_forward)

    # Find permanent break
    break_idx, break_direction = find_permanent_break(
        df_forward, upper_proj, lower_proj, return_threshold
    )

    if break_idx is None:
        # No break found within scan window
        return ChannelLabels(
            duration_bars=n_forward,
            break_direction=BreakDirection.UP,  # Default
            break_trigger_tf=None,
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False
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
        break_trigger_tf=break_trigger_tf,
        new_channel_direction=new_channel_direction,
        permanent_break=True
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
        Dictionary with all label fields
    """
    return {
        'duration_bars': labels.duration_bars,
        'break_direction': labels.break_direction,
        'break_trigger_tf': labels.break_trigger_tf,
        'new_channel_direction': labels.new_channel_direction,
        'permanent_break': labels.permanent_break
    }


def labels_to_array(labels: ChannelLabels, tf_encoding: dict = None) -> np.ndarray:
    """
    Convert ChannelLabels to numpy array for model training.

    Args:
        labels: ChannelLabels object
        tf_encoding: Optional dict mapping TF strings to integers

    Returns:
        Numpy array: [duration, break_dir, trigger_tf_encoded, new_dir, permanent]
    """
    if tf_encoding is None:
        # Default encoding for timeframe strings
        tf_encoding = {
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

    trigger_encoded = tf_encoding.get(labels.break_trigger_tf, 0)

    return np.array([
        labels.duration_bars,
        labels.break_direction,
        trigger_encoded,
        labels.new_channel_direction,
        int(labels.permanent_break)
    ], dtype=np.float32)


def scale_label_params_for_tf(
    tf: str,
    max_scan: int,
    return_threshold: int
) -> Tuple[int, int]:
    """
    Scale label generation parameters for a specific timeframe.

    Keeps the same time horizon by dividing by the number of base bars per TF bar.

    Args:
        tf: Target timeframe (e.g., '15min', '1h', 'daily')
        max_scan: Base max_scan value (in 5min bars)
        return_threshold: Base return_threshold value (in 5min bars)

    Returns:
        Tuple of (scaled_max_scan, scaled_return_threshold)
    """
    bars_per_tf = BARS_PER_TF.get(tf, 1)

    scaled_max_scan = max_scan // bars_per_tf
    scaled_return_threshold = max(1, return_threshold // bars_per_tf)

    return scaled_max_scan, scaled_return_threshold


def generate_labels_per_tf(
    df: pd.DataFrame,
    window: int = 50,
    max_scan: int = 500,
    return_threshold: int = 20,
    fold_end_idx: Optional[int] = None
) -> Dict[str, Optional[ChannelLabels]]:
    """
    Generate labels for each timeframe by resampling and detecting channels.

    For each TF in TIMEFRAMES:
    1. Resamples base 5min data using resample_ohlc()
    2. Detects a channel at that TF
    3. Calls generate_labels() with scaled parameters

    Args:
        df: Base 5min OHLCV DataFrame with DatetimeIndex
        window: Window size for channel detection
        max_scan: Maximum bars to scan forward (in 5min bars, will be scaled)
        return_threshold: Bars outside needed to confirm permanent break (will be scaled)
        fold_end_idx: Optional end index for walk-forward validation fold

    Returns:
        Dict mapping TF name to ChannelLabels (None if channel detection failed)
    """
    labels_per_tf: Dict[str, Optional[ChannelLabels]] = {}

    for tf in TIMEFRAMES:
        try:
            # Resample data to this timeframe
            if tf == '5min':
                df_tf = df
            else:
                df_tf = resample_ohlc(df, tf)

            # Need enough data for channel detection
            if len(df_tf) < window:
                labels_per_tf[tf] = None
                continue

            # Detect channel at this timeframe
            # Use the most recent window of data for channel detection
            channel_start_idx = len(df_tf) - window
            df_channel = df_tf.iloc[channel_start_idx:channel_start_idx + window]
            channel = detect_channel(df_channel, window=window)

            if not channel.valid:
                labels_per_tf[tf] = None
                continue

            # Scale parameters for this timeframe
            scaled_max_scan, scaled_return_threshold = scale_label_params_for_tf(
                tf, max_scan, return_threshold
            )

            # Scale fold_end_idx if provided
            scaled_fold_end_idx = None
            if fold_end_idx is not None:
                bars_per_tf = BARS_PER_TF.get(tf, 1)
                scaled_fold_end_idx = fold_end_idx // bars_per_tf

            # Channel ends at the last bar of the detection window
            channel_end_idx = len(df_tf) - 1

            # Generate labels for this TF
            tf_labels = generate_labels(
                df=df_tf,
                channel=channel,
                channel_end_idx=channel_end_idx,
                current_tf=tf,
                window=window,
                max_scan=scaled_max_scan,
                return_threshold=scaled_return_threshold,
                fold_end_idx=scaled_fold_end_idx
            )

            labels_per_tf[tf] = tf_labels

        except Exception:
            # Channel detection failed for this TF
            labels_per_tf[tf] = None

    return labels_per_tf
