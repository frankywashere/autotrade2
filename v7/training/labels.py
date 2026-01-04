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

from core.channel import (
    detect_channel, detect_channels_multi_window, select_best_channel,
    Channel, Direction, STANDARD_WINDOWS
)
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
            break_trigger_tf=0,  # NO_TRIGGER
            new_channel_direction=NewChannelDirection.SIDEWAYS,
            permanent_break=False,
            duration_valid=False,  # No forward data means duration not observed
            direction_valid=False,
            trigger_tf_valid=False,
            new_channel_valid=False
        )

    df_forward = df.iloc[forward_start:forward_end].copy()
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
    return_threshold: int
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

    Returns:
        Tuple of (scaled_max_scan, scaled_return_threshold)
    """
    bars_per_tf = BARS_PER_TF.get(tf, 1)

    # max_scan per TF - aligned with FORWARD_BARS_PER_TF in label_inspector.py
    tf_max_scan = {
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
    scaled_max_scan = tf_max_scan.get(tf, min(max_scan, 50))

    # Scale return_threshold to keep consistent percentage behavior
    scaled_return_threshold = max(1, return_threshold // bars_per_tf)

    return scaled_max_scan, scaled_return_threshold


def generate_labels_per_tf(
    df: pd.DataFrame,
    channel_end_idx_5min: int,
    window: int = 50,
    max_scan: int = 500,
    return_threshold: int = 20,
    fold_end_idx: Optional[int] = None,
    min_cycles: int = 1,
    channel: Optional[Channel] = None
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

    Returns:
        Dict mapping TF name to ChannelLabels (None if channel detection failed)
    """
    # If a channel is provided, use its window size for consistency
    if channel is not None:
        window = channel.window
    labels_per_tf: Dict[str, Optional[ChannelLabels]] = {}

    # Get the timestamp at the channel end position for TF alignment
    if channel_end_idx_5min >= len(df):
        # Invalid index
        return {tf: None for tf in TIMEFRAMES}

    channel_end_timestamp = df.index[channel_end_idx_5min]

    # Split data: historical (up to sample time) vs full (includes forward bars)
    # This prevents future data leakage in channel detection for longer timeframes
    df_historical = df.iloc[:channel_end_idx_5min + 1]  # Only up to sample time

    for tf in TIMEFRAMES:
        try:
            # Resample data to this timeframe
            # Use separate dataframes for channel detection (historical) vs label scanning (full)
            if tf == '5min':
                df_tf_for_channel = df_historical
                df_tf_full = df
                channel_end_idx_tf = channel_end_idx_5min
            else:
                # Resample historical-only for channel detection (no future leakage)
                df_tf_for_channel = resample_ohlc(df_historical, tf)

                # Resample full data for label scanning (forward bars intentional)
                df_tf_full = resample_ohlc(df, tf)

                # Channel ends at last bar of historical resampled data
                channel_end_idx_tf = len(df_tf_for_channel) - 1

                if channel_end_idx_tf < 0:
                    labels_per_tf[tf] = None
                    continue

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
                tf, max_scan, return_threshold
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
    min_cycles: int = 1
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

    Returns:
        Dict mapping window_size -> {tf_name -> ChannelLabels}
    """
    labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]] = {}

    for window_size, channel in channels.items():
        # Always call generate_labels_per_tf even if 5min channel is invalid,
        # because it does its own multi-window detection per TF now.
        # A valid 1h channel might exist even if the 5min channel at this window is invalid.

        # Generate labels for this window's channel
        labels_per_window[window_size] = generate_labels_per_tf(
            df=df,
            channel_end_idx_5min=channel_end_idx_5min,
            window=window_size,
            max_scan=max_scan,
            return_threshold=return_threshold,
            fold_end_idx=fold_end_idx,
            min_cycles=min_cycles,
            channel=channel
        )

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

    # If no window found (shouldn't happen), return first
    if best_window is None:
        best_window = next(iter(labels_per_window.keys()))

    return best_window
