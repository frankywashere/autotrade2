"""
Channel History Features

Tracks patterns of past channels:
- Last N channel directions
- Last N channel durations
- How each channel ended (break direction)
- RSI at each bounce point
- Patterns like "3 bear channels then sideways"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel, Direction, TouchType
from core.timeframe import resample_ohlc, TIMEFRAMES
from features.rsi import calculate_rsi, calculate_rsi_series


@dataclass
class BounceRecord:
    """Record of a single bounce with RSI and VIX context."""
    bar_index: int
    touch_type: int          # 0=lower, 1=upper
    price: float
    rsi_at_bounce: float     # RSI value when this bounce occurred
    channel_position: float  # Position in channel at bounce (0-1)
    # VIX context at bounce time
    absolute_bar_index: int = 0      # Global bar index in dataset
    timestamp: Optional[pd.Timestamp] = None  # Timestamp of bounce
    vix_at_bounce: float = 20.0      # VIX level when bounce occurred
    vix_regime_at_bounce: int = 1    # 0=low, 1=normal, 2=high, 3=extreme


@dataclass
class ChannelRecord:
    """Record of a completed channel."""
    start_idx: int
    end_idx: int
    duration_bars: int
    direction: int           # 0=bear, 1=sideways, 2=bull
    break_direction: int     # 0=down, 1=up
    bounce_count: int
    complete_cycles: int
    avg_rsi: float           # Average RSI during channel
    rsi_at_start: float      # RSI when channel started
    rsi_at_break: float      # RSI when channel broke
    bounces: List[BounceRecord] = field(default_factory=list)


@dataclass
class ChannelHistoryFeatures:
    """Features derived from channel history."""
    # Recent channel stats
    last_n_directions: List[int]       # Last N channel directions
    last_n_durations: List[int]        # Last N channel durations (bars)
    last_n_break_dirs: List[int]       # Last N break directions (0=down, 1=up)

    # Aggregates
    avg_duration: float                # Average of last N durations
    direction_streak: int              # Consecutive same-direction channels
    bear_count_last_5: int             # How many bear channels in last 5
    bull_count_last_5: int             # How many bull channels in last 5
    sideways_count_last_5: int         # How many sideways in last 5

    # RSI patterns
    avg_rsi_at_upper_bounce: float     # Avg RSI when hitting upper bounds
    avg_rsi_at_lower_bounce: float     # Avg RSI when hitting lower bounds
    rsi_at_last_break: float           # RSI at most recent channel break

    # Break patterns
    break_up_after_bear_pct: float     # % of bear channels that broke up
    break_down_after_bull_pct: float   # % of bull channels that broke down


def _get_vix_at_timestamp(vix_df: Optional[pd.DataFrame], timestamp: pd.Timestamp) -> Tuple[float, int]:
    """
    Get VIX level and regime at a specific timestamp.

    Args:
        vix_df: VIX daily data with DatetimeIndex
        timestamp: Timestamp to lookup

    Returns:
        Tuple of (vix_level, vix_regime)
    """
    if vix_df is None or len(vix_df) == 0:
        return 20.0, 1  # Default: normal regime

    # Get the date from timestamp
    target_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp

    # Find the closest VIX value at or before this timestamp
    try:
        if hasattr(vix_df.index, 'date'):
            # DatetimeIndex - find closest date
            mask = vix_df.index.date <= target_date
            if mask.any():
                vix_level = float(vix_df.loc[mask, 'close'].iloc[-1])
            else:
                vix_level = float(vix_df['close'].iloc[0])
        else:
            # Use iloc for fallback
            vix_level = float(vix_df['close'].iloc[-1])
    except (KeyError, IndexError):
        vix_level = 20.0

    # Determine regime
    if vix_level < 15:
        regime = 0  # Low volatility
    elif vix_level < 25:
        regime = 1  # Normal
    elif vix_level < 35:
        regime = 2  # High
    else:
        regime = 3  # Extreme

    return vix_level, regime


def detect_bounces_with_rsi(
    df: pd.DataFrame,
    channel: Channel,
    rsi_series: np.ndarray,
    vix_df: Optional[pd.DataFrame] = None,
    base_bar_index: int = 0
) -> List[BounceRecord]:
    """
    Detect bounces and record RSI and VIX at each bounce.

    Args:
        df: OHLCV data used for channel
        channel: Detected channel
        rsi_series: RSI values for each bar
        vix_df: Optional VIX daily data for VIX context at bounces
        base_bar_index: Base index for calculating absolute bar positions

    Returns:
        List of BounceRecord objects with RSI and VIX context
    """
    bounces = []

    for touch in channel.touches:
        bar_idx = touch.bar_index

        # Get RSI at this bar (adjust for any offset)
        rsi_idx = len(df) - channel.window + bar_idx
        if 0 <= rsi_idx < len(rsi_series):
            rsi_at_bounce = rsi_series[rsi_idx]
        else:
            rsi_at_bounce = 50.0

        # Channel position at this bar
        close = channel.close[bar_idx]
        upper = channel.upper_line[bar_idx]
        lower = channel.lower_line[bar_idx]
        position = (close - lower) / (upper - lower) if upper != lower else 0.5

        # Get absolute bar index and timestamp
        absolute_idx = base_bar_index + rsi_idx
        timestamp = None
        if hasattr(df, 'index') and 0 <= rsi_idx < len(df):
            try:
                timestamp = df.index[rsi_idx]
            except (IndexError, KeyError):
                pass

        # Get VIX at bounce time
        vix_at_bounce = 20.0
        vix_regime = 1
        if timestamp is not None and vix_df is not None:
            vix_at_bounce, vix_regime = _get_vix_at_timestamp(vix_df, timestamp)

        bounces.append(BounceRecord(
            bar_index=bar_idx,
            touch_type=int(touch.touch_type),
            price=touch.price,
            rsi_at_bounce=float(rsi_at_bounce),
            channel_position=float(np.clip(position, 0, 1)),
            absolute_bar_index=absolute_idx,
            timestamp=timestamp,
            vix_at_bounce=vix_at_bounce,
            vix_regime_at_bounce=vix_regime,
        ))

    return bounces


def scan_channel_history(
    df: pd.DataFrame,
    window: int = 20,
    max_channels: int = 10,
    scan_bars: int = 5000,
    vix_df: Optional[pd.DataFrame] = None
) -> List[ChannelRecord]:
    """
    Scan historical data to find past channels.

    This is a simplified scanner that looks for channel breaks
    and records the channels that existed before each break.

    Args:
        df: OHLCV data
        window: Window size for channel detection
        max_channels: Maximum channels to return
        scan_bars: How many bars back to scan
        vix_df: Optional VIX daily data for VIX context at bounces

    Returns:
        List of ChannelRecord objects (most recent first)
    """
    channels = []
    rsi_series = calculate_rsi_series(df['close'].values, period=14)

    # Start from recent and go back
    end_idx = len(df)
    start_scan = max(window + 100, len(df) - scan_bars)

    current_idx = end_idx
    last_channel_end = end_idx

    while current_idx > start_scan and len(channels) < max_channels:
        # Look for where channel breaks
        df_slice = df.iloc[:current_idx]

        if len(df_slice) < window:
            break

        channel = detect_channel(df_slice, window=window)

        if channel.valid:
            # Scan forward to find where this channel broke
            break_idx = current_idx
            break_direction = 1  # Default up

            for future_idx in range(current_idx, min(current_idx + 200, len(df))):
                future_price = df['close'].iloc[future_idx]
                # Project channel forward
                x_future = window + (future_idx - current_idx)
                upper_proj = channel.slope * x_future + channel.intercept + 2 * channel.std_dev
                lower_proj = channel.slope * x_future + channel.intercept - 2 * channel.std_dev

                if future_price > upper_proj:
                    break_idx = future_idx
                    break_direction = 1  # Broke up
                    break
                elif future_price < lower_proj:
                    break_idx = future_idx
                    break_direction = 0  # Broke down
                    break

            # Record this channel
            rsi_start_idx = current_idx - window
            rsi_end_idx = break_idx

            bounces = detect_bounces_with_rsi(df_slice, channel, rsi_series, vix_df=vix_df)

            # Get RSI values
            rsi_at_start = rsi_series[rsi_start_idx] if rsi_start_idx >= 0 else 50.0
            rsi_at_break = rsi_series[min(rsi_end_idx, len(rsi_series) - 1)]
            avg_rsi = np.mean(rsi_series[max(0, rsi_start_idx):rsi_end_idx]) if rsi_end_idx > rsi_start_idx else 50.0

            channels.append(ChannelRecord(
                start_idx=current_idx - window,
                end_idx=break_idx,
                duration_bars=break_idx - (current_idx - window),
                direction=int(channel.direction),
                break_direction=break_direction,
                bounce_count=channel.bounce_count,
                complete_cycles=channel.complete_cycles,
                avg_rsi=float(avg_rsi),
                rsi_at_start=float(rsi_at_start),
                rsi_at_break=float(rsi_at_break),
                bounces=bounces,
            ))

            # Move back past this channel
            current_idx = current_idx - window - 10
        else:
            # No valid channel, step back
            current_idx -= 10

    return channels


def extract_history_features(
    channel_records: List[ChannelRecord],
    n_recent: int = 5
) -> ChannelHistoryFeatures:
    """
    Extract features from channel history.

    Args:
        channel_records: List of past channels (most recent first)
        n_recent: How many recent channels to use

    Returns:
        ChannelHistoryFeatures object
    """
    if not channel_records:
        return ChannelHistoryFeatures(
            last_n_directions=[1] * n_recent,  # sideways
            last_n_durations=[50] * n_recent,
            last_n_break_dirs=[1] * n_recent,
            avg_duration=50.0,
            direction_streak=0,
            bear_count_last_5=0,
            bull_count_last_5=0,
            sideways_count_last_5=0,
            avg_rsi_at_upper_bounce=50.0,
            avg_rsi_at_lower_bounce=50.0,
            rsi_at_last_break=50.0,
            break_up_after_bear_pct=0.5,
            break_down_after_bull_pct=0.5,
        )

    recent = channel_records[:n_recent]

    # Basic lists
    directions = [c.direction for c in recent]
    durations = [c.duration_bars for c in recent]
    break_dirs = [c.break_direction for c in recent]

    # Pad if not enough
    while len(directions) < n_recent:
        directions.append(1)  # sideways
        durations.append(50)
        break_dirs.append(1)

    # Direction streak (consecutive same direction)
    streak = 1
    if len(channel_records) >= 2:
        first_dir = channel_records[0].direction
        for c in channel_records[1:]:
            if c.direction == first_dir:
                streak += 1
            else:
                break

    # Counts
    dirs_5 = [c.direction for c in recent[:5]]
    bear_count = sum(1 for d in dirs_5 if d == 0)
    bull_count = sum(1 for d in dirs_5 if d == 2)
    sideways_count = sum(1 for d in dirs_5 if d == 1)

    # RSI at bounces
    upper_rsis = []
    lower_rsis = []
    for c in recent:
        for b in c.bounces:
            if b.touch_type == 1:  # upper
                upper_rsis.append(b.rsi_at_bounce)
            else:  # lower
                lower_rsis.append(b.rsi_at_bounce)

    avg_rsi_upper = np.mean(upper_rsis) if upper_rsis else 50.0
    avg_rsi_lower = np.mean(lower_rsis) if lower_rsis else 50.0

    # Break patterns
    bear_channels = [c for c in channel_records if c.direction == 0]
    bull_channels = [c for c in channel_records if c.direction == 2]

    break_up_after_bear = 0.5
    if bear_channels:
        break_up_after_bear = sum(1 for c in bear_channels if c.break_direction == 1) / len(bear_channels)

    break_down_after_bull = 0.5
    if bull_channels:
        break_down_after_bull = sum(1 for c in bull_channels if c.break_direction == 0) / len(bull_channels)

    return ChannelHistoryFeatures(
        last_n_directions=directions[:n_recent],
        last_n_durations=durations[:n_recent],
        last_n_break_dirs=break_dirs[:n_recent],
        avg_duration=float(np.mean(durations)),
        direction_streak=streak,
        bear_count_last_5=bear_count,
        bull_count_last_5=bull_count,
        sideways_count_last_5=sideways_count,
        avg_rsi_at_upper_bounce=float(avg_rsi_upper),
        avg_rsi_at_lower_bounce=float(avg_rsi_lower),
        rsi_at_last_break=float(recent[0].rsi_at_break) if recent else 50.0,
        break_up_after_bear_pct=float(break_up_after_bear),
        break_down_after_bull_pct=float(break_down_after_bull),
    )
