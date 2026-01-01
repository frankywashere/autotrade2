"""
Exit Tracking Module

Tracks exit/return behavior within channels:
- Exit events: when price crosses upper/lower bounds
- Return events: when price comes back inside
- Time spent outside before returning
- Behavior after returning (bounces)
- Exit frequency and acceleration
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import Channel


class ExitDirection(IntEnum):
    """Direction of channel exit."""
    DOWN = 0
    UP = 1


@dataclass
class ExitEvent:
    """
    Record of a single exit from the channel.

    Attributes:
        bar_index: Index of the bar where exit occurred
        exit_direction: Direction of exit (UP or DOWN)
        bars_outside: Number of bars spent outside before returning (or until end)
        did_return: Whether price returned inside the channel
        return_bar: Bar index where price returned (None if didn't return)
    """
    bar_index: int
    exit_direction: ExitDirection
    bars_outside: int
    did_return: bool
    return_bar: Optional[int] = None


@dataclass
class ExitTrackingFeatures:
    """
    Features derived from exit/return behavior within a channel.

    Attributes:
        exit_count: Total number of exit events
        avg_bars_outside: Average bars spent outside before returning
        max_bars_outside: Maximum bars spent outside before returning
        exit_frequency: Exits per 100 bars
        exits_accelerating: Whether exits are happening more frequently recently
        exits_up_count: Number of exits through upper bound
        exits_down_count: Number of exits through lower bound
        avg_return_speed: Average 1/bars_outside for returned exits (higher = faster)
        return_speed_slowing: Whether returns are taking longer recently
        bounces_after_last_return: Number of bounces since last return event
    """
    exit_count: int
    avg_bars_outside: float
    max_bars_outside: int
    exit_frequency: float
    exits_accelerating: bool
    exits_up_count: int
    exits_down_count: int
    avg_return_speed: float
    return_speed_slowing: bool
    bounces_after_last_return: int


def _project_channel_bounds(
    channel: Channel,
    num_bars_forward: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project channel bounds forward from the last known bar.

    Args:
        channel: Channel object with bounds
        num_bars_forward: Number of bars to project forward

    Returns:
        Tuple of (upper_line, lower_line) arrays of length (window + num_bars_forward)
    """
    window = channel.window

    # Existing bounds
    upper = channel.upper_line
    lower = channel.lower_line

    if num_bars_forward <= 0:
        return upper, lower

    # Project forward using slope
    x_start = window
    x_forward = np.arange(x_start, x_start + num_bars_forward)

    # Calculate forward projections
    center_forward = channel.slope * x_forward + channel.intercept
    upper_forward = center_forward + 2 * channel.std_dev
    lower_forward = center_forward - 2 * channel.std_dev

    # Concatenate existing and projected
    upper_full = np.concatenate([upper, upper_forward])
    lower_full = np.concatenate([lower, lower_forward])

    return upper_full, lower_full


def _is_outside_channel(
    high: float,
    low: float,
    upper: float,
    lower: float
) -> Tuple[bool, Optional[ExitDirection]]:
    """
    Check if a bar is outside the channel.

    Args:
        high: Bar high price
        low: Bar low price
        upper: Upper channel bound at this bar
        lower: Lower channel bound at this bar

    Returns:
        Tuple of (is_outside, exit_direction)
        exit_direction is None if not outside
    """
    if high > upper:
        return True, ExitDirection.UP
    elif low < lower:
        return True, ExitDirection.DOWN
    return False, None


def _is_inside_channel(
    high: float,
    low: float,
    upper: float,
    lower: float
) -> bool:
    """
    Check if a bar is fully inside the channel.

    A bar is inside if both high and low are within bounds.

    Args:
        high: Bar high price
        low: Bar low price
        upper: Upper channel bound at this bar
        lower: Lower channel bound at this bar

    Returns:
        True if bar is fully inside channel
    """
    return high <= upper and low >= lower


def track_exits_in_channel(
    df: pd.DataFrame,
    channel: Channel,
    lookforward: int = 200
) -> ExitTrackingFeatures:
    """
    Track exit/return behavior within a channel.

    Scans through bars looking for when price crosses outside the channel
    bounds, then tracks when/if it returns inside.

    Args:
        df: OHLCV DataFrame (should contain data for the channel window
            plus lookforward bars if available)
        channel: Detected Channel object
        lookforward: Number of bars to look ahead from channel end

    Returns:
        ExitTrackingFeatures with all calculated metrics
    """
    if not channel.valid or channel.window < 10:
        return _empty_features()

    window = channel.window

    # Get price data - use channel's stored data plus any additional bars in df
    # Channel data covers bars [len(df) - window : len(df)]
    # We want to look at the channel window and then lookforward bars beyond

    df_start_idx = len(df) - window
    available_forward = len(df) - (df_start_idx + window)
    actual_lookforward = min(lookforward, available_forward) if available_forward > 0 else 0
    total_bars = window + actual_lookforward

    # Get price data for full range
    df_slice = df.iloc[df_start_idx : df_start_idx + total_bars]
    highs = df_slice['high'].values.astype(np.float64)
    lows = df_slice['low'].values.astype(np.float64)

    # Get/project channel bounds
    upper, lower = _project_channel_bounds(channel, actual_lookforward)

    # Ensure arrays are same length
    n_bars = min(len(highs), len(upper), len(lows), len(lower))
    highs = highs[:n_bars]
    lows = lows[:n_bars]
    upper = upper[:n_bars]
    lower = lower[:n_bars]

    # Scan for exits and returns
    exit_events: List[ExitEvent] = []
    i = 0

    while i < n_bars:
        is_out, direction = _is_outside_channel(highs[i], lows[i], upper[i], lower[i])

        if is_out and direction is not None:
            # Found an exit, now look for return
            exit_bar = i
            return_bar = None
            bars_outside = 0
            did_return = False

            # Scan forward to find return
            j = i + 1
            while j < n_bars:
                bars_outside += 1

                # Check if fully back inside
                if _is_inside_channel(highs[j], lows[j], upper[j], lower[j]):
                    did_return = True
                    return_bar = j
                    break
                j += 1

            # If we hit end without returning, count remaining bars as outside
            if not did_return:
                bars_outside = n_bars - exit_bar - 1

            exit_events.append(ExitEvent(
                bar_index=exit_bar,
                exit_direction=direction,
                bars_outside=max(bars_outside, 1),
                did_return=did_return,
                return_bar=return_bar,
            ))

            # Skip to after return (or continue from last checked bar)
            i = return_bar + 1 if return_bar is not None else j
        else:
            i += 1

    # Calculate features from exit events
    return _calculate_features(exit_events, channel, n_bars)


def _empty_features() -> ExitTrackingFeatures:
    """Return empty/default features when no data available."""
    return ExitTrackingFeatures(
        exit_count=0,
        avg_bars_outside=0.0,
        max_bars_outside=0,
        exit_frequency=0.0,
        exits_accelerating=False,
        exits_up_count=0,
        exits_down_count=0,
        avg_return_speed=0.0,
        return_speed_slowing=False,
        bounces_after_last_return=0,
    )


def _calculate_features(
    exit_events: List[ExitEvent],
    channel: Channel,
    n_bars: int
) -> ExitTrackingFeatures:
    """
    Calculate all exit tracking features from detected exit events.

    Args:
        exit_events: List of detected ExitEvent objects
        channel: Original channel for bounce info
        n_bars: Total number of bars analyzed

    Returns:
        ExitTrackingFeatures with all metrics
    """
    if not exit_events:
        # No exits detected
        return ExitTrackingFeatures(
            exit_count=0,
            avg_bars_outside=0.0,
            max_bars_outside=0,
            exit_frequency=0.0,
            exits_accelerating=False,
            exits_up_count=0,
            exits_down_count=0,
            avg_return_speed=0.0,
            return_speed_slowing=False,
            bounces_after_last_return=channel.bounce_count,
        )

    exit_count = len(exit_events)

    # Bars outside statistics
    bars_outside_list = [e.bars_outside for e in exit_events]
    avg_bars_outside = float(np.mean(bars_outside_list))
    max_bars_outside = int(np.max(bars_outside_list))

    # Exit frequency (per 100 bars)
    exit_frequency = (exit_count / n_bars) * 100 if n_bars > 0 else 0.0

    # Direction counts
    exits_up_count = sum(1 for e in exit_events if e.exit_direction == ExitDirection.UP)
    exits_down_count = sum(1 for e in exit_events if e.exit_direction == ExitDirection.DOWN)

    # Check if exits are accelerating (more exits in second half)
    exits_accelerating = False
    if exit_count >= 2:
        mid_bar = n_bars // 2
        first_half_exits = sum(1 for e in exit_events if e.bar_index < mid_bar)
        second_half_exits = sum(1 for e in exit_events if e.bar_index >= mid_bar)
        exits_accelerating = second_half_exits > first_half_exits

    # Return speed (inverse of bars_outside for returned exits)
    # Note: bars_outside is guaranteed >= 1 at creation (line 249), but we use max() defensively
    returned_events = [e for e in exit_events if e.did_return]
    if returned_events:
        return_speeds = [1.0 / max(e.bars_outside, 1) for e in returned_events]
        avg_return_speed = float(np.mean(return_speeds))

        # Check if returns are slowing (compare first half to second half)
        if len(returned_events) >= 2:
            mid = len(returned_events) // 2
            first_half_speed = np.mean([1.0 / max(e.bars_outside, 1) for e in returned_events[:mid]])
            second_half_speed = np.mean([1.0 / max(e.bars_outside, 1) for e in returned_events[mid:]])
            return_speed_slowing = second_half_speed < first_half_speed
        else:
            return_speed_slowing = False
    else:
        avg_return_speed = 0.0
        return_speed_slowing = False

    # Count bounces after last return
    bounces_after_last_return = 0
    last_return_bar = None
    for e in reversed(exit_events):
        if e.did_return and e.return_bar is not None:
            last_return_bar = e.return_bar
            break

    if last_return_bar is not None:
        # Count channel touches after the return bar
        for touch in channel.touches:
            if touch.bar_index > last_return_bar:
                bounces_after_last_return += 1
    else:
        # No returns, so all bounces count
        bounces_after_last_return = channel.bounce_count

    return ExitTrackingFeatures(
        exit_count=exit_count,
        avg_bars_outside=avg_bars_outside,
        max_bars_outside=max_bars_outside,
        exit_frequency=exit_frequency,
        exits_accelerating=exits_accelerating,
        exits_up_count=exits_up_count,
        exits_down_count=exits_down_count,
        avg_return_speed=avg_return_speed,
        return_speed_slowing=return_speed_slowing,
        bounces_after_last_return=bounces_after_last_return,
    )


def features_to_dict(features: ExitTrackingFeatures) -> dict:
    """
    Convert ExitTrackingFeatures to a flat dictionary.

    Useful for model input or logging.

    Args:
        features: ExitTrackingFeatures object

    Returns:
        Dictionary with all feature values
    """
    return {
        'exit_count': features.exit_count,
        'avg_bars_outside': features.avg_bars_outside,
        'max_bars_outside': features.max_bars_outside,
        'exit_frequency': features.exit_frequency,
        'exits_accelerating': int(features.exits_accelerating),
        'exits_up_count': features.exits_up_count,
        'exits_down_count': features.exits_down_count,
        'avg_return_speed': features.avg_return_speed,
        'return_speed_slowing': int(features.return_speed_slowing),
        'bounces_after_last_return': features.bounces_after_last_return,
    }
