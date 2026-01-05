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
        return_rate: Percentage of exits that resulted in returns (0-1)
        channel_resilience_score: Composite score rewarding more returns, faster returns (0-1)
        avg_duration_after_return: Average bars channel remained valid after returns
        max_duration_after_return: Maximum bars channel lasted after a return
        returns_leading_to_new_channel: Count of returns that led to channel reformation (>20 bars inside)
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
    return_rate: float
    channel_resilience_score: float
    avg_duration_after_return: float
    max_duration_after_return: int
    returns_leading_to_new_channel: int


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

    # Vectorized computation of inside/outside status for all bars
    # A bar is "outside up" if high > upper, "outside down" if low < lower
    # A bar is "inside" if high <= upper AND low >= lower
    outside_up = highs > upper
    outside_down = lows < lower
    is_outside = outside_up | outside_down
    is_inside = (highs <= upper) & (lows >= lower)

    # Find exit transitions: bar where we go from inside (or start) to outside
    # Prepend True to represent "inside" state before first bar
    was_inside_or_start = np.concatenate([[True], is_inside[:-1]])
    exit_mask = was_inside_or_start & is_outside
    exit_indices = np.where(exit_mask)[0]

    # For each exit, determine direction (UP takes priority if both conditions met)
    # and find the next return (first bar that is fully inside after the exit)
    exit_events: List[ExitEvent] = []

    # Pre-compute indices where bars are inside for fast return lookup
    inside_indices = np.where(is_inside)[0]

    # Process exits - skip exits that occur before a previous exit's return
    skip_until = -1
    for exit_bar in exit_indices:
        if exit_bar <= skip_until:
            continue

        # Determine exit direction (UP takes priority, matching original logic)
        if outside_up[exit_bar]:
            direction = ExitDirection.UP
        else:
            direction = ExitDirection.DOWN

        # Find next return: first inside bar after exit_bar
        # Use searchsorted for O(log n) lookup instead of linear scan
        search_pos = np.searchsorted(inside_indices, exit_bar, side='right')

        if search_pos < len(inside_indices):
            return_bar = int(inside_indices[search_pos])
            did_return = True
            bars_outside = return_bar - exit_bar
            skip_until = return_bar  # Skip to after return
        else:
            # No return found - this is the last exit event
            # Original code would terminate loop after scanning to end
            return_bar = None
            did_return = False
            bars_outside = n_bars - exit_bar - 1
            # Record this exit and break - no more exits after an unreturned exit
            exit_events.append(ExitEvent(
                bar_index=exit_bar,
                exit_direction=direction,
                bars_outside=max(bars_outside, 1),
                did_return=did_return,
                return_bar=return_bar,
            ))
            break

        exit_events.append(ExitEvent(
            bar_index=exit_bar,
            exit_direction=direction,
            bars_outside=max(bars_outside, 1),
            did_return=did_return,
            return_bar=return_bar,
        ))

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
        return_rate=0.0,
        channel_resilience_score=0.0,
        avg_duration_after_return=0.0,
        max_duration_after_return=0,
        returns_leading_to_new_channel=0,
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
        # No exits detected - perfect resilience (no exits means channel held)
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
            return_rate=1.0,  # No exits means no failures, treat as perfect
            channel_resilience_score=1.0,  # Maximum resilience
            avg_duration_after_return=0.0,
            max_duration_after_return=0,
            returns_leading_to_new_channel=0,
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

    # Calculate return rate: percentage of exits that resulted in returns (0-1)
    return_rate = len(returned_events) / exit_count if exit_count > 0 else 0.0

    # Calculate channel_resilience_score as a composite metric:
    # resilience = return_rate * 0.4 + (1 - avg_normalized_bars_outside) * 0.3 + (bounces_after_returns / max_bounces) * 0.3
    # Where avg_normalized_bars_outside = min(avg_bars_outside / 50, 1.0)
    avg_normalized_bars_outside = min(avg_bars_outside / 50.0, 1.0)

    # For the bounce component, use a reasonable max (e.g., 10 bounces as "excellent")
    max_bounces = 10.0
    bounce_component = min(bounces_after_last_return / max_bounces, 1.0)

    channel_resilience_score = (
        return_rate * 0.4 +
        (1.0 - avg_normalized_bars_outside) * 0.3 +
        bounce_component * 0.3
    )

    # Calculate post-return durability metrics
    # For each return event, track how many bars until the next exit or end of data
    durations_after_return: List[int] = []
    returns_leading_to_new_channel = 0

    for i, event in enumerate(exit_events):
        if event.did_return and event.return_bar is not None:
            return_bar = event.return_bar

            # Find the next exit bar, or use end of data
            next_exit_bar = n_bars  # Default to end of data
            for next_event in exit_events[i + 1:]:
                next_exit_bar = next_event.bar_index
                break

            # Duration is bars from return until next exit (or end of data)
            duration = next_exit_bar - return_bar
            durations_after_return.append(duration)

            # Count if this return led to sustained channel behavior (>20 bars inside)
            if duration > 20:
                returns_leading_to_new_channel += 1

    if durations_after_return:
        avg_duration_after_return = float(np.mean(durations_after_return))
        max_duration_after_return = int(np.max(durations_after_return))
    else:
        avg_duration_after_return = 0.0
        max_duration_after_return = 0

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
        return_rate=return_rate,
        channel_resilience_score=channel_resilience_score,
        avg_duration_after_return=avg_duration_after_return,
        max_duration_after_return=max_duration_after_return,
        returns_leading_to_new_channel=returns_leading_to_new_channel,
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
        'return_rate': features.return_rate,
        'channel_resilience_score': features.channel_resilience_score,
        'avg_duration_after_return': features.avg_duration_after_return,
        'max_duration_after_return': features.max_duration_after_return,
        'returns_leading_to_new_channel': features.returns_leading_to_new_channel,
    }
