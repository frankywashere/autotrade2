"""
Break Scanner Module for V15 Channel Analysis.

This module implements forward scanning from a channel to detect:
- First bar where price breaks outside channel bounds
- Whether the break is permanent or temporary (false break)
- Break magnitude in standard deviations
- Duration outside channel and return behavior

Ported from v7/core/channel.py ExitEvent and durability tracking patterns.

Key concepts:
- ExitEvent: Record of when price exits a channel
- BreakResult: Complete analysis of a channel break
- scan_for_break(): Main function to scan forward and detect breaks

Usage:
    from v15.core.break_scanner import scan_for_break, BreakResult

    result = scan_for_break(
        channel=channel,
        forward_high=forward_high,
        forward_low=forward_low,
        forward_close=forward_close,
        max_scan_bars=300
    )

    if result.break_detected:
        print(f"Break at bar {result.break_bar}")
        print(f"Direction: {'UP' if result.break_direction == 1 else 'DOWN'}")
        print(f"Magnitude: {result.break_magnitude:.2f} std devs")
        print(f"Permanent: {result.is_permanent}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import IntEnum

# Import from v7 for compatibility
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v7.core.channel import Channel

from ..exceptions import V15Error


# =============================================================================
# Exceptions
# =============================================================================

class BreakScannerError(V15Error):
    """Raised when break scanning fails unexpectedly."""
    pass


class InsufficientDataError(BreakScannerError):
    """Raised when there is insufficient forward data to scan."""
    pass


# =============================================================================
# Enums
# =============================================================================

class BreakDirection(IntEnum):
    """Direction of channel break."""
    DOWN = 0
    UP = 1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExitEvent:
    """
    Record of a channel exit event (ported from v7).

    Attributes:
        bar_index: Bar index (relative to scan start) when exit occurred
        exit_type: 'upper' or 'lower' indicating which boundary was breached
        exit_price: Price at exit (high for upper, low for lower breach)
        magnitude: How far outside the channel (in std devs) at exit
        returned: Whether price returned to channel after exit
        bars_outside: Number of bars spent outside channel before return (if returned)
        return_bar: Bar index where price returned to channel (if returned)
    """
    bar_index: int
    exit_type: str  # 'upper' or 'lower'
    exit_price: float
    magnitude: float = 0.0  # Breach magnitude in std devs
    returned: bool = False
    bars_outside: int = 0
    return_bar: Optional[int] = None


@dataclass
class BreakResult:
    """
    Complete result of a forward break scan.

    This captures everything needed for labels.py pass2 to determine:
    - Whether a break occurred
    - When and where the break happened
    - Whether it was permanent or temporary (false break)
    - The severity (magnitude) of the break

    IMPORTANT: Tracks BOTH first break AND permanent break separately.
    First break may be a false break that returns to channel.
    Permanent break is the final/lasting break that didn't return.

    Attributes:
        break_detected: Whether any break was detected in the scan window
        break_bar: Bar index of FIRST break (relative to channel end, 0-indexed)
        break_direction: Direction of FIRST break (0=DOWN, 1=UP)
        break_magnitude: Distance from bound in std devs (FIRST break)
        break_price: Price at FIRST break point

        is_permanent: Whether the FIRST break was permanent (never returned)
        is_false_break: Whether price returned to channel after FIRST break
        bars_until_return: Bars outside before returning (if first break returned)
        return_bar: Bar index where price returned (if first break returned)

        permanent_break_direction: Direction of FINAL/lasting break (-1=none, 0=DOWN, 1=UP)
        permanent_break_bar: Bar index of permanent break (-1 if none)
        permanent_break_magnitude: Magnitude of permanent break in std devs

        all_exit_events: List of all exit events detected during scan
        false_break_count: Count of temporary exits that returned
        false_break_rate: Ratio of false breaks to total exits

        scan_bars_used: Number of bars actually scanned
        projected_upper: Upper bound projected to break point
        projected_lower: Lower bound projected to break point
    """
    # Core break info
    break_detected: bool = False
    break_bar: int = -1
    break_direction: int = 0  # 0=DOWN, 1=UP
    break_magnitude: float = 0.0
    break_price: float = 0.0

    # First touch tracking (when price first went outside bounds, even if it returned)
    # This is the actual visual break point, before magnitude/confirmation checks
    first_touch_bar: int = -1
    first_touch_direction: int = 0  # 0=DOWN, 1=UP
    first_touch_price: float = 0.0

    # Permanence tracking (for FIRST break)
    is_permanent: bool = False
    is_false_break: bool = False
    bars_until_return: int = 0
    return_bar: Optional[int] = None

    # Permanent/final break tracking (the LASTING break, may differ from first)
    permanent_break_direction: int = -1  # -1=none, 0=DOWN, 1=UP
    permanent_break_bar: int = -1        # Bar index of permanent break (-1 if none)
    permanent_break_magnitude: float = 0.0  # Magnitude of permanent break

    # Full exit history
    all_exit_events: Optional[list] = None
    false_break_count: int = 0
    false_break_rate: float = 0.0

    # Metadata
    scan_bars_used: int = 0
    projected_upper: float = 0.0
    projected_lower: float = 0.0

    # Exit verification tracking (NEW)
    scan_timed_out: bool = False           # Did scan hit max_scan without confirming permanence?
    bars_verified_permanent: int = 0       # How many bars was price outside before declaring permanent?
    exits_returned_count: int = 0          # Count of exits that returned (same as false_break_count)
    exits_stayed_out_count: int = 0        # Count of exits that didn't return
    exit_return_rate: float = 0.0          # exits_returned / total_exits

    # Round-trip bounce tracking
    round_trip_bounces: int = 0            # Count of complete round-trip bounces (upper->lower or lower->upper)

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.all_exit_events is None:
            self.all_exit_events = []


# =============================================================================
# Core Functions
# =============================================================================

def project_channel_bounds(
    channel: Channel,
    bars_forward: int
) -> Tuple[float, float, float]:
    """
    Project channel bounds forward from the end of the channel.

    The channel's linear regression is: center = slope * x + intercept
    where x is the bar index relative to the start of the channel window.
    At the end of the channel, x = window - 1.
    To project forward, we add bars_forward to x.

    Args:
        channel: Channel object with slope, intercept, std_dev, window
        bars_forward: Number of bars forward from channel end to project

    Returns:
        Tuple of (projected_center, projected_upper, projected_lower)

    Raises:
        BreakScannerError: If channel lacks required attributes
    """
    if channel.slope is None:
        raise BreakScannerError("Channel missing slope - cannot project bounds")
    if channel.intercept is None:
        raise BreakScannerError("Channel missing intercept - cannot project bounds")
    if channel.std_dev is None or channel.std_dev == 0:
        raise BreakScannerError("Channel missing or zero std_dev - cannot project bounds")

    # x at end of channel = window - 1
    # x for first forward bar = window (one bar after channel ends)
    # x projected forward = window + bars_forward
    projection_x = channel.window + bars_forward

    # Projected center line at that point
    projected_center = channel.slope * projection_x + channel.intercept

    # Use 2 * std_dev for channel width (standard +-2 sigma bounds)
    std_multiplier = 2.0
    projected_upper = projected_center + std_multiplier * channel.std_dev
    projected_lower = projected_center - std_multiplier * channel.std_dev

    return projected_center, projected_upper, projected_lower


def scan_for_break(
    channel: Channel,
    forward_high: np.ndarray,
    forward_low: np.ndarray,
    forward_close: np.ndarray,
    max_scan_bars: int = 300,
    return_threshold_bars: int = 5,
    min_break_magnitude: float = 0.5
) -> BreakResult:
    """
    Scan forward from channel end to detect the first break outside bounds.

    This function projects the channel forward and checks each bar to see if
    price has broken outside the projected bounds. It tracks:
    - The first bar where price breaks out (HIGH > upper or LOW < lower)
    - Whether price subsequently returns to the channel
    - All exit events (for durability analysis)

    Args:
        channel: Channel object with regression parameters (slope, intercept, std_dev)
        forward_high: High prices for bars after channel end
        forward_low: Low prices for bars after channel end
        forward_close: Close prices for bars after channel end
        max_scan_bars: Maximum bars to scan forward
        return_threshold_bars: Max bars to wait for return before declaring permanent
        min_break_magnitude: Minimum magnitude (in std devs) to count as a break.
                             Prevents flagging noise/minor touches as breaks. Default 0.5.

    Returns:
        BreakResult with complete break analysis

    Raises:
        InsufficientDataError: If forward arrays are empty
        BreakScannerError: If channel lacks required attributes

    Example:
        >>> result = scan_for_break(
        ...     channel=my_channel,
        ...     forward_high=df['high'].values[channel_end:],
        ...     forward_low=df['low'].values[channel_end:],
        ...     forward_close=df['close'].values[channel_end:],
        ...     max_scan_bars=300
        ... )
        >>> print(f"Break at bar {result.break_bar}, magnitude {result.break_magnitude:.2f}")
    """
    # Validate inputs
    if len(forward_high) == 0 or len(forward_low) == 0 or len(forward_close) == 0:
        raise InsufficientDataError(
            f"Forward arrays are empty - cannot scan for break. "
            f"high={len(forward_high)}, low={len(forward_low)}, close={len(forward_close)}"
        )

    # Ensure arrays are the same length
    min_len = min(len(forward_high), len(forward_low), len(forward_close))
    actual_scan = min(min_len, max_scan_bars)

    if actual_scan == 0:
        raise InsufficientDataError("No bars available to scan")

    # Initialize result
    result = BreakResult(scan_bars_used=actual_scan)

    # State tracking
    exit_events: list = []
    current_exit: Optional[ExitEvent] = None
    inside_channel = True
    first_break_found = False
    first_permanent_found = False  # Track FIRST exit that stays outside for 5+ bars

    # Scan each bar
    for bar_idx in range(actual_scan):
        high = forward_high[bar_idx]
        low = forward_low[bar_idx]
        close = forward_close[bar_idx]

        # Project bounds to this bar
        _, upper, lower = project_channel_bounds(channel, bar_idx)

        # Check for exit (breach outside bounds)
        # Only count as break if magnitude exceeds min_break_magnitude threshold

        # Track FIRST TOUCH - the first bar where CLOSE went outside bounds
        # This is tracked regardless of magnitude (for visual marker placement)
        if result.first_touch_bar < 0:
            if close > upper:  # Require CLOSE above upper
                result.first_touch_bar = bar_idx
                result.first_touch_direction = int(BreakDirection.UP)
                result.first_touch_price = close  # Use close price
            elif close < lower:  # Require CLOSE below lower
                result.first_touch_bar = bar_idx
                result.first_touch_direction = int(BreakDirection.DOWN)
                result.first_touch_price = close  # Use close price

        # Track current exit direction for opposite-direction break detection
        current_exit_direction = None
        if current_exit is not None:
            current_exit_direction = current_exit.exit_type

        if close > upper:
            # Upper breach - calculate magnitude first (using close)
            magnitude = (close - upper) / channel.std_dev if channel.std_dev > 0 else 0.0

            # Count as break if magnitude exceeds threshold AND either:
            # 1. We're inside the channel (normal break), OR
            # 2. We were outside in the OPPOSITE direction (direction reversal)
            is_direction_reversal = (current_exit_direction == 'lower')
            if magnitude >= min_break_magnitude and (inside_channel or is_direction_reversal):
                # If direction reversal, close out the previous exit first
                if is_direction_reversal and current_exit is not None:
                    current_exit.returned = True  # Treat as returned (crossed back through)
                    current_exit.bars_outside = bar_idx - current_exit.bar_index
                    current_exit.return_bar = bar_idx
                    exit_events.append(current_exit)

                # New exit event
                current_exit = ExitEvent(
                    bar_index=bar_idx,
                    exit_type='upper',
                    exit_price=close,  # Use close price
                    magnitude=magnitude
                )
                inside_channel = False

                # Record first break
                if not first_break_found:
                    first_break_found = True
                    result.break_detected = True
                    result.break_bar = bar_idx
                    result.break_direction = int(BreakDirection.UP)
                    result.break_price = close  # Use close price
                    result.projected_upper = upper
                    result.projected_lower = lower
                    result.break_magnitude = magnitude

        elif close < lower:
            # Lower breach - calculate magnitude first (using close)
            magnitude = (lower - close) / channel.std_dev if channel.std_dev > 0 else 0.0

            # Count as break if magnitude exceeds threshold AND either:
            # 1. We're inside the channel (normal break), OR
            # 2. We were outside in the OPPOSITE direction (direction reversal)
            is_direction_reversal = (current_exit_direction == 'upper')
            if magnitude >= min_break_magnitude and (inside_channel or is_direction_reversal):
                # If direction reversal, close out the previous exit first
                if is_direction_reversal and current_exit is not None:
                    current_exit.returned = True  # Treat as returned (crossed back through)
                    current_exit.bars_outside = bar_idx - current_exit.bar_index
                    current_exit.return_bar = bar_idx
                    exit_events.append(current_exit)

                # New exit event
                current_exit = ExitEvent(
                    bar_index=bar_idx,
                    exit_type='lower',
                    exit_price=close,  # Use close price
                    magnitude=magnitude
                )
                inside_channel = False

                # Record first break
                if not first_break_found:
                    first_break_found = True
                    result.break_detected = True
                    result.break_bar = bar_idx
                    result.break_direction = int(BreakDirection.DOWN)
                    result.break_price = close  # Use close price
                    result.projected_upper = upper
                    result.projected_lower = lower
                    result.break_magnitude = magnitude

        # Check for return to channel (close inside bounds)
        if not inside_channel and lower <= close <= upper:
            # Price returned to channel
            inside_channel = True
            if current_exit is not None:
                current_exit.returned = True
                current_exit.bars_outside = bar_idx - current_exit.bar_index
                current_exit.return_bar = bar_idx
                exit_events.append(current_exit)
                current_exit = None

        # Check for PERMANENT break - FIRST exit that stays outside for 5+ consecutive bars
        # This is checked DURING the scan, not at the end, so we catch it even if price
        # eventually returns after many bars
        if not inside_channel and current_exit is not None and not first_permanent_found:
            bars_outside_so_far = bar_idx - current_exit.bar_index
            if bars_outside_so_far >= return_threshold_bars:
                # This exit has been outside for 5+ bars - mark as permanent
                first_permanent_found = True
                result.permanent_break_bar = current_exit.bar_index
                result.permanent_break_direction = (
                    int(BreakDirection.UP) if current_exit.exit_type == 'upper'
                    else int(BreakDirection.DOWN)
                )
                # Calculate magnitude at the permanent break bar
                _, perm_upper, perm_lower = project_channel_bounds(channel, current_exit.bar_index)
                if channel.std_dev > 0:
                    if current_exit.exit_type == 'upper':
                        result.permanent_break_magnitude = (
                            current_exit.exit_price - perm_upper
                        ) / channel.std_dev
                    else:
                        result.permanent_break_magnitude = (
                            perm_lower - current_exit.exit_price
                        ) / channel.std_dev
                else:
                    result.permanent_break_magnitude = 0.0

    # Handle final exit if we're still outside at end of scan
    if not inside_channel and current_exit is not None:
        current_exit.returned = False
        current_exit.bars_outside = actual_scan - current_exit.bar_index
        exit_events.append(current_exit)

    # Store all exit events
    result.all_exit_events = exit_events

    # Calculate exit statistics (NEW: expanded from false_break_count)
    if exit_events:
        result.exits_returned_count = sum(1 for e in exit_events if e.returned)
        result.exits_stayed_out_count = len(exit_events) - result.exits_returned_count
        result.exit_return_rate = result.exits_returned_count / len(exit_events)

        # Backward compatibility - keep false_break fields as aliases
        result.false_break_count = result.exits_returned_count
        result.false_break_rate = result.exit_return_rate
    else:
        result.exits_returned_count = 0
        result.exits_stayed_out_count = 0
        result.exit_return_rate = 0.0
        result.false_break_count = 0
        result.false_break_rate = 0.0

    # Count round-trip bounces
    # A round-trip bounce is when price alternates between upper and lower bounds:
    # - Exit upper -> return -> exit lower -> return = 1 bounce
    # - Exit lower -> return -> exit upper -> return = 1 bounce
    # We count complete round-trips where both exits returned to channel
    round_trip_bounces = 0
    if len(exit_events) >= 2:
        for i in range(1, len(exit_events)):
            prev_exit = exit_events[i - 1]
            curr_exit = exit_events[i]
            # Check if direction alternated and both exits returned
            if (prev_exit.exit_type != curr_exit.exit_type and
                    prev_exit.returned and curr_exit.returned):
                round_trip_bounces += 1
    result.round_trip_bounces = round_trip_bounces

    # PERMANENT BREAK LOGIC:
    # Primary: Use the FIRST exit that stayed outside for 5+ consecutive bars
    #          (already tracked during the scan loop via first_permanent_found)
    # Fallback: If no exit reached 5+ bars during scan, use the last exit that
    #           didn't return (still outside at scan end)

    if not first_permanent_found:
        # No exit stayed outside for 5+ bars during scan
        # Fall back to finding the last exit that didn't return (if any)
        permanent_exit = None
        for exit_event in reversed(exit_events):
            if not exit_event.returned:
                permanent_exit = exit_event
                break

        if permanent_exit is not None:
            result.permanent_break_bar = permanent_exit.bar_index
            result.permanent_break_direction = (
                int(BreakDirection.UP) if permanent_exit.exit_type == 'upper'
                else int(BreakDirection.DOWN)
            )

            # Calculate magnitude for permanent break
            _, perm_upper, perm_lower = project_channel_bounds(channel, permanent_exit.bar_index)

            if channel.std_dev > 0:
                if permanent_exit.exit_type == 'upper':
                    result.permanent_break_magnitude = (
                        permanent_exit.exit_price - perm_upper
                    ) / channel.std_dev
                else:
                    result.permanent_break_magnitude = (
                        perm_lower - permanent_exit.exit_price
                    ) / channel.std_dev
            else:
                result.permanent_break_magnitude = 0.0

            result.bars_verified_permanent = permanent_exit.bars_outside
        else:
            # No permanent break found - all exits returned within 5 bars
            result.permanent_break_direction = -1
            result.permanent_break_bar = -1
            result.permanent_break_magnitude = 0.0
            result.bars_verified_permanent = 0
    else:
        # Permanent break was found during scan (first exit that stayed 5+ bars)
        # Calculate bars_verified_permanent from the exit events
        for exit_event in exit_events:
            if exit_event.bar_index == result.permanent_break_bar:
                result.bars_verified_permanent = exit_event.bars_outside
                break
        else:
            # If the permanent exit is still current (not in exit_events yet)
            if current_exit is not None and current_exit.bar_index == result.permanent_break_bar:
                result.bars_verified_permanent = actual_scan - current_exit.bar_index

    # Determine scan timeout
    # Scan timed out if we hit max_scan_bars and no permanent break was confirmed
    result.scan_timed_out = False
    if result.scan_bars_used >= max_scan_bars:
        if result.permanent_break_bar < 0:
            # No permanent break found - might have found one if we scanned longer
            result.scan_timed_out = True
        elif result.bars_verified_permanent < return_threshold_bars:
            # Found a break but didn't verify it long enough
            result.scan_timed_out = True

    # Determine if first break was permanent or false break
    if result.break_detected and exit_events:
        first_exit = exit_events[0]
        result.is_false_break = first_exit.returned
        result.is_permanent = not first_exit.returned

        if first_exit.returned:
            result.bars_until_return = first_exit.bars_outside
            result.return_bar = first_exit.return_bar
        else:
            # Check if we have enough data to declare permanent
            # Only declare permanent if we scanned past the threshold
            if first_exit.bars_outside >= return_threshold_bars:
                result.is_permanent = True
            else:
                # Not enough data - leave as undetermined (is_permanent=False)
                pass

    return result


def calculate_durability_score(
    false_break_count: int,
    total_exits: int,
    avg_bars_outside: float
) -> float:
    """
    Calculate channel durability score from exit events (ported from v7).

    A channel that survives many false breaks is more durable/reliable.
    This score combines:
    - False break rate (higher = more resilient)
    - Volume of false breaks (more = more proven)
    - Quick returns (faster returns = stronger channel)

    Args:
        false_break_count: Number of exits that returned (false breaks)
        total_exits: Total number of exit events
        avg_bars_outside: Average bars spent outside before returning

    Returns:
        Durability score (0.0-1.5+, higher is more durable)

    Example:
        - 3 exits, all returned fast -> high score (~1.3)
        - 5 exits, 2 returned -> medium score (~0.5)
        - 2 exits, none returned -> low score (0.0)
    """
    if total_exits == 0:
        return 0.0

    # False break rate (higher = more resilient)
    false_break_rate = false_break_count / total_exits

    # Quick return factor: faster returns = more durable
    # Uses exponential decay: e^(-0.1 * avg_bars) so 0 bars = 1.0, 10 bars = 0.37
    quick_return_factor = np.exp(-0.1 * avg_bars_outside) if false_break_count > 0 else 0.0

    # Volume factor: more false breaks survived = more confidence
    # Uses log scaling to prevent runaway values
    volume_factor = np.log1p(false_break_count) / np.log1p(10)  # Normalized to ~1.0 at 10 breaks

    # Composite durability score
    durability_score = false_break_rate * (1 + 0.3 * quick_return_factor + 0.2 * volume_factor)

    return float(durability_score)


def compute_durability_from_result(result: BreakResult) -> Tuple[int, float, float]:
    """
    Compute durability metrics from a BreakResult.

    Convenience function that extracts statistics from exit events
    and computes the durability score.

    Args:
        result: BreakResult from scan_for_break()

    Returns:
        Tuple of (false_break_count, false_break_rate, durability_score)
    """
    if not result.all_exit_events:
        return 0, 0.0, 0.0

    total_exits = len(result.all_exit_events)
    false_break_count = result.false_break_count
    false_break_rate = result.false_break_rate

    # Calculate average bars outside for returns
    if false_break_count > 0:
        avg_bars_outside = sum(
            e.bars_outside for e in result.all_exit_events if e.returned
        ) / false_break_count
    else:
        avg_bars_outside = 0.0

    durability_score = calculate_durability_score(
        false_break_count, total_exits, avg_bars_outside
    )

    return false_break_count, false_break_rate, durability_score
