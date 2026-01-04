"""
Channel Detection Module

Detects price channels using linear regression with ±2σ bounds.
Key insight: Use HIGHS for upper touches, LOWS for lower touches.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import IntEnum


# Standard window sizes for multi-window channel detection
STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]


class Direction(IntEnum):
    """Channel direction classification."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


class TouchType(IntEnum):
    """Type of channel boundary touch."""
    LOWER = 0
    UPPER = 1


@dataclass
class Touch:
    """Record of a channel boundary touch."""
    bar_index: int
    touch_type: TouchType
    price: float  # The HIGH (for upper) or LOW (for lower) that touched


@dataclass
class Channel:
    """
    Represents a detected price channel.

    Attributes:
        valid: Whether this is a valid channel (has bounces)
        direction: BULL, BEAR, or SIDEWAYS
        slope: Regression slope (price change per bar)
        intercept: Regression intercept
        r_squared: Quality of linear fit (0-1)
        std_dev: Standard deviation of residuals
        upper_line: Upper boundary values (array)
        lower_line: Lower boundary values (array)
        center_line: Center regression line (array)
        touches: List of boundary touches detected
        complete_cycles: Count of full round-trips (L→U→L or U→L→U)
        bounce_count: Count of alternating touches
        width_pct: Channel width as percentage of price
        window: Number of bars used
        alternations: Count of alternating touches (L->U or U->L transitions)
        alternation_ratio: alternations / (len(touches) - 1), measures bounce cleanliness
        upper_touches: Count of upper boundary touches
        lower_touches: Count of lower boundary touches
    """
    valid: bool
    direction: Direction
    slope: float
    intercept: float
    r_squared: float
    std_dev: float
    upper_line: np.ndarray
    lower_line: np.ndarray
    center_line: np.ndarray
    touches: List[Touch]
    complete_cycles: int
    bounce_count: int
    width_pct: float
    window: int
    alternations: int = 0
    alternation_ratio: float = 0.0
    upper_touches: int = 0
    lower_touches: int = 0
    quality_score: float = 0.0

    # Optional: store the OHLC data used
    close: np.ndarray = field(default=None, repr=False)
    high: np.ndarray = field(default=None, repr=False)
    low: np.ndarray = field(default=None, repr=False)

    @property
    def slope_pct(self) -> float:
        """Slope as percentage per bar."""
        if self.close is None or len(self.close) == 0:
            return 0.0
        avg_price = np.mean(self.close)
        return (self.slope / avg_price) * 100 if avg_price > 0 else 0.0

    @property
    def last_touch(self) -> Optional[TouchType]:
        """Type of the most recent touch."""
        if not self.touches:
            return None
        return self.touches[-1].touch_type

    @property
    def bars_since_last_touch(self) -> int:
        """Bars since the last boundary touch."""
        if not self.touches:
            return self.window
        return self.window - 1 - self.touches[-1].bar_index

    def position_at(self, bar_index: int = -1) -> float:
        """
        Get position in channel (0=lower, 0.5=center, 1=upper).

        Args:
            bar_index: Which bar (-1 for last bar)
        """
        if self.close is None:
            return 0.5

        price = self.close[bar_index]
        upper = self.upper_line[bar_index]
        lower = self.lower_line[bar_index]

        if upper == lower:
            return 0.5

        position = (price - lower) / (upper - lower)
        return float(np.clip(position, 0.0, 1.0))

    def distance_to_upper(self, bar_index: int = -1) -> float:
        """Percentage distance to upper bound."""
        if self.close is None:
            return 0.0
        price = self.close[bar_index]
        upper = self.upper_line[bar_index]
        return ((upper - price) / price) * 100 if price > 0 else 0.0

    def distance_to_lower(self, bar_index: int = -1) -> float:
        """Percentage distance to lower bound."""
        if self.close is None:
            return 0.0
        price = self.close[bar_index]
        lower = self.lower_line[bar_index]
        return ((price - lower) / price) * 100 if price > 0 else 0.0


def detect_bounces(
    high: np.ndarray,
    low: np.ndarray,
    upper_line: np.ndarray,
    lower_line: np.ndarray,
    threshold: float = 0.10
) -> Tuple[List[Touch], int, int, int, int]:
    """
    Detect channel boundary touches and count cycles.

    Uses HIGH prices for upper touches, LOW prices for lower touches.
    Threshold is percentage of channel width.

    Args:
        high: High prices for each bar
        low: Low prices for each bar
        upper_line: Upper boundary values
        lower_line: Lower boundary values
        threshold: Touch threshold as fraction of channel width (0.10 = 10%)

    Returns:
        Tuple of (touches, bounce_count, complete_cycles, upper_touches, lower_touches)
    """
    touches = []
    channel_width = upper_line - lower_line

    for i in range(len(high)):
        width = channel_width[i]
        if width <= 0:
            continue

        # Check if HIGH is near/above upper line
        upper_dist = (upper_line[i] - high[i]) / width
        # Check if LOW is near/below lower line
        lower_dist = (low[i] - lower_line[i]) / width

        # Touch upper if HIGH is within threshold of upper (or above it)
        if upper_dist <= threshold:
            touches.append(Touch(bar_index=i, touch_type=TouchType.UPPER, price=high[i]))
        # Touch lower if LOW is within threshold of lower (or below it)
        elif lower_dist <= threshold:
            touches.append(Touch(bar_index=i, touch_type=TouchType.LOWER, price=low[i]))

    # Count alternating touches (bounces)
    bounce_count = 0
    last_type = None
    for touch in touches:
        if last_type is not None and touch.touch_type != last_type:
            bounce_count += 1
        last_type = touch.touch_type

    # Count complete cycles (full round-trips)
    complete_cycles = 0
    i = 0
    while i < len(touches) - 2:
        t1 = touches[i].touch_type
        t2 = touches[i + 1].touch_type
        t3 = touches[i + 2].touch_type

        # Lower → Upper → Lower OR Upper → Lower → Upper
        if (t1 == TouchType.LOWER and t2 == TouchType.UPPER and t3 == TouchType.LOWER) or \
           (t1 == TouchType.UPPER and t2 == TouchType.LOWER and t3 == TouchType.UPPER):
            complete_cycles += 1
            i += 2  # Skip to after this cycle
        else:
            i += 1

    # Count upper and lower touches
    upper_touches = sum(1 for t in touches if t.touch_type == TouchType.UPPER)
    lower_touches = sum(1 for t in touches if t.touch_type == TouchType.LOWER)

    return touches, bounce_count, complete_cycles, upper_touches, lower_touches


def detect_channel(
    df: pd.DataFrame,
    window: int = 50,
    std_multiplier: float = 2.0,
    touch_threshold: float = 0.10,
    min_cycles: int = 1  # Minimum alternating bounces for valid channel
) -> Channel:
    """
    Detect a price channel in OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        window: Number of bars to use for regression
        std_multiplier: Multiplier for standard deviation (default 2.0 = ±2σ)
        touch_threshold: Touch threshold as fraction of channel width
        min_cycles: Minimum alternating bounces (L→U or U→L transitions) for valid channel

    Returns:
        Channel object with all metrics
    """
    # Get last 'window' bars
    if len(df) < window:
        window = len(df)

    df_slice = df.iloc[-window:]
    close = df_slice['close'].values.astype(np.float64)
    high = df_slice['high'].values.astype(np.float64)
    low = df_slice['low'].values.astype(np.float64)

    n = len(close)
    x = np.arange(n)

    # Linear regression on close prices
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
    r_squared = r_value ** 2

    # Center line (regression fit)
    center_line = slope * x + intercept

    # Residuals and standard deviation
    residuals = close - center_line
    std_dev = np.std(residuals)

    # Upper and lower bounds
    upper_line = center_line + std_multiplier * std_dev
    lower_line = center_line - std_multiplier * std_dev

    # Channel width as percentage
    avg_price = np.mean(close)
    width_pct = ((upper_line[-1] - lower_line[-1]) / avg_price) * 100 if avg_price > 0 else 0.0

    # Detect bounces
    touches, bounce_count, complete_cycles, upper_touches, lower_touches = detect_bounces(
        high, low, upper_line, lower_line, touch_threshold
    )

    # Calculate alternation metrics
    # alternations = bounce_count (each L->U or U->L transition)
    alternations = bounce_count
    # alternation_ratio measures how "clean" the bouncing is
    # LULULU -> 5 alternations out of 5 transitions = 1.0 (perfect)
    # UULUU -> 2 alternations out of 4 transitions = 0.5 (some consecutive)
    # UUUUU -> 0 alternations = 0.0 (no bouncing)
    alternation_ratio = bounce_count / max(1, len(touches) - 1) if len(touches) > 1 else 0.0

    # Determine direction based on slope
    slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0
    if slope_pct > 0.05:
        direction = Direction.BULL
    elif slope_pct < -0.05:
        direction = Direction.BEAR
    else:
        direction = Direction.SIDEWAYS

    # Valid if enough alternating bounces
    # bounce_count measures L→U and U→L transitions, which is what we want
    valid = bounce_count >= min_cycles  # min_cycles now means min alternating bounces

    channel = Channel(
        valid=valid,
        direction=direction,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        std_dev=std_dev,
        upper_line=upper_line,
        lower_line=lower_line,
        center_line=center_line,
        touches=touches,
        complete_cycles=complete_cycles,
        bounce_count=bounce_count,
        width_pct=width_pct,
        window=window,
        alternations=alternations,
        alternation_ratio=alternation_ratio,
        upper_touches=upper_touches,
        lower_touches=lower_touches,
        close=close,
        high=high,
        low=low,
    )
    channel.quality_score = calculate_channel_quality_score(channel)
    return channel


def calculate_channel_quality_score(channel: Channel) -> float:
    """
    Calculate quality score emphasizing alternating bounces.

    Score = alternations × (1 + alternation_ratio)

    This rewards:
    - More alternations (bounces) - primary factor
    - Cleaner alternation pattern (higher ratio) - secondary factor

    Examples:
    - 5 alternations with ratio 1.0 → score = 5 × 2.0 = 10.0
    - 5 alternations with ratio 0.5 → score = 5 × 1.5 = 7.5
    - 2 alternations with ratio 1.0 → score = 2 × 2.0 = 4.0

    R² is deliberately NOT used - a channel with many bounces is valid
    regardless of how well a linear regression fits.

    Returns:
        float: Quality score (0.0+, typically 0-20)
    """
    # Use alternations field (same as bounce_count)
    alternations = channel.alternations
    ratio = channel.alternation_ratio

    quality = alternations * (1 + ratio)
    return float(quality)


def detect_channels_multi_window(
    df: pd.DataFrame,
    windows: List[int] = None,
    **kwargs
) -> Dict[int, Channel]:
    """
    Detect channels at multiple window sizes and return all of them.

    Args:
        df: OHLCV DataFrame
        windows: List of window sizes to try (defaults to STANDARD_WINDOWS)
        **kwargs: Additional arguments passed to detect_channel

    Returns:
        Dict mapping window size to detected Channel
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    channels = {}
    for w in windows:
        if len(df) >= w:
            channels[w] = detect_channel(df, window=w, **kwargs)

    return channels


def select_best_channel(channels: Dict[int, Channel]) -> Tuple[Optional[Channel], Optional[int]]:
    """
    Select the best channel from a dictionary of channels.

    Uses bounce-first sorting: more bounces always wins,
    with r_squared as a tiebreaker.

    Args:
        channels: Dict mapping window size to Channel

    Returns:
        Tuple of (best_channel, window_size) or (None, None) if no channels
    """
    if not channels:
        return None, None

    # Find the window size with the best channel (bounce-first sorting)
    best_window = max(
        channels.keys(),
        key=lambda w: (channels[w].bounce_count, channels[w].r_squared)
    )

    return channels[best_window], best_window


def find_best_channel(
    df: pd.DataFrame,
    windows: List[int] = None,
    **kwargs
) -> Optional[Channel]:
    """
    Find the best channel (most complete cycles) among multiple windows.

    Args:
        df: OHLCV DataFrame
        windows: List of window sizes to try
        **kwargs: Additional arguments passed to detect_channel

    Returns:
        Best Channel or None if no valid channels found
    """
    channels = detect_channels_multi_window(df, windows, **kwargs)

    if not channels:
        return None

    # Sort by complete_cycles (descending), then by r_squared (descending)
    best = max(
        channels.values(),
        key=lambda c: (c.complete_cycles, c.r_squared)
    )

    return best if best.valid else None
