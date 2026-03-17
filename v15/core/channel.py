"""
Channel Detection Module

Detects price channels using linear regression with ±2σ bounds.
Key insight: Use HIGHS for upper touches, LOWS for lower touches.

NOTE: This module was originally in v7/core/channel.py and has been
copied to v15/core/ to remove the v7 dependency.

Performance: The inner computation is JIT-compiled with numba for C-speed
execution. Falls back to pure-numpy if numba is unavailable.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import IntEnum

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


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
    # False break tracking fields
    false_break_count: int = 0  # Number of temporary exits that returned
    false_break_rate: float = 0.0  # false_breaks / total_exits (0-1, higher = more resilient)
    channel_durability_score: float = 0.0  # Composite score including false break resilience

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


# ---------------------------------------------------------------------------
# Numba JIT-compiled core (C-speed inner loop)
# ---------------------------------------------------------------------------

def _make_jit_core():
    """Build and return the numba-compiled core function, or None if unavailable."""
    if not _HAS_NUMBA:
        return None

    @numba.njit(cache=True)
    def _detect_channel_core(close, high, low, std_multiplier, touch_threshold):
        """
        JIT-compiled core of detect_channel + detect_bounces.

        Returns a flat tuple of scalars and arrays — no Python objects.

        Returns:
            (slope, intercept, r_squared, std_dev, avg_price,
             upper, lower, center,
             touch_bar_indices, touch_types, touch_prices, n_touches,
             bounce_count, complete_cycles, upper_touches_count, lower_touches_count)
        """
        n = len(close)

        # --- Linear regression on close prices ---
        x = np.arange(n).astype(np.float64)
        x_mean = 0.0
        y_mean = 0.0
        for i in range(n):
            x_mean += x[i]
            y_mean += close[i]
        x_mean /= n
        y_mean /= n

        xy_cov = 0.0
        x_var = 0.0
        ss_tot = 0.0
        for i in range(n):
            xm = x[i] - x_mean
            ym = close[i] - y_mean
            xy_cov += xm * ym
            x_var += xm * xm
            ss_tot += ym * ym

        slope = xy_cov / x_var
        intercept = y_mean - slope * x_mean

        # --- Center line, residuals, std_dev ---
        center = np.empty(n, dtype=np.float64)
        residuals_sq_sum = 0.0
        residuals_sum = 0.0
        for i in range(n):
            center[i] = slope * x[i] + intercept
            r = close[i] - center[i]
            residuals_sum += r
            residuals_sq_sum += r * r

        # Population std (matches np.std)
        res_mean = residuals_sum / n
        # Var = E[X^2] - (E[X])^2
        variance = residuals_sq_sum / n - res_mean * res_mean
        if variance < 0.0:
            variance = 0.0
        std_dev = variance ** 0.5

        # --- R-squared ---
        ss_res = residuals_sq_sum
        if ss_tot > 0.0:
            r_squared = 1.0 - ss_res / ss_tot
        else:
            r_squared = 0.0

        # --- Upper and lower bounds ---
        band = std_multiplier * std_dev
        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        avg_price = 0.0
        for i in range(n):
            upper[i] = center[i] + band
            lower[i] = center[i] - band
            avg_price += close[i]
        avg_price /= n

        # --- Touch detection ---
        # First pass: count touches to allocate arrays
        max_touches = 2 * n  # Upper bound: both upper and lower at every bar
        touch_bar_indices = np.empty(max_touches, dtype=np.int64)
        touch_types = np.empty(max_touches, dtype=np.int8)  # 1=UPPER, 0=LOWER
        touch_prices = np.empty(max_touches, dtype=np.float64)
        t_count = 0
        upper_touches_count = 0
        lower_touches_count = 0

        for i in range(n):
            width = upper[i] - lower[i]
            if width <= 0.0:
                continue
            u_dist = (upper[i] - high[i]) / width
            l_dist = (low[i] - lower[i]) / width

            is_upper = u_dist <= touch_threshold
            is_lower = l_dist <= touch_threshold

            # Upper before lower for same bar (preserving original merge order)
            if is_upper:
                touch_bar_indices[t_count] = i
                touch_types[t_count] = 1  # UPPER
                touch_prices[t_count] = high[i]
                t_count += 1
                upper_touches_count += 1
            if is_lower:
                touch_bar_indices[t_count] = i
                touch_types[t_count] = 0  # LOWER
                touch_prices[t_count] = low[i]
                t_count += 1
                lower_touches_count += 1

        # --- Bounce count (alternating touches) ---
        bounce_count = 0
        if t_count > 1:
            for i in range(t_count - 1):
                if touch_types[i] != touch_types[i + 1]:
                    bounce_count += 1

        # --- Complete cycles ---
        complete_cycles = 0
        i = 0
        while i < t_count - 2:
            t1 = touch_types[i]
            t2 = touch_types[i + 1]
            t3 = touch_types[i + 2]
            if t1 != t2 and t2 != t3 and t1 == t3:
                complete_cycles += 1
                i += 2
            else:
                i += 1

        # Trim touch arrays to actual size
        out_bar_indices = touch_bar_indices[:t_count].copy()
        out_types = touch_types[:t_count].copy()
        out_prices = touch_prices[:t_count].copy()

        return (slope, intercept, r_squared, std_dev, avg_price,
                upper, lower, center,
                out_bar_indices, out_types, out_prices, t_count,
                bounce_count, complete_cycles,
                upper_touches_count, lower_touches_count)

    return _detect_channel_core


# Build JIT function at import time (compilation is deferred until first call)
_detect_channel_core_jit = _make_jit_core()


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
    channel_width = upper_line - lower_line

    # Vectorized touch detection
    valid = channel_width > 0
    # Safe divide (avoid division by zero; invalid entries won't pass threshold)
    safe_width = np.where(valid, channel_width, 1.0)
    upper_dist = (upper_line - high) / safe_width
    lower_dist = (low - lower_line) / safe_width

    upper_touch_mask = valid & (upper_dist <= threshold)
    lower_touch_mask = valid & (lower_dist <= threshold)

    upper_indices = np.nonzero(upper_touch_mask)[0]
    lower_indices = np.nonzero(lower_touch_mask)[0]

    # Build touches list sorted by bar index, with upper before lower for same bar
    # (preserving original loop order: upper checked first, then lower)
    touches = []
    ui, li = 0, 0
    n_upper, n_lower = len(upper_indices), len(lower_indices)
    while ui < n_upper or li < n_lower:
        u_idx = upper_indices[ui] if ui < n_upper else len(high) + 1
        l_idx = lower_indices[li] if li < n_lower else len(high) + 1
        if u_idx <= l_idx:
            touches.append(Touch(bar_index=int(u_idx), touch_type=TouchType.UPPER, price=high[u_idx]))
            ui += 1
            if l_idx == u_idx:
                touches.append(Touch(bar_index=int(l_idx), touch_type=TouchType.LOWER, price=low[l_idx]))
                li += 1
        else:
            touches.append(Touch(bar_index=int(l_idx), touch_type=TouchType.LOWER, price=low[l_idx]))
            li += 1

    # Count alternating touches (bounces)
    if len(touches) > 1:
        touch_types = np.array([t.touch_type for t in touches], dtype=np.int8)
        bounce_count = int(np.count_nonzero(np.diff(touch_types)))
    else:
        bounce_count = 0

    # Count complete cycles (full round-trips)
    complete_cycles = 0
    i = 0
    n_touches = len(touches)
    while i < n_touches - 2:
        t1 = touches[i].touch_type
        t2 = touches[i + 1].touch_type
        t3 = touches[i + 2].touch_type

        # Lower → Upper → Lower OR Upper → Lower → Upper
        if t1 != t2 and t2 != t3 and t1 == t3:
            complete_cycles += 1
            i += 2  # Skip to after this cycle
        else:
            i += 1

    # Count upper and lower touches (already computed)
    upper_touches = n_upper
    lower_touches = n_lower

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

    Uses numba JIT-compiled core for C-speed computation when available,
    falls back to pure-numpy otherwise.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        window: Number of bars to use for regression
        std_multiplier: Multiplier for standard deviation (default 2.0 = ±2σ)
        touch_threshold: Touch threshold as fraction of channel width
        min_cycles: Minimum alternating bounces (L→U or U→L transitions) for valid channel

    Returns:
        Channel object with all metrics
    """
    # Get last 'window' bars, excluding current bar to prevent data leakage
    if len(df) < window + 1:
        window = len(df) - 1

    df_slice = df.iloc[-(window+1):-1]
    close = df_slice['close'].values.astype(np.float64)
    high = df_slice['high'].values.astype(np.float64)
    low = df_slice['low'].values.astype(np.float64)

    if _detect_channel_core_jit is not None:
        # --- Fast path: numba JIT ---
        (slope, intercept, r_squared, std_dev, avg_price,
         upper_line, lower_line, center_line,
         touch_bar_indices, touch_types_arr, touch_prices_arr, n_touches,
         bounce_count, complete_cycles,
         upper_touches, lower_touches) = _detect_channel_core_jit(
            close, high, low, std_multiplier, touch_threshold)

        # Convert scalar numpy types to Python (for dataclass compatibility)
        slope = float(slope)
        intercept = float(intercept)
        r_squared = float(r_squared)
        std_dev = float(std_dev)
        avg_price = float(avg_price)
        bounce_count = int(bounce_count)
        complete_cycles = int(complete_cycles)
        upper_touches = int(upper_touches)
        lower_touches = int(lower_touches)
        n_touches = int(n_touches)

        # Build Touch objects from arrays
        touches = []
        for i in range(n_touches):
            tt = TouchType.UPPER if touch_types_arr[i] == 1 else TouchType.LOWER
            touches.append(Touch(
                bar_index=int(touch_bar_indices[i]),
                touch_type=tt,
                price=float(touch_prices_arr[i]),
            ))

        # Channel width as percentage
        width_pct = ((upper_line[-1] - lower_line[-1]) / avg_price) * 100 if avg_price > 0 else 0.0
    else:
        # --- Fallback: pure-numpy path ---
        n = len(close)
        x = np.arange(n)

        # Linear regression on close prices (pure numpy, avoids scipy overhead)
        x_f = x.astype(np.float64)
        x_mean = x_f.mean()
        y_mean = close.mean()
        xm = x_f - x_mean
        ym = close - y_mean
        xy_cov = (xm * ym).sum()
        x_var = (xm * xm).sum()
        slope = xy_cov / x_var
        intercept = y_mean - slope * x_mean

        # Center line (regression fit)
        center_line = slope * x_f + intercept

        # Residuals and standard deviation
        residuals = close - center_line
        std_dev = np.std(residuals)

        # R-squared (matches r_value**2 from scipy.stats.linregress)
        ss_res = (residuals * residuals).sum()
        ss_tot = (ym * ym).sum()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

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
    alternations = bounce_count
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
    valid = bounce_count >= min_cycles

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


def calculate_channel_quality_score(
    channel: Channel,
    false_break_rate: Optional[float] = None
) -> float:
    """
    Calculate quality score emphasizing alternating bounces.

    Score = alternations × (1 + alternation_ratio) × (1 + 0.2 × false_break_rate)

    This rewards:
    - More alternations (bounces) - primary factor
    - Cleaner alternation pattern (higher ratio) - secondary factor
    - Higher false break resilience (optional) - tertiary factor

    Examples:
    - 5 alternations with ratio 1.0, no false break data → score = 5 × 2.0 = 10.0
    - 5 alternations with ratio 1.0, false_break_rate 0.5 → score = 5 × 2.0 × 1.1 = 11.0
    - 5 alternations with ratio 0.5 → score = 5 × 1.5 = 7.5
    - 2 alternations with ratio 1.0 → score = 2 × 2.0 = 4.0

    R² is deliberately NOT used - a channel with many bounces is valid
    regardless of how well a linear regression fits.

    Args:
        channel: Channel object to score
        false_break_rate: Optional false break rate (0-1). If provided, incorporates
                         durability bonus. If None, uses original formula for
                         backwards compatibility.

    Returns:
        float: Quality score (0.0+, typically 0-20)
    """
    # Use alternations field (same as bounce_count)
    alternations = channel.alternations
    ratio = channel.alternation_ratio

    quality = alternations * (1 + ratio)

    # Optionally incorporate false break resilience
    if false_break_rate is not None:
        quality = quality * (1 + 0.2 * false_break_rate)

    return float(quality)


@dataclass
class ExitEvent:
    """
    Record of a channel exit event.

    Attributes:
        bar_index: Bar index when exit occurred
        exit_type: 'upper' or 'lower' indicating which boundary was breached
        returned: Whether price returned to channel after exit
        bars_outside: Number of bars spent outside channel before return (if returned)
    """
    bar_index: int
    exit_type: str  # 'upper' or 'lower'
    returned: bool
    bars_outside: int = 0


def calculate_channel_durability(
    channel: Channel,
    exit_events: List[ExitEvent]
) -> Tuple[int, float, float]:
    """
    Calculate channel durability based on false break analysis.

    A false break occurs when price temporarily exits the channel but returns.
    Channels that survive many false breaks are more durable/reliable.

    Args:
        channel: Channel object to analyze
        exit_events: List of ExitEvent objects representing channel exits

    Returns:
        Tuple of (false_break_count, false_break_rate, durability_score)
        - false_break_count: Number of exits that returned (false breaks)
        - false_break_rate: false_breaks / total_exits (0-1, higher = more resilient)
        - durability_score: Composite durability score

    Examples:
        - 3 exits, all returned → rate = 1.0, very durable channel
        - 5 exits, 2 returned → rate = 0.4, moderately durable
        - 2 exits, 0 returned → rate = 0.0, breaks are real breakouts
    """
    if not exit_events:
        return 0, 0.0, 0.0

    total_exits = len(exit_events)
    false_break_count = sum(1 for e in exit_events if e.returned)

    # Calculate false break rate (higher = more resilient)
    false_break_rate = false_break_count / total_exits if total_exits > 0 else 0.0

    # Durability score combines:
    # 1. Base: false_break_rate (0-1)
    # 2. Volume bonus: more false breaks survived = more proven durability
    # 3. Quick return bonus: if exits return quickly, channel is stronger
    avg_bars_outside = 0.0
    if false_break_count > 0:
        avg_bars_outside = sum(
            e.bars_outside for e in exit_events if e.returned
        ) / false_break_count

    # Quick return factor: faster returns = more durable (decay with bars outside)
    # Uses exponential decay: e^(-0.1 * avg_bars) so 0 bars = 1.0, 10 bars = 0.37
    quick_return_factor = np.exp(-0.1 * avg_bars_outside) if false_break_count > 0 else 0.0

    # Volume factor: more false breaks survived = more confidence
    # Uses log scaling to prevent runaway values
    volume_factor = np.log1p(false_break_count) / np.log1p(10)  # Normalized to ~1.0 at 10 breaks

    # Composite durability score
    # Base rate contributes most, with bonuses for quick returns and high volume
    durability_score = false_break_rate * (1 + 0.3 * quick_return_factor + 0.2 * volume_factor)

    return false_break_count, false_break_rate, float(durability_score)


def detect_channels_multi_window(
    df: pd.DataFrame,
    windows: List[int] = None,
    max_workers: int = 4,
    **kwargs
) -> Dict[int, Channel]:
    """
    Detect channels at multiple window sizes and return all of them.

    Uses parallel execution via ThreadPoolExecutor for improved performance.

    Args:
        df: OHLCV DataFrame
        windows: List of window sizes to try (defaults to STANDARD_WINDOWS)
        max_workers: Maximum number of parallel workers (default 4)
        **kwargs: Additional arguments passed to detect_channel

    Returns:
        Dict mapping window size to detected Channel
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    valid_windows = [w for w in windows if len(df) >= w]

    def detect_for_window(w):
        return w, detect_channel(df, window=w, **kwargs)

    channels = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(detect_for_window, valid_windows)
        for w, channel in results:
            channels[w] = channel

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
