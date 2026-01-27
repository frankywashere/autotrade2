"""
v15/features/spy_channel.py - SPY Channel Feature Extraction

Extracts 58 channel-specific features from a SPY Channel object.
All features return valid floats with safe defaults (no NaN, no Inf).
All feature names are prefixed with 'spy_'.

Features:
- 50 base channel features (mirroring TSLA channel features)
- 8 excursion features for price movements beyond channel boundaries

Total features: 58 per window/TF combination
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, TYPE_CHECKING

from .utils import safe_float, safe_divide, safe_pct_change, get_last_valid, atr

if TYPE_CHECKING:
    from v15.core.channel import Channel


def extract_spy_channel_features(
    spy_df: pd.DataFrame,
    channel: "Channel",
    window: int,
    tf: str
) -> Dict[str, float]:
    """
    Extract 58 SPY channel features from a Channel object detected on SPY data.

    Args:
        spy_df: SPY DataFrame with OHLC data
        channel: Channel object detected on SPY data
        window: Window size used for channel detection
        tf: Timeframe name (e.g., 'daily', '1h')

    Returns:
        Dict[str, float] with 58 features, all guaranteed to be valid floats
        All keys are prefixed with 'spy_'
    """
    features: Dict[str, float] = {}

    # Handle None or invalid channel
    if channel is None:
        return _get_default_features()

    is_valid = getattr(channel, 'valid', False)

    # ==========================================================================
    # 1. spy_channel_valid (0/1)
    # ==========================================================================
    features['spy_channel_valid'] = 1.0 if is_valid else 0.0

    # ==========================================================================
    # 2. spy_channel_direction (0=bear, 1=sideways, 2=bull)
    # ==========================================================================
    direction = getattr(channel, 'direction', 1)
    features['spy_channel_direction'] = safe_float(int(direction), 1.0)

    # ==========================================================================
    # 3. spy_channel_slope
    # ==========================================================================
    slope = getattr(channel, 'slope', 0.0)
    features['spy_channel_slope'] = safe_float(slope, 0.0)

    # ==========================================================================
    # 4. spy_channel_slope_normalized (slope / price level)
    # ==========================================================================
    close = getattr(channel, 'close', None)
    if close is not None and len(close) > 0:
        avg_price = safe_float(np.mean(close), 1.0)
        features['spy_channel_slope_normalized'] = safe_divide(slope, avg_price, 0.0)
    else:
        features['spy_channel_slope_normalized'] = 0.0

    # ==========================================================================
    # 5. spy_channel_intercept
    # ==========================================================================
    features['spy_channel_intercept'] = safe_float(getattr(channel, 'intercept', 0.0), 0.0)

    # ==========================================================================
    # 6. spy_channel_r_squared
    # ==========================================================================
    features['spy_channel_r_squared'] = safe_float(getattr(channel, 'r_squared', 0.0), 0.0)

    # ==========================================================================
    # 7. spy_channel_width_pct
    # ==========================================================================
    features['spy_channel_width_pct'] = safe_float(getattr(channel, 'width_pct', 0.0), 0.0)

    # ==========================================================================
    # 8. spy_channel_width_atr_ratio
    # ==========================================================================
    high = getattr(channel, 'high', None)
    low = getattr(channel, 'low', None)
    if high is not None and low is not None and close is not None and len(close) >= 14:
        atr_values = atr(high, low, close, period=14)
        current_atr = get_last_valid(atr_values, default=1.0)
        upper_line = getattr(channel, 'upper_line', None)
        lower_line = getattr(channel, 'lower_line', None)
        if upper_line is not None and lower_line is not None:
            channel_width = safe_float(upper_line[-1] - lower_line[-1], 0.0)
            features['spy_channel_width_atr_ratio'] = safe_divide(channel_width, current_atr, 0.0)
        else:
            features['spy_channel_width_atr_ratio'] = 0.0
    else:
        features['spy_channel_width_atr_ratio'] = 0.0

    # ==========================================================================
    # 9. spy_bounce_count
    # ==========================================================================
    features['spy_bounce_count'] = safe_float(getattr(channel, 'bounce_count', 0), 0.0)

    # ==========================================================================
    # 10. spy_complete_cycles
    # ==========================================================================
    features['spy_complete_cycles'] = safe_float(getattr(channel, 'complete_cycles', 0), 0.0)

    # ==========================================================================
    # 11. spy_upper_touches
    # ==========================================================================
    features['spy_upper_touches'] = safe_float(getattr(channel, 'upper_touches', 0), 0.0)

    # ==========================================================================
    # 12. spy_lower_touches
    # ==========================================================================
    features['spy_lower_touches'] = safe_float(getattr(channel, 'lower_touches', 0), 0.0)

    # ==========================================================================
    # 13. spy_alternation_ratio
    # ==========================================================================
    features['spy_alternation_ratio'] = safe_float(getattr(channel, 'alternation_ratio', 0.0), 0.0)

    # ==========================================================================
    # 14. spy_quality_score (r_squared * bounce_count normalized)
    # ==========================================================================
    features['spy_quality_score'] = safe_float(getattr(channel, 'quality_score', 0.0), 0.0)

    # ==========================================================================
    # 15. spy_channel_age_bars
    # ==========================================================================
    channel_window = getattr(channel, 'window', window)
    features['spy_channel_age_bars'] = safe_float(channel_window, float(window))

    # ==========================================================================
    # 16. spy_channel_trend_strength (slope * r_squared)
    # ==========================================================================
    r_squared = features['spy_channel_r_squared']
    features['spy_channel_trend_strength'] = safe_float(slope * r_squared, 0.0)

    # ==========================================================================
    # 17-19. spy_bars_since_last_touch, spy_bars_since_upper_touch, spy_bars_since_lower_touch
    # ==========================================================================
    touches = getattr(channel, 'touches', [])
    bars_since_last = safe_float(getattr(channel, 'bars_since_last_touch', window), float(window))
    features['spy_bars_since_last_touch'] = bars_since_last

    # Calculate bars since upper and lower touches
    bars_since_upper = safe_float(window, float(window))
    bars_since_lower = safe_float(window, float(window))

    if touches:
        for touch in reversed(touches):
            touch_type = getattr(touch, 'touch_type', None)
            bar_idx = getattr(touch, 'bar_index', 0)
            bars_since = window - 1 - bar_idx

            if touch_type is not None:
                # TouchType.UPPER = 1, TouchType.LOWER = 0
                if int(touch_type) == 1 and bars_since_upper == window:
                    bars_since_upper = bars_since
                elif int(touch_type) == 0 and bars_since_lower == window:
                    bars_since_lower = bars_since

            if bars_since_upper < window and bars_since_lower < window:
                break

    features['spy_bars_since_upper_touch'] = safe_float(bars_since_upper, float(window))
    features['spy_bars_since_lower_touch'] = safe_float(bars_since_lower, float(window))

    # ==========================================================================
    # 20. spy_touch_velocity (bounces per bar)
    # ==========================================================================
    bounce_count = features['spy_bounce_count']
    features['spy_touch_velocity'] = safe_divide(bounce_count, window, 0.0)

    # ==========================================================================
    # 21. spy_last_touch_type (0=lower, 1=upper)
    # ==========================================================================
    last_touch = getattr(channel, 'last_touch', None)
    if last_touch is not None:
        features['spy_last_touch_type'] = safe_float(int(last_touch), 0.0)
    elif touches:
        last_t = touches[-1]
        features['spy_last_touch_type'] = safe_float(int(getattr(last_t, 'touch_type', 0)), 0.0)
    else:
        features['spy_last_touch_type'] = 0.0

    # ==========================================================================
    # 22. spy_consecutive_same_touches
    # ==========================================================================
    consecutive = 0
    if touches:
        last_type = None
        for touch in reversed(touches):
            touch_type = getattr(touch, 'touch_type', None)
            if last_type is None:
                last_type = touch_type
                consecutive = 1
            elif touch_type == last_type:
                consecutive += 1
            else:
                break
    features['spy_consecutive_same_touches'] = safe_float(consecutive, 0.0)

    # ==========================================================================
    # 23. spy_channel_maturity (bounces / window)
    # ==========================================================================
    features['spy_channel_maturity'] = safe_divide(bounce_count, window, 0.0)

    # ==========================================================================
    # 24. spy_position_in_channel (0=floor, 1=ceiling)
    # ==========================================================================
    if hasattr(channel, 'position_at'):
        try:
            position = channel.position_at(-1)
            features['spy_position_in_channel'] = safe_float(position, 0.5)
        except Exception:
            features['spy_position_in_channel'] = 0.5
    else:
        features['spy_position_in_channel'] = 0.5

    # ==========================================================================
    # 25. spy_distance_to_upper_pct
    # ==========================================================================
    if hasattr(channel, 'distance_to_upper'):
        try:
            dist_upper = channel.distance_to_upper(-1)
            features['spy_distance_to_upper_pct'] = safe_float(dist_upper, 0.0)
        except Exception:
            features['spy_distance_to_upper_pct'] = 0.0
    else:
        features['spy_distance_to_upper_pct'] = 0.0

    # ==========================================================================
    # 26. spy_distance_to_lower_pct
    # ==========================================================================
    if hasattr(channel, 'distance_to_lower'):
        try:
            dist_lower = channel.distance_to_lower(-1)
            features['spy_distance_to_lower_pct'] = safe_float(dist_lower, 0.0)
        except Exception:
            features['spy_distance_to_lower_pct'] = 0.0
    else:
        features['spy_distance_to_lower_pct'] = 0.0

    # ==========================================================================
    # 27. spy_price_vs_channel_midpoint
    # ==========================================================================
    upper_line = getattr(channel, 'upper_line', None)
    lower_line = getattr(channel, 'lower_line', None)
    center_line = getattr(channel, 'center_line', None)

    if close is not None and center_line is not None and len(close) > 0:
        current_price = safe_float(close[-1], 0.0)
        center_price = safe_float(center_line[-1], current_price)
        # Positive = above midpoint, negative = below
        features['spy_price_vs_channel_midpoint'] = safe_divide(
            current_price - center_price, center_price, 0.0
        ) * 100
    else:
        features['spy_price_vs_channel_midpoint'] = 0.0

    # ==========================================================================
    # 28. spy_channel_momentum (slope change - estimated from regression)
    # ==========================================================================
    # Estimate momentum by comparing slope to what it would be with fewer bars
    if close is not None and len(close) >= 10:
        half_window = len(close) // 2
        close_half = close[half_window:]

        if len(close_half) >= 5:
            try:
                slope_half, _ = np.polyfit(np.arange(len(close_half)), close_half, 1)
                features['spy_channel_momentum'] = safe_float(slope - slope_half, 0.0)
            except Exception:
                features['spy_channel_momentum'] = 0.0
        else:
            features['spy_channel_momentum'] = 0.0
    else:
        features['spy_channel_momentum'] = 0.0

    # ==========================================================================
    # 29. spy_upper_line_slope
    # ==========================================================================
    if upper_line is not None and len(upper_line) >= 2:
        upper_slope = safe_float(upper_line[-1] - upper_line[0], 0.0) / max(len(upper_line) - 1, 1)
        features['spy_upper_line_slope'] = safe_float(upper_slope, 0.0)
    else:
        features['spy_upper_line_slope'] = 0.0

    # ==========================================================================
    # 30. spy_lower_line_slope
    # ==========================================================================
    if lower_line is not None and len(lower_line) >= 2:
        lower_slope = safe_float(lower_line[-1] - lower_line[0], 0.0) / max(len(lower_line) - 1, 1)
        features['spy_lower_line_slope'] = safe_float(lower_slope, 0.0)
    else:
        features['spy_lower_line_slope'] = 0.0

    # ==========================================================================
    # 31. spy_channel_expanding (1 if width increasing)
    # ==========================================================================
    if upper_line is not None and lower_line is not None and len(upper_line) >= 10:
        width_start = (upper_line[0] - lower_line[0])
        width_end = (upper_line[-1] - lower_line[-1])
        features['spy_channel_expanding'] = 1.0 if width_end > width_start * 1.05 else 0.0
    else:
        features['spy_channel_expanding'] = 0.0

    # ==========================================================================
    # 32. spy_channel_contracting (1 if width decreasing)
    # ==========================================================================
    if upper_line is not None and lower_line is not None and len(upper_line) >= 10:
        width_start = (upper_line[0] - lower_line[0])
        width_end = (upper_line[-1] - lower_line[-1])
        features['spy_channel_contracting'] = 1.0 if width_end < width_start * 0.95 else 0.0
    else:
        features['spy_channel_contracting'] = 0.0

    # ==========================================================================
    # 33. spy_std_dev_ratio (std_dev / avg_price)
    # ==========================================================================
    std_dev = getattr(channel, 'std_dev', 0.0)
    if close is not None and len(close) > 0:
        avg_price = safe_float(np.mean(close), 1.0)
        features['spy_std_dev_ratio'] = safe_divide(std_dev, avg_price, 0.0)
    else:
        features['spy_std_dev_ratio'] = 0.0

    # ==========================================================================
    # 34. spy_breakout_pressure_up
    # ==========================================================================
    # Pressure = how close/frequent price approaches upper boundary
    if high is not None and upper_line is not None and len(high) >= 5:
        recent_high = high[-5:]
        recent_upper = upper_line[-5:]
        distances_to_upper = []
        for h, u in zip(recent_high, recent_upper):
            if u > 0:
                dist = safe_divide(u - h, u, 0.0)
                distances_to_upper.append(max(0.0, dist))
        if distances_to_upper:
            # Lower average distance = more pressure (closer to boundary)
            avg_dist = safe_float(np.mean(distances_to_upper), 1.0)
            features['spy_breakout_pressure_up'] = safe_float(1.0 - avg_dist, 0.0)
        else:
            features['spy_breakout_pressure_up'] = 0.0
    else:
        features['spy_breakout_pressure_up'] = 0.0

    # ==========================================================================
    # 35. spy_breakout_pressure_down
    # ==========================================================================
    if low is not None and lower_line is not None and len(low) >= 5:
        recent_low = low[-5:]
        recent_lower = lower_line[-5:]
        distances_to_lower = []
        for l, lb in zip(recent_low, recent_lower):
            if l > 0:
                dist = safe_divide(l - lb, l, 0.0)
                distances_to_lower.append(max(0.0, dist))
        if distances_to_lower:
            avg_dist = safe_float(np.mean(distances_to_lower), 1.0)
            features['spy_breakout_pressure_down'] = safe_float(1.0 - avg_dist, 0.0)
        else:
            features['spy_breakout_pressure_down'] = 0.0
    else:
        features['spy_breakout_pressure_down'] = 0.0

    # ==========================================================================
    # 36. spy_channel_symmetry (how balanced are upper/lower touches)
    # ==========================================================================
    upper_t = features['spy_upper_touches']
    lower_t = features['spy_lower_touches']
    total_touches = upper_t + lower_t
    if total_touches > 0:
        # Perfect symmetry = 1.0 (equal touches), asymmetry approaches 0
        min_touches = min(upper_t, lower_t)
        max_touches = max(upper_t, lower_t)
        features['spy_channel_symmetry'] = safe_divide(min_touches, max_touches, 0.0)
    else:
        features['spy_channel_symmetry'] = 0.0

    # ==========================================================================
    # 37. spy_touch_regularity (std dev of intervals between touches)
    # ==========================================================================
    if len(touches) >= 3:
        intervals = []
        for i in range(1, len(touches)):
            interval = getattr(touches[i], 'bar_index', 0) - getattr(touches[i - 1], 'bar_index', 0)
            intervals.append(interval)
        if intervals:
            avg_interval = safe_float(np.mean(intervals), 1.0)
            std_interval = safe_float(np.std(intervals), 0.0)
            # Lower std relative to mean = more regular (0 = perfect, higher = irregular)
            # Invert so higher = more regular
            regularity = 1.0 - safe_divide(std_interval, avg_interval + 1, 0.0)
            features['spy_touch_regularity'] = safe_float(max(0.0, regularity), 0.0)
        else:
            features['spy_touch_regularity'] = 0.0
    else:
        features['spy_touch_regularity'] = 0.0

    # ==========================================================================
    # 38. spy_recent_touch_bias (bias toward upper or lower in recent touches)
    # ==========================================================================
    if len(touches) >= 3:
        recent_touches = touches[-min(5, len(touches)):]
        recent_upper = sum(1 for t in recent_touches if getattr(t, 'touch_type', 0) == 1)
        recent_lower = len(recent_touches) - recent_upper
        # -1 = all lower, 0 = balanced, 1 = all upper
        features['spy_recent_touch_bias'] = safe_divide(
            recent_upper - recent_lower, len(recent_touches), 0.0
        )
    else:
        features['spy_recent_touch_bias'] = 0.0

    # ==========================================================================
    # 39. spy_channel_curvature (non-linearity measure)
    # ==========================================================================
    if close is not None and len(close) >= 10:
        try:
            x = np.arange(len(close))
            # Fit quadratic and measure curvature coefficient
            coeffs = np.polyfit(x, close, 2)
            curvature = coeffs[0]  # Coefficient of x^2
            # Normalize by price level
            avg_price = safe_float(np.mean(close), 1.0)
            features['spy_channel_curvature'] = safe_divide(curvature, avg_price, 0.0) * 1000
        except Exception:
            features['spy_channel_curvature'] = 0.0
    else:
        features['spy_channel_curvature'] = 0.0

    # ==========================================================================
    # 40. spy_parallel_score (how parallel are upper and lower lines)
    # ==========================================================================
    upper_slope_val = features['spy_upper_line_slope']
    lower_slope_val = features['spy_lower_line_slope']
    avg_slope = safe_divide(upper_slope_val + lower_slope_val, 2.0, 0.0)
    if avg_slope != 0:
        slope_diff = abs(upper_slope_val - lower_slope_val)
        # Lower difference = more parallel (1.0 = perfectly parallel)
        features['spy_parallel_score'] = safe_float(
            1.0 - safe_divide(slope_diff, abs(avg_slope) + 0.0001, 0.0), 0.5
        )
    else:
        features['spy_parallel_score'] = 1.0 if upper_slope_val == lower_slope_val else 0.5

    # ==========================================================================
    # 41-50. Additional derived metrics
    # ==========================================================================

    # 41. spy_touch_density (touches per unit channel width)
    total_touches = features['spy_upper_touches'] + features['spy_lower_touches']
    channel_width_pct = features['spy_channel_width_pct']
    features['spy_touch_density'] = safe_divide(total_touches, channel_width_pct + 1.0, 0.0)

    # 42. spy_bounce_efficiency (complete_cycles / total touches)
    complete_cycles = features['spy_complete_cycles']
    features['spy_bounce_efficiency'] = safe_divide(complete_cycles, total_touches + 1.0, 0.0)

    # 43. spy_channel_stability (r_squared * alternation_ratio)
    r_sq = features['spy_channel_r_squared']
    alt_ratio = features['spy_alternation_ratio']
    features['spy_channel_stability'] = safe_float(r_sq * alt_ratio, 0.0)

    # 44. spy_momentum_direction_alignment (1 if momentum matches direction)
    momentum = features['spy_channel_momentum']
    direction_val = features['spy_channel_direction']
    if direction_val == 2:  # Bull
        features['spy_momentum_direction_alignment'] = 1.0 if momentum > 0 else 0.0
    elif direction_val == 0:  # Bear
        features['spy_momentum_direction_alignment'] = 1.0 if momentum < 0 else 0.0
    else:  # Sideways
        features['spy_momentum_direction_alignment'] = 1.0 if abs(momentum) < 0.01 else 0.5

    # 45. spy_price_position_extreme (how close to boundaries)
    position = features['spy_position_in_channel']
    features['spy_price_position_extreme'] = safe_float(abs(position - 0.5) * 2.0, 0.0)

    # 46. spy_breakout_imminence (combined pressure score)
    pressure_up = features['spy_breakout_pressure_up']
    pressure_down = features['spy_breakout_pressure_down']
    features['spy_breakout_imminence'] = safe_float(max(pressure_up, pressure_down), 0.0)

    # 47. spy_breakout_direction_bias (positive = up, negative = down)
    features['spy_breakout_direction_bias'] = safe_float(pressure_up - pressure_down, 0.0)

    # 48. spy_channel_health_score (composite quality metric)
    health = (
        features['spy_channel_valid'] * 0.2 +
        features['spy_channel_stability'] * 0.3 +
        features['spy_parallel_score'] * 0.2 +
        features['spy_touch_regularity'] * 0.15 +
        features['spy_channel_symmetry'] * 0.15
    )
    features['spy_channel_health_score'] = safe_float(health, 0.0)

    # 49. spy_time_weighted_position (position weighted by time since last touch)
    time_factor = safe_divide(bars_since_last, window, 1.0)
    features['spy_time_weighted_position'] = safe_float(position * (1 - time_factor), 0.0)

    # 50. spy_volatility_adjusted_width (width relative to recent volatility)
    if features['spy_channel_width_atr_ratio'] > 0:
        # Normalize: 1.0 = average, >1 = wide, <1 = narrow
        features['spy_volatility_adjusted_width'] = safe_float(
            features['spy_channel_width_atr_ratio'] / 4.0, 1.0  # Typical ATR ratio ~4
        )
    else:
        features['spy_volatility_adjusted_width'] = 1.0

    # ==========================================================================
    # 51-58. EXCURSION FEATURES - Price movements beyond channel boundaries
    # ==========================================================================
    excursion_features = _extract_excursion_features(
        spy_df=spy_df,
        channel=channel,
        window=window,
        high=high,
        low=low,
        close=close,
        upper_line=upper_line,
        lower_line=lower_line
    )
    features.update(excursion_features)

    # Final safety check
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _extract_excursion_features(
    spy_df: pd.DataFrame,
    channel: "Channel",
    window: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    upper_line: np.ndarray,
    lower_line: np.ndarray
) -> Dict[str, float]:
    """
    Extract 8 excursion features tracking price movements beyond channel boundaries.

    Excursions are temporary price movements outside the channel that may or may not
    return back inside the channel.

    Args:
        spy_df: SPY DataFrame
        channel: Channel object
        window: Window size
        high: High prices array
        low: Low prices array
        close: Close prices array
        upper_line: Upper channel boundary
        lower_line: Lower channel boundary

    Returns:
        Dict with 8 excursion features prefixed with 'spy_'
    """
    features: Dict[str, float] = {}

    # Default values
    features['spy_excursions_above_upper'] = 0.0
    features['spy_excursions_below_lower'] = 0.0
    features['spy_max_excursion_above_pct'] = 0.0
    features['spy_max_excursion_below_pct'] = 0.0
    features['spy_bars_since_last_excursion'] = float(window)
    features['spy_excursion_return_speed_avg'] = 0.0
    features['spy_excursion_rate'] = 0.0
    features['spy_last_excursion_direction'] = 0.0  # 0=none, 1=above, -1=below

    # Need valid arrays to compute excursions
    if (high is None or low is None or close is None or
        upper_line is None or lower_line is None):
        return features

    if len(high) == 0 or len(upper_line) == 0:
        return features

    n = min(len(high), len(low), len(close), len(upper_line), len(lower_line))
    if n < 3:
        return features

    # Truncate to same length
    high = high[:n]
    low = low[:n]
    close = close[:n]
    upper_line = upper_line[:n]
    lower_line = lower_line[:n]

    # Track excursions
    excursions_above = []  # List of (start_bar, max_pct, return_bars)
    excursions_below = []  # List of (start_bar, max_pct, return_bars)

    in_excursion_above = False
    in_excursion_below = False
    excursion_start = 0
    max_excursion_pct = 0.0

    for i in range(n):
        # Check if HIGH is above upper line (excursion above)
        if high[i] > upper_line[i]:
            if not in_excursion_above:
                # Starting new excursion above
                in_excursion_above = True
                excursion_start = i
                max_excursion_pct = 0.0

            # Track max excursion percentage
            excursion_pct = safe_divide(high[i] - upper_line[i], upper_line[i], 0.0) * 100
            max_excursion_pct = max(max_excursion_pct, excursion_pct)

        elif in_excursion_above:
            # Price returned inside channel
            return_bars = i - excursion_start
            excursions_above.append((excursion_start, max_excursion_pct, return_bars))
            in_excursion_above = False

        # Check if LOW is below lower line (excursion below)
        if low[i] < lower_line[i]:
            if not in_excursion_below:
                # Starting new excursion below
                in_excursion_below = True
                excursion_start = i
                max_excursion_pct = 0.0

            # Track max excursion percentage
            excursion_pct = safe_divide(lower_line[i] - low[i], lower_line[i], 0.0) * 100
            max_excursion_pct = max(max_excursion_pct, excursion_pct)

        elif in_excursion_below:
            # Price returned inside channel
            return_bars = i - excursion_start
            excursions_below.append((excursion_start, max_excursion_pct, return_bars))
            in_excursion_below = False

    # Handle ongoing excursions at end of window
    if in_excursion_above:
        return_bars = n - excursion_start
        excursions_above.append((excursion_start, max_excursion_pct, return_bars))

    if in_excursion_below:
        return_bars = n - excursion_start
        excursions_below.append((excursion_start, max_excursion_pct, return_bars))

    # 51. spy_excursions_above_upper - count of excursions above upper boundary
    features['spy_excursions_above_upper'] = float(len(excursions_above))

    # 52. spy_excursions_below_lower - count of excursions below lower boundary
    features['spy_excursions_below_lower'] = float(len(excursions_below))

    # 53. spy_max_excursion_above_pct - maximum percentage excursion above
    if excursions_above:
        max_above = max(e[1] for e in excursions_above)
        features['spy_max_excursion_above_pct'] = safe_float(max_above, 0.0)

    # 54. spy_max_excursion_below_pct - maximum percentage excursion below
    if excursions_below:
        max_below = max(e[1] for e in excursions_below)
        features['spy_max_excursion_below_pct'] = safe_float(max_below, 0.0)

    # 55. spy_bars_since_last_excursion - bars since any excursion ended
    all_excursions = excursions_above + excursions_below
    if all_excursions:
        # Find most recent excursion end
        last_excursion_end = max(e[0] + e[2] for e in all_excursions)
        bars_since = n - 1 - last_excursion_end
        features['spy_bars_since_last_excursion'] = safe_float(max(0, bars_since), float(window))

    # 56. spy_excursion_return_speed_avg - average bars to return from excursion
    all_return_bars = [e[2] for e in all_excursions]
    if all_return_bars:
        avg_return = np.mean(all_return_bars)
        features['spy_excursion_return_speed_avg'] = safe_float(avg_return, 0.0)

    # 57. spy_excursion_rate - excursions per bar (total excursions / window)
    total_excursions = len(excursions_above) + len(excursions_below)
    features['spy_excursion_rate'] = safe_divide(total_excursions, window, 0.0)

    # 58. spy_last_excursion_direction - direction of most recent excursion
    # 0 = none, 1 = above, -1 = below
    if all_excursions:
        # Find the most recent excursion by start bar
        last_above_start = max((e[0] for e in excursions_above), default=-1)
        last_below_start = max((e[0] for e in excursions_below), default=-1)

        if last_above_start > last_below_start:
            features['spy_last_excursion_direction'] = 1.0
        elif last_below_start > last_above_start:
            features['spy_last_excursion_direction'] = -1.0
        else:
            features['spy_last_excursion_direction'] = 0.0

    return features


def _get_default_features() -> Dict[str, float]:
    """Return default feature values for invalid/None channel."""
    return {
        # 50 base features
        'spy_channel_valid': 0.0,
        'spy_channel_direction': 1.0,
        'spy_channel_slope': 0.0,
        'spy_channel_slope_normalized': 0.0,
        'spy_channel_intercept': 0.0,
        'spy_channel_r_squared': 0.0,
        'spy_channel_width_pct': 0.0,
        'spy_channel_width_atr_ratio': 0.0,
        'spy_bounce_count': 0.0,
        'spy_complete_cycles': 0.0,
        'spy_upper_touches': 0.0,
        'spy_lower_touches': 0.0,
        'spy_alternation_ratio': 0.0,
        'spy_quality_score': 0.0,
        'spy_channel_age_bars': 50.0,
        'spy_channel_trend_strength': 0.0,
        'spy_bars_since_last_touch': 50.0,
        'spy_bars_since_upper_touch': 50.0,
        'spy_bars_since_lower_touch': 50.0,
        'spy_touch_velocity': 0.0,
        'spy_last_touch_type': 0.0,
        'spy_consecutive_same_touches': 0.0,
        'spy_channel_maturity': 0.0,
        'spy_position_in_channel': 0.5,
        'spy_distance_to_upper_pct': 0.0,
        'spy_distance_to_lower_pct': 0.0,
        'spy_price_vs_channel_midpoint': 0.0,
        'spy_channel_momentum': 0.0,
        'spy_upper_line_slope': 0.0,
        'spy_lower_line_slope': 0.0,
        'spy_channel_expanding': 0.0,
        'spy_channel_contracting': 0.0,
        'spy_std_dev_ratio': 0.0,
        'spy_breakout_pressure_up': 0.0,
        'spy_breakout_pressure_down': 0.0,
        'spy_channel_symmetry': 0.0,
        'spy_touch_regularity': 0.0,
        'spy_recent_touch_bias': 0.0,
        'spy_channel_curvature': 0.0,
        'spy_parallel_score': 0.5,
        'spy_touch_density': 0.0,
        'spy_bounce_efficiency': 0.0,
        'spy_channel_stability': 0.0,
        'spy_momentum_direction_alignment': 0.5,
        'spy_price_position_extreme': 0.0,
        'spy_breakout_imminence': 0.0,
        'spy_breakout_direction_bias': 0.0,
        'spy_channel_health_score': 0.0,
        'spy_time_weighted_position': 0.0,
        'spy_volatility_adjusted_width': 1.0,
        # 8 excursion features
        'spy_excursions_above_upper': 0.0,
        'spy_excursions_below_lower': 0.0,
        'spy_max_excursion_above_pct': 0.0,
        'spy_max_excursion_below_pct': 0.0,
        'spy_bars_since_last_excursion': 50.0,
        'spy_excursion_return_speed_avg': 0.0,
        'spy_excursion_rate': 0.0,
        'spy_last_excursion_direction': 0.0,
    }


def get_spy_channel_feature_names() -> List[str]:
    """Get ordered list of all SPY channel feature names."""
    return list(_get_default_features().keys())


def get_spy_channel_feature_count() -> int:
    """Get total number of SPY channel features."""
    return len(_get_default_features())  # 58


def extract_spy_channel_features_tf(
    spy_df: pd.DataFrame,
    channel: "Channel",
    tf: str,
    window: int
) -> Dict[str, float]:
    """
    Extract SPY channel features with TF and window prefix.

    Args:
        spy_df: SPY DataFrame with OHLC data
        channel: Channel object detected on SPY data
        tf: Timeframe name (e.g., 'daily', '1h')
        window: Window size (e.g., 50)

    Returns:
        Dict with keys like 'daily_w50_spy_channel_slope', 'daily_w50_spy_position_in_channel'
    """
    base_features = extract_spy_channel_features(spy_df, channel, window, tf)
    prefix = f"{tf}_w{window}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_spy_channel_feature_names_tf(tf: str, window: int) -> List[str]:
    """Get feature names with TF and window prefix."""
    base_names = get_spy_channel_feature_names()
    prefix = f"{tf}_w{window}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_spy_channel_feature_names() -> List[str]:
    """Get ALL SPY channel feature names across all TFs and windows."""
    from v15.config import TIMEFRAMES, STANDARD_WINDOWS

    all_names = []
    for tf in TIMEFRAMES:
        for window in STANDARD_WINDOWS:
            all_names.extend(get_spy_channel_feature_names_tf(tf, window))
    return all_names


def get_total_spy_channel_features() -> int:
    """Total SPY channel features: 58 * 8 windows * 10 TFs = 4,640"""
    return 58 * 8 * 10
