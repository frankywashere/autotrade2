"""
v15/features/tsla_channel.py - TSLA Channel Feature Extraction

Extracts 50 channel-specific features from a Channel object.
All features return valid floats with safe defaults (no NaN, no Inf).

Supports two extraction modes:
1. Base extraction: extract_tsla_channel_features(channel) -> 50 features
2. TF-prefixed extraction: extract_tsla_channel_features_tf(channel, tf, window)
   -> 50 features with keys like 'daily_w50_channel_slope'

Total features across all TFs and windows: 50 * 8 windows * 11 TFs = 4,400
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, TYPE_CHECKING

from .utils import safe_float, safe_divide, safe_pct_change, get_last_valid, atr

if TYPE_CHECKING:
    from v7.core.channel import Channel


def extract_tsla_channel_features(channel: "Channel") -> Dict[str, float]:
    """
    Extract 50 TSLA channel features from a Channel object.

    Args:
        channel: Channel object from v7.core.channel

    Returns:
        Dict[str, float] with 50 features, all guaranteed to be valid floats
    """
    features: Dict[str, float] = {}

    # Handle None or invalid channel
    if channel is None:
        return _get_default_features()

    is_valid = getattr(channel, 'valid', False)

    # ==========================================================================
    # 1. channel_valid (0/1)
    # ==========================================================================
    features['channel_valid'] = 1.0 if is_valid else 0.0

    # ==========================================================================
    # 2. channel_direction (0=bear, 1=sideways, 2=bull)
    # ==========================================================================
    direction = getattr(channel, 'direction', 1)
    features['channel_direction'] = safe_float(int(direction), 1.0)

    # ==========================================================================
    # 3. channel_slope
    # ==========================================================================
    slope = getattr(channel, 'slope', 0.0)
    features['channel_slope'] = safe_float(slope, 0.0)

    # ==========================================================================
    # 4. channel_slope_normalized (slope / price level)
    # ==========================================================================
    close = getattr(channel, 'close', None)
    if close is not None and len(close) > 0:
        avg_price = safe_float(np.mean(close), 1.0)
        features['channel_slope_normalized'] = safe_divide(slope, avg_price, 0.0)
    else:
        features['channel_slope_normalized'] = 0.0

    # ==========================================================================
    # 5. channel_intercept
    # ==========================================================================
    features['channel_intercept'] = safe_float(getattr(channel, 'intercept', 0.0), 0.0)

    # ==========================================================================
    # 6. channel_r_squared
    # ==========================================================================
    features['channel_r_squared'] = safe_float(getattr(channel, 'r_squared', 0.0), 0.0)

    # ==========================================================================
    # 7. channel_width_pct
    # ==========================================================================
    features['channel_width_pct'] = safe_float(getattr(channel, 'width_pct', 0.0), 0.0)

    # ==========================================================================
    # 8. channel_width_atr_ratio
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
            features['channel_width_atr_ratio'] = safe_divide(channel_width, current_atr, 0.0)
        else:
            features['channel_width_atr_ratio'] = 0.0
    else:
        features['channel_width_atr_ratio'] = 0.0

    # ==========================================================================
    # 9. bounce_count
    # ==========================================================================
    features['bounce_count'] = safe_float(getattr(channel, 'bounce_count', 0), 0.0)

    # ==========================================================================
    # 10. complete_cycles
    # ==========================================================================
    features['complete_cycles'] = safe_float(getattr(channel, 'complete_cycles', 0), 0.0)

    # ==========================================================================
    # 11. upper_touches
    # ==========================================================================
    features['upper_touches'] = safe_float(getattr(channel, 'upper_touches', 0), 0.0)

    # ==========================================================================
    # 12. lower_touches
    # ==========================================================================
    features['lower_touches'] = safe_float(getattr(channel, 'lower_touches', 0), 0.0)

    # ==========================================================================
    # 13. alternation_ratio
    # ==========================================================================
    features['alternation_ratio'] = safe_float(getattr(channel, 'alternation_ratio', 0.0), 0.0)

    # ==========================================================================
    # 14. quality_score (r_squared * bounce_count normalized)
    # ==========================================================================
    features['quality_score'] = safe_float(getattr(channel, 'quality_score', 0.0), 0.0)

    # ==========================================================================
    # 15. channel_age_bars
    # ==========================================================================
    window = getattr(channel, 'window', 50)
    features['channel_age_bars'] = safe_float(window, 50.0)

    # ==========================================================================
    # 16. channel_trend_strength (slope * r_squared)
    # ==========================================================================
    r_squared = features['channel_r_squared']
    features['channel_trend_strength'] = safe_float(slope * r_squared, 0.0)

    # ==========================================================================
    # 17-19. bars_since_last_touch, bars_since_upper_touch, bars_since_lower_touch
    # ==========================================================================
    touches = getattr(channel, 'touches', [])
    bars_since_last = safe_float(getattr(channel, 'bars_since_last_touch', window), window)
    features['bars_since_last_touch'] = bars_since_last

    # Calculate bars since upper and lower touches
    bars_since_upper = safe_float(window, window)
    bars_since_lower = safe_float(window, window)

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

    features['bars_since_upper_touch'] = safe_float(bars_since_upper, window)
    features['bars_since_lower_touch'] = safe_float(bars_since_lower, window)

    # ==========================================================================
    # 20. touch_velocity (bounces per bar)
    # ==========================================================================
    bounce_count = features['bounce_count']
    features['touch_velocity'] = safe_divide(bounce_count, window, 0.0)

    # ==========================================================================
    # 21. last_touch_type (0=lower, 1=upper)
    # ==========================================================================
    last_touch = getattr(channel, 'last_touch', None)
    if last_touch is not None:
        features['last_touch_type'] = safe_float(int(last_touch), 0.0)
    elif touches:
        last_t = touches[-1]
        features['last_touch_type'] = safe_float(int(getattr(last_t, 'touch_type', 0)), 0.0)
    else:
        features['last_touch_type'] = 0.0

    # ==========================================================================
    # 22. consecutive_same_touches
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
    features['consecutive_same_touches'] = safe_float(consecutive, 0.0)

    # ==========================================================================
    # 23. channel_maturity (bounces / window)
    # ==========================================================================
    features['channel_maturity'] = safe_divide(bounce_count, window, 0.0)

    # ==========================================================================
    # 24. position_in_channel (0=floor, 1=ceiling)
    # ==========================================================================
    if hasattr(channel, 'position_at'):
        try:
            position = channel.position_at(-1)
            features['position_in_channel'] = safe_float(position, 0.5)
        except Exception:
            features['position_in_channel'] = 0.5
    else:
        features['position_in_channel'] = 0.5

    # ==========================================================================
    # 25. distance_to_upper_pct
    # ==========================================================================
    if hasattr(channel, 'distance_to_upper'):
        try:
            dist_upper = channel.distance_to_upper(-1)
            features['distance_to_upper_pct'] = safe_float(dist_upper, 0.0)
        except Exception:
            features['distance_to_upper_pct'] = 0.0
    else:
        features['distance_to_upper_pct'] = 0.0

    # ==========================================================================
    # 26. distance_to_lower_pct
    # ==========================================================================
    if hasattr(channel, 'distance_to_lower'):
        try:
            dist_lower = channel.distance_to_lower(-1)
            features['distance_to_lower_pct'] = safe_float(dist_lower, 0.0)
        except Exception:
            features['distance_to_lower_pct'] = 0.0
    else:
        features['distance_to_lower_pct'] = 0.0

    # ==========================================================================
    # 27. price_vs_channel_midpoint
    # ==========================================================================
    upper_line = getattr(channel, 'upper_line', None)
    lower_line = getattr(channel, 'lower_line', None)
    center_line = getattr(channel, 'center_line', None)

    if close is not None and center_line is not None and len(close) > 0:
        current_price = safe_float(close[-1], 0.0)
        center_price = safe_float(center_line[-1], current_price)
        # Positive = above midpoint, negative = below
        features['price_vs_channel_midpoint'] = safe_divide(
            current_price - center_price, center_price, 0.0
        ) * 100
    else:
        features['price_vs_channel_midpoint'] = 0.0

    # ==========================================================================
    # 28. channel_momentum (slope change - estimated from regression)
    # ==========================================================================
    # Estimate momentum by comparing slope to what it would be with fewer bars
    if close is not None and len(close) >= 10:
        half_window = len(close) // 2
        x_full = np.arange(len(close))
        x_half = np.arange(half_window, len(close))
        close_half = close[half_window:]

        if len(close_half) >= 5:
            try:
                slope_half, _ = np.polyfit(np.arange(len(close_half)), close_half, 1)
                features['channel_momentum'] = safe_float(slope - slope_half, 0.0)
            except Exception:
                features['channel_momentum'] = 0.0
        else:
            features['channel_momentum'] = 0.0
    else:
        features['channel_momentum'] = 0.0

    # ==========================================================================
    # 29. upper_line_slope
    # ==========================================================================
    if upper_line is not None and len(upper_line) >= 2:
        upper_slope = safe_float(upper_line[-1] - upper_line[0], 0.0) / max(len(upper_line) - 1, 1)
        features['upper_line_slope'] = safe_float(upper_slope, 0.0)
    else:
        features['upper_line_slope'] = 0.0

    # ==========================================================================
    # 30. lower_line_slope
    # ==========================================================================
    if lower_line is not None and len(lower_line) >= 2:
        lower_slope = safe_float(lower_line[-1] - lower_line[0], 0.0) / max(len(lower_line) - 1, 1)
        features['lower_line_slope'] = safe_float(lower_slope, 0.0)
    else:
        features['lower_line_slope'] = 0.0

    # ==========================================================================
    # 31. channel_expanding (1 if width increasing)
    # ==========================================================================
    if upper_line is not None and lower_line is not None and len(upper_line) >= 10:
        width_start = (upper_line[0] - lower_line[0])
        width_end = (upper_line[-1] - lower_line[-1])
        features['channel_expanding'] = 1.0 if width_end > width_start * 1.05 else 0.0
    else:
        features['channel_expanding'] = 0.0

    # ==========================================================================
    # 32. channel_contracting (1 if width decreasing)
    # ==========================================================================
    if upper_line is not None and lower_line is not None and len(upper_line) >= 10:
        width_start = (upper_line[0] - lower_line[0])
        width_end = (upper_line[-1] - lower_line[-1])
        features['channel_contracting'] = 1.0 if width_end < width_start * 0.95 else 0.0
    else:
        features['channel_contracting'] = 0.0

    # ==========================================================================
    # 33. std_dev_ratio (std_dev / avg_price)
    # ==========================================================================
    std_dev = getattr(channel, 'std_dev', 0.0)
    if close is not None and len(close) > 0:
        avg_price = safe_float(np.mean(close), 1.0)
        features['std_dev_ratio'] = safe_divide(std_dev, avg_price, 0.0)
    else:
        features['std_dev_ratio'] = 0.0

    # ==========================================================================
    # 34. breakout_pressure_up
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
            features['breakout_pressure_up'] = safe_float(1.0 - avg_dist, 0.0)
        else:
            features['breakout_pressure_up'] = 0.0
    else:
        features['breakout_pressure_up'] = 0.0

    # ==========================================================================
    # 35. breakout_pressure_down
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
            features['breakout_pressure_down'] = safe_float(1.0 - avg_dist, 0.0)
        else:
            features['breakout_pressure_down'] = 0.0
    else:
        features['breakout_pressure_down'] = 0.0

    # ==========================================================================
    # 36. channel_symmetry (how balanced are upper/lower touches)
    # ==========================================================================
    upper_t = features['upper_touches']
    lower_t = features['lower_touches']
    total_touches = upper_t + lower_t
    if total_touches > 0:
        # Perfect symmetry = 1.0 (equal touches), asymmetry approaches 0
        min_touches = min(upper_t, lower_t)
        max_touches = max(upper_t, lower_t)
        features['channel_symmetry'] = safe_divide(min_touches, max_touches, 0.0)
    else:
        features['channel_symmetry'] = 0.0

    # ==========================================================================
    # 37. touch_regularity (std dev of intervals between touches)
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
            features['touch_regularity'] = safe_float(max(0.0, regularity), 0.0)
        else:
            features['touch_regularity'] = 0.0
    else:
        features['touch_regularity'] = 0.0

    # ==========================================================================
    # 38. recent_touch_bias (bias toward upper or lower in recent touches)
    # ==========================================================================
    if len(touches) >= 3:
        recent_touches = touches[-min(5, len(touches)):]
        recent_upper = sum(1 for t in recent_touches if getattr(t, 'touch_type', 0) == 1)
        recent_lower = len(recent_touches) - recent_upper
        # -1 = all lower, 0 = balanced, 1 = all upper
        features['recent_touch_bias'] = safe_divide(
            recent_upper - recent_lower, len(recent_touches), 0.0
        )
    else:
        features['recent_touch_bias'] = 0.0

    # ==========================================================================
    # 39. channel_curvature (non-linearity measure)
    # ==========================================================================
    if close is not None and len(close) >= 10:
        try:
            x = np.arange(len(close))
            # Fit quadratic and measure curvature coefficient
            coeffs = np.polyfit(x, close, 2)
            curvature = coeffs[0]  # Coefficient of x^2
            # Normalize by price level
            avg_price = safe_float(np.mean(close), 1.0)
            features['channel_curvature'] = safe_divide(curvature, avg_price, 0.0) * 1000
        except Exception:
            features['channel_curvature'] = 0.0
    else:
        features['channel_curvature'] = 0.0

    # ==========================================================================
    # 40. parallel_score (how parallel are upper and lower lines)
    # ==========================================================================
    upper_slope = features['upper_line_slope']
    lower_slope = features['lower_line_slope']
    avg_slope = safe_divide(upper_slope + lower_slope, 2.0, 0.0)
    if avg_slope != 0:
        slope_diff = abs(upper_slope - lower_slope)
        # Lower difference = more parallel (1.0 = perfectly parallel)
        features['parallel_score'] = safe_float(
            1.0 - safe_divide(slope_diff, abs(avg_slope) + 0.0001, 0.0), 0.5
        )
    else:
        features['parallel_score'] = 1.0 if upper_slope == lower_slope else 0.5

    # ==========================================================================
    # 41-50. Additional derived metrics
    # ==========================================================================

    # 41. touch_density (touches per unit channel width)
    total_touches = features['upper_touches'] + features['lower_touches']
    channel_width_pct = features['channel_width_pct']
    features['touch_density'] = safe_divide(total_touches, channel_width_pct + 1.0, 0.0)

    # 42. bounce_efficiency (complete_cycles / total touches)
    complete_cycles = features['complete_cycles']
    features['bounce_efficiency'] = safe_divide(complete_cycles, total_touches + 1.0, 0.0)

    # 43. channel_stability (r_squared * alternation_ratio)
    r_sq = features['channel_r_squared']
    alt_ratio = features['alternation_ratio']
    features['channel_stability'] = safe_float(r_sq * alt_ratio, 0.0)

    # 44. momentum_direction_alignment (1 if momentum matches direction)
    momentum = features['channel_momentum']
    direction = features['channel_direction']
    if direction == 2:  # Bull
        features['momentum_direction_alignment'] = 1.0 if momentum > 0 else 0.0
    elif direction == 0:  # Bear
        features['momentum_direction_alignment'] = 1.0 if momentum < 0 else 0.0
    else:  # Sideways
        features['momentum_direction_alignment'] = 1.0 if abs(momentum) < 0.01 else 0.5

    # 45. price_position_extreme (how close to boundaries)
    position = features['position_in_channel']
    features['price_position_extreme'] = safe_float(abs(position - 0.5) * 2.0, 0.0)

    # 46. breakout_imminence (combined pressure score)
    pressure_up = features['breakout_pressure_up']
    pressure_down = features['breakout_pressure_down']
    features['breakout_imminence'] = safe_float(max(pressure_up, pressure_down), 0.0)

    # 47. breakout_direction_bias (positive = up, negative = down)
    features['breakout_direction_bias'] = safe_float(pressure_up - pressure_down, 0.0)

    # 48. channel_health_score (composite quality metric)
    health = (
        features['channel_valid'] * 0.2 +
        features['channel_stability'] * 0.3 +
        features['parallel_score'] * 0.2 +
        features['touch_regularity'] * 0.15 +
        features['channel_symmetry'] * 0.15
    )
    features['channel_health_score'] = safe_float(health, 0.0)

    # 49. time_weighted_position (position weighted by time since last touch)
    time_factor = safe_divide(bars_since_last, window, 1.0)
    features['time_weighted_position'] = safe_float(position * (1 - time_factor), 0.0)

    # 50. volatility_adjusted_width (width relative to recent volatility)
    if features['channel_width_atr_ratio'] > 0:
        # Normalize: 1.0 = average, >1 = wide, <1 = narrow
        features['volatility_adjusted_width'] = safe_float(
            features['channel_width_atr_ratio'] / 4.0, 1.0  # Typical ATR ratio ~4
        )
    else:
        features['volatility_adjusted_width'] = 1.0

    # Final safety check
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _get_default_features() -> Dict[str, float]:
    """Return default feature values for invalid/None channel."""
    return {
        'channel_valid': 0.0,
        'channel_direction': 1.0,
        'channel_slope': 0.0,
        'channel_slope_normalized': 0.0,
        'channel_intercept': 0.0,
        'channel_r_squared': 0.0,
        'channel_width_pct': 0.0,
        'channel_width_atr_ratio': 0.0,
        'bounce_count': 0.0,
        'complete_cycles': 0.0,
        'upper_touches': 0.0,
        'lower_touches': 0.0,
        'alternation_ratio': 0.0,
        'quality_score': 0.0,
        'channel_age_bars': 50.0,
        'channel_trend_strength': 0.0,
        'bars_since_last_touch': 50.0,
        'bars_since_upper_touch': 50.0,
        'bars_since_lower_touch': 50.0,
        'touch_velocity': 0.0,
        'last_touch_type': 0.0,
        'consecutive_same_touches': 0.0,
        'channel_maturity': 0.0,
        'position_in_channel': 0.5,
        'distance_to_upper_pct': 0.0,
        'distance_to_lower_pct': 0.0,
        'price_vs_channel_midpoint': 0.0,
        'channel_momentum': 0.0,
        'upper_line_slope': 0.0,
        'lower_line_slope': 0.0,
        'channel_expanding': 0.0,
        'channel_contracting': 0.0,
        'std_dev_ratio': 0.0,
        'breakout_pressure_up': 0.0,
        'breakout_pressure_down': 0.0,
        'channel_symmetry': 0.0,
        'touch_regularity': 0.0,
        'recent_touch_bias': 0.0,
        'channel_curvature': 0.0,
        'parallel_score': 0.5,
        'touch_density': 0.0,
        'bounce_efficiency': 0.0,
        'channel_stability': 0.0,
        'momentum_direction_alignment': 0.5,
        'price_position_extreme': 0.0,
        'breakout_imminence': 0.0,
        'breakout_direction_bias': 0.0,
        'channel_health_score': 0.0,
        'time_weighted_position': 0.0,
        'volatility_adjusted_width': 1.0,
    }


def get_tsla_channel_feature_names() -> list:
    """Get ordered list of all TSLA channel feature names."""
    return list(_get_default_features().keys())


def get_tsla_channel_feature_count() -> int:
    """Get total number of TSLA channel features."""
    return len(_get_default_features())


def extract_tsla_channel_features_tf(
    channel: "Channel",
    tf: str,
    window: int
) -> Dict[str, float]:
    """
    Extract channel features with TF and window prefix.

    Args:
        channel: Channel object
        tf: Timeframe name (e.g., 'daily', '1h')
        window: Window size (e.g., 50)

    Returns:
        Dict with keys like 'daily_w50_channel_slope', 'daily_w50_position_in_channel'
    """
    base_features = extract_tsla_channel_features(channel)
    prefix = f"{tf}_w{window}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_tsla_channel_feature_names_tf(tf: str, window: int) -> List[str]:
    """Get feature names with TF and window prefix."""
    base_names = get_tsla_channel_feature_names()
    prefix = f"{tf}_w{window}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_tsla_channel_feature_names() -> List[str]:
    """Get ALL channel feature names across all TFs and windows."""
    from v7.core.timeframe import TIMEFRAMES
    STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]

    all_names = []
    for tf in TIMEFRAMES:
        for window in STANDARD_WINDOWS:
            all_names.extend(get_tsla_channel_feature_names_tf(tf, window))
    return all_names


def get_total_channel_features() -> int:
    """Total channel features: 50 * 8 windows * 11 TFs = 4,400"""
    return 50 * 8 * 11
