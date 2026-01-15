"""
Simplified Feature Extraction for V15

Extracts ~50-100 simple, robust features that are SAFE:
- No NaN values (defaults provided for all edge cases)
- No division by zero (guarded calculations)
- Flat dict output for easy model consumption

Features:
1. Channel Features (~12): direction, slope, r_squared, width, bounces, etc.
2. Price Position Features (~6): position in channel, RSI, momentum
3. Cross-Asset Features (~6): SPY correlation, VIX level/change
4. Window Scores (~24): 8 windows x 3 metrics each
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from v7.core.channel import Channel


# Standard window sizes for multi-window analysis
STANDARD_WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator or invalid result."""
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    if not np.isfinite(result):
        return default
    return float(result)


def safe_pct_change(current: float, previous: float, default: float = 0.0) -> float:
    """Safe percentage change calculation."""
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return default
    result = ((current - previous) / previous) * 100
    if not np.isfinite(result):
        return default
    return float(result)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default for invalid values."""
    try:
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def calculate_rsi_safe(prices: np.ndarray, period: int = 14) -> float:
    """
    Calculate RSI with safe handling of edge cases.

    Args:
        prices: Array of close prices (most recent last)
        period: RSI period (default 14)

    Returns:
        RSI value (0-100), defaults to 50.0 on error
    """
    if len(prices) < 2:
        return 50.0

    try:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use exponential moving average
        alpha = 1.0 / period

        # Initialize with available data
        available_bars = min(period, len(gains))
        avg_gain = np.mean(gains[:available_bars])
        avg_loss = np.mean(losses[:available_bars])

        # Exponential smoothing for remaining bars
        for i in range(available_bars, len(gains)):
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        if not np.isfinite(rsi):
            return 50.0

        return float(np.clip(rsi, 0.0, 100.0))

    except Exception:
        return 50.0


def calculate_correlation_safe(
    series1: np.ndarray,
    series2: np.ndarray,
    window: int = 20
) -> float:
    """
    Calculate rolling correlation between two series safely.

    Args:
        series1: First price series
        series2: Second price series
        window: Rolling window for correlation

    Returns:
        Correlation value (-1 to 1), defaults to 0.0 on error
    """
    try:
        min_len = min(len(series1), len(series2))
        if min_len < window:
            return 0.0

        s1 = series1[-window:]
        s2 = series2[-window:]

        # Check for zero variance
        if np.std(s1) == 0 or np.std(s2) == 0:
            return 0.0

        corr_matrix = np.corrcoef(s1, s2)
        correlation = float(corr_matrix[0, 1])

        if not np.isfinite(correlation):
            return 0.0

        return float(np.clip(correlation, -1.0, 1.0))

    except Exception:
        return 0.0


def calculate_momentum(prices: np.ndarray, lookback: int) -> float:
    """
    Calculate price momentum (percent change over lookback period).

    Args:
        prices: Price array
        lookback: Number of bars to look back

    Returns:
        Momentum as percentage change, defaults to 0.0
    """
    if len(prices) < lookback + 1:
        return 0.0

    current = prices[-1]
    past = prices[-(lookback + 1)]

    return safe_pct_change(current, past, default=0.0)


def extract_channel_features(channel: Optional[Channel]) -> Dict[str, float]:
    """
    Extract features from a Channel object.

    Args:
        channel: Channel object (can be None)

    Returns:
        Dict of channel features with safe defaults
    """
    features = {}

    # Default values for invalid/None channel
    if channel is None or not getattr(channel, 'valid', False):
        features['channel_valid'] = 0.0
        features['channel_direction'] = 1.0  # sideways
        features['channel_slope'] = 0.0
        features['channel_intercept'] = 0.0
        features['channel_r_squared'] = 0.0
        features['channel_width_pct'] = 0.0
        features['bounce_count'] = 0.0
        features['complete_cycles'] = 0.0
        features['upper_touches'] = 0.0
        features['lower_touches'] = 0.0
        features['alternation_ratio'] = 0.0
        features['quality_score'] = 0.0
        return features

    # Extract features from valid channel
    features['channel_valid'] = 1.0
    features['channel_direction'] = safe_float(int(channel.direction), 1.0)
    features['channel_slope'] = safe_float(channel.slope, 0.0)
    features['channel_intercept'] = safe_float(channel.intercept, 0.0)
    features['channel_r_squared'] = safe_float(channel.r_squared, 0.0)
    features['channel_width_pct'] = safe_float(channel.width_pct, 0.0)
    features['bounce_count'] = safe_float(channel.bounce_count, 0.0)
    features['complete_cycles'] = safe_float(channel.complete_cycles, 0.0)
    features['upper_touches'] = safe_float(getattr(channel, 'upper_touches', 0), 0.0)
    features['lower_touches'] = safe_float(getattr(channel, 'lower_touches', 0), 0.0)
    features['alternation_ratio'] = safe_float(getattr(channel, 'alternation_ratio', 0.0), 0.0)
    features['quality_score'] = safe_float(getattr(channel, 'quality_score', 0.0), 0.0)

    return features


def extract_price_position_features(
    tsla_df: pd.DataFrame,
    channel: Optional[Channel]
) -> Dict[str, float]:
    """
    Extract price position features from TSLA data and channel.

    Args:
        tsla_df: TSLA OHLCV DataFrame
        channel: Channel object for position calculations

    Returns:
        Dict of price position features
    """
    features = {}

    # Get close prices
    if len(tsla_df) < 2:
        features['position_in_channel'] = 0.5
        features['distance_to_upper_pct'] = 0.0
        features['distance_to_lower_pct'] = 0.0
        features['rsi_14'] = 50.0
        features['price_momentum_5'] = 0.0
        features['price_momentum_20'] = 0.0
        return features

    close = tsla_df['close'].values
    current_price = float(close[-1])

    # Position in channel (0-1)
    if channel is not None and getattr(channel, 'valid', False):
        features['position_in_channel'] = safe_float(channel.position_at(-1), 0.5)
        features['distance_to_upper_pct'] = safe_float(channel.distance_to_upper(-1), 0.0)
        features['distance_to_lower_pct'] = safe_float(channel.distance_to_lower(-1), 0.0)
    else:
        features['position_in_channel'] = 0.5
        features['distance_to_upper_pct'] = 0.0
        features['distance_to_lower_pct'] = 0.0

    # RSI
    features['rsi_14'] = calculate_rsi_safe(close, period=14)

    # Price momentum
    features['price_momentum_5'] = calculate_momentum(close, lookback=5)
    features['price_momentum_20'] = calculate_momentum(close, lookback=20)

    return features


def extract_cross_asset_features(
    tsla_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame],
    vix_df: Optional[pd.DataFrame]
) -> Dict[str, float]:
    """
    Extract cross-asset features (SPY correlation, VIX).

    Args:
        tsla_df: TSLA OHLCV DataFrame
        spy_df: SPY OHLCV DataFrame (can be None)
        vix_df: VIX OHLCV DataFrame (can be None)

    Returns:
        Dict of cross-asset features
    """
    features = {}

    # SPY correlation
    if spy_df is not None and len(spy_df) >= 20 and len(tsla_df) >= 20:
        tsla_close = tsla_df['close'].values
        spy_close = spy_df['close'].values
        features['spy_correlation'] = calculate_correlation_safe(tsla_close, spy_close, window=20)

        # SPY RSI for comparison
        features['spy_rsi_14'] = calculate_rsi_safe(spy_close, period=14)

        # SPY momentum
        features['spy_momentum_5'] = calculate_momentum(spy_close, lookback=5)
    else:
        features['spy_correlation'] = 0.0
        features['spy_rsi_14'] = 50.0
        features['spy_momentum_5'] = 0.0

    # VIX features
    if vix_df is not None and len(vix_df) >= 5:
        vix_close = vix_df['close'].values
        current_vix = safe_float(vix_close[-1], 20.0)

        features['vix_level'] = current_vix

        # VIX 5-day change
        if len(vix_df) >= 5:
            vix_5d_ago = safe_float(vix_close[-5], current_vix)
            features['vix_change_5d'] = safe_pct_change(current_vix, vix_5d_ago, default=0.0)
        else:
            features['vix_change_5d'] = 0.0

        # VIX regime (0=low <15, 1=normal 15-25, 2=high 25-35, 3=extreme >35)
        if current_vix < 15:
            features['vix_regime'] = 0.0
        elif current_vix < 25:
            features['vix_regime'] = 1.0
        elif current_vix < 35:
            features['vix_regime'] = 2.0
        else:
            features['vix_regime'] = 3.0
    else:
        features['vix_level'] = 20.0
        features['vix_change_5d'] = 0.0
        features['vix_regime'] = 1.0

    return features


def extract_window_scores(
    channels_by_window: Optional[Dict[int, Any]],
    windows: Optional[list] = None
) -> Dict[str, float]:
    """
    Extract window-level quality scores for each window size.

    Args:
        channels_by_window: Dict mapping window size to Channel object
        windows: List of window sizes to analyze (defaults to STANDARD_WINDOWS)

    Returns:
        Dict of window score features
    """
    if windows is None:
        windows = STANDARD_WINDOWS

    features = {}

    for w in windows:
        prefix = f'window_{w}'

        # Default values
        features[f'{prefix}_valid'] = 0.0
        features[f'{prefix}_bounce_count'] = 0.0
        features[f'{prefix}_r_squared'] = 0.0

        if channels_by_window is None:
            continue

        channel = channels_by_window.get(w)

        if channel is None:
            continue

        is_valid = getattr(channel, 'valid', False)
        features[f'{prefix}_valid'] = 1.0 if is_valid else 0.0

        if is_valid:
            features[f'{prefix}_bounce_count'] = safe_float(channel.bounce_count, 0.0)
            features[f'{prefix}_r_squared'] = safe_float(channel.r_squared, 0.0)

    return features


def extract_features(
    tsla_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame],
    vix_df: Optional[pd.DataFrame],
    channel: Optional[Channel],
    window: int,
    channels_by_window: Optional[Dict[int, Any]] = None
) -> Dict[str, float]:
    """
    Extract all features for model input.

    This is the main entry point for feature extraction. Returns a flat dict
    of {feature_name: float_value} that is guaranteed to have no NaN values
    and no division by zero errors.

    Args:
        tsla_df: TSLA OHLCV DataFrame with columns [open, high, low, close, volume]
        spy_df: SPY OHLCV DataFrame (can be None if not available)
        vix_df: VIX OHLCV DataFrame (can be None if not available)
        channel: Primary Channel object for the current window
        window: Current window size being used
        channels_by_window: Optional dict mapping window sizes to Channel objects
                           for multi-window score features

    Returns:
        Dict[str, float] with ~50-100 features, all guaranteed to be valid floats

    Example:
        >>> features = extract_features(tsla_df, spy_df, vix_df, channel, 50)
        >>> print(features['channel_direction'])  # 0, 1, or 2
        >>> print(features['rsi_14'])  # 0-100
        >>> print(features['spy_correlation'])  # -1 to 1
    """
    features: Dict[str, float] = {}

    # 1. Channel Features (~12 features)
    channel_features = extract_channel_features(channel)
    features.update(channel_features)

    # 2. Price Position Features (~6 features)
    position_features = extract_price_position_features(tsla_df, channel)
    features.update(position_features)

    # 3. Cross-Asset Features (~6 features)
    cross_asset_features = extract_cross_asset_features(tsla_df, spy_df, vix_df)
    features.update(cross_asset_features)

    # 4. Window Scores (~24 features: 8 windows x 3 metrics)
    window_scores = extract_window_scores(channels_by_window)
    features.update(window_scores)

    # Add the current window as a feature
    features['current_window'] = safe_float(window, 50.0)

    # Final safety check: ensure all values are valid floats
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def get_feature_names() -> list:
    """
    Get the list of all feature names in order.

    Returns:
        List of feature name strings
    """
    names = []

    # Channel features
    names.extend([
        'channel_valid',
        'channel_direction',
        'channel_slope',
        'channel_intercept',
        'channel_r_squared',
        'channel_width_pct',
        'bounce_count',
        'complete_cycles',
        'upper_touches',
        'lower_touches',
        'alternation_ratio',
        'quality_score',
    ])

    # Price position features
    names.extend([
        'position_in_channel',
        'distance_to_upper_pct',
        'distance_to_lower_pct',
        'rsi_14',
        'price_momentum_5',
        'price_momentum_20',
    ])

    # Cross-asset features
    names.extend([
        'spy_correlation',
        'spy_rsi_14',
        'spy_momentum_5',
        'vix_level',
        'vix_change_5d',
        'vix_regime',
    ])

    # Window scores
    for w in STANDARD_WINDOWS:
        names.extend([
            f'window_{w}_valid',
            f'window_{w}_bounce_count',
            f'window_{w}_r_squared',
        ])

    # Current window
    names.append('current_window')

    return names


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Convert features dict to numpy array in consistent order.

    Args:
        features: Dict of feature name to value

    Returns:
        numpy array of feature values
    """
    names = get_feature_names()
    return np.array([features.get(name, 0.0) for name in names], dtype=np.float32)


def get_feature_count() -> int:
    """Get the total number of features."""
    return len(get_feature_names())
