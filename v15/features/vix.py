"""
VIX Feature Extraction

The VIX (CBOE Volatility Index) measures market fear and expected volatility.
This module extracts 25 features from VIX data across multiple categories:

LEVEL (5): Current value and moving average relationships
CHANGES (4): Short and medium-term momentum
PERCENTILES (3): Historical context at multiple timeframes
REGIME (4): Market fear classification and extreme events
TECHNICALS (5): RSI, Bollinger Bands, momentum indicators
STRUCTURE (4): Recent highs, lows, and volatility of volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from .utils import (
    safe_float,
    safe_divide,
    calc_sma,
    calc_rsi,
    get_last_valid,
)


def _pct_change(current: float, previous: float, default: float = 0.0) -> float:
    """Calculate percentage change between two values."""
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return default
    result = ((current - previous) / previous) * 100
    return float(result) if np.isfinite(result) else default


def extract_vix_features(vix_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract 25 VIX features from OHLCV DataFrame.

    Args:
        vix_df: VIX OHLCV DataFrame with columns [open, high, low, close, volume]
                Must have at least 1 row. More data provides better features.

    Returns:
        Dict[str, float] with exactly 25 features, all guaranteed to be valid floats.

    Feature Categories:
        - LEVEL (5): vix_level, vix_sma_10, vix_sma_20, vix_vs_sma_10, vix_vs_sma_20
        - CHANGES (4): vix_change_1d, vix_change_5d, vix_change_20d, vix_acceleration
        - PERCENTILES (3): vix_percentile_30d, vix_percentile_90d, vix_percentile_252d
        - REGIME (4): vix_regime, vix_spike, vix_crush, vix_extreme
        - TECHNICALS (5): vix_rsi, vix_bb_pct_b, vix_momentum_5, vix_momentum_10, vix_mean_reversion
        - STRUCTURE (4): vix_5d_high, vix_5d_low, vix_range_5d, vix_volatility
    """
    features: Dict[str, float] = {}

    # Handle empty or invalid DataFrame
    if vix_df is None or len(vix_df) < 1:
        return _get_default_features()

    # Extract close prices as numpy array
    close = vix_df['close'].values.astype(float)
    high = vix_df['high'].values.astype(float) if 'high' in vix_df.columns else close
    low = vix_df['low'].values.astype(float) if 'low' in vix_df.columns else close

    current_vix = safe_float(close[-1], 20.0)

    # =========================================================================
    # LEVEL FEATURES (5)
    # =========================================================================

    # vix_level: Current VIX value
    features['vix_level'] = current_vix

    # vix_sma_10: 10-day simple moving average
    sma_10_arr = calc_sma(close, 10)
    sma_10 = get_last_valid(sma_10_arr, current_vix)
    features['vix_sma_10'] = sma_10

    # vix_sma_20: 20-day simple moving average
    sma_20_arr = calc_sma(close, 20)
    sma_20 = get_last_valid(sma_20_arr, current_vix)
    features['vix_sma_20'] = sma_20

    # vix_vs_sma_10: % above/below 10-day SMA
    features['vix_vs_sma_10'] = _pct_change(current_vix, sma_10, 0.0)

    # vix_vs_sma_20: % above/below 20-day SMA
    features['vix_vs_sma_20'] = _pct_change(current_vix, sma_20, 0.0)

    # =========================================================================
    # CHANGES FEATURES (4)
    # =========================================================================

    # vix_change_1d: 1-day % change
    if len(close) >= 2:
        features['vix_change_1d'] = _pct_change(close[-1], close[-2], 0.0)
    else:
        features['vix_change_1d'] = 0.0

    # vix_change_5d: 5-day % change
    if len(close) >= 6:
        features['vix_change_5d'] = _pct_change(close[-1], close[-6], 0.0)
    else:
        features['vix_change_5d'] = 0.0

    # vix_change_20d: 20-day % change
    if len(close) >= 21:
        features['vix_change_20d'] = _pct_change(close[-1], close[-21], 0.0)
    else:
        features['vix_change_20d'] = 0.0

    # vix_acceleration: Change in momentum (today's change minus yesterday's change)
    if len(close) >= 3:
        change_today = _pct_change(close[-1], close[-2], 0.0)
        change_yesterday = _pct_change(close[-2], close[-3], 0.0)
        features['vix_acceleration'] = change_today - change_yesterday
    else:
        features['vix_acceleration'] = 0.0

    # =========================================================================
    # PERCENTILE FEATURES (3)
    # =========================================================================

    # vix_percentile_30d: Where is VIX in last 30 days (0-100)
    features['vix_percentile_30d'] = _calculate_percentile(current_vix, close, 30)

    # vix_percentile_90d: Where is VIX in last 90 days (0-100)
    features['vix_percentile_90d'] = _calculate_percentile(current_vix, close, 90)

    # vix_percentile_252d: Where is VIX in last 252 days / 1 year (0-100)
    features['vix_percentile_252d'] = _calculate_percentile(current_vix, close, 252)

    # =========================================================================
    # REGIME FEATURES (4)
    # =========================================================================

    # vix_regime: Market fear classification
    # 0 = calm (<15), 1 = normal (15-20), 2 = elevated (20-30), 3 = fear (30+)
    if current_vix < 15:
        features['vix_regime'] = 0.0
    elif current_vix < 20:
        features['vix_regime'] = 1.0
    elif current_vix < 30:
        features['vix_regime'] = 2.0
    else:
        features['vix_regime'] = 3.0

    # vix_spike: Sudden jump indicator (1 if >20% up in 1 day)
    change_1d = features['vix_change_1d']
    features['vix_spike'] = 1.0 if change_1d > 20 else 0.0

    # vix_crush: Sudden drop indicator (1 if >15% down in 1 day)
    features['vix_crush'] = 1.0 if change_1d < -15 else 0.0

    # vix_extreme: Extreme fear indicator (1 if VIX > 35)
    features['vix_extreme'] = 1.0 if current_vix > 35 else 0.0

    # =========================================================================
    # TECHNICALS FEATURES (5)
    # =========================================================================

    # vix_rsi: 14-period RSI of VIX
    rsi_arr = calc_rsi(close, 14)
    features['vix_rsi'] = get_last_valid(rsi_arr, 50.0)

    # vix_bb_pct_b: Bollinger Band %B (where VIX is relative to 20-day bands)
    features['vix_bb_pct_b'] = _calculate_bollinger_pct_b(close, 20, 2.0)

    # vix_momentum_5: 5-day rate of change
    if len(close) >= 6:
        features['vix_momentum_5'] = _pct_change(close[-1], close[-6], 0.0)
    else:
        features['vix_momentum_5'] = 0.0

    # vix_momentum_10: 10-day rate of change
    if len(close) >= 11:
        features['vix_momentum_10'] = _pct_change(close[-1], close[-11], 0.0)
    else:
        features['vix_momentum_10'] = 0.0

    # vix_mean_reversion: Distance from 200-day mean (VIX tends to mean-revert)
    if len(close) >= 200:
        mean_200 = np.mean(close[-200:])
    else:
        mean_200 = np.mean(close)
    features['vix_mean_reversion'] = _pct_change(current_vix, safe_float(mean_200, 20.0), 0.0)

    # =========================================================================
    # STRUCTURE FEATURES (4)
    # =========================================================================

    # vix_5d_high: 5-day high of VIX
    if len(high) >= 5:
        features['vix_5d_high'] = safe_float(np.max(high[-5:]), current_vix)
    else:
        features['vix_5d_high'] = safe_float(np.max(high), current_vix)

    # vix_5d_low: 5-day low of VIX
    if len(low) >= 5:
        features['vix_5d_low'] = safe_float(np.min(low[-5:]), current_vix)
    else:
        features['vix_5d_low'] = safe_float(np.min(low), current_vix)

    # vix_range_5d: 5-day range as % of current VIX
    range_5d = features['vix_5d_high'] - features['vix_5d_low']
    features['vix_range_5d'] = safe_divide(range_5d, current_vix, 0.0) * 100

    # vix_volatility: Volatility of volatility (std of 20-day returns)
    if len(close) >= 21:
        returns = np.diff(close[-21:]) / close[-21:-1]
        # Filter out any NaN or inf values
        valid_returns = returns[np.isfinite(returns)]
        if len(valid_returns) >= 2:
            features['vix_volatility'] = safe_float(np.std(valid_returns) * 100, 0.0)
        else:
            features['vix_volatility'] = 0.0
    else:
        features['vix_volatility'] = 0.0

    # Final safety check: ensure all values are valid floats
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _calculate_percentile(value: float, data: np.ndarray, lookback: int) -> float:
    """
    Calculate where a value falls within historical data (0-100).

    Args:
        value: Current value to rank
        data: Historical data array
        lookback: Number of days to look back

    Returns:
        Percentile rank (0-100)
    """
    if len(data) < 2:
        return 50.0

    # Use available data up to lookback
    window = data[-lookback:] if len(data) >= lookback else data

    try:
        count_below = np.sum(window < value)
        percentile = (count_below / len(window)) * 100

        if not np.isfinite(percentile):
            return 50.0

        return float(np.clip(percentile, 0.0, 100.0))
    except Exception:
        return 50.0


def _calculate_bollinger_pct_b(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """
    Calculate Bollinger Band %B.

    %B shows where price is relative to the bands:
    - 0 = at lower band
    - 0.5 = at middle band (SMA)
    - 1 = at upper band

    Args:
        close: Close price array
        period: Period for SMA and standard deviation
        num_std: Number of standard deviations for bands

    Returns:
        %B value (typically 0-1, can exceed bounds)
    """
    if len(close) < 2:
        return 0.5

    try:
        current_price = close[-1]

        # Calculate SMA
        if len(close) >= period:
            sma_val = np.mean(close[-period:])
            std_val = np.std(close[-period:], ddof=1)
        else:
            sma_val = np.mean(close)
            std_val = np.std(close, ddof=1) if len(close) > 1 else 0.0

        if std_val == 0:
            return 0.5

        upper_band = sma_val + (num_std * std_val)
        lower_band = sma_val - (num_std * std_val)
        band_width = upper_band - lower_band

        if band_width == 0:
            return 0.5

        pct_b = (current_price - lower_band) / band_width

        if not np.isfinite(pct_b):
            return 0.5

        return float(pct_b)

    except Exception:
        return 0.5


def _get_default_features() -> Dict[str, float]:
    """Return default feature values when no VIX data is available."""
    return {
        # LEVEL (5)
        'vix_level': 20.0,
        'vix_sma_10': 20.0,
        'vix_sma_20': 20.0,
        'vix_vs_sma_10': 0.0,
        'vix_vs_sma_20': 0.0,
        # CHANGES (4)
        'vix_change_1d': 0.0,
        'vix_change_5d': 0.0,
        'vix_change_20d': 0.0,
        'vix_acceleration': 0.0,
        # PERCENTILES (3)
        'vix_percentile_30d': 50.0,
        'vix_percentile_90d': 50.0,
        'vix_percentile_252d': 50.0,
        # REGIME (4)
        'vix_regime': 1.0,
        'vix_spike': 0.0,
        'vix_crush': 0.0,
        'vix_extreme': 0.0,
        # TECHNICALS (5)
        'vix_rsi': 50.0,
        'vix_bb_pct_b': 0.5,
        'vix_momentum_5': 0.0,
        'vix_momentum_10': 0.0,
        'vix_mean_reversion': 0.0,
        # STRUCTURE (4)
        'vix_5d_high': 20.0,
        'vix_5d_low': 20.0,
        'vix_range_5d': 0.0,
        'vix_volatility': 0.0,
    }


def get_vix_feature_names() -> list:
    """
    Get the list of all VIX feature names in order.

    Returns:
        List of 25 feature name strings
    """
    return [
        # LEVEL (5)
        'vix_level',
        'vix_sma_10',
        'vix_sma_20',
        'vix_vs_sma_10',
        'vix_vs_sma_20',
        # CHANGES (4)
        'vix_change_1d',
        'vix_change_5d',
        'vix_change_20d',
        'vix_acceleration',
        # PERCENTILES (3)
        'vix_percentile_30d',
        'vix_percentile_90d',
        'vix_percentile_252d',
        # REGIME (4)
        'vix_regime',
        'vix_spike',
        'vix_crush',
        'vix_extreme',
        # TECHNICALS (5)
        'vix_rsi',
        'vix_bb_pct_b',
        'vix_momentum_5',
        'vix_momentum_10',
        'vix_mean_reversion',
        # STRUCTURE (4)
        'vix_5d_high',
        'vix_5d_low',
        'vix_range_5d',
        'vix_volatility',
    ]


def extract_vix_features_tf(
    vix_df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract VIX features with TF prefix.

    Args:
        vix_df: VIX OHLCV DataFrame (already resampled to target TF)
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_vix_level', '1h_vix_regime', 'weekly_vix_rsi'
    """
    base_features = extract_vix_features(vix_df)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_vix_feature_names_tf(tf: str) -> List[str]:
    """
    Get VIX feature names with TF prefix.

    Args:
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        List of 25 feature names with TF prefix (e.g., 'daily_vix_level')
    """
    base_names = get_vix_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_vix_feature_names() -> List[str]:
    """
    Get ALL VIX feature names across all 10 timeframes.

    Returns:
        List of 250 feature names (25 features x 10 TFs)
    """
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_vix_feature_names_tf(tf))
    return all_names


def get_vix_feature_count() -> int:
    """
    Get base VIX feature count.

    Returns:
        25 (number of VIX features per timeframe)
    """
    return 25


def get_total_vix_features() -> int:
    """
    Get total VIX features across all timeframes.

    Returns:
        250 (25 features x 10 TFs)
    """
    return 25 * 10
