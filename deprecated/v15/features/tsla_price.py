"""
TSLA Price Features Module

Extracts 60 price-based features from TSLA OHLCV data.
All features return valid floats (no NaN or inf values).

Feature Categories:
- Basic Price (11): close, gaps, ranges, shadows
- Volume (7): volume metrics and trends
- Moving Averages (14): SMA, EMA, crossovers, alignment
- Momentum (15): ROC, RSI, stochastic, Williams %R
- Volatility (8): ATR, standard deviation, regime
- Trend (5): higher highs, lower lows, consecutive bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from .utils import (
    safe_float,
    safe_divide,
    ema,
    sma,
    atr,
    rsi,
    get_last_valid,
)


def _pct_change(current: float, previous: float, default: float = 0.0) -> float:
    """Calculate percentage change between two scalar values."""
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return default
    result = ((current - previous) / previous) * 100
    if not np.isfinite(result):
        return default
    return float(result)


def extract_tsla_price_features(tsla_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract 60 TSLA price features from OHLCV DataFrame.

    Args:
        tsla_df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                 Index should be datetime, most recent data last.

    Returns:
        Dict[str, float] with 60 features, all guaranteed to be valid floats.

    Feature Groups:
        - Basic Price (11): close, close_vs_open, close_vs_open_pct, high_low_range,
                           high_low_range_pct, close_vs_high_pct, close_vs_low_pct,
                           upper_shadow_pct, lower_shadow_pct, body_pct, gap_pct
        - Volume (7): volume, volume_vs_avg_10, volume_vs_avg_20, volume_vs_avg_50,
                     volume_trend, volume_price_trend, relative_volume
        - Moving Averages (14): sma_10, sma_20, sma_50, ema_10, ema_20,
                               price_vs_sma_10, price_vs_sma_20, price_vs_sma_50,
                               sma_10_vs_sma_20, sma_20_vs_sma_50, ma_spread,
                               ma_converging, ma_diverging, trend_alignment
        - Momentum (15): momentum_1, momentum_3, momentum_5, momentum_10,
                        momentum_20, momentum_50, acceleration, rsi_5, rsi_9,
                        rsi_14, rsi_21, rsi_divergence, stochastic_k, stochastic_d,
                        williams_r
        - Volatility (8): atr_14, atr_pct, volatility_5, volatility_20,
                         volatility_ratio, range_pct_5, range_pct_20, volatility_regime
        - Trend (5): higher_highs_count, lower_lows_count, up_bars_ratio_10,
                    consecutive_up, consecutive_down
    """
    features: Dict[str, float] = {}

    # Extract arrays
    open_arr = tsla_df['open'].values.astype(float)
    high_arr = tsla_df['high'].values.astype(float)
    low_arr = tsla_df['low'].values.astype(float)
    close_arr = tsla_df['close'].values.astype(float)
    volume_arr = tsla_df['volume'].values.astype(float)

    n = len(close_arr)

    # Previous bar values (use [-2] to avoid data leakage - features should only use data BEFORE prediction point)
    curr_open = safe_float(open_arr[-2]) if n > 1 else 0.0
    curr_high = safe_float(high_arr[-2]) if n > 1 else 0.0
    curr_low = safe_float(low_arr[-2]) if n > 1 else 0.0
    curr_close = safe_float(close_arr[-2]) if n > 1 else 0.0
    curr_volume = safe_float(volume_arr[-2]) if n > 1 else 0.0
    prev_close = safe_float(close_arr[-3]) if n > 2 else curr_close

    # =========================================================================
    # BASIC PRICE FEATURES (11)
    # =========================================================================
    features['close'] = curr_close
    features['close_vs_open'] = curr_close - curr_open
    features['close_vs_open_pct'] = _pct_change(curr_close, curr_open)
    features['high_low_range'] = curr_high - curr_low
    features['high_low_range_pct'] = safe_divide(
        curr_high - curr_low, curr_close, default=0.0
    ) * 100

    # Close position within the bar's range
    bar_range = curr_high - curr_low
    features['close_vs_high_pct'] = safe_divide(
        curr_high - curr_close, bar_range, default=0.0
    ) * 100
    features['close_vs_low_pct'] = safe_divide(
        curr_close - curr_low, bar_range, default=0.0
    ) * 100

    # Shadow analysis (candlestick patterns)
    body_high = max(curr_open, curr_close)
    body_low = min(curr_open, curr_close)
    body_size = abs(curr_close - curr_open)

    features['upper_shadow_pct'] = safe_divide(
        curr_high - body_high, bar_range, default=0.0
    ) * 100
    features['lower_shadow_pct'] = safe_divide(
        body_low - curr_low, bar_range, default=0.0
    ) * 100
    features['body_pct'] = safe_divide(body_size, bar_range, default=0.0) * 100

    # Gap from previous close
    features['gap_pct'] = _pct_change(curr_open, prev_close)

    # =========================================================================
    # VOLUME FEATURES (7)
    # =========================================================================
    features['volume'] = curr_volume

    # Volume vs moving averages
    vol_avg_10 = np.mean(volume_arr[-10:]) if n >= 10 else curr_volume
    vol_avg_20 = np.mean(volume_arr[-20:]) if n >= 20 else curr_volume
    vol_avg_50 = np.mean(volume_arr[-50:]) if n >= 50 else curr_volume

    features['volume_vs_avg_10'] = safe_divide(curr_volume, vol_avg_10, default=1.0)
    features['volume_vs_avg_20'] = safe_divide(curr_volume, vol_avg_20, default=1.0)
    features['volume_vs_avg_50'] = safe_divide(curr_volume, vol_avg_50, default=1.0)

    # Volume trend (5-day vs 20-day average)
    vol_avg_5 = np.mean(volume_arr[-5:]) if n >= 5 else curr_volume
    features['volume_trend'] = safe_divide(vol_avg_5, vol_avg_20, default=1.0)

    # Volume-price trend (positive if price up with high volume)
    price_change = curr_close - prev_close
    vol_ratio = safe_divide(curr_volume, vol_avg_20, default=1.0)
    features['volume_price_trend'] = price_change * vol_ratio if n > 1 else 0.0

    # Relative volume (current vs max of last 20)
    vol_max_20 = np.max(volume_arr[-20:]) if n >= 20 else curr_volume
    features['relative_volume'] = safe_divide(curr_volume, vol_max_20, default=0.0)

    # =========================================================================
    # MOVING AVERAGE FEATURES (14)
    # =========================================================================
    # Calculate SMAs
    sma_10_arr = sma(close_arr, 10)
    sma_20_arr = sma(close_arr, 20)
    sma_50_arr = sma(close_arr, 50)

    sma_10_val = get_last_valid(sma_10_arr, curr_close)
    sma_20_val = get_last_valid(sma_20_arr, curr_close)
    sma_50_val = get_last_valid(sma_50_arr, curr_close)

    features['sma_10'] = sma_10_val
    features['sma_20'] = sma_20_val
    features['sma_50'] = sma_50_val

    # Calculate EMAs
    ema_10_arr = ema(close_arr, 10)
    ema_20_arr = ema(close_arr, 20)

    ema_10_val = get_last_valid(ema_10_arr, curr_close)
    ema_20_val = get_last_valid(ema_20_arr, curr_close)

    features['ema_10'] = ema_10_val
    features['ema_20'] = ema_20_val

    # Price vs SMAs (as percentage)
    features['price_vs_sma_10'] = _pct_change(curr_close, sma_10_val)
    features['price_vs_sma_20'] = _pct_change(curr_close, sma_20_val)
    features['price_vs_sma_50'] = _pct_change(curr_close, sma_50_val)

    # MA crossovers (as percentage difference)
    features['sma_10_vs_sma_20'] = _pct_change(sma_10_val, sma_20_val)
    features['sma_20_vs_sma_50'] = _pct_change(sma_20_val, sma_50_val)

    # MA spread (distance between fastest and slowest MA)
    features['ma_spread'] = _pct_change(sma_10_val, sma_50_val)

    # MA converging/diverging detection
    if n >= 5:
        sma_10_prev = get_last_valid(sma_10_arr[:-1], sma_10_val)
        sma_50_prev = get_last_valid(sma_50_arr[:-1], sma_50_val)
        spread_now = abs(sma_10_val - sma_50_val)
        spread_prev = abs(sma_10_prev - sma_50_prev)
        features['ma_converging'] = 1.0 if spread_now < spread_prev else 0.0
        features['ma_diverging'] = 1.0 if spread_now > spread_prev else 0.0
    else:
        features['ma_converging'] = 0.0
        features['ma_diverging'] = 0.0

    # Trend alignment (1 if SMA10 > SMA20 > SMA50 or SMA10 < SMA20 < SMA50)
    bullish_aligned = sma_10_val > sma_20_val > sma_50_val
    bearish_aligned = sma_10_val < sma_20_val < sma_50_val
    features['trend_alignment'] = 1.0 if (bullish_aligned or bearish_aligned) else 0.0

    # =========================================================================
    # MOMENTUM FEATURES (15)
    # =========================================================================
    # Rate of change (momentum) at various lookbacks
    def momentum(lookback: int) -> float:
        if n <= lookback:
            return 0.0
        return _pct_change(curr_close, safe_float(close_arr[-(lookback + 1)]))

    # momentum_1 removed - use price change instead
    features['momentum_3'] = momentum(3)
    features['momentum_5'] = momentum(5)
    features['momentum_10'] = momentum(10)
    features['momentum_20'] = momentum(20)
    features['momentum_50'] = momentum(50)

    # Acceleration (momentum of momentum)
    mom_5_now = momentum(5)
    if n > 10:
        close_5_ago = safe_float(close_arr[-6])
        close_10_ago = safe_float(close_arr[-11])
        mom_5_prev = _pct_change(close_5_ago, close_10_ago)
        features['acceleration'] = mom_5_now - mom_5_prev
    else:
        features['acceleration'] = 0.0

    # RSI at multiple periods
    rsi_5_arr = rsi(close_arr, 5)
    rsi_9_arr = rsi(close_arr, 9)
    rsi_14_arr = rsi(close_arr, 14)
    rsi_21_arr = rsi(close_arr, 21)

    features['rsi_5'] = get_last_valid(rsi_5_arr, 50.0)
    features['rsi_9'] = get_last_valid(rsi_9_arr, 50.0)
    features['rsi_14'] = get_last_valid(rsi_14_arr, 50.0)
    features['rsi_21'] = get_last_valid(rsi_21_arr, 50.0)

    # RSI divergence (price making new high but RSI lower, or vice versa)
    if n >= 14:
        price_5d_change = momentum(5)
        rsi_14_prev = get_last_valid(rsi_14_arr[:-5], 50.0) if n > 5 else 50.0
        rsi_14_now = features['rsi_14']
        rsi_change = rsi_14_now - rsi_14_prev

        # Bearish divergence: price up, RSI down
        # Bullish divergence: price down, RSI up
        if price_5d_change > 1.0 and rsi_change < -5.0:
            features['rsi_divergence'] = -1.0  # Bearish
        elif price_5d_change < -1.0 and rsi_change > 5.0:
            features['rsi_divergence'] = 1.0  # Bullish
        else:
            features['rsi_divergence'] = 0.0
    else:
        features['rsi_divergence'] = 0.0

    # Stochastic oscillator
    stoch_k, stoch_d = _calculate_stochastic(high_arr, low_arr, close_arr)
    features['stochastic_k'] = stoch_k
    features['stochastic_d'] = stoch_d

    # Williams %R removed - inverse of stochastic_k (redundant)

    # =========================================================================
    # VOLATILITY FEATURES (8)
    # =========================================================================
    # ATR
    atr_arr = atr(high_arr, low_arr, close_arr, 14)
    atr_14_val = get_last_valid(atr_arr, 0.0)
    features['atr_14'] = atr_14_val
    features['atr_pct'] = safe_divide(atr_14_val, curr_close, default=0.0) * 100

    # Standard deviation of returns
    features['volatility_5'] = _calculate_volatility(close_arr, 5)
    features['volatility_20'] = _calculate_volatility(close_arr, 20)

    # Volatility ratio (short-term vs long-term)
    vol_5 = features['volatility_5']
    vol_20 = features['volatility_20']
    features['volatility_ratio'] = safe_divide(vol_5, vol_20, default=1.0)

    # Range percentage (high-low range as % of close)
    features['range_pct_5'] = _calculate_range_pct(high_arr, low_arr, close_arr, 5)
    features['range_pct_20'] = _calculate_range_pct(high_arr, low_arr, close_arr, 20)

    # Volatility regime (0=low, 1=normal, 2=high)
    # Based on current ATR% vs typical levels
    atr_pct = features['atr_pct']
    if atr_pct < 1.5:
        features['volatility_regime'] = 0.0  # Low volatility
    elif atr_pct < 3.0:
        features['volatility_regime'] = 1.0  # Normal
    else:
        features['volatility_regime'] = 2.0  # High volatility

    # =========================================================================
    # TREND FEATURES (5)
    # =========================================================================
    # Higher highs and lower lows count (last 10 bars)
    hh_count, ll_count = _count_higher_highs_lower_lows(high_arr, low_arr, 10)
    features['higher_highs_count'] = hh_count
    features['lower_lows_count'] = ll_count

    # Up bars ratio (percentage of bars that closed up)
    features['up_bars_ratio_10'] = _calculate_up_bars_ratio(close_arr, 10)

    # Consecutive up/down bars
    consec_up, consec_down = _count_consecutive_bars(close_arr)
    features['consecutive_up'] = consec_up
    features['consecutive_down'] = consec_down

    # Final safety check
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> tuple:
    """Calculate Stochastic %K and %D."""
    n = len(high)
    if n < k_period:
        return 50.0, 50.0

    try:
        # Calculate %K for last d_period bars
        k_values = []
        for i in range(d_period):
            idx = n - d_period + i
            start_idx = max(0, idx - k_period + 1)
            highest = np.max(high[start_idx:idx + 1])
            lowest = np.min(low[start_idx:idx + 1])
            current = close[idx]

            range_val = highest - lowest
            if range_val == 0:
                k_values.append(50.0)
            else:
                k = ((current - lowest) / range_val) * 100
                k_values.append(float(np.clip(k, 0.0, 100.0)))

        k = k_values[-1]
        d = float(np.mean(k_values))

        return k, d
    except Exception:
        return 50.0, 50.0


def _calculate_williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> float:
    """Calculate Williams %R."""
    n = len(high)
    if n < period:
        return -50.0

    try:
        # Use [-2] to avoid data leakage - features should only use data BEFORE prediction point
        highest = np.max(high[-period-1:-1])
        lowest = np.min(low[-period-1:-1])
        current = close[-2]

        range_val = highest - lowest
        if range_val == 0:
            return -50.0

        wr = ((highest - current) / range_val) * -100
        return float(np.clip(wr, -100.0, 0.0))
    except Exception:
        return -50.0


def _calculate_volatility(close: np.ndarray, period: int) -> float:
    """Calculate standard deviation of returns as percentage."""
    n = len(close)
    if n < period + 1:
        return 0.0

    try:
        returns = np.diff(close[-(period + 1):]) / close[-(period + 1):-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            return 0.0

        vol = float(np.std(returns) * 100)
        return vol if np.isfinite(vol) else 0.0
    except Exception:
        return 0.0


def _calculate_range_pct(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> float:
    """Calculate average high-low range as percentage of close."""
    n = len(high)
    if n < period:
        return 0.0

    try:
        ranges = high[-period:] - low[-period:]
        closes = close[-period:]
        pct_ranges = ranges / closes * 100
        pct_ranges = pct_ranges[np.isfinite(pct_ranges)]

        if len(pct_ranges) == 0:
            return 0.0

        return float(np.mean(pct_ranges))
    except Exception:
        return 0.0


def _count_higher_highs_lower_lows(
    high: np.ndarray,
    low: np.ndarray,
    period: int
) -> tuple:
    """Count higher highs and lower lows in the period."""
    n = len(high)
    if n < period:
        return 0.0, 0.0

    hh_count = 0.0
    ll_count = 0.0

    for i in range(1, period):
        idx = -(period - i)
        if idx == 0:
            curr_high = high[-1]
            curr_low = low[-1]
        else:
            curr_high = high[idx]
            curr_low = low[idx]

        prev_high = high[idx - 1]
        prev_low = low[idx - 1]

        if curr_high > prev_high:
            hh_count += 1
        if curr_low < prev_low:
            ll_count += 1

    return hh_count, ll_count


def _calculate_up_bars_ratio(close: np.ndarray, period: int) -> float:
    """Calculate ratio of up bars in the period."""
    n = len(close)
    if n < period + 1:
        return 0.5

    try:
        changes = np.diff(close[-(period + 1):])
        up_count = np.sum(changes > 0)
        return float(up_count / period)
    except Exception:
        return 0.5


def _count_consecutive_bars(close: np.ndarray) -> tuple:
    """Count consecutive up and down bars from the most recent bar."""
    n = len(close)
    if n < 2:
        return 0.0, 0.0

    consec_up = 0.0
    consec_down = 0.0

    # Count from most recent backwards
    for i in range(n - 1, 0, -1):
        if close[i] > close[i - 1]:
            if consec_down == 0:
                consec_up += 1
            else:
                break
        elif close[i] < close[i - 1]:
            if consec_up == 0:
                consec_down += 1
            else:
                break
        else:
            break

    return consec_up, consec_down


# Feature names for reference
TSLA_PRICE_FEATURE_NAMES = [
    # Basic Price (11)
    'close', 'close_vs_open', 'close_vs_open_pct', 'high_low_range',
    'high_low_range_pct', 'close_vs_high_pct', 'close_vs_low_pct',
    'upper_shadow_pct', 'lower_shadow_pct', 'body_pct', 'gap_pct',
    # Volume (7)
    'volume', 'volume_vs_avg_10', 'volume_vs_avg_20', 'volume_vs_avg_50',
    'volume_trend', 'volume_price_trend', 'relative_volume',
    # Moving Averages (14)
    'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
    'price_vs_sma_10', 'price_vs_sma_20', 'price_vs_sma_50',
    'sma_10_vs_sma_20', 'sma_20_vs_sma_50', 'ma_spread',
    'ma_converging', 'ma_diverging', 'trend_alignment',
    # Momentum (13) - removed momentum_1, williams_r
    'momentum_3', 'momentum_5', 'momentum_10',
    'momentum_20', 'momentum_50', 'acceleration',
    'rsi_5', 'rsi_9', 'rsi_14', 'rsi_21', 'rsi_divergence',
    'stochastic_k', 'stochastic_d',
    # Volatility (8)
    'atr_14', 'atr_pct', 'volatility_5', 'volatility_20',
    'volatility_ratio', 'range_pct_5', 'range_pct_20', 'volatility_regime',
    # Trend (5)
    'higher_highs_count', 'lower_lows_count', 'up_bars_ratio_10',
    'consecutive_up', 'consecutive_down',
]


def get_tsla_price_feature_names() -> list:
    """Return list of all TSLA price feature names."""
    return TSLA_PRICE_FEATURE_NAMES.copy()


def get_tsla_price_feature_count() -> int:
    """Return the number of TSLA price features (58)."""
    return len(TSLA_PRICE_FEATURE_NAMES)


# =============================================================================
# TF-PREFIXED FUNCTIONS
# =============================================================================

def extract_tsla_price_features_tf(
    tsla_df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract price features with TF prefix.

    Args:
        tsla_df: OHLCV DataFrame (already resampled to target TF)
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_close', 'daily_rsi_14', '1h_momentum_5'
    """
    base_features = extract_tsla_price_features(tsla_df)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_tsla_price_feature_names_tf(tf: str) -> list:
    """Get feature names with TF prefix."""
    base_names = get_tsla_price_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_tsla_price_feature_names() -> list:
    """Get ALL price feature names across all TFs."""
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_tsla_price_feature_names_tf(tf))
    return all_names


def get_total_price_features() -> int:
    """Total price features: 60 * 10 TFs = 600"""
    return 60 * 10
