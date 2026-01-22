"""
Technical Indicator Features for V15

Implements 77 technical indicator features from OHLCV data.
All functions return Dict[str, float] with valid float values (no NaN/inf).

Categories:
- MACD (5): macd_line, macd_signal, macd_histogram, macd_crossover, macd_divergence
- Bollinger Bands (8): bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b,
                       price_vs_bb_upper, price_vs_bb_lower, bb_squeeze
- Keltner (5): keltner_upper, keltner_middle, keltner_lower, keltner_width, keltner_position
- ADX (4): adx, plus_di, minus_di, di_crossover
- Ichimoku (6): tenkan, kijun, senkou_a, senkou_b, price_vs_cloud, cloud_thickness
- Volume Indicators (10): obv, obv_trend, obv_divergence, mfi, mfi_divergence,
                          accumulation_dist, chaikin_mf, force_index, volume_oscillator, vwap_distance
- Other Oscillators (8): aroon_up, aroon_down, aroon_oscillator, trix,
                         ultimate_oscillator, ppo, dpo, cmo
- Pivot Points (7): pivot, r1, r2, r3, s1, s2, s3
- Fibonacci (6): fib_236, fib_382, fib_500, fib_618, fib_786, nearest_fib_distance
- Candlestick Patterns (12): is_doji, is_hammer, is_shooting_star, is_engulfing_bull,
                             is_engulfing_bear, is_morning_star, is_evening_star,
                             is_harami_bull, is_harami_bear, is_three_white, is_three_black,
                             is_spinning_top
- Additional (6): cci, roc_5, roc_10, roc_20, price_channel_upper, price_channel_lower
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from .utils import (
    safe_float,
    safe_divide,
    ema,
    sma,
    atr,
    rsi,
    true_range,
    get_last_valid,
)


def _scalar_pct_change(current: float, previous: float, default: float = 0.0) -> float:
    """Calculate percentage change between two scalar values."""
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return default
    result = ((current - previous) / previous) * 100
    if not np.isfinite(result):
        return default
    return float(result)


def extract_technical_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract 77 technical indicator features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

    Returns:
        Dict[str, float] with 77 features, all guaranteed to be valid floats
    """
    features: Dict[str, float] = {}

    # Extract arrays
    open_arr = df['open'].values.astype(float)
    high_arr = df['high'].values.astype(float)
    low_arr = df['low'].values.astype(float)
    close_arr = df['close'].values.astype(float)

    # Volume is optional (VIX doesn't have volume)
    has_volume = 'volume' in df.columns
    volume_arr = df['volume'].values.astype(float) if has_volume else np.zeros(len(close_arr))

    n = len(close_arr)
    # Use previous bar's close to avoid data leakage (features should not see current bar)
    current_close = safe_float(close_arr[-2]) if n > 1 else 0.0

    # MACD (5 features)
    macd_features = _calculate_macd(close_arr)
    features.update(macd_features)

    # Bollinger Bands (8 features)
    bb_features = _calculate_bollinger_bands(close_arr, current_close)
    features.update(bb_features)

    # Keltner Channels (5 features)
    keltner_features = _calculate_keltner(close_arr, high_arr, low_arr, current_close)
    features.update(keltner_features)

    # ADX (4 features)
    adx_features = _calculate_adx(high_arr, low_arr, close_arr)
    features.update(adx_features)

    # Ichimoku (6 features)
    ichimoku_features = _calculate_ichimoku(high_arr, low_arr, current_close)
    features.update(ichimoku_features)

    # Volume Indicators (10 features)
    volume_features = _calculate_volume_indicators(
        open_arr, high_arr, low_arr, close_arr, volume_arr
    )
    features.update(volume_features)

    # Other Oscillators (8 features)
    oscillator_features = _calculate_oscillators(high_arr, low_arr, close_arr)
    features.update(oscillator_features)

    # Pivot Points (7 features)
    pivot_features = _calculate_pivot_points(high_arr, low_arr, close_arr)
    features.update(pivot_features)

    # Fibonacci (6 features)
    fib_features = _calculate_fibonacci(high_arr, low_arr, current_close)
    features.update(fib_features)

    # Candlestick Patterns (12 features)
    candle_features = _calculate_candlestick_patterns(open_arr, high_arr, low_arr, close_arr)
    features.update(candle_features)

    # Additional (6 features)
    additional_features = _calculate_additional(high_arr, low_arr, close_arr)
    features.update(additional_features)

    # Final safety check
    for key, value in features.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            features[key] = 0.0

    return features


def _calculate_macd(close: np.ndarray) -> Dict[str, float]:
    """Calculate MACD indicators (5 features)."""
    features = {}

    if len(close) < 26:
        return {
            'macd_line': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'macd_crossover': 0.0,
            'macd_divergence': 0.0,
        }

    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)

    macd_line = ema_12 - ema_26
    macd_signal = ema(macd_line, 9)
    macd_histogram = macd_line - macd_signal

    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    features['macd_line'] = get_last_valid(macd_line[:-1], 0.0) if len(macd_line) > 1 else 0.0
    features['macd_signal'] = get_last_valid(macd_signal[:-1], 0.0) if len(macd_signal) > 1 else 0.0
    features['macd_histogram'] = get_last_valid(macd_histogram[:-1], 0.0) if len(macd_histogram) > 1 else 0.0

    # Crossover: 1 if MACD crossed above signal, -1 if below, 0 otherwise
    # Use [-2] and [-3] to avoid data leakage (features should not see current bar)
    crossover = 0.0
    if len(macd_line) >= 3 and len(macd_signal) >= 3:
        prev_macd = macd_line[-3] if np.isfinite(macd_line[-3]) else 0.0
        prev_signal = macd_signal[-3] if np.isfinite(macd_signal[-3]) else 0.0
        curr_macd = macd_line[-2] if np.isfinite(macd_line[-2]) else 0.0
        curr_signal = macd_signal[-2] if np.isfinite(macd_signal[-2]) else 0.0

        if prev_macd <= prev_signal and curr_macd > curr_signal:
            crossover = 1.0
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            crossover = -1.0

    features['macd_crossover'] = crossover

    # Divergence: compare price direction vs MACD direction over last 10 bars
    # Use [-2] to avoid data leakage (features should not see current bar)
    divergence = 0.0
    if len(close) >= 11 and len(macd_line) >= 11:
        price_change = close[-2] - close[-11]
        macd_change = get_last_valid(macd_line[-2:-1], 0.0) - get_last_valid(macd_line[-11:-10], 0.0)

        if price_change > 0 and macd_change < 0:
            divergence = -1.0  # Bearish divergence
        elif price_change < 0 and macd_change > 0:
            divergence = 1.0  # Bullish divergence

    features['macd_divergence'] = divergence

    return features


def _calculate_bollinger_bands(close: np.ndarray, current_close: float) -> Dict[str, float]:
    """Calculate Bollinger Bands indicators (8 features)."""
    if len(close) < 20:
        return {
            'bb_upper': 0.0,
            'bb_middle': 0.0,
            'bb_lower': 0.0,
            'bb_width': 0.0,
            'bb_pct_b': 0.5,
            'price_vs_bb_upper': 0.0,
            'price_vs_bb_lower': 0.0,
            'bb_squeeze': 0.0,
        }

    period = 20
    std_dev = 2.0

    middle = sma(close, period)
    rolling_std = np.zeros_like(close, dtype=float)

    for i in range(period - 1, len(close)):
        rolling_std[i] = np.std(close[i - period + 1:i + 1])

    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    bb_upper = get_last_valid(upper[:-1], 0.0) if len(upper) > 1 else 0.0
    bb_middle = get_last_valid(middle[:-1], 0.0) if len(middle) > 1 else 0.0
    bb_lower = get_last_valid(lower[:-1], 0.0) if len(lower) > 1 else 0.0

    bb_width = safe_divide(bb_upper - bb_lower, bb_middle, 0.0)

    # %B: (Price - Lower) / (Upper - Lower)
    band_range = bb_upper - bb_lower
    bb_pct_b = safe_divide(current_close - bb_lower, band_range, 0.5)
    bb_pct_b = float(np.clip(bb_pct_b, 0.0, 1.0))

    # Price vs bands (percentage distance)
    price_vs_bb_upper = _scalar_pct_change(current_close, bb_upper, 0.0) if bb_upper > 0 else 0.0
    price_vs_bb_lower = _scalar_pct_change(current_close, bb_lower, 0.0) if bb_lower > 0 else 0.0

    # Squeeze detection: compare current width to historical width
    # Use [-2] to avoid data leakage (features should not see current bar)
    bb_squeeze = 0.0
    if len(rolling_std) >= 51:
        current_std = rolling_std[-2] if np.isfinite(rolling_std[-2]) else 0.0
        recent_std = rolling_std[-51:-1]
        if len(recent_std) > 0 and not np.all(np.isnan(recent_std)):
            avg_std = np.nanmean(recent_std)
        else:
            avg_std = 0.0
        if avg_std > 0 and current_std < avg_std * 0.5:
            bb_squeeze = 1.0

    return {
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'bb_width': bb_width,
        'bb_pct_b': bb_pct_b,
        'price_vs_bb_upper': price_vs_bb_upper,
        'price_vs_bb_lower': price_vs_bb_lower,
        'bb_squeeze': bb_squeeze,
    }


def _calculate_keltner(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    current_close: float
) -> Dict[str, float]:
    """Calculate Keltner Channel indicators (5 features)."""
    if len(close) < 20:
        return {
            'keltner_upper': 0.0,
            'keltner_middle': 0.0,
            'keltner_lower': 0.0,
            'keltner_width': 0.0,
            'keltner_position': 0.5,
        }

    period = 20
    multiplier = 2.0

    middle = ema(close, period)
    atr_values = atr(high, low, close, period)

    upper = middle + multiplier * atr_values
    lower = middle - multiplier * atr_values

    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    keltner_upper = get_last_valid(upper[:-1], 0.0) if len(upper) > 1 else 0.0
    keltner_middle = get_last_valid(middle[:-1], 0.0) if len(middle) > 1 else 0.0
    keltner_lower = get_last_valid(lower[:-1], 0.0) if len(lower) > 1 else 0.0

    keltner_width = safe_divide(keltner_upper - keltner_lower, keltner_middle, 0.0)

    # Position within channel (0=lower, 1=upper)
    channel_range = keltner_upper - keltner_lower
    keltner_position = safe_divide(current_close - keltner_lower, channel_range, 0.5)
    keltner_position = float(np.clip(keltner_position, 0.0, 1.0))

    return {
        'keltner_upper': keltner_upper,
        'keltner_middle': keltner_middle,
        'keltner_lower': keltner_lower,
        'keltner_width': keltner_width,
        'keltner_position': keltner_position,
    }


def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
    """Calculate ADX indicators (4 features)."""
    if len(close) < 28:
        return {
            'adx': 0.0,
            'plus_di': 0.0,
            'minus_di': 0.0,
            'di_crossover': 0.0,
        }

    period = 14
    n = len(close)

    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed values
    tr_values = true_range(high, low, close)
    atr_smooth = ema(tr_values, period)
    plus_dm_smooth = ema(plus_dm, period)
    minus_dm_smooth = ema(minus_dm, period)

    # Calculate +DI and -DI
    plus_di = np.zeros(n, dtype=float)
    minus_di = np.zeros(n, dtype=float)

    for i in range(n):
        if np.isfinite(atr_smooth[i]) and atr_smooth[i] > 0:
            plus_di[i] = 100 * safe_divide(plus_dm_smooth[i], atr_smooth[i], 0.0)
            minus_di[i] = 100 * safe_divide(minus_dm_smooth[i], atr_smooth[i], 0.0)

    # Calculate DX and ADX
    dx = np.zeros(n, dtype=float)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    adx_values = ema(dx, period)

    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    adx = get_last_valid(adx_values[:-1], 0.0) if len(adx_values) > 1 else 0.0
    plus_di_val = get_last_valid(plus_di[:-1], 0.0) if len(plus_di) > 1 else 0.0
    minus_di_val = get_last_valid(minus_di[:-1], 0.0) if len(minus_di) > 1 else 0.0

    # DI crossover
    # Use [-2] and [-3] to avoid data leakage (features should not see current bar)
    di_crossover = 0.0
    if len(plus_di) >= 3 and len(minus_di) >= 3:
        prev_plus = plus_di[-3] if np.isfinite(plus_di[-3]) else 0.0
        prev_minus = minus_di[-3] if np.isfinite(minus_di[-3]) else 0.0
        curr_plus = plus_di[-2] if np.isfinite(plus_di[-2]) else 0.0
        curr_minus = minus_di[-2] if np.isfinite(minus_di[-2]) else 0.0

        if prev_plus <= prev_minus and curr_plus > curr_minus:
            di_crossover = 1.0
        elif prev_plus >= prev_minus and curr_plus < curr_minus:
            di_crossover = -1.0

    return {
        'adx': adx,
        'plus_di': plus_di_val,
        'minus_di': minus_di_val,
        'di_crossover': di_crossover,
    }


def _calculate_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    current_close: float
) -> Dict[str, float]:
    """Calculate Ichimoku indicators (6 features)."""
    if len(high) < 52:
        return {
            'tenkan': 0.0,
            'kijun': 0.0,
            'senkou_a': 0.0,
            'senkou_b': 0.0,
            'price_vs_cloud': 0.0,
            'cloud_thickness': 0.0,
        }

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    tenkan_high = np.max(high[-10:-1])
    tenkan_low = np.min(low[-10:-1])
    tenkan = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = np.max(high[-27:-1])
    kijun_low = np.min(low[-27:-1])
    kijun = (kijun_high + kijun_low) / 2

    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B: (52-period high + 52-period low) / 2
    senkou_b_high = np.max(high[-53:-1])
    senkou_b_low = np.min(low[-53:-1])
    senkou_b = (senkou_b_high + senkou_b_low) / 2

    # Price vs Cloud
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    if current_close > cloud_top:
        price_vs_cloud = 1.0  # Above cloud
    elif current_close < cloud_bottom:
        price_vs_cloud = -1.0  # Below cloud
    else:
        price_vs_cloud = 0.0  # Inside cloud

    # Cloud thickness (normalized by price)
    cloud_thickness = safe_divide(cloud_top - cloud_bottom, current_close, 0.0)

    return {
        'tenkan': safe_float(tenkan, 0.0),
        'kijun': safe_float(kijun, 0.0),
        'senkou_a': safe_float(senkou_a, 0.0),
        'senkou_b': safe_float(senkou_b, 0.0),
        'price_vs_cloud': price_vs_cloud,
        'cloud_thickness': cloud_thickness,
    }


def _calculate_volume_indicators(
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> Dict[str, float]:
    """Calculate Volume indicators (10 features)."""
    n = len(close)

    if n < 20:
        return {
            'obv': 0.0,
            'obv_trend': 0.0,
            'obv_divergence': 0.0,
            'mfi': 50.0,
            'mfi_divergence': 0.0,
            'accumulation_dist': 0.0,
            'chaikin_mf': 0.0,
            'force_index': 0.0,
            'volume_oscillator': 0.0,
            'vwap_distance': 0.0,
        }

    # OBV (On Balance Volume)
    obv = np.zeros(n, dtype=float)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    # Use [-2] to avoid data leakage (features should not see current bar)
    obv_val = safe_float(obv[-2], 0.0) if n >= 2 else 0.0

    # OBV Trend (slope over last 10 periods, normalized)
    # Use [-2] to avoid data leakage
    obv_trend = 0.0
    if n >= 11:
        obv_change = obv[-2] - obv[-11]
        avg_vol = np.mean(volume[-11:-1])
        obv_trend = safe_divide(obv_change, avg_vol * 10, 0.0)

    # OBV Divergence
    # Use [-2] to avoid data leakage
    obv_divergence = 0.0
    if n >= 11:
        price_change = close[-2] - close[-11]
        obv_change = obv[-2] - obv[-11]
        if price_change > 0 and obv_change < 0:
            obv_divergence = -1.0
        elif price_change < 0 and obv_change > 0:
            obv_divergence = 1.0

    # MFI (Money Flow Index)
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    positive_flow = np.zeros(n, dtype=float)
    negative_flow = np.zeros(n, dtype=float)

    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            positive_flow[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            negative_flow[i] = raw_money_flow[i]

    period = 14
    pos_sum = np.sum(positive_flow[-period:])
    neg_sum = np.sum(negative_flow[-period:])

    if neg_sum == 0:
        mfi = 100.0 if pos_sum > 0 else 50.0
    else:
        money_ratio = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + money_ratio))
    mfi = float(np.clip(mfi, 0.0, 100.0))

    # MFI Divergence
    mfi_divergence = 0.0
    # Calculate MFI for 10 bars ago
    if n >= 24:  # Need enough data for historical MFI
        pos_sum_old = np.sum(positive_flow[-24:-10])
        neg_sum_old = np.sum(negative_flow[-24:-10])
        if neg_sum_old > 0:
            mr_old = pos_sum_old / neg_sum_old
            mfi_old = 100.0 - (100.0 / (1.0 + mr_old))
        else:
            mfi_old = 100.0 if pos_sum_old > 0 else 50.0

        # Use [-2] to avoid data leakage (features should not see current bar)
        price_change = close[-2] - close[-11]
        mfi_change = mfi - mfi_old
        if price_change > 0 and mfi_change < 0:
            mfi_divergence = -1.0
        elif price_change < 0 and mfi_change > 0:
            mfi_divergence = 1.0

    # Accumulation/Distribution
    ad = np.zeros(n, dtype=float)
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range > 0:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            ad[i] = clv * volume[i]
        if i > 0:
            ad[i] += ad[i - 1]

    # Use [-2] to avoid data leakage (features should not see current bar)
    accumulation_dist = safe_float(ad[-2], 0.0) if n >= 2 else 0.0

    # Chaikin Money Flow
    # Use range(-cmf_period-1, -1) to avoid data leakage (exclude current bar)
    cmf_period = 20
    cmf_num = 0.0
    cmf_den = 0.0
    for i in range(-cmf_period - 1, -1):
        hl_range = high[i] - low[i]
        if hl_range > 0:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            cmf_num += clv * volume[i]
        cmf_den += volume[i]

    chaikin_mf = safe_divide(cmf_num, cmf_den, 0.0)
    chaikin_mf = float(np.clip(chaikin_mf, -1.0, 1.0))

    # Force Index
    force_idx = np.zeros(n, dtype=float)
    for i in range(1, n):
        force_idx[i] = (close[i] - close[i - 1]) * volume[i]

    force_ema = ema(force_idx, 13)
    # Use [:-1] to avoid data leakage (features should not see current bar)
    force_index = get_last_valid(force_ema[:-1], 0.0) if len(force_ema) > 1 else 0.0

    # Volume Oscillator (short EMA / long EMA - 1)
    vol_short = ema(volume, 5)
    vol_long = ema(volume, 20)
    # Use [:-1] to avoid data leakage (features should not see current bar)
    vol_short_val = get_last_valid(vol_short[:-1], 1.0) if len(vol_short) > 1 else 1.0
    vol_long_val = get_last_valid(vol_long[:-1], 1.0) if len(vol_long) > 1 else 1.0
    volume_oscillator = safe_divide(vol_short_val - vol_long_val, vol_long_val, 0.0)

    # VWAP Distance
    # Simple VWAP approximation for the day
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)

    vwap = np.zeros(n, dtype=float)
    for i in range(n):
        if cumulative_vol[i] > 0:
            vwap[i] = cumulative_tp_vol[i] / cumulative_vol[i]

    # Use [-2] to avoid data leakage (features should not see current bar)
    vwap_val = get_last_valid(vwap[:-1], close[-2]) if n >= 2 else 0.0
    vwap_distance = _scalar_pct_change(close[-2], vwap_val, 0.0) if n >= 2 else 0.0

    return {
        'obv': obv_val,
        'obv_trend': obv_trend,
        'obv_divergence': obv_divergence,
        'mfi': mfi,
        'mfi_divergence': mfi_divergence,
        'accumulation_dist': accumulation_dist,
        'chaikin_mf': chaikin_mf,
        'force_index': force_index,
        'volume_oscillator': volume_oscillator,
        'vwap_distance': vwap_distance,
    }


def _calculate_oscillators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Dict[str, float]:
    """Calculate other oscillators (8 features)."""
    n = len(close)

    if n < 28:
        return {
            'aroon_up': 0.0,
            'aroon_down': 0.0,
            'aroon_oscillator': 0.0,
            'trix': 0.0,
            'ultimate_oscillator': 50.0,
            'ppo': 0.0,
            'dpo': 0.0,
            'cmo': 0.0,
        }

    # Aroon (25 period)
    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    period = 25
    high_window = high[-period - 1:-1]
    low_window = low[-period - 1:-1]

    days_since_high = period - 1 - np.argmax(high_window)
    days_since_low = period - 1 - np.argmin(low_window)

    aroon_up = ((period - days_since_high) / period) * 100
    aroon_down = ((period - days_since_low) / period) * 100
    aroon_oscillator = aroon_up - aroon_down

    # TRIX (15 period)
    ema1 = ema(close, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)

    # Use [-2:-1] and [-3:-2] to avoid data leakage (features should not see current bar)
    trix_val = 0.0
    if len(ema3) >= 3:
        curr = get_last_valid(ema3[-2:-1], 0.0)
        prev = get_last_valid(ema3[-3:-2], curr)
        if prev > 0:
            trix_val = ((curr - prev) / prev) * 100

    # Ultimate Oscillator
    bp = np.zeros(n, dtype=float)  # Buying Pressure
    tr_vals = true_range(high, low, close)

    for i in range(1, n):
        bp[i] = close[i] - min(low[i], close[i - 1])

    # Three periods: 7, 14, 28
    # Use [:-1] slices to avoid data leakage (exclude current bar)
    bp7 = np.sum(bp[-8:-1])
    bp14 = np.sum(bp[-15:-1])
    bp28 = np.sum(bp[-29:-1])

    tr7 = np.sum(tr_vals[-8:-1])
    tr14 = np.sum(tr_vals[-15:-1])
    tr28 = np.sum(tr_vals[-29:-1])

    avg7 = safe_divide(bp7, tr7, 0.5)
    avg14 = safe_divide(bp14, tr14, 0.5)
    avg28 = safe_divide(bp28, tr28, 0.5)

    ultimate_oscillator = ((4 * avg7) + (2 * avg14) + avg28) / 7 * 100
    ultimate_oscillator = float(np.clip(ultimate_oscillator, 0.0, 100.0))

    # PPO (Percentage Price Oscillator)
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)

    ppo_line = np.zeros(n, dtype=float)
    for i in range(n):
        if np.isfinite(ema_26[i]) and ema_26[i] > 0:
            ppo_line[i] = ((ema_12[i] - ema_26[i]) / ema_26[i]) * 100

    # Use [:-1] to avoid data leakage (features should not see current bar)
    ppo = get_last_valid(ppo_line[:-1], 0.0) if len(ppo_line) > 1 else 0.0

    # DPO (Detrended Price Oscillator)
    dpo_period = 20
    sma_vals = sma(close, dpo_period)
    shift = dpo_period // 2 + 1

    # Use [-2] to avoid data leakage (features should not see current bar)
    dpo = 0.0
    if n > shift + 1 and np.isfinite(sma_vals[-shift - 1]):
        dpo = close[-2] - sma_vals[-shift - 1]

    # CMO (Chande Momentum Oscillator)
    cmo_period = 14
    gains = np.zeros(n - 1, dtype=float)
    losses = np.zeros(n - 1, dtype=float)

    for i in range(n - 1):
        diff = close[i + 1] - close[i]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    # Use [:-1] slices to avoid data leakage (exclude current bar)
    # gains/losses arrays are already n-1 length, so -1 more to exclude current bar's change
    sum_gains = np.sum(gains[-cmo_period - 1:-1]) if len(gains) > cmo_period else np.sum(gains[:-1]) if len(gains) > 1 else 0.0
    sum_losses = np.sum(losses[-cmo_period - 1:-1]) if len(losses) > cmo_period else np.sum(losses[:-1]) if len(losses) > 1 else 0.0

    cmo = 0.0
    total = sum_gains + sum_losses
    if total > 0:
        cmo = ((sum_gains - sum_losses) / total) * 100
    cmo = float(np.clip(cmo, -100.0, 100.0))

    return {
        'aroon_up': safe_float(aroon_up, 0.0),
        'aroon_down': safe_float(aroon_down, 0.0),
        'aroon_oscillator': safe_float(aroon_oscillator, 0.0),
        'trix': safe_float(trix_val, 0.0),
        'ultimate_oscillator': ultimate_oscillator,
        'ppo': ppo,
        'dpo': safe_float(dpo, 0.0),
        'cmo': cmo,
    }


def _calculate_pivot_points(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Dict[str, float]:
    """Calculate Pivot Points (7 features)."""
    if len(close) < 2:
        return {
            'pivot': 0.0,
            'r1': 0.0,
            'r2': 0.0,
            'r3': 0.0,
            's1': 0.0,
            's2': 0.0,
            's3': 0.0,
        }

    # Use previous bar's HLC for pivot calculation
    prev_high = safe_float(high[-2], high[-1])
    prev_low = safe_float(low[-2], low[-1])
    prev_close = safe_float(close[-2], close[-1])

    pivot = (prev_high + prev_low + prev_close) / 3

    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high

    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)

    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)

    return {
        'pivot': safe_float(pivot, 0.0),
        'r1': safe_float(r1, 0.0),
        'r2': safe_float(r2, 0.0),
        'r3': safe_float(r3, 0.0),
        's1': safe_float(s1, 0.0),
        's2': safe_float(s2, 0.0),
        's3': safe_float(s3, 0.0),
    }


def _calculate_fibonacci(
    high: np.ndarray,
    low: np.ndarray,
    current_close: float
) -> Dict[str, float]:
    """Calculate Fibonacci levels (6 features)."""
    if len(high) < 20:
        return {
            'fib_236': 0.0,
            'fib_382': 0.0,
            'fib_500': 0.0,
            'fib_618': 0.0,
            'fib_786': 0.0,
            'nearest_fib_distance': 0.0,
        }

    # Use last 50 bars (or available) for high/low
    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    lookback = min(50, len(high) - 1)
    swing_high = np.max(high[-lookback - 1:-1])
    swing_low = np.min(low[-lookback - 1:-1])

    range_size = swing_high - swing_low

    if range_size == 0:
        return {
            'fib_236': 0.0,
            'fib_382': 0.0,
            'fib_500': 0.0,
            'fib_618': 0.0,
            'fib_786': 0.0,
            'nearest_fib_distance': 0.0,
        }

    # Calculate Fibonacci retracement levels (from high)
    fib_236 = swing_high - 0.236 * range_size
    fib_382 = swing_high - 0.382 * range_size
    fib_500 = swing_high - 0.500 * range_size
    fib_618 = swing_high - 0.618 * range_size
    fib_786 = swing_high - 0.786 * range_size

    # Nearest Fibonacci distance (normalized)
    fib_levels = [fib_236, fib_382, fib_500, fib_618, fib_786]
    distances = [abs(current_close - level) for level in fib_levels]
    nearest_distance = min(distances)
    nearest_fib_distance = safe_divide(nearest_distance, current_close, 0.0)

    return {
        'fib_236': safe_float(fib_236, 0.0),
        'fib_382': safe_float(fib_382, 0.0),
        'fib_500': safe_float(fib_500, 0.0),
        'fib_618': safe_float(fib_618, 0.0),
        'fib_786': safe_float(fib_786, 0.0),
        'nearest_fib_distance': nearest_fib_distance,
    }


def _calculate_candlestick_patterns(
    open_arr: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Dict[str, float]:
    """Calculate Candlestick Pattern features (12 features)."""
    n = len(close)

    defaults = {
        'is_doji': 0.0,
        'is_hammer': 0.0,
        'is_shooting_star': 0.0,
        'is_engulfing_bull': 0.0,
        'is_engulfing_bear': 0.0,
        'is_morning_star': 0.0,
        'is_evening_star': 0.0,
        'is_harami_bull': 0.0,
        'is_harami_bear': 0.0,
        'is_three_white': 0.0,
        'is_three_black': 0.0,
        'is_spinning_top': 0.0,
    }

    if n < 4:
        return defaults

    # Use previous candle properties to avoid data leakage (features should not see current bar)
    o, h, l, c = open_arr[-2], high[-2], low[-2], close[-2]
    body = abs(c - o)
    full_range = h - l
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    # Average body size for reference (exclude current bar)
    avg_body = np.mean([abs(close[i] - open_arr[i]) for i in range(-11, -1)])
    if avg_body == 0:
        avg_body = 1.0

    features = defaults.copy()

    # Doji: body is very small compared to range
    if full_range > 0 and body / full_range < 0.1:
        features['is_doji'] = 1.0

    # Hammer: small body at top, long lower shadow
    if full_range > 0:
        if lower_shadow >= 2 * body and upper_shadow < body and body > 0:
            features['is_hammer'] = 1.0

    # Shooting Star: small body at bottom, long upper shadow
    if full_range > 0:
        if upper_shadow >= 2 * body and lower_shadow < body and body > 0:
            features['is_shooting_star'] = 1.0

    # Spinning Top: small body, shadows on both sides
    if full_range > 0 and body < avg_body * 0.3:
        if upper_shadow > body and lower_shadow > body:
            features['is_spinning_top'] = 1.0

    if n >= 3:
        # Two bars ago (since o,h,l,c is now [-2], this is [-3])
        o1, h1, l1, c1 = open_arr[-3], high[-3], low[-3], close[-3]
        body1 = abs(c1 - o1)

        # Bullish Engulfing
        if c1 < o1 and c > o:  # Prev bearish, current bullish
            if c > o1 and o < c1:  # Current body engulfs previous
                features['is_engulfing_bull'] = 1.0

        # Bearish Engulfing
        if c1 > o1 and c < o:  # Prev bullish, current bearish
            if c < o1 and o > c1:  # Current body engulfs previous
                features['is_engulfing_bear'] = 1.0

        # Bullish Harami
        if c1 < o1 and c > o:  # Prev bearish, current bullish
            if o > c1 and c < o1:  # Current inside previous
                features['is_harami_bull'] = 1.0

        # Bearish Harami
        if c1 > o1 and c < o:  # Prev bullish, current bearish
            if o < c1 and c > o1:  # Current inside previous
                features['is_harami_bear'] = 1.0

    if n >= 4:
        # Shifted by 1 to avoid data leakage: [-4], [-3], [-2] instead of [-3], [-2], [-1]
        o2, h2, l2, c2 = open_arr[-4], high[-4], low[-4], close[-4]
        o1, h1, l1, c1 = open_arr[-3], high[-3], low[-3], close[-3]
        body2 = abs(c2 - o2)
        body1 = abs(c1 - o1)

        # Morning Star
        if c2 < o2 and body1 < body2 * 0.3 and c > o and c > (o2 + c2) / 2:
            features['is_morning_star'] = 1.0

        # Evening Star
        if c2 > o2 and body1 < body2 * 0.3 and c < o and c < (o2 + c2) / 2:
            features['is_evening_star'] = 1.0

        # Three White Soldiers (use [-4], [-3], [-2] to avoid data leakage)
        if (close[-4] > open_arr[-4] and
            close[-3] > open_arr[-3] and
            close[-2] > open_arr[-2] and
            close[-3] > close[-4] and
            close[-2] > close[-3]):
            features['is_three_white'] = 1.0

        # Three Black Crows (use [-4], [-3], [-2] to avoid data leakage)
        if (close[-4] < open_arr[-4] and
            close[-3] < open_arr[-3] and
            close[-2] < open_arr[-2] and
            close[-3] < close[-4] and
            close[-2] < close[-3]):
            features['is_three_black'] = 1.0

    return features


def _calculate_additional(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Dict[str, float]:
    """Calculate additional indicators (6 features)."""
    n = len(close)

    if n < 20:
        return {
            'cci': 0.0,
            'roc_5': 0.0,
            'roc_10': 0.0,
            'roc_20': 0.0,
            'price_channel_upper': 0.0,
            'price_channel_lower': 0.0,
        }

    # CCI (Commodity Channel Index)
    period = 20
    typical_price = (high + low + close) / 3
    tp_sma = sma(typical_price, period)

    # Mean deviation
    mean_dev = np.zeros(n, dtype=float)
    for i in range(period - 1, n):
        mean_dev[i] = np.mean(np.abs(typical_price[i - period + 1:i + 1] - tp_sma[i]))

    cci = np.zeros(n, dtype=float)
    for i in range(period - 1, n):
        if mean_dev[i] > 0:
            cci[i] = (typical_price[i] - tp_sma[i]) / (0.015 * mean_dev[i])

    # Use [:-1] to avoid data leakage (features should not see current bar)
    cci_val = get_last_valid(cci[:-1], 0.0) if len(cci) > 1 else 0.0

    # ROC (Rate of Change)
    # Use [-2] to avoid data leakage (features should not see current bar)
    def calc_roc(periods: int) -> float:
        if n > periods + 1:
            prev = close[-periods - 2]
            if prev > 0:
                return ((close[-2] - prev) / prev) * 100
        return 0.0

    roc_5 = calc_roc(5)
    roc_10 = calc_roc(10)
    roc_20 = calc_roc(20)

    # Price Channel (Donchian)
    # Use [:-1] slices to avoid data leakage (features should not see current bar)
    channel_period = 20
    price_channel_upper = np.max(high[-channel_period - 1:-1])
    price_channel_lower = np.min(low[-channel_period - 1:-1])

    return {
        'cci': safe_float(cci_val, 0.0),
        'roc_5': safe_float(roc_5, 0.0),
        'roc_10': safe_float(roc_10, 0.0),
        'roc_20': safe_float(roc_20, 0.0),
        'price_channel_upper': safe_float(price_channel_upper, 0.0),
        'price_channel_lower': safe_float(price_channel_lower, 0.0),
    }


def get_technical_feature_names() -> list:
    """Get the list of all 77 technical feature names."""
    return [
        # MACD (5)
        'macd_line', 'macd_signal', 'macd_histogram', 'macd_crossover', 'macd_divergence',
        # Bollinger Bands (8)
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct_b',
        'price_vs_bb_upper', 'price_vs_bb_lower', 'bb_squeeze',
        # Keltner (5)
        'keltner_upper', 'keltner_middle', 'keltner_lower', 'keltner_width', 'keltner_position',
        # ADX (4)
        'adx', 'plus_di', 'minus_di', 'di_crossover',
        # Ichimoku (6)
        'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'price_vs_cloud', 'cloud_thickness',
        # Volume Indicators (10)
        'obv', 'obv_trend', 'obv_divergence', 'mfi', 'mfi_divergence',
        'accumulation_dist', 'chaikin_mf', 'force_index', 'volume_oscillator', 'vwap_distance',
        # Other Oscillators (8)
        'aroon_up', 'aroon_down', 'aroon_oscillator', 'trix',
        'ultimate_oscillator', 'ppo', 'dpo', 'cmo',
        # Pivot Points (7)
        'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3',
        # Fibonacci (6)
        'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'nearest_fib_distance',
        # Candlestick Patterns (12)
        'is_doji', 'is_hammer', 'is_shooting_star', 'is_engulfing_bull', 'is_engulfing_bear',
        'is_morning_star', 'is_evening_star', 'is_harami_bull', 'is_harami_bear',
        'is_three_white', 'is_three_black', 'is_spinning_top',
        # Additional (6)
        'cci', 'roc_5', 'roc_10', 'roc_20', 'price_channel_upper', 'price_channel_lower',
    ]


def get_technical_feature_count() -> int:
    """Get the total number of technical features (77)."""
    return len(get_technical_feature_names())


# =============================================================================
# TF-Prefixed Feature Extraction
# =============================================================================

def extract_technical_features_tf(
    df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract technical features with TF prefix.

    Args:
        df: OHLCV DataFrame (already resampled to target TF)
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_macd_line', '1h_bb_upper', 'weekly_adx'
    """
    base_features = extract_technical_features(df)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_technical_feature_names_tf(tf: str) -> List[str]:
    """Get feature names with TF prefix."""
    base_names = get_technical_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_technical_feature_names() -> List[str]:
    """Get ALL technical feature names across all TFs."""
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_technical_feature_names_tf(tf))
    return all_names


def get_total_technical_features() -> int:
    """Total technical features: 77 * 10 TFs = 770"""
    return 77 * 10
