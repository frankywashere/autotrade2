"""
SPY Feature Extraction Module

Extracts 80 SPY-specific features that MIRROR TSLA features.
All features are prefixed with "spy_" and return a flat Dict[str, float].

Features Categories (80 total):
- BASIC PRICE (10): Close, gap, range, shadows, volume
- MOVING AVERAGES (10): SMA at multiple periods, price vs SMA, trend alignment
- MOMENTUM (15): Multi-period momentum, RSI variants, stochastic, divergence
- VOLATILITY (8): ATR, volatility at multiple periods, range, regime
- TECHNICAL (20): MACD, Bollinger, ADX/DMI, OBV, MFI, Aroon, CCI, pivots, Ichimoku, Keltner
- TREND (7): Higher highs/lower lows counts, up/down bars, consecutive, trend strength
- MARKET REGIME (10): Intraday position, gap, expansion, acceleration, VPT, efficiency, choppiness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

from .utils import (
    safe_float,
    safe_divide,
    calc_ema,
    calc_sma,
    calc_atr,
    calc_rsi,
    get_last_valid,
    true_range,
)


def _pct_change(current: float, previous: float, default: float = 0.0) -> float:
    """Calculate percentage change between two values safely."""
    if previous == 0 or not np.isfinite(previous) or not np.isfinite(current):
        return default
    result = ((current - previous) / previous) * 100
    if not np.isfinite(result):
        return default
    return float(result)


# ============================================================================
# BASIC PRICE FEATURES (10)
# ============================================================================

def _extract_basic_price_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract basic price features.

    Features:
        spy_close: Current close price
        spy_close_vs_open_pct: % change from open to close
        spy_high_low_range_pct: High-low range as % of close
        spy_close_vs_high_pct: Distance from high as % of range
        spy_close_vs_low_pct: Distance from low as % of range
        spy_body_pct: Candle body as % of high-low range
        spy_gap_pct: Gap from previous close as %
        spy_upper_shadow_pct: Upper shadow as % of high-low range
        spy_lower_shadow_pct: Lower shadow as % of high-low range
        spy_volume_vs_avg: Volume relative to 20-day average
    """
    features = {}

    if len(df) < 3:
        return {
            "spy_close": 0.0,
            "spy_close_vs_open_pct": 0.0,
            "spy_high_low_range_pct": 0.0,
            "spy_close_vs_high_pct": 0.5,
            "spy_close_vs_low_pct": 0.5,
            "spy_body_pct": 0.0,
            "spy_gap_pct": 0.0,
            "spy_upper_shadow_pct": 0.0,
            "spy_lower_shadow_pct": 0.0,
            "spy_volume_vs_avg": 1.0,
        }

    o = float(df["open"].iloc[-2])
    h = float(df["high"].iloc[-2])
    l = float(df["low"].iloc[-2])
    c = float(df["close"].iloc[-2])
    v = float(df["volume"].iloc[-2])
    prev_c = float(df["close"].iloc[-3])

    hl_range = h - l
    body = abs(c - o)

    features["spy_close"] = c
    features["spy_close_vs_open_pct"] = _pct_change(c, o, 0.0)
    features["spy_high_low_range_pct"] = safe_divide(hl_range, c, 0.0) * 100

    # Position in range (0=at low, 1=at high)
    features["spy_close_vs_high_pct"] = safe_divide(h - c, hl_range, 0.5)
    features["spy_close_vs_low_pct"] = safe_divide(c - l, hl_range, 0.5)

    # Body as % of range
    features["spy_body_pct"] = safe_divide(body, hl_range, 0.0)

    # Gap from previous close
    features["spy_gap_pct"] = _pct_change(o, prev_c, 0.0)

    # Shadows
    if c >= o:  # Bullish candle
        upper_shadow = h - c
        lower_shadow = o - l
    else:  # Bearish candle
        upper_shadow = h - o
        lower_shadow = c - l

    features["spy_upper_shadow_pct"] = safe_divide(upper_shadow, hl_range, 0.0)
    features["spy_lower_shadow_pct"] = safe_divide(lower_shadow, hl_range, 0.0)

    # Volume vs 20-day average
    if len(df) >= 20 and "volume" in df.columns:
        avg_vol = df["volume"].iloc[-20:].mean()
        features["spy_volume_vs_avg"] = safe_divide(v, avg_vol, 1.0)
    else:
        features["spy_volume_vs_avg"] = 1.0

    return features


# ============================================================================
# MOVING AVERAGE FEATURES (10)
# ============================================================================

def _extract_moving_average_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract moving average features.

    Features:
        spy_sma_10: 10-period SMA
        spy_sma_20: 20-period SMA
        spy_sma_50: 50-period SMA
        spy_price_vs_sma_10: Price position relative to SMA 10 (%)
        spy_price_vs_sma_20: Price position relative to SMA 20 (%)
        spy_price_vs_sma_50: Price position relative to SMA 50 (%)
        spy_sma_10_vs_sma_20: SMA 10 position vs SMA 20 (%)
        spy_sma_20_vs_sma_50: SMA 20 position vs SMA 50 (%)
        spy_ma_spread: Spread between SMA 10 and SMA 50 (%)
        spy_trend_alignment: Trend alignment score (-1 to 1)
    """
    features = {}

    close = df["close"].values
    c = float(close[-1]) if len(close) > 0 else 0.0

    # Calculate SMAs
    sma_10 = get_last_valid(calc_sma(close, 10), c)
    sma_20 = get_last_valid(calc_sma(close, 20), c)
    sma_50 = get_last_valid(calc_sma(close, 50), c)

    features["spy_sma_10"] = sma_10
    features["spy_sma_20"] = sma_20
    features["spy_sma_50"] = sma_50

    # Price vs SMAs (as %)
    features["spy_price_vs_sma_10"] = _pct_change(c, sma_10, 0.0)
    features["spy_price_vs_sma_20"] = _pct_change(c, sma_20, 0.0)
    features["spy_price_vs_sma_50"] = _pct_change(c, sma_50, 0.0)

    # SMA relationships
    features["spy_sma_10_vs_sma_20"] = _pct_change(sma_10, sma_20, 0.0)
    features["spy_sma_20_vs_sma_50"] = _pct_change(sma_20, sma_50, 0.0)

    # MA spread (10 vs 50)
    features["spy_ma_spread"] = _pct_change(sma_10, sma_50, 0.0)

    # Trend alignment: +1 if c > sma10 > sma20 > sma50 (bullish), -1 if reverse
    alignment = 0.0
    if c > sma_10 > sma_20 > sma_50:
        alignment = 1.0
    elif c < sma_10 < sma_20 < sma_50:
        alignment = -1.0
    elif c > sma_20 > sma_50:
        alignment = 0.5
    elif c < sma_20 < sma_50:
        alignment = -0.5

    features["spy_trend_alignment"] = alignment

    return features


# ============================================================================
# MOMENTUM FEATURES (15)
# ============================================================================

def _extract_momentum_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract momentum features.

    Features:
        spy_momentum_1: 1-bar momentum (%)
        spy_momentum_3: 3-bar momentum (%)
        spy_momentum_5: 5-bar momentum (%)
        spy_momentum_10: 10-bar momentum (%)
        spy_momentum_20: 20-bar momentum (%)
        spy_acceleration: Rate of change of momentum
        spy_rsi_5: 5-period RSI
        spy_rsi_9: 9-period RSI
        spy_rsi_14: 14-period RSI
        spy_rsi_21: 21-period RSI
        spy_stochastic_k: Stochastic %K
        spy_stochastic_d: Stochastic %D (smoothed %K)
        spy_williams_r: Williams %R
        spy_rsi_divergence: RSI divergence indicator (-1, 0, 1)
        spy_momentum_regime: Momentum regime classification
    """
    features = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    c = float(close[-1]) if n > 0 else 0.0

    # Multi-period momentum
    def momentum(lookback):
        if n <= lookback:
            return 0.0
        return _pct_change(c, float(close[-(lookback + 1)]), 0.0)

    features["spy_momentum_1"] = momentum(1)
    features["spy_momentum_3"] = momentum(3)
    features["spy_momentum_5"] = momentum(5)
    features["spy_momentum_10"] = momentum(10)
    features["spy_momentum_20"] = momentum(20)

    # Acceleration (change in momentum)
    if n > 10:
        mom_5 = momentum(5)
        prev_mom_5 = _pct_change(float(close[-6]), float(close[-11]), 0.0) if n > 11 else 0.0
        features["spy_acceleration"] = mom_5 - prev_mom_5
    else:
        features["spy_acceleration"] = 0.0

    # RSI at multiple periods
    features["spy_rsi_5"] = get_last_valid(calc_rsi(close, 5), 50.0)
    features["spy_rsi_9"] = get_last_valid(calc_rsi(close, 9), 50.0)
    features["spy_rsi_14"] = get_last_valid(calc_rsi(close, 14), 50.0)
    features["spy_rsi_21"] = get_last_valid(calc_rsi(close, 21), 50.0)

    # Stochastic oscillator (14-period default)
    period = 14
    if n >= period:
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        stoch_k = safe_divide(c - lowest_low, highest_high - lowest_low, 0.5) * 100

        # %D is 3-period SMA of %K - approximate with last 3 closes
        if n >= period + 2:
            stoch_k_values = []
            for i in range(3):
                idx = -(1 + i)
                if abs(idx) <= n:
                    ll = np.min(low[max(0, n + idx - period):n + idx + 1])
                    hh = np.max(high[max(0, n + idx - period):n + idx + 1])
                    sk = safe_divide(close[idx] - ll, hh - ll, 0.5) * 100
                    stoch_k_values.append(sk)
            stoch_d = np.mean(stoch_k_values) if stoch_k_values else stoch_k
        else:
            stoch_d = stoch_k
    else:
        stoch_k = 50.0
        stoch_d = 50.0

    features["spy_stochastic_k"] = stoch_k
    features["spy_stochastic_d"] = stoch_d

    # Williams %R
    if n >= period:
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        williams_r = safe_divide(highest_high - c, highest_high - lowest_low, 0.5) * -100
    else:
        williams_r = -50.0

    features["spy_williams_r"] = williams_r

    # RSI divergence (simple: compare RSI direction to price direction)
    rsi_vals = calc_rsi(close, 14)
    if n >= 5:
        price_trend = np.sign(close[-1] - close[-5])
        rsi_trend = np.sign(rsi_vals[-1] - rsi_vals[-5]) if np.isfinite(rsi_vals[-5]) else 0

        if price_trend > 0 and rsi_trend < 0:
            divergence = -1  # Bearish divergence
        elif price_trend < 0 and rsi_trend > 0:
            divergence = 1  # Bullish divergence
        else:
            divergence = 0
    else:
        divergence = 0

    features["spy_rsi_divergence"] = float(divergence)

    # Momentum regime
    # 2=strong bullish, 1=bullish, 0=neutral, -1=bearish, -2=strong bearish
    mom_10 = features["spy_momentum_10"]
    rsi_14 = features["spy_rsi_14"]

    if mom_10 > 5 and rsi_14 > 70:
        regime = 2
    elif mom_10 > 2 and rsi_14 > 55:
        regime = 1
    elif mom_10 < -5 and rsi_14 < 30:
        regime = -2
    elif mom_10 < -2 and rsi_14 < 45:
        regime = -1
    else:
        regime = 0

    features["spy_momentum_regime"] = float(regime)

    return features


# ============================================================================
# VOLATILITY FEATURES (8)
# ============================================================================

def _extract_volatility_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract volatility features.

    Features:
        spy_atr_14: 14-period Average True Range
        spy_atr_pct: ATR as % of close
        spy_volatility_5: 5-period returns volatility (std)
        spy_volatility_20: 20-period returns volatility (std)
        spy_volatility_ratio: Short-term / long-term volatility ratio
        spy_range_pct_5: Average 5-day high-low range as % of close
        spy_range_pct_20: Average 20-day high-low range as % of close
        spy_volatility_regime: Volatility regime (0=low, 1=normal, 2=high)
    """
    features = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    c = float(close[-1]) if n > 0 else 1.0

    # ATR
    atr_vals = calc_atr(high, low, close, 14)
    atr_14 = get_last_valid(atr_vals, 0.0)

    features["spy_atr_14"] = atr_14
    features["spy_atr_pct"] = safe_divide(atr_14, c, 0.0) * 100

    # Returns-based volatility
    if n >= 6:
        returns = np.diff(close[-6:]) / close[-6:-1]
        vol_5 = float(np.std(returns)) * 100 if len(returns) > 0 else 0.0
    else:
        vol_5 = 0.0

    if n >= 21:
        returns = np.diff(close[-21:]) / close[-21:-1]
        vol_20 = float(np.std(returns)) * 100 if len(returns) > 0 else 0.0
    else:
        vol_20 = vol_5

    features["spy_volatility_5"] = vol_5
    features["spy_volatility_20"] = vol_20
    features["spy_volatility_ratio"] = safe_divide(vol_5, vol_20, 1.0)

    # Range-based volatility
    if n >= 5:
        ranges_5 = (high[-5:] - low[-5:]) / close[-5:] * 100
        features["spy_range_pct_5"] = float(np.mean(ranges_5))
    else:
        features["spy_range_pct_5"] = 0.0

    if n >= 20:
        ranges_20 = (high[-20:] - low[-20:]) / close[-20:] * 100
        features["spy_range_pct_20"] = float(np.mean(ranges_20))
    else:
        features["spy_range_pct_20"] = features["spy_range_pct_5"]

    # Volatility regime
    atr_pct = features["spy_atr_pct"]
    if atr_pct < 0.5:
        regime = 0  # Low volatility
    elif atr_pct < 1.5:
        regime = 1  # Normal
    else:
        regime = 2  # High volatility

    features["spy_volatility_regime"] = float(regime)

    return features


# ============================================================================
# TECHNICAL FEATURES (20)
# ============================================================================

def _extract_technical_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract technical indicator features.

    Features:
        spy_macd_line: MACD line (12-26 EMA)
        spy_macd_signal: MACD signal line (9-period EMA of MACD)
        spy_macd_histogram: MACD histogram
        spy_macd_crossover: MACD crossover signal (-1, 0, 1)
        spy_bb_pct_b: Bollinger Band %B
        spy_bb_width: Bollinger Band width (%)
        spy_bb_squeeze: Bollinger Band squeeze indicator
        spy_adx: Average Directional Index
        spy_plus_di: +DI line
        spy_minus_di: -DI line
        spy_di_crossover: DI crossover signal (-1, 0, 1)
        spy_obv_trend: OBV trend indicator (-1, 0, 1)
        spy_mfi: Money Flow Index
        spy_aroon_oscillator: Aroon oscillator
        spy_cci: Commodity Channel Index
        spy_pivot: Daily pivot point
        spy_r1: First resistance level
        spy_s1: First support level
        spy_price_vs_cloud: Price position vs Ichimoku cloud (-1 to 1)
        spy_keltner_position: Price position in Keltner Channel (0-1)
    """
    features = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
    n = len(close)
    c = float(close[-1]) if n > 0 else 1.0
    h = float(high[-1]) if n > 0 else c
    l = float(low[-1]) if n > 0 else c

    # MACD (12, 26, 9)
    ema_12 = calc_ema(close, 12)
    ema_26 = calc_ema(close, 26)
    macd_line_arr = ema_12 - ema_26
    macd_signal_arr = calc_ema(macd_line_arr[~np.isnan(macd_line_arr)], 9) if np.sum(~np.isnan(macd_line_arr)) >= 9 else np.array([0.0])

    macd_line = get_last_valid(macd_line_arr, 0.0)
    macd_signal = get_last_valid(macd_signal_arr, 0.0)
    macd_histogram = macd_line - macd_signal

    features["spy_macd_line"] = macd_line
    features["spy_macd_signal"] = macd_signal
    features["spy_macd_histogram"] = macd_histogram

    # MACD crossover
    if n >= 2 and len(macd_line_arr) >= 2 and len(macd_signal_arr) >= 2:
        prev_macd = macd_line_arr[-2] if np.isfinite(macd_line_arr[-2]) else macd_line
        prev_signal = macd_signal_arr[-2] if len(macd_signal_arr) >= 2 and np.isfinite(macd_signal_arr[-2]) else macd_signal

        if macd_line > macd_signal and prev_macd <= prev_signal:
            crossover = 1  # Bullish crossover
        elif macd_line < macd_signal and prev_macd >= prev_signal:
            crossover = -1  # Bearish crossover
        else:
            crossover = 0
    else:
        crossover = 0

    features["spy_macd_crossover"] = float(crossover)

    # Bollinger Bands (20, 2)
    bb_period = 20
    if n >= bb_period:
        bb_sma = np.mean(close[-bb_period:])
        bb_std = np.std(close[-bb_period:])
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std

        bb_pct_b = safe_divide(c - bb_lower, bb_upper - bb_lower, 0.5)
        bb_width = safe_divide(bb_upper - bb_lower, bb_sma, 0.0) * 100

        # Squeeze: compare current width to average width
        if n >= bb_period + 20:
            widths = []
            for i in range(20):
                idx = -(1 + i)
                sma_i = np.mean(close[max(0, n + idx - bb_period):n + idx + 1])
                std_i = np.std(close[max(0, n + idx - bb_period):n + idx + 1])
                w = safe_divide((sma_i + 2 * std_i) - (sma_i - 2 * std_i), sma_i, 0.0) * 100
                widths.append(w)
            avg_width = np.mean(widths)
            bb_squeeze = 1.0 if bb_width < avg_width * 0.75 else 0.0
        else:
            bb_squeeze = 0.0
    else:
        bb_pct_b = 0.5
        bb_width = 0.0
        bb_squeeze = 0.0

    features["spy_bb_pct_b"] = bb_pct_b
    features["spy_bb_width"] = bb_width
    features["spy_bb_squeeze"] = bb_squeeze

    # ADX and DMI (14-period)
    adx_period = 14
    if n >= adx_period + 1:
        # True Range
        tr = true_range(high, low, close)

        # +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with EMA
        tr_smooth = get_last_valid(calc_ema(tr, adx_period), 1.0)
        plus_dm_smooth = get_last_valid(calc_ema(plus_dm, adx_period), 0.0)
        minus_dm_smooth = get_last_valid(calc_ema(minus_dm, adx_period), 0.0)

        plus_di = safe_divide(plus_dm_smooth, tr_smooth, 0.0) * 100
        minus_di = safe_divide(minus_dm_smooth, tr_smooth, 0.0) * 100

        dx = safe_divide(abs(plus_di - minus_di), plus_di + minus_di, 0.0) * 100

        # ADX is smoothed DX
        dx_arr = np.zeros(n)
        dx_arr[-1] = dx
        adx_val = dx  # Simplified - would need full DX series for proper ADX
    else:
        plus_di = 25.0
        minus_di = 25.0
        adx_val = 25.0

    features["spy_adx"] = adx_val
    features["spy_plus_di"] = plus_di
    features["spy_minus_di"] = minus_di

    # DI crossover
    if plus_di > minus_di + 5:
        di_cross = 1
    elif minus_di > plus_di + 5:
        di_cross = -1
    else:
        di_cross = 0

    features["spy_di_crossover"] = float(di_cross)

    # OBV trend
    if n >= 5:
        obv = np.cumsum(np.where(np.diff(close) > 0, volume[1:],
                                  np.where(np.diff(close) < 0, -volume[1:], 0)))
        if len(obv) >= 5:
            obv_trend = np.sign(obv[-1] - obv[-5])
        else:
            obv_trend = 0
    else:
        obv_trend = 0

    features["spy_obv_trend"] = float(obv_trend)

    # Money Flow Index (14-period)
    mfi_period = 14
    if n >= mfi_period:
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume

        pos_mf = 0.0
        neg_mf = 0.0
        for i in range(n - mfi_period, n):
            if i > 0 and typical_price[i] > typical_price[i - 1]:
                pos_mf += raw_mf[i]
            elif i > 0:
                neg_mf += raw_mf[i]

        mfi = safe_divide(pos_mf, pos_mf + neg_mf, 0.5) * 100
    else:
        mfi = 50.0

    features["spy_mfi"] = mfi

    # Aroon oscillator (25-period)
    aroon_period = 25
    if n >= aroon_period:
        high_idx = np.argmax(high[-aroon_period:])
        low_idx = np.argmin(low[-aroon_period:])

        aroon_up = ((aroon_period - 1 - (aroon_period - 1 - high_idx)) / (aroon_period - 1)) * 100
        aroon_down = ((aroon_period - 1 - (aroon_period - 1 - low_idx)) / (aroon_period - 1)) * 100
        aroon_osc = aroon_up - aroon_down
    else:
        aroon_osc = 0.0

    features["spy_aroon_oscillator"] = aroon_osc

    # CCI (20-period)
    cci_period = 20
    if n >= cci_period:
        typical_price = (high + low + close) / 3
        tp_sma = np.mean(typical_price[-cci_period:])
        tp_mad = np.mean(np.abs(typical_price[-cci_period:] - tp_sma))
        cci = safe_divide(typical_price[-1] - tp_sma, 0.015 * tp_mad, 0.0)
    else:
        cci = 0.0

    features["spy_cci"] = cci

    # Pivot points (classic formula)
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    s1 = 2 * pivot - h

    features["spy_pivot"] = pivot
    features["spy_r1"] = r1
    features["spy_s1"] = s1

    # Ichimoku cloud position (simplified)
    # Conversion line (9), Base line (26), Span A, Span B (52)
    if n >= 52:
        conv_period = 9
        base_period = 26
        span_b_period = 52

        conv = (np.max(high[-conv_period:]) + np.min(low[-conv_period:])) / 2
        base = (np.max(high[-base_period:]) + np.min(low[-base_period:])) / 2
        span_a = (conv + base) / 2
        span_b = (np.max(high[-span_b_period:]) + np.min(low[-span_b_period:])) / 2

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)

        if c > cloud_top:
            cloud_pos = 1.0  # Above cloud
        elif c < cloud_bottom:
            cloud_pos = -1.0  # Below cloud
        else:
            cloud_pos = safe_divide(c - cloud_bottom, cloud_top - cloud_bottom, 0.0) * 2 - 1  # In cloud
    else:
        cloud_pos = 0.0

    features["spy_price_vs_cloud"] = cloud_pos

    # Keltner Channel position (20-period EMA, 2x ATR)
    kc_period = 20
    if n >= kc_period:
        kc_mid = get_last_valid(calc_ema(close, kc_period), c)
        kc_atr = get_last_valid(calc_atr(high, low, close, kc_period), 0.0)
        kc_upper = kc_mid + 2 * kc_atr
        kc_lower = kc_mid - 2 * kc_atr
        kc_pos = safe_divide(c - kc_lower, kc_upper - kc_lower, 0.5)
    else:
        kc_pos = 0.5

    features["spy_keltner_position"] = kc_pos

    return features


# ============================================================================
# TREND FEATURES (7)
# ============================================================================

def _extract_trend_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract trend features.

    Features:
        spy_higher_highs_count: Count of higher highs in last 10 bars
        spy_lower_lows_count: Count of lower lows in last 10 bars
        spy_up_bars_ratio_10: Ratio of up bars in last 10 bars
        spy_consecutive_up: Consecutive up bars count
        spy_consecutive_down: Consecutive down bars count
        spy_trend_strength: Overall trend strength (-1 to 1)
        spy_trend_direction: Trend direction (-1=down, 0=sideways, 1=up)
    """
    features = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    # Higher highs and lower lows in last 10 bars
    lookback = min(10, n - 1)
    if lookback >= 2:
        higher_highs = 0
        lower_lows = 0
        for i in range(1, lookback + 1):
            if high[-(i)] > high[-(i + 1)]:
                higher_highs += 1
            if low[-(i)] < low[-(i + 1)]:
                lower_lows += 1
    else:
        higher_highs = 0
        lower_lows = 0

    features["spy_higher_highs_count"] = float(higher_highs)
    features["spy_lower_lows_count"] = float(lower_lows)

    # Up bars ratio
    if n >= 10:
        up_bars = sum(1 for i in range(1, 11) if close[-i] > close[-(i + 1)])
        features["spy_up_bars_ratio_10"] = up_bars / 10.0
    else:
        features["spy_up_bars_ratio_10"] = 0.5

    # Consecutive up/down bars
    consecutive_up = 0
    consecutive_down = 0

    for i in range(1, min(n, 20)):
        if close[-i] > close[-(i + 1)] if (i + 1) <= n else False:
            if consecutive_down == 0:
                consecutive_up += 1
            else:
                break
        elif close[-i] < close[-(i + 1)] if (i + 1) <= n else False:
            if consecutive_up == 0:
                consecutive_down += 1
            else:
                break
        else:
            break

    features["spy_consecutive_up"] = float(consecutive_up)
    features["spy_consecutive_down"] = float(consecutive_down)

    # Trend strength (based on HH/LL ratio and up bar ratio)
    hh_ll_diff = higher_highs - lower_lows
    up_ratio = features["spy_up_bars_ratio_10"]

    # Combine metrics: [-1, 1] range
    trend_strength = (hh_ll_diff / max(lookback, 1)) * 0.5 + (up_ratio - 0.5) * 2 * 0.5
    trend_strength = float(np.clip(trend_strength, -1.0, 1.0))

    features["spy_trend_strength"] = trend_strength

    # Trend direction
    if trend_strength > 0.3:
        trend_dir = 1
    elif trend_strength < -0.3:
        trend_dir = -1
    else:
        trend_dir = 0

    features["spy_trend_direction"] = float(trend_dir)

    return features


# ============================================================================
# MARKET REGIME FEATURES (10)
# ============================================================================

def _extract_market_regime_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract market regime and additional market analysis features.

    Features:
        spy_intraday_range_position: Position within today's range (0-1)
        spy_open_gap_filled: Whether gap from open has been filled (0/1)
        spy_daily_range_expansion: Range expansion vs average
        spy_price_acceleration: Second derivative of price
        spy_volume_price_trend: Volume-weighted price trend
        spy_buying_pressure: Close position in range adjusted by volume
        spy_roc_5: 5-period rate of change
        spy_roc_10: 10-period rate of change
        spy_efficiency_ratio: Price efficiency (directional movement vs total)
        spy_choppiness_index: Market choppiness indicator (0-100)
    """
    features = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
    n = len(close)

    if n < 3:
        return {
            "spy_intraday_range_position": 0.5,
            "spy_open_gap_filled": 0.0,
            "spy_daily_range_expansion": 1.0,
            "spy_price_acceleration": 0.0,
            "spy_volume_price_trend": 0.0,
            "spy_buying_pressure": 0.5,
            "spy_roc_5": 0.0,
            "spy_roc_10": 0.0,
            "spy_efficiency_ratio": 0.0,
            "spy_choppiness_index": 50.0,
        }

    o = float(df["open"].iloc[-2])
    h = float(high[-2])
    l = float(low[-2])
    c = float(close[-2])
    v = float(volume[-2])

    # Intraday range position
    hl_range = h - l
    features["spy_intraday_range_position"] = safe_divide(c - l, hl_range, 0.5)

    # Gap filled check (did price return to previous close?)
    if n >= 3:
        prev_c = float(close[-3])
        gap = o - prev_c
        if gap > 0:  # Gap up
            gap_filled = 1.0 if l <= prev_c else 0.0
        elif gap < 0:  # Gap down
            gap_filled = 1.0 if h >= prev_c else 0.0
        else:
            gap_filled = 1.0  # No gap
    else:
        gap_filled = 0.0
    features["spy_open_gap_filled"] = gap_filled

    # Daily range expansion vs 10-day average
    if n >= 10:
        avg_range = np.mean(high[-10:] - low[-10:])
        features["spy_daily_range_expansion"] = safe_divide(hl_range, avg_range, 1.0)
    else:
        features["spy_daily_range_expansion"] = 1.0

    # Price acceleration (second derivative approximation)
    if n >= 3:
        velocity_1 = close[-1] - close[-2]
        velocity_2 = close[-2] - close[-3]
        features["spy_price_acceleration"] = velocity_1 - velocity_2
    else:
        features["spy_price_acceleration"] = 0.0

    # Volume-price trend (simplified VPT)
    if n >= 5:
        vpt = 0.0
        for i in range(1, 6):
            if close[-(i)] != close[-(i+1)] and close[-(i+1)] != 0:
                pct_change = (close[-(i)] - close[-(i+1)]) / close[-(i+1)]
                vpt += volume[-(i)] * pct_change
        features["spy_volume_price_trend"] = vpt / 1e6  # Normalize
    else:
        features["spy_volume_price_trend"] = 0.0

    # Buying pressure (Accumulation/Distribution approximation)
    if hl_range > 0:
        clv = ((c - l) - (h - c)) / hl_range  # Close Location Value
        features["spy_buying_pressure"] = (clv + 1) / 2  # Normalize to 0-1
    else:
        features["spy_buying_pressure"] = 0.5

    # Rate of change at different periods
    if n >= 6:
        features["spy_roc_5"] = _pct_change(c, float(close[-6]), 0.0)
    else:
        features["spy_roc_5"] = 0.0

    if n >= 11:
        features["spy_roc_10"] = _pct_change(c, float(close[-11]), 0.0)
    else:
        features["spy_roc_10"] = 0.0

    # Efficiency ratio (Kaufman)
    lookback = min(10, n - 1)
    if lookback >= 2:
        direction = abs(close[-1] - close[-(lookback + 1)])
        volatility = sum(abs(close[-(i)] - close[-(i + 1)]) for i in range(1, lookback + 1))
        features["spy_efficiency_ratio"] = safe_divide(direction, volatility, 0.0)
    else:
        features["spy_efficiency_ratio"] = 0.0

    # Choppiness index (simplified)
    chop_period = min(14, n - 1)
    if chop_period >= 2:
        sum_tr = sum(
            max(high[-(i)] - low[-(i)],
                abs(high[-(i)] - close[-(i + 1)]) if (i + 1) <= n else 0,
                abs(low[-(i)] - close[-(i + 1)]) if (i + 1) <= n else 0)
            for i in range(1, chop_period + 1)
        )
        highest_high = np.max(high[-chop_period:])
        lowest_low = np.min(low[-chop_period:])
        hl_range_period = highest_high - lowest_low

        if hl_range_period > 0 and sum_tr > 0:
            chop = 100 * np.log10(sum_tr / hl_range_period) / np.log10(chop_period)
            features["spy_choppiness_index"] = float(np.clip(chop, 0, 100))
        else:
            features["spy_choppiness_index"] = 50.0
    else:
        features["spy_choppiness_index"] = 50.0

    return features


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_spy_features(spy_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract all 80 SPY features from an OHLCV DataFrame.

    This function mirrors the TSLA feature extraction approach, providing
    comprehensive technical analysis features for SPY market data.

    Args:
        spy_df: pandas DataFrame with OHLCV columns:
                - open, high, low, close, volume
                Index should be datetime

    Returns:
        Dict[str, float] with 80 features, all prefixed with "spy_":

        BASIC PRICE (10):
            spy_close, spy_close_vs_open_pct, spy_high_low_range_pct,
            spy_close_vs_high_pct, spy_close_vs_low_pct, spy_body_pct,
            spy_gap_pct, spy_upper_shadow_pct, spy_lower_shadow_pct,
            spy_volume_vs_avg

        MOVING AVERAGES (10):
            spy_sma_10, spy_sma_20, spy_sma_50,
            spy_price_vs_sma_10, spy_price_vs_sma_20, spy_price_vs_sma_50,
            spy_sma_10_vs_sma_20, spy_sma_20_vs_sma_50,
            spy_ma_spread, spy_trend_alignment

        MOMENTUM (15):
            spy_momentum_1, spy_momentum_3, spy_momentum_5,
            spy_momentum_10, spy_momentum_20, spy_acceleration,
            spy_rsi_5, spy_rsi_9, spy_rsi_14, spy_rsi_21,
            spy_stochastic_k, spy_stochastic_d, spy_williams_r,
            spy_rsi_divergence, spy_momentum_regime

        VOLATILITY (8):
            spy_atr_14, spy_atr_pct, spy_volatility_5, spy_volatility_20,
            spy_volatility_ratio, spy_range_pct_5, spy_range_pct_20,
            spy_volatility_regime

        TECHNICAL (20):
            spy_macd_line, spy_macd_signal, spy_macd_histogram,
            spy_macd_crossover, spy_bb_pct_b, spy_bb_width, spy_bb_squeeze,
            spy_adx, spy_plus_di, spy_minus_di, spy_di_crossover,
            spy_obv_trend, spy_mfi, spy_aroon_oscillator, spy_cci,
            spy_pivot, spy_r1, spy_s1, spy_price_vs_cloud, spy_keltner_position

        TREND (7):
            spy_higher_highs_count, spy_lower_lows_count,
            spy_up_bars_ratio_10, spy_consecutive_up, spy_consecutive_down,
            spy_trend_strength, spy_trend_direction

        MARKET REGIME (10):
            spy_intraday_range_position, spy_open_gap_filled,
            spy_daily_range_expansion, spy_price_acceleration,
            spy_volume_price_trend, spy_buying_pressure,
            spy_roc_5, spy_roc_10, spy_efficiency_ratio, spy_choppiness_index

    Example:
        >>> import pandas as pd
        >>> # Assume spy_data is a DataFrame with OHLCV columns
        >>> features = extract_spy_features(spy_data)
        >>> print(f"RSI: {features['spy_rsi_14']:.2f}")
        >>> print(f"MACD: {features['spy_macd_line']:.4f}")
        >>> print(f"Trend: {features['spy_trend_direction']}")

    Notes:
        - All features are guaranteed to be finite (no NaN/inf)
        - Missing data is handled gracefully with sensible defaults
        - Volume column is optional (defaults to 1.0 if missing)
    """
    # Validate input
    if spy_df is None or len(spy_df) == 0:
        return _get_default_features()

    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in spy_df.columns:
            return _get_default_features()

    # Extract all feature groups
    features: Dict[str, float] = {}

    # Basic price features (10)
    features.update(_extract_basic_price_features(spy_df))

    # Moving average features (10)
    features.update(_extract_moving_average_features(spy_df))

    # Momentum features (15)
    features.update(_extract_momentum_features(spy_df))

    # Volatility features (8)
    features.update(_extract_volatility_features(spy_df))

    # Technical features (20)
    features.update(_extract_technical_features(spy_df))

    # Trend features (7)
    features.update(_extract_trend_features(spy_df))

    # Market regime features (10)
    features.update(_extract_market_regime_features(spy_df))

    # Final validation: ensure all values are finite
    for key, value in features.items():
        if not np.isfinite(value):
            features[key] = 0.0

    return features


def _get_default_features() -> Dict[str, float]:
    """Return default feature values when input is invalid."""
    return {
        # Basic price (10)
        "spy_close": 0.0,
        "spy_close_vs_open_pct": 0.0,
        "spy_high_low_range_pct": 0.0,
        "spy_close_vs_high_pct": 0.5,
        "spy_close_vs_low_pct": 0.5,
        "spy_body_pct": 0.0,
        "spy_gap_pct": 0.0,
        "spy_upper_shadow_pct": 0.0,
        "spy_lower_shadow_pct": 0.0,
        "spy_volume_vs_avg": 1.0,
        # Moving averages (10)
        "spy_sma_10": 0.0,
        "spy_sma_20": 0.0,
        "spy_sma_50": 0.0,
        "spy_price_vs_sma_10": 0.0,
        "spy_price_vs_sma_20": 0.0,
        "spy_price_vs_sma_50": 0.0,
        "spy_sma_10_vs_sma_20": 0.0,
        "spy_sma_20_vs_sma_50": 0.0,
        "spy_ma_spread": 0.0,
        "spy_trend_alignment": 0.0,
        # Momentum (15)
        "spy_momentum_1": 0.0,
        "spy_momentum_3": 0.0,
        "spy_momentum_5": 0.0,
        "spy_momentum_10": 0.0,
        "spy_momentum_20": 0.0,
        "spy_acceleration": 0.0,
        "spy_rsi_5": 50.0,
        "spy_rsi_9": 50.0,
        "spy_rsi_14": 50.0,
        "spy_rsi_21": 50.0,
        "spy_stochastic_k": 50.0,
        "spy_stochastic_d": 50.0,
        "spy_williams_r": -50.0,
        "spy_rsi_divergence": 0.0,
        "spy_momentum_regime": 0.0,
        # Volatility (8)
        "spy_atr_14": 0.0,
        "spy_atr_pct": 0.0,
        "spy_volatility_5": 0.0,
        "spy_volatility_20": 0.0,
        "spy_volatility_ratio": 1.0,
        "spy_range_pct_5": 0.0,
        "spy_range_pct_20": 0.0,
        "spy_volatility_regime": 1.0,
        # Technical (20)
        "spy_macd_line": 0.0,
        "spy_macd_signal": 0.0,
        "spy_macd_histogram": 0.0,
        "spy_macd_crossover": 0.0,
        "spy_bb_pct_b": 0.5,
        "spy_bb_width": 0.0,
        "spy_bb_squeeze": 0.0,
        "spy_adx": 25.0,
        "spy_plus_di": 25.0,
        "spy_minus_di": 25.0,
        "spy_di_crossover": 0.0,
        "spy_obv_trend": 0.0,
        "spy_mfi": 50.0,
        "spy_aroon_oscillator": 0.0,
        "spy_cci": 0.0,
        "spy_pivot": 0.0,
        "spy_r1": 0.0,
        "spy_s1": 0.0,
        "spy_price_vs_cloud": 0.0,
        "spy_keltner_position": 0.5,
        # Trend (7)
        "spy_higher_highs_count": 0.0,
        "spy_lower_lows_count": 0.0,
        "spy_up_bars_ratio_10": 0.5,
        "spy_consecutive_up": 0.0,
        "spy_consecutive_down": 0.0,
        "spy_trend_strength": 0.0,
        "spy_trend_direction": 0.0,
        # Market regime (10)
        "spy_intraday_range_position": 0.5,
        "spy_open_gap_filled": 0.0,
        "spy_daily_range_expansion": 1.0,
        "spy_price_acceleration": 0.0,
        "spy_volume_price_trend": 0.0,
        "spy_buying_pressure": 0.5,
        "spy_roc_5": 0.0,
        "spy_roc_10": 0.0,
        "spy_efficiency_ratio": 0.0,
        "spy_choppiness_index": 50.0,
    }


def get_spy_feature_names() -> list:
    """
    Get ordered list of all SPY feature names.

    Returns:
        List of 80 feature name strings in consistent order
    """
    return list(_get_default_features().keys())


def get_spy_feature_count() -> int:
    """
    Get total number of SPY features.

    Returns:
        80 (the total feature count)
    """
    return len(_get_default_features())


# ============================================================================
# TF-PREFIXED EXTRACTION FUNCTIONS
# ============================================================================

def extract_spy_features_tf(
    spy_df: pd.DataFrame,
    tf: str
) -> Dict[str, float]:
    """
    Extract SPY features with TF prefix.

    Args:
        spy_df: SPY OHLCV DataFrame (already resampled to target TF)
        tf: Timeframe name (e.g., 'daily', '1h', '5min')

    Returns:
        Dict with keys like 'daily_spy_close', '1h_spy_rsi_14', 'weekly_spy_macd_line'

    Note: Features are double-prefixed: tf + spy_
          e.g., 'daily_spy_momentum_5' not just 'daily_momentum_5'
    """
    base_features = extract_spy_features(spy_df)
    prefix = f"{tf}_"
    return {f"{prefix}{k}": v for k, v in base_features.items()}


def get_spy_feature_names_tf(tf: str) -> List[str]:
    """Get feature names with TF prefix."""
    base_names = get_spy_feature_names()
    prefix = f"{tf}_"
    return [f"{prefix}{name}" for name in base_names]


def get_all_spy_feature_names() -> List[str]:
    """Get ALL SPY feature names across all TFs."""
    from v15.config import TIMEFRAMES

    all_names = []
    for tf in TIMEFRAMES:
        all_names.extend(get_spy_feature_names_tf(tf))
    return all_names


def get_total_spy_features() -> int:
    """Total SPY features: 80 * 10 TFs = 800"""
    return 80 * 10
