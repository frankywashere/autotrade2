"""
SPY Feature Extraction Module

Extracts SPY features that include both SPY-specific market features and
shared technical indicators from technical.py for feature parity with TSLA.
All features are prefixed with "spy_" and return a flat Dict[str, float].

Features Categories (135 total = 60 SPY-specific + 77 shared technical - 2 overlapping):

SPY-SPECIFIC FEATURES (60):
- BASIC PRICE (10): Close, gap, range, shadows, volume
- MOVING AVERAGES (10): SMA at multiple periods, price vs SMA, trend alignment
- MOMENTUM (15): Multi-period momentum, RSI variants, stochastic, divergence
- VOLATILITY (8): ATR, volatility at multiple periods, range, regime
- TREND (7): Higher highs/lower lows counts, up/down bars, consecutive, trend strength
- MARKET REGIME (10): Intraday position, gap, expansion, acceleration, VPT, efficiency, choppiness

SHARED TECHNICAL FEATURES (77 from technical.py):
- MACD (5): macd_line, macd_signal, macd_histogram, macd_crossover, macd_divergence
- Bollinger Bands (8): bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b, etc.
- Keltner (5): keltner_upper, keltner_middle, keltner_lower, keltner_width, keltner_position
- ADX (4): adx, plus_di, minus_di, di_crossover
- Ichimoku (6): tenkan, kijun, senkou_a, senkou_b, price_vs_cloud, cloud_thickness
- Volume Indicators (10): obv, obv_trend, mfi, accumulation_dist, chaikin_mf, etc.
- Other Oscillators (8): aroon_up, aroon_down, aroon_oscillator, trix, etc.
- Pivot Points (7): pivot, r1, r2, r3, s1, s2, s3
- Fibonacci (6): fib_236, fib_382, fib_500, fib_618, fib_786, nearest_fib_distance
- Candlestick Patterns (12): is_doji, is_hammer, is_shooting_star, etc.
- Additional (6): cci, roc_5, roc_10, roc_20, price_channel_upper, price_channel_lower
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
from .technical import extract_technical_features, get_technical_feature_names


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
    Extract all 135 SPY features from an OHLCV DataFrame.

    This function provides comprehensive technical analysis features for SPY,
    combining SPY-specific market features with shared technical indicators
    from technical.py for feature parity with TSLA.

    Args:
        spy_df: pandas DataFrame with OHLCV columns:
                - open, high, low, close, volume
                Index should be datetime

    Returns:
        Dict[str, float] with 135 features, all prefixed with "spy_":

        SPY-SPECIFIC FEATURES (60):

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

        TREND (7):
            spy_higher_highs_count, spy_lower_lows_count,
            spy_up_bars_ratio_10, spy_consecutive_up, spy_consecutive_down,
            spy_trend_strength, spy_trend_direction

        MARKET REGIME (10):
            spy_intraday_range_position, spy_open_gap_filled,
            spy_daily_range_expansion, spy_price_acceleration,
            spy_volume_price_trend, spy_buying_pressure,
            spy_roc_5, spy_roc_10, spy_efficiency_ratio, spy_choppiness_index

        SHARED TECHNICAL FEATURES (77 from technical.py):
            See technical.py for full list. Includes MACD (5), Bollinger (8),
            Keltner (5), ADX (4), Ichimoku (6), Volume indicators (10),
            Oscillators (8), Pivot Points (7), Fibonacci (6),
            Candlestick Patterns (12), Additional (6).

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
        - Technical features from technical.py ensure parity with TSLA
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

    # Trend features (7)
    features.update(_extract_trend_features(spy_df))

    # Market regime features (10)
    features.update(_extract_market_regime_features(spy_df))

    # Extract shared technical indicators (same as TSLA)
    # This ensures feature parity between assets
    tech_features = extract_technical_features(spy_df)
    features.update({f'spy_{k}': v for k, v in tech_features.items()})

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
        # Shared technical features (77 from technical.py)
        # MACD (5)
        "spy_macd_line": 0.0,
        "spy_macd_signal": 0.0,
        "spy_macd_histogram": 0.0,
        "spy_macd_crossover": 0.0,
        "spy_macd_divergence": 0.0,
        # Bollinger Bands (8)
        "spy_bb_upper": 0.0,
        "spy_bb_middle": 0.0,
        "spy_bb_lower": 0.0,
        "spy_bb_width": 0.0,
        "spy_bb_pct_b": 0.5,
        "spy_price_vs_bb_upper": 0.0,
        "spy_price_vs_bb_lower": 0.0,
        "spy_bb_squeeze": 0.0,
        # Keltner (5)
        "spy_keltner_upper": 0.0,
        "spy_keltner_middle": 0.0,
        "spy_keltner_lower": 0.0,
        "spy_keltner_width": 0.0,
        "spy_keltner_position": 0.5,
        # ADX (4)
        "spy_adx": 0.0,
        "spy_plus_di": 0.0,
        "spy_minus_di": 0.0,
        "spy_di_crossover": 0.0,
        # Ichimoku (6)
        "spy_tenkan": 0.0,
        "spy_kijun": 0.0,
        "spy_senkou_a": 0.0,
        "spy_senkou_b": 0.0,
        "spy_price_vs_cloud": 0.0,
        "spy_cloud_thickness": 0.0,
        # Volume Indicators (10)
        "spy_obv": 0.0,
        "spy_obv_trend": 0.0,
        "spy_obv_divergence": 0.0,
        "spy_mfi": 50.0,
        "spy_mfi_divergence": 0.0,
        "spy_accumulation_dist": 0.0,
        "spy_chaikin_mf": 0.0,
        "spy_force_index": 0.0,
        "spy_volume_oscillator": 0.0,
        "spy_vwap_distance": 0.0,
        # Other Oscillators (8)
        "spy_aroon_up": 0.0,
        "spy_aroon_down": 0.0,
        "spy_aroon_oscillator": 0.0,
        "spy_trix": 0.0,
        "spy_ultimate_oscillator": 50.0,
        "spy_ppo": 0.0,
        "spy_dpo": 0.0,
        "spy_cmo": 0.0,
        # Pivot Points (7)
        "spy_pivot": 0.0,
        "spy_r1": 0.0,
        "spy_r2": 0.0,
        "spy_r3": 0.0,
        "spy_s1": 0.0,
        "spy_s2": 0.0,
        "spy_s3": 0.0,
        # Fibonacci (6)
        "spy_fib_236": 0.0,
        "spy_fib_382": 0.0,
        "spy_fib_500": 0.0,
        "spy_fib_618": 0.0,
        "spy_fib_786": 0.0,
        "spy_nearest_fib_distance": 0.0,
        # Candlestick Patterns (12)
        "spy_is_doji": 0.0,
        "spy_is_hammer": 0.0,
        "spy_is_shooting_star": 0.0,
        "spy_is_engulfing_bull": 0.0,
        "spy_is_engulfing_bear": 0.0,
        "spy_is_morning_star": 0.0,
        "spy_is_evening_star": 0.0,
        "spy_is_harami_bull": 0.0,
        "spy_is_harami_bear": 0.0,
        "spy_is_three_white": 0.0,
        "spy_is_three_black": 0.0,
        "spy_is_spinning_top": 0.0,
        # Additional (6)
        "spy_cci": 0.0,
        "spy_roc_5": 0.0,
        "spy_roc_10": 0.0,
        "spy_roc_20": 0.0,
        "spy_price_channel_upper": 0.0,
        "spy_price_channel_lower": 0.0,
    }


def get_spy_feature_names() -> list:
    """
    Get ordered list of all SPY feature names.

    Returns:
        List of 135 feature name strings in consistent order
        (60 SPY-specific + 77 shared technical - 2 overlapping = 135)
    """
    return list(_get_default_features().keys())


def get_spy_feature_count() -> int:
    """
    Get total number of SPY features.

    Returns:
        135 (60 SPY-specific + 77 shared technical - 2 overlapping)
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
    """Total SPY features: 135 * 10 TFs = 1350"""
    return get_spy_feature_count() * 10
