"""
Channel detection module for RSI Monitor.

Provides linear regression channel detection for TSLA across multiple
intraday timeframes. Uses scipy.stats.linregress for fitting and
computes channel bands, position, direction, and age.
"""

import logging
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# Backtest-derived win rates for display
CHANNEL_WIN_RATES = {
    # (timeframe, signal_type) -> win_rate_pct
    # RSI < 30 only (no channel)
    ('5m', 'rsi_only_buy'): 60.1,
    ('15m', 'rsi_only_buy'): 58.3,
    ('1h', 'rsi_only_buy'): 58.4,
    ('4h', 'rsi_only_buy'): 57.0,
    # RSI < 30 AND channel lower band
    ('5m', 'rsi_channel_buy'): 59.1,
    ('15m', 'rsi_channel_buy'): 59.3,
    ('1h', 'rsi_channel_buy'): 56.8,
    ('4h', 'rsi_channel_buy'): 66.7,
    # RSI < 30 inside valid channel (not necessarily at band)
    ('5m', 'rsi_in_channel_buy'): 60.9,
    ('15m', 'rsi_in_channel_buy'): 60.8,
    ('1h', 'rsi_in_channel_buy'): 59.4,
    ('4h', 'rsi_in_channel_buy'): 66.7,
    # Channel lower band touch only
    ('5m', 'channel_only_buy'): 59.6,
    ('15m', 'channel_only_buy'): 64.2,
    ('1h', 'channel_only_buy'): 54.5,
    ('4h', 'channel_only_buy'): 64.7,
    # Profit factors
    ('5m', 'rsi_only_pf'): 1.13,
    ('15m', 'rsi_only_pf'): 1.10,
    ('1h', 'rsi_only_pf'): 1.01,
    ('4h', 'rsi_only_pf'): 1.30,
    ('5m', 'rsi_channel_pf'): 1.69,
    ('15m', 'rsi_channel_pf'): 1.30,
    ('1h', 'rsi_channel_pf'): 1.28,
    ('4h', 'rsi_channel_pf'): 1.46,
}

# Default timeframes for channel analysis (skip 1d, 1wk -- useless for TSLA)
DEFAULT_TIMEFRAMES = ['5m', '15m', '1h', '4h']

# Period to fetch for each interval (enough bars for lookback)
FETCH_PERIODS = {
    '5m': '5d',
    '15m': '5d',
    '1h': '60d',
    '4h': '60d',  # fetched as 1h then resampled
}


def _invalid_channel() -> dict:
    """Return a default invalid channel result."""
    return {
        'valid': False,
        'r_squared': 0.0,
        'slope': 0.0,
        'intercept': 0.0,
        'std_dev': 0.0,
        'upper_band': 0.0,
        'lower_band': 0.0,
        'midline': 0.0,
        'position': 0.5,
        'direction': 'sideways',
        'width_pct': 0.0,
        'near_lower': False,
        'near_upper': False,
        'age': 0,
    }


def detect_channel(closes: np.ndarray, lookback: int = 50, r2_threshold: float = 0.7) -> dict:
    """
    Detect linear regression channel for the last `lookback` bars.

    Args:
        closes: Array of close prices.
        lookback: Number of bars to use for regression.
        r2_threshold: Minimum R-squared to consider channel valid.

    Returns:
        Dict with channel metrics. See module docstring for full field list.
    """
    try:
        if len(closes) < lookback:
            return _invalid_channel()

        window = closes[-lookback:]
        x = np.arange(lookback)

        result = linregress(x, window)
        slope = result.slope
        intercept = result.intercept
        r_squared = result.rvalue ** 2

        # Residuals and standard deviation
        fitted = slope * x + intercept
        residuals = window - fitted
        std_dev = float(np.std(residuals))

        # Current bar values (last point in the regression window)
        midline = slope * (lookback - 1) + intercept
        upper_band = midline + 2 * std_dev
        lower_band = midline - 2 * std_dev

        # Position within channel (0 = lower band, 0.5 = midline, 1 = upper band)
        width = upper_band - lower_band
        if width > 0:
            position = float(np.clip((closes[-1] - lower_band) / width, 0.0, 1.0))
        else:
            position = 0.5

        # Near band detection
        # Total width = 4 std_dev, 0.5 std_dev = 0.5/4 = 0.125 of total width
        # near_lower = position < 0.25 (within 0.5 std_dev fraction of lower)
        near_lower = position < 0.25
        near_upper = position > 0.75

        # Direction based on total slope as percentage of price
        if midline > 0:
            slope_pct = (slope / midline) * 100 * lookback
        else:
            slope_pct = 0.0

        if slope_pct > 2.0:
            direction = 'uptrend'
        elif slope_pct < -2.0:
            direction = 'downtrend'
        else:
            direction = 'sideways'

        # Channel width as percentage of price
        width_pct = (width / midline * 100) if midline > 0 else 0.0

        # Valid channel check
        valid = r_squared >= r2_threshold

        # Age: count consecutive bars (backwards) where R² stays above threshold
        # Cap at 50 steps back to avoid being slow
        age = 0
        if valid:
            age = 1  # current bar qualifies
            max_age_check = min(50, len(closes) - lookback)
            for step in range(1, max_age_check + 1):
                end_idx = len(closes) - step
                if end_idx < lookback:
                    break
                past_window = closes[end_idx - lookback:end_idx]
                past_x = np.arange(lookback)
                past_result = linregress(past_x, past_window)
                past_r2 = past_result.rvalue ** 2
                if past_r2 >= r2_threshold:
                    age += 1
                else:
                    break

        return {
            'valid': valid,
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept),
            'std_dev': std_dev,
            'upper_band': float(upper_band),
            'lower_band': float(lower_band),
            'midline': float(midline),
            'position': position,
            'direction': direction,
            'width_pct': float(width_pct),
            'near_lower': near_lower,
            'near_upper': near_upper,
            'age': age,
        }

    except Exception as e:
        logger.warning("detect_channel failed: %s", e)
        return _invalid_channel()


def analyze_channels(
    df: pd.DataFrame,
    lookback: int = 50,
    r2_threshold: float = 0.7,
    min_age: int = 20,
) -> dict:
    """
    Analyze channel for a single timeframe's DataFrame.

    Args:
        df: OHLCV DataFrame from DataFetcher (uppercase columns: Open, High, Low, Close, Volume).
        lookback: Regression lookback period.
        r2_threshold: Minimum R-squared for valid channel.
        min_age: Minimum consecutive bars above r2_threshold.

    Returns:
        Dict with 'channel' (detect_channel result) and 'meets_criteria' bool.
    """
    try:
        if df is None or df.empty or 'Close' not in df.columns:
            return {
                'channel': _invalid_channel(),
                'meets_criteria': False,
            }

        closes = df['Close'].dropna().values.astype(float)
        channel = detect_channel(closes, lookback=lookback, r2_threshold=r2_threshold)

        meets_criteria = (
            channel['valid']
            and channel['age'] >= min_age
            and channel['r_squared'] >= r2_threshold
        )

        return {
            'channel': channel,
            'meets_criteria': meets_criteria,
        }

    except Exception as e:
        logger.warning("analyze_channels failed: %s", e)
        return {
            'channel': _invalid_channel(),
            'meets_criteria': False,
        }


def get_channel_context_with_data(
    rsi_results: dict,
    data_fetcher,
    symbol: str = "TSLA",
    timeframes: Optional[list] = None,
    prepost: bool = True,
) -> tuple:
    """
    Get channel analysis across multiple timeframes, also returning raw OHLCV data.

    Returns:
        Tuple of (channel_results, ohlcv_data) where:
        - channel_results: Dict mapping timeframe -> channel analysis result.
        - ohlcv_data: Dict mapping timeframe -> pandas DataFrame (OHLCV).
    """
    if timeframes is None:
        timeframes = list(DEFAULT_TIMEFRAMES)

    skip_tf = {'1d', '1wk', '5d', '1mo'}
    timeframes = [tf for tf in timeframes if tf not in skip_tf]

    results = {}
    ohlcv_data = {}

    df_1h = None

    for tf in timeframes:
        try:
            if tf == '4h':
                if df_1h is None:
                    df_1h = data_fetcher.fetch(
                        symbol, interval='1h', period=FETCH_PERIODS['4h'], prepost=prepost
                    )
                if df_1h is None or df_1h.empty:
                    results[tf] = {
                        'channel': _invalid_channel(),
                        'meets_criteria': False,
                    }
                    continue

                df_4h = df_1h.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                }).dropna()

                ohlcv_data[tf] = df_4h
                results[tf] = analyze_channels(df_4h)
            else:
                period = FETCH_PERIODS.get(tf, '5d')
                fetch_interval = tf

                df = data_fetcher.fetch(
                    symbol, interval=fetch_interval, period=period, prepost=prepost
                )

                if tf == '1h':
                    df_1h = df

                ohlcv_data[tf] = df
                results[tf] = analyze_channels(df)

        except Exception as e:
            logger.warning("get_channel_context failed for %s/%s: %s", symbol, tf, e)
            results[tf] = {
                'channel': _invalid_channel(),
                'meets_criteria': False,
            }

    return results, ohlcv_data


def get_channel_context(
    rsi_results: dict,
    data_fetcher,
    symbol: str = "TSLA",
    timeframes: Optional[list] = None,
    prepost: bool = True,
) -> dict:
    """
    Get channel analysis across multiple timeframes for a symbol.
    Called from dashboard.py for TSLA only.

    Args:
        rsi_results: Not used directly, but available if needed later.
        data_fetcher: DataFetcher instance (has .fetch(symbol, interval, period, prepost)).
        symbol: Ticker symbol (default TSLA).
        timeframes: List of timeframe strings. Default: ['5m', '15m', '1h', '4h'].
        prepost: Whether to include pre/post market data.

    Returns:
        Dict mapping timeframe -> channel analysis result.
        Skips '1d' and '1wk' even if passed in timeframes.
    """
    results, _ = get_channel_context_with_data(
        rsi_results, data_fetcher, symbol, timeframes, prepost
    )
    return results
