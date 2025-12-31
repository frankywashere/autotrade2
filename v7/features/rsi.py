"""
RSI (Relative Strength Index) Calculator

Calculates RSI at various periods for momentum analysis.
"""

import numpy as np
import pandas as pd
from typing import Union


def calculate_rsi(prices: Union[np.ndarray, pd.Series], period: int = 14) -> float:
    """
    Calculate RSI for the most recent bar.

    Args:
        prices: Array of close prices (most recent last)
        period: RSI period (default 14)

    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral if not enough data

    prices = np.asarray(prices)
    deltas = np.diff(prices[-(period + 1):])

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def calculate_rsi_series(prices: Union[np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Calculate RSI for all bars (vectorized).

    Args:
        prices: Array of close prices
        period: RSI period (default 14)

    Returns:
        Array of RSI values (first `period` values will be NaN)
    """
    prices = np.asarray(prices)
    n = len(prices)

    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use exponential moving average for smoother RSI
    alpha = 1.0 / period

    # Initialize with simple average
    avg_gain = np.zeros(n - 1)
    avg_loss = np.zeros(n - 1)

    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Exponential smoothing
    for i in range(period, n - 1):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i - 1]

    # Calculate RSI
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))

    # Pad with NaN for first period
    result = np.full(n, np.nan)
    result[period:] = rsi[period - 1:]

    # Replace NaN with neutral 50
    result = np.nan_to_num(result, nan=50.0)

    return result


def detect_rsi_divergence(prices: np.ndarray, rsi: np.ndarray, lookback: int = 10) -> int:
    """
    Detect RSI divergence (price vs RSI disagreement).

    Args:
        prices: Close prices
        rsi: RSI values
        lookback: Bars to look back for divergence

    Returns:
        1 = bullish divergence (price lower low, RSI higher low)
        -1 = bearish divergence (price higher high, RSI lower high)
        0 = no divergence
    """
    if len(prices) < lookback or len(rsi) < lookback:
        return 0

    recent_prices = prices[-lookback:]
    recent_rsi = rsi[-lookback:]

    price_trend = recent_prices[-1] - recent_prices[0]
    rsi_trend = recent_rsi[-1] - recent_rsi[0]

    # Bullish divergence: price down, RSI up
    if price_trend < 0 and rsi_trend > 5:
        return 1

    # Bearish divergence: price up, RSI down
    if price_trend > 0 and rsi_trend < -5:
        return -1

    return 0
