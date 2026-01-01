"""
RSI (Relative Strength Index) Calculator

Calculates RSI at various periods for momentum analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def calculate_rsi(prices: Union[np.ndarray, pd.Series], period: int = 14) -> float:
    """
    Calculate RSI for the most recent bar.

    Uses exponential smoothing to match calculate_rsi_series() methodology.
    Works with as few as 2 bars (minimum for calculating price changes).

    Args:
        prices: Array of close prices (most recent last)
        period: RSI period (default 14)

    Returns:
        RSI value (0-100)
    """
    prices = np.asarray(prices)

    if len(prices) < 2:
        return 50.0  # Neutral if not enough data for deltas

    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use exponential moving average
    alpha = 1.0 / period

    # Initialize with available data
    available_bars = min(period, len(gains))
    avg_gain = np.mean(gains[:available_bars])
    avg_loss = np.mean(losses[:available_bars])

    # Exponential smoothing for remaining bars (if any)
    for i in range(available_bars, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def calculate_rsi_with_confidence(
    prices: Union[np.ndarray, pd.Series],
    period: int = 14
) -> Tuple[float, float]:
    """
    Calculate RSI with confidence score.

    Confidence is based on data availability relative to the required period.
    The more bars available, the more reliable the RSI calculation.

    Args:
        prices: Array of close prices (most recent last)
        period: RSI period (default 14)

    Returns:
        (rsi_value, confidence) where confidence is 0.0-1.0
        based on how many bars are available vs required.
    """
    prices_array = np.asarray(prices)

    if len(prices_array) < 2:
        return (50.0, 0.0)  # Neutral, no confidence

    # Calculate confidence based on data availability
    # Full confidence when we have period+1 bars or more
    confidence = min(len(prices_array) / (period + 1), 1.0)

    # Calculate RSI (works with ≥2 bars using EMA)
    rsi = calculate_rsi(prices_array, period=period)

    return (float(rsi), float(confidence))


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

    # Calculate RSI with proper handling of edge cases
    rsi = np.zeros_like(avg_gain)
    for i in range(len(avg_gain)):
        if avg_loss[i] == 0:
            # No losses: RSI = 100 if gains exist, else 50
            rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))

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
