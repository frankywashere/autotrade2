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
    Calculate RSI for all bars (optimized).

    Args:
        prices: Array of close prices
        period: RSI period (default 14)

    Returns:
        Array of RSI values (first `period` values will be 50.0)
    """
    prices = np.asarray(prices)
    n = len(prices)

    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(prices)

    # Use maximum/minimum with 0 instead of np.where for efficiency
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)

    # Use exponential moving average for smoother RSI
    alpha = 1.0 / period
    decay = 1.0 - alpha

    # Number of RSI values to compute (from period-1 onwards in gains/losses)
    num_rsi = n - period

    # Pre-allocate only what we need for the output portion
    avg_gain_out = np.empty(num_rsi)
    avg_loss_out = np.empty(num_rsi)

    # Initialize with simple average of first `period` values
    avg_gain_prev = np.mean(gains[:period])
    avg_loss_prev = np.mean(losses[:period])
    avg_gain_out[0] = avg_gain_prev
    avg_loss_out[0] = avg_loss_prev

    # Exponential smoothing - only compute values we need
    for i in range(1, num_rsi):
        idx = period - 1 + i  # Index into gains/losses arrays
        avg_gain_prev = alpha * gains[idx] + decay * avg_gain_prev
        avg_loss_prev = alpha * losses[idx] + decay * avg_loss_prev
        avg_gain_out[i] = avg_gain_prev
        avg_loss_out[i] = avg_loss_prev

    # Vectorized RSI calculation with edge case handling
    # Handle division by zero: where avg_loss == 0
    zero_loss_mask = avg_loss_out == 0
    has_gain_mask = avg_gain_out > 0

    # Default RSI calculation (safe division)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.divide(avg_gain_out, avg_loss_out, out=np.zeros_like(avg_gain_out), where=~zero_loss_mask)
    rsi_values = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases: zero loss
    # RSI = 100 if gains > 0 and loss == 0, else 50 if both zero
    rsi_values[zero_loss_mask & has_gain_mask] = 100.0
    rsi_values[zero_loss_mask & ~has_gain_mask] = 50.0

    # Build result array: first `period` values are 50.0
    result = np.empty(n)
    result[:period] = 50.0
    result[period:] = rsi_values

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
