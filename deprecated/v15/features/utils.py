"""
Feature Utilities for x14 Feature System v15

Safe mathematical operations and technical indicator calculations.
All functions guarantee no NaN or inf returns with sensible defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Any

# Type aliases for clarity
ArrayLike = Union[np.ndarray, pd.Series, list]
Numeric = Union[int, float, np.number]


# =============================================================================
# Safe Basic Operations
# =============================================================================

def safe_divide(
    numerator: Union[Numeric, ArrayLike],
    denominator: Union[Numeric, ArrayLike],
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Safe division that handles zero, NaN, and inf.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return when division is invalid

    Returns:
        Result of division or default value
    """
    # Handle scalar case efficiently
    if np.isscalar(numerator) and np.isscalar(denominator):
        if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
            return default
        result = numerator / denominator
        return float(result) if np.isfinite(result) else default

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)

    # Broadcast to common shape
    num, den = np.broadcast_arrays(num, den)

    # Create output array filled with default
    result = np.full_like(num, default, dtype=np.float64)

    # Identify valid divisions (non-zero, non-nan denominator)
    valid_mask = (den != 0) & np.isfinite(den) & np.isfinite(num)

    # Perform division only where valid
    if np.any(valid_mask):
        result[valid_mask] = num[valid_mask] / den[valid_mask]

    # Replace any inf results with default
    result[~np.isfinite(result)] = default

    # Return scalar if inputs were scalar-like
    if result.ndim == 0:
        return float(result.item())

    return result


def safe_pct_change(
    values: ArrayLike,
    periods: int = 1,
    default: float = 0.0
) -> np.ndarray:
    """
    Safe percentage change calculation.

    Args:
        values: Input values
        periods: Number of periods for change calculation
        default: Value for invalid calculations

    Returns:
        Array of percentage changes
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n <= periods or periods <= 0:
        return result

    # Calculate percentage change
    prev_values = arr[:-periods]
    curr_values = arr[periods:]

    # Safe division for percentage change
    pct = safe_divide(curr_values - prev_values, np.abs(prev_values), default=default)

    # Place results in correct positions
    if isinstance(pct, np.ndarray):
        result[periods:] = pct
    else:
        result[periods:] = pct

    return result


def safe_float(
    value: Union[Numeric, ArrayLike, None],
    default: float = 0.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    Safe conversion to float, handling None, NaN, inf.

    Args:
        value: Value to convert
        default: Default if conversion fails
        min_val: Optional minimum value (clip if below)
        max_val: Optional maximum value (clip if above)

    Returns:
        Float value or default
    """
    if value is None:
        return default

    try:
        if isinstance(value, (list, np.ndarray, pd.Series)):
            # For arrays, return first valid value or default
            arr = np.asarray(value, dtype=np.float64).flatten()
            valid = arr[np.isfinite(arr)]
            result = float(valid[0]) if len(valid) > 0 else default
        else:
            result = float(value)
            if not np.isfinite(result):
                return default

        # Apply optional clipping
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val

        return result
    except (TypeError, ValueError):
        return default


# =============================================================================
# Safe Array Operations
# =============================================================================

def safe_mean(values: ArrayLike, default: float = 0.0) -> float:
    """
    Safe mean calculation with NaN handling.

    Args:
        values: Input array
        default: Default if calculation fails

    Returns:
        Mean value or default
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return default

    result = np.mean(valid)
    return float(result) if np.isfinite(result) else default


def safe_std(values: ArrayLike, default: float = 0.0, ddof: int = 1) -> float:
    """
    Safe standard deviation calculation.

    Args:
        values: Input array
        default: Default if calculation fails
        ddof: Degrees of freedom

    Returns:
        Standard deviation or default
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) <= ddof:
        return default

    result = np.std(valid, ddof=ddof)
    return float(result) if np.isfinite(result) else default


def safe_min(values: ArrayLike, default: float = 0.0) -> float:
    """
    Safe minimum calculation.

    Args:
        values: Input array
        default: Default if no valid values

    Returns:
        Minimum value or default
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return default

    return float(np.min(valid))


def safe_max(values: ArrayLike, default: float = 0.0) -> float:
    """
    Safe maximum calculation.

    Args:
        values: Input array
        default: Default if no valid values

    Returns:
        Maximum value or default
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return default

    return float(np.max(valid))


def safe_sum(values: ArrayLike, default: float = 0.0) -> float:
    """
    Safe sum calculation.

    Args:
        values: Input array
        default: Default if no valid values

    Returns:
        Sum or default
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return default

    result = np.sum(valid)
    return float(result) if np.isfinite(result) else default


# =============================================================================
# Rolling Calculations
# =============================================================================

def rolling_mean(
    values: ArrayLike,
    window: int,
    min_periods: Optional[int] = None,
    default: float = 0.0
) -> np.ndarray:
    """
    Rolling mean calculation with NaN handling.

    Args:
        values: Input array
        window: Window size
        min_periods: Minimum observations required
        default: Default for invalid calculations

    Returns:
        Array of rolling means
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n == 0 or window <= 0:
        return result

    min_periods = min_periods if min_periods is not None else window
    min_periods = max(1, min(min_periods, window))

    # Use pandas for efficient rolling calculation
    series = pd.Series(arr)
    rolling = series.rolling(window=window, min_periods=min_periods).mean()

    # Replace NaN with default
    result = rolling.to_numpy()
    result[~np.isfinite(result)] = default

    return result


def rolling_std(
    values: ArrayLike,
    window: int,
    min_periods: Optional[int] = None,
    default: float = 0.0,
    ddof: int = 1
) -> np.ndarray:
    """
    Rolling standard deviation calculation.

    Args:
        values: Input array
        window: Window size
        min_periods: Minimum observations required
        default: Default for invalid calculations
        ddof: Degrees of freedom

    Returns:
        Array of rolling standard deviations
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n == 0 or window <= 0:
        return result

    min_periods = min_periods if min_periods is not None else window
    min_periods = max(1, min(min_periods, window))

    series = pd.Series(arr)
    rolling = series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)

    result = rolling.to_numpy()
    result[~np.isfinite(result)] = default

    return result


def rolling_correlation(
    values1: ArrayLike,
    values2: ArrayLike,
    window: int,
    min_periods: Optional[int] = None,
    default: float = 0.0
) -> np.ndarray:
    """
    Rolling correlation calculation.

    Args:
        values1: First input array
        values2: Second input array
        window: Window size
        min_periods: Minimum observations required
        default: Default for invalid calculations

    Returns:
        Array of rolling correlations
    """
    arr1 = np.asarray(values1, dtype=np.float64)
    arr2 = np.asarray(values2, dtype=np.float64)

    n = min(len(arr1), len(arr2))
    result = np.full(n, default, dtype=np.float64)

    if n == 0 or window <= 0:
        return result

    min_periods = min_periods if min_periods is not None else window
    min_periods = max(1, min(min_periods, window))

    series1 = pd.Series(arr1[:n])
    series2 = pd.Series(arr2[:n])

    rolling = series1.rolling(window=window, min_periods=min_periods).corr(series2)

    result = rolling.to_numpy()
    result[~np.isfinite(result)] = default

    return result


# =============================================================================
# Technical Indicators - Moving Averages
# =============================================================================

def calc_sma(
    values: ArrayLike,
    period: int,
    default: float = 0.0
) -> np.ndarray:
    """
    Simple Moving Average calculation.

    Args:
        values: Price or value array
        period: SMA period
        default: Default for invalid values

    Returns:
        Array of SMA values
    """
    return rolling_mean(values, window=period, min_periods=1, default=default)


def calc_ema(
    values: ArrayLike,
    period: int,
    default: float = 0.0
) -> np.ndarray:
    """
    Exponential Moving Average calculation.

    Args:
        values: Price or value array
        period: EMA period
        default: Default for invalid values

    Returns:
        Array of EMA values
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n == 0 or period <= 0:
        return result

    # Use pandas EMA for efficiency
    series = pd.Series(arr)
    ema = series.ewm(span=period, min_periods=1, adjust=False).mean()

    result = ema.to_numpy()
    result[~np.isfinite(result)] = default

    return result


# =============================================================================
# Technical Indicators - Momentum
# =============================================================================

def calc_rsi(
    values: ArrayLike,
    period: int = 14,
    default: float = 50.0
) -> np.ndarray:
    """
    Relative Strength Index calculation.

    Args:
        values: Price array (typically close prices)
        period: RSI period
        default: Default value (50 = neutral)

    Returns:
        Array of RSI values (0-100)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n <= 1 or period <= 0:
        return result

    # Calculate price changes
    delta = np.diff(arr, prepend=arr[0])

    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Calculate EMA of gains and losses
    avg_gains = calc_ema(gains, period, default=0.0)
    avg_losses = calc_ema(losses, period, default=0.0)

    # Calculate RS and RSI
    # When avg_losses is 0 and avg_gains > 0, RSI = 100
    # When both are 0, RSI = default (50)
    for i in range(n):
        if avg_losses[i] == 0:
            if avg_gains[i] > 0:
                result[i] = 100.0
            else:
                result[i] = default
        else:
            rs = avg_gains[i] / avg_losses[i]
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    # Ensure result is in valid range
    result = np.clip(result, 0.0, 100.0)
    result[~np.isfinite(result)] = default

    return result


def calc_momentum(
    values: ArrayLike,
    period: int = 10,
    default: float = 0.0
) -> np.ndarray:
    """
    Momentum calculation (price difference over period).

    Args:
        values: Price array
        period: Momentum period
        default: Default for invalid values

    Returns:
        Array of momentum values
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n <= period or period <= 0:
        return result

    # Momentum = current price - price n periods ago
    result[period:] = arr[period:] - arr[:-period]
    result[~np.isfinite(result)] = default

    return result


def calc_roc(
    values: ArrayLike,
    period: int = 10,
    default: float = 0.0
) -> np.ndarray:
    """
    Rate of Change calculation.

    Args:
        values: Price array
        period: ROC period
        default: Default for invalid values

    Returns:
        Array of ROC values (percentage)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n <= period or period <= 0:
        return result

    # ROC = (current - previous) / previous * 100
    prev_values = arr[:-period]
    curr_values = arr[period:]

    roc = safe_divide(curr_values - prev_values, np.abs(prev_values), default=0.0) * 100.0
    if isinstance(roc, np.ndarray):
        result[period:] = roc
    else:
        result[period:] = roc
    result[~np.isfinite(result)] = default

    return result


def calc_stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
    default: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator calculation.

    Args:
        high: High price array
        low: Low price array
        close: Close price array
        k_period: %K period
        d_period: %D smoothing period
        default: Default value (50 = neutral)

    Returns:
        Tuple of (%K, %D) arrays
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = min(len(h), len(l), len(c))
    k_result = np.full(n, default, dtype=np.float64)
    d_result = np.full(n, default, dtype=np.float64)

    if n <= k_period or k_period <= 0:
        return k_result, d_result

    # Truncate to same length
    h, l, c = h[:n], l[:n], c[:n]

    # Calculate rolling highest high and lowest low
    series_h = pd.Series(h)
    series_l = pd.Series(l)

    highest_high = series_h.rolling(window=k_period, min_periods=1).max().to_numpy()
    lowest_low = series_l.rolling(window=k_period, min_periods=1).min().to_numpy()

    # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    hl_range = highest_high - lowest_low
    k_values = safe_divide(c - lowest_low, hl_range, default=0.5)
    if isinstance(k_values, np.ndarray):
        k_result = k_values * 100.0
    else:
        k_result = np.full(n, k_values * 100.0, dtype=np.float64)
    k_result = np.clip(k_result, 0.0, 100.0)
    k_result[~np.isfinite(k_result)] = default

    # %D = SMA of %K
    d_result = calc_sma(k_result, d_period, default=default)
    d_result = np.clip(d_result, 0.0, 100.0)

    return k_result, d_result


# =============================================================================
# Technical Indicators - Volatility
# =============================================================================

def calc_atr(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 14,
    default: float = 0.0
) -> np.ndarray:
    """
    Average True Range calculation.

    Args:
        high: High price array
        low: Low price array
        close: Close price array
        period: ATR period
        default: Default for invalid values

    Returns:
        Array of ATR values
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = min(len(h), len(l), len(c))
    result = np.full(n, default, dtype=np.float64)

    if n <= 1 or period <= 0:
        return result

    # Truncate to same length
    h, l, c = h[:n], l[:n], c[:n]

    # Calculate True Range
    # TR = max(H-L, |H-Cp|, |L-Cp|) where Cp is previous close
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]

    tr1 = h - l  # High - Low
    tr2 = np.abs(h - prev_close)  # |High - Previous Close|
    tr3 = np.abs(l - prev_close)  # |Low - Previous Close|

    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    true_range[~np.isfinite(true_range)] = 0.0

    # ATR = EMA of True Range
    result = calc_ema(true_range, period, default=default)
    result[~np.isfinite(result)] = default

    return result


def calc_bollinger_bands(
    values: ArrayLike,
    period: int = 20,
    num_std: float = 2.0,
    default: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands calculation.

    Args:
        values: Price array (typically close)
        period: Moving average period
        num_std: Number of standard deviations
        default: Default for invalid values

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)

    upper = np.full(n, default, dtype=np.float64)
    middle = np.full(n, default, dtype=np.float64)
    lower = np.full(n, default, dtype=np.float64)

    if n == 0 or period <= 0:
        return upper, middle, lower

    # Middle band = SMA
    middle = rolling_mean(arr, window=period, min_periods=1, default=default)

    # Standard deviation
    std = rolling_std(arr, window=period, min_periods=1, default=0.0)

    # Upper and lower bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    # Ensure no invalid values
    upper[~np.isfinite(upper)] = default
    middle[~np.isfinite(middle)] = default
    lower[~np.isfinite(lower)] = default

    return upper, middle, lower


# =============================================================================
# Technical Indicators - Trend
# =============================================================================

def calc_macd(
    values: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    default: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence) calculation.

    Args:
        values: Price array (typically close)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        default: Default for invalid values

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)

    macd_line = np.full(n, default, dtype=np.float64)
    signal_line = np.full(n, default, dtype=np.float64)
    histogram = np.full(n, default, dtype=np.float64)

    if n == 0 or fast_period <= 0 or slow_period <= 0:
        return macd_line, signal_line, histogram

    # Fast and slow EMAs
    fast_ema = calc_ema(arr, fast_period, default=default)
    slow_ema = calc_ema(arr, slow_period, default=default)

    # MACD line = Fast EMA - Slow EMA
    macd_line = fast_ema - slow_ema
    macd_line[~np.isfinite(macd_line)] = default

    # Signal line = EMA of MACD line
    signal_line = calc_ema(macd_line, signal_period, default=default)
    signal_line[~np.isfinite(signal_line)] = default

    # Histogram = MACD line - Signal line
    histogram = macd_line - signal_line
    histogram[~np.isfinite(histogram)] = default

    return macd_line, signal_line, histogram


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_values(
    values: ArrayLike,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: float = 0.5
) -> np.ndarray:
    """
    Normalize values to 0-1 range.

    Args:
        values: Input array
        min_val: Minimum value (computed if None)
        max_val: Maximum value (computed if None)
        default: Default for invalid calculations

    Returns:
        Normalized array
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n == 0:
        return result

    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return result

    if min_val is None:
        min_val = float(np.min(valid))
    if max_val is None:
        max_val = float(np.max(valid))

    range_val = max_val - min_val
    if range_val == 0 or not np.isfinite(range_val):
        return result

    norm = safe_divide(arr - min_val, range_val, default=default)
    if isinstance(norm, np.ndarray):
        result = np.clip(norm, 0.0, 1.0)
    else:
        result = np.full(n, np.clip(norm, 0.0, 1.0), dtype=np.float64)
    result[~np.isfinite(result)] = default

    return result


def zscore(
    values: ArrayLike,
    window: Optional[int] = None,
    default: float = 0.0
) -> np.ndarray:
    """
    Z-score calculation (standard score).

    Args:
        values: Input array
        window: Rolling window (None for full array)
        default: Default for invalid calculations

    Returns:
        Array of z-scores
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    result = np.full(n, default, dtype=np.float64)

    if n == 0:
        return result

    if window is None:
        # Full array z-score
        mean = safe_mean(arr, default=0.0)
        std = safe_std(arr, default=1.0)
        if std == 0:
            return result
        z = safe_divide(arr - mean, std, default=default)
        if isinstance(z, np.ndarray):
            result = z
        else:
            result = np.full(n, z, dtype=np.float64)
    else:
        # Rolling z-score
        roll_mean = rolling_mean(arr, window=window, min_periods=1, default=0.0)
        roll_std = rolling_std(arr, window=window, min_periods=1, default=1.0)
        roll_std[roll_std == 0] = 1.0  # Avoid division by zero
        z = safe_divide(arr - roll_mean, roll_std, default=default)
        if isinstance(z, np.ndarray):
            result = z
        else:
            result = np.full(n, z, dtype=np.float64)

    result[~np.isfinite(result)] = default
    return result


def crossover(
    values1: ArrayLike,
    values2: ArrayLike,
    default: int = 0
) -> np.ndarray:
    """
    Detect crossovers between two series.

    Args:
        values1: First series
        values2: Second series
        default: Default value

    Returns:
        Array: 1 for bullish crossover, -1 for bearish, 0 otherwise
    """
    arr1 = np.asarray(values1, dtype=np.float64)
    arr2 = np.asarray(values2, dtype=np.float64)

    n = min(len(arr1), len(arr2))
    result = np.full(n, default, dtype=np.int32)

    if n <= 1:
        return result

    arr1, arr2 = arr1[:n], arr2[:n]

    # Previous comparison
    prev_above = np.roll(arr1 > arr2, 1)
    prev_above[0] = False

    curr_above = arr1 > arr2

    # Bullish crossover: was below, now above
    bullish = (~prev_above) & curr_above
    # Bearish crossover: was above, now below
    bearish = prev_above & (~curr_above)

    result[bullish] = 1
    result[bearish] = -1

    return result


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def apply_to_columns(
    df: pd.DataFrame,
    func: callable,
    columns: Optional[list] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Apply a function to multiple columns efficiently.

    Args:
        df: Input DataFrame
        func: Function to apply
        columns: Columns to process (None for all numeric)
        **kwargs: Additional arguments for func

    Returns:
        DataFrame with transformed columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result = df.copy()
    for col in columns:
        if col in df.columns:
            result[col] = func(df[col].values, **kwargs)

    return result


def ensure_finite(
    values: ArrayLike,
    default: float = 0.0
) -> np.ndarray:
    """
    Ensure all values in array are finite.

    Args:
        values: Input array
        default: Replacement for non-finite values

    Returns:
        Array with all finite values
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr.copy()  # Don't modify input
    arr[~np.isfinite(arr)] = default
    return arr


def get_last_valid(arr: ArrayLike, default: float = 0.0) -> float:
    """
    Get the last valid (non-NaN) value from an array.

    Args:
        arr: Input array
        default: Default if no valid values

    Returns:
        Last valid value or default
    """
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return default

    for i in range(len(arr) - 1, -1, -1):
        if np.isfinite(arr[i]):
            return float(arr[i])

    return default


# =============================================================================
# Legacy Compatibility Aliases
# =============================================================================

# Maintain compatibility with existing code
ema = calc_ema
sma = calc_sma
rsi = calc_rsi
atr = calc_atr


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> np.ndarray:
    """
    Calculate true range (legacy compatibility).

    Args:
        high: High price array
        low: Low price array
        close: Close price array

    Returns:
        Array of true range values
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = min(len(h), len(l), len(c))
    if n <= 1:
        return h[:n] - l[:n]

    h, l, c = h[:n], l[:n], c[:n]

    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]

    tr1 = h - l
    tr2 = np.abs(h - prev_close)
    tr3 = np.abs(l - prev_close)

    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[~np.isfinite(tr)] = 0.0

    return tr
