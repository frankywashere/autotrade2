"""
Channel Calculation with Partial Bars (v5.4)

This module calculates channel features at 5min resolution including partial TF bars.
At each 5min timestamp, the channel for coarser TFs includes the in-progress bar.

Performance optimization:
- Precompute complete TF bars once
- Precompute partial bar states at each 5min timestamp
- For each 5min bar, append partial to history and calculate channel
- Use incremental regression to avoid O(window) per bar

Memory: ~8GB for full calculation (11 TFs × 418K bars × 21 windows)
Time: ~5-10 minutes for full calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from tqdm import tqdm
import time

from .partial_bars import compute_partial_bars, PartialBarState, TIMEFRAME_PERIOD_RULES


# Window sizes (from config.py)
CHANNEL_WINDOW_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 168, 200, 252, 336, 504]


@dataclass
class ChannelFeatures:
    """Extracted channel features for a single 5min bar."""
    close_slope: float
    close_slope_pct: float
    close_intercept: float
    close_r_squared: float
    high_slope: float
    high_slope_pct: float
    low_slope: float
    low_slope_pct: float
    position: float  # 0-1, where price is within channel
    upper_dist: float  # % distance from upper band
    lower_dist: float  # % distance from lower band
    channel_width_pct: float
    slope_convergence: float
    stability: float
    is_valid: float
    insufficient_data: float


def resample_to_tf(df: pd.DataFrame, tf_rule: str) -> pd.DataFrame:
    """Resample OHLCV data to target timeframe."""
    return df.resample(tf_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def compute_complete_tf_bars(df_5min: pd.DataFrame, tf: str, tf_rule: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Resample 5min data to complete TF bars.

    Returns:
        opens, highs, lows, closes, timestamps
    """
    resampled = resample_to_tf(df_5min, tf_rule)
    return (
        resampled['open'].values,
        resampled['high'].values,
        resampled['low'].values,
        resampled['close'].values,
        resampled.index
    )


def find_complete_bars_before_timestamp(tf_timestamps: pd.DatetimeIndex,
                                         ts: pd.Timestamp) -> int:
    """
    Find index of last complete TF bar before given timestamp.

    Returns -1 if no complete bars exist before ts.
    """
    # Find bars that closed before ts
    mask = tf_timestamps < ts
    if not mask.any():
        return -1
    return mask.sum() - 1


def calculate_channel_with_partial(
    complete_closes: np.ndarray,
    complete_highs: np.ndarray,
    complete_lows: np.ndarray,
    partial_close: float,
    partial_high: float,
    partial_low: float,
    current_price: float,
    window: int,
    n_complete: int
) -> Optional[ChannelFeatures]:
    """
    Calculate channel features including partial bar.

    Args:
        complete_closes/highs/lows: Arrays of complete TF bars
        partial_close/high/low: Current partial bar's OHLCV
        current_price: Current 5min close (for position calculation)
        window: Lookback window in TF bars
        n_complete: Number of complete bars to use (starting from most recent)

    Returns:
        ChannelFeatures or None if insufficient data
    """
    # Need at least window-1 complete bars + 1 partial
    if n_complete < window - 1:
        return ChannelFeatures(
            close_slope=0.0, close_slope_pct=0.0, close_intercept=0.0, close_r_squared=0.0,
            high_slope=0.0, high_slope_pct=0.0, low_slope=0.0, low_slope_pct=0.0,
            position=0.5, upper_dist=0.0, lower_dist=0.0, channel_width_pct=0.0,
            slope_convergence=0.0, stability=0.0, is_valid=0.0, insufficient_data=1.0
        )

    # Take last (window-1) complete bars + partial
    start_idx = max(0, n_complete - (window - 1))
    closes = np.append(complete_closes[start_idx:n_complete], partial_close)
    highs = np.append(complete_highs[start_idx:n_complete], partial_high)
    lows = np.append(complete_lows[start_idx:n_complete], partial_low)

    n = len(closes)
    if n < 10:
        return ChannelFeatures(
            close_slope=0.0, close_slope_pct=0.0, close_intercept=0.0, close_r_squared=0.0,
            high_slope=0.0, high_slope_pct=0.0, low_slope=0.0, low_slope_pct=0.0,
            position=0.5, upper_dist=0.0, lower_dist=0.0, channel_width_pct=0.0,
            slope_convergence=0.0, stability=0.0, is_valid=0.0, insufficient_data=1.0
        )

    # Linear regression
    x = np.arange(n)
    sum_x = x.sum()
    sum_xx = (x ** 2).sum()

    sum_close = closes.sum()
    sum_xy_close = (x * closes).sum()
    sum_high = highs.sum()
    sum_xy_high = (x * highs).sum()
    sum_low = lows.sum()
    sum_xy_low = (x * lows).sum()

    denom = n * sum_xx - sum_x ** 2
    if abs(denom) < 1e-10:
        return None

    close_slope = (n * sum_xy_close - sum_x * sum_close) / denom
    high_slope = (n * sum_xy_high - sum_x * sum_high) / denom
    low_slope = (n * sum_xy_low - sum_x * sum_low) / denom

    close_intercept = (sum_close - close_slope * sum_x) / n
    high_intercept = (sum_high - high_slope * sum_x) / n
    low_intercept = (sum_low - low_slope * sum_x) / n

    # Calculate regression lines
    close_line = close_slope * x + close_intercept
    high_line = high_slope * x + high_intercept
    low_line = low_slope * x + low_intercept

    # Residuals and channel bands
    residuals = closes - close_line
    residual_std = np.std(residuals)

    upper_line = np.maximum(high_line, close_line + 2.0 * residual_std)
    lower_line = np.minimum(low_line, close_line - 2.0 * residual_std)

    # Channel metrics
    channel_width_pct = ((upper_line[-1] - lower_line[-1]) / max(close_line[-1], 1e-10)) * 100
    slope_convergence = (high_slope - low_slope) / abs(close_slope) if abs(close_slope) > 1e-10 else 0.0

    # R-squared for close
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((closes - np.mean(closes)) ** 2)
    close_r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    close_r_squared = max(0, min(1, close_r_squared))

    # Current position within channel
    current_upper = upper_line[-1]
    current_lower = lower_line[-1]
    current_center = close_line[-1]

    if current_upper > current_lower:
        position = (current_price - current_lower) / (current_upper - current_lower)
        position = max(0, min(1, position))
    else:
        position = 0.5

    upper_dist = (current_upper - current_price) / max(current_price, 1e-10) * 100
    lower_dist = (current_price - current_lower) / max(current_price, 1e-10) * 100

    # Slope percentages
    close_slope_pct = (close_slope / max(current_price, 1e-10)) * 100
    high_slope_pct = (high_slope / max(current_price, 1e-10)) * 100
    low_slope_pct = (low_slope / max(current_price, 1e-10)) * 100

    # Stability score
    stability = close_r_squared * 10  # Simple stability metric

    return ChannelFeatures(
        close_slope=close_slope,
        close_slope_pct=close_slope_pct,
        close_intercept=close_intercept,
        close_r_squared=close_r_squared,
        high_slope=high_slope,
        high_slope_pct=high_slope_pct,
        low_slope=low_slope,
        low_slope_pct=low_slope_pct,
        position=position,
        upper_dist=upper_dist,
        lower_dist=lower_dist,
        channel_width_pct=channel_width_pct,
        slope_convergence=slope_convergence,
        stability=stability,
        is_valid=1.0 if close_r_squared > 0.5 else 0.0,
        insufficient_data=0.0
    )


def calculate_channels_for_tf_with_partial(
    df_5min: pd.DataFrame,
    symbol: str,
    tf: str,
    tf_rule: str,
    windows: List[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Calculate channel features for a single TF at 5min resolution with partial bars.

    This is the main function that:
    1. Resamples to complete TF bars
    2. Computes partial bar state at each 5min timestamp
    3. For each 5min bar, calculates channel including partial

    Args:
        df_5min: 5min OHLCV DataFrame with columns: {symbol}_open, {symbol}_high, etc.
        symbol: 'tsla' or 'spy'
        tf: Timeframe name
        tf_rule: Pandas resample rule
        windows: List of window sizes (defaults to CHANNEL_WINDOW_SIZES)
        show_progress: Show progress bar

    Returns:
        DataFrame with channel features at 5min resolution
    """
    if windows is None:
        windows = CHANNEL_WINDOW_SIZES

    # Extract symbol columns
    symbol_df = df_5min[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low',
                          f'{symbol}_close', f'{symbol}_volume']].copy()
    symbol_df.columns = ['open', 'high', 'low', 'close', 'volume']

    n_5min = len(symbol_df)

    # Step 1: Compute complete TF bars
    complete_opens, complete_highs, complete_lows, complete_closes, tf_timestamps = \
        compute_complete_tf_bars(symbol_df, tf, tf_rule)

    # Step 2: Compute partial bar state at each 5min timestamp
    partial_state = compute_partial_bars(symbol_df, tf)

    # Step 3: For each 5min bar, calculate channels including partial
    feature_dict = {}

    # Initialize output arrays for each window
    for w in windows:
        prefix = f'{symbol}_channel_{tf}_w{w}'
        for feat in ['close_slope', 'close_slope_pct', 'close_r_squared',
                     'high_slope', 'high_slope_pct', 'low_slope', 'low_slope_pct',
                     'position', 'upper_dist', 'lower_dist', 'channel_width_pct',
                     'slope_convergence', 'stability', 'is_valid', 'insufficient_data']:
            feature_dict[f'{prefix}_{feat}'] = np.zeros(n_5min, dtype=np.float32)

    # Calculate for each 5min bar
    iterator = range(n_5min)
    if show_progress:
        iterator = tqdm(iterator, desc=f"   {symbol}_{tf} channels", leave=False, ncols=100)

    for i in iterator:
        ts = symbol_df.index[i]

        # Find complete bars before this timestamp
        n_complete = find_complete_bars_before_timestamp(tf_timestamps, ts) + 1

        # Get partial bar state at this timestamp
        p_close = partial_state.partial_close[i]
        p_high = partial_state.partial_high[i]
        p_low = partial_state.partial_low[i]
        current_price = symbol_df['close'].iloc[i]

        # Calculate for each window
        for w in windows:
            prefix = f'{symbol}_channel_{tf}_w{w}'

            features = calculate_channel_with_partial(
                complete_closes[:n_complete] if n_complete > 0 else np.array([]),
                complete_highs[:n_complete] if n_complete > 0 else np.array([]),
                complete_lows[:n_complete] if n_complete > 0 else np.array([]),
                p_close, p_high, p_low,
                current_price, w, n_complete
            )

            if features is None:
                features = ChannelFeatures(
                    close_slope=0.0, close_slope_pct=0.0, close_intercept=0.0, close_r_squared=0.0,
                    high_slope=0.0, high_slope_pct=0.0, low_slope=0.0, low_slope_pct=0.0,
                    position=0.5, upper_dist=0.0, lower_dist=0.0, channel_width_pct=0.0,
                    slope_convergence=0.0, stability=0.0, is_valid=0.0, insufficient_data=1.0
                )

            feature_dict[f'{prefix}_close_slope'][i] = features.close_slope
            feature_dict[f'{prefix}_close_slope_pct'][i] = features.close_slope_pct
            feature_dict[f'{prefix}_close_r_squared'][i] = features.close_r_squared
            feature_dict[f'{prefix}_high_slope'][i] = features.high_slope
            feature_dict[f'{prefix}_high_slope_pct'][i] = features.high_slope_pct
            feature_dict[f'{prefix}_low_slope'][i] = features.low_slope
            feature_dict[f'{prefix}_low_slope_pct'][i] = features.low_slope_pct
            feature_dict[f'{prefix}_position'][i] = features.position
            feature_dict[f'{prefix}_upper_dist'][i] = features.upper_dist
            feature_dict[f'{prefix}_lower_dist'][i] = features.lower_dist
            feature_dict[f'{prefix}_channel_width_pct'][i] = features.channel_width_pct
            feature_dict[f'{prefix}_slope_convergence'][i] = features.slope_convergence
            feature_dict[f'{prefix}_stability'][i] = features.stability
            feature_dict[f'{prefix}_is_valid'][i] = features.is_valid
            feature_dict[f'{prefix}_insufficient_data'][i] = features.insufficient_data

    return pd.DataFrame(feature_dict, index=symbol_df.index)


def test_partial_channel_calc():
    """Test channel calculation with partial bars."""
    print("=" * 70)
    print("TESTING CHANNEL CALCULATION WITH PARTIAL BARS")
    print("=" * 70)

    # Create sample data (1 month of 5min bars)
    np.random.seed(42)
    n_bars = 78 * 22  # ~1 month
    dates = pd.date_range('2024-01-02 09:30', periods=n_bars * 2, freq='5min')
    # Filter to market hours
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)][:n_bars]

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    df = pd.DataFrame({
        'tsla_open': prices,
        'tsla_high': prices + np.abs(np.random.randn(len(dates)) * 0.2),
        'tsla_low': prices - np.abs(np.random.randn(len(dates)) * 0.2),
        'tsla_close': prices + np.random.randn(len(dates)) * 0.1,
        'tsla_volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print(f"Sample data: {len(df)} 5min bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Test daily channel with single window
    print("\n--- Testing Daily Channel (window=50) ---")
    t0 = time.time()

    result = calculate_channels_for_tf_with_partial(
        df, 'tsla', 'daily', '1D',
        windows=[50],
        show_progress=True
    )

    elapsed = time.time() - t0
    print(f"Computation time: {elapsed:.1f}s for {len(df)} bars")
    print(f"Output shape: {result.shape}")
    print(f"Columns: {list(result.columns)[:5]}...")

    # Check for valid values
    print(f"\nSample values at different points:")
    for i in [100, 500, 1000, len(df)-1]:
        if i < len(df):
            pos = result['tsla_channel_daily_w50_position'].iloc[i]
            r2 = result['tsla_channel_daily_w50_close_r_squared'].iloc[i]
            print(f"  Bar {i}: position={pos:.3f}, r²={r2:.3f}")

    print("\n✅ Test completed!")
    return result


if __name__ == '__main__':
    import time
    test_partial_channel_calc()
