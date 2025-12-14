"""
Vectorized Channel Calculation with Partial Bars (v5.4)

This module calculates channel features at 5min resolution including partial TF bars.
Uses vectorized numpy operations for efficiency - ~100x faster than loop-based approach.

Key optimization: Process all 5min bars within a TF period in parallel
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

from .partial_bars import compute_partial_bars, PartialBarState, TIMEFRAME_PERIOD_RULES


def calculate_channel_features_vectorized(
    df_5min: pd.DataFrame,
    symbol: str,
    tf: str,
    tf_rule: str,
    window: int = 50,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Calculate channel features at 5min resolution with partial bars.
    Vectorized implementation - processes all bars in each TF period together.

    Args:
        df_5min: 5min OHLCV DataFrame
        symbol: 'tsla' or 'spy'
        tf: Timeframe name
        tf_rule: Pandas resample rule
        window: Lookback window in TF bars
        show_progress: Show progress bar

    Returns:
        DataFrame with channel features at 5min resolution
    """
    # Extract symbol columns
    symbol_df = df_5min[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low',
                          f'{symbol}_close', f'{symbol}_volume']].copy()
    symbol_df.columns = ['open', 'high', 'low', 'close', 'volume']

    n_5min = len(symbol_df)

    # FAST PATH: For 5min TF, use simple rolling regression (no partial bar concept)
    # Each 5min bar IS complete - no "partial" applies. Using the period loop would be O(n²)!
    if tf == '5min':
        return _calculate_5min_channel_features_rolling(symbol_df, symbol, window)

    # Resample to complete TF bars
    resampled = symbol_df.resample(tf_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    complete_closes = resampled['close'].values
    complete_highs = resampled['high'].values
    complete_lows = resampled['low'].values
    tf_timestamps = resampled.index
    n_complete = len(complete_closes)

    # Compute partial bar state at each 5min timestamp
    partial_state = compute_partial_bars(symbol_df, tf)

    # Initialize output arrays
    prefix = f'{symbol}_channel_{tf}_w{window}'
    output = {
        f'{prefix}_close_slope': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_close_slope_pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_close_r_squared': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_high_slope': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_low_slope': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_position': np.full(n_5min, 0.5, dtype=np.float32),
        f'{prefix}_upper_dist': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_lower_dist': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_channel_width_pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_stability': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_is_valid': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_insufficient_data': np.ones(n_5min, dtype=np.float32),
    }

    # Process by TF period (vectorized within each period)
    # Find unique periods and their boundaries
    period = symbol_df.index.to_period(TIMEFRAME_PERIOD_RULES.get(tf, tf))
    unique_periods = period.unique()

    iterator = range(len(unique_periods))
    if show_progress:
        iterator = tqdm(iterator, desc=f"   {symbol}_{tf}_w{window}", leave=False, ncols=100)

    for period_idx in iterator:
        current_period = unique_periods[period_idx]

        # Find 5min bars in this period
        period_mask = period == current_period
        bar_indices = np.where(period_mask)[0]
        if len(bar_indices) == 0:
            continue

        # Find complete TF bars before this period
        period_start_ts = symbol_df.index[bar_indices[0]]
        n_complete_before = (tf_timestamps < period_start_ts).sum()

        # Need at least window-1 complete bars + 1 partial
        if n_complete_before < window - 1:
            # Not enough data - leave as defaults (insufficient_data=1)
            continue

        # Get historical bars for regression
        start_idx = max(0, n_complete_before - (window - 1))
        hist_closes = complete_closes[start_idx:n_complete_before]
        hist_highs = complete_highs[start_idx:n_complete_before]
        hist_lows = complete_lows[start_idx:n_complete_before]
        n_hist = len(hist_closes)

        # Precompute sums from historical bars
        x_hist = np.arange(n_hist)
        sum_x_hist = x_hist.sum()
        sum_xx_hist = (x_hist ** 2).sum()
        sum_y_close_hist = hist_closes.sum()
        sum_xy_close_hist = (x_hist * hist_closes).sum()
        sum_y_high_hist = hist_highs.sum()
        sum_xy_high_hist = (x_hist * hist_highs).sum()
        sum_y_low_hist = hist_lows.sum()
        sum_xy_low_hist = (x_hist * hist_lows).sum()
        sum_yy_close_hist = (hist_closes ** 2).sum()

        # Get partial bar values for all 5min bars in this period (vectorized)
        partial_closes = partial_state.partial_close[bar_indices]
        partial_highs = partial_state.partial_high[bar_indices]
        partial_lows = partial_state.partial_low[bar_indices]
        current_prices = symbol_df['close'].iloc[bar_indices].values

        # Total number of bars including partial
        n_total = n_hist + 1
        x_partial = n_hist  # Position of partial bar

        # Update sums with partial bar (vectorized for all bars in period)
        sum_x = sum_x_hist + x_partial
        sum_xx = sum_xx_hist + x_partial ** 2
        sum_y_close = sum_y_close_hist + partial_closes
        sum_xy_close = sum_xy_close_hist + x_partial * partial_closes
        sum_y_high = sum_y_high_hist + partial_highs
        sum_xy_high = sum_xy_high_hist + x_partial * partial_highs
        sum_y_low = sum_y_low_hist + partial_lows
        sum_xy_low = sum_xy_low_hist + x_partial * partial_lows

        # Calculate regression coefficients (vectorized)
        denom = n_total * sum_xx - sum_x ** 2

        close_slope = (n_total * sum_xy_close - sum_x * sum_y_close) / denom
        high_slope = (n_total * sum_xy_high - sum_x * sum_y_high) / denom
        low_slope = (n_total * sum_xy_low - sum_x * sum_y_low) / denom

        close_intercept = (sum_y_close - close_slope * sum_x) / n_total
        high_intercept = (sum_y_high - high_slope * sum_x) / n_total
        low_intercept = (sum_y_low - low_slope * sum_x) / n_total

        # Calculate channel bounds at partial bar position
        x_partial_arr = np.full(len(bar_indices), x_partial)
        center_at_partial = close_slope * x_partial + close_intercept
        upper_at_partial = high_slope * x_partial + high_intercept
        lower_at_partial = low_slope * x_partial + low_intercept

        # Estimate residual std from historical bars
        hist_predicted = close_slope[:, None] * x_hist[None, :] + close_intercept[:, None]
        hist_residuals = hist_closes[None, :] - hist_predicted
        residual_std = np.std(hist_residuals, axis=1)

        # Adjust bounds with std dev
        upper_at_partial = np.maximum(upper_at_partial, center_at_partial + 2.0 * residual_std)
        lower_at_partial = np.minimum(lower_at_partial, center_at_partial - 2.0 * residual_std)

        # Calculate position
        channel_range = upper_at_partial - lower_at_partial
        position = np.where(
            channel_range > 1e-10,
            (current_prices - lower_at_partial) / channel_range,
            0.5
        )
        position = np.clip(position, 0, 1)

        # Calculate distances
        upper_dist = (upper_at_partial - current_prices) / np.maximum(current_prices, 1e-10) * 100
        lower_dist = (current_prices - lower_at_partial) / np.maximum(current_prices, 1e-10) * 100

        # Channel width
        channel_width_pct = channel_range / np.maximum(center_at_partial, 1e-10) * 100

        # R-squared (simplified - using historical only for speed)
        ss_res = np.sum(hist_residuals ** 2, axis=1)
        ss_tot = n_hist * np.var(hist_closes)
        close_r_squared = np.where(ss_tot > 1e-10, 1 - ss_res / ss_tot, 0)
        close_r_squared = np.clip(close_r_squared, 0, 1)

        # Slope percentages
        close_slope_pct = close_slope / np.maximum(current_prices, 1e-10) * 100

        # Stability
        stability = close_r_squared * 10

        # Is valid
        is_valid = np.where(close_r_squared > 0.5, 1.0, 0.0)

        # Store results
        output[f'{prefix}_close_slope'][bar_indices] = close_slope.astype(np.float32)
        output[f'{prefix}_close_slope_pct'][bar_indices] = close_slope_pct.astype(np.float32)
        output[f'{prefix}_close_r_squared'][bar_indices] = close_r_squared.astype(np.float32)
        output[f'{prefix}_high_slope'][bar_indices] = high_slope.astype(np.float32)
        output[f'{prefix}_low_slope'][bar_indices] = low_slope.astype(np.float32)
        output[f'{prefix}_position'][bar_indices] = position.astype(np.float32)
        output[f'{prefix}_upper_dist'][bar_indices] = upper_dist.astype(np.float32)
        output[f'{prefix}_lower_dist'][bar_indices] = lower_dist.astype(np.float32)
        output[f'{prefix}_channel_width_pct'][bar_indices] = channel_width_pct.astype(np.float32)
        output[f'{prefix}_stability'][bar_indices] = stability.astype(np.float32)
        output[f'{prefix}_is_valid'][bar_indices] = is_valid.astype(np.float32)
        output[f'{prefix}_insufficient_data'][bar_indices] = 0.0

    return pd.DataFrame(output, index=symbol_df.index)


def calculate_all_channel_features_vectorized(
    df_5min: pd.DataFrame,
    symbol: str,
    tf: str,
    tf_rule: str,
    windows: List[int] = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Calculate channel features for all window sizes.
    """
    if windows is None:
        windows = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 168, 200, 252, 336, 504]

    results = []
    for w in windows:
        result = calculate_channel_features_vectorized(
            df_5min, symbol, tf, tf_rule, w, show_progress=show_progress
        )
        results.append(result)

    return pd.concat(results, axis=1)


def test_vectorized():
    """Test vectorized channel calculation."""
    print("=" * 70)
    print("TESTING VECTORIZED CHANNEL CALCULATION")
    print("=" * 70)

    # Create sample data
    np.random.seed(42)
    n_bars = 50000  # 50K bars
    dates = pd.date_range('2015-01-02 09:30', periods=n_bars * 2, freq='5min')
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)][:n_bars]

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    df = pd.DataFrame({
        'tsla_open': prices,
        'tsla_high': prices + np.abs(np.random.randn(len(dates)) * 0.2),
        'tsla_low': prices - np.abs(np.random.randn(len(dates)) * 0.2),
        'tsla_close': prices + np.random.randn(len(dates)) * 0.1,
        'tsla_volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print(f"Test data: {len(df):,} 5min bars")

    # Test single window
    print("\n--- Single Window (w50) ---")
    t0 = time.time()
    result = calculate_channel_features_vectorized(
        df, 'tsla', 'daily', '1D', window=50, show_progress=True
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s for {len(df):,} bars")
    print(f"Per bar: {elapsed/len(df)*1000:.4f}ms")

    # Verify values
    print(f"\nSample values:")
    for i in [1000, 10000, 30000, len(df)-1]:
        if i < len(df):
            pos = result['tsla_channel_daily_w50_position'].iloc[i]
            r2 = result['tsla_channel_daily_w50_close_r_squared'].iloc[i]
            print(f"  Bar {i}: position={pos:.3f}, r²={r2:.3f}")

    # Test all windows
    print("\n--- All 21 Windows ---")
    t0 = time.time()
    result_all = calculate_all_channel_features_vectorized(
        df, 'tsla', 'daily', '1D', show_progress=True
    )
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s for {len(df):,} bars × 21 windows")
    print(f"Output shape: {result_all.shape}")

    # Estimate full dataset time
    full_estimate = elapsed * 418635 / len(df) * 22  # 11 TFs × 2 symbols
    print(f"\nEstimated for full dataset (all TFs, both symbols): {full_estimate/60:.0f} min")

    print("\n✅ Test completed!")
    return result


def _calculate_5min_channel_features_rolling(symbol_df: pd.DataFrame, symbol: str, window: int) -> pd.DataFrame:
    """
    Fast fully-vectorized channel calculation for 5min timeframe.

    For 5min TF, each bar IS the complete bar - no partial bar concept applies.
    Uses pandas rolling + numpy for O(n) vectorized computation.
    """
    n = len(symbol_df)
    prefix = f'{symbol}_channel_5min_w{window}'

    closes = symbol_df['close'].values.astype(np.float64)
    highs = symbol_df['high'].values.astype(np.float64)
    lows = symbol_df['low'].values.astype(np.float64)

    # Use pandas rolling for efficient computation
    close_series = pd.Series(closes, index=symbol_df.index)
    high_series = pd.Series(highs, index=symbol_df.index)
    low_series = pd.Series(lows, index=symbol_df.index)

    # Pre-compute x values for regression (constant for all windows)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    x_centered = x - x_mean

    # Rolling statistics using pandas (vectorized, fast)
    close_mean = close_series.rolling(window, min_periods=window).mean().values
    high_mean = high_series.rolling(window, min_periods=window).mean().values
    low_mean = low_series.rolling(window, min_periods=window).mean().values
    close_std = close_series.rolling(window, min_periods=window).std().values

    # For slope, we need rolling covariance with x
    # cov(x, y) = E[xy] - E[x]E[y] = sum((x-x_mean)(y-y_mean)) / n
    # slope = cov(x, y) / var(x)

    # Create weighted series for covariance calculation
    # We need sum of (x_i - x_mean) * (y_i - y_mean) for each window
    # This equals sum(x_centered * y) - x_mean * sum(y) + n * x_mean * y_mean
    # Simplified: sum(x_centered * y) since x is the same for all windows

    # Use stride tricks to create rolling windows efficiently
    from numpy.lib.stride_tricks import sliding_window_view

    if n >= window:
        # Create sliding windows of closes, highs, lows
        close_windows = sliding_window_view(closes, window)  # Shape: (n - window + 1, window)
        high_windows = sliding_window_view(highs, window)
        low_windows = sliding_window_view(lows, window)

        # Compute covariances with x (vectorized across all windows)
        # x_centered has shape (window,), windows have shape (n-window+1, window)
        # Result: shape (n-window+1,)
        close_cov = np.sum(x_centered * (close_windows - close_windows.mean(axis=1, keepdims=True)), axis=1)
        high_cov = np.sum(x_centered * (high_windows - high_windows.mean(axis=1, keepdims=True)), axis=1)
        low_cov = np.sum(x_centered * (low_windows - low_windows.mean(axis=1, keepdims=True)), axis=1)

        # Slopes
        close_slope = close_cov / x_var
        high_slope = high_cov / x_var
        low_slope = low_cov / x_var

        # Intercepts (at x = x_mean)
        close_intercept = close_windows.mean(axis=1) - close_slope * x_mean
        high_intercept = high_windows.mean(axis=1) - high_slope * x_mean
        low_intercept = low_windows.mean(axis=1) - low_slope * x_mean

        # Channel bounds at end of window (x = window - 1)
        x_now = window - 1
        center = close_slope * x_now + close_intercept
        upper_raw = high_slope * x_now + high_intercept
        lower_raw = low_slope * x_now + low_intercept

        # Residual std for bound adjustment
        y_pred = close_slope[:, None] * x[None, :] + close_intercept[:, None]
        residual_std = np.std(close_windows - y_pred, axis=1)

        # Adjust bounds
        upper = np.maximum(upper_raw, center + 2.0 * residual_std)
        lower = np.minimum(lower_raw, center - 2.0 * residual_std)

        # R-squared
        ss_res = np.sum((close_windows - y_pred) ** 2, axis=1)
        ss_tot = np.sum((close_windows - close_windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
        r_squared = np.where(ss_tot > 1e-10, 1 - ss_res / ss_tot, 0)
        r_squared = np.clip(r_squared, 0, 1)

        # Position (using the close at end of each window's range, which is closes[window-1:])
        # Window i contains closes[i:i+window], and current price is closes[i+window-1]
        # We have n - window + 1 windows, so current_prices should have that many elements
        current_prices = closes[window - 1:]  # Length: n - window + 1
        channel_range = upper - lower
        position = np.where(
            channel_range > 1e-10,
            (current_prices[:len(channel_range)] - lower) / channel_range,
            0.5
        )
        position = np.clip(position, 0, 1)

        # Distances
        current_prices_safe = np.maximum(current_prices[:len(channel_range)], 1e-10)
        upper_dist = (upper - current_prices[:len(channel_range)]) / current_prices_safe * 100
        lower_dist = (current_prices[:len(channel_range)] - lower) / current_prices_safe * 100
        channel_width = channel_range / np.maximum(center, 1e-10) * 100
        close_slope_pct = close_slope / current_prices_safe * 100

        # Initialize output arrays
        output = {
            f'{prefix}_close_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_close_slope_pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_close_r_squared': np.zeros(n, dtype=np.float32),
            f'{prefix}_high_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_low_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_position': np.full(n, 0.5, dtype=np.float32),
            f'{prefix}_upper_dist': np.zeros(n, dtype=np.float32),
            f'{prefix}_lower_dist': np.zeros(n, dtype=np.float32),
            f'{prefix}_channel_width_pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_stability': np.zeros(n, dtype=np.float32),
            f'{prefix}_is_valid': np.zeros(n, dtype=np.float32),
            f'{prefix}_insufficient_data': np.ones(n, dtype=np.float32),
        }

        # Fill in valid values (starting from index window-1)
        start_idx = window - 1
        end_idx = start_idx + len(close_slope)

        output[f'{prefix}_close_slope'][start_idx:end_idx] = close_slope.astype(np.float32)
        output[f'{prefix}_close_slope_pct'][start_idx:end_idx] = close_slope_pct.astype(np.float32)
        output[f'{prefix}_close_r_squared'][start_idx:end_idx] = r_squared.astype(np.float32)
        output[f'{prefix}_high_slope'][start_idx:end_idx] = high_slope.astype(np.float32)
        output[f'{prefix}_low_slope'][start_idx:end_idx] = low_slope.astype(np.float32)
        output[f'{prefix}_position'][start_idx:end_idx] = position.astype(np.float32)
        output[f'{prefix}_upper_dist'][start_idx:end_idx] = upper_dist.astype(np.float32)
        output[f'{prefix}_lower_dist'][start_idx:end_idx] = lower_dist.astype(np.float32)
        output[f'{prefix}_channel_width_pct'][start_idx:end_idx] = channel_width.astype(np.float32)
        output[f'{prefix}_stability'][start_idx:end_idx] = (r_squared * 10).astype(np.float32)
        output[f'{prefix}_is_valid'][start_idx:end_idx] = (r_squared > 0.5).astype(np.float32)
        output[f'{prefix}_insufficient_data'][start_idx:end_idx] = 0.0

    else:
        # Not enough data - return defaults
        output = {
            f'{prefix}_close_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_close_slope_pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_close_r_squared': np.zeros(n, dtype=np.float32),
            f'{prefix}_high_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_low_slope': np.zeros(n, dtype=np.float32),
            f'{prefix}_position': np.full(n, 0.5, dtype=np.float32),
            f'{prefix}_upper_dist': np.zeros(n, dtype=np.float32),
            f'{prefix}_lower_dist': np.zeros(n, dtype=np.float32),
            f'{prefix}_channel_width_pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_stability': np.zeros(n, dtype=np.float32),
            f'{prefix}_is_valid': np.zeros(n, dtype=np.float32),
            f'{prefix}_insufficient_data': np.ones(n, dtype=np.float32),
        }

    return pd.DataFrame(output, index=symbol_df.index)


if __name__ == '__main__':
    test_vectorized()
