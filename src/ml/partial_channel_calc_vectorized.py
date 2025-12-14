"""
Vectorized Channel Calculation with Partial Bars (v5.6)

This module calculates channel features at 5min resolution including partial TF bars.
Uses vectorized numpy operations for efficiency - ~100x faster than loop-based approach.

Key optimization: Process all 5min bars within a TF period in parallel

v5.5: Added missing channel features for parity with old LinearRegressionChannel:
  - high_slope_pct, low_slope_pct, high_r_squared, low_r_squared, r_squared_avg
  - slope_convergence, is_bull, is_bear, is_sideways, quality_score, duration
  - ping_pongs (4 thresholds), complete_cycles (4 thresholds)

v5.6: Removed projection features (projected_high/low/center)
  - Projections are now calculated at INFERENCE time using learned duration predictions
  - Model predicts how long channel will continue, then projects geometrically
  - See projection_calculator.py for the new approach
  - Feature count: 31 per window (was 34)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

# Numba JIT compilation for performance-critical loops
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .partial_bars import compute_partial_bars, PartialBarState, TIMEFRAME_PERIOD_RULES


# Numba JIT-compiled ping-pong detection functions
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, fastmath=True)
    def _detect_ping_pongs_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                threshold: float = 0.02) -> int:
        """
        JIT-compiled ping-pong detection - counts alternating touches of upper/lower bounds.
        """
        bounces = 0
        last_touch = 0  # 0=none, 1=upper, 2=lower

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances as percentage
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
            else:
                upper_dist = 1.0
            if abs(lower_val) > 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
            else:
                lower_dist = 1.0

            # Check if price touches upper line
            if upper_dist <= threshold:
                if last_touch == 2:  # Was at lower
                    bounces += 1
                last_touch = 1

            # Check if price touches lower line
            elif lower_dist <= threshold:
                if last_touch == 1:  # Was at upper
                    bounces += 1
                last_touch = 2

        return bounces

    @numba.jit(nopython=True, fastmath=True)
    def _detect_complete_cycles_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                     threshold: float = 0.02) -> int:
        """
        JIT-compiled complete cycles detection - counts full round-trips.
        Lower → Upper → Lower = 1 cycle
        Upper → Lower → Upper = 1 cycle
        """
        touches = np.empty(len(prices), dtype=np.int8)  # 0=none, 1=upper, 2=lower
        touch_count = 0
        last_touch = 0

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            # Calculate distances with zero protection
            if upper_val > 0:
                upper_dist = abs(price - upper_val) / upper_val
            else:
                upper_dist = 1.0
            if lower_val != 0:
                lower_dist = abs(price - lower_val) / abs(lower_val)
            else:
                lower_dist = 1.0

            # Record touches (only when transitioning)
            if upper_dist <= threshold and last_touch != 1:
                touches[touch_count] = 1  # upper
                touch_count += 1
                last_touch = 1
            elif lower_dist <= threshold and last_touch != 2:
                touches[touch_count] = 2  # lower
                touch_count += 1
                last_touch = 2

        # Count complete cycles
        complete_cycles = 0
        i = 0
        while i < touch_count - 2:
            # Lower → Upper → Lower (2 → 1 → 2)
            if touches[i] == 2 and touches[i+1] == 1 and touches[i+2] == 2:
                complete_cycles += 1
                i += 2
            # Upper → Lower → Upper (1 → 2 → 1)
            elif touches[i] == 1 and touches[i+1] == 2 and touches[i+2] == 1:
                complete_cycles += 1
                i += 2
            else:
                i += 1

        return complete_cycles
else:
    # Fallback Python implementations (slower but functional)
    def _detect_ping_pongs_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                threshold: float = 0.02) -> int:
        """Python fallback for ping-pong detection."""
        bounces = 0
        last_touch = 0

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            upper_dist = abs(price - upper_val) / max(upper_val, 1e-10)
            lower_dist = abs(price - lower_val) / max(abs(lower_val), 1e-10)

            if upper_dist <= threshold:
                if last_touch == 2:
                    bounces += 1
                last_touch = 1
            elif lower_dist <= threshold:
                if last_touch == 1:
                    bounces += 1
                last_touch = 2

        return bounces

    def _detect_complete_cycles_jit(prices: np.ndarray, upper: np.ndarray, lower: np.ndarray,
                                     threshold: float = 0.02) -> int:
        """Python fallback for cycle detection."""
        touches = []
        last_touch = 0

        for i in range(len(prices)):
            price = prices[i]
            upper_val = upper[i]
            lower_val = lower[i]

            upper_dist = abs(price - upper_val) / max(upper_val, 1e-10)
            lower_dist = abs(price - lower_val) / max(abs(lower_val), 1e-10)

            if upper_dist <= threshold and last_touch != 1:
                touches.append(1)
                last_touch = 1
            elif lower_dist <= threshold and last_touch != 2:
                touches.append(2)
                last_touch = 2

        complete_cycles = 0
        i = 0
        while i < len(touches) - 2:
            if touches[i] == 2 and touches[i+1] == 1 and touches[i+2] == 2:
                complete_cycles += 1
                i += 2
            elif touches[i] == 1 and touches[i+1] == 2 and touches[i+2] == 1:
                complete_cycles += 1
                i += 2
            else:
                i += 1

        return complete_cycles


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

    v5.6: Removed fixed projection features (projected_high/low/center).
    Projections are now calculated at inference time using learned duration predictions.
    This reduces feature count from 34 to 31 per window.

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

    # Initialize output arrays (31 features per window - v5.6: removed projections)
    prefix = f'{symbol}_channel_{tf}_w{window}'
    output = {
        # Original 12 features
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

        # NEW: Slope percentage variants (2 features)
        f'{prefix}_high_slope_pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_low_slope_pct': np.zeros(n_5min, dtype=np.float32),

        # NEW: R-squared variants (3 features)
        f'{prefix}_high_r_squared': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_low_r_squared': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_r_squared_avg': np.zeros(n_5min, dtype=np.float32),

        # NEW: Derived metrics (3 features)
        f'{prefix}_slope_convergence': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_quality_score': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_duration': np.zeros(n_5min, dtype=np.float32),

        # NEW: Direction flags (3 features)
        f'{prefix}_is_bull': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_is_bear': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_is_sideways': np.zeros(n_5min, dtype=np.float32),

        # v5.6: Removed projected_high/low/center - now calculated at inference from learned duration

        # NEW: Ping-pongs at 4 thresholds (4 features)
        f'{prefix}_ping_pongs': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_ping_pongs_0_5pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_ping_pongs_1_0pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_ping_pongs_3_0pct': np.zeros(n_5min, dtype=np.float32),

        # NEW: Complete cycles at 4 thresholds (4 features)
        f'{prefix}_complete_cycles': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_complete_cycles_0_5pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_complete_cycles_1_0pct': np.zeros(n_5min, dtype=np.float32),
        f'{prefix}_complete_cycles_3_0pct': np.zeros(n_5min, dtype=np.float32),
    }

    # Process by TF period (vectorized within each period)
    # Find unique periods and their boundaries
    period = symbol_df.index.to_period(TIMEFRAME_PERIOD_RULES.get(tf, tf))

    # OPTIMIZATION: Use pd.factorize to get sequential codes (0, 1, 2, ...) that map to unique periods
    # This is much faster than period.codes which returns ordinal numbers
    period_codes, unique_periods = pd.factorize(period)

    # OPTIMIZATION: Precompute period-to-indices mapping ONCE (O(n) instead of O(n²))
    # period_codes[i] gives the index into unique_periods for bar i
    period_to_indices = {}
    for i, code in enumerate(period_codes):
        if code not in period_to_indices:
            period_to_indices[code] = []
        period_to_indices[code].append(i)
    # Convert lists to numpy arrays for efficiency
    period_to_indices = {k: np.array(v) for k, v in period_to_indices.items()}

    iterator = range(len(unique_periods))
    if show_progress:
        iterator = tqdm(iterator, desc=f"   {symbol}_{tf}_w{window}", leave=False, ncols=100)

    # OPTIMIZATION: Convert tf_timestamps to numpy for binary search
    tf_timestamps_ns = tf_timestamps.view(np.int64) if hasattr(tf_timestamps, 'view') else tf_timestamps.astype(np.int64)

    for period_idx in iterator:
        # Use precomputed mapping - period_idx is guaranteed to be in the dict
        bar_indices = period_to_indices[period_idx]
        if len(bar_indices) == 0:
            continue

        # Find complete TF bars before this period using binary search (O(log n) instead of O(n))
        period_start_ts = symbol_df.index[bar_indices[0]]
        period_start_ns = period_start_ts.value  # nanoseconds
        n_complete_before = np.searchsorted(tf_timestamps_ns, period_start_ns, side='left')

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

        # R-squared for close (using historical only for speed)
        ss_res_close = np.sum(hist_residuals ** 2, axis=1)
        ss_tot_close = n_hist * np.var(hist_closes)
        close_r_squared = np.where(ss_tot_close > 1e-10, 1 - ss_res_close / ss_tot_close, 0)
        close_r_squared = np.clip(close_r_squared, 0, 1)

        # R-squared for high
        high_predicted = high_slope[:, None] * x_hist[None, :] + high_intercept[:, None]
        high_residuals = hist_highs[None, :] - high_predicted
        ss_res_high = np.sum(high_residuals ** 2, axis=1)
        ss_tot_high = n_hist * np.var(hist_highs)
        high_r_squared = np.where(ss_tot_high > 1e-10, 1 - ss_res_high / ss_tot_high, 0)
        high_r_squared = np.clip(high_r_squared, 0, 1)

        # R-squared for low
        low_predicted = low_slope[:, None] * x_hist[None, :] + low_intercept[:, None]
        low_residuals = hist_lows[None, :] - low_predicted
        ss_res_low = np.sum(low_residuals ** 2, axis=1)
        ss_tot_low = n_hist * np.var(hist_lows)
        low_r_squared = np.where(ss_tot_low > 1e-10, 1 - ss_res_low / ss_tot_low, 0)
        low_r_squared = np.clip(low_r_squared, 0, 1)

        # Average R-squared
        r_squared_avg = (close_r_squared + high_r_squared + low_r_squared) / 3

        # Slope percentages
        current_prices_safe = np.maximum(current_prices, 1e-10)
        close_slope_pct = close_slope / current_prices_safe * 100
        high_slope_pct = high_slope / current_prices_safe * 100
        low_slope_pct = low_slope / current_prices_safe * 100

        # Slope convergence: how parallel are the channel lines (1 = parallel, 0 = diverging)
        slope_range = np.abs(high_slope - low_slope)
        slope_convergence = 1 - slope_range / (np.abs(close_slope) + 1e-10)
        slope_convergence = np.clip(slope_convergence, 0, 1)

        # Direction flags (based on close_slope_pct)
        is_bull = (close_slope_pct > 0.1).astype(np.float32)
        is_bear = (close_slope_pct < -0.1).astype(np.float32)
        is_sideways = (np.abs(close_slope_pct) <= 0.1).astype(np.float32)

        # v5.6: Removed fixed projection calculation
        # Projections are now calculated at inference time using learned duration predictions
        # The model predicts how long the channel will continue, then we project by that duration

        # Duration (TF bars since start of window - constant for this period)
        duration = np.full(len(bar_indices), float(n_hist + 1), dtype=np.float32)

        # Compute channel lines over historical window for ping-pong detection
        # Lines at each historical position
        x_full = np.arange(n_hist + 1)  # Include partial bar position
        center_line = close_slope[:, None] * x_full[None, :] + close_intercept[:, None]
        upper_line_raw = high_slope[:, None] * x_full[None, :] + high_intercept[:, None]
        lower_line_raw = low_slope[:, None] * x_full[None, :] + low_intercept[:, None]

        # Adjust lines with std dev
        upper_line = np.maximum(upper_line_raw, center_line + 2.0 * residual_std[:, None])
        lower_line = np.minimum(lower_line_raw, center_line - 2.0 * residual_std[:, None])

        # Ping-pong and cycle detection - need to iterate over each bar in period
        # For efficiency, we compute for the first bar only (same channel for all bars in period)
        # since the historical bars don't change within a TF period
        n_bars_in_period = len(bar_indices)

        # Build price array: historical closes + partial close for each bar
        # For ping-pong detection, use historical closes only (partial doesn't count as complete)
        pp_2pct = np.zeros(n_bars_in_period, dtype=np.float32)
        pp_0_5pct = np.zeros(n_bars_in_period, dtype=np.float32)
        pp_1_0pct = np.zeros(n_bars_in_period, dtype=np.float32)
        pp_3_0pct = np.zeros(n_bars_in_period, dtype=np.float32)
        cc_2pct = np.zeros(n_bars_in_period, dtype=np.float32)
        cc_0_5pct = np.zeros(n_bars_in_period, dtype=np.float32)
        cc_1_0pct = np.zeros(n_bars_in_period, dtype=np.float32)
        cc_3_0pct = np.zeros(n_bars_in_period, dtype=np.float32)

        # Use the first bar's channel for ping-pong detection (all bars in period share same historical data)
        if n_bars_in_period > 0:
            upper_for_pp = upper_line[0, :n_hist]  # Historical portion only
            lower_for_pp = lower_line[0, :n_hist]

            # Detect ping-pongs at each threshold
            pp_2pct[:] = _detect_ping_pongs_jit(hist_closes, upper_for_pp, lower_for_pp, 0.02)
            pp_0_5pct[:] = _detect_ping_pongs_jit(hist_closes, upper_for_pp, lower_for_pp, 0.005)
            pp_1_0pct[:] = _detect_ping_pongs_jit(hist_closes, upper_for_pp, lower_for_pp, 0.01)
            pp_3_0pct[:] = _detect_ping_pongs_jit(hist_closes, upper_for_pp, lower_for_pp, 0.03)

            # Detect complete cycles at each threshold
            cc_2pct[:] = _detect_complete_cycles_jit(hist_closes, upper_for_pp, lower_for_pp, 0.02)
            cc_0_5pct[:] = _detect_complete_cycles_jit(hist_closes, upper_for_pp, lower_for_pp, 0.005)
            cc_1_0pct[:] = _detect_complete_cycles_jit(hist_closes, upper_for_pp, lower_for_pp, 0.01)
            cc_3_0pct[:] = _detect_complete_cycles_jit(hist_closes, upper_for_pp, lower_for_pp, 0.03)

        # Quality score: cycles × (0.5 + 0.5 × r²)
        quality_score = cc_2pct * (0.5 + 0.5 * r_squared_avg)

        # Stability
        stability = close_r_squared * 10

        # Is valid (based on cycles >= 2, like old system)
        is_valid = np.where(cc_2pct >= 2, 1.0, 0.0)

        # Store results - Original 12 features
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

        # Store NEW features - Slope percentages
        output[f'{prefix}_high_slope_pct'][bar_indices] = high_slope_pct.astype(np.float32)
        output[f'{prefix}_low_slope_pct'][bar_indices] = low_slope_pct.astype(np.float32)

        # Store NEW features - R-squared variants
        output[f'{prefix}_high_r_squared'][bar_indices] = high_r_squared.astype(np.float32)
        output[f'{prefix}_low_r_squared'][bar_indices] = low_r_squared.astype(np.float32)
        output[f'{prefix}_r_squared_avg'][bar_indices] = r_squared_avg.astype(np.float32)

        # Store NEW features - Derived metrics
        output[f'{prefix}_slope_convergence'][bar_indices] = slope_convergence.astype(np.float32)
        output[f'{prefix}_quality_score'][bar_indices] = quality_score.astype(np.float32)
        output[f'{prefix}_duration'][bar_indices] = duration

        # Store NEW features - Direction flags
        output[f'{prefix}_is_bull'][bar_indices] = is_bull
        output[f'{prefix}_is_bear'][bar_indices] = is_bear
        output[f'{prefix}_is_sideways'][bar_indices] = is_sideways

        # v5.6: Removed projected_high/low/center storage - calculated at inference

        # Store NEW features - Ping-pongs
        output[f'{prefix}_ping_pongs'][bar_indices] = pp_2pct
        output[f'{prefix}_ping_pongs_0_5pct'][bar_indices] = pp_0_5pct
        output[f'{prefix}_ping_pongs_1_0pct'][bar_indices] = pp_1_0pct
        output[f'{prefix}_ping_pongs_3_0pct'][bar_indices] = pp_3_0pct

        # Store NEW features - Complete cycles
        output[f'{prefix}_complete_cycles'][bar_indices] = cc_2pct
        output[f'{prefix}_complete_cycles_0_5pct'][bar_indices] = cc_0_5pct
        output[f'{prefix}_complete_cycles_1_0pct'][bar_indices] = cc_1_0pct
        output[f'{prefix}_complete_cycles_3_0pct'][bar_indices] = cc_3_0pct

    return pd.DataFrame(output, index=symbol_df.index)


def calculate_all_channel_features_vectorized(
    df_5min: pd.DataFrame,
    symbol: str,
    tf: str,
    tf_rule: str,
    windows: List[int] = None,
    show_progress: bool = True,
    debug: bool = False
) -> pd.DataFrame:
    """
    Calculate channel features for all window sizes.
    """
    import sys
    if windows is None:
        windows = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 168, 200, 252, 336, 504]

    results = []
    for i, w in enumerate(windows):
        if debug:
            t0 = time.time()
        result = calculate_channel_features_vectorized(
            df_5min, symbol, tf, tf_rule, w, show_progress=show_progress
        )
        if debug:
            print(f"         [{symbol}_{tf}] Window {w}: {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
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

    v5.6: Returns 31 features per window (removed projected_high/low/center).
    Projections are now calculated at inference time using learned duration predictions.
    """
    n = len(symbol_df)
    prefix = f'{symbol}_channel_5min_w{window}'

    closes = symbol_df['close'].values.astype(np.float64)
    highs = symbol_df['high'].values.astype(np.float64)
    lows = symbol_df['low'].values.astype(np.float64)

    # Pre-compute x values for regression (constant for all windows)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    x_centered = x - x_mean

    # Use stride tricks to create rolling windows efficiently
    from numpy.lib.stride_tricks import sliding_window_view

    # Helper to create default output dict with zeros
    def _create_default_output():
        return {
            # Original 12 features
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

            # NEW: Slope percentage variants (2 features)
            f'{prefix}_high_slope_pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_low_slope_pct': np.zeros(n, dtype=np.float32),

            # NEW: R-squared variants (3 features)
            f'{prefix}_high_r_squared': np.zeros(n, dtype=np.float32),
            f'{prefix}_low_r_squared': np.zeros(n, dtype=np.float32),
            f'{prefix}_r_squared_avg': np.zeros(n, dtype=np.float32),

            # NEW: Derived metrics (3 features)
            f'{prefix}_slope_convergence': np.zeros(n, dtype=np.float32),
            f'{prefix}_quality_score': np.zeros(n, dtype=np.float32),
            f'{prefix}_duration': np.zeros(n, dtype=np.float32),

            # NEW: Direction flags (3 features)
            f'{prefix}_is_bull': np.zeros(n, dtype=np.float32),
            f'{prefix}_is_bear': np.zeros(n, dtype=np.float32),
            f'{prefix}_is_sideways': np.zeros(n, dtype=np.float32),

            # v5.6: Removed projected_high/low/center - now calculated at inference from learned duration

            # NEW: Ping-pongs at 4 thresholds (4 features)
            f'{prefix}_ping_pongs': np.zeros(n, dtype=np.float32),
            f'{prefix}_ping_pongs_0_5pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_ping_pongs_1_0pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_ping_pongs_3_0pct': np.zeros(n, dtype=np.float32),

            # NEW: Complete cycles at 4 thresholds (4 features)
            f'{prefix}_complete_cycles': np.zeros(n, dtype=np.float32),
            f'{prefix}_complete_cycles_0_5pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_complete_cycles_1_0pct': np.zeros(n, dtype=np.float32),
            f'{prefix}_complete_cycles_3_0pct': np.zeros(n, dtype=np.float32),
        }

    if n < window:
        # Not enough data - return defaults
        return pd.DataFrame(_create_default_output(), index=symbol_df.index)

    # Create sliding windows of closes, highs, lows
    close_windows = sliding_window_view(closes, window)  # Shape: (n - window + 1, window)
    high_windows = sliding_window_view(highs, window)
    low_windows = sliding_window_view(lows, window)
    n_windows = len(close_windows)

    # Compute covariances with x (vectorized across all windows)
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

    # Predicted values for each window
    close_pred = close_slope[:, None] * x[None, :] + close_intercept[:, None]
    high_pred = high_slope[:, None] * x[None, :] + high_intercept[:, None]
    low_pred = low_slope[:, None] * x[None, :] + low_intercept[:, None]

    # Residual std for bound adjustment
    residual_std = np.std(close_windows - close_pred, axis=1)

    # Adjust bounds
    upper = np.maximum(upper_raw, center + 2.0 * residual_std)
    lower = np.minimum(lower_raw, center - 2.0 * residual_std)

    # R-squared for close
    ss_res_close = np.sum((close_windows - close_pred) ** 2, axis=1)
    ss_tot_close = np.sum((close_windows - close_windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ss_tot_close_safe = np.maximum(ss_tot_close, 1e-10)
    close_r_squared = np.where(ss_tot_close > 1e-10, 1 - ss_res_close / ss_tot_close_safe, 0)
    close_r_squared = np.clip(close_r_squared, 0, 1)

    # R-squared for high
    ss_res_high = np.sum((high_windows - high_pred) ** 2, axis=1)
    ss_tot_high = np.sum((high_windows - high_windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ss_tot_high_safe = np.maximum(ss_tot_high, 1e-10)
    high_r_squared = np.where(ss_tot_high > 1e-10, 1 - ss_res_high / ss_tot_high_safe, 0)
    high_r_squared = np.clip(high_r_squared, 0, 1)

    # R-squared for low
    ss_res_low = np.sum((low_windows - low_pred) ** 2, axis=1)
    ss_tot_low = np.sum((low_windows - low_windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    ss_tot_low_safe = np.maximum(ss_tot_low, 1e-10)
    low_r_squared = np.where(ss_tot_low > 1e-10, 1 - ss_res_low / ss_tot_low_safe, 0)
    low_r_squared = np.clip(low_r_squared, 0, 1)

    # Average R-squared
    r_squared_avg = (close_r_squared + high_r_squared + low_r_squared) / 3

    # Position
    current_prices = closes[window - 1:]  # Length: n - window + 1
    channel_range = upper - lower
    channel_range_safe = np.maximum(channel_range, 1e-10)
    position = np.where(
        channel_range > 1e-10,
        (current_prices[:n_windows] - lower) / channel_range_safe,
        0.5
    )
    position = np.clip(position, 0, 1)

    # Distances and width
    current_prices_safe = np.maximum(current_prices[:n_windows], 1e-10)
    upper_dist = (upper - current_prices[:n_windows]) / current_prices_safe * 100
    lower_dist = (current_prices[:n_windows] - lower) / current_prices_safe * 100
    channel_width = channel_range / np.maximum(center, 1e-10) * 100

    # Slope percentages
    close_slope_pct = close_slope / current_prices_safe * 100
    high_slope_pct = high_slope / current_prices_safe * 100
    low_slope_pct = low_slope / current_prices_safe * 100

    # Slope convergence
    slope_range = np.abs(high_slope - low_slope)
    slope_convergence = 1 - slope_range / (np.abs(close_slope) + 1e-10)
    slope_convergence = np.clip(slope_convergence, 0, 1)

    # Direction flags
    is_bull = (close_slope_pct > 0.1).astype(np.float32)
    is_bear = (close_slope_pct < -0.1).astype(np.float32)
    is_sideways = (np.abs(close_slope_pct) <= 0.1).astype(np.float32)

    # v5.6: Removed fixed projection calculation
    # Projections are now calculated at inference time using learned duration predictions
    # The model predicts how long the channel will continue, then we project by that duration

    # Duration (window size - constant for 5min)
    duration = np.full(n_windows, float(window), dtype=np.float32)

    # Ping-pongs and cycles - compute for each window
    # For 5min path, we need to iterate since each window has different bounds
    pp_2pct = np.zeros(n_windows, dtype=np.float32)
    pp_0_5pct = np.zeros(n_windows, dtype=np.float32)
    pp_1_0pct = np.zeros(n_windows, dtype=np.float32)
    pp_3_0pct = np.zeros(n_windows, dtype=np.float32)
    cc_2pct = np.zeros(n_windows, dtype=np.float32)
    cc_0_5pct = np.zeros(n_windows, dtype=np.float32)
    cc_1_0pct = np.zeros(n_windows, dtype=np.float32)
    cc_3_0pct = np.zeros(n_windows, dtype=np.float32)

    # Compute channel lines for each window (upper/lower at each position)
    # upper_lines[i, j] = upper bound for window i at position j
    upper_lines = high_slope[:, None] * x[None, :] + high_intercept[:, None]
    lower_lines = low_slope[:, None] * x[None, :] + low_intercept[:, None]
    center_lines = close_slope[:, None] * x[None, :] + close_intercept[:, None]

    # Adjust with residual std
    upper_lines = np.maximum(upper_lines, center_lines + 2.0 * residual_std[:, None])
    lower_lines = np.minimum(lower_lines, center_lines - 2.0 * residual_std[:, None])

    # Iterate over windows to compute ping-pongs and cycles
    for i in range(n_windows):
        prices_in_window = close_windows[i]
        upper_in_window = upper_lines[i]
        lower_in_window = lower_lines[i]

        # Detect at each threshold
        pp_2pct[i] = _detect_ping_pongs_jit(prices_in_window, upper_in_window, lower_in_window, 0.02)
        pp_0_5pct[i] = _detect_ping_pongs_jit(prices_in_window, upper_in_window, lower_in_window, 0.005)
        pp_1_0pct[i] = _detect_ping_pongs_jit(prices_in_window, upper_in_window, lower_in_window, 0.01)
        pp_3_0pct[i] = _detect_ping_pongs_jit(prices_in_window, upper_in_window, lower_in_window, 0.03)

        cc_2pct[i] = _detect_complete_cycles_jit(prices_in_window, upper_in_window, lower_in_window, 0.02)
        cc_0_5pct[i] = _detect_complete_cycles_jit(prices_in_window, upper_in_window, lower_in_window, 0.005)
        cc_1_0pct[i] = _detect_complete_cycles_jit(prices_in_window, upper_in_window, lower_in_window, 0.01)
        cc_3_0pct[i] = _detect_complete_cycles_jit(prices_in_window, upper_in_window, lower_in_window, 0.03)

    # Quality score: cycles × (0.5 + 0.5 × r²)
    quality_score = cc_2pct * (0.5 + 0.5 * r_squared_avg)

    # Stability
    stability = close_r_squared * 10

    # Is valid (based on cycles >= 2)
    is_valid = np.where(cc_2pct >= 2, 1.0, 0.0)

    # Initialize output arrays
    output = _create_default_output()

    # Fill in valid values (starting from index window-1)
    start_idx = window - 1
    end_idx = start_idx + n_windows

    # Original 12 features
    output[f'{prefix}_close_slope'][start_idx:end_idx] = close_slope.astype(np.float32)
    output[f'{prefix}_close_slope_pct'][start_idx:end_idx] = close_slope_pct.astype(np.float32)
    output[f'{prefix}_close_r_squared'][start_idx:end_idx] = close_r_squared.astype(np.float32)
    output[f'{prefix}_high_slope'][start_idx:end_idx] = high_slope.astype(np.float32)
    output[f'{prefix}_low_slope'][start_idx:end_idx] = low_slope.astype(np.float32)
    output[f'{prefix}_position'][start_idx:end_idx] = position.astype(np.float32)
    output[f'{prefix}_upper_dist'][start_idx:end_idx] = upper_dist.astype(np.float32)
    output[f'{prefix}_lower_dist'][start_idx:end_idx] = lower_dist.astype(np.float32)
    output[f'{prefix}_channel_width_pct'][start_idx:end_idx] = channel_width.astype(np.float32)
    output[f'{prefix}_stability'][start_idx:end_idx] = stability.astype(np.float32)
    output[f'{prefix}_is_valid'][start_idx:end_idx] = is_valid.astype(np.float32)
    output[f'{prefix}_insufficient_data'][start_idx:end_idx] = 0.0

    # NEW: Slope percentages
    output[f'{prefix}_high_slope_pct'][start_idx:end_idx] = high_slope_pct.astype(np.float32)
    output[f'{prefix}_low_slope_pct'][start_idx:end_idx] = low_slope_pct.astype(np.float32)

    # NEW: R-squared variants
    output[f'{prefix}_high_r_squared'][start_idx:end_idx] = high_r_squared.astype(np.float32)
    output[f'{prefix}_low_r_squared'][start_idx:end_idx] = low_r_squared.astype(np.float32)
    output[f'{prefix}_r_squared_avg'][start_idx:end_idx] = r_squared_avg.astype(np.float32)

    # NEW: Derived metrics
    output[f'{prefix}_slope_convergence'][start_idx:end_idx] = slope_convergence.astype(np.float32)
    output[f'{prefix}_quality_score'][start_idx:end_idx] = quality_score.astype(np.float32)
    output[f'{prefix}_duration'][start_idx:end_idx] = duration

    # NEW: Direction flags
    output[f'{prefix}_is_bull'][start_idx:end_idx] = is_bull
    output[f'{prefix}_is_bear'][start_idx:end_idx] = is_bear
    output[f'{prefix}_is_sideways'][start_idx:end_idx] = is_sideways

    # v5.6: Removed projected_high/low/center storage - calculated at inference

    # NEW: Ping-pongs
    output[f'{prefix}_ping_pongs'][start_idx:end_idx] = pp_2pct
    output[f'{prefix}_ping_pongs_0_5pct'][start_idx:end_idx] = pp_0_5pct
    output[f'{prefix}_ping_pongs_1_0pct'][start_idx:end_idx] = pp_1_0pct
    output[f'{prefix}_ping_pongs_3_0pct'][start_idx:end_idx] = pp_3_0pct

    # NEW: Complete cycles
    output[f'{prefix}_complete_cycles'][start_idx:end_idx] = cc_2pct
    output[f'{prefix}_complete_cycles_0_5pct'][start_idx:end_idx] = cc_0_5pct
    output[f'{prefix}_complete_cycles_1_0pct'][start_idx:end_idx] = cc_1_0pct
    output[f'{prefix}_complete_cycles_3_0pct'][start_idx:end_idx] = cc_3_0pct

    return pd.DataFrame(output, index=symbol_df.index)


if __name__ == '__main__':
    test_vectorized()
