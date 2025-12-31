"""
Partial Bar Computation for Rolling Channel Features (v5.4)

This module implements efficient partial bar calculation for real-time channel features.
At each 5min timestamp, we compute what the "partial TF bar" would look like,
enabling channels to include in-progress data.

Key concepts:
- Partial bar: The OHLCV of the current TF period from its start until now
- Example: At Monday 2pm, the "partial weekly bar" contains Mon 9:30am - Mon 2pm data
- This gives the model access to "how is this week evolving" rather than just "what happened last week"

Performance: O(n) vectorized operations per TF, ~few seconds total for all 11 TFs
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# TF period rules for pandas (using Period-compatible freq codes)
TIMEFRAME_PERIOD_RULES = {
    '5min': '5min',    # Each bar is its own period (no partial)
    '15min': '15min',
    '30min': '30min',
    '1h': 'h',
    '2h': '2h',
    '3h': '3h',
    '4h': '4h',
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'M',    # Period uses 'M', not 'ME'
    '3month': 'Q',     # Quarter for 3-month periods
}

# 5min bars per TF period (approximate, for trading hours)
BARS_PER_TF_PERIOD = {
    '5min': 1,
    '15min': 3,
    '30min': 6,
    '1h': 12,
    '2h': 24,
    '3h': 36,
    '4h': 48,
    'daily': 78,       # 6.5 hours * 12 bars/hour
    'weekly': 390,     # 5 days * 78
    'monthly': 1716,   # ~22 trading days * 78
    '3month': 5148,    # ~66 trading days * 78
}


@dataclass
class PartialBarState:
    """Precomputed partial bar state for efficient channel calculation."""
    # For each 5min bar, the partial TF bar's OHLCV
    partial_open: np.ndarray    # First bar's open in current period
    partial_high: np.ndarray    # Expanding max of highs
    partial_low: np.ndarray     # Expanding min of lows
    partial_close: np.ndarray   # Current bar's close
    partial_volume: np.ndarray  # Expanding sum of volumes

    # Period boundaries
    period_ids: np.ndarray      # Which TF period each 5min bar belongs to
    period_start_idx: np.ndarray  # Index of first bar in each period
    is_first_bar: np.ndarray    # True if this is the first bar of a new period


def compute_partial_bars(df: pd.DataFrame, tf: str) -> PartialBarState:
    """
    Compute partial bar state at each 5min timestamp for a given TF.

    For each 5min bar, we compute what the "in-progress" TF bar looks like,
    using expanding aggregations within each TF period.

    Args:
        df: 5min OHLCV DataFrame with columns: open, high, low, close, volume
            Index must be DatetimeIndex
        tf: Target timeframe ('15min', '1h', 'daily', 'weekly', etc.)

    Returns:
        PartialBarState with partial OHLCV at each 5min timestamp

    Example:
        For weekly TF at Monday 2pm:
        - partial_open = Monday 9:30am's open
        - partial_high = max(Monday 9:30am-2pm highs)
        - partial_low = min(Monday 9:30am-2pm lows)
        - partial_close = Monday 2pm's close
        - partial_volume = sum(Monday 9:30am-2pm volumes)
    """
    if tf == '5min':
        # No partial for 5min - each bar IS the complete bar
        return PartialBarState(
            partial_open=df['open'].values,
            partial_high=df['high'].values,
            partial_low=df['low'].values,
            partial_close=df['close'].values,
            partial_volume=df['volume'].values,
            period_ids=np.arange(len(df)),
            period_start_idx=np.arange(len(df)),
            is_first_bar=np.ones(len(df), dtype=bool)
        )

    # Get the period rule for this TF
    period_rule = TIMEFRAME_PERIOD_RULES.get(tf, tf)

    # Assign each bar to its TF period
    period = df.index.to_period(period_rule)
    period_ids = period.astype(np.int64)  # Convert to numeric for efficient grouping

    # Compute period boundaries FIRST (needed for vectorized aggregations)
    period_values = period.values
    is_first_bar = np.zeros(len(df), dtype=bool)
    is_first_bar[0] = True  # First bar is always start of period
    is_first_bar[1:] = period_values[1:] != period_values[:-1]

    # OPTIMIZED: Compute expanding aggregations using numpy (10-100x faster than groupby transform)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values

    n = len(df)
    partial_open = np.empty(n, dtype=np.float64)
    partial_high = np.empty(n, dtype=np.float64)
    partial_low = np.empty(n, dtype=np.float64)
    partial_volume = np.empty(n, dtype=np.float64)

    # Single pass through data - O(n) instead of O(n * n_groups)
    current_open = opens[0]
    current_high = highs[0]
    current_low = lows[0]
    current_volume = volumes[0]

    for i in range(n):
        if is_first_bar[i]:
            # Start of new period - reset accumulators
            current_open = opens[i]
            current_high = highs[i]
            current_low = lows[i]
            current_volume = volumes[i]
        else:
            # Continue period - update accumulators
            current_high = max(current_high, highs[i])
            current_low = min(current_low, lows[i])
            current_volume += volumes[i]

        partial_open[i] = current_open
        partial_high[i] = current_high
        partial_low[i] = current_low
        partial_volume[i] = current_volume

    # Close: current bar's close (always the most recent)
    partial_close = closes.copy()

    # period_start_idx: index of first bar in each period
    period_start_idx = np.zeros(n, dtype=np.int64)
    current_start = 0
    for i in range(n):
        if is_first_bar[i]:
            current_start = i
        period_start_idx[i] = current_start

    return PartialBarState(
        partial_open=partial_open.astype(np.float64),
        partial_high=partial_high.astype(np.float64),
        partial_low=partial_low.astype(np.float64),
        partial_close=partial_close.astype(np.float64),
        partial_volume=partial_volume.astype(np.float64),
        period_ids=period_ids,
        period_start_idx=period_start_idx,
        is_first_bar=is_first_bar
    )


def compute_all_partial_bars(df: pd.DataFrame, timeframes: list = None) -> Dict[str, PartialBarState]:
    """
    Compute partial bar states for all timeframes.

    Args:
        df: 5min OHLCV DataFrame
        timeframes: List of TFs to compute (defaults to all)

    Returns:
        Dict mapping tf -> PartialBarState
    """
    if timeframes is None:
        timeframes = list(TIMEFRAME_PERIOD_RULES.keys())

    return {tf: compute_partial_bars(df, tf) for tf in timeframes}


@dataclass
class RegressionSums:
    """Precomputed sums for incremental linear regression."""
    n: int                    # Number of complete bars
    sum_x: float              # Σx
    sum_xx: float             # Σx²
    sum_y_close: float        # Σy (close)
    sum_xy_close: float       # Σxy (close)
    sum_y_high: float         # Σy (high)
    sum_xy_high: float        # Σxy (high)
    sum_y_low: float          # Σy (low)
    sum_xy_low: float         # Σxy (low)
    sum_yy_close: float       # Σy² (for r²)


def precompute_regression_sums(closes: np.ndarray, highs: np.ndarray,
                                lows: np.ndarray, window: int) -> RegressionSums:
    """
    Precompute regression sums for a window of complete bars.

    These sums can be incrementally updated when adding a partial bar.
    """
    n = len(closes)
    if n < window:
        return None

    # Use last `window` bars
    closes = closes[-window:]
    highs = highs[-window:]
    lows = lows[-window:]

    x = np.arange(window)

    return RegressionSums(
        n=window,
        sum_x=x.sum(),
        sum_xx=(x ** 2).sum(),
        sum_y_close=closes.sum(),
        sum_xy_close=(x * closes).sum(),
        sum_y_high=highs.sum(),
        sum_xy_high=(x * highs).sum(),
        sum_y_low=lows.sum(),
        sum_xy_low=(x * lows).sum(),
        sum_yy_close=(closes ** 2).sum()
    )


def compute_regression_with_partial(sums: RegressionSums,
                                     partial_close: float,
                                     partial_high: float,
                                     partial_low: float) -> Dict[str, float]:
    """
    Compute linear regression including a partial bar at the end.

    Uses precomputed sums for efficiency - O(1) per partial bar.

    Args:
        sums: Precomputed RegressionSums from complete bars
        partial_close/high/low: Current partial bar's values

    Returns:
        Dict with slope, intercept, r_squared for close/high/low
    """
    if sums is None:
        return None

    # Add partial bar at position n (after the window)
    n_total = sums.n + 1
    x_partial = sums.n  # Position of partial bar

    # Update sums with partial bar
    sum_x = sums.sum_x + x_partial
    sum_xx = sums.sum_xx + x_partial ** 2

    # Close regression
    sum_y_close = sums.sum_y_close + partial_close
    sum_xy_close = sums.sum_xy_close + x_partial * partial_close
    sum_yy_close = sums.sum_yy_close + partial_close ** 2

    # Calculate regression for close
    denom = n_total * sum_xx - sum_x ** 2
    if abs(denom) < 1e-10:
        return None

    close_slope = (n_total * sum_xy_close - sum_x * sum_y_close) / denom
    close_intercept = (sum_y_close - close_slope * sum_x) / n_total

    # R² for close
    y_mean = sum_y_close / n_total
    ss_tot = sum_yy_close - n_total * y_mean ** 2
    ss_res = (sum_yy_close - 2 * close_slope * sum_xy_close
              - 2 * close_intercept * sum_y_close
              + close_slope ** 2 * sum_xx
              + 2 * close_slope * close_intercept * sum_x
              + n_total * close_intercept ** 2)
    close_r_squared = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0
    close_r_squared = max(0, min(1, close_r_squared))  # Clamp to [0, 1]

    # High regression
    sum_y_high = sums.sum_y_high + partial_high
    sum_xy_high = sums.sum_xy_high + x_partial * partial_high
    high_slope = (n_total * sum_xy_high - sum_x * sum_y_high) / denom
    high_intercept = (sum_y_high - high_slope * sum_x) / n_total

    # Low regression
    sum_y_low = sums.sum_y_low + partial_low
    sum_xy_low = sums.sum_xy_low + x_partial * partial_low
    low_slope = (n_total * sum_xy_low - sum_x * sum_y_low) / denom
    low_intercept = (sum_y_low - low_slope * sum_x) / n_total

    return {
        'close_slope': close_slope,
        'close_intercept': close_intercept,
        'close_r_squared': close_r_squared,
        'high_slope': high_slope,
        'high_intercept': high_intercept,
        'low_slope': low_slope,
        'low_intercept': low_intercept,
        'n_bars': n_total
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_partial_bars():
    """Test partial bar computation with sample data."""
    import time

    print("=" * 70)
    print("TESTING PARTIAL BAR COMPUTATION")
    print("=" * 70)

    # Create sample 5min data (2 weeks)
    np.random.seed(42)
    n_bars = 390 * 2  # 2 weeks
    dates = pd.date_range('2024-01-08 09:30', periods=n_bars, freq='5min')
    # Filter to market hours
    dates = dates[(dates.hour >= 9) & (dates.hour < 16)][:n_bars]

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(len(dates)) * 0.2),
        'low': prices - np.abs(np.random.randn(len(dates)) * 0.2),
        'close': prices + np.random.randn(len(dates)) * 0.1,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print(f"\nSample data: {len(df)} 5min bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Test weekly partial bars
    print("\n--- Weekly Partial Bars ---")
    t0 = time.time()
    weekly_state = compute_partial_bars(df, 'weekly')
    print(f"Computation time: {(time.time() - t0) * 1000:.1f}ms")

    # Verify: at end of first week, partial should equal complete weekly bar
    first_week_end = df.index.to_period('W')[0]
    first_week_mask = df.index.to_period('W') == first_week_end
    first_week_idx = np.where(first_week_mask)[0][-1]  # Last bar of first week

    week_data = df[first_week_mask]
    complete_weekly = {
        'open': week_data['open'].iloc[0],
        'high': week_data['high'].max(),
        'low': week_data['low'].min(),
        'close': week_data['close'].iloc[-1]
    }

    print(f"\nFirst week ends at index {first_week_idx}")
    print(f"Complete weekly bar: O={complete_weekly['open']:.2f}, "
          f"H={complete_weekly['high']:.2f}, "
          f"L={complete_weekly['low']:.2f}, "
          f"C={complete_weekly['close']:.2f}")
    print(f"Last partial:        O={weekly_state.partial_open[first_week_idx]:.2f}, "
          f"H={weekly_state.partial_high[first_week_idx]:.2f}, "
          f"L={weekly_state.partial_low[first_week_idx]:.2f}, "
          f"C={weekly_state.partial_close[first_week_idx]:.2f}")

    # Check match
    assert np.isclose(weekly_state.partial_open[first_week_idx], complete_weekly['open']), "Open mismatch!"
    assert np.isclose(weekly_state.partial_high[first_week_idx], complete_weekly['high']), "High mismatch!"
    assert np.isclose(weekly_state.partial_low[first_week_idx], complete_weekly['low']), "Low mismatch!"
    assert np.isclose(weekly_state.partial_close[first_week_idx], complete_weekly['close']), "Close mismatch!"
    print("✅ Partial bars match complete bars at period end!")

    # Test evolution within first week (first 10 bars)
    print("\n--- Partial Bar Evolution (first 10 bars of week 1) ---")
    for i in range(min(10, len(df))):
        print(f"Bar {i}: partial_high={weekly_state.partial_high[i]:.2f}, "
              f"partial_low={weekly_state.partial_low[i]:.2f}, "
              f"is_first={weekly_state.is_first_bar[i]}")

    # Performance test with larger dataset
    print("\n--- Performance Test (418K bars) ---")
    n_large = 418_635
    large_df = pd.DataFrame({
        'open': np.random.randn(n_large) + 100,
        'high': np.random.randn(n_large) + 101,
        'low': np.random.randn(n_large) + 99,
        'close': np.random.randn(n_large) + 100,
        'volume': np.random.randint(1000, 10000, n_large)
    }, index=pd.date_range('2015-01-02', periods=n_large, freq='5min'))

    for tf in ['daily', 'weekly', 'monthly', '3month']:
        t0 = time.time()
        state = compute_partial_bars(large_df, tf)
        elapsed = (time.time() - t0) * 1000
        print(f"  {tf:8s}: {elapsed:.1f}ms")

    print("\n✅ All tests passed!")
    return True


if __name__ == '__main__':
    test_partial_bars()
