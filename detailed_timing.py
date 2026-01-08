#!/usr/bin/env python3
"""
Even more detailed timing: Break down window-dependent operations.
Focus on what's being repeated across the 8 windows.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from core.channel import detect_channel, detect_channels_multi_window, STANDARD_WINDOWS
from core.timeframe import resample_ohlc, TIMEFRAMES
from features.full_features import extract_tsla_channel_features
from features.rsi import calculate_rsi_series

def timeit(func, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def create_sample_data(n_bars=500):
    """Create sample data."""
    np.random.seed(42)

    dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')
    close = 250 + np.cumsum(np.random.randn(n_bars) * 0.5)

    tsla_df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 0.2,
        'high': close + np.abs(np.random.randn(n_bars) * 0.5),
        'low': close - np.abs(np.random.randn(n_bars) * 0.5),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    return tsla_df

def investigate_window_operations():
    """Break down what happens in window-dependent operations."""
    print("\n" + "="*80)
    print("WINDOW-DEPENDENT OPERATIONS: What's being repeated?")
    print("="*80)

    tsla_df = create_sample_data(n_bars=500)

    # First: Resampling is SHARED (computed once)
    print("\n1. RESAMPLING (Shared, computed once)")
    print("-"*80)

    resampled = {}
    total_resample_time = 0
    for tf in TIMEFRAMES:
        if tf == '5min':
            resampled[tf] = tsla_df
        else:
            _, t = timeit(resample_ohlc, tsla_df, tf)
            resampled[tf] = resample_ohlc(tsla_df, tf)
            total_resample_time += t
            print(f"  {tf}: {t*1000:.2f}ms")

    print(f"  Total resampling: {total_resample_time*1000:.2f}ms (one-time)")

    # Second: RSI series is SHARED (computed once per TF)
    print("\n2. RSI CALCULATION (Shared, computed once per TF)")
    print("-"*80)

    rsi_series = {}
    total_rsi_time = 0
    for tf in TIMEFRAMES:
        df_tf = resampled[tf]
        _, t = timeit(calculate_rsi_series, df_tf['close'].values, period=14)
        rsi_series[tf] = calculate_rsi_series(df_tf['close'].values, period=14)
        total_rsi_time += t

    print(f"  Total RSI series (11 TFs): {total_rsi_time*1000:.2f}ms (one-time)")

    # Third: Channel detection at different windows
    print("\n3. CHANNEL DETECTION (Per window)")
    print("-"*80)

    channel_times = []
    for window in STANDARD_WINDOWS:
        _, t = timeit(detect_channel, tsla_df, window=window)
        channel_times.append(t)
        print(f"  Window {window}: {t*1000:.2f}ms")

    print(f"  Total for 8 windows: {sum(channel_times)*1000:.2f}ms")
    print(f"  Average per window: {np.mean(channel_times)*1000:.2f}ms")

    # Fourth: What about detect_channels_multi_window?
    print("\n4. MULTI-WINDOW DETECTION (All windows at once)")
    print("-"*80)

    _, multiwindow_time = timeit(detect_channels_multi_window, tsla_df, windows=STANDARD_WINDOWS)
    print(f"  detect_channels_multi_window: {multiwindow_time*1000:.2f}ms")
    print(f"  vs 8 separate calls: {sum(channel_times)*1000:.2f}ms")
    print(f"  Savings: {(sum(channel_times) - multiwindow_time)*1000:.2f}ms ({((sum(channel_times) - multiwindow_time)/sum(channel_times))*100:.1f}%)")

    # Fifth: Per-TF features extraction
    print("\n5. PER-TF FEATURE EXTRACTION (Done for each window)")
    print("-"*80)
    print("\n  This is where the real time goes! Each window calls extract_tsla_channel_features")
    print("  for all 11 timeframes...")

    # Simulate what happens for one window
    tf_times = []
    window = 20
    channel_20 = detect_channel(tsla_df, window=window)

    print(f"\n  For window={window}, per-TF extraction times:")
    for i, tf in enumerate(TIMEFRAMES[:3]):  # Just show first 3
        df_tf = resampled[tf]
        _, t = timeit(
            extract_tsla_channel_features,
            df_tf,
            tf,
            window=window,
            longer_tf_channels=None,
            lookforward_bars=200
        )
        tf_times.append(t)
        print(f"    {tf}: {t*1000:.2f}ms")

    print(f"    ...")
    avg_tf_time = np.mean(tf_times)
    total_per_window_tf_time = avg_tf_time * 11
    print(f"  Average per TF: {avg_tf_time*1000:.2f}ms")
    print(f"  Total for 11 TFs (one window): {total_per_window_tf_time*1000:.2f}ms")
    print(f"  Total for 11 TFs × 8 windows: {total_per_window_tf_time * 8*1000:.2f}ms")

    # What is extract_tsla_channel_features doing each time?
    print("\n  Inside extract_tsla_channel_features:")
    print("    - Detect channel (DIFFERENT window each time)")
    print("    - Calculate RSI series (SAME data, already computed but recomputed!)")
    print("    - Detect RSI divergence (recomputed)")
    print("    - Detect bounces with RSI (recomputed)")
    print("    - Check containment")
    print("    - Track exits")

    print("\n6. ACTUAL REDUNDANCY FOUND!")
    print("-"*80)
    print("\n  Problem: RSI series and bounce detection are recomputed for EACH WINDOW")
    print("  even though they depend only on the price data, not the window size!")

    # Verify this
    print("\n  Testing: Extract features for same TF, different windows")
    print(f"    - Data: TSLA 5min")

    for window in [10, 20, 30]:
        _, t = timeit(
            extract_tsla_channel_features,
            tsla_df,
            '5min',
            window=window,
            longer_tf_channels=None,
            lookforward_bars=200
        )
        print(f"    - Window {window}: {t*1000:.2f}ms")

    print("\n  Expected savings from fixing:")
    print("    - Each window: Compute RSI series once instead of 11 times")
    print("    - Each window: Compute divergence once instead of 11 times")
    print("    - Each window: Compute bounces once instead of 11 times")

    return {
        'resample_time': total_resample_time,
        'rsi_time': total_rsi_time,
        'channel_times': channel_times,
        'multiwindow_time': multiwindow_time,
        'avg_tf_time': avg_tf_time,
        'total_per_window_tf_time': total_per_window_tf_time
    }

if __name__ == '__main__':
    results = investigate_window_operations()

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. SHARED OPERATIONS (Already Optimized)")
    print("   - Resampling: Done once")
    print("   - RSI series: Done once per timeframe")
    print("   - History scanning: Done once")

    print("\n2. WINDOW-DEPENDENT OPERATIONS")
    print("   - Channel detection: Efficiently done with detect_channels_multi_window")
    print("   - Per-TF features: Extracted for each window")

    print("\n3. HIDDEN REDUNDANCY IN FEATURE EXTRACTION")
    print("   - RSI series is calculated AGAIN inside extract_tsla_channel_features")
    print("   - For each window, for each of 11 TFs")
    print("   - Total redundancy: 8 windows × 11 TFs = 88 redundant RSI calculations")

    print("\n4. ACTUAL SPEEDUP OPPORTUNITY")
    print("   - Share RSI series across windows and TFs")
    print("   - Share bounce detection results")
    print("   - Share divergence detection")
    print("   - Estimated savings: 10-15% of per-window time")
