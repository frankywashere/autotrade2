#!/usr/bin/env python3
"""
Detailed timing investigation: Break down feature extraction per position
to understand the actual time spent in each component.

This investigates the discrepancy between:
- Agent 4 report: 560ms per position
- Agent 9 report: 3 seconds per position (history scanning)
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
from features.full_features import (
    extract_full_features,
    extract_shared_features,
    extract_window_features,
    extract_all_window_features
)
from features.history import scan_channel_history, extract_history_features
from features.rsi import calculate_rsi_series

def timeit(func, *args, **kwargs):
    """Time a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def create_sample_data(n_bars=500):
    """Create sample TSLA, SPY, VIX data."""
    np.random.seed(42)

    # Create TSLA data
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')
    close = 250 + np.cumsum(np.random.randn(n_bars) * 0.5)

    tsla_df = pd.DataFrame({
        'open': close + np.random.randn(n_bars) * 0.2,
        'high': close + np.abs(np.random.randn(n_bars) * 0.5),
        'low': close - np.abs(np.random.randn(n_bars) * 0.5),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    }, index=dates)

    # Create SPY data
    spy_close = 450 + np.cumsum(np.random.randn(n_bars) * 0.3)
    spy_df = pd.DataFrame({
        'open': spy_close + np.random.randn(n_bars) * 0.1,
        'high': spy_close + np.abs(np.random.randn(n_bars) * 0.3),
        'low': spy_close - np.abs(np.random.randn(n_bars) * 0.3),
        'close': spy_close,
        'volume': np.random.randint(5000000, 20000000, n_bars)
    }, index=dates)

    # Create VIX data (daily)
    dates_daily = pd.date_range('2023-01-01', periods=n_bars//288 + 1, freq='D')
    vix_close = 20 + np.cumsum(np.random.randn(len(dates_daily)) * 0.5)
    vix_df = pd.DataFrame({
        'open': vix_close,
        'high': vix_close + 1,
        'low': vix_close - 1,
        'close': vix_close,
    }, index=dates_daily)

    return tsla_df, spy_df, vix_df

def breakdown_component_timing():
    """Break down timing for each component of feature extraction."""
    print("\n" + "="*80)
    print("DETAILED TIMING BREAKDOWN: Feature Extraction Per Position")
    print("="*80)

    # Create sample data
    tsla_df, spy_df, vix_df = create_sample_data(n_bars=500)

    print(f"\nData Shape:")
    print(f"  TSLA: {len(tsla_df)} bars")
    print(f"  SPY:  {len(spy_df)} bars")
    print(f"  VIX:  {len(vix_df)} bars")

    # Test 1: Shared features extraction (computed once)
    print("\n" + "-"*80)
    print("1. SHARED FEATURES EXTRACTION (computed once, reused for all 8 windows)")
    print("-"*80)

    shared, shared_time = timeit(
        extract_shared_features,
        tsla_df, spy_df, vix_df,
        lookforward_bars=200
    )
    print(f"Total shared features time: {shared_time*1000:.2f}ms")

    # Break down shared features
    print("\n  Subcomponents of shared features:")

    # Resampling
    resample_time = 0
    for tf in TIMEFRAMES:
        if tf != '5min':
            _, t = timeit(resample_ohlc, tsla_df, tf)
            resample_time += t
    print(f"    Resampling (11 TFs): {resample_time*1000:.2f}ms")

    # RSI series
    rsi_time = 0
    for tf in TIMEFRAMES:
        df_tf = tsla_df if tf == '5min' else resample_ohlc(tsla_df, tf)
        _, t = timeit(calculate_rsi_series, df_tf['close'].values, period=14)
        rsi_time += t
    print(f"    RSI series (11 TFs): {rsi_time*1000:.2f}ms")

    # Channel history scanning
    hist_time = 0
    for asset, df in [('TSLA', tsla_df), ('SPY', spy_df)]:
        _, t = timeit(scan_channel_history, df, window=20, max_channels=10, vix_df=vix_df)
        hist_time += t
        print(f"    History scanning ({asset}): {t*1000:.2f}ms")

    print(f"    Total history scanning: {hist_time*1000:.2f}ms")

    # Test 2: Window-dependent features extraction (per window)
    print("\n" + "-"*80)
    print("2. WINDOW-DEPENDENT FEATURES (computed per window)")
    print("-"*80)

    window_times = []
    for window in STANDARD_WINDOWS[:3]:  # Test 3 windows
        _, wt = timeit(
            extract_window_features,
            shared=shared,
            window=window,
            tsla_df=tsla_df,
            spy_df=spy_df,
            include_history=True,
            lookforward_bars=200
        )
        window_times.append(wt)
        print(f"  Window {window}: {wt*1000:.2f}ms")

    avg_window_time = np.mean(window_times)
    total_window_time = avg_window_time * 8
    print(f"  Average per window: {avg_window_time*1000:.2f}ms")
    print(f"  Total for 8 windows: {total_window_time*1000:.2f}ms")

    # Test 3: Complete pipeline with extract_all_window_features
    print("\n" + "-"*80)
    print("3. COMPLETE PIPELINE: extract_all_window_features()")
    print("-"*80)

    features_per_window, pipeline_time = timeit(
        extract_all_window_features,
        tsla_df, spy_df, vix_df,
        windows=STANDARD_WINDOWS,
        include_history=True,
        lookforward_bars=200
    )
    print(f"Total pipeline time (shared + 8 windows): {pipeline_time*1000:.2f}ms")

    # Test 4: Full feature extraction (old API, no window optimization)
    print("\n" + "-"*80)
    print("4. LEGACY API: extract_full_features() - single window")
    print("-"*80)

    full_features, full_time = timeit(
        extract_full_features,
        tsla_df, spy_df, vix_df,
        window=20,
        include_history=True,
        lookforward_bars=200
    )
    print(f"Single call with history: {full_time*1000:.2f}ms")

    # Test 5: Multiple calls (what would be needed without optimization)
    print("\n" + "-"*80)
    print("5. INEFFICIENT: Calling extract_full_features() 8 times")
    print("-"*80)

    start = time.perf_counter()
    for window in STANDARD_WINDOWS:
        _ = extract_full_features(
            tsla_df, spy_df, vix_df,
            window=window,
            include_history=True,
            lookforward_bars=200
        )
    inefficient_time = time.perf_counter() - start
    print(f"8 separate calls: {inefficient_time*1000:.2f}ms")
    print(f"Per call: {(inefficient_time/8)*1000:.2f}ms")

    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY & ANALYSIS")
    print("="*80)

    print(f"\nShared features (one-time): {shared_time*1000:.2f}ms")
    print(f"  - Resampling: {resample_time*1000:.2f}ms")
    print(f"  - RSI series: {rsi_time*1000:.2f}ms")
    print(f"  - History scanning: {hist_time*1000:.2f}ms")

    print(f"\nOptimized (shared + 8 windows):")
    print(f"  - Shared once: {shared_time*1000:.2f}ms")
    print(f"  - Window features (avg {avg_window_time*1000:.2f}ms × 8): {total_window_time*1000:.2f}ms")
    print(f"  - Total: {pipeline_time*1000:.2f}ms")

    print(f"\nUnoptimized (8 separate calls):")
    print(f"  - Total: {inefficient_time*1000:.2f}ms")
    print(f"  - Overhead from redundancy: {(inefficient_time - pipeline_time)*1000:.2f}ms")

    speedup = inefficient_time / pipeline_time
    print(f"  - Speedup factor: {speedup:.2f}x")

    # Analyze the discrepancy
    print("\n" + "="*80)
    print("DISCREPANCY ANALYSIS")
    print("="*80)

    print(f"\nAgent 4 reported: ~560ms per position")
    print(f"  Our measurement (optimized): {pipeline_time*1000:.2f}ms")
    print(f"  Difference: {abs(560 - pipeline_time*1000):.2f}ms")

    print(f"\nAgent 9 reported: ~3 seconds (from history scanning)")
    print(f"  Our history scanning (both assets): {hist_time*1000:.2f}ms")
    print(f"  Note: This is PART of shared features, not total per position")

    print(f"\nKey insight:")
    print(f"  - History scanning is {hist_time*1000:.1f}ms, not 3000ms")
    print(f"  - It's optional (include_history parameter)")
    print(f"  - It's computed ONCE with shared features, not per window")
    print(f"  - For 8 windows, it costs {hist_time*1000:.1f}ms total, not {hist_time*8*1000:.1f}ms")

    return {
        'shared_time': shared_time,
        'resampling_time': resample_time,
        'rsi_time': rsi_time,
        'history_time': hist_time,
        'avg_window_time': avg_window_time,
        'total_window_time': total_window_time,
        'pipeline_time': pipeline_time,
        'inefficient_time': inefficient_time,
        'speedup': speedup
    }

if __name__ == '__main__':
    results = breakdown_component_timing()

    print("\n" + "="*80)
    print("SPEEDUP CALCULATION: Fixing all three redundancies")
    print("="*80)

    # Calculate current time
    current_total = results['pipeline_time'] * 1000

    # Estimate savings
    print(f"\nCurrent time per position: {current_total:.2f}ms")
    print(f"\nRedundancies to fix:")
    print(f"1. Channel detection: Shared via detect_channels_multi_window")
    print(f"   - Current: {results['pipeline_time']*1000:.2f}ms (already shared)")
    print(f"   - Savings: ~0ms (already optimized)")

    print(f"\n2. RSI calculation: Computed once via calculate_rsi_series")
    print(f"   - Current: {results['rsi_time']*1000:.2f}ms per 11 TFs")
    print(f"   - Savings: ~0ms (already optimized)")

    print(f"\n3. History scanning: Computed once with shared features")
    print(f"   - Current: {results['history_time']*1000:.2f}ms total (not per window!)")
    print(f"   - If optional: Can save {results['history_time']*1000:.2f}ms")

    savings_ms = results['history_time'] * 1000
    final_time = current_total - savings_ms if results['history_time'] > 0 else current_total

    print(f"\nFinal time after fixing all redundancies:")
    print(f"  - With history: {current_total:.2f}ms")
    print(f"  - Without history (optional): {final_time:.2f}ms")
    print(f"  - Savings: {savings_ms:.2f}ms ({(savings_ms/current_total)*100:.1f}%)")
