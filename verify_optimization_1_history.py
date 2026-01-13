"""
Verify Optimization 1: Fixed-size windows vs growing windows in history.py

Tests whether scan_channel_history produces identical channel detection results
when using fixed-size windows (optimized) vs growing windows (original).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from v7.features.history import scan_channel_history
from v7.core.channel import detect_channel


def scan_channel_history_original(
    df: pd.DataFrame,
    window: int = 20,
    max_channels: int = 10,
    scan_bars: int = 1500,
    vix_df: pd.DataFrame = None
):
    """
    Original implementation using GROWING window slices.
    This is the baseline to verify against.
    """
    from v7.features.rsi import calculate_rsi_series
    from v7.features.history import detect_bounces_with_rsi, ChannelRecord

    channels = []
    rsi_series = calculate_rsi_series(df['close'].values, period=14)

    end_idx = len(df)
    start_scan = max(window + 100, len(df) - scan_bars)

    current_idx = end_idx
    step_size = 30

    while current_idx > start_scan and len(channels) < max_channels:
        # ORIGINAL: Use growing slice from start to current_idx
        df_slice = df.iloc[:current_idx]

        if len(df_slice) < window:
            break

        channel = detect_channel(df_slice, window=window)

        if channel.valid:
            # Scan forward to find where this channel broke
            break_idx = current_idx
            break_direction = 1

            for future_idx in range(current_idx, min(current_idx + 200, len(df))):
                future_price = df['close'].iloc[future_idx]
                x_future = window + (future_idx - current_idx)
                upper_proj = channel.slope * x_future + channel.intercept + 2 * channel.std_dev
                lower_proj = channel.slope * x_future + channel.intercept - 2 * channel.std_dev

                if future_price > upper_proj:
                    break_idx = future_idx
                    break_direction = 1
                    break
                elif future_price < lower_proj:
                    break_idx = future_idx
                    break_direction = 0
                    break

            # Record this channel
            rsi_start_idx = current_idx - window
            rsi_end_idx = min(break_idx, len(rsi_series) - 1)

            rsi_period = 14
            if rsi_period <= rsi_start_idx < len(rsi_series):
                rsi_at_start = rsi_series[rsi_start_idx]
            else:
                rsi_at_start = 50.0

            if rsi_period <= rsi_end_idx < len(rsi_series):
                rsi_at_break = rsi_series[rsi_end_idx]
            else:
                rsi_at_break = 50.0

            safe_start = max(rsi_period, rsi_start_idx)
            safe_end = min(rsi_end_idx, len(rsi_series))
            if safe_end > safe_start:
                avg_rsi = np.mean(rsi_series[safe_start:safe_end])
            else:
                avg_rsi = 50.0

            # For simplicity, skip bounce detection in this test
            bounces = []

            channels.append(ChannelRecord(
                start_idx=current_idx - window,
                end_idx=break_idx,
                duration_bars=break_idx - (current_idx - window),
                direction=int(channel.direction),
                break_direction=break_direction,
                bounce_count=channel.bounce_count,
                complete_cycles=channel.complete_cycles,
                avg_rsi=float(avg_rsi),
                rsi_at_start=float(rsi_at_start),
                rsi_at_break=float(rsi_at_break),
                bounces=bounces,
            ))

            current_idx = current_idx - window - step_size
        else:
            current_idx -= step_size

    return channels


def test_fixed_vs_growing_windows():
    """
    Test if fixed-size windows produce identical results to growing windows.
    """
    print("=" * 80)
    print("OPTIMIZATION 1: Fixed-size windows vs Growing windows (history.py)")
    print("=" * 80)
    print()

    # Generate synthetic test data
    np.random.seed(42)
    n_bars = 2000
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='5min')

    # Create trending data with noise
    trend = np.linspace(100, 150, n_bars)
    noise = np.random.randn(n_bars) * 2
    close = trend + noise
    high = close + np.abs(np.random.randn(n_bars) * 0.5)
    low = close - np.abs(np.random.randn(n_bars) * 0.5)
    open_price = close + np.random.randn(n_bars) * 0.3
    volume = np.random.randint(1000, 10000, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # Create dummy VIX data
    vix_dates = pd.date_range('2023-01-01', periods=n_bars // 78 + 1, freq='D')
    vix_df = pd.DataFrame({
        'close': np.random.uniform(15, 25, len(vix_dates))
    }, index=vix_dates)

    print("Test data: {} bars".format(len(df)))
    print()

    # Test with optimized implementation (current)
    print("Running OPTIMIZED implementation (fixed-size windows)...")
    try:
        channels_optimized = scan_channel_history(
            df,
            window=20,
            max_channels=5,
            scan_bars=500,
            vix_df=vix_df
        )
        print("  Found {} channels".format(len(channels_optimized)))
    except Exception as e:
        print("  ERROR: {}".format(e))
        channels_optimized = []
    print()

    # Test with original implementation (growing windows)
    print("Running ORIGINAL implementation (growing windows)...")
    try:
        channels_original = scan_channel_history_original(
            df,
            window=20,
            max_channels=5,
            scan_bars=500,
            vix_df=vix_df
        )
        print("  Found {} channels".format(len(channels_original)))
    except Exception as e:
        print("  ERROR: {}".format(e))
        channels_original = []
    print()

    # Compare results
    print("-" * 80)
    print("COMPARISON:")
    print("-" * 80)

    if len(channels_optimized) != len(channels_original):
        print("DIFFERENCE DETECTED:")
        print("  Optimized found {} channels".format(len(channels_optimized)))
        print("  Original found {} channels".format(len(channels_original)))
        print()
        print("RESULT: Optimizations may change channel detection count")
        return False

    print("Number of channels: {} (both implementations)".format(len(channels_optimized)))

    # Compare each channel's properties
    all_match = True
    for i, (opt, orig) in enumerate(zip(channels_optimized, channels_original)):
        print("\nChannel {}:".format(i + 1))

        # Compare key properties
        properties = [
            ('start_idx', opt.start_idx, orig.start_idx),
            ('end_idx', opt.end_idx, orig.end_idx),
            ('duration_bars', opt.duration_bars, orig.duration_bars),
            ('direction', opt.direction, orig.direction),
            ('break_direction', opt.break_direction, orig.break_direction),
            ('bounce_count', opt.bounce_count, orig.bounce_count),
        ]

        for prop_name, opt_val, orig_val in properties:
            match = opt_val == orig_val
            status = "MATCH" if match else "DIFFER"
            print("  {}: {} vs {} [{}]".format(prop_name, opt_val, orig_val, status))
            if not match:
                all_match = False

    print()
    print("-" * 80)
    if all_match and len(channels_optimized) == len(channels_original):
        print("RESULT: Optimizations produce IDENTICAL results")
        return True
    else:
        print("RESULT: Optimizations produce DIFFERENT results")
        return False


if __name__ == '__main__':
    try:
        result = test_fixed_vs_growing_windows()
        sys.exit(0 if result else 1)
    except Exception as e:
        print("\nFATAL ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(2)
