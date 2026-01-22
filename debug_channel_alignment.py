#!/usr/bin/env python3
"""
Debug script to investigate channel alignment issues across timeframes.

For sample index 1, checks each TF (5min, 1h, daily, weekly) to understand
why daily aligns correctly but others don't.

Prints:
- source_channel_start_ts and source_channel_end_ts
- Window size
- Whether end_ts exists in the resampled data's index
- The x_offset calculation result
- Compare stored slope/intercept to freshly computed values
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v15.dtypes import ChannelSample, ChannelLabels, TIMEFRAMES, STANDARD_WINDOWS, BARS_PER_TF
from v15.data import load_market_data, resample_ohlc
from v7.core.channel import detect_channel


def main():
    # Load samples
    samples_path = '/tmp/test_timestamps.pkl'
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load market data
    print("Loading market data...")
    tsla_df, _, _ = load_market_data('data')
    print(f"Loaded {len(tsla_df)} 5min bars")

    # Sample index to analyze
    sample_idx = 1
    sample = samples[sample_idx]

    print("\n" + "=" * 80)
    print(f"ANALYZING SAMPLE {sample_idx}")
    print("=" * 80)
    print(f"Sample timestamp: {sample.timestamp}")
    print(f"Channel end idx (5min): {sample.channel_end_idx}")
    print(f"Best window: {sample.best_window}")

    # Timeframes to check
    tfs_to_check = ['5min', '1h', 'daily', 'weekly']

    # TF mapping for resampling
    TF_MAP = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W', 'monthly': '1MS'
    }

    for tf in tfs_to_check:
        print("\n" + "-" * 80)
        print(f"TIMEFRAME: {tf}")
        print("-" * 80)

        # Get best window for this TF
        best_window = sample.best_window

        # Get labels for this TF at best window
        labels = None
        for window in STANDARD_WINDOWS:
            if window in sample.labels_per_window:
                if 'tsla' in sample.labels_per_window[window]:
                    tf_labels = sample.labels_per_window[window]['tsla'].get(tf)
                    if tf_labels is not None:
                        if tf_labels.source_channel_direction >= 0:
                            labels = tf_labels
                            best_window = window
                            break

        if labels is None:
            print(f"  No valid labels found for {tf}")
            continue

        print(f"  Using window: {best_window}")
        print(f"\n  STORED CHANNEL PARAMETERS:")
        print(f"    source_channel_start_ts: {labels.source_channel_start_ts}")
        print(f"    source_channel_end_ts: {labels.source_channel_end_ts}")
        print(f"    source_channel_slope: {labels.source_channel_slope}")
        print(f"    source_channel_intercept: {labels.source_channel_intercept}")
        print(f"    source_channel_std_dev: {labels.source_channel_std_dev}")
        print(f"    source_channel_r_squared: {labels.source_channel_r_squared}")
        print(f"    source_channel_direction: {labels.source_channel_direction}")

        # Resample TSLA data to this timeframe
        channel_end_idx = sample.channel_end_idx
        bars_per_tf = BARS_PER_TF.get(tf, 1)
        lookback_5min = (best_window + 50) * bars_per_tf
        forward_5min = 100 * bars_per_tf

        start_idx = max(0, channel_end_idx - lookback_5min)
        end_idx = min(len(tsla_df), channel_end_idx + forward_5min)

        df_slice = tsla_df.iloc[start_idx:end_idx].copy()

        # Resample
        resample_tf = TF_MAP.get(tf, tf)
        if tf != '5min':
            result = resample_ohlc(df_slice, resample_tf)
            df_tf = result[0] if isinstance(result, tuple) else result
        else:
            df_tf = df_slice

        print(f"\n  RESAMPLED DATA:")
        print(f"    df_tf length: {len(df_tf)}")
        print(f"    df_tf index range: {df_tf.index[0]} to {df_tf.index[-1]}")

        # Check if end_ts exists in resampled data
        end_ts = labels.source_channel_end_ts
        start_ts = labels.source_channel_start_ts

        print(f"\n  TIMESTAMP CHECKS:")

        # Check exact match
        end_ts_in_index = end_ts in df_tf.index if end_ts is not None else False
        start_ts_in_index = start_ts in df_tf.index if start_ts is not None else False
        print(f"    end_ts exact match in df_tf.index: {end_ts_in_index}")
        print(f"    start_ts exact match in df_tf.index: {start_ts_in_index}")

        # If not exact match, find closest
        if end_ts is not None and not end_ts_in_index:
            searchsorted_idx = df_tf.index.searchsorted(end_ts)
            print(f"    end_ts searchsorted result: {searchsorted_idx}")
            if 0 < searchsorted_idx < len(df_tf):
                print(f"      nearest before: {df_tf.index[searchsorted_idx - 1]}")
                print(f"      nearest after (or at): {df_tf.index[searchsorted_idx]}")
            elif searchsorted_idx == 0:
                print(f"      end_ts is BEFORE all data")
            else:
                print(f"      end_ts is AFTER all data")

        if start_ts is not None and not start_ts_in_index:
            searchsorted_idx = df_tf.index.searchsorted(start_ts)
            print(f"    start_ts searchsorted result: {searchsorted_idx}")

        # Calculate x_offset as the inspector does
        print(f"\n  X_OFFSET CALCULATION (as done by inspector):")

        # Simulate what the inspector does in _draw_panel
        channel_end_ts_5min = tsla_df.index[channel_end_idx]
        print(f"    channel_end_ts (from 5min idx): {channel_end_ts_5min}")

        # Find closest index in resampled data
        tf_end_idx = df_tf.index.searchsorted(channel_end_ts_5min)
        tf_end_idx = min(tf_end_idx, len(df_tf) - 1)
        print(f"    tf_end_idx (searchsorted of 5min timestamp): {tf_end_idx}")
        print(f"    df_tf timestamp at tf_end_idx: {df_tf.index[tf_end_idx]}")

        # Calculate plot range (simulated)
        plot_start = max(0, tf_end_idx - best_window + 1)
        forward_bars = min(50, len(df_tf) - tf_end_idx - 1)
        plot_end = tf_end_idx + forward_bars + 1

        df_plot = df_tf.iloc[plot_start:plot_end]
        print(f"\n    df_plot range: {plot_start} to {plot_end} ({len(df_plot)} bars)")
        print(f"    df_plot index range: {df_plot.index[0]} to {df_plot.index[-1]}")

        # Now simulate _draw_channel x_offset calculation
        end_ts_labels = labels.source_channel_end_ts
        print(f"\n    end_ts from labels: {end_ts_labels}")

        if end_ts_labels is not None and end_ts_labels in df_plot.index:
            channel_end_plot_idx = df_plot.index.get_loc(end_ts_labels)
            print(f"    end_ts FOUND in df_plot.index at position: {channel_end_plot_idx}")
        elif end_ts_labels is not None:
            channel_end_plot_idx = df_plot.index.searchsorted(end_ts_labels)
            if channel_end_plot_idx >= len(df_plot):
                channel_end_plot_idx = len(df_plot) - 1
            print(f"    end_ts NOT FOUND - searchsorted gives: {channel_end_plot_idx}")
            print(f"    (clamped to {channel_end_plot_idx})")
        else:
            channel_end_plot_idx = best_window - 1
            print(f"    end_ts is None - fallback to window-1: {channel_end_plot_idx}")

        x_offset = channel_end_plot_idx - (best_window - 1)
        print(f"\n    x_offset = channel_end_plot_idx - (window - 1)")
        print(f"    x_offset = {channel_end_plot_idx} - ({best_window} - 1) = {x_offset}")

        # Now let's do fresh channel detection to compare
        print(f"\n  FRESH CHANNEL DETECTION:")

        # For fresh detection, we need to get the data slice ending at end_ts
        if end_ts_labels is not None:
            # Find the index in df_tf closest to end_ts
            if end_ts_labels in df_tf.index:
                fresh_end_idx = df_tf.index.get_loc(end_ts_labels)
            else:
                fresh_end_idx = df_tf.index.searchsorted(end_ts_labels)
                if fresh_end_idx >= len(df_tf):
                    fresh_end_idx = len(df_tf) - 1

            fresh_start_idx = max(0, fresh_end_idx - best_window + 1)
            df_for_detection = df_tf.iloc[fresh_start_idx:fresh_end_idx + 1]

            print(f"    Detection slice: {fresh_start_idx} to {fresh_end_idx + 1}")
            print(f"    Detection slice length: {len(df_for_detection)} (window={best_window})")

            if len(df_for_detection) >= best_window:
                try:
                    fresh_channel = detect_channel(df_for_detection, window=best_window)
                    print(f"\n    FRESH CHANNEL PARAMS:")
                    print(f"      slope: {fresh_channel.slope}")
                    print(f"      intercept: {fresh_channel.intercept}")
                    print(f"      std_dev: {fresh_channel.std_dev}")
                    print(f"      r_squared: {fresh_channel.r_squared}")
                    print(f"      direction: {fresh_channel.direction}")
                    print(f"      valid: {fresh_channel.valid}")

                    # Compare stored vs fresh
                    print(f"\n    COMPARISON (stored vs fresh):")
                    slope_diff = labels.source_channel_slope - fresh_channel.slope
                    intercept_diff = labels.source_channel_intercept - fresh_channel.intercept
                    print(f"      slope diff: {slope_diff:.6f}")
                    print(f"      intercept diff: {intercept_diff:.6f}")

                    if abs(slope_diff) > 0.001 or abs(intercept_diff) > 1.0:
                        print(f"      *** SIGNIFICANT DIFFERENCE DETECTED ***")
                    else:
                        print(f"      Parameters match closely")

                except Exception as e:
                    print(f"    Fresh detection failed: {e}")
            else:
                print(f"    Not enough data for fresh detection")
        else:
            print(f"    Cannot do fresh detection - end_ts is None")

        # Key insight: check if timestamps align with TF bar boundaries
        print(f"\n  TIMESTAMP ALIGNMENT ANALYSIS:")
        if end_ts_labels is not None:
            # For daily, timestamps should be at 09:30:00 (market open) or similar
            # For 1h, timestamps should be at :00:00 or :30:00
            # For 5min, timestamps are at :00, :05, :10, etc.

            hour = end_ts_labels.hour
            minute = end_ts_labels.minute
            second = end_ts_labels.second
            print(f"    end_ts time component: {hour:02d}:{minute:02d}:{second:02d}")

            if tf == 'daily':
                # Daily bars typically start at 09:30
                print(f"    Daily: expecting ~09:30 (market open)")
                if hour == 9 and minute == 30:
                    print(f"      ALIGNED with daily bar boundary")
                else:
                    print(f"      NOT ALIGNED with daily bar boundary")
            elif tf == '1h':
                # Hourly bars at :00 or :30
                if minute % 60 == 0 or minute % 60 == 30:
                    print(f"      ALIGNED with hourly bar boundary")
                else:
                    print(f"      NOT ALIGNED with hourly bar boundary")
            elif tf == '5min':
                # 5min bars at :00, :05, :10, etc.
                if minute % 5 == 0:
                    print(f"      ALIGNED with 5min bar boundary")
                else:
                    print(f"      NOT ALIGNED with 5min bar boundary")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The key insight is how the inspector handles timestamp alignment:

1. The labels store source_channel_end_ts which is the timestamp in the
   RESAMPLED timeframe data (e.g., daily bars have specific timestamps).

2. When drawing, the inspector tries to find end_ts in df_plot.index:
   - If found exactly: uses get_loc() - correct position
   - If not found: uses searchsorted() - may give wrong position

3. For daily timeframe, the stored end_ts likely matches exactly because
   daily resampling produces consistent bar boundaries.

4. For other TFs (5min, 1h), the mismatch may come from:
   - The stored end_ts not aligning with the resampled bar boundaries
   - Different resampling producing slightly different indices
   - The lookback/slice approach cutting at different points

The x_offset calculation determines where the channel lines are drawn.
If x_offset is wrong, the channel appears misaligned.
""")


if __name__ == '__main__':
    main()
