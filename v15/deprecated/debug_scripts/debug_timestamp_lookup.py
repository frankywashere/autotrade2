#!/usr/bin/env python3
"""
Debug script to analyze timestamp lookup issues.

This script loads samples from /tmp/test_timestamps.pkl and prints detailed
information about timestamp lookup for each TF, specifically focusing on
sample index 1 which shows issues with 1h timeframe.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Import v15 types and data loading
from v15.dtypes import ChannelLabels, ChannelSample, TIMEFRAMES, STANDARD_WINDOWS, BARS_PER_TF
from v15.data import load_market_data

# Forward bars configuration per timeframe
FORWARD_BARS_PER_TF = {
    '5min': 100,
    '15min': 100,
    '30min': 50,
    '1h': 50,
    '2h': 50,
    '3h': 50,
    '4h': 50,
    'daily': 50,
    'weekly': 50,
    'monthly': 10,
}


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5min OHLCV data to target timeframe."""
    if timeframe == '5min':
        return df

    rule_map = {
        '15min': '15min', '30min': '30min', '1h': '1h',
        '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W', 'monthly': '1ME'
    }

    rule = rule_map.get(timeframe)
    if not rule:
        return df

    return df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


def debug_sample_timestamps(sample: ChannelSample, tsla_df: pd.DataFrame, sample_idx: int):
    """
    Debug timestamp lookup for a single sample across all TFs.
    """
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx}")
    print(f"{'='*80}")

    print(f"\n1. sample.timestamp = {sample.timestamp}")
    print(f"   sample.channel_end_idx = {sample.channel_end_idx}")
    print(f"   sample.best_window = {sample.best_window}")

    # Find position in 5min data using timestamp
    try:
        channel_end_idx_5min = tsla_df.index.get_loc(sample.timestamp)
        print(f"   5min idx via get_loc(timestamp) = {channel_end_idx_5min}")
    except KeyError:
        idx = tsla_df.index.searchsorted(sample.timestamp)
        channel_end_idx_5min = min(idx, len(tsla_df) - 1)
        print(f"   5min idx via searchsorted (KeyError) = {channel_end_idx_5min}")

    # Check labels for each TF
    timeframes_to_check = ['5min', '1h', 'daily', 'weekly']
    best_window = sample.best_window

    for tf in timeframes_to_check:
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf}")
        print(f"{'='*60}")

        # Get labels from sample
        labels = None
        if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
            if best_window in sample.labels_per_window:
                window_labels = sample.labels_per_window[best_window]
                if 'tsla' in window_labels:
                    labels = window_labels['tsla'].get(tf)
                else:
                    labels = window_labels.get(tf)

        if labels is None:
            print(f"   NO LABELS for {tf} at window {best_window}")
            continue

        # Print source channel timestamps
        start_ts = getattr(labels, 'source_channel_start_ts', None)
        end_ts = getattr(labels, 'source_channel_end_ts', None)

        print(f"\n2. Labels source_channel timestamps:")
        print(f"   source_channel_start_ts = {start_ts}")
        print(f"   source_channel_end_ts   = {end_ts}")

        # Calculate window size that was used
        window = best_window
        print(f"\n3. Window size selected = {window}")

        # Now simulate what get_sample_data_window does
        # Resample data to this TF
        df_historical = tsla_df.iloc[:channel_end_idx_5min + 1]

        if tf == '5min':
            df_channel_tf = df_historical.copy()
            channel_end_idx_tf = channel_end_idx_5min
        else:
            df_channel_tf = resample_ohlc(df_historical, tf)
            channel_end_idx_tf = len(df_channel_tf) - 1

        print(f"\n4. Resampled data analysis:")
        print(f"   len(df_channel_tf) = {len(df_channel_tf)}")
        print(f"   channel_end_idx_tf (len-1) = {channel_end_idx_tf}")

        if len(df_channel_tf) > 0:
            print(f"   df_channel_tf.index[0] = {df_channel_tf.index[0]}")
            print(f"   df_channel_tf.index[-1] = {df_channel_tf.index[-1]}")

        # What channel_end_plot_idx would be if we look up end_ts
        channel_end_plot_idx = None
        last_data_ts = df_channel_tf.index[-1] if len(df_channel_tf) > 0 else None

        if end_ts is not None:
            # KEY DIAGNOSTIC: Is end_ts in the future relative to available data?
            if last_data_ts is not None and end_ts > last_data_ts:
                print(f"\n   *** KEY ISSUE: end_ts ({end_ts}) is AFTER last data ({last_data_ts}) ***")
                print(f"   This means the stored channel ends in the FUTURE relative to this sample!")
                print(f"   The labels may have been computed with forward-looking data.")

            try:
                channel_end_plot_idx = df_channel_tf.index.get_loc(end_ts)
                print(f"\n5. Looking up end_ts in resampled data:")
                print(f"   channel_end_plot_idx = df_channel_tf.index.get_loc(end_ts) = {channel_end_plot_idx}")
            except KeyError:
                # Try searchsorted
                idx = df_channel_tf.index.searchsorted(end_ts)
                channel_end_plot_idx = min(idx, len(df_channel_tf) - 1) if len(df_channel_tf) > 0 else 0
                print(f"\n5. Looking up end_ts in resampled data (KeyError, using searchsorted):")
                print(f"   channel_end_plot_idx = searchsorted result = {channel_end_plot_idx}")

                # Also show nearest timestamps
                if len(df_channel_tf) > 0 and channel_end_plot_idx < len(df_channel_tf):
                    print(f"   Nearest timestamp in data = {df_channel_tf.index[channel_end_plot_idx]}")
                if channel_end_plot_idx > 0:
                    print(f"   Previous timestamp in data = {df_channel_tf.index[channel_end_plot_idx - 1]}")
        else:
            print(f"\n5. end_ts is None - cannot look up channel_end_plot_idx")
            # Fall back to len-1
            channel_end_plot_idx = channel_end_idx_tf
            print(f"   Using fallback: channel_end_plot_idx = channel_end_idx_tf = {channel_end_plot_idx}")

        # Calculate x_offset
        if channel_end_plot_idx is not None:
            x_offset = channel_end_plot_idx - (window - 1)
            print(f"\n6. x_offset calculation:")
            print(f"   x_offset = channel_end_plot_idx - (window - 1)")
            print(f"   x_offset = {channel_end_plot_idx} - ({window} - 1)")
            print(f"   x_offset = {channel_end_plot_idx} - {window - 1}")
            print(f"   x_offset = {x_offset}")

            if x_offset < 0:
                print(f"\n   *** PROBLEM: x_offset is NEGATIVE ({x_offset}) ***")
                print(f"   This means the channel would start before the data begins!")
                print(f"   Need at least {window} bars, but only have {channel_end_plot_idx + 1} bars up to channel_end_plot_idx")

        # Also calculate what channel_start_idx would be
        channel_start_idx = max(0, channel_end_plot_idx - window + 1) if channel_end_plot_idx is not None else 0
        print(f"\n7. Channel visualization range:")
        print(f"   channel_start_idx = max(0, {channel_end_plot_idx} - {window} + 1) = {channel_start_idx}")
        print(f"   channel_end_plot_idx = {channel_end_plot_idx}")
        print(f"   Actual visible bars = {channel_end_plot_idx - channel_start_idx + 1 if channel_end_plot_idx else 0}")

        # Check if this matches the window
        if channel_end_plot_idx is not None:
            actual_visible = channel_end_plot_idx - channel_start_idx + 1
            if actual_visible < window:
                print(f"\n   *** PROBLEM: Only {actual_visible} visible bars, but window is {window} ***")
                print(f"   Channel lines won't align with the data properly!")


def main():
    # Load samples
    samples_path = Path('/tmp/test_timestamps.pkl')
    if not samples_path.exists():
        print(f"Error: {samples_path} not found")
        return

    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples")

    # Load market data
    print("Loading market data...")
    tsla_df, _, _ = load_market_data('data')
    print(f"Loaded {len(tsla_df)} bars of TSLA 5min data")
    print(f"Date range: {tsla_df.index[0]} to {tsla_df.index[-1]}")

    # Debug sample index 1 (the one showing issues)
    if len(samples) > 1:
        debug_sample_timestamps(samples[1], tsla_df, sample_idx=1)
    else:
        print("Not enough samples - debugging sample 0 instead")
        debug_sample_timestamps(samples[0], tsla_df, sample_idx=0)


if __name__ == '__main__':
    main()
