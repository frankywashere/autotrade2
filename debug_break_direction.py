#!/usr/bin/env python3
"""
Debug script to print the actual values being used in break direction calculation.

This script:
1. Loads the test samples from /tmp/test_samples_fixed.pkl
2. For each sample, regenerates the channel detection for the daily timeframe
3. Prints all the values used in the break direction calculation
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v15.data import load_market_data
from v15.labels import (
    detect_all_channels,
    generate_all_labels,
    BreakDirection,
    NewChannelDirection,
)
from v7.core.timeframe import resample_ohlc


def debug_break_direction_calculation(
    channel_map,
    labeled_map,
    tf: str = 'daily',
    window: int = 50,
    max_channels: int = 10
):
    """
    Print detailed break direction calculation for channels.

    Args:
        channel_map: The channel map from detect_all_channels()
        labeled_map: The labeled channel map from generate_all_labels()
        tf: Timeframe to debug (default: 'daily')
        window: Window size to debug (default: 50)
        max_channels: Maximum number of channels to print (default: 10)
    """
    key = (tf, window)
    channels = channel_map.get(key, [])
    labeled_channels = labeled_map.get(key, [])

    print(f"\n{'='*80}")
    print(f"DEBUG: Break Direction Calculation for TF={tf}, Window={window}")
    print(f"{'='*80}")
    print(f"Total channels detected: {len(channels)}")
    print(f"Total labeled channels: {len(labeled_channels)}")

    if not channels:
        print("No channels found for this TF/window combination!")
        return

    # Process each channel pair (current + next)
    for i, (detected, labeled) in enumerate(zip(channels[:-1], labeled_channels[:-1])):
        if i >= max_channels:
            print(f"\n... (showing first {max_channels} channels only)")
            break

        next_detected = channels[i + 1]
        next_labeled = labeled_channels[i + 1]

        curr_channel = detected.channel
        next_channel = next_detected.channel

        print(f"\n{'-'*80}")
        print(f"CHANNEL {i}: {detected.start_timestamp} to {detected.end_timestamp}")
        print(f"{'-'*80}")

        # Current channel properties
        print(f"\n  CURRENT CHANNEL PROPERTIES:")
        print(f"    curr_channel.slope        = {curr_channel.slope:.8f}")
        print(f"    curr_channel.intercept    = {curr_channel.intercept:.4f}")
        print(f"    curr_channel.std_dev      = {curr_channel.std_dev:.4f}")
        print(f"    curr_channel.window       = {curr_channel.window}")
        print(f"    curr_channel.direction    = {curr_channel.direction}")

        # Close prices
        if curr_channel.close is not None and len(curr_channel.close) > 0:
            print(f"    curr_channel.close[-1]    = {curr_channel.close[-1]:.4f} (last price)")
            print(f"    curr_channel.close[0]     = {curr_channel.close[0]:.4f} (first price)")
            print(f"    curr_channel.close.shape  = {curr_channel.close.shape}")
        else:
            print(f"    curr_channel.close        = None or empty!")

        # Next channel properties
        print(f"\n  NEXT CHANNEL PROPERTIES:")
        print(f"    Start timestamp:          = {next_detected.start_timestamp}")
        print(f"    End timestamp:            = {next_detected.end_timestamp}")
        print(f"    next_channel.slope        = {next_channel.slope:.8f}")
        print(f"    next_channel.intercept    = {next_channel.intercept:.4f}")
        print(f"    next_channel.direction    = {next_channel.direction}")

        if next_channel.close is not None and len(next_channel.close) > 0:
            print(f"    next_channel.close[0]     = {next_channel.close[0]:.4f} (first price = next_start_price)")
        else:
            print(f"    next_channel.close        = None or empty!")

        # Duration calculation
        duration_bars = next_detected.start_idx - detected.end_idx
        print(f"\n  DURATION:")
        print(f"    current.end_idx           = {detected.end_idx}")
        print(f"    next.start_idx            = {next_detected.start_idx}")
        print(f"    duration_bars             = {duration_bars}")

        # Projection calculation
        print(f"\n  PROJECTION CALCULATION:")
        projection_x = curr_channel.window - 1 + duration_bars
        print(f"    projection_x              = window - 1 + duration_bars")
        print(f"                              = {curr_channel.window} - 1 + {duration_bars}")
        print(f"                              = {projection_x}")

        projected_center = curr_channel.slope * projection_x + curr_channel.intercept
        print(f"    projected_center          = slope * projection_x + intercept")
        print(f"                              = {curr_channel.slope:.8f} * {projection_x} + {curr_channel.intercept:.4f}")
        print(f"                              = {projected_center:.4f}")

        std_multiplier = 2.0
        projected_upper = projected_center + std_multiplier * curr_channel.std_dev
        projected_lower = projected_center - std_multiplier * curr_channel.std_dev

        print(f"    std_multiplier            = {std_multiplier}")
        print(f"    projected_upper           = projected_center + 2.0 * std_dev")
        print(f"                              = {projected_center:.4f} + 2.0 * {curr_channel.std_dev:.4f}")
        print(f"                              = {projected_upper:.4f}")
        print(f"    projected_lower           = projected_center - 2.0 * std_dev")
        print(f"                              = {projected_center:.4f} - 2.0 * {curr_channel.std_dev:.4f}")
        print(f"                              = {projected_lower:.4f}")

        # Break direction determination
        print(f"\n  BREAK DIRECTION DETERMINATION:")
        if next_channel.close is not None and len(next_channel.close) > 0:
            next_start_price = next_channel.close[0]
            print(f"    next_start_price          = {next_start_price:.4f}")
            print(f"    projected_upper           = {projected_upper:.4f}")
            print(f"    projected_lower           = {projected_lower:.4f}")

            if next_start_price > projected_upper:
                calc_break_dir = BreakDirection.UP
                print(f"    CONDITION: next_start_price ({next_start_price:.4f}) > projected_upper ({projected_upper:.4f})")
                print(f"    => break_direction = UP (1)")
            elif next_start_price < projected_lower:
                calc_break_dir = BreakDirection.DOWN
                print(f"    CONDITION: next_start_price ({next_start_price:.4f}) < projected_lower ({projected_lower:.4f})")
                print(f"    => break_direction = DOWN (0)")
            else:
                print(f"    CONDITION: next_start_price ({next_start_price:.4f}) is WITHIN bounds")
                print(f"               [{projected_lower:.4f}, {projected_upper:.4f}]")
                # Fallback logic
                if curr_channel.close is not None and len(curr_channel.close) > 0:
                    curr_end_price = curr_channel.close[-1]
                    print(f"    FALLBACK: Compare to curr_end_price = {curr_end_price:.4f}")
                    if next_start_price > curr_end_price:
                        calc_break_dir = BreakDirection.UP
                        print(f"    => break_direction = UP (1) (price moved up)")
                    else:
                        calc_break_dir = BreakDirection.DOWN
                        print(f"    => break_direction = DOWN (0) (price moved down)")
                else:
                    calc_break_dir = BreakDirection.UP
                    print(f"    FALLBACK: No curr_channel.close, defaulting to UP")
        else:
            print(f"    WARNING: No next_channel.close data available!")
            calc_break_dir = BreakDirection.UP

        # Compare to stored label
        stored_label = labeled.labels
        print(f"\n  RESULT:")
        print(f"    Calculated break_direction = {calc_break_dir} ({'UP' if calc_break_dir == 1 else 'DOWN'})")
        print(f"    Stored break_direction     = {stored_label.break_direction} ({'UP' if stored_label.break_direction == 1 else 'DOWN'})")
        print(f"    direction_valid            = {stored_label.direction_valid}")
        print(f"    new_channel_direction      = {stored_label.new_channel_direction} ({['BEAR', 'SIDEWAYS', 'BULL'][stored_label.new_channel_direction]})")

        if calc_break_dir != stored_label.break_direction:
            print(f"    *** MISMATCH! Calculated != Stored ***")


def main():
    # Try to load existing samples first
    samples_path = Path("/tmp/test_samples_fixed.pkl")
    if not samples_path.exists():
        samples_path = Path("/tmp/test_samples_v15.pkl")

    if samples_path.exists():
        print(f"Loading samples from {samples_path}...")
        with open(samples_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"Loaded {len(samples)} samples")

        # Show sample timestamps for reference
        print("\nSample timestamps:")
        for i, s in enumerate(samples):
            print(f"  [{i}] {s.timestamp}, channel_end_idx={s.channel_end_idx}")
    else:
        print(f"No samples file found at {samples_path}")
        samples = []

    # Load market data to regenerate channel maps
    print("\nLoading market data...")
    tsla, spy, vix = load_market_data("data")
    print(f"Loaded {len(tsla)} bars, range: {tsla.index[0]} to {tsla.index[-1]}")

    # Get a focused slice around the sample timestamps
    if samples:
        # Use the first sample's position as reference
        sample_idx = samples[0].channel_end_idx

        # Get a reasonable slice (include some context before and after)
        start_idx = max(0, sample_idx - 5000)  # ~5000 bars before
        end_idx = min(len(tsla), sample_idx + 5000)  # ~5000 bars after

        print(f"\nUsing data slice: idx {start_idx} to {end_idx}")
        tsla_slice = tsla.iloc[start_idx:end_idx]
    else:
        # Use a reasonable default slice
        print("\nUsing last 10000 bars of data")
        tsla_slice = tsla.iloc[-10000:]

    print(f"Data slice: {len(tsla_slice)} bars, {tsla_slice.index[0]} to {tsla_slice.index[-1]}")

    # Run channel detection on this slice
    print("\n" + "="*80)
    print("RUNNING CHANNEL DETECTION (Pass 1)")
    print("="*80)

    # Only detect for daily timeframe and a few windows for debugging
    test_windows = [10, 20, 30, 40, 50, 60]

    channel_map = detect_all_channels(
        df=tsla_slice,
        timeframes=['daily'],
        windows=test_windows,
        step=1,  # Dense scanning for debug
        min_cycles=1,
        min_gap_bars=5,
        verbose=True,
        workers=1  # Sequential for debugging
    )

    # Run label generation
    print("\n" + "="*80)
    print("RUNNING LABEL GENERATION (Pass 2)")
    print("="*80)

    labeled_map = generate_all_labels(
        channel_map=channel_map,
        verbose=True
    )

    # Debug each window size
    for window in test_windows:
        debug_break_direction_calculation(
            channel_map=channel_map,
            labeled_map=labeled_map,
            tf='daily',
            window=window,
            max_channels=5  # Show first 5 channels per window
        )

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for window in test_windows:
        key = ('daily', window)
        labeled_channels = labeled_map.get(key, [])

        if not labeled_channels:
            print(f"\nWindow {window}: No channels")
            continue

        up_count = sum(1 for lc in labeled_channels if lc.labels.direction_valid and lc.labels.break_direction == 1)
        down_count = sum(1 for lc in labeled_channels if lc.labels.direction_valid and lc.labels.break_direction == 0)
        invalid_count = sum(1 for lc in labeled_channels if not lc.labels.direction_valid)

        total_valid = up_count + down_count
        if total_valid > 0:
            up_pct = 100 * up_count / total_valid
            down_pct = 100 * down_count / total_valid
        else:
            up_pct = down_pct = 0

        print(f"\nWindow {window}:")
        print(f"  Total channels: {len(labeled_channels)}")
        print(f"  Valid labels: {total_valid}")
        print(f"  Invalid labels: {invalid_count}")
        print(f"  UP breaks: {up_count} ({up_pct:.1f}%)")
        print(f"  DOWN breaks: {down_count} ({down_pct:.1f}%)")


if __name__ == "__main__":
    main()
