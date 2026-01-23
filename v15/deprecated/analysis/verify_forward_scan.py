#!/usr/bin/env python3
"""
Verify that forward_scan labeling is working in the generated cache.
"""

import pickle
import random
from collections import defaultdict

def verify_forward_scan(sample_file):
    """Verify forward_scan labeling in cache."""
    print(f"\nLoading samples from: {sample_file}")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples\n")

    if len(samples) < 200:
        print(f"Warning: Only {len(samples)} samples available, adjusting range")
        start_pos = max(0, len(samples) - 100)
        end_pos = len(samples)
    else:
        start_pos = 100
        end_pos = 200

    # Pick 10 random samples from positions 100-200
    available_positions = list(range(start_pos, min(end_pos, len(samples))))
    num_samples = min(10, len(available_positions))
    selected_positions = sorted(random.sample(available_positions, num_samples))

    print("=" * 80)
    print(f"VERIFYING FORWARD_SCAN LABELING (checking {num_samples} random samples)")
    print("=" * 80)
    print(f"Selected positions: {selected_positions}\n")

    # Stats tracking
    stats = {
        'tsla_valid': 0,
        'tsla_has_bars_to_break': 0,
        'tsla_permanent_break': 0,
        'tsla_returned_to_channel': 0,
        'spy_valid': 0,
        'spy_has_bars_to_break': 0,
        'spy_permanent_break': 0,
        'spy_returned_to_channel': 0,
    }

    # Check window 50, 5min timeframe labels
    timeframe = '5min'
    window = 50

    for i, pos in enumerate(selected_positions):
        sample = samples[pos]
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {i+1}/{num_samples} - Position {pos}")
        print(f"Timestamp: {sample.timestamp}")
        print(f"{'=' * 80}")

        # Check TSLA labels (primary asset)
        print(f"\n{timeframe} w{window} TSLA Labels:")
        print("-" * 40)

        tsla_prefix = f"{timeframe}_w{window}_"

        # Check break_scan_valid
        break_scan_valid_key = f"{tsla_prefix}break_scan_valid"
        bars_to_first_break_key = f"{tsla_prefix}bars_to_first_break"
        permanent_break_key = f"{tsla_prefix}permanent_break"
        returned_to_channel_key = f"{tsla_prefix}returned_to_channel"

        # Also check for old method indicator
        next_channel_exists_key = f"{tsla_prefix}next_channel_exists"

        break_scan_valid = sample.tf_features.get(break_scan_valid_key, None)
        bars_to_first_break = sample.tf_features.get(bars_to_first_break_key, None)
        permanent_break = sample.tf_features.get(permanent_break_key, None)
        returned_to_channel = sample.tf_features.get(returned_to_channel_key, None)
        next_channel_exists = sample.tf_features.get(next_channel_exists_key, None)

        print(f"  break_scan_valid: {break_scan_valid}")
        print(f"  bars_to_first_break: {bars_to_first_break}")
        print(f"  permanent_break: {permanent_break}")
        print(f"  returned_to_channel: {returned_to_channel}")
        print(f"  next_channel_exists (old method): {next_channel_exists}")

        # Update stats
        if break_scan_valid == 1.0:
            stats['tsla_valid'] += 1
        if bars_to_first_break is not None and bars_to_first_break > 0:
            stats['tsla_has_bars_to_break'] += 1
        if permanent_break == 1.0:
            stats['tsla_permanent_break'] += 1
        if returned_to_channel == 1.0:
            stats['tsla_returned_to_channel'] += 1

        # Check SPY labels
        print(f"\n{timeframe} w{window} SPY Labels:")
        print("-" * 40)

        spy_prefix = f"{timeframe}_w{window}_spy_"

        spy_break_scan_valid_key = f"{spy_prefix}break_scan_valid"
        spy_bars_to_first_break_key = f"{spy_prefix}bars_to_first_break"
        spy_permanent_break_key = f"{spy_prefix}permanent_break"
        spy_returned_to_channel_key = f"{spy_prefix}returned_to_channel"
        spy_next_channel_exists_key = f"{spy_prefix}next_channel_exists"

        spy_break_scan_valid = sample.tf_features.get(spy_break_scan_valid_key, None)
        spy_bars_to_first_break = sample.tf_features.get(spy_bars_to_first_break_key, None)
        spy_permanent_break = sample.tf_features.get(spy_permanent_break_key, None)
        spy_returned_to_channel = sample.tf_features.get(spy_returned_to_channel_key, None)
        spy_next_channel_exists = sample.tf_features.get(spy_next_channel_exists_key, None)

        print(f"  break_scan_valid: {spy_break_scan_valid}")
        print(f"  bars_to_first_break: {spy_bars_to_first_break}")
        print(f"  permanent_break: {spy_permanent_break}")
        print(f"  returned_to_channel: {spy_returned_to_channel}")
        print(f"  next_channel_exists (old method): {spy_next_channel_exists}")

        # Update stats
        if spy_break_scan_valid == 1.0:
            stats['spy_valid'] += 1
        if spy_bars_to_first_break is not None and spy_bars_to_first_break > 0:
            stats['spy_has_bars_to_break'] += 1
        if spy_permanent_break == 1.0:
            stats['spy_permanent_break'] += 1
        if spy_returned_to_channel == 1.0:
            stats['spy_returned_to_channel'] += 1

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTSLA ({timeframe} w{window}):")
    print(f"  Samples with break_scan_valid=True: {stats['tsla_valid']}/{num_samples} ({stats['tsla_valid']/num_samples*100:.1f}%)")
    print(f"  Samples with bars_to_first_break > 0: {stats['tsla_has_bars_to_break']}/{num_samples} ({stats['tsla_has_bars_to_break']/num_samples*100:.1f}%)")
    print(f"  Samples with permanent_break=True: {stats['tsla_permanent_break']}/{num_samples} ({stats['tsla_permanent_break']/num_samples*100:.1f}%)")
    print(f"  Samples with returned_to_channel=True: {stats['tsla_returned_to_channel']}/{num_samples} ({stats['tsla_returned_to_channel']/num_samples*100:.1f}%)")

    print(f"\nSPY ({timeframe} w{window}):")
    print(f"  Samples with break_scan_valid=True: {stats['spy_valid']}/{num_samples} ({stats['spy_valid']/num_samples*100:.1f}%)")
    print(f"  Samples with bars_to_first_break > 0: {stats['spy_has_bars_to_break']}/{num_samples} ({stats['spy_has_bars_to_break']/num_samples*100:.1f}%)")
    print(f"  Samples with permanent_break=True: {stats['spy_permanent_break']}/{num_samples} ({stats['spy_permanent_break']/num_samples*100:.1f}%)")
    print(f"  Samples with returned_to_channel=True: {stats['spy_returned_to_channel']}/{num_samples} ({stats['spy_returned_to_channel']/num_samples*100:.1f}%)")

    # Determine if forward_scan is working
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if stats['tsla_valid'] > 0 or stats['spy_valid'] > 0:
        print("\n✓ FORWARD_SCAN IS WORKING!")
        print(f"  Found {stats['tsla_valid']} TSLA samples and {stats['spy_valid']} SPY samples with valid break_scan data")

        if stats['tsla_has_bars_to_break'] > 0 or stats['spy_has_bars_to_break'] > 0:
            print(f"  Detected breaks in {stats['tsla_has_bars_to_break']} TSLA and {stats['spy_has_bars_to_break']} SPY samples")

        if stats['tsla_valid'] == num_samples and stats['spy_valid'] == num_samples:
            print("\n  ALL samples have valid forward_scan data - excellent!")
        elif stats['tsla_valid'] < num_samples or stats['spy_valid'] < num_samples:
            print("\n  ⚠ Warning: Not all samples have valid forward_scan data")
            print(f"    TSLA missing: {num_samples - stats['tsla_valid']} samples")
            print(f"    SPY missing: {num_samples - stats['spy_valid']} samples")
    else:
        print("\n✗ FORWARD_SCAN IS NOT WORKING!")
        print("  No samples found with break_scan_valid=True")
        print("  The cache may still be using the old next_channel method")


if __name__ == '__main__':
    verify_forward_scan('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl')
