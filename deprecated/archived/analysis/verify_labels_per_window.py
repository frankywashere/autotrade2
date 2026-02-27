#!/usr/bin/env python3
"""
Verify that forward_scan labels exist in labels_per_window structure.
"""

import pickle
import random

def verify_labels_in_sample(sample_file):
    """Verify labels_per_window structure has forward_scan labels."""
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
    print(f"VERIFYING FORWARD_SCAN LABELS IN labels_per_window (checking {num_samples} random samples)")
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

    window = 50
    timeframe = '5min'

    for i, pos in enumerate(selected_positions):
        sample = samples[pos]
        print(f"\n{'=' * 80}")
        print(f"SAMPLE {i+1}/{num_samples} - Position {pos}")
        print(f"Timestamp: {sample.timestamp}")
        print(f"Best window: {sample.best_window}")
        print(f"{'=' * 80}")

        # Check if labels_per_window exists
        if not hasattr(sample, 'labels_per_window') or sample.labels_per_window is None:
            print("\n  ✗ ERROR: sample.labels_per_window is None or missing!")
            continue

        # Check if window 50 exists in labels_per_window
        if window not in sample.labels_per_window:
            print(f"\n  ✗ ERROR: window {window} not found in labels_per_window")
            print(f"  Available windows: {list(sample.labels_per_window.keys())}")
            continue

        # Check if assets exist
        window_labels = sample.labels_per_window[window]
        if 'tsla' not in window_labels or 'spy' not in window_labels:
            print(f"\n  ✗ ERROR: tsla/spy not found in window {window} labels")
            print(f"  Available keys: {list(window_labels.keys())}")
            continue

        # Check TSLA labels
        print(f"\n{timeframe} w{window} TSLA Labels:")
        print("-" * 40)

        tsla_tf_labels = window_labels['tsla']
        if timeframe not in tsla_tf_labels:
            print(f"  ✗ ERROR: {timeframe} not found in TSLA labels")
            print(f"  Available timeframes: {list(tsla_tf_labels.keys())}")
        else:
            tsla_labels = tsla_tf_labels[timeframe]
            if tsla_labels is None:
                print("  ✗ TSLA labels are None")
            else:
                print(f"  break_scan_valid: {tsla_labels.break_scan_valid}")
                print(f"  bars_to_first_break: {tsla_labels.bars_to_first_break}")
                print(f"  permanent_break: {tsla_labels.permanent_break}")
                print(f"  returned_to_channel: {tsla_labels.returned_to_channel}")
                print(f"  break_direction: {tsla_labels.break_direction}")
                print(f"  first_break_direction: {tsla_labels.first_break_direction}")
                print(f"  break_magnitude: {tsla_labels.break_magnitude:.2f}")

                # Update stats
                if tsla_labels.break_scan_valid:
                    stats['tsla_valid'] += 1
                if tsla_labels.bars_to_first_break is not None and tsla_labels.bars_to_first_break > 0:
                    stats['tsla_has_bars_to_break'] += 1
                if tsla_labels.permanent_break:
                    stats['tsla_permanent_break'] += 1
                if tsla_labels.returned_to_channel:
                    stats['tsla_returned_to_channel'] += 1

        # Check SPY labels
        print(f"\n{timeframe} w{window} SPY Labels:")
        print("-" * 40)

        spy_tf_labels = window_labels['spy']
        if timeframe not in spy_tf_labels:
            print(f"  ✗ ERROR: {timeframe} not found in SPY labels")
            print(f"  Available timeframes: {list(spy_tf_labels.keys())}")
        else:
            spy_labels = spy_tf_labels[timeframe]
            if spy_labels is None:
                print("  ✗ SPY labels are None")
            else:
                print(f"  break_scan_valid: {spy_labels.break_scan_valid}")
                print(f"  bars_to_first_break: {spy_labels.bars_to_first_break}")
                print(f"  permanent_break: {spy_labels.permanent_break}")
                print(f"  returned_to_channel: {spy_labels.returned_to_channel}")
                print(f"  break_direction: {spy_labels.break_direction}")
                print(f"  first_break_direction: {spy_labels.first_break_direction}")
                print(f"  break_magnitude: {spy_labels.break_magnitude:.2f}")

                # Update stats
                if spy_labels.break_scan_valid:
                    stats['spy_valid'] += 1
                if spy_labels.bars_to_first_break is not None and spy_labels.bars_to_first_break > 0:
                    stats['spy_has_bars_to_break'] += 1
                if spy_labels.permanent_break:
                    stats['spy_permanent_break'] += 1
                if spy_labels.returned_to_channel:
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

        print("\n  NOTE: Labels are stored in sample.labels_per_window[window][asset][tf]")
        print("        They are NOT in sample.tf_features dict")
    else:
        print("\n✗ FORWARD_SCAN IS NOT WORKING!")
        print("  No samples found with break_scan_valid=True")
        print("  The cache may still be using the old next_channel method")


if __name__ == '__main__':
    verify_labels_in_sample('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl')
