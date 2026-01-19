#!/usr/bin/env python3
"""
Debug channel detection across timeframes.
Check why channels are only detected for 5min and not higher TFs.
"""

import pickle
import sys
from collections import defaultdict

def check_channel_detection(sample_file):
    """Check channel detection success across timeframes."""
    print(f"\nLoading samples from: {sample_file}\n")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    if len(samples) == 0:
        print("No samples found!")
        return

    # Analyze first sample
    sample = samples[0]

    timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']
    windows = [10, 20, 30, 40, 50, 60, 70, 80]

    print("=" * 80)
    print("CHANNEL DETECTION SUCCESS BY TIMEFRAME AND WINDOW")
    print("=" * 80)
    print(f"\nSample: {sample.timestamp}")
    print(f"Best window: {sample.best_window}\n")

    # Check for each TF and window
    results = defaultdict(lambda: defaultdict(dict))

    for tf in timeframes:
        for window in windows:
            # Check if channel features exist and are non-zero
            slope_key = f"{tf}_w{window}_channel_slope"
            intercept_key = f"{tf}_w{window}_channel_intercept"
            std_dev_key = f"{tf}_w{window}_std_dev_ratio"
            valid_key = f"{tf}_w{window}_channel_valid"

            slope = sample.tf_features.get(slope_key, None)
            intercept = sample.tf_features.get(intercept_key, None)
            std_dev = sample.tf_features.get(std_dev_key, None)
            valid = sample.tf_features.get(valid_key, None)

            # Check if channel was detected (non-zero values)
            if slope is None or intercept is None or std_dev is None:
                status = "MISSING"
            elif valid == 1.0 and abs(slope) > 1e-9:
                status = "VALID"
            else:
                status = "ZERO"

            results[tf][window] = {
                'status': status,
                'slope': slope,
                'intercept': intercept,
                'std_dev': std_dev,
                'valid': valid
            }

    # Print summary table
    print("Status codes: VALID = channel detected, ZERO = no channel, MISSING = feature not found")
    print()
    print("TF        " + "".join(f"  w{w:2d}  " for w in windows))
    print("-" * 80)

    for tf in timeframes:
        status_str = ""
        for window in windows:
            status = results[tf][window]['status']
            if status == "VALID":
                status_str += f"   ✓   "
            elif status == "ZERO":
                status_str += f"   -   "
            else:
                status_str += f"   ✗   "

        print(f"{tf:8s}  {status_str}")

    # Print detailed info for each TF
    print("\n" + "=" * 80)
    print("DETAILED CHANNEL PARAMETERS")
    print("=" * 80)

    for tf in timeframes:
        print(f"\n{tf} timeframe:")

        valid_count = sum(1 for w in windows if results[tf][w]['status'] == 'VALID')
        zero_count = sum(1 for w in windows if results[tf][w]['status'] == 'ZERO')
        missing_count = sum(1 for w in windows if results[tf][w]['status'] == 'MISSING')

        print(f"  Valid: {valid_count}, Zero: {zero_count}, Missing: {missing_count}")

        # Show example from window 50 if available
        w = 50
        if w in results[tf]:
            data = results[tf][w]
            print(f"  Window {w} example:")
            print(f"    slope: {data['slope']}")
            print(f"    intercept: {data['intercept']}")
            print(f"    std_dev: {data['std_dev']}")
            print(f"    valid: {data['valid']}")

    # Check SPY channels too
    print("\n" + "=" * 80)
    print("SPY CHANNEL DETECTION")
    print("=" * 80)

    spy_results = defaultdict(lambda: defaultdict(dict))

    for tf in timeframes:
        for window in windows:
            slope_key = f"{tf}_w{window}_spy_channel_slope"
            intercept_key = f"{tf}_w{window}_spy_channel_intercept"
            std_dev_key = f"{tf}_w{window}_spy_std_dev_ratio"
            valid_key = f"{tf}_w{window}_spy_channel_valid"

            slope = sample.tf_features.get(slope_key, None)
            intercept = sample.tf_features.get(intercept_key, None)
            std_dev = sample.tf_features.get(std_dev_key, None)
            valid = sample.tf_features.get(valid_key, None)

            if slope is None or intercept is None or std_dev is None:
                status = "MISSING"
            elif valid == 1.0 and abs(slope) > 1e-9:
                status = "VALID"
            else:
                status = "ZERO"

            spy_results[tf][window] = {
                'status': status,
                'slope': slope,
                'intercept': intercept,
                'std_dev': std_dev,
                'valid': valid
            }

    print("TF        " + "".join(f"  w{w:2d}  " for w in windows))
    print("-" * 80)

    for tf in timeframes:
        status_str = ""
        for window in windows:
            status = spy_results[tf][window]['status']
            if status == "VALID":
                status_str += f"   ✓   "
            elif status == "ZERO":
                status_str += f"   -   "
            else:
                status_str += f"   ✗   "

        print(f"{tf:8s}  {status_str}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_tf_window_combos = len(timeframes) * len(windows)
    tsla_valid = sum(1 for tf in timeframes for w in windows if results[tf][w]['status'] == 'VALID')
    spy_valid = sum(1 for tf in timeframes for w in windows if spy_results[tf][w]['status'] == 'VALID')

    print(f"\nTotal TF/window combinations: {total_tf_window_combos}")
    print(f"TSLA channels detected: {tsla_valid} ({100*tsla_valid/total_tf_window_combos:.1f}%)")
    print(f"SPY channels detected: {spy_valid} ({100*spy_valid/total_tf_window_combos:.1f}%)")

    # Check which TFs have any valid channels
    print("\nTimeframes with valid channels:")
    for tf in timeframes:
        tsla_valid_tf = sum(1 for w in windows if results[tf][w]['status'] == 'VALID')
        spy_valid_tf = sum(1 for w in windows if spy_results[tf][w]['status'] == 'VALID')
        if tsla_valid_tf > 0 or spy_valid_tf > 0:
            print(f"  {tf}: TSLA={tsla_valid_tf}, SPY={spy_valid_tf}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_channel_detection.py <path_to_samples.pkl>")
        print("\nExample:")
        print("  python debug_channel_detection.py test_samples.pkl")
        sys.exit(1)

    check_channel_detection(sys.argv[1])
