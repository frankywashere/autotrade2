#!/usr/bin/env python3
"""
Debug script to check what features actually exist in samples.
"""

import pickle
import sys
from collections import defaultdict

def analyze_sample_features(sample_file):
    """Load a sample and analyze its features."""
    print(f"\nLoading samples from: {sample_file}")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples\n")

    if len(samples) == 0:
        print("No samples found!")
        return

    # Analyze first sample
    sample = samples[0]

    print(f"Sample timestamp: {sample.timestamp}")
    print(f"Best window: {sample.best_window}")
    print(f"Total features: {len(sample.tf_features)}\n")

    # Group features by timeframe and window
    feature_groups = defaultdict(list)

    for feature_name in sorted(sample.tf_features.keys()):
        parts = feature_name.split('_')

        # Try to identify timeframe
        if feature_name.startswith('5min_'):
            tf = '5min'
        elif feature_name.startswith('15min_'):
            tf = '15min'
        elif feature_name.startswith('30min_'):
            tf = '30min'
        elif feature_name.startswith('1h_'):
            tf = '1h'
        elif feature_name.startswith('2h_'):
            tf = '2h'
        elif feature_name.startswith('3h_'):
            tf = '3h'
        elif feature_name.startswith('4h_'):
            tf = '4h'
        elif feature_name.startswith('daily_'):
            tf = 'daily'
        elif feature_name.startswith('weekly_'):
            tf = 'weekly'
        elif feature_name.startswith('monthly_'):
            tf = 'monthly'
        else:
            tf = 'global'

        # Try to identify window
        window = None
        for i, part in enumerate(parts):
            if part.startswith('w') and len(part) > 1 and part[1:].isdigit():
                window = part
                break

        key = (tf, window)
        feature_groups[key].append(feature_name)

    # Print summary by timeframe
    print("=" * 80)
    print("FEATURE COUNT BY TIMEFRAME AND WINDOW")
    print("=" * 80)

    # Sort with proper handling of None values
    sorted_groups = sorted(feature_groups.items(), key=lambda x: (x[0][0], x[0][1] or ''))

    for (tf, window), features in sorted_groups:
        if window:
            print(f"\n{tf} {window}: {len(features)} features")
        else:
            print(f"\n{tf} (no window): {len(features)} features")

        # Show first 10 features as examples
        print("  Examples:")
        for feat in features[:10]:
            value = sample.tf_features[feat]
            print(f"    {feat}: {value:.6f}")

        if len(features) > 10:
            print(f"    ... and {len(features) - 10} more")

    # Now specifically check for the features we're looking for
    print("\n" + "=" * 80)
    print("CHECKING SPECIFIC FEATURES")
    print("=" * 80)

    test_cases = [
        ('5min', 50, 'tsla'),
        ('5min', 50, 'spy'),
        ('15min', 50, 'tsla'),
        ('1h', 30, 'tsla'),
    ]

    for tf, window, asset in test_cases:
        print(f"\n{tf} w{window} {asset.upper()}:")

        if asset == 'tsla':
            slope_key = f"{tf}_w{window}_channel_slope"
            intercept_key = f"{tf}_w{window}_channel_intercept"
            std_dev_key = f"{tf}_w{window}_std_dev_ratio"
        else:
            slope_key = f"{tf}_w{window}_spy_channel_slope"
            intercept_key = f"{tf}_w{window}_spy_channel_intercept"
            std_dev_key = f"{tf}_w{window}_spy_std_dev_ratio"

        for key in [slope_key, intercept_key, std_dev_key]:
            if key in sample.tf_features:
                print(f"  ✓ {key}: {sample.tf_features[key]:.6f}")
            else:
                print(f"  ✗ {key}: NOT FOUND")

                # Try to find similar keys
                similar = [k for k in sample.tf_features.keys() if 'channel_slope' in k or 'intercept' in k or 'std_dev' in k]
                if similar:
                    print(f"    Similar features found:")
                    for s in sorted(similar)[:5]:
                        print(f"      - {s}")

    # Check what windows are actually available for 5min timeframe
    print("\n" + "=" * 80)
    print("AVAILABLE WINDOWS FOR 5MIN TIMEFRAME")
    print("=" * 80)

    windows_5min = set()
    for feature_name in sample.tf_features.keys():
        if feature_name.startswith('5min_w'):
            parts = feature_name.split('_')
            if len(parts) >= 2 and parts[1].startswith('w'):
                window_str = parts[1]
                windows_5min.add(window_str)

    print(f"\nWindows found: {sorted(windows_5min)}")

    # Sample some channel features from each window
    for window_str in sorted(windows_5min)[:3]:
        print(f"\n{window_str} features (sample):")
        matching = [k for k in sorted(sample.tf_features.keys()) if f'5min_{window_str}_' in k]
        for feat in matching[:5]:
            print(f"  {feat}: {sample.tf_features[feat]:.6f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_features.py <path_to_samples.pkl>")
        print("\nExample:")
        print("  python debug_features.py v15/cache/scanner_samples_2024-08-15_to_2024-08-16.pkl")
        sys.exit(1)

    analyze_sample_features(sys.argv[1])
