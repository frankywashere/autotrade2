#!/usr/bin/env python3
"""
Detailed analysis of break-related features in the cache.
"""

import pickle
import random

def analyze_break_features(sample_file):
    """Analyze break-related features."""
    print(f"\nLoading samples from: {sample_file}")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples\n")

    # Check first sample for all break features with "5min_w50" in the name
    sample = samples[100]
    print(f"Sample timestamp: {sample.timestamp}")

    print("\n" + "=" * 80)
    print("SEARCHING FOR BREAK_SCAN FEATURES")
    print("=" * 80)

    # Search for any feature with "break_scan" in the name
    break_scan_features = sorted([k for k in sample.tf_features.keys() if 'break_scan' in k.lower()])
    print(f"\nFeatures with 'break_scan' in name: {len(break_scan_features)}")

    if break_scan_features:
        for feat in break_scan_features[:30]:
            value = sample.tf_features[feat]
            print(f"  {feat}: {value}")
    else:
        print("  NONE FOUND - break_scan labeling is NOT working!")

    # Search for specific break_scan labels we're looking for
    print("\n" + "=" * 80)
    print("SEARCHING FOR EXPECTED BREAK_SCAN LABELS")
    print("=" * 80)

    expected_labels = [
        '5min_w50_break_scan_valid',
        '5min_w50_bars_to_first_break',
        '5min_w50_permanent_break',
        '5min_w50_returned_to_channel',
        '5min_w50_spy_break_scan_valid',
        '5min_w50_spy_bars_to_first_break',
        '5min_w50_spy_permanent_break',
        '5min_w50_spy_returned_to_channel',
    ]

    print("\nExpected labels:")
    found_count = 0
    for label in expected_labels:
        if label in sample.tf_features:
            value = sample.tf_features[label]
            print(f"  ✓ {label}: {value}")
            found_count += 1
        else:
            print(f"  ✗ {label}: NOT FOUND")

    print(f"\nFound {found_count}/{len(expected_labels)} expected labels")

    # Check what break-related features DO exist for 5min w50
    print("\n" + "=" * 80)
    print("ACTUAL BREAK-RELATED FEATURES FOR 5min w50")
    print("=" * 80)

    tsla_break = sorted([k for k in sample.tf_features.keys()
                         if k.startswith('5min_w50_') and 'break' in k.lower() and '_spy_' not in k])
    spy_break = sorted([k for k in sample.tf_features.keys()
                        if k.startswith('5min_w50_spy_') and 'break' in k.lower()])

    print(f"\nTSLA break features: {len(tsla_break)}")
    for feat in tsla_break:
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")

    print(f"\nSPY break features: {len(spy_break)}")
    for feat in spy_break:
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")

    # Check for old "next_channel" features
    print("\n" + "=" * 80)
    print("CHECKING FOR OLD 'next_channel' FEATURES")
    print("=" * 80)

    old_features = [
        '5min_w50_next_channel_exists',
        '5min_w50_next_channel_distance',
        '5min_w50_spy_next_channel_exists',
        '5min_w50_spy_next_channel_distance',
    ]

    print("\nOld next_channel labels:")
    old_found = 0
    for label in old_features:
        if label in sample.tf_features:
            value = sample.tf_features[label]
            print(f"  ✓ {label}: {value}")
            old_found += 1
        else:
            print(f"  ✗ {label}: NOT FOUND")

    print(f"\nFound {old_found}/{len(old_features)} old labels")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if found_count == 0:
        print("\n✗ FORWARD_SCAN LABELING IS NOT WORKING!")
        print("  No break_scan labels found in the cache.")
        print("  The cache was likely generated with the old labeling code.")

        if old_found > 0:
            print(f"\n  Found {old_found} old 'next_channel' labels - cache is using OLD method")

        print("\n  RECOMMENDATION: Regenerate the cache with the new labeling code.")
    else:
        print("\n✓ FORWARD_SCAN LABELING IS PRESENT!")
        print(f"  Found {found_count}/{len(expected_labels)} expected labels")

        if found_count < len(expected_labels):
            print("\n  ⚠ Warning: Some expected labels are missing")


if __name__ == '__main__':
    analyze_break_features('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl')
