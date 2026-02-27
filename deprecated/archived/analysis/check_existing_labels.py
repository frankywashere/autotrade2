#!/usr/bin/env python3
"""
Check what labels actually exist in the samples.
"""

import pickle

def check_existing_labels(sample_file):
    """Check what labels exist in samples."""
    print(f"\nLoading samples from: {sample_file}")

    with open(sample_file, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples\n")

    # Check first sample
    sample = samples[100]
    print(f"Sample timestamp: {sample.timestamp}")
    print(f"Sample best_window: {sample.best_window}")

    # Find all 5min w50 features
    print("\n" + "=" * 80)
    print("ALL 5min w50 FEATURES (TSLA)")
    print("=" * 80)

    tsla_features = sorted([k for k in sample.tf_features.keys() if k.startswith('5min_w50_') and '_spy_' not in k])
    print(f"\nFound {len(tsla_features)} features:\n")

    for feat in tsla_features:
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")

    # Find all 5min w50 SPY features
    print("\n" + "=" * 80)
    print("ALL 5min w50 FEATURES (SPY)")
    print("=" * 80)

    spy_features = sorted([k for k in sample.tf_features.keys() if k.startswith('5min_w50_spy_')])
    print(f"\nFound {len(spy_features)} features:\n")

    for feat in spy_features:
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")

    # Check for any break-related features
    print("\n" + "=" * 80)
    print("ALL BREAK-RELATED FEATURES (any timeframe/window)")
    print("=" * 80)

    break_features = sorted([k for k in sample.tf_features.keys() if 'break' in k.lower()])
    print(f"\nFound {len(break_features)} break-related features:\n")

    for feat in break_features[:20]:  # Show first 20
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")

    if len(break_features) > 20:
        print(f"\n  ... and {len(break_features) - 20} more")

    # Check for any channel-related features
    print("\n" + "=" * 80)
    print("ALL CHANNEL-RELATED FEATURES (5min w50 only)")
    print("=" * 80)

    channel_features = sorted([k for k in sample.tf_features.keys() if 'channel' in k.lower() and '5min_w50' in k])
    print(f"\nFound {len(channel_features)} channel-related features:\n")

    for feat in channel_features:
        value = sample.tf_features[feat]
        print(f"  {feat}: {value}")


if __name__ == '__main__':
    check_existing_labels('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl')
