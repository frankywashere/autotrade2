#!/usr/bin/env python3
"""
Test that scanner populates channel history features.

This test validates that the channel history tracking implementation
correctly populates the 670 channel history features (67 per TF × 10 TFs).

Usage:
    # Run a small scan first to generate test samples
    ./build_manual/bin/v15_scanner \
        --data-dir /path/to/data \
        --output /tmp/history_test_samples.bin \
        --max-samples 200 \
        --warmup-bars 35000

    # Then run this test
    python3 tests/test_history_scan.py /tmp/history_test_samples.bin
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_samples import load_samples


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_history_scan.py <samples.bin>")
        print("\nRun a scan first to generate samples:")
        print("  ./build_manual/bin/v15_scanner --data-dir data --output /tmp/test.bin --max-samples 200")
        return 1

    samples_path = sys.argv[1]

    if not os.path.exists(samples_path):
        print(f"❌ Samples file not found: {samples_path}")
        return 1

    # Load samples
    print(f"Loading samples from {samples_path}...")
    version, num_samples, num_features, samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} samples (version={version}, features={num_features})")

    if len(samples) == 0:
        print("❌ No samples loaded")
        return 1

    # Get feature names from first sample
    sample = samples[0]
    feature_names = list(sample.tf_features.keys())
    print(f"Sample has {len(feature_names)} features")

    # Find channel history features (67 per TF × 10 TFs = 670 total)
    # They have names like: 5min_tsla_last5_avg_duration, 1h_spy_channel_momentum, etc.
    history_prefixes = [
        'last5_', 'channel_momentum', 'channel_regime_shift',
        'slope_trend', 'duration_trend', 'quality_trend',
        'cross_asset_channel'
    ]

    history_features = []
    for fname in feature_names:
        for prefix in history_prefixes:
            if prefix in fname:
                history_features.append(fname)
                break

    print(f"\nFound {len(history_features)} channel history features")

    if len(history_features) == 0:
        print("⚠️  No history features found by name pattern")
        # Try listing some feature names
        print("Sample feature names:")
        for fname in sorted(feature_names)[:20]:
            print(f"  {fname}")
        return 1

    # Check default values
    # Known defaults: duration=50.0, slope=0.0, direction_pattern=0.5, r_squared=0.0
    default_values = {0.0, 50.0, 0.5, 2.0}

    non_default_count = 0
    default_count = 0
    total_checked = 0

    # Sample specific feature values for debugging
    sample_values = {}

    for sample in samples[:100]:
        for fname in history_features:
            if fname in sample.tf_features:
                val = sample.tf_features[fname]
                total_checked += 1

                # Track sample values
                if fname not in sample_values:
                    sample_values[fname] = []
                if len(sample_values[fname]) < 5:
                    sample_values[fname].append(val)

                if val in default_values:
                    default_count += 1
                else:
                    non_default_count += 1

    if total_checked > 0:
        real_pct = 100.0 * non_default_count / total_checked
        print(f"\nResults for {total_checked} feature values across {min(100, len(samples))} samples:")
        print(f"  Non-default values: {non_default_count} ({real_pct:.1f}%)")
        print(f"  Default values: {default_count} ({100.0 - real_pct:.1f}%)")

        # Show sample values
        print("\nSample values for first few history features:")
        for fname, values in list(sample_values.items())[:10]:
            print(f"  {fname}: {values}")

        # Acceptance criteria
        if real_pct >= 10.0:
            print(f"\n✓ PASSED: {real_pct:.1f}% of history features have non-default values")
            return 0
        else:
            print(f"\n❌ FAILED: Only {real_pct:.1f}% of history features populated (need ≥10%)")
            print("\nThis suggests channel history is not being tracked correctly.")
            print("Check that the scanner is using the new history-enabled extract_all_features().")
            return 1
    else:
        print("\n❌ Could not check feature values (no features found in samples)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
