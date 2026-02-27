#!/usr/bin/env python3
"""
Validate that history feature values are within expected ranges.

This test ensures the channel history features contain sensible values,
not garbage or obviously wrong numbers.

Usage:
    python3 tests/test_history_values.py /tmp/history_test_samples.bin
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_samples import load_samples
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_history_values.py <samples.bin>")
        return 1

    samples_path = sys.argv[1]

    if not os.path.exists(samples_path):
        print(f"❌ Samples file not found: {samples_path}")
        return 1

    # Load samples
    print(f"Loading samples from {samples_path}...")
    version, num_samples, num_features, samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("❌ No samples loaded")
        return 1

    # Get feature names
    feature_names = list(samples[0].features.keys())

    # Find specific history features by type
    duration_features = [f for f in feature_names if 'last5_avg_duration' in f]
    slope_features = [f for f in feature_names if 'last5_avg_slope' in f]
    r2_features = [f for f in feature_names if 'last5_avg_quality' in f]
    momentum_features = [f for f in feature_names if 'channel_momentum' in f]

    print(f"\nFound feature counts:")
    print(f"  Duration features: {len(duration_features)}")
    print(f"  Slope features: {len(slope_features)}")
    print(f"  Quality/R² features: {len(r2_features)}")
    print(f"  Momentum features: {len(momentum_features)}")

    errors = []

    # Check duration values (should be 10-500 bars typically)
    durations = []
    for sample in samples:
        for feat in duration_features:
            val = sample.features.get(feat, 0.0)
            if val not in (0.0, 50.0):  # Not default
                durations.append(val)

    if durations:
        print(f"\nDuration features (n={len(durations)} non-default):")
        print(f"  Min: {min(durations):.1f}")
        print(f"  Max: {max(durations):.1f}")
        print(f"  Mean: {np.mean(durations):.1f}")

        if min(durations) < 0:
            errors.append(f"Duration has negative values: min={min(durations)}")
        if max(durations) > 10000:
            errors.append(f"Duration suspiciously high: max={max(durations)}")
    else:
        print("\n⚠️  No non-default duration values found")

    # Check slope values (should be small, typically -1 to 1 per bar)
    slopes = []
    for sample in samples:
        for feat in slope_features:
            val = sample.features.get(feat, 0.0)
            if val != 0.0:
                slopes.append(val)

    if slopes:
        print(f"\nSlope features (n={len(slopes)} non-zero):")
        print(f"  Min: {min(slopes):.6f}")
        print(f"  Max: {max(slopes):.6f}")
        print(f"  Mean: {np.mean(slopes):.6f}")

        if abs(min(slopes)) > 100 or abs(max(slopes)) > 100:
            errors.append(f"Slope values suspiciously large: [{min(slopes)}, {max(slopes)}]")
    else:
        print("\n⚠️  No non-zero slope values found")

    # Check R²/quality values (should be 0.0-1.0)
    r2_values = []
    for sample in samples:
        for feat in r2_features:
            val = sample.features.get(feat, 0.0)
            if val not in (0.0, 2.0):  # Not default
                r2_values.append(val)

    if r2_values:
        print(f"\nQuality/R² features (n={len(r2_values)} non-default):")
        print(f"  Min: {min(r2_values):.3f}")
        print(f"  Max: {max(r2_values):.3f}")
        print(f"  Mean: {np.mean(r2_values):.3f}")

        if min(r2_values) < -0.1:
            errors.append(f"R² has invalid negative values: min={min(r2_values)}")
        if max(r2_values) > 1.1:
            errors.append(f"R² exceeds valid range: max={max(r2_values)}")
    else:
        print("\n⚠️  No non-default quality/R² values found")

    # Check momentum values
    momentum_vals = []
    for sample in samples:
        for feat in momentum_features:
            val = sample.features.get(feat, 0.0)
            if val != 0.0:
                momentum_vals.append(val)

    if momentum_vals:
        print(f"\nMomentum features (n={len(momentum_vals)} non-zero):")
        print(f"  Min: {min(momentum_vals):.6f}")
        print(f"  Max: {max(momentum_vals):.6f}")
        print(f"  Mean: {np.mean(momentum_vals):.6f}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("❌ VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("✓ All feature values are within expected ranges")
        return 0


if __name__ == "__main__":
    sys.exit(main())
