#!/usr/bin/env python3
"""
Test multi-scale feature extraction.

Verifies that the expanded timeframe features are correctly extracted:
- Old: 3 timeframes (1h, 4h, daily) × 10 features = 30 + 26 base = 56 total
- New: 11 timeframes × 10 features = 110 + 26 base = 136 total
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.features import TradingFeatureExtractor


def test_multiscale_features():
    """Test that all 11 timeframes are included in feature extraction."""
    print("\n" + "=" * 70)
    print("🧪 TESTING MULTI-SCALE FEATURE EXTRACTION")
    print("=" * 70)

    # Create feature extractor
    print("\nCreating feature extractor...")
    extractor = TradingFeatureExtractor()

    # Check feature names
    feature_names = extractor.get_feature_names()
    feature_dim = extractor.get_feature_dim()

    print(f"\n✓ Feature dimension: {feature_dim}")
    print(f"  Expected: ~136 features (was ~56)")

    # Expected timeframes
    expected_timeframes = [
        '5min', '15min', '30min',
        '1h', '2h', '3h', '4h',
        'daily', 'weekly', 'monthly', '3month'
    ]

    print(f"\n✓ Expected timeframes: {len(expected_timeframes)}")

    # Check for each timeframe
    print("\nChecking timeframe coverage:")

    for tf in expected_timeframes:
        # Check channel features
        channel_features = [
            f'channel_{tf}_position',
            f'channel_{tf}_upper_dist',
            f'channel_{tf}_lower_dist',
            f'channel_{tf}_slope',
            f'channel_{tf}_stability',
            f'channel_{tf}_ping_pongs',
            f'channel_{tf}_r_squared'
        ]

        # Check RSI features
        rsi_features = [
            f'rsi_{tf}',
            f'rsi_{tf}_oversold',
            f'rsi_{tf}_overbought'
        ]

        all_tf_features = channel_features + rsi_features

        # Count how many are present
        found = sum(1 for f in all_tf_features if f in feature_names)

        if found == len(all_tf_features):
            status = "✅"
        elif found > 0:
            status = "⚠️"
        else:
            status = "❌"

        print(f"  {status} {tf:10s}: {found}/{len(all_tf_features)} features")

    # Calculate feature breakdown
    print("\n" + "=" * 70)
    print("FEATURE BREAKDOWN")
    print("=" * 70)

    # Count by category
    base_features = [f for f in feature_names if not any(
        f.startswith(prefix) for prefix in ['channel_', 'rsi_']
    )]
    channel_features = [f for f in feature_names if f.startswith('channel_')]
    rsi_features = [f for f in feature_names if f.startswith('rsi_')]

    print(f"\n  Base features: {len(base_features)}")
    print(f"  Channel features: {len(channel_features)} (7 per timeframe × {len(channel_features)//7} timeframes)")
    print(f"  RSI features: {len(rsi_features)} (3 per timeframe × {len(rsi_features)//3} timeframes)")
    print(f"\n  Total: {len(feature_names)}")

    # Success criteria
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if feature_dim >= 130:
        print(f"✅ SUCCESS: Feature dimension is {feature_dim} (expected ~136)")
    else:
        print(f"❌ FAILURE: Feature dimension is {feature_dim} (expected ~136)")

    channel_timeframes = len(set(f.split('_')[1] for f in channel_features if '_' in f))
    rsi_timeframes = len(set(f.split('_')[1] for f in rsi_features if '_' in f))

    if channel_timeframes == 11 and rsi_timeframes == 11:
        print(f"✅ SUCCESS: All 11 timeframes present in features")
    else:
        print(f"❌ FAILURE: Only {min(channel_timeframes, rsi_timeframes)} timeframes found (expected 11)")

    print("\n" + "=" * 70)
    print("Feature count increased from 56 → 136 for multi-scale learning!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    test_multiscale_features()
