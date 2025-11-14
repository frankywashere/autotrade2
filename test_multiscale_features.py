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

    # Check for each timeframe (NOW CHECKS BOTH TSLA AND SPY)
    print("\nChecking timeframe coverage:")

    for tf in expected_timeframes:
        # Check TSLA and SPY features
        all_tf_features = []

        for symbol in ['tsla', 'spy']:
            # Channel features
            all_tf_features.extend([
                f'{symbol}_channel_{tf}_position',
                f'{symbol}_channel_{tf}_upper_dist',
                f'{symbol}_channel_{tf}_lower_dist',
                f'{symbol}_channel_{tf}_slope',
                f'{symbol}_channel_{tf}_stability',
                f'{symbol}_channel_{tf}_ping_pongs',
                f'{symbol}_channel_{tf}_r_squared'
            ])

            # RSI features
            all_tf_features.extend([
                f'{symbol}_rsi_{tf}',
                f'{symbol}_rsi_{tf}_oversold',
                f'{symbol}_rsi_{tf}_overbought'
            ])

        # Count how many are present
        found = sum(1 for f in all_tf_features if f in feature_names)

        if found == len(all_tf_features):
            status = "✅"
        elif found > 0:
            status = "⚠️"
        else:
            status = "❌"

        print(f"  {status} {tf:10s}: {found}/{len(all_tf_features)} features (TSLA+SPY)")

    # Calculate feature breakdown
    print("\n" + "=" * 70)
    print("FEATURE BREAKDOWN (v3.4 - WITH SPY FEATURES)")
    print("=" * 70)

    # Count by category (new naming with symbol prefixes)
    base_features = [f for f in feature_names if not any(
        f.startswith(prefix) for prefix in ['tsla_channel_', 'spy_channel_', 'tsla_rsi_', 'spy_rsi_']
    )]
    tsla_channel_features = [f for f in feature_names if f.startswith('tsla_channel_')]
    spy_channel_features = [f for f in feature_names if f.startswith('spy_channel_')]
    tsla_rsi_features = [f for f in feature_names if f.startswith('tsla_rsi_')]
    spy_rsi_features = [f for f in feature_names if f.startswith('spy_rsi_')]

    print(f"\n  Base features: {len(base_features)}")
    print(f"  TSLA Channel features: {len(tsla_channel_features)} (7 per timeframe × {len(tsla_channel_features)//7} timeframes)")
    print(f"  SPY Channel features: {len(spy_channel_features)} (7 per timeframe × {len(spy_channel_features)//7} timeframes)")
    print(f"  TSLA RSI features: {len(tsla_rsi_features)} (3 per timeframe × {len(tsla_rsi_features)//3} timeframes)")
    print(f"  SPY RSI features: {len(spy_rsi_features)} (3 per timeframe × {len(spy_rsi_features)//3} timeframes)")
    print(f"\n  Total: {len(feature_names)} (expected: 245)")

    # Success criteria (v3.4 - updated for 245 features)
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    if feature_dim >= 240:
        print(f"✅ SUCCESS: Feature dimension is {feature_dim} (expected 245)")
    else:
        print(f"❌ FAILURE: Feature dimension is {feature_dim} (expected 245, got {feature_dim})")

    # Check TSLA and SPY channel/RSI timeframes separately
    tsla_channel_timeframes = len(set(f.split('_')[2] for f in tsla_channel_features if f.count('_') >= 2))
    spy_channel_timeframes = len(set(f.split('_')[2] for f in spy_channel_features if f.count('_') >= 2))
    tsla_rsi_timeframes = len(set(f.split('_')[2] for f in tsla_rsi_features if f.count('_') >= 2))
    spy_rsi_timeframes = len(set(f.split('_')[2] for f in spy_rsi_features if f.count('_') >= 2))

    # Check that both TSLA and SPY have all 11 timeframes
    all_timeframes_ok = (tsla_channel_timeframes == 11 and spy_channel_timeframes == 11 and
                        tsla_rsi_timeframes == 11 and spy_rsi_timeframes == 11)

    if all_timeframes_ok:
        print(f"✅ SUCCESS: All 11 timeframes present for both TSLA and SPY")
    else:
        print(f"❌ FAILURE: Timeframe coverage incomplete")
        print(f"   TSLA channels: {tsla_channel_timeframes}/11, SPY channels: {spy_channel_timeframes}/11")
        print(f"   TSLA RSI: {tsla_rsi_timeframes}/11, SPY RSI: {spy_rsi_timeframes}/11")

    print("\n" + "=" * 70)
    print("Feature count progression:")
    print("  v1.0: 56 features (basic)")
    print("  v2.0-v3.3: 135 features (multi-scale TSLA only)")
    print("  v3.4: 245 features (multi-scale TSLA + SPY) ✨")
    print("=" * 70)
    print()


if __name__ == '__main__':
    test_multiscale_features()
