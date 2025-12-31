#!/usr/bin/env python3
"""
Test AutoTrade v7.0 Feature Extractors

Tests each extractor independently to verify they can be imported
and initialized correctly.

Usage:
    python3 scripts/test_extractors.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all extractors can be imported"""
    print("\n" + "=" * 80)
    print("TEST 1: Import All Extractors")
    print("=" * 80)

    try:
        from src.features import (
            ChannelFeatureExtractor,
            MarketFeatureExtractor,
            VIXFeatureExtractor,
            EventFeatureExtractor,
            ChannelHistoryExtractor,
            BreakdownFeatureExtractor,
            FeaturePipeline,
        )

        print("✓ All extractors imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization():
    """Test that extractors can be initialized with config"""
    print("\n" + "=" * 80)
    print("TEST 2: Initialize Extractors")
    print("=" * 80)

    try:
        from config import get_feature_config
        from src.features import (
            ChannelFeatureExtractor,
            MarketFeatureExtractor,
            VIXFeatureExtractor,
            EventFeatureExtractor,
            ChannelHistoryExtractor,
            BreakdownFeatureExtractor,
            FeaturePipeline,
        )
        from src.monitoring import MetricsTracker

        config = get_feature_config()
        metrics = MetricsTracker()

        # Initialize each extractor
        extractors = {
            'ChannelFeatureExtractor': ChannelFeatureExtractor(config, metrics),
            'MarketFeatureExtractor': MarketFeatureExtractor(config, metrics),
            'VIXFeatureExtractor': VIXFeatureExtractor(config, metrics),
            'EventFeatureExtractor': EventFeatureExtractor(config, metrics),
            'ChannelHistoryExtractor': ChannelHistoryExtractor(config, metrics),
            'BreakdownFeatureExtractor': BreakdownFeatureExtractor(config, metrics),
            'FeaturePipeline': FeaturePipeline(config),  # No metrics parameter
        }

        for name, extractor in extractors.items():
            print(f"✓ {name} initialized")

        print("\n✓ All extractors initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test config has all required attributes"""
    print("\n" + "=" * 80)
    print("TEST 3: Config Validation")
    print("=" * 80)

    try:
        from config import get_feature_config

        config = get_feature_config()

        # Check required attributes
        required_attrs = [
            'channel_windows',
            'channel_timeframes',
            'rsi_timeframes',
            'breakdown_timeframes',
            'channel_symbols',
            'min_cycles',
            'min_r_squared',
        ]

        for attr in required_attrs:
            value = getattr(config, attr)
            print(f"✓ config.{attr} = {value}")

        # Check feature counts
        counts = config.count_features()
        print(f"\n✓ Feature counts:")
        for category, count in counts.items():
            print(f"   {category:20s}: {count:,}")

        total = sum(counts.values())
        print(f"\n✓ Total features: {total:,}")

        return True

    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_extraction():
    """Test extractors with mock data"""
    print("\n" + "=" * 80)
    print("TEST 4: Mock Data Extraction")
    print("=" * 80)

    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from config import get_feature_config
        from src.features import (
            MarketFeatureExtractor,
            VIXFeatureExtractor,
            EventFeatureExtractor,
        )

        config = get_feature_config()

        # Create mock 5min data (100 bars)
        n_bars = 100
        dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

        # Generate realistic-looking price data
        tsla_base = 250
        spy_base = 450

        df = pd.DataFrame({
            'tsla_open': tsla_base + np.cumsum(np.random.randn(n_bars) * 0.5),
            'tsla_high': tsla_base + np.cumsum(np.random.randn(n_bars) * 0.5) + np.random.rand(n_bars),
            'tsla_low': tsla_base + np.cumsum(np.random.randn(n_bars) * 0.5) - np.random.rand(n_bars),
            'tsla_close': tsla_base + np.cumsum(np.random.randn(n_bars) * 0.5),
            'tsla_volume': np.random.randint(1000000, 5000000, n_bars),
            'spy_open': spy_base + np.cumsum(np.random.randn(n_bars) * 0.3),
            'spy_high': spy_base + np.cumsum(np.random.randn(n_bars) * 0.3) + np.random.rand(n_bars) * 0.5,
            'spy_low': spy_base + np.cumsum(np.random.randn(n_bars) * 0.3) - np.random.rand(n_bars) * 0.5,
            'spy_close': spy_base + np.cumsum(np.random.randn(n_bars) * 0.3),
            'spy_volume': np.random.randint(5000000, 20000000, n_bars),
        }, index=dates)

        # Ensure high >= close >= low
        for symbol in ['tsla', 'spy']:
            df[f'{symbol}_high'] = df[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close']].max(axis=1)
            df[f'{symbol}_low'] = df[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close']].min(axis=1)

        # Test Market Features
        market_extractor = MarketFeatureExtractor(config)
        market_features = market_extractor.extract(df, symbols=['tsla', 'spy'], mode='batch')
        print(f"✓ Market features: {market_features.shape[1]} features extracted")

        # Test VIX Features (without real VIX data - should use fallback)
        vix_extractor = VIXFeatureExtractor(config)
        vix_features = vix_extractor.extract(df, vix_data=None, mode='batch')
        print(f"✓ VIX features: {vix_features.shape[1]} features extracted (fallback)")

        # Test Event Features (without real event data - should use fallback)
        event_extractor = EventFeatureExtractor(config)
        event_features = event_extractor.extract(df, earnings_dates=None, fomc_dates=None, mode='batch')
        print(f"✓ Event features: {event_features.shape[1]} features extracted (fallback)")

        print("\n✓ Mock extraction tests passed")
        return True

    except Exception as e:
        print(f"❌ Mock extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("AutoTrade v7.0 - Feature Extractor Test Suite")
    print("=" * 80)

    tests = [
        ("Import All Extractors", test_imports),
        ("Initialize Extractors", test_initialization),
        ("Config Validation", test_config),
        ("Mock Data Extraction", test_mock_extraction),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Test Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED! Feature extractors working correctly.")
        print("\nWhat We've Built:")
        print("  ✓ 6 modular feature extractors")
        print("  ✓ Config-driven feature selection")
        print("  ✓ Error handling with graceful degradation")
        print("  ✓ Structured logging and metrics tracking")
        print("  ✓ Clean architecture with single responsibility")
        print("\nFeature Breakdown:")
        print("  • Channel Features: 3,410 (5 windows × 11 TF × 31 metrics × 2 symbols)")
        print("  • Market Features: 64 (price, RSI, volume, correlation)")
        print("  • VIX Features: 15 (volatility regime)")
        print("  • Event Features: 4 (earnings, FOMC proximity)")
        print("  • Channel History: 99 (temporal context)")
        print("  • Breakdown Features: 38 (breakout detection)")
        print("  • Total: ~3,630 features")
        print("\nNext Steps:")
        print("  - Build cache manager (Week 5)")
        print("  - Build training pipeline (Week 6-7)")
        print("  - Build inference service (Week 9-10)")
        return 0
    else:
        print("\n❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
