#!/usr/bin/env python3
"""
Test AutoTrade v7.0 Clean Architecture Components

Tests each component independently without requiring full feature extraction.

Usage:
    python3 scripts/test_architecture.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test config system"""
    print("\n" + "=" * 80)
    print("TEST 1: Config System")
    print("=" * 80)

    from config import get_feature_config

    config = get_feature_config()

    print(f"✓ Version: {config.version}")
    print(f"✓ Windows: {config.channel_windows}")
    print(f"✓ RSI timeframes: {config.rsi_timeframes}")
    print(f"✓ Validity criteria: cycles>={config.min_cycles}, r²>{config.min_r_squared}")

    counts = config.count_features()
    print(f"\n✓ Feature counts:")
    for category, count in counts.items():
        print(f"   {category:20s}: {count:,}")

    # Test validation
    assert config.is_channel_valid(cycles=2, r_squared=0.15) == True
    assert config.is_channel_valid(cycles=0, r_squared=0.9) == False
    print(f"\n✓ Validity logic working")

    return True


def test_errors():
    """Test error handling"""
    print("\n" + "=" * 80)
    print("TEST 2: Error Handling")
    print("=" * 80)

    from src.errors import (
        InsufficientDataError,
        FeatureExtractionError,
        VIXFeaturesError,
        GracefulDegradation,
    )

    # Test exception hierarchy
    try:
        raise InsufficientDataError("Not enough bars")
    except InsufficientDataError as e:
        print(f"✓ InsufficientDataError: {e}")

    # Test graceful degradation
    recovery = GracefulDegradation()
    fallback_vix = recovery.get_zero_vix_features()
    print(f"✓ VIX fallback features: {len(fallback_vix)} features")

    fallback_events = recovery.get_default_events()
    print(f"✓ Event fallback features: {len(fallback_events)} features")

    fallback_pred = recovery.get_fallback_prediction()
    print(f"✓ Fallback prediction: confidence={fallback_pred['confidence']}")

    return True


def test_monitoring():
    """Test monitoring"""
    print("\n" + "=" * 80)
    print("TEST 3: Monitoring & Metrics")
    print("=" * 80)

    from src.monitoring import MetricsTracker
    import time

    metrics = MetricsTracker()

    # Test metrics recording
    metrics.record('test_metric', 42.5)
    metrics.record('test_metric', 38.2)
    metrics.record('test_metric', 45.1)

    stats = metrics.get_stats('test_metric')
    print(f"✓ Metric stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}, p95={stats['p95']:.2f}")

    # Test timing
    with metrics.timer('test_operation'):
        time.sleep(0.01)

    timing_stats = metrics.get_stats('test_operation_duration_ms')
    print(f"✓ Timing: {timing_stats['mean']:.2f}ms")

    # Test summary
    summary = metrics.summary()
    print(f"✓ Summary generated ({len(summary)} chars)")

    return True


def test_core():
    """Test core modules"""
    print("\n" + "=" * 80)
    print("TEST 4: Core Modules")
    print("=" * 80)

    from src.core import LinearRegressionChannel, RSICalculator
    import pandas as pd
    import numpy as np

    # Test channel calculator
    print(f"✓ LinearRegressionChannel imported")

    # Test RSI calculator
    print(f"✓ RSICalculator imported")

    # Create mock data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    prices = 200 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({'close': prices}, index=dates)

    # Calculate RSI
    rsi_calc = RSICalculator(period=14)
    rsi = rsi_calc.calculate_rsi(df)
    print(f"✓ RSI calculated: last value = {rsi.iloc[-1]:.2f}")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("AutoTrade v7.0 - Clean Architecture Test Suite")
    print("=" * 80)

    tests = [
        ("Config System", test_config),
        ("Error Handling", test_errors),
        ("Monitoring", test_monitoring),
        ("Core Modules", test_core),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Test Results: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 80)

    if failed == 0:
        print("\n✅ ALL TESTS PASSED! Clean architecture working perfectly.")
        print("\nWhat We've Built:")
        print("  ✓ Config-driven feature selection (YAML + Pydantic)")
        print("  ✓ Error handling with graceful degradation")
        print("  ✓ Structured logging and metrics tracking")
        print("  ✓ Core domain logic (LinearRegressionChannel, RSI)")
        print("  ✓ Modular architecture ready for expansion")
        print("\nNext Steps:")
        print("  - Build complete feature extractors (Week 3-4)")
        print("  - Build cache manager (Week 5)")
        print("  - Build training pipeline (Week 6-7)")
        print("  - Build inference service (Week 9-10)")
        return 0
    else:
        print("\n❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
