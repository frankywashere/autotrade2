#!/usr/bin/env python3
"""
Standalone test runner for optimization correctness tests.

Runs all tests without requiring pytest.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import the test module
from test_optimization_correctness import (
    random_price_data,
    sample_market_data,
    TestRSIOptimization,
    TestChannelCaching,
    TestResamplingCache,
    TestFullFeatureExtraction,
    TestLabelGeneration,
    TestPerformanceBenchmarks,
    TestEndToEndCorrectness,
    generate_performance_report,
)

import numpy as np
import pandas as pd


def run_test_class(test_class, fixtures):
    """Run all test methods in a class."""
    class_name = test_class.__name__
    print(f"\n{'='*80}")
    print(f"Running {class_name}")
    print('='*80)

    instance = test_class()
    test_methods = [m for m in dir(instance) if m.startswith('test_')]

    passed = 0
    failed = 0
    errors = []

    for method_name in test_methods:
        method = getattr(instance, method_name)
        test_name = f"{class_name}::{method_name}"

        try:
            # Call method with appropriate fixtures
            if 'random_price_data' in method.__code__.co_varnames:
                method(fixtures['random_price_data'])
            elif 'sample_market_data' in method.__code__.co_varnames:
                method(fixtures['sample_market_data'])
            else:
                method()

            print(f"  ✓ {method_name}")
            passed += 1

        except Exception as e:
            print(f"  ✗ {method_name}")
            print(f"     Error: {str(e)[:200]}")
            failed += 1
            errors.append((test_name, e))

    return passed, failed, errors


def generate_fixtures():
    """Generate test fixtures."""
    # Random price data
    np.random.seed(42)
    n_bars = 1000

    close_prices = 250.0 + np.cumsum(np.random.randn(n_bars) * 2)
    high_prices = close_prices + np.abs(np.random.randn(n_bars) * 1.5)
    low_prices = close_prices - np.abs(np.random.randn(n_bars) * 1.5)
    open_prices = close_prices + np.random.randn(n_bars) * 0.5
    volume = np.random.randint(100000, 1000000, n_bars)

    dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

    random_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    # Sample market data
    np.random.seed(123)
    n_bars = 500

    dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='5min')

    tsla_close = 250.0 + np.cumsum(np.random.randn(n_bars) * 2)
    tsla_df = pd.DataFrame({
        'open': tsla_close + np.random.randn(n_bars) * 0.5,
        'high': tsla_close + np.abs(np.random.randn(n_bars) * 1.5),
        'low': tsla_close - np.abs(np.random.randn(n_bars) * 1.5),
        'close': tsla_close,
        'volume': np.random.randint(100000, 1000000, n_bars)
    }, index=dates)

    spy_close = 450.0 + np.cumsum(np.random.randn(n_bars) * 1)
    spy_df = pd.DataFrame({
        'open': spy_close + np.random.randn(n_bars) * 0.3,
        'high': spy_close + np.abs(np.random.randn(n_bars) * 0.8),
        'low': spy_close - np.abs(np.random.randn(n_bars) * 0.8),
        'close': spy_close,
        'volume': np.random.randint(500000, 2000000, n_bars)
    }, index=dates)

    n_days = n_bars // 78 + 1
    vix_dates = pd.date_range('2024-01-01', periods=n_days, freq='1D')
    vix_df = pd.DataFrame({
        'open': 15.0 + np.random.randn(n_days) * 2,
        'high': 16.0 + np.random.randn(n_days) * 2,
        'low': 14.0 + np.random.randn(n_days) * 2,
        'close': 15.0 + np.random.randn(n_days) * 2,
    }, index=vix_dates)

    return {
        'random_price_data': random_df,
        'sample_market_data': (tsla_df, spy_df, vix_df)
    }


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("OPTIMIZATION CORRECTNESS TEST SUITE")
    print("="*80)

    # Generate fixtures
    print("\nGenerating test fixtures...")
    fixtures = generate_fixtures()
    print("✓ Fixtures ready")

    # Test classes to run
    test_classes = [
        TestRSIOptimization,
        TestChannelCaching,
        TestResamplingCache,
        TestFullFeatureExtraction,
        TestLabelGeneration,
        TestPerformanceBenchmarks,
        TestEndToEndCorrectness,
    ]

    total_passed = 0
    total_failed = 0
    all_errors = []

    # Run each test class
    for test_class in test_classes:
        passed, failed, errors = run_test_class(test_class, fixtures)
        total_passed += passed
        total_failed += failed
        all_errors.extend(errors)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\nTotal tests run: {total_passed + total_failed}")
    print(f"  ✓ Passed: {total_passed}")
    print(f"  ✗ Failed: {total_failed}")

    if total_failed > 0:
        print("\nFailed tests:")
        for test_name, error in all_errors:
            print(f"  - {test_name}")
            print(f"    {str(error)[:200]}")

    # Generate performance report
    print("\n" + "="*80)
    print("GENERATING PERFORMANCE REPORT")
    print("="*80)
    generate_performance_report()

    # Exit code
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == '__main__':
    main()
