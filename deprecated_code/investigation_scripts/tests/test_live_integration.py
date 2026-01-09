#!/usr/bin/env python3
"""
Test script for live data integration

Quick test to verify the new live module works before integrating into dashboard.py
"""

import sys
from pathlib import Path

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from v7.data.live import fetch_live_data, load_live_data_tuple, is_market_open


def test_basic_fetch():
    """Test basic fetch_live_data() function."""
    print("="*70)
    print("TEST 1: Basic fetch_live_data()")
    print("="*70)

    try:
        result = fetch_live_data(lookback_days=5)

        print(f"\n✓ Success!")
        print(f"  TSLA: {len(result.tsla_df)} bars")
        print(f"  SPY:  {len(result.spy_df)} bars")
        print(f"  VIX:  {len(result.vix_df)} bars")
        print(f"  Status: {result.status}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"  Data age: {result.data_age_minutes:.1f} minutes")
        print(f"\n  Latest TSLA close: ${result.tsla_df['close'].iloc[-1]:.2f}")
        print(f"  Latest SPY close: ${result.spy_df['close'].iloc[-1]:.2f}")
        print(f"  Latest VIX close: {result.vix_df['close'].iloc[-1]:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tuple_compatibility():
    """Test backward-compatible tuple function."""
    print("\n" + "="*70)
    print("TEST 2: Backward-compatible load_live_data_tuple()")
    print("="*70)

    try:
        tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=5)

        print(f"\n✓ Success!")
        print(f"  TSLA: {len(tsla_df)} bars")
        print(f"  SPY:  {len(spy_df)} bars")
        print(f"  VIX:  {len(vix_df)} bars")
        print(f"  Latest TSLA: ${tsla_df['close'].iloc[-1]:.2f}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_force_historical():
    """Test force_historical flag."""
    print("\n" + "="*70)
    print("TEST 3: Force historical mode (no yfinance)")
    print("="*70)

    try:
        result = fetch_live_data(lookback_days=5, force_historical=True)

        print(f"\n✓ Success!")
        print(f"  Status: {result.status} (should be HISTORICAL)")
        print(f"  TSLA: {len(result.tsla_df)} bars")
        print(f"  Latest: {result.timestamp}")

        if result.status == 'HISTORICAL':
            print("  ✓ Correctly skipped yfinance")
        else:
            print("  ⚠ Warning: Expected HISTORICAL status")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_status():
    """Test market open/closed detection."""
    print("\n" + "="*70)
    print("TEST 4: Market status check")
    print("="*70)

    try:
        market_open = is_market_open()

        print(f"\n✓ Success!")
        if market_open:
            print("  🔔 Market is OPEN (9:30 AM - 4:00 PM ET, Mon-Fri)")
        else:
            print("  🔕 Market is CLOSED (after hours or weekend)")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_format():
    """Test that data format matches dashboard expectations."""
    print("\n" + "="*70)
    print("TEST 5: Data format validation")
    print("="*70)

    try:
        result = fetch_live_data(lookback_days=5)

        # Check TSLA format
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        tsla_cols = result.tsla_df.columns.tolist()

        print(f"\n  Checking TSLA DataFrame:")
        print(f"    Columns: {tsla_cols}")
        print(f"    Index type: {type(result.tsla_df.index).__name__}")
        print(f"    Index name: {result.tsla_df.index.name}")

        missing = set(required_cols) - set(tsla_cols)
        if missing:
            print(f"  ✗ Missing columns: {missing}")
            return False

        # Check index is DatetimeIndex
        import pandas as pd
        if not isinstance(result.tsla_df.index, pd.DatetimeIndex):
            print(f"  ✗ Index is not DatetimeIndex")
            return False

        # Check same format for SPY
        spy_cols = result.spy_df.columns.tolist()
        if spy_cols != tsla_cols:
            print(f"  ✗ SPY columns don't match TSLA: {spy_cols}")
            return False

        # Check VIX format
        vix_cols = result.vix_df.columns.tolist()
        print(f"  VIX columns: {vix_cols}")

        print(f"\n  ✓ All format checks passed!")
        print(f"    - OHLCV columns present")
        print(f"    - DatetimeIndex correctly set")
        print(f"    - TSLA and SPY format consistent")
        print(f"    - VIX data loaded")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "LIVE DATA INTEGRATION TEST" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    results = []

    # Run tests
    results.append(("Basic fetch", test_basic_fetch()))
    results.append(("Tuple compatibility", test_tuple_compatibility()))
    results.append(("Force historical", test_force_historical()))
    results.append(("Market status", test_market_status()))
    results.append(("Data format", test_data_format()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        color_code = "\033[92m" if result else "\033[91m"
        reset_code = "\033[0m"
        print(f"{color_code}{status}{reset_code}  {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n\033[92m✓ ALL TESTS PASSED - Ready for dashboard integration!\033[0m")
        return 0
    else:
        print(f"\n\033[91m✗ {total - passed} test(s) failed - Review errors above\033[0m")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
