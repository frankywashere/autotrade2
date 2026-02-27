"""
Test VIX Fetcher

This script tests the VIX fetcher with all three data sources:
1. FRED API
2. yfinance
3. Local CSV

Run this to verify the VIX fetching system works correctly.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v7.data.vix_fetcher import FREDVixFetcher, fetch_vix_data
import pandas as pd


def test_fred_api():
    """Test fetching VIX from FRED API."""
    print("=" * 80)
    print("TEST 1: FRED API")
    print("=" * 80)

    # Try to get API key from environment
    fred_key = os.getenv('FRED_API_KEY')

    if not fred_key:
        print("SKIPPED: No FRED_API_KEY found in environment")
        print("To test FRED API, set FRED_API_KEY environment variable:")
        print("  export FRED_API_KEY='your_api_key_here'")
        print("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return False

    try:
        fetcher = FREDVixFetcher(fred_api_key=fred_key)
        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31")

        print(f"\nSuccess! Fetched {len(vix_df)} records from FRED")
        print(f"\nDate range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
        print(f"\nFirst 5 rows:")
        print(vix_df.head())
        print(f"\nLast 5 rows:")
        print(vix_df.tail())

        # Check source info
        source_info = fetcher.get_source_info()
        if source_info:
            print(f"\nSource: {source_info.source}")
            print(f"Has gaps: {source_info.has_gaps}")

        # Validate data
        assert len(vix_df) > 0, "No data returned"
        assert all(col in vix_df.columns for col in ['open', 'high', 'low', 'close']), "Missing columns"
        assert (vix_df >= 0).all().all(), "Negative VIX values found"

        print("\nPASSED: FRED API test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def test_yfinance():
    """Test fetching VIX from yfinance."""
    print("\n" + "=" * 80)
    print("TEST 2: yfinance")
    print("=" * 80)

    try:
        # Use fetcher without FRED API key to force yfinance
        fetcher = FREDVixFetcher(fred_api_key=None)
        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31")

        print(f"\nSuccess! Fetched {len(vix_df)} records from yfinance")
        print(f"\nDate range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
        print(f"\nFirst 5 rows:")
        print(vix_df.head())
        print(f"\nData summary:")
        print(vix_df.describe())

        # Check source info
        source_info = fetcher.get_source_info()
        if source_info:
            print(f"\nSource: {source_info.source}")
            print(f"Has gaps before forward-fill: {source_info.has_gaps}")

        # Validate data
        assert len(vix_df) > 0, "No data returned"
        assert all(col in vix_df.columns for col in ['open', 'high', 'low', 'close']), "Missing columns"
        assert (vix_df >= 0).all().all(), "Negative VIX values found"
        assert (vix_df['high'] >= vix_df['low']).all(), "High < Low found"

        print("\nPASSED: yfinance test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def test_csv():
    """Test loading VIX from local CSV."""
    print("\n" + "=" * 80)
    print("TEST 3: Local CSV")
    print("=" * 80)

    # Find CSV file
    csv_path = Path(__file__).parent.parent.parent / "data" / "VIX_History.csv"

    if not csv_path.exists():
        print(f"SKIPPED: CSV file not found at {csv_path}")
        return False

    try:
        fetcher = FREDVixFetcher(
            fred_api_key=None,  # Force skip FRED
            csv_path=str(csv_path)
        )

        # Temporarily disable yfinance by simulating failure
        original_fetch = fetcher._fetch_from_yfinance
        fetcher._fetch_from_yfinance = lambda *args, **kwargs: None

        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31")

        # Restore
        fetcher._fetch_from_yfinance = original_fetch

        print(f"\nSuccess! Loaded {len(vix_df)} records from CSV")
        print(f"\nDate range: {vix_df.index.min().date()} to {vix_df.index.max().date()}")
        print(f"\nFirst 5 rows:")
        print(vix_df.head())

        # Check source info
        source_info = fetcher.get_source_info()
        if source_info:
            print(f"\nSource: {source_info.source}")

        # Validate data
        assert len(vix_df) > 0, "No data returned"
        assert all(col in vix_df.columns for col in ['open', 'high', 'low', 'close']), "Missing columns"

        print("\nPASSED: CSV test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_fill():
    """Test forward-fill logic for missing dates."""
    print("\n" + "=" * 80)
    print("TEST 4: Forward-Fill Logic")
    print("=" * 80)

    try:
        # Fetch data with forward fill
        vix_df_filled = fetch_vix_data(
            start_date="2023-01-01",
            end_date="2023-01-31",
            forward_fill=True
        )

        # Count expected days (should be 31 for January)
        expected_days = 31
        actual_days = len(vix_df_filled)

        print(f"\nForward-fill enabled:")
        print(f"  Expected days: {expected_days}")
        print(f"  Actual days: {actual_days}")
        print(f"  Date range: {vix_df_filled.index.min().date()} to {vix_df_filled.index.max().date()}")

        # Check for gaps
        date_diffs = vix_df_filled.index.to_series().diff()
        max_gap = date_diffs.max()
        print(f"  Maximum gap: {max_gap}")

        assert actual_days == expected_days, f"Expected {expected_days} days, got {actual_days}"
        assert max_gap <= pd.Timedelta(days=1), f"Found gap larger than 1 day: {max_gap}"

        print("\nPASSED: Forward-fill test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def test_fallback_chain():
    """Test the fallback chain works correctly."""
    print("\n" + "=" * 80)
    print("TEST 5: Fallback Chain")
    print("=" * 80)

    try:
        # Test with invalid FRED key to trigger fallback
        fetcher = FREDVixFetcher(
            fred_api_key="invalid_key_12345",  # Invalid key
            csv_path=str(Path(__file__).parent.parent.parent / "data" / "VIX_History.csv")
        )

        print("\nAttempting fetch with invalid FRED key (should fall back to yfinance)...")
        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-10")

        source_info = fetcher.get_source_info()
        print(f"\nData source used: {source_info.source}")
        print(f"Records fetched: {len(vix_df)}")

        # Should have fallen back to yfinance or CSV
        assert source_info.source in ['yfinance', 'csv'], f"Expected yfinance or csv, got {source_info.source}"
        assert len(vix_df) > 0, "No data returned"

        print("\nPASSED: Fallback chain test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def test_data_validation():
    """Test data validation catches errors."""
    print("\n" + "=" * 80)
    print("TEST 6: Data Validation")
    print("=" * 80)

    try:
        fetcher = FREDVixFetcher()
        vix_df = fetcher.fetch(start_date="2023-01-01", end_date="2023-01-31")

        print("\nValidating VIX data...")

        # Check 1: No negative values
        assert (vix_df >= 0).all().all(), "FAILED: Found negative values"
        print("  ✓ No negative values")

        # Check 2: High >= Low
        assert (vix_df['high'] >= vix_df['low']).all(), "FAILED: Found high < low"
        print("  ✓ High >= Low")

        # Check 3: Close within [low, high]
        assert ((vix_df['close'] >= vix_df['low']) & (vix_df['close'] <= vix_df['high'])).all(), \
            "FAILED: Close outside [low, high]"
        print("  ✓ Close within [low, high]")

        # Check 4: No extreme values
        assert (vix_df['close'] < 200).all(), "FAILED: Extreme values (>200)"
        print("  ✓ No extreme values")

        # Check 5: Index is DatetimeIndex
        assert isinstance(vix_df.index, pd.DatetimeIndex), "FAILED: Index is not DatetimeIndex"
        print("  ✓ Index is DatetimeIndex")

        # Check 6: Sorted by date
        assert vix_df.index.is_monotonic_increasing, "FAILED: Index not sorted"
        print("  ✓ Sorted by date")

        print("\nPASSED: Data validation test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "=" * 80)
    print("TEST 7: Convenience Function")
    print("=" * 80)

    try:
        # Test simple usage
        vix_df = fetch_vix_data(
            start_date="2023-01-01",
            end_date="2023-01-10"
        )

        print(f"\nFetched {len(vix_df)} records using convenience function")
        print(f"\nFirst 3 rows:")
        print(vix_df.head(3))

        assert len(vix_df) > 0, "No data returned"
        assert isinstance(vix_df, pd.DataFrame), "Return type is not DataFrame"

        print("\nPASSED: Convenience function test")
        return True

    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VIX FETCHER TEST SUITE")
    print("=" * 80)

    tests = [
        ("FRED API", test_fred_api),
        ("yfinance", test_yfinance),
        ("Local CSV", test_csv),
        ("Forward-Fill", test_forward_fill),
        ("Fallback Chain", test_fallback_chain),
        ("Data Validation", test_data_validation),
        ("Convenience Function", test_convenience_function),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nTest {name} crashed: {str(e)}")
            results[name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED/SKIPPED"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests PASSED!")
    else:
        print(f"\n{total - passed} test(s) failed or skipped")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
