"""
Test script for dashboard.py live data integration.

This script verifies that the dashboard can successfully:
1. Import from v7.data.live
2. Load live data
3. Display data freshness status
"""

import sys
from pathlib import Path

# Add v7 to path
sys.path.insert(0, str(Path(__file__).parent / 'v7'))

from v7.data.live import fetch_live_data, is_market_open

def test_live_data_import():
    """Test that live data module can be imported."""
    print("Testing live data import...")
    try:
        from v7.data.live import fetch_live_data, LiveDataResult, is_market_open
        print("✓ Live data module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import live data module: {e}")
        return False


def test_fetch_live_data():
    """Test that live data can be fetched."""
    print("\nTesting live data fetch...")
    try:
        result = fetch_live_data(lookback_days=30)
        print(f"✓ Live data fetched successfully")
        print(f"  - TSLA: {len(result.tsla_df)} bars")
        print(f"  - SPY: {len(result.spy_df)} bars")
        print(f"  - VIX: {len(result.vix_df)} bars")
        print(f"  - Status: {result.status}")
        print(f"  - Data age: {result.data_age_minutes:.1f} minutes")
        return True
    except Exception as e:
        print(f"✗ Failed to fetch live data: {e}")
        return False


def test_market_open():
    """Test market open detection."""
    print("\nTesting market open detection...")
    try:
        market_open = is_market_open()
        print(f"✓ Market open check completed: {'OPEN' if market_open else 'CLOSED'}")
        return True
    except Exception as e:
        print(f"✗ Failed to check market status: {e}")
        return False


def test_dashboard_integration():
    """Test that dashboard.py has been updated correctly."""
    print("\nTesting dashboard.py integration...")
    try:
        # Read dashboard.py
        dashboard_path = Path(__file__).parent / 'dashboard.py'
        content = dashboard_path.read_text()

        # Check for required imports
        checks = [
            ('from v7.data.live import fetch_live_data', 'Live data import'),
            ('LiveDataResult', 'LiveDataResult type'),
            ('is_market_open', 'Market open function'),
            ('data_status', 'Data status field'),
            ('data_age_minutes', 'Data age field'),
            ('market_open', 'Market open field'),
        ]

        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"  ✓ {description} found")
            else:
                print(f"  ✗ {description} NOT found")
                all_passed = False

        if all_passed:
            print("✓ Dashboard integration looks good")
            return True
        else:
            print("✗ Dashboard integration incomplete")
            return False

    except Exception as e:
        print(f"✗ Failed to check dashboard integration: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Dashboard Live Data Integration Test")
    print("=" * 60)

    results = []
    results.append(test_live_data_import())
    results.append(test_fetch_live_data())
    results.append(test_market_open())
    results.append(test_dashboard_integration())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed! Dashboard is ready for live data.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        sys.exit(1)
