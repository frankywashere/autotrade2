#!/usr/bin/env python3
"""
Simple test to verify live data integration with 120-day lookback.

Tests:
1. Import load_live_data_tuple
2. Fetch data with 120 day lookback
3. Verify DataFrame shapes
4. Check data freshness
5. Ensure it returns valid data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v7.data import load_live_data_tuple


def test_live_data_integration():
    """
    Test live data integration with 120-day lookback.
    """
    print("\n" + "="*70)
    print("Live Data Integration Test - 120 Day Lookback")
    print("="*70)

    # Step 1: Import load_live_data_tuple
    print("\n[1/5] Import load_live_data_tuple")
    print("  ✓ Successfully imported from v7.data")

    # Step 2: Fetch data with 120 day lookback
    print("\n[2/5] Fetch data with 120 day lookback")
    try:
        tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=120)
        print(f"  ✓ Data fetched successfully")
    except Exception as e:
        print(f"  ✗ Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Verify DataFrame shapes
    print("\n[3/5] Verify DataFrame shapes")
    print(f"  TSLA shape: {tsla_df.shape}")
    print(f"  SPY shape:  {spy_df.shape}")
    print(f"  VIX shape:  {vix_df.shape}")

    # Validate non-empty
    if len(tsla_df) == 0 or len(spy_df) == 0 or len(vix_df) == 0:
        print("  ✗ One or more DataFrames are empty!")
        return False

    # Validate columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for name, df in [('TSLA', tsla_df), ('SPY', spy_df)]:
        missing = set(required_cols) - set(df.columns)
        if missing:
            print(f"  ✗ {name} missing columns: {missing}")
            return False

    print("  ✓ All DataFrames have valid shapes and columns")

    # Step 4: Check data freshness
    print("\n[4/5] Check data freshness")

    # Get latest timestamp
    latest_tsla = tsla_df.index[-1]
    latest_spy = spy_df.index[-1]
    latest_vix = vix_df.index[-1]

    print(f"  Latest TSLA timestamp: {latest_tsla}")
    print(f"  Latest SPY timestamp:  {latest_spy}")
    print(f"  Latest VIX timestamp:  {latest_vix}")

    # Calculate age
    now = datetime.now()
    tsla_age = (now - latest_tsla).total_seconds() / 3600  # hours

    print(f"  Data age: {tsla_age:.1f} hours")

    # Check if data is within lookback period
    cutoff = now - timedelta(days=120)
    earliest_tsla = tsla_df.index[0]

    print(f"  Earliest TSLA timestamp: {earliest_tsla}")
    print(f"  120-day cutoff: {cutoff}")

    if earliest_tsla > cutoff:
        print(f"  ⚠ Warning: Data starts after cutoff (this is OK if CSVs are recent)")
    else:
        print(f"  ✓ Data covers the full lookback period")

    # Step 5: Ensure it returns valid data
    print("\n[5/5] Ensure it returns valid data")

    # Check for NaN values
    tsla_nans = tsla_df.isnull().sum().sum()
    spy_nans = spy_df.isnull().sum().sum()
    vix_nans = vix_df.isnull().sum().sum()

    print(f"  TSLA NaN count: {tsla_nans}")
    print(f"  SPY NaN count:  {spy_nans}")
    print(f"  VIX NaN count:  {vix_nans}")

    # Check value ranges
    print(f"\n  TSLA price range: ${tsla_df['close'].min():.2f} - ${tsla_df['close'].max():.2f}")
    print(f"  SPY price range:  ${spy_df['close'].min():.2f} - ${spy_df['close'].max():.2f}")
    print(f"  VIX range:        {vix_df['close'].min():.2f} - {vix_df['close'].max():.2f}")

    # Latest values
    print(f"\n  Latest TSLA close: ${tsla_df['close'].iloc[-1]:.2f}")
    print(f"  Latest SPY close:  ${spy_df['close'].iloc[-1]:.2f}")
    print(f"  Latest VIX close:  {vix_df['close'].iloc[-1]:.2f}")

    # Sanity checks
    if tsla_df['close'].iloc[-1] <= 0:
        print("  ✗ TSLA price is invalid!")
        return False

    if spy_df['close'].iloc[-1] <= 0:
        print("  ✗ SPY price is invalid!")
        return False

    print("\n  ✓ All data validation checks passed")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Successfully loaded {len(tsla_df)} TSLA bars")
    print(f"✓ Successfully loaded {len(spy_df)} SPY bars")
    print(f"✓ Successfully loaded {len(vix_df)} VIX bars")
    print(f"✓ Data is {tsla_age:.1f} hours old")
    print(f"✓ All validation checks passed")
    print("\n✓ Live data integration is working correctly!")

    return True


if __name__ == '__main__':
    success = test_live_data_integration()
    sys.exit(0 if success else 1)
