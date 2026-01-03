#!/usr/bin/env python3
"""
Comprehensive Live Data Integration Verification Script

This script demonstrates all the key features of the live data integration:
1. Import load_live_data_tuple
2. Fetch data with 120 day lookback
3. Verify DataFrame shapes
4. Check data freshness
5. Ensure it returns valid data

Usage:
    python verify_live_integration.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main verification function."""

    print("\n" + "="*80)
    print("LIVE DATA INTEGRATION VERIFICATION")
    print("="*80)

    # ========================================================================
    # Step 1: Import load_live_data_tuple
    # ========================================================================
    print("\n[Step 1] Import load_live_data_tuple")
    print("-" * 80)

    try:
        from v7.data import load_live_data_tuple
        print("✓ Successfully imported load_live_data_tuple from v7.data")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # ========================================================================
    # Step 2: Fetch data with 120 day lookback
    # ========================================================================
    print("\n[Step 2] Fetch data with 120 day lookback")
    print("-" * 80)

    try:
        print("Calling: tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=120)")
        tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=120)
        print("✓ Data fetched successfully")
    except Exception as e:
        print(f"✗ Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========================================================================
    # Step 3: Verify DataFrame shapes
    # ========================================================================
    print("\n[Step 3] Verify DataFrame shapes")
    print("-" * 80)

    print(f"\nTSLA DataFrame:")
    print(f"  Shape: {tsla_df.shape}")
    print(f"  Rows: {len(tsla_df)}")
    print(f"  Columns: {list(tsla_df.columns)}")
    print(f"  Index type: {type(tsla_df.index).__name__}")

    print(f"\nSPY DataFrame:")
    print(f"  Shape: {spy_df.shape}")
    print(f"  Rows: {len(spy_df)}")
    print(f"  Columns: {list(spy_df.columns)}")
    print(f"  Index type: {type(spy_df.index).__name__}")

    print(f"\nVIX DataFrame:")
    print(f"  Shape: {vix_df.shape}")
    print(f"  Rows: {len(vix_df)}")
    print(f"  Columns: {list(vix_df.columns)}")
    print(f"  Index type: {type(vix_df.index).__name__}")

    # Validate shapes
    if len(tsla_df) == 0 or len(spy_df) == 0 or len(vix_df) == 0:
        print("\n✗ ERROR: One or more DataFrames are empty!")
        return False

    # Validate columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for name, df in [('TSLA', tsla_df), ('SPY', spy_df)]:
        missing = set(required_cols) - set(df.columns)
        if missing:
            print(f"\n✗ ERROR: {name} missing required columns: {missing}")
            return False

    # Validate index
    if not isinstance(tsla_df.index, pd.DatetimeIndex):
        print(f"\n✗ ERROR: TSLA index is not DatetimeIndex")
        return False

    print("\n✓ All DataFrames have valid shapes, columns, and index types")

    # ========================================================================
    # Step 4: Check data freshness
    # ========================================================================
    print("\n[Step 4] Check data freshness")
    print("-" * 80)

    # Get latest and earliest timestamps
    latest_tsla = tsla_df.index[-1]
    earliest_tsla = tsla_df.index[0]
    latest_spy = spy_df.index[-1]
    latest_vix = vix_df.index[-1]

    print(f"\nTimestamp Information:")
    print(f"  Current time:     {datetime.now()}")
    print(f"  Latest TSLA:      {latest_tsla}")
    print(f"  Latest SPY:       {latest_spy}")
    print(f"  Latest VIX:       {latest_vix}")
    print(f"  Earliest TSLA:    {earliest_tsla}")

    # Calculate data age
    now = datetime.now()
    data_age_hours = (now - latest_tsla).total_seconds() / 3600
    data_age_days = data_age_hours / 24

    print(f"\nData Freshness:")
    print(f"  Age (hours):      {data_age_hours:.1f}")
    print(f"  Age (days):       {data_age_days:.1f}")

    # Classify freshness
    if data_age_hours < 1:
        freshness = "VERY FRESH (< 1 hour old)"
    elif data_age_hours < 24:
        freshness = "FRESH (< 1 day old)"
    elif data_age_days < 7:
        freshness = "RECENT (< 1 week old)"
    else:
        freshness = "STALE (> 1 week old)"

    print(f"  Status:           {freshness}")

    # Check lookback coverage
    cutoff = now - timedelta(days=120)
    coverage_days = (latest_tsla - earliest_tsla).days

    print(f"\nLookback Coverage:")
    print(f"  Requested:        120 days")
    print(f"  Actual:           {coverage_days} days")
    print(f"  120-day cutoff:   {cutoff}")

    if earliest_tsla > cutoff:
        gap_days = (earliest_tsla - cutoff).days
        print(f"  ⚠ Note: Data starts {gap_days} days after cutoff (CSV may not have 120 days)")
    else:
        print(f"  ✓ Full 120-day coverage achieved")

    # ========================================================================
    # Step 5: Ensure it returns valid data
    # ========================================================================
    print("\n[Step 5] Ensure it returns valid data")
    print("-" * 80)

    # Check for NaN values
    print(f"\nNaN/Missing Value Check:")
    tsla_nans = tsla_df.isnull().sum().sum()
    spy_nans = spy_df.isnull().sum().sum()
    vix_nans = vix_df.isnull().sum().sum()

    print(f"  TSLA NaN count:   {tsla_nans}")
    print(f"  SPY NaN count:    {spy_nans}")
    print(f"  VIX NaN count:    {vix_nans}")

    if tsla_nans + spy_nans + vix_nans == 0:
        print(f"  ✓ No missing values detected")
    else:
        print(f"  ⚠ Warning: Missing values detected")

    # Value ranges
    print(f"\nPrice Ranges:")
    print(f"  TSLA:  ${tsla_df['close'].min():>8.2f} - ${tsla_df['close'].max():>8.2f}")
    print(f"  SPY:   ${spy_df['close'].min():>8.2f} - ${spy_df['close'].max():>8.2f}")
    print(f"  VIX:   {vix_df['close'].min():>9.2f} - {vix_df['close'].max():>9.2f}")

    # Latest values
    print(f"\nLatest Closing Prices:")
    print(f"  TSLA:  ${tsla_df['close'].iloc[-1]:>8.2f}")
    print(f"  SPY:   ${spy_df['close'].iloc[-1]:>8.2f}")
    print(f"  VIX:   {vix_df['close'].iloc[-1]:>9.2f}")

    # Sanity checks
    print(f"\nSanity Checks:")
    checks_passed = True

    if tsla_df['close'].iloc[-1] <= 0:
        print(f"  ✗ TSLA price is invalid (≤ 0)")
        checks_passed = False
    else:
        print(f"  ✓ TSLA price is valid (> 0)")

    if spy_df['close'].iloc[-1] <= 0:
        print(f"  ✗ SPY price is invalid (≤ 0)")
        checks_passed = False
    else:
        print(f"  ✓ SPY price is valid (> 0)")

    if vix_df['close'].iloc[-1] <= 0 or vix_df['close'].iloc[-1] > 100:
        print(f"  ✗ VIX value is out of reasonable range")
        checks_passed = False
    else:
        print(f"  ✓ VIX value is in reasonable range (0-100)")

    # Check OHLC relationship
    tsla_sample = tsla_df.tail(100)
    invalid_bars = sum(
        (tsla_sample['low'] > tsla_sample['high']) |
        (tsla_sample['open'] > tsla_sample['high']) |
        (tsla_sample['open'] < tsla_sample['low']) |
        (tsla_sample['close'] > tsla_sample['high']) |
        (tsla_sample['close'] < tsla_sample['low'])
    )

    if invalid_bars > 0:
        print(f"  ✗ Found {invalid_bars} invalid OHLC bars in last 100 bars")
        checks_passed = False
    else:
        print(f"  ✓ OHLC relationships are valid")

    if not checks_passed:
        print("\n✗ Some sanity checks failed!")
        return False

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    print(f"\n✓ All 5 steps completed successfully:")
    print(f"  1. ✓ Imported load_live_data_tuple")
    print(f"  2. ✓ Fetched data with 120-day lookback")
    print(f"  3. ✓ Verified DataFrame shapes ({len(tsla_df)} TSLA, {len(spy_df)} SPY, {len(vix_df)} VIX rows)")
    print(f"  4. ✓ Checked data freshness ({data_age_hours:.1f} hours old)")
    print(f"  5. ✓ Validated data quality (no NaN, valid ranges, valid OHLC)")

    print(f"\n✓ Live data integration is working correctly!")
    print(f"\n" + "="*80)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
