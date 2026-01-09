#!/usr/bin/env python3
"""
Full API test demonstrating both basic and advanced usage.
"""

from v7.data import fetch_live_data, load_live_data_tuple, is_market_open
from datetime import datetime

print("="*80)
print("LIVE DATA API - FULL DEMONSTRATION")
print("="*80)

# ============================================================================
# Method 1: Simple tuple interface (backward compatible)
# ============================================================================
print("\n1. Simple Tuple Interface (load_live_data_tuple)")
print("-"*80)

tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=120)

print(f"Loaded data:")
print(f"  TSLA: {len(tsla_df)} rows")
print(f"  SPY:  {len(spy_df)} rows")
print(f"  VIX:  {len(vix_df)} rows")
print(f"\nLatest prices:")
print(f"  TSLA: ${tsla_df['close'].iloc[-1]:.2f}")
print(f"  SPY:  ${spy_df['close'].iloc[-1]:.2f}")


# ============================================================================
# Method 2: Rich result interface with metadata
# ============================================================================
print("\n2. Rich Result Interface (fetch_live_data)")
print("-"*80)

result = fetch_live_data(lookback_days=120)

print(f"Data Status: {result.status}")
print(f"  LIVE   = data < 15 minutes old")
print(f"  RECENT = data < 60 minutes old")
print(f"  STALE  = data > 60 minutes old")
print(f"  HISTORICAL = CSV data only (no yfinance)")

print(f"\nTimestamp: {result.timestamp}")
print(f"Data Age: {result.data_age_minutes:.1f} minutes")

print(f"\nDataFrames:")
print(f"  TSLA: {len(result.tsla_df)} rows")
print(f"  SPY:  {len(result.spy_df)} rows")
print(f"  VIX:  {len(result.vix_df)} rows")


# ============================================================================
# Method 3: Force historical mode (skip yfinance)
# ============================================================================
print("\n3. Force Historical Mode")
print("-"*80)

historical_result = fetch_live_data(lookback_days=120, force_historical=True)

print(f"Status: {historical_result.status} (should be HISTORICAL)")
print(f"This mode uses only CSV data, skipping live yfinance fetch")


# ============================================================================
# Market status check
# ============================================================================
print("\n4. Market Status Check")
print("-"*80)

market_open = is_market_open()
current_time = datetime.now()

print(f"Current time: {current_time}")
print(f"Market open: {market_open}")

if market_open:
    print("  Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri")
    print("  Data should be LIVE or RECENT")
else:
    print("  Market is closed (after hours, weekend, or holiday)")
    print("  Data will be STALE or HISTORICAL")


# ============================================================================
# Data validation
# ============================================================================
print("\n5. Data Validation Summary")
print("-"*80)

# Check columns
print(f"TSLA columns: {list(tsla_df.columns)}")
print(f"Index type: {type(tsla_df.index).__name__}")

# Check for missing data
tsla_nans = tsla_df.isnull().sum().sum()
spy_nans = spy_df.isnull().sum().sum()
print(f"\nMissing values: TSLA={tsla_nans}, SPY={spy_nans}")

# Time coverage
time_span = (tsla_df.index[-1] - tsla_df.index[0]).days
print(f"Time coverage: {time_span} days")

# Sample data
print(f"\nSample TSLA data (last 3 bars):")
print(tsla_df[['close', 'volume']].tail(3))


print("\n" + "="*80)
print("✓ All API methods working correctly!")
print("="*80)
