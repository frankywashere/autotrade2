"""
Test to investigate precomputed resampling lookahead risk.

This test demonstrates:
1. How searchsorted with side='right' behaves
2. Whether there's a lookahead risk when using precomputed daily data
3. The difference between resampling df[:i] directly vs using precomputed slice
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample 5-minute intraday data for a trading day
# Market opens at 9:30, closes at 16:00 (6.5 hours = 78 bars of 5-min data)

def create_sample_data():
    """Create sample 5-min data for demonstration."""
    # Start at 9:30 AM
    start = pd.Timestamp('2024-01-02 09:30:00')

    # Create 5-min timestamps for multiple days
    timestamps = []
    for day in range(3):  # 3 trading days
        day_start = start + pd.Timedelta(days=day)
        for i in range(78):  # 78 bars per day (6.5 hours * 12 bars/hour)
            timestamps.append(day_start + pd.Timedelta(minutes=5*i))

    # Create data
    n = len(timestamps)
    df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 101,
        'low': np.random.randn(n).cumsum() + 99,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n)
    }, index=pd.DatetimeIndex(timestamps))

    return df

def resample_to_daily(df):
    """Resample to daily using pandas resample."""
    return df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

def demonstrate_searchsorted():
    """Demonstrate searchsorted behavior with side='right'."""
    print("=" * 80)
    print("1. Understanding searchsorted with side='right'")
    print("=" * 80)

    # Simple example
    index = pd.DatetimeIndex([
        '2024-01-02 00:00:00',  # Daily bar for Jan 2
        '2024-01-03 00:00:00',  # Daily bar for Jan 3
        '2024-01-04 00:00:00',  # Daily bar for Jan 4
    ])

    print("\nDaily bar index (00:00:00 = start of day):")
    for i, ts in enumerate(index):
        print(f"  [{i}] {ts}")

    # Test searchsorted with different query times
    test_times = [
        '2024-01-02 14:30:00',  # Midday on Jan 2
        '2024-01-02 23:59:59',  # End of Jan 2
        '2024-01-03 00:00:00',  # Exactly start of Jan 3
        '2024-01-03 14:30:00',  # Midday on Jan 3
    ]

    print("\nSearchsorted behavior with side='right':")
    for query in test_times:
        query_ts = pd.Timestamp(query)
        idx_right = index.searchsorted(query_ts, side='right')
        idx_left = index.searchsorted(query_ts, side='left')
        print(f"\nQuery: {query}")
        print(f"  side='right': idx={idx_right} (slice [:idx_right] includes bars 0 to {idx_right-1})")
        print(f"  side='left':  idx={idx_left} (slice [:idx_left] includes bars 0 to {idx_left-1})")

def test_lookahead_scenario():
    """Test the specific lookahead scenario: historical data at 14:30 vs daily bar."""
    print("\n\n" + "=" * 80)
    print("2. Testing Lookahead Scenario: Historical data ends at 14:30")
    print("=" * 80)

    df_5min = create_sample_data()

    # Simulate: we're at 14:30 on Jan 2 (after 60 bars of 5-min data)
    # 60 bars * 5 min = 300 min = 5 hours after 9:30 = 14:30
    historical_end_bar = 60  # This is bar index 60 (0-indexed)
    df_historical = df_5min.iloc[:historical_end_bar + 1]

    print(f"\n5-min data:")
    print(f"  First timestamp: {df_5min.index[0]}")
    print(f"  Last timestamp:  {df_5min.index[-1]}")
    print(f"  Total bars: {len(df_5min)}")

    print(f"\nHistorical data (up to bar {historical_end_bar}):")
    print(f"  First timestamp: {df_historical.index[0]}")
    print(f"  Last timestamp:  {df_historical.index[-1]}")
    print(f"  Total bars: {len(df_historical)}")

    # Resample historical data directly
    df_historical_daily = resample_to_daily(df_historical)

    # Resample full data (precomputed approach)
    df_full_daily = resample_to_daily(df_5min)

    print(f"\n--- METHOD 1: Direct resample of historical data (df[:60+1]) ---")
    print(f"Daily bars: {len(df_historical_daily)}")
    for i, (ts, row) in enumerate(df_historical_daily.iterrows()):
        print(f"  [{i}] {ts}: open={row['open']:.2f}, close={row['close']:.2f}")

    print(f"\n--- METHOD 2: Precomputed full data, then slice ---")
    print(f"Full daily bars: {len(df_full_daily)}")
    for i, (ts, row) in enumerate(df_full_daily.iterrows()):
        print(f"  [{i}] {ts}: open={row['open']:.2f}, close={row['close']:.2f}")

    # Now use searchsorted to slice precomputed data
    end_timestamp = df_historical.index[-1]
    print(f"\nUsing searchsorted to slice precomputed data:")
    print(f"  end_timestamp (last historical bar): {end_timestamp}")

    idx_right = df_full_daily.index.searchsorted(end_timestamp, side='right')
    idx_left = df_full_daily.index.searchsorted(end_timestamp, side='left')

    print(f"  searchsorted(side='right'): idx={idx_right}")
    print(f"  searchsorted(side='left'):  idx={idx_left}")

    df_precomputed_slice_right = df_full_daily.iloc[:idx_right]
    df_precomputed_slice_left = df_full_daily.iloc[:idx_left]

    print(f"\nPrecomputed slice with side='right':")
    print(f"  Bars included: {len(df_precomputed_slice_right)}")
    for i, (ts, row) in enumerate(df_precomputed_slice_right.iterrows()):
        print(f"  [{i}] {ts}: open={row['open']:.2f}, close={row['close']:.2f}")

    print(f"\nPrecomputed slice with side='left':")
    print(f"  Bars included: {len(df_precomputed_slice_left)}")
    for i, (ts, row) in enumerate(df_precomputed_slice_left.iterrows()):
        print(f"  [{i}] {ts}: open={row['open']:.2f}, close={row['close']:.2f}")

def test_partial_bar_data():
    """Test what data is included in the partial daily bar."""
    print("\n\n" + "=" * 80)
    print("3. Comparing data in the partial daily bar")
    print("=" * 80)

    df_5min = create_sample_data()

    # Historical data ends at 14:30 (bar 60)
    historical_end_bar = 60
    df_historical = df_5min.iloc[:historical_end_bar + 1]

    # Get the Jan 2 daily bar when resampling historical data
    df_historical_daily = resample_to_daily(df_historical)
    jan2_historical = df_historical_daily.iloc[0]

    # Get the Jan 2 daily bar when resampling full data
    df_full_daily = resample_to_daily(df_5min)
    jan2_full = df_full_daily.iloc[0]

    print("\nJan 2 daily bar - Historical resample (9:30 to 14:30):")
    print(f"  Timestamp: {df_historical_daily.index[0]}")
    print(f"  Open:   {jan2_historical['open']:.2f}")
    print(f"  High:   {jan2_historical['high']:.2f}")
    print(f"  Low:    {jan2_historical['low']:.2f}")
    print(f"  Close:  {jan2_historical['close']:.2f}")
    print(f"  Volume: {jan2_historical['volume']:.0f}")

    print("\nJan 2 daily bar - Full resample (9:30 to 16:00):")
    print(f"  Timestamp: {df_full_daily.index[0]}")
    print(f"  Open:   {jan2_full['open']:.2f}")
    print(f"  High:   {jan2_full['high']:.2f}")
    print(f"  Low:    {jan2_full['low']:.2f}")
    print(f"  Close:  {jan2_full['close']:.2f}")
    print(f"  Volume: {jan2_full['volume']:.0f}")

    print("\nDifference (Full - Historical):")
    print(f"  Open:   {jan2_full['open'] - jan2_historical['open']:.2f} (should be 0, open is 'first')")
    print(f"  High:   {jan2_full['high'] - jan2_historical['high']:.2f} (lookahead if > 0)")
    print(f"  Low:    {jan2_full['low'] - jan2_historical['low']:.2f} (lookahead if < 0)")
    print(f"  Close:  {jan2_full['close'] - jan2_historical['close']:.2f} (lookahead if != 0)")
    print(f"  Volume: {jan2_full['volume'] - jan2_historical['volume']:.0f} (lookahead if > 0)")

    # Count how many 5-min bars are in each
    jan2_start = pd.Timestamp('2024-01-02 09:30:00')
    jan2_end_historical = df_historical.index[-1]
    jan2_end_full = pd.Timestamp('2024-01-02 15:55:00')  # Last 5-min bar

    print(f"\n5-min bars included:")
    print(f"  Historical: {historical_end_bar + 1} bars (9:30 to {jan2_end_historical})")
    print(f"  Full: 78 bars (9:30 to 16:00)")
    print(f"  Lookahead: {78 - (historical_end_bar + 1)} bars")

def analyze_lookahead_risk():
    """Analyze whether this is a real lookahead risk."""
    print("\n\n" + "=" * 80)
    print("4. LOOKAHEAD RISK ANALYSIS")
    print("=" * 80)

    print("""
The precomputed resampling approach has a REAL lookahead risk:

PROBLEM:
--------
When historical 5-min data ends at 14:30, and we use searchsorted(side='right')
to slice the precomputed daily data:

1. The 5-min timestamp 14:30 falls WITHIN the daily bar (Jan 2, 00:00:00)
2. searchsorted(side='right') returns idx=1, including the full Jan 2 daily bar
3. The full Jan 2 daily bar contains data from 9:30 to 16:00
4. This includes 14:35, 14:40, ..., 16:00 which are FUTURE relative to 14:30

EVIDENCE:
---------
- Historical resample: Uses only 9:30-14:30 data (61 bars)
- Precomputed slice: Uses the full 9:30-16:00 data (78 bars)
- The daily bar's high/low/close/volume will be different
- This is LOOKAHEAD because we're using future data to compute the daily bar

WHY side='right'?
-----------------
searchsorted(side='right') means: find the position AFTER all elements <= query
- If query is 14:30 and daily bars are [00:00, 00:00, 00:00]
- side='right' finds the position after all bars <= 14:30
- This includes the Jan 2 bar (since 00:00 < 14:30)
- Result: idx=1, slice [:1] includes Jan 2 bar

But the Jan 2 bar in precomputed data contains all of Jan 2's data,
not just the data up to 14:30!

VERDICT:
--------
This IS a lookahead risk. The precomputed approach includes future data
in the partial daily bar that wouldn't be available at 14:30.

IMPACT:
-------
- For channel detection on daily timeframes, this means the channel bounds
  could be computed using high/low values that occur AFTER the historical point
- For features, this means daily OHLCV values include future data
- The magnitude depends on how much the market moves from 14:30 to 16:00

CORRECT APPROACH:
-----------------
Should use fresh resampling of df[:i] to ensure the partial bar only includes
data up to the historical point, not the full day's data.
""")

if __name__ == '__main__':
    demonstrate_searchsorted()
    test_lookahead_scenario()
    test_partial_bar_data()
    analyze_lookahead_risk()
