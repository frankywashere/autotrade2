"""
Comprehensive test demonstrating the lookahead risk impact in the actual codebase flow.

This shows:
1. How the precomputed optimization is used in labels.py
2. The actual lookahead when used in generate_labels_per_tf
3. Impact on channel detection and features
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from v7.core.timeframe import resample_ohlc

def create_realistic_data():
    """Create realistic 5-min intraday data with a clear trend in afternoon."""
    # Start at 9:30 AM on Jan 2, 2024
    start = pd.Timestamp('2024-01-02 09:30:00')

    # Create timestamps for 2 full trading days
    timestamps = []
    for day in range(2):
        day_start = start + pd.Timedelta(days=day)
        for i in range(78):  # 78 bars per day
            timestamps.append(day_start + pd.Timedelta(minutes=5*i))

    n = len(timestamps)

    # Create price data with specific pattern:
    # - Morning (9:30-14:30): sideways around 100
    # - Afternoon (14:30-16:00): strong rally to 110
    prices = []
    for i, ts in enumerate(timestamps):
        bar_of_day = i % 78
        if bar_of_day < 60:  # Morning (9:30-14:30)
            # Sideways around 100
            price = 100 + np.random.randn() * 0.5
        else:  # Afternoon (14:30-16:00)
            # Strong rally
            afternoon_bars = bar_of_day - 60
            price = 100 + (afternoon_bars / 18) * 10 + np.random.randn() * 0.5

        prices.append(price)

    prices = np.array(prices)

    df = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.1,
        'high': prices + np.abs(np.random.randn(n)) * 0.3,
        'low': prices - np.abs(np.random.randn(n)) * 0.3,
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=pd.DatetimeIndex(timestamps))

    return df


def simulate_actual_flow():
    """Simulate the actual flow from generate_labels_per_tf."""
    print("=" * 80)
    print("SIMULATING ACTUAL FLOW FROM generate_labels_per_tf")
    print("=" * 80)

    df_5min = create_realistic_data()

    # Simulate being at position i=60 (14:30 on Jan 2)
    channel_end_idx_5min = 60
    df_historical = df_5min.iloc[:channel_end_idx_5min + 1]

    print(f"\nScenario: Detecting channel at 5-min bar {channel_end_idx_5min}")
    print(f"  Historical data: {df_historical.index[0]} to {df_historical.index[-1]}")
    print(f"  This is 14:30 on Jan 2 (61 bars into the trading day)")

    print("\n" + "-" * 80)
    print("RESAMPLING TO DAILY")
    print("-" * 80)

    # Method 1: Correct approach (used in generate_labels_per_tf for df_tf_for_channel)
    print("\n[1] CORRECT: Resample historical data only (no lookahead)")
    df_daily_historical = resample_ohlc(df_historical, 'daily')
    print(f"  Result: {len(df_daily_historical)} daily bar(s)")
    for i, (ts, row) in enumerate(df_daily_historical.iterrows()):
        print(f"    [{i}] {ts}: O={row['open']:.2f}, H={row['high']:.2f}, "
              f"L={row['low']:.2f}, C={row['close']:.2f}, V={row['volume']:.0f}")

    # Method 2: Precomputed approach (what _try_get_precomputed_slice does)
    print("\n[2] PRECOMPUTED: Slice precomputed full data (LOOKAHEAD RISK)")
    df_daily_full = resample_ohlc(df_5min, 'daily')
    print(f"  Full precomputed: {len(df_daily_full)} daily bars")

    # Simulate searchsorted logic from _try_get_precomputed_slice
    end_timestamp = df_historical.index[-1]
    idx = df_daily_full.index.searchsorted(end_timestamp, side='right')
    df_daily_precomputed = df_daily_full.iloc[:idx]

    print(f"  Sliced using searchsorted('{end_timestamp}', side='right'): idx={idx}")
    print(f"  Result: {len(df_daily_precomputed)} daily bar(s)")
    for i, (ts, row) in enumerate(df_daily_precomputed.iterrows()):
        print(f"    [{i}] {ts}: O={row['open']:.2f}, H={row['high']:.2f}, "
              f"L={row['low']:.2f}, C={row['close']:.2f}, V={row['volume']:.0f}")

    print("\n" + "-" * 80)
    print("COMPARING RESULTS")
    print("-" * 80)

    jan2_historical = df_daily_historical.iloc[0]
    jan2_precomputed = df_daily_precomputed.iloc[0]

    print("\nJan 2 daily bar comparison:")
    print(f"  Open:   Historical={jan2_historical['open']:.2f}, "
          f"Precomputed={jan2_precomputed['open']:.2f}, "
          f"Diff={jan2_precomputed['open'] - jan2_historical['open']:.2f}")
    print(f"  High:   Historical={jan2_historical['high']:.2f}, "
          f"Precomputed={jan2_precomputed['high']:.2f}, "
          f"Diff={jan2_precomputed['high'] - jan2_historical['high']:.2f}")
    print(f"  Low:    Historical={jan2_historical['low']:.2f}, "
          f"Precomputed={jan2_precomputed['low']:.2f}, "
          f"Diff={jan2_precomputed['low'] - jan2_historical['low']:.2f}")
    print(f"  Close:  Historical={jan2_historical['close']:.2f}, "
          f"Precomputed={jan2_precomputed['close']:.2f}, "
          f"Diff={jan2_precomputed['close'] - jan2_historical['close']:.2f}")
    print(f"  Volume: Historical={jan2_historical['volume']:.0f}, "
          f"Precomputed={jan2_precomputed['volume']:.0f}, "
          f"Diff={jan2_precomputed['volume'] - jan2_historical['volume']:.0f}")

    # Show the 5-min data that's included/excluded
    print("\n5-min bars included in each:")
    print(f"  Historical: bars 0-60 (9:30 to 14:30)")
    print(f"  Precomputed: bars 0-77 (9:30 to 16:00)")
    print(f"  LOOKAHEAD: bars 61-77 (14:35 to 16:00) - 17 bars of future data")

    # Show afternoon data that's the lookahead
    df_afternoon = df_5min.iloc[61:78]  # Bars 61-77 (14:35-16:00)
    afternoon_high = df_afternoon['high'].max()
    afternoon_low = df_afternoon['low'].min()
    afternoon_close = df_afternoon['close'].iloc[-1]
    afternoon_volume = df_afternoon['volume'].sum()

    print("\nAfternoon data (LOOKAHEAD):")
    print(f"  High:   {afternoon_high:.2f} (contributes to daily high)")
    print(f"  Low:    {afternoon_low:.2f} (contributes to daily low)")
    print(f"  Close:  {afternoon_close:.2f} (becomes daily close)")
    print(f"  Volume: {afternoon_volume:.0f} (adds to daily volume)")


def analyze_impact_on_channel_detection():
    """Analyze how lookahead affects channel detection."""
    print("\n\n" + "=" * 80)
    print("IMPACT ON CHANNEL DETECTION")
    print("=" * 80)

    df_5min = create_realistic_data()
    channel_end_idx_5min = 60
    df_historical = df_5min.iloc[:channel_end_idx_5min + 1]

    # Resample to daily using both approaches
    df_daily_correct = resample_ohlc(df_historical, 'daily')
    df_daily_full = resample_ohlc(df_5min, 'daily')
    end_timestamp = df_historical.index[-1]
    idx = df_daily_full.index.searchsorted(end_timestamp, side='right')
    df_daily_lookahead = df_daily_full.iloc[:idx]

    print("\nDaily OHLC for channel detection:")
    print("\nCORRECT (no lookahead):")
    print(df_daily_correct[['open', 'high', 'low', 'close']].to_string())

    print("\nPRECOMPUTED (with lookahead):")
    print(df_daily_lookahead[['open', 'high', 'low', 'close']].to_string())

    print("\n\nIMPACT:")
    print("-------")
    print("If channel detection uses these daily bars:")
    print("  - High/Low range is WIDER with lookahead (includes afternoon rally)")
    print("  - Channel std_dev will be LARGER")
    print("  - Channel bounds will be DIFFERENT")
    print("  - This affects whether price is 'in channel' or 'broke out'")
    print("  - Labels (break direction, duration) will be INCORRECT")


def final_verdict():
    """Print final analysis."""
    print("\n\n" + "=" * 80)
    print("FINAL VERDICT ON LOOKAHEAD RISK")
    print("=" * 80)

    print("""
1. HOW _try_get_precomputed_slice() USES searchsorted:
   ===================================================
   Line 240: idx = precomputed_df.index.searchsorted(end_timestamp, side='right')
   Line 243: return precomputed_df.iloc[:idx]

   - Takes the last timestamp from historical data (e.g., 14:30)
   - Uses searchsorted to find position in precomputed daily index
   - Returns all daily bars up to that position

2. WHAT side='right' MEANS:
   =========================
   searchsorted(side='right') finds the insertion point AFTER all elements <= query.

   Example with daily bars indexed at 00:00:00:
   - Daily index: [2024-01-02 00:00:00, 2024-01-03 00:00:00, ...]
   - Query: 2024-01-02 14:30:00 (midday timestamp)
   - side='right' returns: idx=1 (position after Jan 2 bar)
   - Slice [:1] includes: Jan 2 bar

   This INCLUDES the Jan 2 bar, even though query is 14:30.

3. CONCRETE EXAMPLE WITH TIMESTAMPS:
   ==================================
   Historical data: 9:30 to 14:30 (61 bars)
   Daily bar: 00:00:00 (represents entire day 9:30-16:00)

   Direct resample of df[:61]:
   - High: max(9:30 to 14:30)
   - Low: min(9:30 to 14:30)
   - Close: close at 14:30
   - Volume: sum(9:30 to 14:30)

   Precomputed slice with searchsorted:
   - High: max(9:30 to 16:00) ← INCLUDES FUTURE
   - Low: min(9:30 to 16:00) ← INCLUDES FUTURE
   - Close: close at 16:00 ← INCLUDES FUTURE
   - Volume: sum(9:30 to 16:00) ← INCLUDES FUTURE

4. IS THERE ACTUAL LOOKAHEAD?
   ===========================
   YES. The precomputed daily bar includes data from 14:35 to 16:00
   which occurs AFTER the historical endpoint of 14:30.

   Evidence from test:
   - Close differs by several points
   - Volume differs by ~89k shares
   - High/Low can differ significantly if afternoon has volatility

5. IS THIS A FALSE ALARM?
   =======================
   NO. This is a REAL lookahead bug.

   The comment on line 216-224 says:
   "The pre-computed approach is mathematically equivalent to fresh
   resampling for all COMPLETE time periods. For the last partial period,
   there may be a minor difference..."

   This is INCORRECT. It's not a "minor difference in the last partial bar."
   It's FUTURE DATA being included in the current bar.

RECOMMENDATION:
===============
The precomputed optimization should NOT be used for partial bars.
Either:
a) Fall back to fresh resampling when detecting a partial bar
b) Use searchsorted with the START of the next period to exclude partial data
c) Pre-compute partial bars correctly for each historical position (expensive)

The current implementation trades correctness for speed, introducing
lookahead bias that can affect model training.
""")


if __name__ == '__main__':
    simulate_actual_flow()
    analyze_impact_on_channel_detection()
    final_verdict()
