#!/usr/bin/env python3
"""
Investigate resample boundary issue in detail.

This script answers:
1. When does a daily bar "close"? At 16:00 ET market close?
2. If historical data ends at 14:30, what does resample_ohlc() include in that day's bar?
3. What does the precomputed full resample include?
4. How does searchsorted with side='right' work exactly?
5. Is there a concrete example where parallel includes future data that sequential doesn't?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ============================================================================
# 1. Understanding when a daily bar "closes"
# ============================================================================

print("=" * 80)
print("1. WHEN DOES A DAILY BAR CLOSE?")
print("=" * 80)

# Create sample data with 5-minute bars on a trading day
# US Eastern Time - market hours 9:30 AM to 4:00 PM
et = pytz.timezone('US/Eastern')
base_date = datetime(2024, 1, 2, 9, 30, tzinfo=et)  # Tuesday 9:30 AM

# Generate 5-minute bars for a full trading day (9:30 AM to 4:00 PM = 6.5 hours = 78 bars)
timestamps = [base_date + timedelta(minutes=5*i) for i in range(78)]
df = pd.DataFrame({
    'open': np.random.uniform(100, 101, 78),
    'high': np.random.uniform(101, 102, 78),
    'low': np.random.uniform(99, 100, 78),
    'close': np.random.uniform(100, 101, 78),
    'volume': np.random.randint(1000, 10000, 78)
}, index=pd.DatetimeIndex(timestamps))

print(f"\nOriginal 5min data:")
print(f"First bar: {df.index[0]}")
print(f"Last bar:  {df.index[-1]}")
print(f"Total bars: {len(df)}")

# Resample to daily
daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nResampled to daily:")
print(f"Daily bar timestamp: {daily.index[0]}")
print(f"Number of daily bars: {len(daily)}")

# Test with UTC timestamps (no timezone)
print("\n" + "-" * 80)
print("Testing with UTC/naive timestamps:")
timestamps_utc = [datetime(2024, 1, 2, 14, 30) + timedelta(minutes=5*i) for i in range(78)]
df_utc = pd.DataFrame({
    'open': np.random.uniform(100, 101, 78),
    'high': np.random.uniform(101, 102, 78),
    'low': np.random.uniform(99, 100, 78),
    'close': np.random.uniform(100, 101, 78),
    'volume': np.random.randint(1000, 10000, 78)
}, index=pd.DatetimeIndex(timestamps_utc))

daily_utc = df_utc.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nOriginal 5min data (UTC):")
print(f"First bar: {df_utc.index[0]}")
print(f"Last bar:  {df_utc.index[-1]}")
print(f"\nResampled daily bar:")
print(f"Timestamp: {daily_utc.index[0]}")
print(f"Close value from bar at: {df_utc.index[-1]}")

# Key finding: Daily bar timestamp is at MIDNIGHT of the day
print(f"\n*** KEY FINDING: Daily bars are labeled at MIDNIGHT (00:00) of the day ***")
print(f"*** The 'close' of a daily bar is the last value from that calendar day ***")

# ============================================================================
# 2. Partial day behavior
# ============================================================================

print("\n" + "=" * 80)
print("2. PARTIAL DAY BEHAVIOR - What if data ends at 14:30?")
print("=" * 80)

# Create data that ends mid-day (at 2:30 PM instead of 4:00 PM)
partial_timestamps = [datetime(2024, 1, 2, 14, 30) + timedelta(minutes=5*i) for i in range(50)]
df_partial = pd.DataFrame({
    'open': np.random.uniform(100, 101, 50),
    'high': np.random.uniform(101, 102, 50),
    'low': np.random.uniform(99, 100, 50),
    'close': np.random.uniform(100, 101, 50),
    'volume': np.random.randint(1000, 10000, 50)
}, index=pd.DatetimeIndex(partial_timestamps))

print(f"\nPartial day data (ends at 2:30 PM):")
print(f"First bar: {df_partial.index[0]}")
print(f"Last bar:  {df_partial.index[-1]}")
print(f"Total bars: {len(df_partial)}")

daily_partial = df_partial.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nResampled to daily:")
print(f"Daily bar timestamp: {daily_partial.index[0]}")
print(f"Daily 'close' comes from: {df_partial.index[-1]} (last available bar)")
print(f"\n*** FINDING: Partial day creates a daily bar with data up to last timestamp ***")
print(f"*** The daily bar includes ALL data from that calendar day, even if incomplete ***")

# ============================================================================
# 3. Precomputed resample comparison
# ============================================================================

print("\n" + "=" * 80)
print("3. PRECOMPUTED FULL RESAMPLE vs PARTIAL RESAMPLE")
print("=" * 80)

# Simulate the parallel scenario: We have full data resampled once
full_timestamps = [datetime(2024, 1, 2, 14, 30) + timedelta(minutes=5*i) for i in range(78)]
df_full = pd.DataFrame({
    'open': [100.0 + i*0.01 for i in range(78)],  # Deterministic for comparison
    'high': [101.0 + i*0.01 for i in range(78)],
    'low': [99.0 + i*0.01 for i in range(78)],
    'close': [100.5 + i*0.01 for i in range(78)],
    'volume': [1000 + i*10 for i in range(78)]
}, index=pd.DatetimeIndex(full_timestamps))

# Full resample (what parallel mode precomputes)
daily_full = df_full.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nFull data (78 bars, full day):")
print(f"First: {df_full.index[0]}")
print(f"Last:  {df_full.index[-1]}")
print(f"\nFull daily resample:")
print(daily_full)

# Partial resample (what sequential mode would do at position i=50)
df_partial_seq = df_full.iloc[:50]
daily_partial_seq = df_partial_seq.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nPartial data (50 bars, up to 2:30 PM):")
print(f"First: {df_partial_seq.index[0]}")
print(f"Last:  {df_partial_seq.index[-1]}")
print(f"\nPartial daily resample (SEQUENTIAL approach):")
print(daily_partial_seq)

print(f"\n*** COMPARISON ***")
print(f"Full daily 'close':    {daily_full['close'].iloc[0]:.4f} (from bar at {df_full.index[-1]})")
print(f"Partial daily 'close': {daily_partial_seq['close'].iloc[0]:.4f} (from bar at {df_partial_seq.index[-1]})")
print(f"\nDifference in 'close': {daily_full['close'].iloc[0] - daily_partial_seq['close'].iloc[0]:.4f}")
print(f"*** FINDING: Sequential and full resample give DIFFERENT values for partial days ***")

# ============================================================================
# 4. searchsorted with side='right' behavior
# ============================================================================

print("\n" + "=" * 80)
print("4. HOW SEARCHSORTED WITH SIDE='RIGHT' WORKS")
print("=" * 80)

# Create a simple example
times = pd.DatetimeIndex([
    '2024-01-02 00:00',  # idx 0 - Daily bar 1
    '2024-01-03 00:00',  # idx 1 - Daily bar 2
    '2024-01-04 00:00',  # idx 2 - Daily bar 3
    '2024-01-05 00:00',  # idx 3 - Daily bar 4
])

print(f"\nDaily bar index:")
for i, t in enumerate(times):
    print(f"  idx {i}: {t}")

# Test searchsorted with different end timestamps
test_timestamps = [
    pd.Timestamp('2024-01-02 14:30'),  # Mid-day of bar 1
    pd.Timestamp('2024-01-02 23:59'),  # End of day 1
    pd.Timestamp('2024-01-03 00:00'),  # Exact boundary (bar 2 timestamp)
    pd.Timestamp('2024-01-03 14:30'),  # Mid-day of bar 2
]

print(f"\nSearchsorted results (side='right'):")
for ts in test_timestamps:
    idx_right = times.searchsorted(ts, side='right')
    print(f"  timestamp: {ts}")
    print(f"  -> searchsorted(side='right') = {idx_right}")
    print(f"  -> slice [:idx] includes bars: {list(range(idx_right))}")
    print()

print("*** KEY FINDING: side='right' with exact match returns position AFTER the match ***")
print("*** This means if end_timestamp = '2024-01-03 00:00', we include bars [0, 1] ***")
print("*** If a 5min bar timestamp equals a daily bar boundary, we INCLUDE that daily bar ***")

# ============================================================================
# 5. Concrete lookahead example
# ============================================================================

print("\n" + "=" * 80)
print("5. CONCRETE LOOKAHEAD EXAMPLE")
print("=" * 80)

# Scenario: We're at position i=50 (2:30 PM on Jan 2)
# We want to generate features for a channel ending at this position

# Create two days of data
day1_timestamps = [datetime(2024, 1, 2, 14, 30) + timedelta(minutes=5*i) for i in range(78)]
day2_timestamps = [datetime(2024, 1, 3, 14, 30) + timedelta(minutes=5*i) for i in range(78)]
all_timestamps = day1_timestamps + day2_timestamps

df_two_days = pd.DataFrame({
    'open': [100.0 + i*0.01 for i in range(156)],
    'high': [101.0 + i*0.01 for i in range(156)],
    'low': [99.0 + i*0.01 for i in range(156)],
    'close': [100.5 + i*0.01 for i in range(156)],
    'volume': [1000 + i*10 for i in range(156)]
}, index=pd.DatetimeIndex(all_timestamps))

print(f"\nTwo days of 5min data:")
print(f"Day 1 first bar: {df_two_days.index[0]}")
print(f"Day 1 last bar:  {df_two_days.index[77]}")
print(f"Day 2 first bar: {df_two_days.index[78]}")
print(f"Day 2 last bar:  {df_two_days.index[155]}")

# Position i=50 (mid-day on day 1)
position_i = 50
channel_end_timestamp = df_two_days.index[position_i]

print(f"\n*** SCENARIO: Channel ends at position i={position_i} ***")
print(f"*** Timestamp: {channel_end_timestamp} ***")

# SEQUENTIAL approach: Resample df[:i+1]
df_sequential = df_two_days.iloc[:position_i + 1]
daily_sequential = df_sequential.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nSEQUENTIAL approach (resample df[:51]):")
print(f"Data ends at: {df_sequential.index[-1]}")
print(f"Daily bars:")
print(daily_sequential)

# PARALLEL approach: Precompute full resample, then slice by timestamp
daily_full_parallel = df_two_days.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"\nPARALLEL approach (precomputed full resample):")
print(f"Full daily bars:")
print(daily_full_parallel)

# Now slice using searchsorted with end_timestamp
end_timestamp = df_two_days.index[position_i]
idx = daily_full_parallel.index.searchsorted(end_timestamp, side='right')
daily_parallel_sliced = daily_full_parallel.iloc[:idx]

print(f"\nPARALLEL slice using searchsorted('{end_timestamp}', side='right'):")
print(f"searchsorted returned idx={idx}")
print(f"Sliced daily bars ([:idx]):")
print(daily_parallel_sliced)

# CRITICAL COMPARISON
print(f"\n{'='*80}")
print("*** CRITICAL COMPARISON ***")
print(f"{'='*80}")
print(f"\nPosition: i={position_i} (timestamp: {channel_end_timestamp})")
print(f"\nSEQUENTIAL daily 'close': {daily_sequential['close'].iloc[0]:.4f}")
print(f"PARALLEL daily 'close':   {daily_parallel_sliced['close'].iloc[0]:.4f}")
print(f"\nDifference: {daily_parallel_sliced['close'].iloc[0] - daily_sequential['close'].iloc[0]:.4f}")

if abs(daily_parallel_sliced['close'].iloc[0] - daily_sequential['close'].iloc[0]) > 0.0001:
    print(f"\n*** LOOKAHEAD DETECTED! ***")
    print(f"*** Parallel includes future data that sequential doesn't! ***")
    print(f"\nWhy?")
    print(f"- Sequential resamples df[:51], creating a partial day bar")
    print(f"- Parallel precomputes full df with complete day, then slices")
    print(f"- The partial day bar has different 'close' than the complete day")
    print(f"- When parallel slices by timestamp, it includes the COMPLETE daily bar")
    print(f"- This daily bar includes data from positions 51-77 (future relative to position 50)")
else:
    print(f"\n*** No significant difference - both approaches equivalent here ***")

# Show exactly which 5min bars contribute to each approach
print(f"\n{'='*80}")
print("DETAILED BREAKDOWN:")
print(f"{'='*80}")

day1_end_idx = 77
print(f"\nSEQUENTIAL (i=50):")
print(f"  5min bars contributing to daily 'close': 0 to 50")
print(f"  Last 5min bar timestamp: {df_two_days.index[50]}")
print(f"  Daily 'close' value: {df_two_days['close'].iloc[50]:.4f}")

print(f"\nPARALLEL (precomputed full, sliced):")
print(f"  Full day 1 includes 5min bars: 0 to {day1_end_idx}")
print(f"  Last 5min bar timestamp: {df_two_days.index[day1_end_idx]}")
print(f"  Daily 'close' value: {df_two_days['close'].iloc[day1_end_idx]:.4f}")

print(f"\n*** The parallel approach uses data from bars 51-77, which is FUTURE data! ***")
print(f"*** These bars occur AFTER the channel end position at i=50 ***")

# ============================================================================
# 6. Solution verification
# ============================================================================

print("\n" + "=" * 80)
print("6. SOLUTION: How to fix this?")
print("=" * 80)

print("""
The issue: Precomputed full resample creates complete daily bars, but when we
slice by a mid-day timestamp, we include a daily bar that has future data.

Solution approaches:

1. SEQUENTIAL ONLY (safest, but slow):
   - Always resample df[:i+1] for each position
   - Guarantees no lookahead, but redundant computation

2. SMART SLICING (fast, requires careful implementation):
   - Precompute full resample for speed
   - When slicing, check if end_timestamp is mid-day
   - If mid-day, re-resample just that day's data to get correct partial bar
   - Use precomputed bars for all COMPLETE days before the partial day

3. DAILY BOUNDARY ONLY (compromise):
   - Only use precomputed resamples when channel ends at daily boundary
   - For mid-day channels, fall back to sequential resampling
   - Hybrid approach: fast for most cases, safe for edge cases

Current code uses approach 2 but incorrectly implements the slicing.
The searchsorted includes the full daily bar even for mid-day timestamps.
""")

# Demonstrate the fix
print("\nDemonstrating CORRECT smart slicing:")
print("-" * 80)

# Get the daily bar that contains end_timestamp
daily_bar_containing_end = daily_full_parallel.index.asof(end_timestamp)
print(f"End timestamp: {end_timestamp}")
print(f"Daily bar containing end: {daily_bar_containing_end}")

# Check if end_timestamp is at the start of a daily bar (exact boundary)
is_exact_boundary = (end_timestamp == daily_bar_containing_end)
print(f"Is exact boundary? {is_exact_boundary}")

if is_exact_boundary:
    # Use precomputed slice (safe - boundary is complete)
    idx = daily_full_parallel.index.searchsorted(end_timestamp, side='right')
    result = daily_full_parallel.iloc[:idx]
    print(f"Using precomputed slice (boundary case)")
else:
    # Need to handle partial day - slice BEFORE the partial day, then re-resample the partial
    idx = daily_full_parallel.index.searchsorted(end_timestamp, side='left')
    complete_days = daily_full_parallel.iloc[:idx]

    # Re-resample the partial day from 5min data
    df_partial_day = df_sequential[df_sequential.index.date == end_timestamp.date()]
    partial_day_resampled = df_partial_day.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Combine complete days + partial day
    if len(complete_days) > 0:
        result = pd.concat([complete_days, partial_day_resampled])
    else:
        result = partial_day_resampled

    print(f"Using hybrid approach: {len(complete_days)} complete days + 1 partial day")

print(f"\nCorrected result:")
print(result)
print(f"\nCorrected daily 'close': {result['close'].iloc[-1]:.4f}")
print(f"Sequential daily 'close': {daily_sequential['close'].iloc[0]:.4f}")
print(f"Match? {abs(result['close'].iloc[-1] - daily_sequential['close'].iloc[0]) < 0.0001}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
1. Daily bars are labeled at MIDNIGHT (00:00) of each calendar day
2. A daily bar's 'close' is the last 5min bar from that day
3. If data ends mid-day, resample creates a PARTIAL daily bar with data up to that point
4. Precomputed full resample creates COMPLETE daily bars with all day's data
5. searchsorted(side='right') includes bars UP TO AND INCLUDING the search timestamp
6. LOOKAHEAD OCCURS when:
   - Parallel: precomputes full days, slices by mid-day timestamp
   - This includes a complete daily bar with future data (rest of the day)
   - Sequential: resamples up to position, creates partial day bar
   - The two approaches give DIFFERENT values for the same position

RECOMMENDATION:
- For walk-forward validation, MUST use sequential approach or smart hybrid
- Current parallel optimization is UNSAFE for mid-day positions
- Daily bar boundaries are safe, but intraday positions are problematic
""")
