#!/usr/bin/env python3
"""
Debug script to visualize the timestamp issue with channel data.
Shows whether channel end timestamps are before or after the sample timestamp.
"""

import pickle
from datetime import datetime, timezone
import pandas as pd

# Load the test data
with open('/tmp/test_timestamps.pkl', 'rb') as f:
    samples = pickle.load(f)

sample = samples[0]  # Get sample 1

print("=" * 80)
print("TIMESTAMP ANALYSIS - Sample 1")
print("=" * 80)

# Sample timestamp - handle both int ms and pandas Timestamp
sample_ts_raw = sample.timestamp
if isinstance(sample_ts_raw, pd.Timestamp):
    sample_ts = int(sample_ts_raw.timestamp() * 1000)  # Convert to ms
    sample_dt = sample_ts_raw.to_pydatetime()
else:
    sample_ts = sample_ts_raw
    sample_dt = datetime.fromtimestamp(sample_ts / 1000, tz=timezone.utc)

print(f"\nSample Timestamp (when we're 'standing' making prediction):")
print(f"  Timestamp: {sample_ts}")
print(f"  DateTime:  {sample_dt}")

# Helper to convert timestamps
def to_ms(ts):
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp() * 1000)
    return ts

def to_dt(ts):
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

# Get labels for the best window (or first available)
best_window = sample.best_window
print(f"\nBest Window: {best_window}")

# Get labels for this window
labels_dict = sample.labels_per_window.get(best_window, {})

# Look at TSLA labels
tsla_labels = labels_dict.get('tsla', {})

print("\n" + "-" * 80)
print("TIMEFRAME ANALYSIS FROM LABELS")
print("-" * 80)

# Check available timeframes
if isinstance(tsla_labels, dict):
    timeframes = list(tsla_labels.keys())
else:
    timeframes = []

print(f"\nAvailable timeframes in labels: {timeframes}")

# Analyze each timeframe
for tf in sorted(timeframes):
    tf_labels = tsla_labels[tf]

    # Check if it has channel timestamps
    if not hasattr(tf_labels, 'source_channel_start_ts'):
        continue

    ch_start_ts_raw = tf_labels.source_channel_start_ts
    ch_end_ts_raw = tf_labels.source_channel_end_ts

    if ch_start_ts_raw is None or ch_end_ts_raw is None:
        print(f"\n{tf}: No channel timestamps available")
        continue

    ch_start = to_ms(ch_start_ts_raw)
    ch_end = to_ms(ch_end_ts_raw)
    ch_start_dt = to_dt(ch_start_ts_raw)
    ch_end_dt = to_dt(ch_end_ts_raw)

    print(f"\n{'=' * 80}")
    print(f"TIMEFRAME: {tf}")
    print(f"{'=' * 80}")

    print(f"\nChannel Metadata:")
    print(f"  source_channel_start_ts: {ch_start}")
    print(f"  source_channel_start:    {ch_start_dt}")
    print(f"  source_channel_end_ts:   {ch_end}")
    print(f"  source_channel_end:      {ch_end_dt}")
    print(f"  window size:             {best_window}")

    # Calculate time spans
    channel_duration_ms = ch_end - ch_start

    # Calculate bars based on timeframe
    tf_minutes = {
        '1min': 1, '5min': 5, '15min': 15, '30min': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }

    tf_min = tf_minutes.get(tf, 5)
    channel_duration_minutes = channel_duration_ms / (1000 * 60)
    channel_bars = channel_duration_minutes / tf_min

    print(f"\nChannel Duration:")
    print(f"  Duration (minutes): {channel_duration_minutes:.1f}")
    print(f"  Number of bars:     {channel_bars:.1f} (for {tf} timeframe)")
    print(f"  Expected bars:      {best_window} (from window parameter)")

    # Check relationship
    end_vs_sample = ch_end - sample_ts
    end_vs_sample_minutes = end_vs_sample / (1000 * 60)

    print(f"\n" + "=" * 80)
    print(f"CRITICAL CHECK: Is channel_end BEFORE or AFTER sample.timestamp?")
    print("=" * 80)

    if ch_end > sample_ts:
        print(f"\n  !!! PROBLEM: channel_end is AFTER sample.timestamp !!!")
        print(f"  Difference: {end_vs_sample_minutes:.1f} minutes INTO THE FUTURE")
        future_leak = True
    elif ch_end == sample_ts:
        print(f"\n  channel_end EQUALS sample.timestamp (edge case)")
        future_leak = False
    else:
        print(f"\n  OK: channel_end is BEFORE sample.timestamp")
        print(f"  Difference: {abs(end_vs_sample_minutes):.1f} minutes in the past")
        future_leak = False

    # ASCII Timeline
    print(f"\n" + "-" * 80)
    print("ASCII TIMELINE VISUALIZATION")
    print("-" * 80)

    # Normalize timestamps to create a visual scale
    min_ts = ch_start
    max_ts = max(ch_end, sample_ts)
    span = max_ts - min_ts

    # Scale to 60 characters
    scale = 60

    def pos(ts):
        return int((ts - min_ts) / span * scale) if span > 0 else 0

    start_pos = pos(ch_start)
    end_pos = pos(ch_end)
    sample_pos = pos(sample_ts)

    # Build timeline
    timeline = ['-'] * (scale + 1)

    # Mark positions
    timeline[start_pos] = 'S'  # Start

    # Mark sample position
    if abs(sample_pos - end_pos) <= 1:
        # They're very close, mark with special char
        if ch_end > sample_ts:
            timeline[sample_pos] = 'T'
            if end_pos != sample_pos:
                timeline[end_pos] = 'E'
        else:
            timeline[end_pos] = 'E'
            if sample_pos != end_pos:
                timeline[sample_pos] = 'T'
    else:
        timeline[end_pos] = 'E'
        timeline[sample_pos] = 'T'

    print(f"\n  Legend: S=channel_start, E=channel_end, T=sample.timestamp (prediction point)")
    print()
    print(f"  |{''.join(timeline)}|")

    # Show pointer to "now"
    pointer_line = [' '] * (scale + 3)
    pointer_line[sample_pos + 1] = '^'

    print(f"  {''.join(pointer_line)}")
    print(f"  {' ' * (sample_pos)}we are HERE (making prediction)")

    if future_leak:
        print(f"\n  *** The channel data extends {end_vs_sample_minutes:.1f} minutes")
        print(f"      PAST the point where we're making predictions! ***")
        print(f"\n  This means the model sees FUTURE price action when making predictions!")

    print()
    print(f"  Timeline:")
    print(f"    Channel Start: {ch_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Sample (NOW):  {sample_dt.strftime('%Y-%m-%d %H:%M:%S')} <-- prediction point")
    print(f"    Channel End:   {ch_end_dt.strftime('%Y-%m-%d %H:%M:%S')}", end="")
    if future_leak:
        print(f" <-- FUTURE DATA!")
    else:
        print()

print("\n" + "=" * 80)
print("SUMMARY TABLE: ALL TIMEFRAMES")
print("=" * 80)

print(f"\n{'Timeframe':<10} {'Ch Start':<20} {'Ch End':<20} {'Sample TS':<20} {'Status'}")
print("-" * 90)

for tf in sorted(timeframes):
    tf_labels = tsla_labels[tf]

    if not hasattr(tf_labels, 'source_channel_start_ts'):
        continue

    ch_start_ts_raw = tf_labels.source_channel_start_ts
    ch_end_ts_raw = tf_labels.source_channel_end_ts

    if ch_start_ts_raw is None or ch_end_ts_raw is None:
        print(f"{tf:<10} {'N/A':<20} {'N/A':<20} {'N/A':<20} MISSING")
        continue

    ch_start = to_ms(ch_start_ts_raw)
    ch_end = to_ms(ch_end_ts_raw)
    ch_start_dt = to_dt(ch_start_ts_raw)
    ch_end_dt = to_dt(ch_end_ts_raw)

    diff_ms = ch_end - sample_ts
    diff_minutes = diff_ms / (1000 * 60)

    if diff_ms > 0:
        status = f"FUTURE LEAK! (+{diff_minutes:.0f}m)"
    elif diff_ms == 0:
        status = "EXACT"
    else:
        status = f"OK ({diff_minutes:.0f}m)"

    print(f"{tf:<10} {ch_start_dt.strftime('%H:%M:%S'):<20} {ch_end_dt.strftime('%H:%M:%S'):<20} {sample_dt.strftime('%H:%M:%S'):<20} {status}")

print("\n" + "=" * 80)
