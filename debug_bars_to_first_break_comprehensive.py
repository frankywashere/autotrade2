"""
Comprehensive debug script to investigate bars_to_first_break discrepancy.

Load sample 109 from small_sample.pkl and:
1. Load the actual SPY 5min price data
2. Extract the channel bounds
3. Manually check which bar first exceeded the upper/lower bounds
4. Compare manual calculation vs what the label says
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from v7.core.timeframe import resample_ohlc
from v7.core.channel import detect_channel

# Load the sample
sample_path = Path("/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl")
with open(sample_path, 'rb') as f:
    samples = pickle.load(f)

# Load SPY 1min data and resample to 5min
spy_data_path = Path("/Users/frank/Desktop/CodingProjects/x14/data/SPY_1min.csv")
spy_df = pd.read_csv(spy_data_path)
spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'])
spy_df.set_index('timestamp', inplace=True)
spy_df.sort_index(inplace=True)

# Resample to 5min
spy_5min = resample_ohlc(spy_df, '5min')

print("=" * 80)
print("SPY DATA LOADED")
print("=" * 80)
print(f"SPY 5min data shape: {spy_5min.shape}")
print(f"SPY 5min date range: {spy_5min.index[0]} to {spy_5min.index[-1]}")
print()

# Get sample 109
sample_idx = 109
sample = samples[sample_idx]

print("=" * 80)
print(f"DEBUGGING SAMPLE {sample_idx}")
print("=" * 80)
print()
print(f"Timestamp: {sample.timestamp}")
print(f"Channel end idx (5min): {sample.channel_end_idx}")
print()

# Configuration
window = 10
asset = 'spy'
tf = '5min'

print("=" * 80)
print(f"ANALYZING: Window={window}, Asset={asset.upper()}, TF={tf}")
print("=" * 80)
print()

# Get labels from sample
labels = None
if window in sample.labels_per_window:
    window_data = sample.labels_per_window[window]
    if asset in window_data:
        asset_data = window_data[asset]
        if tf in asset_data:
            labels = asset_data[tf]

if labels is None:
    print("ERROR: Could not find labels for this configuration")
    sys.exit(1)

print("LABELS FROM SAMPLE:")
print("-" * 80)
print(f"  bars_to_first_break: {labels.bars_to_first_break}")
print(f"  first_break_direction: {labels.first_break_direction} (0=DOWN, 1=UP)")
print(f"  break_magnitude: {labels.break_magnitude:.4f} std devs")
print(f"  bars_outside: {labels.bars_outside}")
print(f"  returned_to_channel: {labels.returned_to_channel}")
print(f"  permanent_break: {labels.permanent_break}")
print(f"  break_scan_valid: {labels.break_scan_valid}")
print()

# Now reconstruct the channel at this position
channel_end_idx = sample.channel_end_idx
channel_start_idx = channel_end_idx - window + 1

print("=" * 80)
print("RECONSTRUCTING CHANNEL")
print("=" * 80)
print(f"Channel start idx: {channel_start_idx}")
print(f"Channel end idx: {channel_end_idx}")
print(f"Window size: {window}")
print()

# Extract the channel data
channel_data = spy_5min.iloc[channel_start_idx:channel_end_idx + 1].copy()
print(f"Channel data shape: {channel_data.shape}")
print(f"Channel timestamp range: {channel_data.index[0]} to {channel_data.index[-1]}")
print()

# Detect the channel using the same algorithm
try:
    channel = detect_channel(channel_data, window=window, min_cycles=1)
    print(f"Channel detected: valid={channel.valid}, direction={channel.direction}")

    if channel.valid:
        print(f"Channel slope: {channel.slope:.6f}")
        print(f"Channel intercept: {channel.intercept:.6f}")
        print(f"Channel std_dev: {channel.std_dev:.6f}")
        print()
    else:
        print("ERROR: Channel is not valid!")
        sys.exit(1)
except Exception as e:
    print(f"ERROR detecting channel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now scan forward from the channel end
print("=" * 80)
print("MANUAL FORWARD SCAN")
print("=" * 80)
print()

# Get forward data (next 30 bars for inspection)
forward_start = channel_end_idx + 1
forward_end = min(channel_end_idx + 31, len(spy_5min))
forward_data = spy_5min.iloc[forward_start:forward_end].copy()

print(f"Forward scan range: idx {forward_start} to {forward_end - 1}")
print(f"Forward data shape: {forward_data.shape}")
print()

# Calculate projected bounds for each forward bar
print("BAR-BY-BAR ANALYSIS:")
print("-" * 80)
print(f"{'Bar':<4} {'Idx':<6} {'Timestamp':<20} {'High':<10} {'Low':<10} {'Close':<10} {'Upper':<10} {'Lower':<10} {'Status':<20}")
print("-" * 80)

std_multiplier = 2.0
first_break_bar = None
first_break_direction = None

for i, (ts, row) in enumerate(forward_data.iterrows()):
    # Calculate projection: x at end of channel = window - 1
    # x at this forward bar = (window - 1) + i
    projection_x = channel.window - 1 + i

    # Projected center
    projected_center = channel.slope * projection_x + channel.intercept

    # Bounds
    projected_upper = projected_center + std_multiplier * channel.std_dev
    projected_lower = projected_center - std_multiplier * channel.std_dev

    # Check if price is outside bounds
    high = row['high']
    low = row['low']
    close = row['close']

    status = "INSIDE"
    if high > projected_upper:
        status = "BREAK UP"
        if first_break_bar is None:
            first_break_bar = i
            first_break_direction = 1
    elif low < projected_lower:
        status = "BREAK DOWN"
        if first_break_bar is None:
            first_break_bar = i
            first_break_direction = 0

    # Print bar details
    actual_idx = forward_start + i
    print(f"{i:<4} {actual_idx:<6} {str(ts):<20} {high:<10.4f} {low:<10.4f} {close:<10.4f} {projected_upper:<10.4f} {projected_lower:<10.4f} {status:<20}")

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print(f"MANUAL CALCULATION:")
print(f"  First break at bar: {first_break_bar if first_break_bar is not None else 'NO BREAK'}")
print(f"  First break direction: {first_break_direction if first_break_direction is not None else 'N/A'} (0=DOWN, 1=UP)")
print()

print(f"LABEL VALUES:")
print(f"  bars_to_first_break: {labels.bars_to_first_break}")
print(f"  first_break_direction: {labels.first_break_direction} (0=DOWN, 1=UP)")
print()

if first_break_bar is not None:
    if first_break_bar == labels.bars_to_first_break:
        print("✓ MATCH: bars_to_first_break matches manual calculation")
    else:
        print(f"✗ DISCREPANCY: Manual={first_break_bar}, Label={labels.bars_to_first_break}")
        print(f"   Difference: {abs(first_break_bar - labels.bars_to_first_break)} bars")

    if first_break_direction == labels.first_break_direction:
        print("✓ MATCH: first_break_direction matches manual calculation")
    else:
        print(f"✗ DISCREPANCY: Manual direction={first_break_direction}, Label direction={labels.first_break_direction}")
else:
    print("✗ NO BREAK DETECTED in manual scan (but label shows a break)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
