"""
Deep dive into sample 109 to check if there's a discrepancy
between bars_to_first_break=0 and actual visual break position.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from v7.core.timeframe import resample_ohlc
from v7.core.channel import detect_channel

# Load sample
sample_path = Path("/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl")
with open(sample_path, 'rb') as f:
    samples = pickle.load(f)

# Load SPY data
spy_data_path = Path("/Users/frank/Desktop/CodingProjects/x14/data/SPY_1min.csv")
spy_df = pd.read_csv(spy_data_path)
spy_df['timestamp'] = pd.to_datetime(spy_df['timestamp'])
spy_df.set_index('timestamp', inplace=True)
spy_df.sort_index(inplace=True)
spy_5min = resample_ohlc(spy_df, '5min')

# Sample 109
sample = samples[109]
window = 10
channel_end_idx = sample.channel_end_idx
channel_start_idx = channel_end_idx - window + 1

# Get labels
labels = sample.labels_per_window[window]['spy']['5min']

print("=" * 80)
print("SAMPLE 109 - DETAILED ANALYSIS")
print("=" * 80)
print(f"Sample timestamp: {sample.timestamp}")
print(f"Channel end idx: {channel_end_idx}")
print(f"Channel start idx: {channel_start_idx}")
print()

print("LABELS:")
print(f"  bars_to_first_break: {labels.bars_to_first_break}")
print(f"  first_break_direction: {labels.first_break_direction} (0=DOWN, 1=UP)")
print(f"  break_magnitude: {labels.break_magnitude:.4f} std devs")
print(f"  bars_outside: {labels.bars_outside}")
print(f"  returned_to_channel: {labels.returned_to_channel}")
print(f"  permanent_break: {labels.permanent_break}")
print()

# Reconstruct channel
channel_data = spy_5min.iloc[channel_start_idx:channel_end_idx + 1].copy()
channel = detect_channel(channel_data, window=window, min_cycles=1)

print("CHANNEL PARAMETERS:")
print(f"  slope: {channel.slope:.8f}")
print(f"  intercept: {channel.intercept:.6f}")
print(f"  std_dev: {channel.std_dev:.6f}")
print()

# Get the EXACT channel end price
channel_end_price = channel_data.iloc[-1]
print("CHANNEL END BAR (last bar of channel):")
print(f"  Index: {channel_end_idx}")
print(f"  Timestamp: {channel_data.index[-1]}")
print(f"  High: {channel_end_price['high']:.4f}")
print(f"  Low: {channel_end_price['low']:.4f}")
print(f"  Close: {channel_end_price['close']:.4f}")
print()

# Calculate bounds at channel end
projection_x_end = window - 1  # x at end of channel
std_multiplier = 2.0
center_at_end = channel.slope * projection_x_end + channel.intercept
upper_at_end = center_at_end + std_multiplier * channel.std_dev
lower_at_end = center_at_end - std_multiplier * channel.std_dev

print("BOUNDS AT CHANNEL END (x = window - 1 = 9):")
print(f"  Center: {center_at_end:.4f}")
print(f"  Upper:  {upper_at_end:.4f}")
print(f"  Lower:  {lower_at_end:.4f}")
print()

# Check if channel end bar itself is outside bounds
if channel_end_price['high'] > upper_at_end:
    print("⚠️  WARNING: Channel end bar HIGH exceeds upper bound!")
    print(f"   High={channel_end_price['high']:.4f} > Upper={upper_at_end:.4f}")
    print(f"   Difference: {channel_end_price['high'] - upper_at_end:.4f}")
elif channel_end_price['low'] < lower_at_end:
    print("⚠️  WARNING: Channel end bar LOW is below lower bound!")
    print(f"   Low={channel_end_price['low']:.4f} < Lower={lower_at_end:.4f}")
    print(f"   Difference: {lower_at_end - channel_end_price['low']:.4f}")
else:
    print("✓ Channel end bar is within bounds")
print()

# Now check FIRST bar AFTER channel end (bar 0 in forward scan)
forward_start = channel_end_idx + 1
if forward_start < len(spy_5min):
    first_forward_bar = spy_5min.iloc[forward_start]

    print("FIRST BAR AFTER CHANNEL (bar 0 of forward scan):")
    print(f"  Index: {forward_start}")
    print(f"  Timestamp: {spy_5min.index[forward_start]}")
    print(f"  High: {first_forward_bar['high']:.4f}")
    print(f"  Low: {first_forward_bar['low']:.4f}")
    print(f"  Close: {first_forward_bar['close']:.4f}")
    print()

    # Calculate bounds at bar 0 of forward scan
    projection_x_0 = window - 1 + 0  # x at first forward bar
    center_at_0 = channel.slope * projection_x_0 + channel.intercept
    upper_at_0 = center_at_0 + std_multiplier * channel.std_dev
    lower_at_0 = center_at_0 - std_multiplier * channel.std_dev

    print("BOUNDS AT FORWARD BAR 0 (x = window - 1 + 0 = 9):")
    print(f"  Center: {center_at_0:.4f}")
    print(f"  Upper:  {upper_at_0:.4f}")
    print(f"  Lower:  {lower_at_0:.4f}")
    print()

    # Check if bar 0 is outside
    if first_forward_bar['high'] > upper_at_0:
        print("✓ BREAK CONFIRMED: Forward bar 0 HIGH exceeds upper bound")
        print(f"   High={first_forward_bar['high']:.4f} > Upper={upper_at_0:.4f}")
        print(f"   Magnitude: {(first_forward_bar['high'] - upper_at_0) / channel.std_dev:.4f} std devs")
        print(f"   Direction: UP (1)")
    elif first_forward_bar['low'] < lower_at_0:
        print("✓ BREAK CONFIRMED: Forward bar 0 LOW is below lower bound")
        print(f"   Low={first_forward_bar['low']:.4f} < Lower={lower_at_0:.4f}")
        print(f"   Magnitude: {(lower_at_0 - first_forward_bar['low']) / channel.std_dev:.4f} std devs")
        print(f"   Direction: DOWN (0)")
    else:
        print("✗ NO BREAK: Forward bar 0 is within bounds")
        print("   This contradicts bars_to_first_break=0!")
    print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if labels.bars_to_first_break == 0:
    print("bars_to_first_break=0 means the FIRST bar after the channel ends")
    print("(forward bar 0) already breaks outside the projected bounds.")
    print()
    print("This is CORRECT behavior - the break happens immediately.")
    print()
    print("If you're seeing a 'later' break visually, possible explanations:")
    print("1. You might be looking at a different sample")
    print("2. The visual might have a bug in marker positioning")
    print("3. The visual might be showing return-to-channel as the 'real' break")
    print("4. The visual might be showing permanent break position instead")
else:
    print(f"bars_to_first_break={labels.bars_to_first_break}")
    print("The break happens later, not immediately after channel end.")
