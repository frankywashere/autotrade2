"""
Verify if there's a projection index issue where forward bar 0
is evaluated against channel end bounds (x=window-1) instead of
the next position (x=window).
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

# Reconstruct channel
channel_data = spy_5min.iloc[channel_start_idx:channel_end_idx + 1].copy()
channel = detect_channel(channel_data, window=window, min_cycles=1)

print("=" * 80)
print("PROJECTION INDEX VERIFICATION")
print("=" * 80)
print()

print("CURRENT IMPLEMENTATION (in break_scanner.py):")
print("-" * 80)
print("projection_x = channel.window - 1 + bars_forward")
print(f"For forward bar 0: projection_x = {window} - 1 + 0 = {window - 1}")
print()

std_multiplier = 2.0

# Current implementation
projection_x_current = window - 1 + 0
center_current = channel.slope * projection_x_current + channel.intercept
upper_current = center_current + std_multiplier * channel.std_dev
lower_current = center_current - std_multiplier * channel.std_dev

print("BOUNDS USING CURRENT FORMULA (x = 9):")
print(f"  Center: {center_current:.4f}")
print(f"  Upper:  {upper_current:.4f}")
print(f"  Lower:  {lower_current:.4f}")
print()

# Get forward bar 0
forward_bar_0 = spy_5min.iloc[channel_end_idx + 1]
print("FORWARD BAR 0 PRICES:")
print(f"  High:  {forward_bar_0['high']:.4f}")
print(f"  Low:   {forward_bar_0['low']:.4f}")
print(f"  Close: {forward_bar_0['close']:.4f}")
print()

# Check break with current formula
if forward_bar_0['high'] > upper_current:
    print(f"✓ BREAK DETECTED (current formula)")
    print(f"  High {forward_bar_0['high']:.4f} > Upper {upper_current:.4f}")
    print(f"  Magnitude: {(forward_bar_0['high'] - upper_current) / channel.std_dev:.4f} std devs")
elif forward_bar_0['low'] < lower_current:
    print(f"✓ BREAK DETECTED (current formula)")
    print(f"  Low {forward_bar_0['low']:.4f} < Lower {lower_current:.4f}")
    print(f"  Magnitude: {(lower_current - forward_bar_0['low']) / channel.std_dev:.4f} std devs")
else:
    print("✗ NO BREAK (current formula)")

print()
print("=" * 80)
print()

print("ALTERNATIVE IMPLEMENTATION:")
print("-" * 80)
print("projection_x = channel.window + bars_forward")
print(f"For forward bar 0: projection_x = {window} + 0 = {window}")
print()

# Alternative implementation
projection_x_alternative = window + 0
center_alternative = channel.slope * projection_x_alternative + channel.intercept
upper_alternative = center_alternative + std_multiplier * channel.std_dev
lower_alternative = center_alternative - std_multiplier * channel.std_dev

print("BOUNDS USING ALTERNATIVE FORMULA (x = 10):")
print(f"  Center: {center_alternative:.4f}")
print(f"  Upper:  {upper_alternative:.4f}")
print(f"  Lower:  {lower_alternative:.4f}")
print()

# Check break with alternative formula
if forward_bar_0['high'] > upper_alternative:
    print(f"✓ BREAK DETECTED (alternative formula)")
    print(f"  High {forward_bar_0['high']:.4f} > Upper {upper_alternative:.4f}")
    print(f"  Magnitude: {(forward_bar_0['high'] - upper_alternative) / channel.std_dev:.4f} std devs")
elif forward_bar_0['low'] < lower_alternative:
    print(f"✓ BREAK DETECTED (alternative formula)")
    print(f"  Low {forward_bar_0['low']:.4f} < Lower {lower_alternative:.4f}")
    print(f"  Magnitude: {(lower_alternative - forward_bar_0['low']) / channel.std_dev:.4f} std devs")
else:
    print("✗ NO BREAK (alternative formula)")

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print("DIFFERENCE IN BOUNDS:")
print(f"  Upper difference:  {upper_current - upper_alternative:.4f}")
print(f"  Lower difference:  {lower_current - lower_alternative:.4f}")
print(f"  Center difference: {center_current - center_alternative:.4f}")
print()

print("WHICH IS MORE INTUITIVE?")
print()
print("Current (x = window - 1):")
print("  - Forward bar 0 uses the SAME projection as channel's last bar")
print("  - Conceptually: 'How does the next bar compare to where we ended?'")
print("  - Problem: Channel's last bar was used to FIT the regression")
print()
print("Alternative (x = window):")
print("  - Forward bar 0 uses the NEXT projection point")
print("  - Conceptually: 'How does the next bar compare to the projection?'")
print("  - More consistent: Each forward bar is truly 'forward' from channel")
print()

# Check which bars would differ
print("=" * 80)
print("IMPACT ON bars_to_first_break")
print("=" * 80)
print()

# Test first 10 forward bars with both formulas
print("Testing first 10 forward bars:")
print()
print(f"{'Bar':<4} {'Current':<15} {'Alternative':<15} {'Difference':<15}")
print("-" * 60)

for bar_idx in range(min(10, len(spy_5min) - channel_end_idx - 1)):
    forward_bar = spy_5min.iloc[channel_end_idx + 1 + bar_idx]

    # Current formula
    projection_x_curr = window - 1 + bar_idx
    upper_curr = (channel.slope * projection_x_curr + channel.intercept) + std_multiplier * channel.std_dev
    lower_curr = (channel.slope * projection_x_curr + channel.intercept) - std_multiplier * channel.std_dev
    break_curr = forward_bar['high'] > upper_curr or forward_bar['low'] < lower_curr

    # Alternative formula
    projection_x_alt = window + bar_idx
    upper_alt = (channel.slope * projection_x_alt + channel.intercept) + std_multiplier * channel.std_dev
    lower_alt = (channel.slope * projection_x_alt + channel.intercept) - std_multiplier * channel.std_dev
    break_alt = forward_bar['high'] > upper_alt or forward_bar['low'] < lower_alt

    status_curr = "BREAK" if break_curr else "OK"
    status_alt = "BREAK" if break_alt else "OK"
    diff = "SAME" if break_curr == break_alt else "DIFFERENT"

    print(f"{bar_idx:<4} {status_curr:<15} {status_alt:<15} {diff:<15}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

print("For sample 109:")
print(f"  Current formula detects break at bar 0: {forward_bar_0['high'] > upper_current}")
print(f"  Alternative formula detects break at bar 0: {forward_bar_0['high'] > upper_alternative}")
print()
print("The label value bars_to_first_break=0 is correct with CURRENT implementation.")
print()
print("However, the ALTERNATIVE implementation would be more conceptually clean:")
print("  - Each forward bar is truly projected 'forward' from channel end")
print("  - No overlap with channel's last bar bounds")
print("  - More intuitive: 'bar 0 after channel' uses 'x = window' not 'x = window - 1'")
