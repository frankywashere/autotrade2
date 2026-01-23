"""
Debug script to investigate bars_to_first_break discrepancy.

Load sample 109 from small_sample.pkl and check:
1. Window 10, SPY 5min labels - what is bars_to_first_break?
2. What is the channel upper bound at bar 10?
3. Get the actual SPY price data for bars 10-25
4. Manually check which bar first exceeded the upper bound
5. Compare manual calculation vs what the label says
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load the sample
sample_path = Path("/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl")
with open(sample_path, 'rb') as f:
    samples = pickle.load(f)

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

# Look at window 10, SPY, 5min
window = 10
asset = 'spy'
tf = '5min'

print("=" * 80)
print(f"ANALYZING: Window={window}, Asset={asset.upper()}, TF={tf}")
print("=" * 80)
print()

# Get labels
if window in sample.labels_per_window:
    window_data = sample.labels_per_window[window]
    if asset in window_data:
        asset_data = window_data[asset]
        if tf in asset_data:
            labels = asset_data[tf]

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

            # For SPY, also check the spy_ prefixed fields
            print("  spy_bars_to_first_break: {labels.spy_bars_to_first_break}")
            print(f"  spy_first_break_direction: {labels.spy_first_break_direction}")
            print(f"  spy_break_magnitude: {labels.spy_break_magnitude:.4f} std devs")
            print()
        else:
            print(f"ERROR: TF '{tf}' not found in asset data")
            print(f"Available TFs: {list(asset_data.keys())}")
    else:
        print(f"ERROR: Asset '{asset}' not found in window data")
        print(f"Available assets: {list(window_data.keys())}")
else:
    print(f"ERROR: Window {window} not found in labels_per_window")
    print(f"Available windows: {list(sample.labels_per_window.keys())}")

print()
print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print()
print("We need to:")
print("1. Load the actual SPY 5min price data")
print("2. Extract the channel from the sample")
print("3. Calculate the upper/lower bounds")
print("4. Manually check which bar first exceeded the bounds")
print("5. Compare with what the label says")
print()
print("This requires access to the original price data and channel detection.")
print("Let me check if we have the raw data available...")
