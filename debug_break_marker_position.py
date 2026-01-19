#!/usr/bin/env python3
"""
Debug script to check break marker positions and x-axis limits.
"""

import pickle
from pathlib import Path
from v15.dtypes import STANDARD_WINDOWS

# Load the samples
cache_path = Path("small_sample.pkl")
with open(cache_path, 'rb') as f:
    samples = pickle.load(f)

# Check first sample
sample = samples[0]
window = 50  # From screenshot

print(f"Sample timestamp: {sample.timestamp}")
print(f"Window: {window}")
print()

# Check TSLA labels for each timeframe
for tf in ['5min', '1h', 'daily']:
    if window in sample.labels_per_window:
        window_data = sample.labels_per_window[window]
        if 'tsla' in window_data and tf in window_data['tsla']:
            labels = window_data['tsla'][tf]

            print(f"\n{'='*60}")
            print(f"TSLA - {tf}")
            print(f"{'='*60}")
            print(f"  permanent_break: {labels.permanent_break}")
            print(f"  break_scan_valid: {labels.break_scan_valid}")
            print(f"  bars_to_first_break: {labels.bars_to_first_break}")
            print(f"  break_direction: {labels.break_direction}")

            if labels.permanent_break and labels.break_scan_valid:
                # Calculate break position as it's done in dual_inspector.py line 740
                break_bar = window - 1 + labels.bars_to_first_break
                print(f"\n  CALCULATED BREAK POSITION:")
                print(f"    window - 1 = {window - 1}")
                print(f"    + bars_to_first_break = {labels.bars_to_first_break}")
                print(f"    = break_bar = {break_bar}")

                # Calculate projection as done in dual_inspector.py line 726-729
                project_forward = labels.bars_to_first_break + 5
                print(f"\n  PROJECTION DISTANCE:")
                print(f"    bars_to_first_break + 5 = {project_forward}")

                # Calculate max_x as done in dual_inspector.py line 775
                max_x = window - 1 + labels.bars_to_first_break + 5
                print(f"\n  X-AXIS LIMITS:")
                print(f"    window - 1 + bars_to_first_break + 5")
                print(f"    = {window - 1} + {labels.bars_to_first_break} + 5")
                print(f"    = max_x = {max_x}")
                print(f"    xlim will be set to (-0.5, {max_x + 0.5})")

                # Check if break_bar is within visible range
                if break_bar < max_x:
                    print(f"\n  ✓ Break marker at x={break_bar} SHOULD BE VISIBLE (within [0, {max_x}])")
                else:
                    print(f"\n  ✗ Break marker at x={break_bar} IS OFF-SCREEN (beyond {max_x})")

# Check SPY labels too
print("\n" + "="*60)
print("SPY LABELS")
print("="*60)
if window in sample.labels_per_window:
    window_data = sample.labels_per_window[window]
    if 'spy' in window_data:
        for tf in ['5min', '1h', 'daily']:
            if tf in window_data['spy']:
                labels = window_data['spy'][tf]
                print(f"\n{tf}:")
                print(f"  permanent_break: {labels.permanent_break}")
                print(f"  break_scan_valid: {labels.break_scan_valid}")
                print(f"  bars_to_first_break: {labels.bars_to_first_break}")

                if labels.permanent_break and labels.break_scan_valid:
                    break_bar = window - 1 + labels.bars_to_first_break
                    print(f"  break_bar position: {break_bar}")
