#!/usr/bin/env python3
"""
Detailed analysis of a specific sample to understand break marker positioning.
"""

import pickle
from pathlib import Path

# Load the samples
cache_path = Path("small_sample.pkl")
with open(cache_path, 'rb') as f:
    samples = pickle.load(f)

# Use sample 100 which has bars_to_first_break = 58
sample = samples[100]
window = 50

print(f"Sample 100 - {sample.timestamp}")
print(f"Window: {window}")
print("=" * 70)

if window in sample.labels_per_window:
    window_data = sample.labels_per_window[window]
    if 'tsla' in window_data and '5min' in window_data['tsla']:
        labels = window_data['tsla']['5min']

        print("\nTSLA 5min Labels:")
        print(f"  permanent_break: {labels.permanent_break}")
        print(f"  break_scan_valid: {labels.break_scan_valid}")
        print(f"  bars_to_first_break: {labels.bars_to_first_break}")
        print(f"  break_direction: {labels.break_direction}")
        print(f"  break_magnitude: {labels.break_magnitude}")
        print(f"  duration_bars: {labels.duration_bars}")

        print("\n" + "=" * 70)
        print("VISUALIZATION CALCULATIONS")
        print("=" * 70)

        # From dual_inspector.py line 740
        break_bar = window - 1 + labels.bars_to_first_break
        print(f"\n1. Break Marker Position (line 740):")
        print(f"   break_bar = window - 1 + labels.bars_to_first_break")
        print(f"   break_bar = {window} - 1 + {labels.bars_to_first_break}")
        print(f"   break_bar = {break_bar}")

        # From dual_inspector.py line 726-729
        if labels.break_scan_valid and labels.bars_to_first_break > 0:
            project_forward = labels.bars_to_first_break + 5
        else:
            project_forward = 20
        print(f"\n2. Forward Projection Distance (lines 726-729):")
        print(f"   if break_scan_valid and bars_to_first_break > 0:")
        print(f"       project_forward = bars_to_first_break + 5")
        print(f"       project_forward = {labels.bars_to_first_break} + 5 = {project_forward}")

        # Channel plotting range (line 734)
        channel_start = 0
        channel_end = window
        print(f"\n3. Channel Plotting Range (line 734):")
        print(f"   plot_channel_bounds(ax, channel, start=0, length={window}, project_forward={project_forward})")
        print(f"   - Channel line spans: x = [0, {window-1}]")
        print(f"   - Projection spans: x = [{window}, {window + project_forward - 1}]")
        print(f"   - Total range: x = [0, {window + project_forward - 1}]")

        # X-axis limits (line 775-780)
        if labels.break_scan_valid and labels.bars_to_first_break > 0:
            max_x = window - 1 + labels.bars_to_first_break + 5
        else:
            max_x = window - 1 + 20
        print(f"\n4. X-Axis Limits (lines 775-780):")
        print(f"   max_x = window - 1 + bars_to_first_break + 5")
        print(f"   max_x = {window} - 1 + {labels.bars_to_first_break} + 5")
        print(f"   max_x = {max_x}")
        print(f"   ax.set_xlim(-0.5, {max_x + 0.5})")

        print("\n" + "=" * 70)
        print("VISIBILITY CHECK")
        print("=" * 70)

        print(f"\nChannel window ends at: x = {window - 1}")
        print(f"Projection ends at: x = {window + project_forward - 1}")
        print(f"Break marker is at: x = {break_bar}")
        print(f"X-axis visible range: x = [-0.5, {max_x + 0.5}]")

        print(f"\nIs break marker visible?")
        if -0.5 <= break_bar <= max_x + 0.5:
            print(f"  ✓ YES - break_bar ({break_bar}) is within xlim [-0.5, {max_x + 0.5}]")
        else:
            print(f"  ✗ NO - break_bar ({break_bar}) is outside xlim [-0.5, {max_x + 0.5}]")

        # Double-check the math
        print("\n" + "=" * 70)
        print("PROBLEM ANALYSIS")
        print("=" * 70)

        print(f"\nGiven:")
        print(f"  - Window size: {window} bars")
        print(f"  - Channel spans bars 0 to {window-1}")
        print(f"  - Break occurs at bars_to_first_break = {labels.bars_to_first_break} bars AFTER channel end")
        print(f"  - So break is at absolute position: {window - 1} + {labels.bars_to_first_break} = {break_bar}")

        print(f"\nProjection:")
        print(f"  - project_forward = {project_forward} bars")
        print(f"  - Projection starts at x = {window}")
        print(f"  - Projection ends at x = {window + project_forward - 1}")

        print(f"\nDoes projection reach the break?")
        if window + project_forward - 1 >= break_bar:
            print(f"  ✓ YES - projection ends at {window + project_forward - 1}, break at {break_bar}")
        else:
            gap = break_bar - (window + project_forward - 1)
            print(f"  ✗ NO - projection ends at {window + project_forward - 1}, but break is at {break_bar}")
            print(f"  Gap: {gap} bars")

        print(f"\nDoes x-axis reach the break?")
        if max_x >= break_bar:
            print(f"  ✓ YES - x-axis extends to {max_x}, break at {break_bar}")
        else:
            gap = break_bar - max_x
            print(f"  ✗ NO - x-axis extends to {max_x}, but break is at {break_bar}")
            print(f"  Gap: {gap} bars")
