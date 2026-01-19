#!/usr/bin/env python3
"""
Check if break markers are being drawn based on the condition.
"""

import pickle
from pathlib import Path

# Load the samples
cache_path = Path("small_sample.pkl")
with open(cache_path, 'rb') as f:
    samples = pickle.load(f)

print("ANALYSIS: Break Marker Drawing Logic")
print("=" * 70)
print("\nThe code at dual_inspector.py line 738 says:")
print("  if labels.permanent_break and labels.break_scan_valid:")
print("      # Draw break marker")
print("\nSo break markers are ONLY drawn when BOTH conditions are True.")
print("\n" + "=" * 70)

# Check first 10 samples
window = 50
print(f"\nChecking first 10 samples with window={window}:")
print("=" * 70)

for idx in range(min(10, len(samples))):
    sample = samples[idx]
    if window not in sample.labels_per_window:
        continue

    window_data = sample.labels_per_window[window]
    if 'tsla' not in window_data or '5min' not in window_data['tsla']:
        continue

    labels = window_data['tsla']['5min']

    # Check if break marker would be drawn
    will_draw_marker = labels.permanent_break and labels.break_scan_valid

    print(f"\nSample {idx}:")
    print(f"  permanent_break: {labels.permanent_break}")
    print(f"  break_scan_valid: {labels.break_scan_valid}")
    print(f"  bars_to_first_break: {labels.bars_to_first_break}")
    print(f"  → Will draw break marker? {will_draw_marker}")

    if not will_draw_marker and labels.break_scan_valid and labels.bars_to_first_break > 0:
        print(f"  ⚠️  WARNING: Break detected at {labels.bars_to_first_break} bars but marker won't be drawn!")
        print(f"     (x-axis will extend but marker is missing)")

print("\n" + "=" * 70)
print("CHECKING SAMPLE 100 (from screenshot)")
print("=" * 70)

sample = samples[100]
if window in sample.labels_per_window:
    window_data = sample.labels_per_window[window]
    if 'tsla' in window_data and '5min' in window_data['tsla']:
        labels = window_data['tsla']['5min']

        print(f"\nSample 100 - TSLA 5min:")
        print(f"  permanent_break: {labels.permanent_break}")
        print(f"  break_scan_valid: {labels.break_scan_valid}")
        print(f"  bars_to_first_break: {labels.bars_to_first_break}")

        will_draw = labels.permanent_break and labels.break_scan_valid

        print(f"\n  Condition check:")
        print(f"    permanent_break AND break_scan_valid")
        print(f"    = {labels.permanent_break} AND {labels.break_scan_valid}")
        print(f"    = {will_draw}")

        if will_draw:
            print(f"\n  ✓ Break marker WILL be drawn at x={window - 1 + labels.bars_to_first_break}")
        else:
            print(f"\n  ✗ Break marker will NOT be drawn")
            print(f"     But x-axis will extend to show break at x={window - 1 + labels.bars_to_first_break}")
            print(f"\n  This explains the screenshot:")
            print(f"    - Text shows 'Bars to 1st Break: 58'")
            print(f"    - X-axis extends to ~112 to accommodate the break")
            print(f"    - But no vertical line/arrow is visible at x=107")
            print(f"    - Because permanent_break=False, marker is not drawn")

print("\n" + "=" * 70)
print("SUMMARY OF THE BUG")
print("=" * 70)
print("""
The issue is a logic inconsistency in dual_inspector.py:

1. X-axis calculation (lines 774-777):
   - Uses: if break_scan_valid and bars_to_first_break > 0
   - Result: X-axis extends to accommodate the break

2. Break marker drawing (line 738):
   - Uses: if permanent_break AND break_scan_valid
   - Result: Marker only drawn if break is permanent

This causes:
- X-axis extends to show breaks (even non-permanent ones)
- But break markers are only drawn for permanent breaks
- Non-permanent breaks have extended axis but no visual marker

Possible solutions:
A) Draw markers for all breaks (remove permanent_break check)
B) Only extend x-axis for permanent breaks (add permanent_break check)
C) Use different marker style for non-permanent breaks
""")
