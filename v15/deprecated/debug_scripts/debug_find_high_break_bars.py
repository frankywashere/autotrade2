#!/usr/bin/env python3
"""
Find samples with high bars_to_first_break values.
"""

import pickle
from pathlib import Path

# Load the samples
cache_path = Path("small_sample.pkl")
with open(cache_path, 'rb') as f:
    samples = pickle.load(f)

print(f"Total samples: {len(samples)}")
print("\nSearching for high bars_to_first_break values (>20)...\n")

window = 50
found = []

for idx, sample in enumerate(samples):
    if window in sample.labels_per_window:
        window_data = sample.labels_per_window[window]
        if 'tsla' in window_data:
            for tf in ['5min', '1h', 'daily']:
                if tf in window_data['tsla']:
                    labels = window_data['tsla'][tf]
                    if labels.break_scan_valid and labels.bars_to_first_break > 20:
                        found.append({
                            'idx': idx,
                            'timestamp': sample.timestamp,
                            'tf': tf,
                            'bars_to_first_break': labels.bars_to_first_break,
                            'permanent_break': labels.permanent_break,
                            'break_direction': labels.break_direction,
                        })

# Sort by bars_to_first_break
found.sort(key=lambda x: x['bars_to_first_break'], reverse=True)

print(f"Found {len(found)} cases with bars_to_first_break > 20")
print("\nTop 10 highest:")
for i, case in enumerate(found[:10]):
    break_bar = window - 1 + case['bars_to_first_break']
    max_x = window - 1 + case['bars_to_first_break'] + 5
    print(f"\n{i+1}. Sample {case['idx']} - {case['timestamp']}")
    print(f"   TF: {case['tf']}")
    print(f"   bars_to_first_break: {case['bars_to_first_break']}")
    print(f"   permanent_break: {case['permanent_break']}")
    print(f"   break_bar position: {break_bar}")
    print(f"   x-axis max: {max_x}")
    print(f"   xlim: (-0.5, {max_x + 0.5})")

    # Check if it matches screenshot (bars_to_first_break ~58)
    if 55 <= case['bars_to_first_break'] <= 60:
        print(f"   *** MATCHES SCREENSHOT VALUE (58) ***")
