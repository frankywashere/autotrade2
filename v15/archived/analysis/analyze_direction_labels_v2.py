#!/usr/bin/env python3
"""
Analyze direction labels in the channel samples cache - optimized version.
"""

import pickle
import numpy as np
from collections import Counter, defaultdict
import sys

# Load the cache
cache_path = '/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/channel_samples.pkl'
print("Loading cache...", flush=True)
sys.stdout.flush()

with open(cache_path, 'rb') as f:
    data = pickle.load(f)

print(f"Cache contains {len(data)} samples", flush=True)
print("Processing...", flush=True)
sys.stdout.flush()

# Extract direction labels incrementally
all_directions = []
per_timeframe_directions = defaultdict(list)
sample_inconsistencies = []

# Process in batches to show progress
batch_size = 10000
for batch_start in range(0, len(data), batch_size):
    batch_end = min(batch_start + batch_size, len(data))
    print(f"Processing samples {batch_start} to {batch_end}...", flush=True)

    for i in range(batch_start, batch_end):
        sample = data[i]
        timeframe_dirs = {}

        for key in sample.keys():
            if 'direction' in key.lower():
                direction = sample[key]
                all_directions.append(direction)

                # Extract timeframe from key
                if '_' in key:
                    timeframe = key.split('_')[-1]
                    per_timeframe_directions[timeframe].append(direction)
                    timeframe_dirs[timeframe] = direction
                else:
                    per_timeframe_directions['default'].append(direction)
                    timeframe_dirs['default'] = direction

        # Check consistency
        if len(timeframe_dirs) > 1:
            unique_dirs = set(timeframe_dirs.values())
            if len(unique_dirs) > 1 and len(sample_inconsistencies) < 10:
                sample_inconsistencies.append((i, timeframe_dirs))

print("\n" + "=" * 80)
print("1. OVERALL CLASS BALANCE")
print("=" * 80)

if all_directions:
    unique_values = sorted(set(all_directions))
    print(f"Unique direction values: {unique_values}")

    counter = Counter(all_directions)
    total = len(all_directions)
    print(f"\nTotal direction labels: {total:,}")

    for direction in sorted(counter.keys()):
        count = counter[direction]
        percentage = (count / total) * 100
        label = "DOWN" if direction == 0 else "UP" if direction == 1 else f"UNKNOWN({direction})"
        print(f"  {label} (value={direction}): {count:,} ({percentage:.2f}%)")

    if len(counter) == 2:
        values = list(counter.values())
        imbalance_ratio = max(values) / min(values)
        print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}:1")
else:
    print("No direction labels found!")

print("\n" + "=" * 80)
print("2. PER-TIMEFRAME DISTRIBUTION")
print("=" * 80)

if per_timeframe_directions:
    for timeframe in sorted(per_timeframe_directions.keys()):
        directions = per_timeframe_directions[timeframe]
        counter = Counter(directions)
        total = len(directions)

        print(f"\nTimeframe: {timeframe}")
        print(f"Total samples: {total:,}")

        for direction in sorted(counter.keys()):
            count = counter[direction]
            percentage = (count / total) * 100
            label = "DOWN" if direction == 0 else "UP" if direction == 1 else f"UNKNOWN({direction})"
            print(f"  {label} (value={direction}): {count:,} ({percentage:.2f}%)")

        if len(counter) == 2:
            values = list(counter.values())
            imbalance_ratio = max(values) / min(values)
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
else:
    print("No per-timeframe direction labels found!")

print("\n" + "=" * 80)
print("3. CROSS-TIMEFRAME CONSISTENCY")
print("=" * 80)

if sample_inconsistencies:
    print(f"Found {len(sample_inconsistencies)} inconsistent samples (showing first 10):")
    for sample_idx, dirs in sample_inconsistencies:
        print(f"  Sample {sample_idx}: {dirs}")
else:
    print("All samples with multiple timeframes have consistent directions")

print("\n" + "=" * 80)
print("4. LABEL VALIDITY CHECK")
print("=" * 80)

if all_directions:
    unique_values = sorted(set(all_directions))
    is_binary = set(unique_values).issubset({0, 1})

    print(f"All values are binary (0 or 1): {is_binary}")
    print(f"Unique values found: {unique_values}")
    print(f"Data type: {type(all_directions[0])}")

    # Check for NaN or None
    nan_count = sum(1 for d in all_directions if d is None or (isinstance(d, float) and np.isnan(d)))
    print(f"NaN/None values: {nan_count}")

    min_val = min(all_directions)
    max_val = max(all_directions)
    print(f"Value range: [{min_val}, {max_val}]")
else:
    print("No direction labels to validate!")

print("\n" + "=" * 80)
print("5. SAMPLE STRUCTURE INSPECTION")
print("=" * 80)

if data:
    first_sample = data[0]
    direction_keys = [k for k in first_sample.keys() if 'direction' in k.lower()]
    print(f"Direction-related keys: {direction_keys}")

    print("\nAll keys in first sample:")
    for key in sorted(first_sample.keys()):
        value = first_sample[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: ndarray shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if all_directions:
    counter = Counter(all_directions)
    unique_values = sorted(set(all_directions))

    if len(counter) == 2 and set(counter.keys()) == {0, 1}:
        values = list(counter.values())
        min_class = min(values)
        max_class = max(values)
        total = sum(values)

        print(f"✓ Direction labels are valid binary (0/1)")
        print(f"✓ Total labels: {total:,}")
        print(f"  - Class 0 (DOWN): {counter[0]:,} ({counter[0]/total*100:.2f}%)")
        print(f"  - Class 1 (UP): {counter[1]:,} ({counter[1]/total*100:.2f}%)")
        print(f"  - Imbalance: {max_class/min_class:.2f}:1")

        majority_baseline = max_class / total * 100
        print(f"\nMajority class baseline accuracy: {majority_baseline:.2f}%")

        if 48 <= majority_baseline <= 52:
            print("⚠ Classes are nearly balanced - cannot explain ~50% accuracy via imbalance")
        elif majority_baseline > 60:
            print(f"⚠ Strong imbalance detected - majority class baseline is {majority_baseline:.2f}%")
    else:
        print(f"⚠ WARNING: Labels are not binary 0/1!")
        print(f"  Found values: {unique_values}")
else:
    print("✗ No direction labels found in cache!")

print("\nAnalysis complete!", flush=True)
