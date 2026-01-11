#!/usr/bin/env python3
"""
Analyze next_channel labels in the cache file.
"""
import pickle
import sys

# Load the cache
cache_path = '/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/channel_samples.pkl'
print(f"Loading cache from: {cache_path}", flush=True)

try:
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    print(f"Cache loaded successfully", flush=True)
except Exception as e:
    print(f"Error loading cache: {e}", flush=True)
    sys.exit(1)

print(f"Cache type: {type(cache)}", flush=True)
print(f"Cache size: {len(cache) if hasattr(cache, '__len__') else 'unknown'}", flush=True)

if isinstance(cache, dict):
    print(f"Cache keys sample: {list(cache.keys())[:3]}", flush=True)

    # Extract all next_channel labels
    all_labels = []
    timeframe_labels = {}
    none_count = 0
    nan_count = 0

    print("Processing samples...", flush=True)
    for i, (key, data) in enumerate(cache.items()):
        if i % 10000 == 0:
            print(f"Processed {i} samples...", flush=True)

        if isinstance(data, dict) and 'next_channel' in data:
            label = data['next_channel']

            # Check for None/NaN
            if label is None:
                none_count += 1
                continue

            try:
                import math
                if isinstance(label, float) and math.isnan(label):
                    nan_count += 1
                    continue
            except:
                pass

            all_labels.append(label)

            # Extract timeframe from key if it's a tuple
            if isinstance(key, tuple) and len(key) >= 2:
                timeframe = key[1]
                if timeframe not in timeframe_labels:
                    timeframe_labels[timeframe] = []
                timeframe_labels[timeframe].append(label)

    print(f"\nTotal samples: {len(cache)}", flush=True)
    print(f"Samples with next_channel: {len(all_labels)}", flush=True)
    print(f"None values: {none_count}", flush=True)
    print(f"NaN values: {nan_count}", flush=True)

    # Count labels
    from collections import Counter
    label_counts = Counter(all_labels)
    total = len(all_labels)

    # Map labels
    label_map = {0: 'bear', 1: 'sideways', 2: 'bull'}

    print("\n" + "="*60)
    print("OVERALL CLASS DISTRIBUTION")
    print("="*60)
    for label_val in sorted(label_counts.keys()):
        count = label_counts[label_val]
        pct = (count / total) * 100
        label_name = label_map.get(label_val, f'unknown({label_val})')
        print(f"{label_name:12} ({label_val}): {count:6} ({pct:5.2f}%)")

    # Validation
    print("\n" + "="*60)
    print("LABEL VALIDATION")
    print("="*60)
    unique_labels = sorted(set(all_labels))
    print(f"Unique label values: {unique_labels}")

    invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
    if invalid_labels:
        print(f"WARNING: Found invalid labels: {invalid_labels}")
    else:
        print("All labels are valid (0, 1, 2)")

    # Balance analysis
    print("\n" + "="*60)
    print("CLASS BALANCE ANALYSIS")
    print("="*60)
    if len(label_counts) > 0:
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

        if imbalance_ratio < 1.5:
            print("Status: Well balanced")
        elif imbalance_ratio < 3.0:
            print("Status: Moderately imbalanced")
        else:
            print("Status: Heavily imbalanced")

    # Per-timeframe distribution
    print("\n" + "="*60)
    print("PER-TIMEFRAME DISTRIBUTION")
    print("="*60)

    for timeframe in sorted(timeframe_labels.keys()):
        labels = timeframe_labels[timeframe]
        tf_counts = Counter(labels)
        tf_total = len(labels)

        print(f"\n{timeframe} (n={tf_total}):")
        for label_val in sorted(tf_counts.keys()):
            count = tf_counts[label_val]
            pct = (count / tf_total) * 100
            label_name = label_map.get(label_val, f'unknown({label_val})')
            print(f"  {label_name:12} ({label_val}): {count:5} ({pct:5.2f}%)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(cache)}")
    print(f"Valid labels: {len(all_labels)}")
    print(f"Missing/Invalid: {none_count + nan_count}")
    if len(cache) > 0:
        print(f"Coverage: {(len(all_labels) / len(cache) * 100):.2f}%")

else:
    print("Cache is not a dictionary")
    print(f"Type: {type(cache)}")
