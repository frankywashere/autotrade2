#!/usr/bin/env python3
"""
Analyze next_channel labels in the cache file.
"""
import pickle
import numpy as np
import pandas as pd
from collections import Counter

# Load the cache
cache_path = '/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/channel_samples.pkl'
print(f"Loading cache from: {cache_path}")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

print(f"\nCache structure: {type(cache)}")
if isinstance(cache, dict):
    print(f"Cache keys: {list(cache.keys())}")

    # Extract all next_channel labels
    all_labels = []
    timeframe_labels = {}
    none_count = 0
    nan_count = 0

    for key, data in cache.items():
        if isinstance(data, dict) and 'next_channel' in data:
            label = data['next_channel']

            # Check for None/NaN
            if label is None:
                none_count += 1
                continue
            elif isinstance(label, (float, np.floating)) and np.isnan(label):
                nan_count += 1
                continue

            all_labels.append(label)

            # Extract timeframe from key if it's a tuple
            if isinstance(key, tuple) and len(key) >= 2:
                timeframe = key[1]
                if timeframe not in timeframe_labels:
                    timeframe_labels[timeframe] = []
                timeframe_labels[timeframe].append(label)

    print(f"\nTotal samples: {len(cache)}")
    print(f"Samples with next_channel: {len(all_labels)}")
    print(f"None values: {none_count}")
    print(f"NaN values: {nan_count}")

    # Overall distribution
    print("\n" + "="*60)
    print("OVERALL CLASS DISTRIBUTION")
    print("="*60)
    label_counts = Counter(all_labels)
    total = len(all_labels)

    # Map labels
    label_map = {0: 'bear', 1: 'sideways', 2: 'bull'}

    for label_val in sorted(label_counts.keys()):
        count = label_counts[label_val]
        pct = (count / total) * 100
        label_name = label_map.get(label_val, f'unknown({label_val})')
        print(f"{label_name:12} ({label_val}): {count:6} ({pct:5.2f}%)")

    # Check if labels are valid categorical (0, 1, 2)
    print("\n" + "="*60)
    print("LABEL VALIDATION")
    print("="*60)
    unique_labels = sorted(set(all_labels))
    print(f"Unique label values: {unique_labels}")

    invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
    if invalid_labels:
        print(f"WARNING: Found invalid labels: {invalid_labels}")
    else:
        print("✓ All labels are valid (0, 1, 2)")

    # Check data types
    label_types = set(type(l) for l in all_labels)
    print(f"Label data types: {label_types}")

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

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(cache)}")
    print(f"Valid labels: {len(all_labels)}")
    print(f"Missing/Invalid: {none_count + nan_count}")
    print(f"Coverage: {(len(all_labels) / len(cache) * 100):.2f}%")

else:
    print("Cache is not a dictionary. Investigating structure...")
    print(f"Type: {type(cache)}")
    if hasattr(cache, '__len__'):
        print(f"Length: {len(cache)}")
    if hasattr(cache, 'shape'):
        print(f"Shape: {cache.shape}")
