#!/usr/bin/env python3
"""
Analyze window mismatch between features and labels in cached samples.

This script checks how often best_window (used for features, selected by bounce_count)
differs from best_labels_window (used for labels, selected by valid TF count).
"""

import pickle
from pathlib import Path
from typing import Dict, Optional
from collections import Counter

# Import the selection logic
import sys
sys.path.insert(0, str(Path(__file__).parent))

from v7.training.labels import ChannelLabels


def select_best_window_by_labels(labels_per_window: Dict[int, Dict[str, Optional[ChannelLabels]]]) -> int:
    """
    Replicate the selection logic from v7/training/labels.py.

    Selects the window with the most valid TF labels.
    """
    if not labels_per_window:
        raise ValueError("labels_per_window cannot be empty")

    best_window = None
    best_valid_count = -1

    # Sort by window size to prefer smaller windows on ties
    for window_size in sorted(labels_per_window.keys()):
        tf_labels = labels_per_window[window_size]

        # Count valid (non-None) labels
        valid_count = sum(1 for labels in tf_labels.values() if labels is not None)

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_window = window_size

    # If no window found (shouldn't happen), return first
    if best_window is None:
        best_window = next(iter(labels_per_window.keys()))

    return best_window


def analyze_cache(cache_path: Path):
    """Analyze window mismatch in the cache."""

    print("="*70)
    print("WINDOW MISMATCH ANALYSIS")
    print("="*70)
    print()

    # Load cache
    print(f"Loading cache from: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Cache file not found: {cache_path}")
        return
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return

    print(f"✓ Loaded {len(samples):,} samples")
    print()

    # Analyze each sample
    mismatches = 0
    matches = 0
    window_usage = Counter()
    labels_window_usage = Counter()
    mismatch_pairs = Counter()

    valid_count_by_window = {}  # Track how many valid labels per window

    for i, sample in enumerate(samples):
        best_window = sample.best_window

        # Infer what best_labels_window would have been
        if hasattr(sample, 'labels_per_window') and sample.labels_per_window:
            best_labels_window = select_best_window_by_labels(sample.labels_per_window)

            # Count valid labels for this sample
            for window_size, tf_labels in sample.labels_per_window.items():
                valid_count = sum(1 for labels in tf_labels.values() if labels is not None)
                if window_size not in valid_count_by_window:
                    valid_count_by_window[window_size] = []
                valid_count_by_window[window_size].append(valid_count)
        else:
            # Old cache format or no multi-window labels
            best_labels_window = best_window  # Assume same

        # Track statistics
        window_usage[best_window] += 1
        labels_window_usage[best_labels_window] += 1

        if best_window != best_labels_window:
            mismatches += 1
            mismatch_pairs[(best_window, best_labels_window)] += 1
        else:
            matches += 1

    # Print results
    total = len(samples)
    mismatch_rate = (mismatches / total) * 100 if total > 0 else 0

    print("─" * 70)
    print("OVERALL STATISTICS")
    print("─" * 70)
    print(f"Total samples:        {total:,}")
    print(f"Matches:              {matches:,} ({matches/total*100:.1f}%)")
    print(f"Mismatches:           {mismatches:,} ({mismatch_rate:.1f}%)")
    print()

    # Interpretation
    print("INTERPRETATION:")
    if mismatch_rate < 10:
        print(f"✓ Low mismatch rate ({mismatch_rate:.1f}%) - probably not a major issue")
    elif mismatch_rate < 30:
        print(f"⚠️  Moderate mismatch rate ({mismatch_rate:.1f}%) - alignment fix recommended")
    else:
        print(f"❌ High mismatch rate ({mismatch_rate:.1f}%) - alignment fix strongly recommended")
    print()

    # Window usage for features (bounce-based)
    print("─" * 70)
    print("WINDOW USAGE FOR FEATURES (bounce_count based)")
    print("─" * 70)
    for window in sorted(window_usage.keys()):
        count = window_usage[window]
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"Window {window:3d}: {count:6,} ({pct:5.1f}%) {bar}")
    print()

    # Window usage for labels (validity-based)
    print("─" * 70)
    print("WINDOW USAGE FOR LABELS (valid TF count based)")
    print("─" * 70)
    for window in sorted(labels_window_usage.keys()):
        count = labels_window_usage[window]
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"Window {window:3d}: {count:6,} ({pct:5.1f}%) {bar}")
    print()

    # Average valid labels per window
    if valid_count_by_window:
        print("─" * 70)
        print("AVERAGE VALID TF LABELS PER WINDOW")
        print("─" * 70)
        for window in sorted(valid_count_by_window.keys()):
            counts = valid_count_by_window[window]
            avg = sum(counts) / len(counts) if counts else 0
            print(f"Window {window:3d}: {avg:.2f} / 11 TFs ({avg/11*100:.1f}% valid)")
        print()

    # Most common mismatch pairs
    if mismatch_pairs:
        print("─" * 70)
        print("TOP 10 MISMATCH PAIRS (feature_window → label_window)")
        print("─" * 70)
        for (feat_win, label_win), count in mismatch_pairs.most_common(10):
            pct = (count / mismatches) * 100 if mismatches > 0 else 0
            print(f"Window {feat_win:3d} → {label_win:3d}: {count:5,} ({pct:5.1f}% of mismatches)")
        print()

    # Recommendations
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if mismatch_rate < 10:
        print("1. Mismatch rate is low - current system is working reasonably well")
        print("2. Consider alignment fix if you want perfect consistency")
        print("3. Multi-window features probably not needed")
    elif mismatch_rate < 30:
        print("1. ✅ Implement alignment fix: Use same window for features AND labels")
        print("2. Options:")
        print("   a) Force best_labels_window = best_window (prioritize bounces)")
        print("   b) Force best_window = best_labels_window (prioritize labels)")
        print("   c) Use combined score: alpha*bounces + beta*valid_labels")
        print("3. Measure impact on validation metrics after fixing")
    else:
        print("1. ⚠️  HIGH PRIORITY: Fix alignment immediately")
        print("2. Recommended: Use combined score for selection")
        print("3. Consider multi-window features if single-window plateaus")
        print("4. Add window selection loss to training")

    print()
    print("="*70)


if __name__ == "__main__":
    # Default cache path
    cache_path = Path(__file__).parent / "data" / "feature_cache" / "channel_samples.pkl"

    # Allow override from command line
    if len(sys.argv) > 1:
        cache_path = Path(sys.argv[1])

    analyze_cache(cache_path)
