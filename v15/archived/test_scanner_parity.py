#!/usr/bin/env python3
"""
Test script to verify parallel and sequential scanners produce identical results.

This test compares scan_channels() (parallel) with scan_channels_sequential()
to ensure they produce bit-for-bit identical results.

Usage:
    python test_scanner_parity.py
"""

import sys
import time
import numpy as np
from typing import List, Dict, Any, Tuple

# Test parameters
WARMUP_BARS = 1000
FORWARD_BARS = 500
STEP = 100
WORKERS = 2
FLOAT_TOLERANCE = 1e-10


def compare_tf_features(
    features1: Dict[str, float],
    features2: Dict[str, float],
    tolerance: float = FLOAT_TOLERANCE
) -> Tuple[bool, List[str]]:
    """
    Compare two tf_features dictionaries.

    Returns:
        Tuple of (all_match, list of difference messages)
    """
    diffs = []

    keys1 = set(features1.keys())
    keys2 = set(features2.keys())

    # Check for missing keys
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    if only_in_1:
        diffs.append(f"  Keys only in parallel: {sorted(only_in_1)[:5]}{'...' if len(only_in_1) > 5 else ''}")
    if only_in_2:
        diffs.append(f"  Keys only in sequential: {sorted(only_in_2)[:5]}{'...' if len(only_in_2) > 5 else ''}")

    # Compare common keys
    common_keys = keys1 & keys2
    value_mismatches = []

    for key in sorted(common_keys):
        val1 = features1[key]
        val2 = features2[key]

        # Handle NaN
        if np.isnan(val1) and np.isnan(val2):
            continue
        if np.isnan(val1) or np.isnan(val2):
            value_mismatches.append((key, val1, val2))
            continue

        # Check tolerance
        if abs(val1 - val2) > tolerance:
            value_mismatches.append((key, val1, val2))

    if value_mismatches:
        diffs.append(f"  Value mismatches ({len(value_mismatches)} features):")
        for key, v1, v2 in value_mismatches[:5]:
            diffs.append(f"    {key}: parallel={v1}, sequential={v2}, diff={abs(v1-v2):.2e}")
        if len(value_mismatches) > 5:
            diffs.append(f"    ... and {len(value_mismatches) - 5} more")

    all_match = len(diffs) == 0
    return all_match, diffs


def compare_labels(
    labels1: Dict[int, Dict[str, Any]],
    labels2: Dict[int, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Compare two labels_per_window dictionaries.

    Returns:
        Tuple of (all_match, list of difference messages)
    """
    diffs = []

    windows1 = set(labels1.keys())
    windows2 = set(labels2.keys())

    if windows1 != windows2:
        diffs.append(f"  Window mismatch: parallel={sorted(windows1)}, sequential={sorted(windows2)}")
        return False, diffs

    for window in sorted(windows1):
        tfs1 = labels1[window]
        tfs2 = labels2[window]

        if tfs1 is None and tfs2 is None:
            continue
        if tfs1 is None or tfs2 is None:
            diffs.append(f"  Window {window}: one is None (parallel={tfs1 is not None}, sequential={tfs2 is not None})")
            continue

        tfs_keys1 = set(tfs1.keys()) if tfs1 else set()
        tfs_keys2 = set(tfs2.keys()) if tfs2 else set()

        if tfs_keys1 != tfs_keys2:
            diffs.append(f"  Window {window}: timeframe mismatch parallel={sorted(tfs_keys1)}, sequential={sorted(tfs_keys2)}")
            continue

        for tf in sorted(tfs_keys1):
            label1 = tfs1[tf]
            label2 = tfs2[tf]

            if label1 is None and label2 is None:
                continue
            if label1 is None or label2 is None:
                diffs.append(f"  Window {window}, TF {tf}: one is None")
                continue

            # Compare ChannelLabels attributes
            attrs = ['duration_bars', 'break_direction', 'break_trigger_tf',
                     'new_channel_direction', 'permanent_break',
                     'duration_valid', 'direction_valid', 'trigger_tf_valid', 'new_channel_valid']

            for attr in attrs:
                v1 = getattr(label1, attr, None)
                v2 = getattr(label2, attr, None)
                if v1 != v2:
                    diffs.append(f"  Window {window}, TF {tf}, {attr}: parallel={v1}, sequential={v2}")

    all_match = len(diffs) == 0
    return all_match, diffs


def main():
    """Run parity test between parallel and sequential scanners."""
    print("=" * 70)
    print("Scanner Parity Test")
    print("=" * 70)
    print(f"\nTest Parameters:")
    print(f"  warmup_bars: {WARMUP_BARS}")
    print(f"  forward_bars: {FORWARD_BARS}")
    print(f"  step: {STEP}")
    print(f"  workers: {WORKERS}")
    print(f"  float_tolerance: {FLOAT_TOLERANCE}")
    print()

    # Import scanner functions
    print("Loading scanner module...")
    try:
        from v15.scanner import scan_channels, scan_channels_sequential
        from v15.data import load_market_data
    except ImportError as e:
        print(f"ERROR: Failed to import scanner module: {e}")
        print("Make sure you're running from the x14 directory.")
        sys.exit(1)

    # Load data
    print("\nLoading market data...")
    try:
        tsla_full, spy_full, vix_full = load_market_data("data")
        print(f"  Full dataset: {len(tsla_full)} bars")
    except FileNotFoundError as e:
        print(f"ERROR: Data files not found: {e}")
        sys.exit(1)

    # Check if we have enough data
    min_required = WARMUP_BARS + FORWARD_BARS + STEP
    if len(tsla_full) < min_required:
        print(f"ERROR: Not enough data. Have {len(tsla_full)} bars, need at least {min_required}")
        sys.exit(1)

    # Slice data for quick test: warmup + test_positions * step + forward
    # This gives us exactly the data we need for a controlled number of positions
    test_positions = 20  # Number of positions to test
    slice_end = WARMUP_BARS + (test_positions * STEP) + FORWARD_BARS

    # Use min to avoid exceeding available data
    slice_end = min(slice_end, len(tsla_full))

    tsla = tsla_full.iloc[:slice_end].copy()
    spy = spy_full.iloc[:slice_end].copy()
    vix = vix_full.iloc[:slice_end].copy()

    print(f"  Using slice of {len(tsla)} bars for quick test (~{test_positions} positions)")

    # Run parallel scan
    print("\n" + "-" * 70)
    print("Running PARALLEL scan...")
    print("-" * 70)
    t_start = time.perf_counter()
    parallel_samples = scan_channels(
        tsla, spy, vix,
        step=STEP,
        warmup_bars=WARMUP_BARS,
        forward_bars=FORWARD_BARS,
        workers=WORKERS,
        progress=True
    )
    t_parallel = time.perf_counter() - t_start
    print(f"Parallel scan completed in {t_parallel:.2f}s")

    # Run sequential scan
    print("\n" + "-" * 70)
    print("Running SEQUENTIAL scan...")
    print("-" * 70)
    t_start = time.perf_counter()
    sequential_samples = scan_channels_sequential(
        tsla, spy, vix,
        step=STEP,
        warmup_bars=WARMUP_BARS,
        forward_bars=FORWARD_BARS,
        progress=True
    )
    t_sequential = time.perf_counter() - t_start
    print(f"Sequential scan completed in {t_sequential:.2f}s")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    all_pass = True

    # 1. Compare sample counts
    print(f"\n1. Sample Count Check:")
    print(f"   Parallel samples:   {len(parallel_samples)}")
    print(f"   Sequential samples: {len(sequential_samples)}")
    if len(parallel_samples) == len(sequential_samples):
        print("   PASS: Same number of samples")
    else:
        print("   FAIL: Different number of samples!")
        all_pass = False

    if len(parallel_samples) == 0 and len(sequential_samples) == 0:
        print("\nWARNING: Both scanners produced 0 samples. Test inconclusive.")
        sys.exit(1)

    if len(parallel_samples) != len(sequential_samples):
        print("\nERROR: Cannot compare samples with different counts.")
        print("Stopping comparison.")
        sys.exit(1)

    # 2. Compare timestamps
    print(f"\n2. Timestamp Check:")
    timestamps_match = True
    timestamp_diffs = []
    for i, (p_sample, s_sample) in enumerate(zip(parallel_samples, sequential_samples)):
        if p_sample.timestamp != s_sample.timestamp:
            timestamps_match = False
            timestamp_diffs.append((i, p_sample.timestamp, s_sample.timestamp))

    if timestamps_match:
        print("   PASS: All timestamps match")
    else:
        print(f"   FAIL: {len(timestamp_diffs)} timestamp mismatches!")
        for i, p_ts, s_ts in timestamp_diffs[:3]:
            print(f"     Sample {i}: parallel={p_ts}, sequential={s_ts}")
        if len(timestamp_diffs) > 3:
            print(f"     ... and {len(timestamp_diffs) - 3} more")
        all_pass = False

    # 3. Compare tf_features
    print(f"\n3. TF Features Check:")
    features_match_count = 0
    features_mismatch_samples = []

    for i, (p_sample, s_sample) in enumerate(zip(parallel_samples, sequential_samples)):
        p_features = p_sample.tf_features or {}
        s_features = s_sample.tf_features or {}

        match, diffs = compare_tf_features(p_features, s_features)
        if match:
            features_match_count += 1
        else:
            features_mismatch_samples.append((i, diffs))

    if features_match_count == len(parallel_samples):
        print(f"   PASS: All {features_match_count} samples have matching tf_features")
    else:
        print(f"   FAIL: {len(features_mismatch_samples)} samples have tf_features mismatches")
        for i, diffs in features_mismatch_samples[:3]:
            print(f"   Sample {i} (timestamp={parallel_samples[i].timestamp}):")
            for d in diffs[:3]:
                print(f"   {d}")
        if len(features_mismatch_samples) > 3:
            print(f"   ... and {len(features_mismatch_samples) - 3} more samples with differences")
        all_pass = False

    # 4. Compare labels
    print(f"\n4. Labels Check:")
    labels_match_count = 0
    labels_mismatch_samples = []

    for i, (p_sample, s_sample) in enumerate(zip(parallel_samples, sequential_samples)):
        p_labels = p_sample.labels_per_window or {}
        s_labels = s_sample.labels_per_window or {}

        match, diffs = compare_labels(p_labels, s_labels)
        if match:
            labels_match_count += 1
        else:
            labels_mismatch_samples.append((i, diffs))

    if labels_match_count == len(parallel_samples):
        print(f"   PASS: All {labels_match_count} samples have matching labels")
    else:
        print(f"   FAIL: {len(labels_mismatch_samples)} samples have label mismatches")
        for i, diffs in labels_mismatch_samples[:3]:
            print(f"   Sample {i} (timestamp={parallel_samples[i].timestamp}):")
            for d in diffs[:5]:
                print(f"   {d}")
        if len(labels_mismatch_samples) > 3:
            print(f"   ... and {len(labels_mismatch_samples) - 3} more samples with differences")
        all_pass = False

    # 5. Compare best_window
    print(f"\n5. Best Window Check:")
    best_window_mismatches = []
    for i, (p_sample, s_sample) in enumerate(zip(parallel_samples, sequential_samples)):
        if p_sample.best_window != s_sample.best_window:
            best_window_mismatches.append((i, p_sample.best_window, s_sample.best_window))

    if len(best_window_mismatches) == 0:
        print(f"   PASS: All samples have matching best_window")
    else:
        print(f"   FAIL: {len(best_window_mismatches)} samples have best_window mismatches")
        for i, p_bw, s_bw in best_window_mismatches[:5]:
            print(f"     Sample {i}: parallel={p_bw}, sequential={s_bw}")
        all_pass = False

    # 6. Compare channel_end_idx
    print(f"\n6. Channel End Index Check:")
    idx_mismatches = []
    for i, (p_sample, s_sample) in enumerate(zip(parallel_samples, sequential_samples)):
        if p_sample.channel_end_idx != s_sample.channel_end_idx:
            idx_mismatches.append((i, p_sample.channel_end_idx, s_sample.channel_end_idx))

    if len(idx_mismatches) == 0:
        print(f"   PASS: All samples have matching channel_end_idx")
    else:
        print(f"   FAIL: {len(idx_mismatches)} samples have channel_end_idx mismatches")
        for i, p_idx, s_idx in idx_mismatches[:5]:
            print(f"     Sample {i}: parallel={p_idx}, sequential={s_idx}")
        all_pass = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Samples compared: {len(parallel_samples)}")
    print(f"  Parallel time:    {t_parallel:.2f}s")
    print(f"  Sequential time:  {t_sequential:.2f}s")
    print(f"  Speedup:          {t_sequential/t_parallel:.2f}x")
    print()

    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
        print("  Parallel and sequential scanners produce IDENTICAL results.")
        return 0
    else:
        print("  RESULT: SOME CHECKS FAILED")
        print("  Parallel and sequential scanners produce DIFFERENT results!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
