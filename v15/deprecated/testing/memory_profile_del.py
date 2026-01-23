#!/usr/bin/env python3
"""
Memory profiler to understand why `del` operation increases memory.

Uses tracemalloc to track allocations before/after:
1. Creating labeled maps
2. Creating slim maps
3. Deleting full labeled maps

This helps identify what memory operations happen during these phases.
"""
import gc
import sys
import tracemalloc
import psutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pandas as pd


def get_rss_mb():
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024**2)


def print_top_allocations(snapshot, title="Top Allocations", limit=15):
    """Print top memory allocations from a tracemalloc snapshot."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Group by filename
    stats = snapshot.statistics('filename')
    print(f"\nBy File (top {limit}):")
    for stat in stats[:limit]:
        print(f"  {stat.size / (1024**2):8.2f} MB | {stat.count:8d} blocks | {stat.traceback}")

    # Group by lineno
    stats = snapshot.statistics('lineno')
    print(f"\nBy Line (top {limit}):")
    for stat in stats[:limit]:
        print(f"  {stat.size / (1024**2):8.2f} MB | {stat.count:8d} blocks | {stat.traceback}")


def print_snapshot_diff(snapshot1, snapshot2, title="Snapshot Diff", limit=15):
    """Print difference between two snapshots."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Get diff by filename
    stats = snapshot2.compare_to(snapshot1, 'filename')
    print(f"\nBy File (top {limit} changes):")
    for stat in stats[:limit]:
        sign = "+" if stat.size_diff > 0 else ""
        print(f"  {sign}{stat.size_diff / (1024**2):8.2f} MB | {sign}{stat.count_diff:8d} blocks | {stat.traceback}")

    # Get diff by lineno
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    print(f"\nBy Line (top {limit} changes):")
    for stat in stats[:limit]:
        sign = "+" if stat.size_diff > 0 else ""
        print(f"  {sign}{stat.size_diff / (1024**2):8.2f} MB | {sign}{stat.count_diff:8d} blocks | {stat.traceback}")


def log_mem(label):
    """Log current memory state."""
    rss = get_rss_mb()
    traced = tracemalloc.get_traced_memory()
    current_mb = traced[0] / (1024**2)
    peak_mb = traced[1] / (1024**2)
    print(f"[MEM] {label}: RSS={rss:.1f}MB, Traced={current_mb:.1f}MB, Peak={peak_mb:.1f}MB")
    return rss


def main():
    print("="*70)
    print("MEMORY PROFILING: del OPERATION ANALYSIS")
    print("="*70)
    print(f"Python {sys.version}")
    print(f"PID: {psutil.Process().pid}")

    # Start tracemalloc BEFORE any imports
    tracemalloc.start(25)  # Store 25 frames for detailed tracebacks

    log_mem("START")

    # Import modules (this itself allocates memory)
    print("\n[1] Importing modules...")
    from v15.data import load_market_data
    from v15.labels import detect_all_channels, generate_all_labels
    from v15.scanner import _create_slim_labeled_map, SlimLabeledChannel

    snapshot_after_import = tracemalloc.take_snapshot()
    log_mem("AFTER_IMPORTS")

    # Load data (small subset)
    print("\n[2] Loading market data...")
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")

    snapshot_after_data = tracemalloc.take_snapshot()
    log_mem("AFTER_DATA_LOAD")

    # Use very small test configuration
    test_tfs = ['5min']  # Just 1 timeframe
    test_windows = [10]   # Just 1 window

    # Pass 1: Detect channels
    print("\n[3] Pass 1 - Detecting channels (small test)...")
    print(f"    Timeframes: {test_tfs}")
    print(f"    Windows: {test_windows}")

    tsla_channel_map, tsla_resampled = detect_all_channels(
        df=tsla,
        timeframes=test_tfs,
        windows=test_windows,
        step=50,  # Large step = fewer channels
        workers=1,
        verbose=False
    )

    snapshot_after_pass1 = tracemalloc.take_snapshot()
    log_mem("AFTER_PASS1")

    n_channels = sum(len(v) for v in tsla_channel_map.values())
    print(f"    Detected {n_channels} channels")

    # Pass 2: Generate labels
    print("\n[4] Pass 2 - Generating labels...")

    gc.collect()  # Clean up before snapshot
    snapshot_before_labels = tracemalloc.take_snapshot()
    log_mem("BEFORE_LABELS")

    tsla_labeled_map = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled,
        labeling_method="forward_scan",
        verbose=False
    )

    gc.collect()
    snapshot_after_labels = tracemalloc.take_snapshot()
    log_mem("AFTER_LABELS")

    n_labeled = sum(len(v) for v in tsla_labeled_map.values())
    print(f"    Labeled {n_labeled} channels")

    # Analyze labeled map contents
    print("\n[5] Analyzing labeled map memory footprint...")
    print(f"    Number of keys: {len(tsla_labeled_map)}")

    for key, channels in tsla_labeled_map.items():
        print(f"    Key {key}: {len(channels)} labeled channels")
        if channels:
            lc = channels[0]
            print(f"      Sample LabeledChannel:")
            print(f"        - detected.channel type: {type(lc.detected.channel).__name__}")
            print(f"        - labels type: {type(lc.labels).__name__}")
            # Check Channel size
            ch = lc.detected.channel
            print(f"        - Channel attributes: {[a for a in dir(ch) if not a.startswith('_')]}")

    print_snapshot_diff(snapshot_before_labels, snapshot_after_labels,
                       "DIFF: Before vs After Labels Generation", limit=20)

    # Free Pass-1 artifacts first
    print("\n[6] Freeing Pass-1 artifacts...")
    del tsla_channel_map, tsla_resampled
    gc.collect()

    snapshot_after_pass1_del = tracemalloc.take_snapshot()
    log_mem("AFTER_PASS1_DEL")

    # Create slim maps
    print("\n[7] Creating slim labeled map...")

    gc.collect()
    snapshot_before_slim = tracemalloc.take_snapshot()
    log_mem("BEFORE_SLIM_CREATE")

    tsla_slim = _create_slim_labeled_map(tsla_labeled_map)

    gc.collect()
    snapshot_after_slim = tracemalloc.take_snapshot()
    log_mem("AFTER_SLIM_CREATE")

    print_snapshot_diff(snapshot_before_slim, snapshot_after_slim,
                       "DIFF: Before vs After Slim Map Creation", limit=20)

    # Analyze slim map
    print("\n[8] Analyzing slim map contents...")
    for key, channels in tsla_slim.items():
        print(f"    Key {key}: {len(channels)} slim channels")
        if channels:
            sc = channels[0]
            print(f"      Sample SlimLabeledChannel:")
            print(f"        - start_timestamp: {sc.start_timestamp}")
            print(f"        - end_timestamp: {sc.end_timestamp}")
            print(f"        - labels type: {type(sc.labels).__name__}")

    # Check reference counts
    print("\n[9] Checking reference counts before del...")
    print(f"    tsla_labeled_map refcount: {sys.getrefcount(tsla_labeled_map)}")

    # Check if slim maps share references with labeled maps
    if tsla_labeled_map and tsla_slim:
        key = list(tsla_labeled_map.keys())[0]
        if tsla_labeled_map[key] and tsla_slim[key]:
            full_labels = tsla_labeled_map[key][0].labels
            slim_labels = tsla_slim[key][0].labels
            print(f"    Same labels object? {full_labels is slim_labels}")
            print(f"    full_labels id: {id(full_labels)}")
            print(f"    slim_labels id: {id(slim_labels)}")
            print(f"    Labels refcount: {sys.getrefcount(full_labels)}")

    # Now the critical del operation
    print("\n[10] Deleting full labeled map (THE CRITICAL OPERATION)...")

    gc.collect()
    gc.collect()
    snapshot_before_del = tracemalloc.take_snapshot()
    rss_before = get_rss_mb()
    log_mem("BEFORE_DEL")

    # Check what garbage collector knows
    gc_counts_before = gc.get_count()
    print(f"    GC counts before: {gc_counts_before}")

    # THE DEL OPERATION
    del tsla_labeled_map

    # Check immediately after del (before GC)
    rss_after_del = get_rss_mb()
    snapshot_after_del_no_gc = tracemalloc.take_snapshot()
    log_mem("IMMEDIATELY_AFTER_DEL (no gc)")

    gc_counts_after_del = gc.get_count()
    print(f"    GC counts after del: {gc_counts_after_del}")

    # Now run GC
    print("\n[11] Running garbage collection...")
    collected = gc.collect()
    print(f"    GC collected {collected} objects")

    gc_counts_after_gc = gc.get_count()
    print(f"    GC counts after gc: {gc_counts_after_gc}")

    snapshot_after_gc = tracemalloc.take_snapshot()
    rss_after_gc = get_rss_mb()
    log_mem("AFTER_GC")

    # Multiple GC passes
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            print(f"    GC pass {i+2} collected {collected} objects")

    snapshot_final = tracemalloc.take_snapshot()
    rss_final = get_rss_mb()
    log_mem("AFTER_MULTIPLE_GC")

    # Print diffs
    print_snapshot_diff(snapshot_before_del, snapshot_after_del_no_gc,
                       "DIFF: Before del vs After del (NO GC)", limit=25)

    print_snapshot_diff(snapshot_before_del, snapshot_after_gc,
                       "DIFF: Before del vs After GC", limit=25)

    print_snapshot_diff(snapshot_before_del, snapshot_final,
                       "DIFF: Before del vs Final (after multiple GC)", limit=25)

    # Summary
    print("\n" + "="*70)
    print("MEMORY SUMMARY")
    print("="*70)
    print(f"RSS before del:      {rss_before:.1f} MB")
    print(f"RSS after del (no gc): {rss_after_del:.1f} MB  (delta: {rss_after_del - rss_before:+.1f} MB)")
    print(f"RSS after gc:        {rss_after_gc:.1f} MB  (delta: {rss_after_gc - rss_before:+.1f} MB)")
    print(f"RSS final:           {rss_final:.1f} MB  (delta: {rss_final - rss_before:+.1f} MB)")

    traced_current, traced_peak = tracemalloc.get_traced_memory()
    print(f"\nTracemalloc current: {traced_current / (1024**2):.1f} MB")
    print(f"Tracemalloc peak:    {traced_peak / (1024**2):.1f} MB")

    # Check if slim map is still valid
    print("\n[12] Verifying slim map still works after full map deletion...")
    for key, channels in tsla_slim.items():
        print(f"    Key {key}: {len(channels)} channels still accessible")
        if channels:
            sc = channels[0]
            print(f"      - start: {sc.start_timestamp}")
            print(f"      - labels.duration_bars: {sc.labels.duration_bars}")

    tracemalloc.stop()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
