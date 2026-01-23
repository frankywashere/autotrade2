#!/usr/bin/env python3
"""
GC Debug Script: Analyze circular references and unreachable objects
around the labeled maps deletion in the scanner pipeline.

This script helps identify if there are:
1. Circular references preventing memory from being freed
2. Unreachable objects that are stuck in gc.garbage
3. Objects holding references to labeled_map that prevent cleanup

Usage:
    python gc_debug_labeled_maps.py
"""

import gc
import sys
import time
import weakref
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Enable GC debugging
gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_UNCOLLECTABLE)


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_gc_counts(label: str):
    """Print current GC generation counts."""
    counts = gc.get_count()
    print(f"[GC] {label}: gen0={counts[0]}, gen1={counts[1]}, gen2={counts[2]}")
    return counts


def analyze_referrers(obj: Any, obj_name: str, max_depth: int = 2):
    """
    Analyze what objects are referring to the given object.

    Args:
        obj: The object to analyze
        obj_name: Human-readable name for the object
        max_depth: How deep to search for referrers
    """
    print(f"\n[REFERRERS] Analyzing referrers for: {obj_name}")
    print(f"  Object type: {type(obj)}")
    print(f"  Object id: {id(obj)}")

    referrers = gc.get_referrers(obj)

    # Filter out frame objects and module-level references
    filtered_referrers = []
    for r in referrers:
        r_type = type(r).__name__
        # Skip frame objects (current execution frame)
        if r_type == 'frame':
            continue
        # Skip the local scope dict
        if isinstance(r, dict) and '__name__' in r:
            continue
        filtered_referrers.append(r)

    print(f"  Total referrers (filtered): {len(filtered_referrers)}")

    for i, ref in enumerate(filtered_referrers[:10]):  # Limit to first 10
        ref_type = type(ref).__name__
        ref_size = sys.getsizeof(ref) if hasattr(ref, '__sizeof__') else 'unknown'

        print(f"\n  Referrer {i+1}:")
        print(f"    Type: {ref_type}")
        print(f"    Size: {ref_size} bytes")
        print(f"    ID: {id(ref)}")

        # Show some content for dicts
        if isinstance(ref, dict):
            keys = list(ref.keys())[:5]
            print(f"    Sample keys: {keys}")
        elif isinstance(ref, list):
            print(f"    Length: {len(ref)}")
        elif isinstance(ref, tuple):
            print(f"    Length: {len(ref)}")

    if len(filtered_referrers) > 10:
        print(f"\n  ... and {len(filtered_referrers) - 10} more referrers")

    return len(filtered_referrers)


def check_gc_garbage():
    """Check if there are any uncollectable objects in gc.garbage."""
    print(f"\n[GC.GARBAGE] Uncollectable objects: {len(gc.garbage)}")

    if gc.garbage:
        # Analyze the garbage
        type_counts: Dict[str, int] = {}
        for obj in gc.garbage:
            t = type(obj).__name__
            type_counts[t] = type_counts.get(t, 0) + 1

        print("  Types in garbage:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {t}: {count}")

        # Show first few objects
        print("\n  First 5 uncollectable objects:")
        for i, obj in enumerate(gc.garbage[:5]):
            print(f"    {i+1}. {type(obj).__name__}: {repr(obj)[:100]}...")
    else:
        print("  No uncollectable objects (good!)")


def run_gc_collect_verbose():
    """Run gc.collect() with verbose output."""
    print("\n[GC.COLLECT] Running garbage collection...")

    # Collect each generation separately to see what's happening
    gen0 = gc.collect(0)
    gen1 = gc.collect(1)
    gen2 = gc.collect(2)

    print(f"  Gen 0 collected: {gen0} objects")
    print(f"  Gen 1 collected: {gen1} objects")
    print(f"  Gen 2 collected: {gen2} objects")
    print(f"  Total collected: {gen0 + gen1 + gen2} objects")

    return gen0 + gen1 + gen2


def main():
    """Main test function."""
    print_separator("GC DEBUG: LABELED MAPS MEMORY ANALYSIS")
    print(f"Python version: {sys.version}")
    print(f"GC thresholds: {gc.get_threshold()}")

    # Clear any existing garbage
    gc.garbage.clear()
    gc.collect()

    print_gc_counts("INITIAL")
    check_gc_garbage()

    # =========================================================================
    # STEP 1: Load data and create labeled maps (small test)
    # =========================================================================
    print_separator("STEP 1: Loading data and creating labeled maps")

    print("\n[IMPORT] Importing v15 modules...")
    from v15.data import load_market_data
    from v15.labels import (
        detect_all_channels,
        generate_all_labels,
        LabeledChannelMap,
    )
    from v15.config import TIMEFRAMES, STANDARD_WINDOWS
    from v15.scanner import _create_slim_labeled_map

    print_gc_counts("AFTER_IMPORTS")

    print("\n[DATA] Loading market data...")
    tsla_df, spy_df, vix_df = load_market_data("data")

    # Use a small slice for faster testing
    # Use just 10000 bars for testing
    test_size = 10000
    tsla_slice = tsla_df.iloc[:test_size].copy()
    spy_slice = spy_df.iloc[:test_size].copy()

    del tsla_df, spy_df, vix_df  # Free full data
    gc.collect()

    print(f"  Using {len(tsla_slice)} bars for testing")
    print_gc_counts("AFTER_DATA_SLICE")

    print("\n[PASS1] Running channel detection (TSLA only for speed)...")

    # Use fewer timeframes and windows for testing
    test_timeframes = ['5min', '15min', '1h']
    test_windows = [20, 50]

    tsla_channel_map, tsla_resampled_dfs = detect_all_channels(
        df=tsla_slice,
        timeframes=test_timeframes,
        windows=test_windows,
        step=10,  # Larger step for speed
        min_cycles=1,
        min_gap_bars=5,
        workers=4,
        verbose=False
    )

    total_channels = sum(len(chs) for chs in tsla_channel_map.values())
    print(f"  Detected {total_channels} channels")
    print_gc_counts("AFTER_CHANNEL_DETECTION")

    print("\n[PASS2] Generating labels...")

    tsla_labeled_map: LabeledChannelMap = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled_dfs,
        labeling_method="forward_scan",
        verbose=False
    )

    total_labeled = sum(len(lcs) for lcs in tsla_labeled_map.values())
    print(f"  Labeled {total_labeled} channels")
    print_gc_counts("AFTER_LABEL_GENERATION")

    # =========================================================================
    # STEP 2: Create slim maps
    # =========================================================================
    print_separator("STEP 2: Creating slim labeled maps")

    print("\n[SLIM] Creating slim labeled map...")
    tsla_slim_map = _create_slim_labeled_map(tsla_labeled_map)

    slim_entries = sum(len(lcs) for lcs in tsla_slim_map.values())
    print(f"  Slim map has {slim_entries} entries")
    print_gc_counts("AFTER_SLIM_MAP_CREATION")

    # =========================================================================
    # STEP 3: Before del - check referrers
    # =========================================================================
    print_separator("STEP 3: Checking referrers BEFORE deletion")

    print("\n[REFS] Tracking objects by ID (dicts don't support weak refs)...")

    # Track object IDs since dicts don't support weak references
    labeled_map_id = id(tsla_labeled_map)
    channel_map_id = id(tsla_channel_map)
    resampled_dfs_id = id(tsla_resampled_dfs)

    print(f"  labeled_map id: {labeled_map_id}")
    print(f"  channel_map id: {channel_map_id}")
    print(f"  resampled_dfs id: {resampled_dfs_id}")

    # Analyze referrers
    print("\n[ANALYSIS] Checking what's holding references...")

    n_refs_labeled = analyze_referrers(tsla_labeled_map, "tsla_labeled_map")
    n_refs_channel = analyze_referrers(tsla_channel_map, "tsla_channel_map")
    n_refs_resampled = analyze_referrers(tsla_resampled_dfs, "tsla_resampled_dfs")

    print_gc_counts("BEFORE_DELETION")

    # =========================================================================
    # STEP 4: Delete and run GC with DEBUG_STATS
    # =========================================================================
    print_separator("STEP 4: Deleting labeled maps and running GC")

    print("\n[DELETE] Deleting tsla_labeled_map...")
    del tsla_labeled_map

    print("[DELETE] Deleting tsla_channel_map...")
    del tsla_channel_map

    print("[DELETE] Deleting tsla_resampled_dfs...")
    del tsla_resampled_dfs

    print_gc_counts("AFTER_DELETE_BEFORE_GC")

    print("\n[GC] Running garbage collection with DEBUG_STATS...")
    # The DEBUG_STATS flag will print stats to stderr
    collected = run_gc_collect_verbose()

    print_gc_counts("AFTER_GC")

    # =========================================================================
    # STEP 5: Check gc.garbage for uncollectable objects
    # =========================================================================
    print_separator("STEP 5: Checking gc.garbage for uncollectable objects")

    check_gc_garbage()

    # Check if objects still exist by searching for their IDs
    print("\n[OBJECT CHECK] Checking if objects were collected...")

    # Search for objects with the tracked IDs
    all_objects = gc.get_objects()
    labeled_map_found = any(id(obj) == labeled_map_id for obj in all_objects)
    channel_map_found = any(id(obj) == channel_map_id for obj in all_objects)
    resampled_dfs_found = any(id(obj) == resampled_dfs_id for obj in all_objects)

    print(f"  labeled_map collected: {not labeled_map_found}")
    print(f"  channel_map collected: {not channel_map_found}")
    print(f"  resampled_dfs collected: {not resampled_dfs_found}")

    # If any are still alive, analyze why
    if labeled_map_found:
        print("\n  WARNING: labeled_map is STILL ALIVE after del + gc.collect()!")
        for obj in all_objects:
            if id(obj) == labeled_map_id:
                analyze_referrers(obj, "labeled_map (should be dead)")
                break

    if channel_map_found:
        print("\n  WARNING: channel_map is STILL ALIVE after del + gc.collect()!")
        for obj in all_objects:
            if id(obj) == channel_map_id:
                analyze_referrers(obj, "channel_map (should be dead)")
                break

    if resampled_dfs_found:
        print("\n  WARNING: resampled_dfs is STILL ALIVE after del + gc.collect()!")
        for obj in all_objects:
            if id(obj) == resampled_dfs_id:
                analyze_referrers(obj, "resampled_dfs (should be dead)")
                break

    del all_objects  # Free the list

    # =========================================================================
    # STEP 6: Final gc.get_count() comparison
    # =========================================================================
    print_separator("STEP 6: Final GC counts comparison")

    # Run one more collection
    final_collected = run_gc_collect_verbose()

    final_counts = print_gc_counts("FINAL")
    check_gc_garbage()

    # =========================================================================
    # Summary
    # =========================================================================
    print_separator("SUMMARY")

    print("\n[RESULTS]")
    print(f"  Objects collected in main GC: {collected}")
    print(f"  Objects collected in final GC: {final_collected}")
    print(f"  Uncollectable objects (gc.garbage): {len(gc.garbage)}")
    print(f"  labeled_map freed: {not labeled_map_found}")
    print(f"  channel_map freed: {not channel_map_found}")
    print(f"  resampled_dfs freed: {not resampled_dfs_found}")
    print(f"  slim_map still alive: {tsla_slim_map is not None}")

    # Diagnosis
    print("\n[DIAGNOSIS]")

    if gc.garbage:
        print("  ISSUE: Uncollectable objects found in gc.garbage!")
        print("         This indicates circular references with __del__ methods")
        print("         or C-extension cycles that Python cannot break.")
    else:
        print("  OK: No uncollectable objects.")

    if labeled_map_found:
        print("  ISSUE: labeled_map not freed - something is still holding a reference!")
    else:
        print("  OK: labeled_map was properly freed.")

    if channel_map_found:
        print("  ISSUE: channel_map not freed - something is still holding a reference!")
    else:
        print("  OK: channel_map was properly freed.")

    if resampled_dfs_found:
        print("  ISSUE: resampled_dfs not freed - something is still holding a reference!")
    else:
        print("  OK: resampled_dfs was properly freed.")

    # Check if slim_map is holding references to the deleted objects
    print("\n[SLIM MAP CHECK]")
    print("  Checking if slim_map entries contain references to deleted objects...")

    for key, channels in tsla_slim_map.items():
        if channels:
            first_ch = channels[0]
            # Check what attributes the slim channel has
            print(f"  Key {key}: {len(channels)} entries")
            print(f"    First entry type: {type(first_ch)}")
            print(f"    First entry attrs: {[a for a in dir(first_ch) if not a.startswith('_')]}")
            break

    print("\n" + "=" * 70)
    print("GC DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
