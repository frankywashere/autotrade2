#!/usr/bin/env python3
"""
Simple memory test - just test the scan phase with pre-made labeled maps.
"""
import gc
import os
import sys
import time
import psutil

def get_mem():
    proc = psutil.Process()
    main = proc.memory_info().rss / (1024**2)
    children = proc.children(recursive=True)
    child_mem = sum(c.memory_info().rss / (1024**2) for c in children)
    return main, child_mem, len(children)

def log_mem(label):
    main, child, n = get_mem()
    print(f"[MEM] {label}: Main={main:.0f}MB, Children={child:.0f}MB ({n} workers), Total={main+child:.0f}MB")

# Define worker function at module level for pickling
def dummy_worker(x):
    """Simple worker that accesses globals."""
    time.sleep(0.1)
    return x * 2

def real_worker(positions_batch):
    """Worker that actually processes like the scanner does."""
    from v15.scanner import (
        _WORKER_TSLA_DF, _WORKER_SPY_DF, _WORKER_VIX_DF,
        _WORKER_TSLA_LABELED_MAP, _WORKER_SPY_LABELED_MAP,
        _WORKER_TIMEFRAMES, _WORKER_WINDOWS
    )
    from v15.labels import get_labels_for_position

    results = []
    for idx in positions_batch:
        # Simulate accessing labeled maps like the real scanner
        for tf in _WORKER_TIMEFRAMES:
            for window in _WORKER_WINDOWS:
                labels = get_labels_for_position(
                    _WORKER_TSLA_LABELED_MAP, _WORKER_TSLA_DF, idx, tf, window
                )
        results.append(idx)
    return results

def main():
    print("="*60)
    print("SIMPLE MEMORY TEST")
    print("="*60)

    log_mem("START")

    # Load data
    print("\n[1] Loading data...")
    from v15.data import load_market_data
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")
    log_mem("AFTER_LOAD")

    # Create channel maps (Pass 1)
    print("\n[2] Pass 1 - Detecting channels...")
    from v15.labels import detect_all_channels

    # Use only 2 timeframes and 2 windows for speed
    test_tfs = ['5min', '15min']
    test_windows = [10, 20]

    tsla_channel_map, tsla_resampled = detect_all_channels(
        df=tsla,
        timeframes=test_tfs,
        windows=test_windows,
        step=10,
        workers=4,
        verbose=False
    )
    log_mem("AFTER_PASS1_TSLA")

    spy_channel_map, spy_resampled = detect_all_channels(
        df=spy,
        timeframes=test_tfs,
        windows=test_windows,
        step=10,
        workers=4,
        verbose=False
    )
    log_mem("AFTER_PASS1_SPY")

    print(f"    TSLA channels: {sum(len(v) for v in tsla_channel_map.values())}")
    print(f"    SPY channels: {sum(len(v) for v in spy_channel_map.values())}")

    # Generate labels (Pass 2)
    print("\n[3] Pass 2 - Generating labels...")
    from v15.labels import generate_all_labels

    tsla_labeled_map = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    log_mem("AFTER_PASS2_TSLA")

    spy_labeled_map = generate_all_labels(
        channel_map=spy_channel_map,
        resampled_dfs=spy_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    log_mem("AFTER_PASS2_SPY")

    # Free Pass-1 artifacts
    print("\n[4] Freeing Pass-1 artifacts...")
    del tsla_channel_map, spy_channel_map
    del tsla_resampled, spy_resampled
    gc.collect()
    log_mem("AFTER_DEL_PASS1")

    # Test slim labeled maps
    print("\n[5] Creating slim labeled maps...")
    from v15.scanner import _create_slim_labeled_map

    # Count entries before
    full_entries = sum(len(v) for v in tsla_labeled_map.values())
    print(f"    Full TSLA labeled_map entries: {full_entries}")

    tsla_slim = _create_slim_labeled_map(tsla_labeled_map)
    spy_slim = _create_slim_labeled_map(spy_labeled_map)
    log_mem("AFTER_SLIM_CREATE")

    slim_entries = sum(len(v) for v in tsla_slim.values())
    print(f"    Slim TSLA map entries: {slim_entries}")

    # Delete full maps
    print("\n[6] Deleting full labeled maps...")
    del tsla_labeled_map, spy_labeled_map
    log_mem("BEFORE_GC")
    gc.collect()
    gc.collect()  # Sometimes need multiple passes
    gc.collect()
    log_mem("AFTER_DEL_FULL_MAPS")

    # Test simple Pool with slim maps
    print("\n[7] Testing Pool with slim maps...")
    from multiprocessing import Pool
    from v15.scanner import _convert_df_to_pickle_safe, _init_worker

    tsla_data = _convert_df_to_pickle_safe(tsla)
    spy_data = _convert_df_to_pickle_safe(spy)
    vix_data = _convert_df_to_pickle_safe(vix)
    log_mem("AFTER_PICKLE_SAFE")

    print("    Creating Pool with 4 workers...")
    with Pool(
        processes=4,
        initializer=_init_worker,
        initargs=(tsla_data, spy_data, vix_data, tsla_slim, spy_slim,
                  test_tfs, test_windows),
        maxtasksperchild=50
    ) as pool:
        log_mem("AFTER_POOL_CREATE")

        # Test with real-ish work
        print("    Running 20 batches of real label lookups...")
        batches = [[i*100 + 5000 for i in range(j*5, (j+1)*5)] for j in range(20)]

        for i, batch_result in enumerate(pool.imap_unordered(real_worker, batches)):
            if i % 5 == 0:
                log_mem(f"AFTER_BATCH_{i}")

        log_mem("AFTER_ALL_BATCHES")

    log_mem("AFTER_POOL_CLOSE")

    gc.collect()
    log_mem("FINAL")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
