#!/usr/bin/env python3
"""
Test with FULL timeframes/windows to see if size causes hanging.
"""
import time
import sys
import psutil

def get_mem():
    proc = psutil.Process()
    main = proc.memory_info().rss / (1024**2)
    children = proc.children(recursive=True)
    child_mem = sum(c.memory_info().rss / (1024**2) for c in children)
    return main, child_mem

def log_mem(label):
    main, child = get_mem()
    print(f"[MEM] {label}: Main={main:.0f}MB, Children={child:.0f}MB, Total={main+child:.0f}MB")
    sys.stdout.flush()

def main():
    print("Testing with FULL timeframes/windows...")
    log_mem("START")

    from multiprocessing import Pool
    from v15.data import load_market_data
    from v15.scanner import (
        _convert_df_to_pickle_safe,
        _init_worker,
        _create_slim_labeled_map,
        _process_batch_with_globals,
    )
    from v15.labels import detect_all_channels, generate_all_labels
    from v15.config import TIMEFRAMES
    from v15.dtypes import STANDARD_WINDOWS

    # Use FULL TFs and windows like real scanner
    full_tfs = list(TIMEFRAMES)
    full_windows = list(STANDARD_WINDOWS)
    print(f"Using {len(full_tfs)} TFs: {full_tfs}")
    print(f"Using {len(full_windows)} windows: {full_windows}")

    # Load data
    print("\n[1] Loading data...")
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")
    log_mem("AFTER_LOAD")

    # Create channel maps with FULL timeframes/windows
    print("\n[2] Creating channel maps (FULL TFs/windows)...")
    print("    This will take a while...")
    sys.stdout.flush()

    tsla_channel_map, tsla_resampled = detect_all_channels(
        df=tsla,
        timeframes=full_tfs,
        windows=full_windows,
        step=1,  # Like real scanner
        workers=4,
        verbose=False
    )
    log_mem("AFTER_TSLA_CHANNELS")
    print(f"    TSLA: {sum(len(v) for v in tsla_channel_map.values())} channels")

    spy_channel_map, spy_resampled = detect_all_channels(
        df=spy,
        timeframes=full_tfs,
        windows=full_windows,
        step=1,
        workers=4,
        verbose=False
    )
    log_mem("AFTER_SPY_CHANNELS")
    print(f"    SPY: {sum(len(v) for v in spy_channel_map.values())} channels")

    # Generate labels
    print("\n[3] Generating labels...")
    sys.stdout.flush()

    tsla_labeled_map = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    log_mem("AFTER_TSLA_LABELS")

    spy_labeled_map = generate_all_labels(
        channel_map=spy_channel_map,
        resampled_dfs=spy_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    log_mem("AFTER_SPY_LABELS")

    # Free Pass-1 artifacts
    del tsla_channel_map, spy_channel_map
    del tsla_resampled, spy_resampled
    import gc
    gc.collect()
    log_mem("AFTER_DEL_PASS1")

    # Create slim maps
    print("\n[4] Creating slim maps...")
    tsla_slim = _create_slim_labeled_map(tsla_labeled_map)
    spy_slim = _create_slim_labeled_map(spy_labeled_map)
    log_mem("AFTER_SLIM_CREATE")

    # Delete full maps
    del tsla_labeled_map, spy_labeled_map
    gc.collect()
    log_mem("AFTER_DEL_FULL_MAPS")

    # Create pickle-safe data
    print("\n[5] Creating pickle-safe data...")
    tsla_data = _convert_df_to_pickle_safe(tsla)
    spy_data = _convert_df_to_pickle_safe(spy)
    vix_data = _convert_df_to_pickle_safe(vix)
    log_mem("AFTER_PICKLE_SAFE")

    # Create small test batches
    batches = [[5000 + i*100] for i in range(5)]  # Just 5 single-position batches
    print(f"    Created {len(batches)} test batches")

    # Test Pool with 2 workers
    print("\n[6] Testing Pool with 2 workers and FULL TFs/windows...")
    sys.stdout.flush()

    start = time.time()
    try:
        print("    Creating Pool...")
        sys.stdout.flush()

        with Pool(
            processes=2,
            initializer=_init_worker,
            initargs=(tsla_data, spy_data, vix_data, tsla_slim, spy_slim,
                      full_tfs, full_windows),
            maxtasksperchild=50
        ) as pool:
            elapsed = time.time() - start
            print(f"    Pool created in {elapsed:.1f}s")
            log_mem("AFTER_POOL_CREATE")
            sys.stdout.flush()

            print("    Processing batches...")
            sys.stdout.flush()

            processed = 0
            for batch_results in pool.imap_unordered(_process_batch_with_globals, batches):
                processed += 1
                n_samples = sum(1 for r in batch_results if r.get('sample'))
                n_errors = sum(1 for r in batch_results if r.get('error'))
                print(f"      Batch {processed}/{len(batches)}: {n_samples} samples, {n_errors} errors")
                log_mem(f"AFTER_BATCH_{processed}")
                sys.stdout.flush()

            print(f"    All {processed} batches processed!")

        log_mem("AFTER_POOL_CLOSE")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    gc.collect()
    log_mem("FINAL")

    print("\nDone!")


if __name__ == "__main__":
    main()
