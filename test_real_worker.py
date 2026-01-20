#!/usr/bin/env python3
"""
Test with the REAL worker function to see if that's where it hangs.
"""
import time
import sys

def main():
    print("Testing with REAL worker function...")

    from multiprocessing import Pool
    from v15.data import load_market_data
    from v15.scanner import (
        _convert_df_to_pickle_safe,
        _init_worker,
        _create_slim_labeled_map,
        _process_batch_with_globals,  # The REAL worker
    )
    from v15.labels import detect_all_channels, generate_all_labels
    from v15.config import TIMEFRAMES
    from v15.dtypes import STANDARD_WINDOWS

    # Load data
    print("[1] Loading data...")
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")

    # Create channel maps with FULL timeframes/windows like real scanner
    print("[2] Creating channel maps (2 TFs, 2 windows for speed)...")
    test_tfs = ['5min', '15min']
    test_windows = [10, 20]

    tsla_channel_map, tsla_resampled = detect_all_channels(
        df=tsla,
        timeframes=test_tfs,
        windows=test_windows,
        step=10,
        workers=2,
        verbose=False
    )
    spy_channel_map, spy_resampled = detect_all_channels(
        df=spy,
        timeframes=test_tfs,
        windows=test_windows,
        step=10,
        workers=2,
        verbose=False
    )
    print(f"    TSLA: {sum(len(v) for v in tsla_channel_map.values())} channels")
    print(f"    SPY: {sum(len(v) for v in spy_channel_map.values())} channels")

    # Generate labels
    print("[3] Generating labels...")
    tsla_labeled_map = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    spy_labeled_map = generate_all_labels(
        channel_map=spy_channel_map,
        resampled_dfs=spy_resampled,
        labeling_method="forward_scan",
        verbose=False
    )
    print(f"    Labels generated")

    # Create slim maps
    print("[4] Creating slim maps...")
    tsla_slim = _create_slim_labeled_map(tsla_labeled_map)
    spy_slim = _create_slim_labeled_map(spy_labeled_map)
    print(f"    Slim maps created")

    # Create pickle-safe data
    print("[5] Creating pickle-safe data...")
    tsla_data = _convert_df_to_pickle_safe(tsla)
    spy_data = _convert_df_to_pickle_safe(spy)
    vix_data = _convert_df_to_pickle_safe(vix)

    # Create test batches
    batches = [[5000 + i*50 for i in range(j*5, (j+1)*5)] for j in range(4)]
    print(f"    Created {len(batches)} batches")

    # Test Pool with REAL worker
    print("[6] Testing Pool with REAL _process_batch_with_globals...")
    print(f"    Workers: 2")
    sys.stdout.flush()

    start = time.time()
    try:
        with Pool(
            processes=2,
            initializer=_init_worker,
            initargs=(tsla_data, spy_data, vix_data, tsla_slim, spy_slim,
                      test_tfs, test_windows),
            maxtasksperchild=50
        ) as pool:
            print(f"    Pool created in {time.time()-start:.2f}s")
            sys.stdout.flush()

            print("    Processing batches with imap_unordered...")
            sys.stdout.flush()

            processed = 0
            for batch_results in pool.imap_unordered(_process_batch_with_globals, batches):
                processed += 1
                n_samples = sum(1 for r in batch_results if r.get('sample'))
                n_errors = sum(1 for r in batch_results if r.get('error'))
                print(f"      Batch {processed}/{len(batches)}: {n_samples} samples, {n_errors} errors")
                sys.stdout.flush()

            print(f"    All {processed} batches processed!")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
