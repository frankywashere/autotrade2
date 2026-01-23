#!/usr/bin/env python3
"""
Minimal test to check if Pool initialization with slim maps hangs.
"""
import time
import sys

def simple_worker(x):
    """Minimal worker."""
    return x * 2

def main():
    print("Testing Pool initialization...")

    # Import what we need
    from multiprocessing import Pool
    from v15.data import load_market_data
    from v15.scanner import (
        _convert_df_to_pickle_safe,
        _init_worker,
        _create_slim_labeled_map
    )
    from v15.labels import detect_all_channels, generate_all_labels

    # Load data
    print("[1] Loading data...")
    tsla, spy, vix = load_market_data("data")
    print(f"    Loaded {len(tsla)} bars")

    # Create minimal channel maps (small subset)
    print("[2] Creating small channel maps...")
    test_tfs = ['5min']
    test_windows = [10]

    tsla_channel_map, tsla_resampled = detect_all_channels(
        df=tsla,
        timeframes=test_tfs,
        windows=test_windows,
        step=100,  # Very coarse
        workers=1,  # Sequential for this part
        verbose=False
    )
    print(f"    TSLA channels: {sum(len(v) for v in tsla_channel_map.values())}")

    spy_channel_map, spy_resampled = detect_all_channels(
        df=spy,
        timeframes=test_tfs,
        windows=test_windows,
        step=100,
        workers=1,
        verbose=False
    )
    print(f"    SPY channels: {sum(len(v) for v in spy_channel_map.values())}")

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
    print(f"    Pickle-safe data ready")

    # Test Pool initialization
    print("[6] Testing Pool initialization with 2 workers...")
    print("    Creating Pool...")
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
            elapsed = time.time() - start
            print(f"    Pool created in {elapsed:.2f}s")

            # Try running a simple task
            print("    Running simple task...")
            sys.stdout.flush()

            results = list(pool.map(simple_worker, range(5), chunksize=1))
            print(f"    Results: {results}")

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
