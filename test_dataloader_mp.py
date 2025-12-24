#!/usr/bin/env python3
"""
Test DataLoader multiprocessing with mmap sharing (v5.9.4).

Tests both:
1. preload_tf_to_ram=False (mmap) - should have small pickle, shared memory
2. preload_tf_to_ram=True - should work but copies data to each worker
"""

import sys
import time
import pickle
import torch
from torch.utils.data import DataLoader

def get_memory_mb():
    """Get current process memory in MB."""
    import psutil
    return psutil.Process().memory_info().rss / 1e6

def test_mmap_mode():
    """Test with mmap (preload_tf_to_ram=False) - should share memory."""
    print("\n" + "="*70)
    print("TEST 1: MMAP MODE (preload_tf_to_ram=False)")
    print("="*70)

    from src.ml.hierarchical_dataset import HierarchicalDataset

    tf_meta_path = '/Users/frank/Desktop/CodingProjects/exp/data/feature_cache/tf_meta_v5.9.0_vixv1_evv1_projv2_bdv3_pbv4_contv2_20150102_20250927_1692233_vix1764643608_ev1765934884_h24.json'

    mem_before = get_memory_mb()
    print(f"\n1. Creating dataset with mmap (preload_tf_to_ram=False)...")
    start = time.time()
    dataset = HierarchicalDataset(
        use_native_timeframes=True,
        tf_meta_path=tf_meta_path,
        preload_tf_to_ram=False,  # Use mmap
    )
    print(f"   Dataset created in {time.time() - start:.1f}s")
    print(f"   Samples: {len(dataset):,}")
    print(f"   Memory after init: {get_memory_mb() - mem_before:.0f} MB")

    # Check pickle size
    print("\n2. Testing pickle size (should be ~5-10 MB, not 3.2 GB)...")
    start = time.time()
    pickled = pickle.dumps(dataset)
    pickle_time = time.time() - start
    pickle_mb = len(pickled) / 1e6
    print(f"   ✓ Pickle size: {pickle_mb:.1f} MB (took {pickle_time:.1f}s)")

    if pickle_mb < 50:
        print(f"   ✓ GOOD: Pickle is small - mmap arrays excluded!")
    else:
        print(f"   ⚠️  WARNING: Pickle is large - may be copying data")

    # Test unpickle
    print("\n3. Testing unpickle (simulates worker spawn)...")
    start = time.time()
    dataset2 = pickle.loads(pickled)
    unpickle_time = time.time() - start
    print(f"   ✓ Unpickled in {unpickle_time:.1f}s")
    print(f"   ✓ Samples: {len(dataset2):,}")

    # Verify data access works
    sample = dataset2[0]
    print(f"   ✓ Sample access works")

    del pickled, dataset2

    # Test DataLoader
    print("\n4. Baseline: DataLoader with num_workers=0...")
    loader_0 = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    _ = next(iter(loader_0))  # Warmup

    start = time.time()
    for i, batch in enumerate(loader_0):
        if i >= 50:
            break
    time_0 = (time.time() - start) / 50
    print(f"   50 batches @ {time_0*1000:.1f}ms per batch")

    # Test with workers
    print("\n5. Testing DataLoader with num_workers=4 (mmap sharing)...")
    mem_before_workers = get_memory_mb()

    try:
        loader_4 = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )

        start = time.time()
        batch = next(iter(loader_4))
        warmup_time = time.time() - start
        print(f"   Warmup (spawn workers): {warmup_time:.1f}s")

        mem_after_workers = get_memory_mb()
        print(f"   Memory increase from workers: {mem_after_workers - mem_before_workers:.0f} MB")

        start = time.time()
        for i, batch in enumerate(loader_4):
            if i >= 50:
                break
        time_4 = (time.time() - start) / 50
        print(f"   ✓ 50 batches @ {time_4*1000:.1f}ms per batch")
        print(f"   Speedup vs num_workers=0: {time_0/time_4:.2f}x")

        del loader_4

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        time_4 = float('inf')

    return time_0, time_4, pickle_mb


def test_preload_mode():
    """Test with preload (preload_tf_to_ram=True) - for comparison."""
    print("\n" + "="*70)
    print("TEST 2: PRELOAD MODE (preload_tf_to_ram=True)")
    print("="*70)

    from src.ml.hierarchical_dataset import HierarchicalDataset

    tf_meta_path = '/Users/frank/Desktop/CodingProjects/exp/data/feature_cache/tf_meta_v5.9.0_vixv1_evv1_projv2_bdv3_pbv4_contv2_20150102_20250927_1692233_vix1764643608_ev1765934884_h24.json'

    mem_before = get_memory_mb()
    print(f"\n1. Creating dataset with preload_tf_to_ram=True...")
    start = time.time()
    dataset = HierarchicalDataset(
        use_native_timeframes=True,
        tf_meta_path=tf_meta_path,
        preload_tf_to_ram=True,
    )
    print(f"   Dataset created in {time.time() - start:.1f}s")
    print(f"   Memory after init: {get_memory_mb() - mem_before:.0f} MB")

    # Check pickle size
    print("\n2. Testing pickle size...")
    start = time.time()
    pickled = pickle.dumps(dataset)
    pickle_time = time.time() - start
    pickle_mb = len(pickled) / 1e6
    print(f"   Pickle size: {pickle_mb:.1f} MB (took {pickle_time:.1f}s)")

    if pickle_mb > 1000:
        print(f"   ⚠️  Large pickle - each worker gets a copy!")

    del pickled

    # Test DataLoader
    print("\n3. Baseline: DataLoader with num_workers=0...")
    loader_0 = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    _ = next(iter(loader_0))

    start = time.time()
    for i, batch in enumerate(loader_0):
        if i >= 50:
            break
    time_0 = (time.time() - start) / 50
    print(f"   50 batches @ {time_0*1000:.1f}ms per batch")

    print("\n4. Testing DataLoader with num_workers=4 (copies data)...")
    mem_before_workers = get_memory_mb()

    try:
        loader_4 = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )

        start = time.time()
        batch = next(iter(loader_4))
        warmup_time = time.time() - start
        print(f"   Warmup (spawn workers): {warmup_time:.1f}s")

        mem_after_workers = get_memory_mb()
        print(f"   Memory increase from workers: {mem_after_workers - mem_before_workers:.0f} MB")

        start = time.time()
        for i, batch in enumerate(loader_4):
            if i >= 50:
                break
        time_4 = (time.time() - start) / 50
        print(f"   ✓ 50 batches @ {time_4*1000:.1f}ms per batch")
        print(f"   Speedup vs num_workers=0: {time_0/time_4:.2f}x")

        del loader_4

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        time_4 = float('inf')

    return time_0, time_4, pickle_mb


def main():
    print("="*70)
    print("DataLoader Multiprocessing Test - v5.9.4 mmap sharing")
    print("="*70)

    # Test mmap mode (should share memory)
    mmap_time_0, mmap_time_4, mmap_pickle = test_mmap_mode()

    # Test preload mode (copies data but may be faster per-sample)
    preload_time_0, preload_time_4, preload_pickle = test_preload_mode()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Mode':<20} {'Pickle Size':<15} {'num_workers=0':<15} {'num_workers=4':<15} {'Speedup':<10}")
    print("-"*70)
    print(f"{'mmap':<20} {mmap_pickle:>10.1f} MB   {mmap_time_0*1000:>10.1f}ms   {mmap_time_4*1000:>10.1f}ms   {mmap_time_0/mmap_time_4:>6.2f}x")
    print(f"{'preload':<20} {preload_pickle:>10.1f} MB   {preload_time_0*1000:>10.1f}ms   {preload_time_4*1000:>10.1f}ms   {preload_time_0/preload_time_4:>6.2f}x")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    if mmap_pickle < 50:
        print("✓ MMAP mode with num_workers=4: Best for memory efficiency")
        print("  - Small pickle = fast worker spawn, shared memory")
        print("  - Use: preload_tf_to_ram=False + Standard DataLoader + num_workers=4")
    print("="*70)


if __name__ == '__main__':
    main()
