#!/usr/bin/env python3
"""Quick test to verify TF32 and data loading settings."""

import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)

# Check TF32 status
print(f"\n1. TF32 Status:")
print(f"   torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}")
print(f"   Expected: 'medium' for TF32, 'highest' for standard FP32")

# Check if we can detect mmap vs RAM
print(f"\n2. Data Loading Test:")
cache_dir = Path('data/feature_cache')
test_file = list(cache_dir.glob('tf_sequence_5min*.npy'))[0]

print(f"   Loading {test_file.name}...")

# Method 1: mmap
import time
t0 = time.perf_counter()
mmap_arr = np.load(str(test_file), mmap_mode='r')
t1 = time.perf_counter()
print(f"   mmap load: {(t1-t0)*1000:.1f} ms (should be ~1ms - just creates pointer)")

# Method 2: Full load to RAM (what preload does)
t0 = time.perf_counter()
ram_arr = np.array(mmap_arr)
t1 = time.perf_counter()
print(f"   Copy to RAM: {(t1-t0)*1000:.1f} ms (should be ~500-1000ms for 1.76 GB)")

print(f"   Array size: {ram_arr.nbytes / 1e9:.2f} GB")
print(f"   Is mmap: {isinstance(mmap_arr, np.memmap)}")
print(f"   Is ndarray: {isinstance(ram_arr, np.ndarray) and not isinstance(ram_arr, np.memmap)}")

# GPU info
print(f"\n3. GPU Info:")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")

    # Check if GPU supports TF32
    major = torch.cuda.get_device_capability()[0]
    print(f"   Compute capability: {torch.cuda.get_device_capability()}")
    if major >= 8:
        print(f"   ✓ GPU supports TF32 (Ampere or newer)")
    else:
        print(f"   ✗ GPU does NOT support TF32 (needs Ampere/8.0+)")
else:
    print(f"   ✗ CUDA not available")

print("\n" + "=" * 70)
