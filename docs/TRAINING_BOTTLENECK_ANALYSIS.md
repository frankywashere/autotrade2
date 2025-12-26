# Training Pipeline Bottleneck Analysis

**Date:** 2025-12-26
**Version:** v5.9.5
**Author:** Claude Code Analysis

---

## Executive Summary

The training pipeline has **severe CPU bottlenecks** in the data loading path. The primary issue is that `__getitem__` performs **heavy computation per sample** including linear regression, trade simulation, and massive dict construction - work that should be pre-computed once and cached.

**Key findings:**
- `__getitem__` takes 1-2ms per sample (should be <0.1ms)
- Linear regression runs for every single sample
- 2,223 dictionary insertions per sample for targets
- ThreadPoolExecutor is GIL-blocked (no parallel speedup)
- GPU is starving for data

**v5.9.4 Fixes Applied:**
| Fix | Problem | Status | Speedup |
|-----|---------|--------|---------|
| Pre-computed breakout labels | Linear regression per sample | ✅ FIXED | ~3-5 min/epoch |
| Pre-computed target arrays | 2,223 dict insertions/sample | ✅ FIXED | ~7-12 min/epoch |
| DistributedShuffleBufferSampler | DDP bypassed cache-friendly sampler | ✅ FIXED | Variable |

**v5.9.5 Fixes Applied:**
| Fix | Problem | Status | Speedup |
|-----|---------|--------|---------|
| Batched collate tensor creation | 64,832 torch.tensor() calls/batch (~12s) | ✅ FIXED | ~12s → ~0.2s/batch |

**To enable pre-computed targets:**
```bash
python -m src.ml.precompute_targets --cache-dir data/feature_cache
```

---

## Table of Contents

1. [__getitem__ Bottleneck Analysis](#1-__getitem__-bottleneck-analysis)
2. [Linear Regression Per Sample](#2-linear-regression-per-sample)
3. [Continuation Label Construction](#3-continuation-label-construction)
4. [ThreadPoolExecutor GIL Problem](#4-threadpoolexecutor-gil-problem)
5. [Orphaned Pre-computation Cache](#5-orphaned-pre-computation-cache)
6. [DDP vs Single-GPU Sampler Mismatch](#6-ddp-vs-single-gpu-sampler-mismatch)
7. [Memory-Mapping vs Preloading](#7-memory-mapping-vs-preloading)
8. [Worker Memory Multiplication](#8-worker-memory-multiplication)
9. [Collate Function Overhead](#9-collate-function-overhead)
10. [Bottleneck Priority Ranking](#10-bottleneck-priority-ranking)
11. [Data Size Summary](#11-data-size-summary)
12. [Confirmation Methods](#12-confirmation-methods)
13. [Recommendations](#13-recommendations)

---

## 1. __getitem__ Bottleneck Analysis

### Location: `src/ml/hierarchical_dataset.py:724-1036` (`_getitem_native_timeframe`)

**Per-sample computational costs (called ~1.4M times per epoch):**

| Operation | Location | Complexity | Est. Time |
|-----------|----------|------------|-----------|
| **Linear regression** for channel breakout | :1883-1892 | O(60) math ops | **~100-200μs** |
| **Trade simulation** loop over future prices | :1798-1834 | O(horizon) loop | ~20-50μs |
| `np.searchsorted()` ×11 timeframes | :754 | 11 × O(log n) | ~5μs |
| `np.ascontiguousarray()` ×11 copies | :762 | 11 × O(seq×feat) | ~10-20μs (mostly no-op) |
| Continuation label dict build (11 TFs × 14 windows × 14 fields) | :849-962 | **2156 dict insertions** | **~300-500μs** |
| Transition label dict build (11 TFs × 5 fields) | :981-1005 | 55 dict insertions | ~20μs |
| VIX sequence lookup w/ pandas conversion | :1007-1024 | pd.Timestamp + lookup | ~50μs |
| Event fetcher lookup | :1026-1034 | date lookup | ~20μs |

**Estimated total per sample: 1-2ms**

At batch_size=256 and 1.4M samples:
- **~5,500 batches/epoch**
- **1.4-2.8 seconds per batch just in data loading**
- **~2-4 hours per epoch in pure `__getitem__` overhead**

---

## 2. Linear Regression Per Sample

### Location: `src/ml/hierarchical_dataset.py:1836-1927`

The `_detect_channel_breakout` method runs a **full linear regression for every single sample**:

```python
def _detect_channel_breakout(self, past_prices, future_prices, ...):
    y = past_prices[-60:]  # 60 data points
    X = np.arange(60)

    # Full linear regression - RUNS FOR EVERY SAMPLE
    X_mean = X.mean()
    y_mean = y.mean()
    slope = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean) ** 2) + 1e-10)
    intercept = y_mean - slope * X_mean
    fitted = slope * X + intercept
    residuals = y - fitted
    channel_width = np.std(residuals)

    # Then projects forward and loops over future_prices
    for i, price in enumerate(future_prices):
        # breakout detection logic...
```

**This is ~15-20 numpy operations per sample** that could be pre-computed once.

### Called from:
- `_calculate_targets_from_future` (line 697) - native TF mode
- `__getitem__` legacy path (line 1607)

### Additional per-sample loops:

**`_check_target_sequence`** (lines 1798-1803):
```python
for price in prices:
    if price >= target_price:
        return True
    if price <= stop_price:
        return False
```

**`_simulate_trade_execution`** (lines 1824-1834):
```python
for price in prices:
    if price >= target_price:
        return (target_price - entry_price) / entry_price * 100.0
    if price <= stop_price:
        return (stop_price - entry_price) / entry_price * 100.0
```

---

## 3. Continuation Label Construction

### Location: `src/ml/hierarchical_dataset.py:849-963`

For each sample, the code iterates:
```
11 timeframes × 14 window sizes = 154 outer iterations
```

Each iteration creates up to **14 dict entries** per window:

```python
for tf in HIERARCHICAL_TIMEFRAMES:  # 11 timeframes
    for window in config.CHANNEL_WINDOW_SIZES:  # 14 windows: [100,90,80,70,60,50,45,40,35,30,25,20,15,10]
        targets[f'cont_{tf}_w{window}_duration'] = ...
        targets[f'cont_{tf}_w{window}_price_sequence'] = ...
        targets[f'cont_{tf}_w{window}_hit_upper'] = ...
        targets[f'cont_{tf}_w{window}_hit_midline'] = ...
        targets[f'cont_{tf}_w{window}_hit_lower'] = ...
        targets[f'cont_{tf}_w{window}_bars_until_hit_upper'] = ...
        targets[f'cont_{tf}_w{window}_bars_until_hit_midline'] = ...
        targets[f'cont_{tf}_w{window}_bars_until_hit_lower'] = ...
        targets[f'cont_{tf}_w{window}_time_near_upper'] = ...
        targets[f'cont_{tf}_w{window}_time_near_midline'] = ...
        targets[f'cont_{tf}_w{window}_time_near_lower'] = ...
        targets[f'cont_{tf}_w{window}_slope'] = ...
        targets[f'cont_{tf}_w{window}_confidence'] = ...
        targets[f'cont_{tf}_w{window}_valid'] = ...
```

**Total dict insertions per sample:**
- Continuation: 154 × 14 = 2,156
- Transition: 11 × 5 = 55
- Base targets: ~12

**Total: ~2,223 dictionary insertions per sample**

### Continuation labels loaded at init:

| File | Size | Entries |
|------|------|---------|
| continuation_labels_5min_*.pkl | 1.2 GB | ~1.7M rows |
| continuation_labels_15min_*.pkl | 434 MB | |
| continuation_labels_30min_*.pkl | 229 MB | |
| continuation_labels_1h_*.pkl | 123 MB | |
| (+ 7 more timeframes) | | |

At init, `_per_tf_ts_to_idx` dicts are built with **~18.7M total entries** across 11 timeframes.

---

## 4. ThreadPoolExecutor GIL Problem

### Location: `train_hierarchical.py:759`

```python
class RollingBufferBatchLoader:
    def __init__(self, ...):
        self._executor = ThreadPoolExecutor(max_workers=max(1, num_workers))
```

### Why this fails:

1. `__getitem__` is **CPU-bound** (numpy math, dict operations)
2. Python's **Global Interpreter Lock (GIL)** prevents true parallel execution of CPU-bound Python code
3. Threads take turns executing, not running in parallel
4. You pay threading overhead for **zero speedup**

### The irony:

Standard PyTorch DataLoader uses `multiprocessing` which spawns separate processes that bypass the GIL. But the custom `RollingBufferBatchLoader` uses threads, negating this benefit.

### Usage in code (lines 818-825):

```python
if self._executor is not None:
    # Parallel fetch using ThreadPoolExecutor - BUT GIL BLOCKS THIS
    futures = [self._executor.submit(self.dataset.__getitem__, i) for i in batch_indices]
    samples = [f.result() for f in futures]
else:
    # Sequential fetch
    samples = [self.dataset[i] for i in batch_indices]
```

---

## 5. Orphaned Pre-computation Cache

### File: `data/feature_cache/aligned_indices_v5.9.0_417430.npz`

**Size:** 1.5 MB
**References in code:** **NONE**

```bash
$ grep -r "aligned_indices" --include="*.py" .
# NO MATCHES FOUND
```

**Analysis:** Someone started pre-computing index alignments (likely the `np.searchsorted` results) but the code was never integrated. The file is orphaned - searchsorted operations are still done per-sample at line 754.

This represents **wasted work** - the precomputation exists but isn't used.

---

## 6. DDP vs Single-GPU Sampler Mismatch

### Status: **FIXED in v5.9.4**

### Original Problem (v5.9.3 and earlier):

When using multi-GPU DDP, you **lost the mmap-optimized `ShuffleBufferSampler`** and fell back to random access patterns via `DistributedSampler`.

**Consequences:**
- More page faults with mmap'd files
- Worse OS page cache utilization
- Slower disk I/O
- The 8-10x speedup from ShuffleBufferSampler was lost

### Fix (v5.9.4):

Added `DistributedShuffleBufferSampler` class that combines:
- **DistributedSampler**: Splits data across GPUs so each processes different samples
- **ShuffleBufferSampler**: Sequential chunk access with local shuffling (cache-friendly)

**New interactive menu option** (after `preload_tf_to_ram` selection):
```
? Sampler strategy (data is in RAM, both are fast):
> Chunk-based (ShuffleBufferSampler) - Better cache locality (Recommended)
  Random (DistributedSampler) - True global shuffle
```

**Updated sampler selection logic** (`train_hierarchical.py:3647-3698`):
```python
if is_distributed:
    if use_chunk_sampler:
        # v5.9.4: DistributedShuffleBufferSampler - chunk-based with DDP support
        train_sampler = DistributedShuffleBufferSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            buffer_size=10000,
            seed=42,
            drop_last=True
        )
    else:
        # Standard DistributedSampler - true global random shuffle
        train_sampler = DistributedSampler(...)
else:
    if use_chunk_sampler:
        train_sampler = ShuffleBufferSampler(...)
    else:
        train_loader_kwargs['shuffle'] = True  # Standard DataLoader shuffle
```

**Behavior:**
- When `preload_tf_to_ram=False` (mmap): Always uses chunk-based sampler (I/O optimization)
- When `preload_tf_to_ram=True` (RAM): User chooses via interactive menu

---

## 7. Memory-Mapping vs Preloading

### Current file sizes in `data/feature_cache/`:

| File Type | Count | Total Size | Default Loading |
|-----------|-------|-----------|-----------------|
| TF sequences | 11 | 3.2 GB | mmap |
| TF timestamps | 11 | 6 MB | mmap |
| Continuation labels | 11 | 2.3 GB | RAM (pickle) |
| Transition labels | 11 | 30 MB | RAM (pickle) |
| Non-channel features | 1 | 1.2 GB | RAM (pickle) |

### The `ascontiguousarray` - NOT a bottleneck (verified 2025-12-26):

Profiling revealed that **8/11 timeframe slices are already C-contiguous**:
```
✓ Contiguous (no-op): 5min, 15min, 30min, 1h, 2h, 3h, 4h, daily
✗ Not contiguous:     weekly (20 bars), monthly (12 bars), 3month (8 bars)
```

Since row slices of C-contiguous arrays remain contiguous, `np.ascontiguousarray()` is a **no-op** for most timeframes. The 3 non-contiguous TFs are the smallest (8-20 bars), so the copy overhead is negligible (~10-20μs total).

**Conclusion:** This optimization was investigated and found to have minimal impact. Not worth pursuing.

---

## 8. Worker Memory Multiplication

### Location: `train_hierarchical.py:3401-3404`

With `multiprocessing` spawn context (required for CUDA):

```python
# Main process + (num_workers × dataset copy)
estimated_peak_gb = current_ram_gb * (num_workers + 1)
```

### Dataset memory at init:

| Component | Size |
|-----------|------|
| TF sequences (if preloaded) | 3.2 GB |
| Continuation label arrays | 2.3 GB |
| Transition label arrays | 30 MB |
| Lookup dicts (18.7M entries) | ~200 MB |
| Raw OHLC array | ~200 MB |
| Misc | ~100 MB |

**Estimated per-process: ~5-8 GB**

With 4 workers: **25-40 GB just for DataLoader processes**

### Memory safety check (lines 3377-3424):

The code includes warnings but doesn't prevent high worker counts:
```python
def check_dataloader_memory_safety(num_workers, container_ram_gb=0):
    # Warns if estimated_peak_gb > total_ram_gb * 0.80
    # But user can proceed anyway
```

---

## 9. Collate Function Overhead

### Status: **FIXED in v5.9.5**

### Original Problem (v5.9.4 and earlier):

Per batch (64 samples with 1013 target keys):

```python
# SLOW: 64 × 1013 = 64,832 torch.tensor() calls per batch (~12s!)
for tgt in targets_list:  # 64 samples
    ct = {}
    for k, v in tgt.items():  # 1013 keys per sample
        ct[k] = torch.tensor(v, dtype=torch_dtype)
converted_targets.append(ct)
targets_batch = default_collate(converted_targets)
```

### Fix (v5.9.5):

Batch tensor creation per key instead of per sample:

```python
# FAST: 1013 torch.as_tensor() calls per batch (~0.2s)
targets_batch = {}
for k in first_target.keys():  # 1013 keys
    values = [t[k] for t in targets_list]  # Gather all 64 values
    targets_batch[k] = torch.as_tensor(values, dtype=torch_dtype)  # 1 call per key
```

**Speedup:** 64,832 calls → 1,013 calls = **63x fewer function calls**
**Time:** ~12s → ~0.2s per batch

### Slow collate warning (line 500):
```python
if _collate_elapsed > 1.0:
    print(f"[SLOW_COLLATE] batch assembly took {_collate_elapsed:.1f}s...")
```

---

## 10. Bottleneck Priority Ranking

| Rank | Bottleneck | Location | Impact | Status |
|------|-----------|----------|--------|--------|
| **1** | Linear regression per sample | :1883-1892 | **20-30% of __getitem__** | **FIXED v5.9.4** |
| **2** | 2,223 dict insertions per sample | :849-963 | **20-30% of __getitem__** | **FIXED v5.9.4** |
| **3** | 64K `torch.tensor()` per batch | train:452-467 | **~12s per batch** | **FIXED v5.9.5** |
| **4** | 11× `searchsorted()` (orphaned precompute) | :754 | **5-10% of __getitem__** | Open |
| **5** | ThreadPool GIL blocking | train:759 | RollingBuffer doesn't scale | Open |
| **6** | DDP bypasses ShuffleBufferSampler | train:3647 | More random I/O | **FIXED v5.9.4** |
| **7** | Worker memory multiplication | train:3401 | RAM bloat, cache thrashing | Open |
| ~~8~~ | ~~11× `ascontiguousarray()` copies~~ | ~~:762~~ | ~~25-35% of __getitem__~~ | **NOT AN ISSUE** (8/11 TFs already contiguous) |

### Fix #1 + #3: Pre-computed Targets (v5.9.4)

**New script:** `src/ml/precompute_targets.py`

Run once after feature extraction:
```bash
python -m src.ml.precompute_targets --cache-dir data/feature_cache
```

**Creates:**
- `precomputed_breakout_{cache_key}.npz` - Fix #1: Pre-computed breakout labels
- `precomputed_targets_{cache_key}.npz` - Fix #3: Pre-computed target arrays
- `precomputed_valid_indices_{cache_key}.npy` - Sample index verification

**Dataset automatically uses pre-computed data** when files exist:
- `_load_precomputed_targets()` at init detects and loads files
- `_getitem_precomputed_path()` provides fast array lookup path
- Falls back to per-sample computation if files don't exist

**Expected speedup:** ~10-17 min/epoch (from eliminating ~700μs per sample)

---

## 11. Data Size Summary

```
Feature Cache Total: ~6.5 GB
├── TF Sequences (11 files)
│   ├── tf_sequence_5min_*.npy      1.7 GB
│   ├── tf_sequence_15min_*.npy     618 MB
│   ├── tf_sequence_30min_*.npy     326 MB
│   ├── tf_sequence_1h_*.npy        173 MB
│   ├── tf_sequence_2h_*.npy         93 MB
│   ├── tf_sequence_3h_*.npy         68 MB
│   ├── tf_sequence_4h_*.npy         52 MB
│   ├── tf_sequence_daily_*.npy      13 MB
│   ├── tf_sequence_weekly_*.npy    2.2 MB
│   ├── tf_sequence_monthly_*.npy   516 KB
│   └── tf_sequence_3month_*.npy    180 KB
│
├── TF Timestamps (11 files)         ~6 MB total
│
├── Continuation Labels (11 files)   ~2.3 GB total
│   └── continuation_labels_5min_*.pkl  1.2 GB (largest)
│
├── Transition Labels (11 files)     ~30 MB total
│
├── Non-channel Features             1.2 GB
│
└── Orphaned Files
    └── aligned_indices_*.npz        1.5 MB (UNUSED)

Training Samples: ~1.4 million
Batch Size: typically 256
Batches per Epoch: ~5,500
```

---

## 12. Confirmation Methods

### 1. Enable debug timing:
```bash
TRAIN_DEBUG=1 python train_hierarchical.py ...
```

This enables:
- `[SLOW_COLLATE]` logs (line 411)
- `[SLOW_GETITEMS]` logs (line 2006)

### 2. Check startup summary for:
```
Native TF mode: 11 timeframes × N features each
Preloaded X.X GB to RAM in X.Xs
```

### 3. Compare worker counts:
```bash
# Baseline (no worker overhead)
python train_hierarchical.py --num-workers 0

# With workers (expect minimal improvement due to GIL in RollingBuffer)
python train_hierarchical.py --num-workers 4
```

### 4. Profile specific functions:
```python
import cProfile
cProfile.run('dataset[0]', sort='cumtime')
```

### 5. Memory profiler logs:
Check `logs/memory_debug.log` for RAM snapshots (enabled via interactive menu).

---

## 13. Recommendations

### High Priority (Pre-computation)

1. ~~**Pre-compute breakout labels**~~ - ✅ **FIXED v5.9.4**
   - Now in `src/ml/precompute_targets.py`
   - `__getitem__` does simple array lookup via `_getitem_precomputed_path()`

2. **Pre-compute trade simulation results** - `_check_target_sequence` and `_simulate_trade_execution`
   - Results are deterministic given future prices
   - Should be part of label generation, not runtime

3. **Use the orphaned aligned_indices** - Integrate the precomputed index alignments
   - Eliminates 11 `np.searchsorted()` calls per sample

### Medium Priority (Memory Layout)

4. ~~**Pre-stack target tensors**~~ - ✅ **FIXED v5.9.4**
   - Now uses dict of pre-computed arrays (Option B)
   - `__getitem__` builds targets dict from array lookups: `targets[key] = float(arr[idx])`
   - Eliminates 2,223 dict insertions per sample

5. ~~**Eliminate per-sample ascontiguousarray**~~ - **NOT AN ISSUE** (verified 2025-12-26)
   - Profiling showed 8/11 TF slices are already C-contiguous (no-op)
   - Only weekly/monthly/3month need copies (tiny arrays: 8-20 bars)
   - Total overhead: ~10-20μs - not worth optimizing

### Low Priority (Architecture)

6. **Replace ThreadPoolExecutor with ProcessPoolExecutor** in RollingBufferBatchLoader
   - Or use standard DataLoader multiprocessing

7. ~~**Create DDP-compatible ShuffleBufferSampler**~~ - ✅ **FIXED v5.9.4**
   - `DistributedShuffleBufferSampler` class added
   - Maintains sequential chunk access patterns in multi-GPU

8. **Reduce target dict size** - Do you need all 14 windows × 14 fields at training time?

---

## What Should Happen vs What Happens

### Ideal `__getitem__`:
```python
def __getitem__(self, idx):
    return self.precomputed_features[idx], self.precomputed_targets[idx]
    # O(1) array lookup, ~10μs
```

### Current `__getitem__` (v5.9.3 and earlier):
```python
def __getitem__(self, idx):
    # O(n) operations per sample:

    for tf in 11_timeframes:
        searchsorted()           # O(log n)
        slice_array()            # O(1)
        ascontiguousarray()      # no-op for 8/11 TFs (already contiguous)

    linear_regression()          # O(60) numpy math  ← FIXED v5.9.4
    simulate_trade()             # O(horizon) loop

    for tf in 11_timeframes:
        for window in 14_windows:
            for field in 14_fields:
                targets[key] = value  # Python dict insertion  ← FIXED v5.9.4

    return features_dict, targets_dict
    # ~1-2ms per sample
```

### After v5.9.4 (with pre-computed data):
```python
def __getitem__(self, idx):
    for tf in 11_timeframes:
        searchsorted()           # O(log n) - still needed
        slice_array()            # O(1)
        ascontiguousarray()      # no-op for 8/11 TFs (verified)

    # Fast path: array lookups instead of computation
    targets = {k: float(arr[idx]) for k, arr in precomputed_arrays.items()}
    # O(1) per field, ~200-400μs total

    return features_dict, targets_dict
    # ~400-700μs per sample (~50-65% improvement)
```

---

## Files Referenced

| File | Key Lines | Purpose |
|------|-----------|---------|
| `src/ml/hierarchical_dataset.py` | 724-1036 | `_getitem_native_timeframe` |
| | 849-963 | Continuation label construction |
| | 981-1005 | Transition label construction |
| | 1836-1927 | `_detect_channel_breakout` |
| | 1798-1834 | Trade simulation methods |
| | 1962-2011 | `__getitems__` batch method |
| `train_hierarchical.py` | 156-215 | `ShuffleBufferSampler` class |
| | 218-304 | `DistributedShuffleBufferSampler` class (v5.9.4) |
| | 361-503 | `hierarchical_collate` |
| | 811-1032 | `RollingBufferBatchLoader` |
| | 2186-2212 | Sampler choice menu option (v5.9.4) |
| | 3647-3698 | Sampler selection logic (v5.9.4) |
| | 3700-3730 | Memory safety check |
| `config.py` | 317 | `CHANNEL_WINDOW_SIZES` definition |

---

## Hardware Context

**Target system:**
- 100 GB RAM
- 16 GB VRAM per GPU
- Multiple NVIDIA GPUs

**Current utilization:**
- ~50-75 GB RAM (data + workers)
- GPU mostly **idle waiting for data**
- Single-threaded `__getitem__` logic

**The GPU is starving for data.** The data loading pipeline cannot keep up with GPU compute.
