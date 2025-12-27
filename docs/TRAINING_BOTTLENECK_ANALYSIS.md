# Training Pipeline Bottleneck Analysis

**Date:** 2025-12-26
**Version:** v5.9.6
**Author:** Claude Code Analysis

---

## Executive Summary

The training pipeline has been **significantly optimized** through targeted fixes to data loading and loss calculation bottlenecks.

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

**v5.9.6 Fixes Applied:**
| Fix | Problem | Status | Speedup |
|-----|---------|--------|---------|
| Vectorized loss calculation | 20,000+ loop iterations per batch | ✅ FIXED | 5700ms → 700ms (8x) |
| Memory-mapped label loading | 2.3 GB copied per worker | ✅ FIXED | ~18 GB RAM saved (8 workers) |
| Label generation NaN bug | Invalid price_sequence data | ✅ FIXED | Data integrity |
| Single-GPU sampler epoch | ShuffleBufferSampler epoch not updated | ✅ FIXED | Better convergence |

**Current Performance (v5.9.6):**
```
Total time per batch: ~7.2s (down from ~12.5s in v5.9.3)
├── data:     140ms (2%)
├── forward: 1240ms (17%)
├── loss:     700ms (10%) ← Was 5700ms before v5.9.6!
├── backward: 5000ms (69%) ← Now the main bottleneck
└── optimizer: 65ms (1%)
```

**To enable pre-computed targets:**
```bash
python -m src.ml.precompute_targets --cache-dir data/feature_cache
```

---

## Top 3 Remaining Optimizations

Based on current timing (v5.9.6), ranked by impact:

### 1. Backward Pass Optimization (5000ms, 69% of batch time)

**Problem:** Computing gradients through the complex loss calculation graph takes 5 seconds.

**Current architecture issues:**
- Multi-TF loss loops (11 timeframes)
- Validity/transition losses with per-TF masking
- Large computation graph from geometric predictions

**Potential fixes:**
- **Model architecture simplification** - Reduce computation graph complexity
- **Gradient checkpointing** - Trade compute for memory, may help with large graphs
- **Mixed precision training** - Use AMP (currently disabled, was using TF32)
- **Reduce loss components** - Remove or simplify low-impact losses

**Expected speedup:** 20-40% (5000ms → 3000-4000ms)
**Effort:** High (requires architectural changes)

### 2. Forward Pass Optimization (1240ms, 17% of batch time)

**Problem:** Model forward pass through 11 hierarchical timeframe layers.

**Current architecture:**
- 11 separate LSTM/GRU layers (one per timeframe)
- Geometric prediction heads per TF
- Compositor networks
- VIX CfC integration

**Potential fixes:**
- **torch.compile()** - JIT compilation (already supported via `--use-compile`)
- **Reduce hidden_size** - Currently 128, try 96 or 64
- **Simplify geometric heads** - Fewer projection calculations
- **Profile which layers are slowest** - Target specific bottlenecks

**Expected speedup:** 10-20% (1240ms → 1000-1100ms)
**Effort:** Medium

### 3. Data Loading (140ms, 2% of batch time)

**Problem:** Already very fast, but could be faster with more workers.

**Current limitation:**
- `num_workers=0` (single-threaded) due to MPS device
- With mmap labels, multi-worker is now RAM-efficient

**Potential fixes:**
- **Enable workers on CPU-only setup** - Test if mmap sharing works
- **Increase to 2-4 workers** - Should reduce data time to ~50-70ms
- **Pre-fetch batches** - Use `prefetch_factor=2` in DataLoader

**Expected speedup:** 50% (140ms → 70ms)
**Effort:** Low (just enable workers, test RAM usage)

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

### Status: **OPTIMIZED in v5.9.6**

### Current file sizes in `data/feature_cache/`:

| File Type | Count | Total Size | v5.9.6 Loading | Shared across workers? |
|-----------|-------|-----------|----------------|------------------------|
| TF sequences | 11 | 3.2 GB | mmap or RAM (user choice) | ✅ Yes (if mmap) |
| TF timestamps | 11 | 6 MB | mmap | ✅ Yes |
| Continuation labels | 11 | ~800 MB | **mmap (.mmap/ dirs)** | ✅ Yes (auto-converts) |
| Transition labels | 11 | ~22 MB | **mmap (.mmap/ dirs)** | ✅ Yes (auto-converts) |
| Precomputed targets | 2 | ~17 MB | **mmap (.npz)** | ✅ Yes |
| Non-channel features | 1 | 1.2 GB | Deleted (not used in native TF) | N/A |

### RAM Usage Comparison (4 workers × 2 GPUs = 8 processes):

| Component | Before v5.9.6 | After v5.9.6 | Saved |
|-----------|---------------|--------------|-------|
| Continuation labels | 2.3 GB × 8 = 18.4 GB | 2.3 GB (shared) | ~16 GB |
| Transition labels | 30 MB × 8 = 240 MB | 30 MB (shared) | ~210 MB |
| Precomputed targets | 17 MB × 8 = 136 MB | 17 MB (shared) | ~120 MB |
| **Total** | **~18.8 GB** | **~2.4 GB** | **~16.4 GB** |

### Auto-Conversion System (v5.9.6):

When pickle files are loaded (first time or after regeneration):
1. Dataset loads from pickle
2. Synchronously converts pickle → `.mmap/` directory with individual `.npy` files
3. Reloads from mmap for current run
4. Future runs use mmap directly (skip pickle)

**Benefits:**
- No manual conversion needed
- Current training run gets RAM savings
- mmap files shared via OS page cache across all workers

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

## 10. Bottleneck Priority Ranking (v5.9.6 Updated)

### Completed Fixes:

| Rank | Bottleneck | Location | Impact | Status |
|------|-----------|----------|--------|--------|
| **1** | Nested Python loops in loss | train:4355-4480 | **5700ms → 700ms** | **FIXED v5.9.6** |
| **2** | Linear regression per sample | dataset:1883-1892 | **20-30% of __getitem__** | **FIXED v5.9.4** |
| **3** | 2,223 dict insertions per sample | dataset:849-963 | **20-30% of __getitem__** | **FIXED v5.9.4** |
| **4** | 64K `torch.tensor()` per batch | train:452-467 | **~12s per batch** | **FIXED v5.9.5** |
| **5** | DDP bypasses ShuffleBufferSampler | train:3647 | More random I/O | **FIXED v5.9.4** |
| **6** | Worker memory multiplication | dataset init | 2.3 GB × workers | **FIXED v5.9.6** (mmap) |
| **7** | Single-GPU sampler epoch bug | train:4040 | Same shuffle/epoch | **FIXED v5.9.6** |

### Remaining Bottlenecks:

| Rank | Bottleneck | Location | Impact | Effort |
|------|-----------|----------|--------|--------|
| **1** | Backward pass | train:4577 | **5000ms (69%)** | High - Model redesign |
| **2** | Forward pass | train:4180 | **1240ms (17%)** | Medium - torch.compile or profile |
| **3** | Data loading | Single-threaded | **140ms (2%)** | Low - Enable workers |
| ~~4~~ | ~~11× `searchsorted()`~~ | ~~dataset:754~~ | ~~Minimal now~~ | Negligible with preload |
| ~~5~~ | ~~ThreadPool GIL~~ | ~~train:759~~ | ~~No longer used~~ | Deprecated |
| ~~6~~ | ~~`ascontiguousarray()`~~ | ~~dataset:762~~ | ~~No-op for 8/11 TFs~~ | **NOT AN ISSUE** |

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

**Validation:** Run `python scripts/validate_precomputed.py` to verify precomputed values match original calculations. Validated 100% match rate (within float32 precision).

**Edge case note:** The breakout calculation handles near-zero channel width slightly differently:
- Original (`hierarchical_dataset.py`): `channel_width = std + 1e-10` (always continues)
- Precompute (`precompute_targets.py`): Returns neutral defaults if `channel_width < 1e-10`

This only affects samples where price is completely flat (std ≈ 0), which is rare in real market data. Validation found 0 edge cases in 1000 random samples.

---

## 10. Loss Calculation Vectorization (v5.9.6)

### Location: `train_hierarchical.py:4355-4493`

### Problem:

The loss calculation had **massive nested Python loops** creating tensors inside:

```python
# Original (v5.9.5 and earlier):
for sample_idx in range(batch_size):        # 128 iterations
    for win_idx, window in enumerate(WINDOWS):  # 14 iterations
        for tf in timeframes:                    # 11 iterations
            # Individual tensor creation:
            target_hit_upper_t = torch.tensor(hit_upper_target, device=device)  # SLOW!
            hit_loss = F.binary_cross_entropy(
                hit_prob_upper_pred[sample_idx:sample_idx+1], ...  # Per-sample call
            )
```

**Total operations per batch:**
- 128 samples × 14 windows × 11 TFs = ~20,000 loop iterations
- ~20,000 `torch.tensor()` calls
- ~20,000 `F.binary_cross_entropy()` calls

**Time:** 5700ms per batch (45% of total iteration time)

### Fix:

Vectorized to batched tensor operations:

```python
# Vectorized (v5.9.6):
# Phase 1: Pre-gather all targets into [batch, num_windows] tensors
validity_mask = torch.stack(validity_list, dim=1)  # [128, 14]
hit_upper_targets = torch.stack(hit_upper_list, dim=1)  # [128, 14]

# Phase 2: Batched weighted blend
valid_weights = window_weights * validity_mask.float()  # [128, 14]
normalized_weights = valid_weights / total_weight  # [128, 14]
blended_hit_upper = (normalized_weights * hit_upper_targets).sum(dim=1)  # [128]

# Phase 3: Single batched BCE call (instead of 128 individual calls)
bce_upper = F.binary_cross_entropy(hit_prob_upper_pred, blended_hit_upper, reduction='none')
```

**Result:**
- 20,000 operations → ~30 tensor operations per TF
- **5700ms → 700ms (8x speedup)**
- Backward pass also improved slightly (less computation graph)

### Impact on training:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Loss calculation | 5700ms | 700ms | 87% faster |
| Total batch time | ~12,500ms | ~7,200ms | 42% faster |
| Epoch time (2,375 batches) | ~8.2 hours | ~4.8 hours | **3.4 hours saved** |

---

## 11. Data Size Summary (v5.9.6)

```
Feature Cache Total: ~5.1 GB (down from ~6.5 GB after cleanup)
├── TF Sequences (11 files)          3.2 GB
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
├── Continuation Labels (11 .mmap/ dirs + 11 .pkl)   ~3.1 GB
│   ├── *.mmap/ directories         ~800 MB (used at runtime)
│   └── *.pkl files                 ~2.3 GB (source, kept for flexibility)
│
├── Transition Labels (10 .mmap/ dirs + 10 .pkl)     ~52 MB
│   ├── *.mmap/ directories         ~22 MB (used at runtime)
│   └── *.pkl files                 ~30 MB (source, kept for flexibility)
│
├── Precomputed Targets              ~17 MB
│   ├── precomputed_breakout_*.npz   382 KB
│   ├── precomputed_targets_*.npz    16.8 MB
│   └── precomputed_valid_indices_*.npy  small
│
└── Metadata
    ├── tf_meta_*.json               small
    └── cache_manifest_*.json        small

Training Samples: ~417,430 (after warmup)
Batch Size: 128
Batches per Epoch: ~2,375
```

**Note:** Non-channel features (1.2 GB) deleted in v5.9.6 - not used in native TF mode.

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

## 13. Recommendations (v5.9.6 Updated)

### ✅ Completed Optimizations:

1. ~~**Pre-compute breakout labels**~~ - ✅ **FIXED v5.9.4**
2. ~~**Pre-stack target tensors**~~ - ✅ **FIXED v5.9.4**
3. ~~**Batched collate tensor creation**~~ - ✅ **FIXED v5.9.5**
4. ~~**DDP-compatible ShuffleBufferSampler**~~ - ✅ **FIXED v5.9.4**
5. ~~**Vectorize loss calculation**~~ - ✅ **FIXED v5.9.6**
6. ~~**Memory-mapped labels**~~ - ✅ **FIXED v5.9.6**
7. ~~**Single-GPU sampler epoch**~~ - ✅ **FIXED v5.9.6**

### High Priority (Model Architecture)

1. **Backward pass optimization** - 5000ms (69% of batch time)
   - Requires model architecture changes to reduce computation graph
   - Options: Simplify loss components, reduce hidden size, gradient checkpointing
   - Expected speedup: 20-40%
   - Effort: High

### Medium Priority (Inference Optimization)

2. **Forward pass optimization** - 1240ms (17% of batch time)
   - Enable `torch.compile()` for JIT compilation
   - Profile specific layers to find bottlenecks
   - Consider reducing hidden_size from 128 to 96
   - Expected speedup: 10-20%
   - Effort: Medium

### Low Priority (Already Fast)

3. **Data loading** - 140ms (2% of batch time)
   - Enable DataLoader workers (test with mmap labels)
   - Increase from 0 → 2-4 workers
   - Expected speedup: 50% (140ms → 70ms)
   - Effort: Low
   - **Note:** Currently using num_workers=0 due to MPS device

4. ~~**Eliminate per-sample ascontiguousarray**~~ - **NOT AN ISSUE**
5. ~~**Use orphaned aligned_indices**~~ - **Negligible** (with preload to RAM)
6. ~~**Replace ThreadPoolExecutor**~~ - **Deprecated** (RollingBuffer not used)

---

## What Should Happen vs What Happens

### Ideal `__getitem__`:
```python
def __getitem__(self, idx):
    return self.precomputed_features[idx], self.precomputed_targets[idx]
    # O(1) array lookup, ~10μs
```

### Current `__getitem__` (v5.9.6 with all optimizations):
```python
def __getitem__(self, idx):
    for tf in 11_timeframes:
        searchsorted()           # O(log n) - negligible with RAM preload
        slice_array()            # O(1) - mmap'd arrays
        ascontiguousarray()      # no-op for 8/11 TFs (verified)

    # Fast path: array lookups from mmap'd precomputed targets
    targets = {k: float(arr[idx]) for k, arr in precomputed_arrays.items()}
    # O(1) per field, ~200-400μs total

    return features_dict, targets_dict
    # ~400-700μs per sample (data loading is no longer a bottleneck)
```

### Training loop timing (v5.9.6):
```python
for batch in dataloader:
    data_load()      # 140ms - Fast ✅
    forward()        # 1240ms - Medium
    loss_calc()      # 700ms - Optimized ✅ (was 5700ms)
    backward()       # 5000ms - BOTTLENECK ❌
    optimizer()      # 65ms - Fast ✅
# Total: ~7200ms per batch
```

**Key insight:** Data loading pipeline is now well-optimized. Further speedups require model architecture changes to reduce backward pass complexity.

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
