# Sample-Indexed Storage Implementation Plan

**Version:** v5.9.5
**Date:** 2025-12-25
**Status:** PLANNING

---

## Executive Summary

**Original Concept:** Store pre-computed sample tensors for O(1) lookup
**Problem Found:** Storage requirements are 354x-2293x current (1,122-7,260 GB vs 3.2 GB)
**Revised Approach:** Multi-phase optimization with pre-computed indices, buffers, full targets, and VIX

The revised approach achieves **~50-70% faster `__getitem__`** while requiring only **~225 MB additional storage** instead of terabytes.

---

## Why Original Option B is Infeasible

Each sample contains 11 timeframes × sequence_length × 1049 features × 4 bytes (float32):

| Preset | Sequence Lengths | Storage Required | vs Current 3.2 GB |
|--------|-----------------|------------------|-------------------|
| Low | 75 bars uniform | 1,122 GB | 354x |
| Medium | 200-600 bars | 4,279 GB | 1,351x |
| High | 300-1200 bars | 7,260 GB | 2,293x |

**Conclusion:** Full sample storage is not practical.

---

## Revised Solution: Multi-Phase Optimization

### Phase 1: Pre-computed Index Mapping
**Speedup:** ~5% | **Storage:** +37 MB

**Problem:** Every `__getitem__` call runs `np.searchsorted()` 11 times to find slice indices.

**Solution:** Pre-compute all slice indices once and store them.

**New File:** `sample_indices_{cache_key}.npz`
```python
{
    'tf_indices': {
        '5min': ndarray[417933, 2],   # [start, end] for each sample
        '15min': ndarray[417933, 2],
        # ... 11 timeframes
    },
    'valid_indices': ndarray[417933],
    'sequence_lengths': {...},
    'cache_key': str
}
```

---

### Phase 2: Contiguous Buffer Optimization
**Speedup:** ~30-40% | **Storage:** +0 MB (RAM only)

**Problem:** `np.ascontiguousarray()` copies data 11 times per sample (~200-500μs).

**Solution:** Pre-allocate reusable buffers, write directly to them.

```python
# At init:
self._batch_buffers = {
    tf: np.empty((batch_size, seq_len, features), dtype=np.float32)
    for tf in timeframes
}

# In __getitem__:
# Instead of: timeframe_data[tf] = np.ascontiguousarray(...)
# Do: self._batch_buffers[tf][idx] = self.tf_mmaps[tf][start:end, :]
```

---

### Phase 2b: Full Target Pre-computation
**Speedup:** ~10-15% | **Storage:** +20 MB

**Problem:** Even with v5.9.4 precomputed targets, base targets (`high`, `low`, `expected_return`, etc.) are still computed per-sample in `_getitem_precomputed_path()`.

**Current fast path still computes:**
```python
# These are STILL computed per-sample:
targets = self._calculate_targets_from_future(
    current_price=current_price,
    future_prices=future_prices,  # Fetched per-sample
    ...
)
```

**Solution:** Pre-compute ALL targets including base targets.

**Add to `precomputed_targets_{cache_key}.npz`:**
```python
# Currently pre-computed (v5.9.4):
- breakout_occurred, breakout_direction, breakout_bars_log, breakout_magnitude
- cont_{tf}_w{window}_{field} (1000+ keys)
- trans_{tf}_{field}

# NEW - also pre-compute:
- high, low
- hit_band, hit_target
- expected_return, overshoot
- price_change_pct, horizon_bars_log, adaptive_confidence
```

**Then `_getitem_precomputed_path()` becomes trivial:**
```python
def _getitem_precomputed_path(self, idx, data_idx_5min, timeframe_data):
    # ALL targets from pre-computed arrays - NO computation!
    targets = {}
    for key, arr in self._precomputed_targets.items():
        targets[key] = float(arr[idx])
    for key, arr in self._precomputed_breakout.items():
        targets[key] = float(arr[idx])

    # Only VIX and events still per-sample (until Phase 2c)
    vix_seq = self._get_vix(data_idx_5min)
    events = self._get_events(data_idx_5min)

    return timeframe_data, targets, vix_seq, events
```

---

### Phase 2c: VIX Pre-computation
**Speedup:** ~5% | **Storage:** +150 MB

**Problem:** VIX sequence lookup happens per-sample (~50μs).

**Current:**
```python
vix_seq = self._vix_loader.get_sequence(ts.date(), 90)  # 90-day lookback
```

**Solution:** Pre-compute VIX sequences for all samples.

**New File:** `precomputed_vix_{cache_key}.npz`
```python
{
    'vix_sequences': ndarray[417933, 90],  # 90-day sequence per sample
    'vix_file_hash': str,                   # For validation
    'generated_at': str
}
```

**Storage:** 417K samples × 90 days × 4 bytes = **~150 MB**

**Validation:** Check VIX source file hash. Regenerate if VIX data updated.

---

### Phase 3: Events Pre-computation (Optional/Future)
**Speedup:** ~2-3% | **Storage:** +50-100 MB

**Problem:** Event lookup happens per-sample (~20-50μs).

**Why separate phase:**
- `events.csv` is user-editable (users add FOMC dates, earnings, etc.)
- More complex validation needed (file hash + modification date)
- Marginal speedup for added complexity

**New File:** `precomputed_events_{cache_key}.npz`
```python
{
    'event_vectors': ndarray[417933, event_dim],
    'events_file_hash': str,
    'generated_at': str
}
```

**Validation:** Check events.csv hash. Regenerate if events updated.

---

## Implementation Plan

### Files to Create

1. **`src/ml/precompute_sample_indices.py`**
   - `compute_sample_indices()` - Compute slice indices for all samples
   - `save_sample_indices()` - Save to .npz
   - `load_sample_indices()` - Load from .npz
   - `validate_sample_indices()` - Check sequence lengths match

### Files to Modify

1. **`src/ml/precompute_targets.py`** (existing)
   - Add base target computation: `high`, `low`, `hit_band`, `hit_target`, `expected_return`, `overshoot`, `price_change_pct`, `horizon_bars_log`, `adaptive_confidence`
   - Add VIX sequence pre-computation
   - Add validation for VIX source file

2. **`src/ml/hierarchical_dataset.py`**
   - Add `_load_sample_indices()` method
   - Add `_validate_sample_indices()` method
   - Modify `_getitem_native_timeframe()` to use pre-computed indices
   - Add `_init_contiguous_buffers()` for Phase 2
   - Simplify `_getitem_precomputed_path()` to pure lookups (Phase 2b)
   - Add `_load_precomputed_vix()` method (Phase 2c)

3. **`train_hierarchical.py`**
   - Add interactive menu option for optimization level:
     ```
     ? Data loading optimization:
       Full (indices + buffers + all targets + VIX) ⭐ Recommended
       Standard (indices + buffers + targets)
       Minimal (indices only)
       None (runtime computation)
     ```

4. **`docs/TRAINING_BOTTLENECK_ANALYSIS.md`**
   - Update with v5.9.5 status

---

## Expected Performance

### Current (v5.9.4)
| Operation | Time/Sample |
|-----------|-------------|
| 11× searchsorted() | ~5μs |
| 11× ascontiguousarray() | ~200-500μs |
| Base target computation | ~100-200μs |
| Precomputed target lookup | ~50μs |
| VIX lookup | ~50μs |
| Events lookup | ~20μs |
| **Total** | **~425-825μs** |

### After Phase 1 (Index Mapping)
| Operation | Time/Sample | Change |
|-----------|-------------|--------|
| 11× index lookup | ~1μs | -4μs |
| 11× ascontiguousarray() | ~200-500μs | — |
| Base target computation | ~100-200μs | — |
| Precomputed target lookup | ~50μs | — |
| VIX lookup | ~50μs | — |
| Events lookup | ~20μs | — |
| **Total** | **~420-820μs** | ~1% faster |

### After Phase 2 (Contiguous Buffers)
| Operation | Time/Sample | Change |
|-----------|-------------|--------|
| 11× index lookup | ~1μs | — |
| 11× buffer write | ~50-100μs | **-150-400μs** |
| Base target computation | ~100-200μs | — |
| Precomputed target lookup | ~50μs | — |
| VIX lookup | ~50μs | — |
| Events lookup | ~20μs | — |
| **Total** | **~270-420μs** | ~35-50% faster |

### After Phase 2b (Full Target Pre-compute)
| Operation | Time/Sample | Change |
|-----------|-------------|--------|
| 11× index lookup | ~1μs | — |
| 11× buffer write | ~50-100μs | — |
| ALL target lookup | ~10μs | **-140-240μs** |
| VIX lookup | ~50μs | — |
| Events lookup | ~20μs | — |
| **Total** | **~130-180μs** | ~55-65% faster |

### After Phase 2c (VIX Pre-compute)
| Operation | Time/Sample | Change |
|-----------|-------------|--------|
| 11× index lookup | ~1μs | — |
| 11× buffer write | ~50-100μs | — |
| ALL target lookup | ~10μs | — |
| VIX lookup | ~1μs | **-49μs** |
| Events lookup | ~20μs | — |
| **Total** | **~80-130μs** | ~60-70% faster |

### After Phase 3 (Events Pre-compute) - Optional
| Operation | Time/Sample | Change |
|-----------|-------------|--------|
| 11× index lookup | ~1μs | — |
| 11× buffer write | ~50-100μs | — |
| ALL target lookup | ~10μs | — |
| VIX lookup | ~1μs | — |
| Events lookup | ~1μs | **-19μs** |
| **Total** | **~60-110μs** | ~65-75% faster |

---

## Interactive Menu Integration

```python
# After preload_tf_to_ram selection:
if args.preload_tf_to_ram:
    optimization_level = inquirer.select(
        message="Data loading optimization:",
        choices=[
            Choice('full', "Full (indices + buffers + all targets + VIX) ⭐ Recommended"),
            Choice('standard', "Standard (indices + buffers + current targets)"),
            Choice('minimal', "Minimal (indices only)"),
            Choice('none', "None (runtime computation - slowest)"),
        ],
        default='full'
    ).execute()

    args.optimization_level = optimization_level
```

---

## Auto-Generation Pattern

Same pattern as precomputed targets (v5.9.4):

```python
def _load_sample_indices(self, cache_dir, cache_key):
    indices_path = cache_dir / f"sample_indices_{cache_key}.npz"

    if indices_path.exists():
        data = dict(np.load(indices_path))
        if self._validate_sample_indices(data):
            self._sample_indices = data
            self._use_sample_indices = True
            print("     ✓ Loaded pre-computed sample indices")
            return

    # Auto-generate if missing
    print("     🔄 Auto-generating sample indices...")
    indices = compute_sample_indices(cache_dir, cache_key, ...)
    save_sample_indices(cache_dir, cache_key, indices)

    self._sample_indices = indices
    self._use_sample_indices = True
    print("     ✓ Generated and loaded sample indices")
```

---

## Cache Validation

**Invalidation triggers by component:**

| Component | Invalidation Trigger | Validation Method |
|-----------|---------------------|-------------------|
| Sample indices | Sequence lengths changed | Compare `sequence_lengths` dict |
| Base targets | Prediction horizon changed | Compare `prediction_horizon` |
| VIX sequences | VIX data file updated | Check file hash/mtime |
| Events (Phase 3) | events.csv updated | Check file hash/mtime |

**Validation in manifest:**
```json
{
  "sample_indices": {
    "path": "sample_indices_v5.9.0_....npz",
    "sample_count": 417933,
    "sequence_lengths": {"5min": 75, "15min": 75, ...},
    "generated_at": "2025-12-25T12:00:00Z"
  },
  "precomputed_targets": {
    "path": "precomputed_targets_v5.9.0_....npz",
    "includes_base_targets": true,
    "prediction_horizon": 24
  },
  "precomputed_vix": {
    "path": "precomputed_vix_v5.9.0_....npz",
    "vix_file_hash": "abc123...",
    "sequence_length": 90
  }
}
```

---

## Migration Path

### Automatic (Recommended)
1. Update to v5.9.5
2. Run training normally
3. All components auto-generate on first run:
   - Sample indices: ~30 seconds
   - Base targets: ~2-3 minutes (added to existing precompute)
   - VIX sequences: ~1 minute
4. Subsequent runs use cached data

### Manual
```bash
# Generate all optimizations
python -m src.ml.precompute_targets --cache-dir data/feature_cache --full

# Or generate specific components
python -m src.ml.precompute_sample_indices --cache-dir data/feature_cache
```

### Force Regeneration
- Use "Force regenerate" in interactive menu
- Or delete specific files:
  - `sample_indices_*.npz` - Regenerate indices
  - `precomputed_targets_*.npz` - Regenerate targets
  - `precomputed_vix_*.npz` - Regenerate VIX

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Backward compatibility | Low | Falls back to runtime computation |
| Storage overhead | Low | ~225 MB additional (7% increase) |
| Sequence length changes | Medium | Auto-detect and regenerate |
| VIX file updates | Low | Validate hash, regenerate if changed |
| Code complexity | Medium | Follows existing patterns |

---

## Disk Space Summary

| Component | Size | Notes |
|-----------|------|-------|
| Sample indices | 37 MB | 11 TFs × 2 int32s × 418K |
| Precomputed targets (expanded) | 40 MB | +20 MB for base targets |
| Precomputed VIX | 150 MB | 418K × 90 days × 4 bytes |
| Row-based TF sequences | 3.17 GB | Unchanged |
| **Total additional** | **~225 MB** | 7% increase |

**Phase 3 (optional):**
| Component | Size | Notes |
|-----------|------|-------|
| Precomputed events | 50-100 MB | Depends on event vector size |

---

## Summary: What Gets Pre-computed

| Component | v5.9.4 | v5.9.5 (This Plan) |
|-----------|--------|-------------------|
| Breakout labels | ✅ Pre-computed | ✅ Pre-computed |
| Continuation/Transition | ✅ Pre-computed | ✅ Pre-computed |
| Base targets (high, low, etc.) | ❌ Per-sample | ✅ Pre-computed |
| Sample slice indices | ❌ Per-sample | ✅ Pre-computed |
| Contiguous buffers | ❌ Copy per-sample | ✅ Pre-allocated |
| VIX sequences | ❌ Per-sample | ✅ Pre-computed |
| Events | ❌ Per-sample | ❌ Per-sample (Phase 3) |

---

## Conclusion

This multi-phase approach achieves **~60-70% faster `__getitem__`** with only **~225 MB additional storage** (7% increase).

**Phase Summary:**
| Phase | Component | Speedup | Storage |
|-------|-----------|---------|---------|
| 1 | Index mapping | ~5% | +37 MB |
| 2 | Contiguous buffers | ~30-40% | +0 MB |
| 2b | Full target pre-compute | ~10-15% | +20 MB |
| 2c | VIX pre-compute | ~5% | +150 MB |
| 3 | Events pre-compute (optional) | ~2-3% | +50-100 MB |
| **Total (1-2c)** | | **~60-70%** | **~225 MB** |

This approach:
- Follows the same pattern as precomputed targets (v5.9.4)
- Is fully backward compatible
- Auto-generates on first run
- Validates and regenerates when configuration changes
- Keeps Events separate due to user-editable nature
