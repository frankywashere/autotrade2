# Sample-Indexed Storage Implementation Plan

**Version:** v5.9.5
**Date:** 2025-12-25
**Status:** PLANNING

---

## Executive Summary

**Original Concept:** Store pre-computed sample tensors for O(1) lookup
**Problem Found:** Storage requirements are 354x-2293x current (1,122-7,260 GB vs 3.2 GB)
**Revised Approach:** Pre-computed Index Mapping + Contiguous Buffer Optimization

The revised approach achieves ~30-50% faster `__getitem__` while requiring only **~37 MB additional storage** instead of terabytes.

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

## Revised Solution: Two-Phase Optimization

### Phase 1: Pre-computed Index Mapping (~5% speedup)

**Problem:** Currently, every `__getitem__` call runs `np.searchsorted()` 11 times to find the right slice indices.

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

**Storage:** 11 TFs × 2 int32s × 418K samples = **37 MB**

### Phase 2: Contiguous Buffer Optimization (~30-50% speedup)

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

## Implementation Plan

### Files to Create

1. **`src/ml/precompute_sample_indices.py`**
   - `compute_sample_indices()` - Main computation
   - `save_sample_indices()` - Save to .npz
   - `load_sample_indices()` - Load from .npz
   - `validate_sample_indices()` - Validation

### Files to Modify

1. **`src/ml/hierarchical_dataset.py`**
   - Add `_load_sample_indices()` method (like `_load_precomputed_targets`)
   - Add `_validate_sample_indices()` method
   - Modify `_getitem_native_timeframe()` to use pre-computed indices
   - Add `_init_contiguous_buffers()` for Phase 2
   - Add `__getitems__()` batch method for Phase 2

2. **`train_hierarchical.py`**
   - Add interactive menu option for sample index mode:
     ```
     ? Sample index optimization:
       Auto-generate if missing (Recommended) ⭐
       Force regenerate (if sequence lengths changed)
       Skip (use runtime index computation)
     ```

3. **`docs/TRAINING_BOTTLENECK_ANALYSIS.md`**
   - Update with v5.9.5 status

---

## Expected Performance

### Current (v5.9.4)
| Operation | Time/Sample |
|-----------|-------------|
| 11× searchsorted() | ~5μs |
| 11× ascontiguousarray() | ~200-500μs |
| Target lookup | ~200-400μs |
| **Total** | **~400-700μs** |

### After Phase 1 (Index Mapping)
| Operation | Time/Sample | Improvement |
|-----------|-------------|-------------|
| 11× index lookup | ~1μs | 5x faster |
| 11× ascontiguousarray() | ~200-500μs | Unchanged |
| Target lookup | ~200-400μs | Unchanged |
| **Total** | **~400-700μs** | Minimal |

### After Phase 2 (Contiguous Buffers)
| Operation | Time/Sample | Improvement |
|-----------|-------------|-------------|
| 11× index lookup | ~1μs | 5x faster |
| 11× buffer write | ~50-100μs | **4-10x faster** |
| Target lookup | ~200-400μs | Unchanged |
| **Total** | **~250-500μs** | **~30-50% faster** |

---

## Interactive Menu Integration

```python
# After preload_tf_to_ram selection:
if args.preload_tf_to_ram:
    index_mode = inquirer.select(
        message="Sample index optimization:",
        choices=[
            Choice('auto', "Auto-generate if missing (Recommended) ⭐"),
            Choice('force', "Force regenerate (if sequence lengths changed)"),
            Choice('skip', "Skip (use runtime index computation)"),
        ],
        default='auto'
    ).execute()
```

---

## Auto-Generation Pattern

Same pattern as precomputed targets (v5.9.4):

```python
def _load_sample_indices(self, cache_dir, cache_key):
    indices_path = cache_dir / f"sample_indices_{cache_key}.npz"

    if indices_path.exists():
        # Load and validate
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

**Invalidation triggers:**
1. Sequence lengths changed (user switched Low/Medium/High preset)
2. Cache key mismatch
3. Sample count mismatch
4. Missing timeframes

**Validation in manifest:**
```json
{
  "sample_indices": {
    "path": "sample_indices_v5.9.0_....npz",
    "sample_count": 417933,
    "sequence_lengths": {"5min": 75, "15min": 75, ...},
    "generated_at": "2025-12-25T12:00:00Z"
  }
}
```

---

## Migration Path

### Automatic (Recommended)
1. Update to v5.9.5
2. Run training normally
3. Indices auto-generate on first run (~30 seconds)
4. Subsequent runs use cached indices

### Manual
```bash
python -m src.ml.precompute_sample_indices --cache-dir data/feature_cache
```

### Force Regeneration
- Use "Force regenerate" in interactive menu
- Or delete `sample_indices_*.npz` files

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Backward compatibility | Low | Falls back to runtime computation |
| Storage overhead | Low | Only 37 MB additional |
| Sequence length changes | Medium | Auto-detect and regenerate |
| Code complexity | Low | Follows existing patterns |

---

## Disk Space Summary

| Component | Size | Notes |
|-----------|------|-------|
| Sample indices (new) | 37 MB | 11 TFs × 2 int32s × 418K |
| Precomputed targets | 20 MB | Already exists (v5.9.4) |
| Row-based TF sequences | 3.17 GB | Unchanged |
| **Total additional** | **~37 MB** | 1.2% increase |

---

## Conclusion

The original "full sample storage" concept is infeasible (1,000+ GB), but the **Pre-computed Index Mapping + Contiguous Buffers** approach achieves meaningful speedup (~30-50%) with minimal storage overhead (37 MB).

This approach:
- Follows the same pattern as precomputed targets (v5.9.4)
- Is fully backward compatible
- Auto-generates on first run
- Validates and regenerates when configuration changes
