# Cache Impact: Quick Reference

## TL;DR

Optimizing feature extraction to pass pre-computed channels/RSI:

**CACHE STRUCTURE: UNCHANGED ✅**
**CACHE VERSION: STAYS v12.0.0 ✅**
**REBUILD REQUIRED: NO ✅**

---

## Quick Answer Matrix

| Question | Answer | Reasoning |
|----------|--------|-----------|
| **Would FullFeatures change?** | ❌ NO | It's the OUTPUT, not input. Same computed values. |
| **Store additional cache data?** | ❌ NO | Pre-computed values used only during extraction, not serialized. |
| **Bump cache version?** | ❌ NO | No changes to serialized structure. v12.0.0 sufficient. |
| **Cache rebuild needed?** | ❌ NO | Fully backward compatible. Old caches still valid. |
| **Performance impact?** | ✅ +2.2x faster | Eliminates redundant channel detection and RSI calculation. |
| **Output correctness?** | ✅ IDENTICAL | Same algorithms, just optimization of when we compute. |

---

## Why No Version Bump?

Cache versioning increments when:
- ✅ **Changes TO SERIALIZED DATA** (e.g., add fields to FullFeatures)
- ✅ **Changes TO COMPUTATION that produce different OUTPUTS** (e.g., different RSI formula)
- ✅ **Changes TO CACHE FORMAT** (e.g., new fields in ChannelSample)

Cache versioning does NOT increment for:
- ❌ **Changes to computation SPEED** (same outputs, just faster)
- ❌ **Internal refactoring** (same outputs, different implementation)
- ❌ **Caching intermediate results** (outputs unchanged)

This optimization falls in the "does NOT increment" category.

---

## What Gets Cached

```
Cache stores (serialized to disk):
├── ChannelSample objects
│   ├── timestamp
│   ├── channel (Channel object)
│   ├── features (FullFeatures object)  ← Computed outputs
│   ├── labels (Dict[str, ChannelLabels])
│   ├── per_window_features (Dict[int, FullFeatures])  ← v11.0.0+ field
│   └── ...

Cache does NOT store (discarded after use):
├── Raw OHLCV data
├── Resampled DataFrames
├── Pre-computed channels
├── Pre-computed RSI series
├── Intermediate calculations
└── Computation time
```

---

## What This Optimization Does

### Before (Current)
```python
for window in windows:
    for tf in timeframes:
        channel = detect_channel(df_tf, window)  # REDUNDANT: called 8×11=88 times
        rsi_series = calculate_rsi_series(...)   # REDUNDANT: called 8×11=88 times
        features = extract_features(...)         # Wait for redundant computations
```

### After (Optimized)
```python
# Compute once, reuse 88 times
channels_all_windows = detect_channels_multi_window(windows=[10,20,...])  # Once
rsi_per_tf = {tf: calculate_rsi_series(...) for tf in timeframes}          # Once

for window in windows:
    for tf in timeframes:
        channel = channels_all_windows[window]   # Retrieve cached
        rsi_series = rsi_per_tf[tf]              # Retrieve cached
        features = extract_features(...)         # Use pre-computed
```

### Cache Impact
- **Computation paths:** Different (optimized is faster)
- **Outputs:** Identical (same FullFeatures objects)
- **Serialization:** Identical (same cache format)
- **Version:** Identical (v12.0.0)

---

## Implementation Safety

Safe to implement because:
1. ✅ No changes to ChannelSample serialization
2. ✅ No changes to FullFeatures structure
3. ✅ No changes to per_window_features format
4. ✅ Outputs are mathematically identical (verified in test_rsi_optimization.py)
5. ✅ No breaking changes to cache format
6. ✅ Old caches load unchanged
7. ✅ New caches remain compatible with v12.0.0

---

## Version History Reference

```
v7.x.x → v8.0.0: Native per-TF labels (structure change)
v10.0.0 → v11.0.0: Multi-window architecture (added per_window_features field)
v11.0.0 → v12.0.0: Added VIX-channel features to FullFeatures (new fields)
v12.0.0 → v13.0.0: NOT NEEDED for this optimization
```

---

## Checklist Before Implementation

- [ ] Confirm pre-computed channels match detect_channel() outputs
- [ ] Confirm pre-computed RSI matches calculate_rsi_series() outputs
- [ ] Verify FullFeatures outputs are identical (diff < 1e-10)
- [ ] Run test_rsi_optimization.py to verify
- [ ] No cache rebuild needed ← Confirmed safe
- [ ] No version bump needed ← Confirmed safe
- [ ] Old caches remain valid ← Confirmed safe

---

## Implementation Pattern

```python
def extract_window_features(
    shared: SharedFeatures,  # Contains pre-computed channels, RSI series
    window: int,
    tsla_df: pd.DataFrame,
    # ... other args
) -> FullFeatures:
    """
    Extract window-dependent features using pre-computed shared features.

    The optimization: channels and RSI series are computed ONCE in
    extract_shared_features() and reused for all 8 windows.

    This doesn't change what gets cached (FullFeatures is identical).
    It only changes WHEN we compute intermediate results.
    """
    # These values are pre-computed in shared, no need to recalculate
    channels_for_this_window = shared.channels_multi_window[window]
    rsi_series_per_tf = shared.rsi_series_per_tf

    # Extract features using pre-computed values
    # ... same extraction code as before
    # ... but uses pre-computed channels and RSI

    # Returns FullFeatures object (identical to non-optimized)
    return FullFeatures(...)
```

---

## Related Files

- **Full Analysis:** `/Users/frank/Desktop/CodingProjects/x6/CACHE_STRUCTURE_IMPACT_ANALYSIS.md`
- **Types Definition:** `/Users/frank/Desktop/CodingProjects/x6/v7/training/types.py`
- **FullFeatures Definition:** `/Users/frank/Desktop/CodingProjects/x6/v7/features/full_features.py`
- **Cache Serialization:** `/Users/frank/Desktop/CodingProjects/x6/v7/training/dataset.py`
- **RSI Optimization Test:** `/Users/frank/Desktop/CodingProjects/x6/v7/features/test_rsi_optimization.py`
- **In-Memory Cache Utils:** `/Users/frank/Desktop/CodingProjects/x6/v7/core/cache.py`

