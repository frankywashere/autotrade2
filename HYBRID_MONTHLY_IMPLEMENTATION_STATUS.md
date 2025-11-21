# Hybrid Monthly/3Month Processing - Implementation Status

**Date:** January 20, 2025
**Status:** IN PROGRESS (60% complete)

---

## What's Been Done ✅

### 1. Created _extract_monthly_3month_features() Method
**File:** src/ml/features.py (lines 931-1047)
- Extracts monthly/3month on FULL dataset
- Uses calculate_multi_window_rolling (same as other TFs)
- Broadcasts to 1-min timestamps
- Returns DataFrame with ~1,302 columns (2 symbols × 2 TFs × 21 windows × 31 metrics)
- Memory: ~565 KB for 10 years

---

## What Remains 🔄

### 2. Modify Chunked Extraction Flow
**File:** src/ml/features.py (around line ~1070-1080)

**BEFORE chunking loop, ADD:**
```python
# Pre-process monthly/3month on full dataset (they're tiny!)
monthly_3month_features = None
if use_chunking:
    monthly_3month_features = self._extract_monthly_3month_features(df)
```

**DURING chunking (line ~732 in parallel tasks):**
```python
# Filter out monthly/3month from parallel extraction
timeframes_for_chunking = {
    k: v for k, v in timeframes.items()
    if k not in ['monthly', '3month']
}

# Build tasks with ONLY 20 TFs (not 22)
tasks = []
for symbol in ['tsla', 'spy']:
    for tf_name, tf_rule in timeframes_for_chunking.items():  # Changed here
        # ... rest unchanged
```

**AFTER chunking (line ~420-460):**
```python
# When concatenating base features:
if channel_df is None and hasattr(self, '_mmap_meta_path'):
    # Mmap mode - channel features in shards
    base_features_df = pd.concat([
        price_df,
        rsi_df,
        correlation_df,
        cycle_df,
        volume_df,
        time_df,
        monthly_3month_features  # ADD HERE - goes to non_channel_array
    ], axis=1)
```

---

### 3. Update Feature Count Comments
**File:** src/ml/features.py (line ~75)

```python
# OLD:
# 21 windows × 11 timeframes × 31 metrics × 2 symbols = 14,322 channel features

# NEW:
# Mmap shards: 21 windows × 9 timeframes (5min-weekly) × 31 metrics × 2 = ~11,718
# Non-channel: 165 base + 1,302 monthly/3month = 1,467
# Total: 11,718 + 1,467 = 13,185 features (when chunked)
#        OR 14,487 features (when not chunked - all TFs together)
```

---

### 4. Update FEATURE_VERSION
**File:** src/ml/features.py (line 28)

```python
# OLD:
FEATURE_VERSION = "v3.17_complete_cycles_31metrics"

# NEW:
FEATURE_VERSION = "v3.18_hybrid_monthly"
```

---

### 5. Handle Non-Chunked Mode
**File:** src/ml/features.py (line ~406-410)

When NOT chunking, monthly/3month are processed normally (already works).

**ADD check:**
```python
if use_chunking:
    # Process monthly/3month separately
    channel_df = self._extract_channel_features_chunked(
        df, ..., skip_long_tfs=True
    )
else:
    # Process all TFs together (current behavior)
    channel_df = self._extract_channel_features(
        df, ..., all_tfs=True
    )
```

---

## Estimated Remaining Work

**Lines to add:** ~80-100
**Complexity:** Medium (mostly wiring existing functions)
**Time:** 1-2 hours
**Risk:** Low (monthly/3month separate, doesn't affect other TFs)

---

## Testing Plan

1. Run with chunked=True
2. Verify monthly/3month processed on full dataset (108 bars)
3. Verify shards have only 20 TFs
4. Verify total features = 13,185
5. Train and verify no dimension mismatch

---

## Benefits

✅ Monthly/3month have proper data (108 bars vs 18)
✅ No insufficient_data flags for monthly/3month
✅ Memory savings in shards (20 TFs vs 22)
✅ ~500 KB extra in non_channel_array (negligible)

---

## Next Session TODO

1. Wire up monthly_3month_features call before chunking
2. Filter timeframes for parallel extraction
3. Merge monthly_3month into concat
4. Update feature count tracking
5. Bump FEATURE_VERSION
6. Test end-to-end
