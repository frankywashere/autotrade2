# Bug Fix Plan: NaN Training Crash & Feature Quality Issues

## Summary of Issues Found

### 1. CRITICAL: NaN Training Crash (Split Calculation Bug)
**Root Cause:** Split calculation doesn't account for warmup offset in tf_mmaps

- `features_df`: 1,434,833 rows (post-warmup 2017-2025)
- `tf_mmaps['5min']`: 418,635 rows (ALL data 2015-2025)
- Training includes 63,881 pre-warmup rows (indices 200-64,081)
- Rows 200-2,389 have NaN in monthly/3month breakdown features
- Model receives NaN → crash at batch 0

**Location:** `src/ml/hierarchical_dataset.py` lines 2085-2097

**Fix:** Apply warmup offset when calculating split indices:
```python
# Current (WRONG):
conversion_factor = total_len / native_tf_len
train_end_idx_adj = int(train_end_idx / conversion_factor)

# Should be:
warmup_idx_5min = <calculate from timestamps>
train_end_idx_adj = warmup_idx_5min + int((native_tf_len - warmup_idx_5min) * 0.85)
```

---

### 2. CRITICAL: RSI Features Are Constants (66 columns)
**Root Cause:** RSI calculation takes only last value, broadcasts to all rows

**Evidence:**
```
tsla_rsi_5min: 1 unique value (46.67 for ALL 1.6M rows)
tsla_rsi_15min: 1 unique value (47.95 for ALL 1.6M rows)
spy_rsi_5min: 1 unique value (56.71 for ALL 1.6M rows)
...all 66 RSI columns (22 base + 44 oversold/overbought) are constants
```

**Location:** `src/ml/features.py` lines 3405-3411

**Current Code:**
```python
rsi_data = self.rsi_calc.get_rsi_data(resampled)  # Only gets LAST value
rsi_value = rsi_data.value
rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, rsi_value)  # Broadcasts!
```

**Fix:** Use full RSI series and map back to original timestamps:
```python
rsi_series = self.rsi_calc.calculate_rsi(resampled)
# Reindex to original df timestamps using ffill
rsi_aligned = rsi_series.reindex(df.index, method='ffill')
rsi_features[f'{prefix}_{tf_name}'] = rsi_aligned.fillna(50.0).values
```

---

### 3. MEDIUM: Cross-TF ffill NaN at Dataset Start
**What:** First 108-2,390 rows have NaN in weekly/monthly/3month breakdown features

**Why:** `reindex(method='ffill')` has no previous value to fill from at start

**Evidence:**
```
weekly breakdown: 108 NaN rows at start
monthly breakdown: 2,390 NaN rows at start
3month breakdown: 2,390 NaN rows at start
```

**Location:** `src/ml/features.py` line 1521

**Fix:** Use `bfill().ffill()` to fill initial NaN:
```python
breakdown_aligned = other_breakdown.reindex(base_df.index).bfill().ffill()
```

---

### 4. LOW: Volume Surge Always Zero
**Root Cause:** `tsla_volume` column not in non-channel features, only derived `tsla_volume_ratio`

**Evidence:**
```
tsla_volume_surge: 418,635 rows ALL ZERO
```

**Location:** `src/ml/features.py` line 4103
```python
if 'tsla_volume' in resampled_df.columns:  # This fails!
    ...
else:
    breakdown_features['tsla_volume_surge'] = np.zeros(num_rows)
```

**Fix:** Either:
1. Include raw `tsla_volume` in non-channel features, OR
2. Use `tsla_volume_ratio` instead in breakdown calculation

---

## Audit Results Summary

### Features That ARE Working (varying values):
- Price features (spy_close, tsla_close, returns, log_returns)
- Volatility features (volatility_10, volatility_50)
- Correlation features (correlation_10/50/200, divergence)
- Cycle features (distance_from_52w_high/low, mega_channel_position)
- VIX features (vix_level, percentile, change, momentum, ma_ratio, trend)
- Time features (hour_of_day, day_of_week, day_of_month, month_of_year)
- Channel features (8,316 columns - all varying)
- Binary flags (is_monday, vix_above_20, etc. - correctly 0/1)

### Features That Are BROKEN (constant values):
- All 66 RSI columns (22 base RSI + 44 oversold/overbought flags)
- tsla_volume_surge (all zeros)

### Labels That ARE Working:
- Continuation labels (duration_bars, max_gain_pct, confidence - all varying)
- Transition labels (transition_type, switch_to_tf, direction - all varying)

---

## Fix Priority & Implementation Status

| Bug | Priority | Status | Location |
|-----|----------|--------|----------|
| Split calculation | CRITICAL | ✅ FIXED | `hierarchical_dataset.py:2082-2103` |
| RSI constant | CRITICAL | ✅ FIXED | `features.py:3404-3426` |
| Cross-TF ffill NaN | MEDIUM | ✅ FIXED | `features.py:1516-1526` |
| Volume surge zero | LOW | ✅ FIXED | `features.py:4114-4133` |

---

## Implementation Notes

### 1. Split Calculation Fix
- Changed from ratio-based conversion to timestamp-based binary search
- Now filters `all_valid` to exclude pre-warmup indices
- Added logging: `📍 Warmup boundary: 5min index X (date)`

### 2. RSI Fix
- Changed from `get_rsi_data()` (last value only) to `calculate_rsi()` (full series)
- Uses `reindex().bfill().ffill()` to map back to original timestamps
- Oversold/overbought now calculated per-bar (not broadcast)

### 3. Cross-TF ffill Fix
- Changed from `reindex(method='ffill')` to `reindex().bfill().ffill()`
- bfill() fills initial NaN with first available value

### 4. Volume Surge Fix
- Added primary path using `tsla_volume_ratio` (which exists)
- Kept fallback to `tsla_volume` if available
- Calculates relative change in volume_ratio

---

## Verification Steps

After fixes, regenerate cache and verify:
1. ✅ No NaN at warmup boundary in tf_sequence files
2. ✅ RSI columns have varying values (>1000 unique per column)
3. ✅ volume_surge has varying values
4. ✅ Training doesn't crash at batch 0
5. ✅ First batch has no NaN in inputs

**To test:**
```bash
# Delete old cache and regenerate
rm -rf data/feature_cache/tf_sequence_*.npy
rm -rf data/feature_cache/tf_timestamps_*.npy
rm -rf data/feature_cache/tf_meta_*.json
rm -rf data/feature_cache/non_channel_features_*.pkl

# Run training (will regenerate)
python train_hierarchical.py --interactive
```

---

## Phase 2: Partial Bar Implementation (v5.4)

### Architecture Change: Rolling Channels with Partial Bars

**Current (v5.3.3):** Complete bars only - ffill to finer TF
```
Monday 9:30am → Uses LAST WEEK's weekly channel (no today's data)
```

**New (v5.4):** Partial bars included - real-time channel evolution
```
Monday 9:30am → Uses [last 49 weeks] + [partial week: Monday 9:30am]
Monday 4:00pm → Uses [last 49 weeks] + [partial week: Mon 9:30-4pm]
```

### Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Partial bar computation | ✅ DONE | `src/ml/partial_bars.py` |
| Vectorized channel calc | ✅ DONE | `src/ml/partial_channel_calc_vectorized.py` |
| Integrate into features.py | 🔄 TODO | - |
| RSI with partial bars | 🔄 TODO | - |
| Breakdown at 5min | 🔄 TODO | - |
| Labels at 5min resolution | 🔄 TODO | - |
| Warmup update (10 months) | 🔄 TODO | - |
| Dataset/label lookup | 🔄 TODO | - |

### Performance Benchmarks

| Approach | Time (full dataset) | Notes |
|----------|---------------------|-------|
| Current (ffill) | ~30-60 min | Resample → calc → ffill |
| Partial (loop) | ~94 min | Too slow |
| Partial (vectorized) | ~14 min | ✅ Faster than current! |

### Verified: Partial Bars Evolve During Day

```
Feb 20, 2024 partial bar evolution:
  09:00: position=0.690, r²=0.626, slope=-0.3674
  10:45: position=0.676, r²=0.628, slope=-0.3713
  12:30: position=0.673, r²=0.628, slope=-0.3720
  14:15: position=0.627, r²=0.633, slope=-0.3848
  15:55: position=0.622, r²=0.634, slope=-0.3862

Position change: 0.690 → 0.622 ✅
```

### Remaining Work

1. **Integrate into features.py**: Replace `_extract_channel_features` with vectorized partial bar version
2. **RSI with partial bars**: Calculate RSI including today's partial data
3. **Breakdown at 5min**: Calculate breakdown features at 5min resolution (no ffill needed)
4. **Labels at 5min**: Regenerate continuation/transition labels at 5min resolution
5. **Warmup**: Increase to ~10 months (need 5148 5min bars for 3month rolling window)
6. **Dataset**: Update label lookup to use 5min labels directly
