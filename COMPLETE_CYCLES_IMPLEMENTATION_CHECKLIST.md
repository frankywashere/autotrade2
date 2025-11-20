# Complete Cycles Implementation Checklist

**Status:** In Progress
**Date:** January 19, 2025

---

## Summary

Switching from ping_pongs (alternating transitions) to complete_cycles (full round-trips) as the primary channel quality metric, while keeping both for model to learn from.

---

## Changes Completed ✅

### File: src/linear_regression.py

1. ✅ Added `_detect_complete_cycles()` method (lines 282-341)
2. ✅ Added `_detect_complete_cycles_vectorized()` method (lines 606-654)
3. ✅ Updated ChannelData dataclass with 4 new fields (lines 45-48)
4. ✅ Updated `calculate_multi_window_rolling()` to call complete_cycles (lines 771-773)
5. ✅ Updated ChannelData construction with complete_cycles values (lines 807-810)
6. ✅ Updated quality_score formula to use complete_cycles (line 783-785)
7. ✅ Added `find_best_channel_any_quality()` method (lines 248-292)

### File: src/ml/features.py

1. ✅ Updated metrics list in `_build_feature_names()` to include 4 complete_cycle metrics (line 81)
2. ✅ Added complete_cycle array allocations (2 locations in GPU extraction paths)
3. ✅ Added complete_cycle array allocations (2 locations in CPU extraction paths)
4. ✅ Added complete_cycle value assignments in chunked extraction (lines 867-870)

---

## Changes Remaining 🔄

### File: src/ml/features.py

**Location 1 (~line 1194-1199): GPU extraction - add complete_cycles calculation**
```python
# After multi_pp calculation:
multi_pp = channel_calc._detect_ping_pongs_multi_threshold(...)

# ADD:
multi_cycles = channel_calc._detect_complete_cycles_multi_threshold(
    window_prices,
    channel.upper_line,
    channel.lower_line,
    thresholds=[0.005, 0.01, 0.02, 0.03]
)
```

**Then store values (~line 1226-1229):**
```python
results[f'{prefix}_ping_pongs'][idx] = multi_pp[0.02]
results[f'{prefix}_ping_pongs_0_5pct'][idx] = multi_pp[0.005]
results[f'{prefix}_ping_pongs_1_0pct'][idx] = multi_pp[0.01]
results[f'{prefix}_ping_pongs_3_0pct'][idx] = multi_pp[0.03]
# ADD:
results[f'{prefix}_complete_cycles'][idx] = multi_cycles[0.02]
results[f'{prefix}_complete_cycles_0_5pct'][idx] = multi_cycles[0.005]
results[f'{prefix}_complete_cycles_1_0pct'][idx] = multi_cycles[0.01]
results[f'{prefix}_complete_cycles_3_0pct'][idx] = multi_cycles[0.03]
```

**Location 2 (~line 1418-1423): CPU extraction - add complete_cycles calculation**
```python
# After multi_pp calculation:
multi_pp = self.channel_calc._detect_ping_pongs_multi_threshold(...)

# ADD:
multi_cycles = self.channel_calc._detect_complete_cycles_multi_threshold(
    window_prices,
    channel.upper_line,
    channel.lower_line,
    thresholds=[0.005, 0.01, 0.02, 0.03]
)
```

**Then store values (~line 1450-1453):**
```python
results['ping_pongs'][mask] = channel.ping_pongs
results['ping_pongs_0_5pct'][mask] = multi_pp[0.005]
results['ping_pongs_1_0pct'][mask] = multi_pp[0.01]
results['ping_pongs_3_0pct'][mask] = multi_pp[0.03]
# ADD:
results['complete_cycles'][mask] = channel.complete_cycles
results['complete_cycles_0_5pct'][mask] = multi_cycles[0.005]
results['complete_cycles_1_0pct'][mask] = multi_cycles[0.01]
results['complete_cycles_3_0pct'][mask] = multi_cycles[0.03]
```

---

### File: src/ml/features.py (Continuation Labels)

**Location (~line 2401-2410): Switch to find_best_channel_any_quality**
```python
# OLD:
channel_1h = self.channel_calc.find_optimal_channel_window(
    one_h_ohlc, timeframe='1h',
    max_lookback=min(60, max(5, len(one_h_ohlc)-2)),
    min_ping_pongs=2  # ← Filters out "bad" channels
)

# NEW:
channel_1h = self.channel_calc.find_best_channel_any_quality(
    one_h_ohlc, timeframe='1h',
    max_lookback=min(60, max(5, len(one_h_ohlc)-2))
    # ← No filtering! Returns best fit regardless of quality
)

# Same for channel_4h
```

**Location (~line 2413): Remove None check (don't skip timestamps)**
```python
# OLD:
if channel_1h is None or channel_4h is None:
    return None  # Skip timestamp

# NEW:
# Allow None channels (very rare, only if no data)
# Will handle in scoring below
```

**Location (~line 2537-2551): Add quality scores to label dict**
```python
return {
    'timestamp': ts,
    'label': label,
    'continues': float(continues),
    'duration_hours': actual_duration_hours,
    'projected_gain': max_gain,
    'confidence': confidence,
    'score': score,
    'rsi_1h': rsi_1h,
    'rsi_4h': rsi_4h,
    'slope_1h': slope_1h,
    'slope_4h': slope_4h,
    # NEW: Channel quality signals for timeframe switching
    'channel_1h_cycles': channel_1h.complete_cycles if channel_1h else 0,
    'channel_4h_cycles': channel_4h.complete_cycles if channel_4h else 0,
    'channel_1h_r_squared': channel_1h.r_squared if channel_1h else 0.0,
    'channel_4h_r_squared': channel_4h.r_squared if channel_4h else 0.0,
    'channel_1h_valid': channel_1h.is_valid if channel_1h else 0.0,
    'channel_4h_valid': channel_4h.is_valid if channel_4h else 0.0,
    # Adaptive fields
    'adaptive_horizon': adaptive_horizon,
    'conf_score': conf_score
}
```

---

### File: src/ml/hierarchical_dataset.py

**Location (~line 351-365): Add new continuation target fields**
```python
# Add adaptive fields if they exist (when using adaptive mode)
if 'adaptive_horizon' in cont_row.columns:
    targets['adaptive_horizon'] = torch.tensor(cont_row['adaptive_horizon'].iloc[0], dtype=config.get_torch_dtype())
if 'conf_score' in cont_row.columns:
    targets['conf_score'] = torch.tensor(cont_row['conf_score'].iloc[0], dtype=config.get_torch_dtype())

# ADD: Channel quality fields (v3.17 - for timeframe switching)
if 'channel_1h_cycles' in cont_row.columns:
    targets['channel_1h_cycles'] = torch.tensor(cont_row['channel_1h_cycles'].iloc[0], dtype=config.get_torch_dtype())
if 'channel_4h_cycles' in cont_row.columns:
    targets['channel_4h_cycles'] = torch.tensor(cont_row['channel_4h_cycles'].iloc[0], dtype=config.get_torch_dtype())
if 'channel_1h_valid' in cont_row.columns:
    targets['channel_1h_valid'] = torch.tensor(cont_row['channel_1h_valid'].iloc[0], dtype=config.get_torch_dtype())
if 'channel_4h_valid' in cont_row.columns:
    targets['channel_4h_valid'] = torch.tensor(cont_row['channel_4h_valid'].iloc[0], dtype=config.get_torch_dtype())
if 'channel_1h_r_squared' in cont_row.columns:
    targets['channel_1h_r_squared'] = torch.tensor(cont_row['channel_1h_r_squared'].iloc[0], dtype=config.get_torch_dtype())
if 'channel_4h_r_squared' in cont_row.columns:
    targets['channel_4h_r_squared'] = torch.tensor(cont_row['channel_4h_r_squared'].iloc[0], dtype=config.get_torch_dtype())
```

---

## Cache Invalidation Required

**MUST DELETE before running:**
```bash
rm -rf data/feature_cache/features_mmap_meta_*.json
rm -rf data/feature_cache/*.npy
rm -rf data/feature_cache/continuation_labels_*.pkl
```

**Why:**
- Feature dimension changes: 12,639 → 14,487 (+1,848 features)
- Continuation label structure changes (+6 quality fields)
- Complete_cycles values different from ping_pongs

---

## Expected Outcomes

### Feature Count
```
Current: 21 windows × 11 tfs × 15 metrics × 2 symbols = 6,930 channels + 165 = 12,639 total
New:     21 windows × 11 tfs × 19 metrics × 2 symbols = 8,778 channels + 165 = 8,943 total
```

**Note:** Actual may differ due to extraction code details, will verify at runtime

### Continuation Labels
```
Current: ~1.2M labels (skips timestamps with bad channels)
New:     ~1.35M labels (keeps all, adds quality scores)
         +150K more training examples!
```

### Memory Impact (17GB Mac, float32, batch_size=8)
```
Current: 8 × 200 × 12,639 × 4 = 81 MB/batch
New:     8 × 200 × 14,487 × 4 = 93 MB/batch (+12 MB)
Still safe! ✅
```

---

## Testing Plan

1. ✅ Verify complete_cycles methods work correctly
2. 🔄 Complete remaining code changes
3. 🔄 Delete all caches
4. 🔄 Run feature extraction (~40 min)
5. 🔄 Run continuation label generation (~60 min with quality calc)
6. 🔄 Verify new dimensions match
7. 🔄 Start training
8. 🔄 Check if model learns to use channel quality for TF switching

---

## What This Enables

**The Model Will Learn:**

**Pattern 1: Timeframe Switching**
```
Input: channel_1h_cycles=0, channel_4h_cycles=4
Label: continues=1, uses 4h_slope
Model learns: "When 1h unreliable + 4h reliable → trust 4h for prediction"
```

**Pattern 2: Ranging Markets**
```
Input: channel_1h_cycles=1, channel_4h_cycles=1 (both weak)
Label: continues=0, ranges
Model learns: "Both TFs weak → ranging, don't predict continuation"
```

**Pattern 3: Your Exact Scenario!**
```
Input: channel_1h_cycles=0, channel_1h at top, channel_4h_cycles=5, rsi_4h=80
Label: breaks_down, gain=-2%
Model learns: "Weak 1h + strong but overbought 4h → downside break imminent"
```

**This is the adaptive timeframe intelligence you wanted! 🎯**
