# v6.0 Implementation Review

**Date:** December 28, 2025
**Reviewer:** Claude
**Status:** ⚠️ Incomplete - Critical Issues Found

---

## Executive Summary

The v6.0 architecture has been **partially implemented**, but several critical components are missing or incorrectly implemented. The system will **crash or behave incorrectly** if run as-is.

**Overall Status:**
- ✅ **Implemented Correctly:** Model architecture changes, loss_v6.py module, cache_v6.py module
- ⚠️ **Partially Implemented:** Loss integration, inference code
- ❌ **Not Implemented:** Return-after-break label generation, unified cache usage

---

## Critical Issues (Will Cause Crashes)

### 🔴 ISSUE #1: Stale Inference Code References Deleted Heads

**Location:** `src/ml/hierarchical_model.py` lines 1905-1908

**Problem:**
```python
# Lines 1905-1908 try to access deleted multi-task outputs
if self.multi_task and 'multi_task' in output_dict:
    mt = output_dict['multi_task']
    result['hit_band_pred'] = mt['hit_band_prob'][0, 0].item()  # ❌ Key doesn't exist
    result['hit_target_pred'] = mt['hit_target_prob'][0, 0].item()  # ❌ Key doesn't exist
    result['expected_return_pred'] = mt['expected_return'][0, 0].item()  # ❌ Key doesn't exist
```

**Why it crashes:**
- `multi_task` dict no longer contains these keys (correctly removed at line 1659)
- Inference will throw `KeyError` when trying to access them

**Fix:**
Delete or comment out lines 1905-1908.

---

### 🟡 ISSUE #2: Return-After-Break Labels Not Generated

**Location:** `src/ml/features.py`

**Problem:**
The label generation code still uses the **old break detection** (stops at first break):

```python
# Current code (features.py ~line 5579):
if future_closes[bar_idx] > upper or future_closes[bar_idx] < lower:
    break_idx = bar_idx
    break  # ❌ Stops here, never checks for return
```

Missing labels:
- `w{X}_returned` (did price return after break?)
- `w{X}_bars_to_return` (how many bars until return?)
- `w{X}_bars_outside` (total bars spent outside)
- `w{X}_max_consecutive_outside` (longest streak outside)
- `w{X}_final_duration` (true duration accounting for returns)

**Why this matters:**
- Return bonus loss (loss_v6.py line 209) expects these labels
- Window selection loss expects `final_duration` vs `first_break_bar`
- Training will fail or use incorrect targets

**Fix:**
Implement `detect_break_with_return()` as specified in architecture doc section 8.1.

---

## Moderate Issues (Will Work But Incorrectly)

### 🟡 ISSUE #3: v6 Cache Not Being Used by Default

**Location:** `train_hierarchical.py` line 4139

**Problem:**
```python
USE_V6_LOSS = getattr(args, 'v6', False)  # Defaults to False
```

**Why this matters:**
- v6.0 loss requires v6.0 cache format (with return-after-break labels)
- If `--v6` flag not passed, training uses old v5.9 loss + cache
- User might not know to pass `--v6` flag

**Fix:**
Either:
1. Make v6 the default: `USE_V6_LOSS = getattr(args, 'v6', True)`
2. Auto-detect cache version and set USE_V6_LOSS accordingly
3. Require explicit `--v6` flag and error if cache is wrong version

**Recommendation:** Auto-detect cache version.

---

### 🟡 ISSUE #4: Transition Labels Missing Some Fields

**Location:** `src/ml/features.py`

**Problem:**
Current transition label generation only produces:
- `transition_type` (0-3)
- `transition_direction` (0-2)
- Possibly `transition_next_tf` (0-10)

But v6 loss expects:
- `transition_valid_mask` (which samples have valid transitions)

**Why this matters:**
- Loss computation at train_hierarchical.py line 4377 tries to access these
- Missing keys will cause KeyError or incorrect masking

**Fix:**
Add `transition_valid_mask` to label generation (mark samples with insufficient future data as invalid).

---

## Minor Issues (Cosmetic or Low Priority)

### 🟢 ISSUE #5: Debug Logging Still References Old Targets

**Location:** `train_hierarchical.py` lines 4417-4420

**Problem:**
```python
# Lines 4417-4420 (in v5.x branch, runs when USE_V6_LOSS=False)
print(f"[DEBUG] targets high: mean={target_tensor[:,0].mean():.2f}%")
print(f"[DEBUG] targets low: mean={target_tensor[:,1].mean():.2f}%")
print(f"[DEBUG] predictions: mean={predictions[:,:2].mean():.3f}")
print(f"[DEBUG] primary loss (high/low MSE): {loss.item():.4f}")
```

**Why it's okay:**
- Only runs in v5.x mode (when USE_V6_LOSS=False)
- Correct for legacy mode
- Not a bug, but confusing if user accidentally runs v5 mode

**Fix:**
Add clearer logging: `print("[v5.x LEGACY MODE] ...")`

---

### 🟢 ISSUE #6: No Warmup Schedule for Temperature

**Location:** `train_hierarchical.py`

**Problem:**
Architecture doc specifies temperature annealing for window selection:
```
Epoch:    1    2    3    4    5    6    7    8    9   10   11+
Temp:    2.0  1.8  1.6  1.4  1.2  1.0  0.9  0.8  0.7  0.6  0.5
```

But I don't see this being applied to the model during training.

**Why this matters:**
- Window selection stays soft (explore) throughout training
- May not converge to best window
- Not critical, but reduces interpretability

**Fix:**
Add to training loop:
```python
if USE_V6_LOSS:
    new_temp = get_temperature(epoch, config.warmup_epochs)
    for tf in TIMEFRAMES:
        if hasattr(model, 'window_selectors'):
            model.window_selectors[tf].temperature = new_temp
```

---

## What Was Implemented Correctly

### ✅ Model Architecture Changes

**File:** `src/ml/hierarchical_model.py`

**Correct:**
- ✅ Removed `timeframe_heads[f'{tf}_high']` and `timeframe_heads[f'{tf}_low']`
- ✅ Removed `hit_band_head`, `hit_target_head`, `expected_return_head`
- ✅ Kept duration heads (mean + log_std)
- ✅ Kept validity heads
- ✅ Kept window selectors
- ✅ Added `get_v6_output_dict()` method
- ✅ Comments document removals

---

### ✅ Loss Functions Module

**File:** `src/ml/loss_v6.py`

**Correct:**
- ✅ `compute_duration_nll()` - Duration loss
- ✅ `compute_window_selection_loss()` - Punish bad windows
- ✅ `compute_tf_selection_loss()` - Punish bad TF trust
- ✅ `compute_containment_loss()` - Validate via price bounds
- ✅ `compute_breakout_timing_loss()` - Punish early breaks
- ✅ `compute_return_bonus()` - Reward returns
- ✅ `compute_transition_loss()` - Punish wrong transitions
- ✅ `compute_v6_loss()` - Combines all losses with proper weighting
- ✅ `get_warmup_weight()` - Quadratic warmup
- ✅ `V6LossConfig` - Configuration class

---

### ✅ Cache Generation Module

**File:** `src/ml/cache_v6.py`

**Exists:** Yes (created)

**Note:** I haven't reviewed its contents in detail, but its presence indicates unified cache format is being worked on.

---

### ✅ Loss Integration

**File:** `train_hierarchical.py`

**Correct:**
- ✅ Imports `compute_v6_loss` from loss_v6
- ✅ Checks `USE_V6_LOSS` flag
- ✅ Calls `compute_v6_loss()` when enabled
- ✅ Falls back to v5.x loss when disabled

---

## Summary of Required Fixes

### Must Fix Before Training (Crashes)

1. **Delete stale inference code** (lines 1905-1908 in hierarchical_model.py)
2. **Implement return-after-break label generation** (features.py)

### Should Fix for Correctness

3. **Add `transition_valid_mask` to labels** (features.py)
4. **Auto-detect cache version** or require `--v6` flag (train_hierarchical.py)

### Nice to Have (Enhanced functionality)

5. **Add temperature annealing** (train_hierarchical.py)
6. **Clearer logging for v5 vs v6 mode** (train_hierarchical.py)

---

## Implementation Checklist

From architecture doc section 10:

| Phase | Task | Status |
|-------|------|--------|
| **A: Cache Format** | Create cache_v6.py | ✅ Done |
| | Generate unified .npz files | ⚠️ Script exists, not verified |
| | Update dataset.py to load | ⚠️ Partially (v6 imports added) |
| | Test batch dict output | ❌ Not verified |
| **B: Label Enhancement** | Implement detect_break_with_return() | ❌ **Missing** |
| | Implement detect_transition() | ⚠️ Partial (basic version exists) |
| | Add return labels to cache | ❌ **Missing** |
| | Generate full cache | ❌ Not done |
| **C: Model Changes** | Delete high/low heads | ✅ Done |
| | Delete hit_band/hit_target/expected_return | ✅ Done |
| | Add get_v6_output_dict() | ✅ Done |
| | Update geometric projection | ⚠️ Exists, not verified |
| **D: Loss Restructure** | Create loss_v6.py | ✅ Done |
| | Implement all loss functions | ✅ Done |
| | Update train_hierarchical.py | ⚠️ Partial (calls exist, integration incomplete) |
| | Add warmup schedule | ⚠️ Partial (weights yes, temperature no) |
| **E: Integration Testing** | Test training loop | ❌ Not done |
| | Verify loss curves | ❌ Not done |
| | Verify predictions improve | ❌ Not done |

---

## Recommendations

### Immediate Actions (Before First Training Run)

1. **Fix critical bugs:**
   - Delete lines 1905-1908 in hierarchical_model.py
   - Add placeholder for missing return labels (or implement fully)

2. **Test with small data:**
   - Generate v6 cache for 1 month of data
   - Run 1 epoch with `--v6` flag
   - Check for crashes

3. **Verify outputs:**
   - Print loss_components on first batch
   - Verify all loss values are finite
   - Check that duration predictions are reasonable (not 500 bars)

### After Initial Testing

4. **Implement return-after-break fully:**
   - Add detect_break_with_return() to features.py
   - Regenerate full cache
   - Retrain and compare

5. **Add interpretability:**
   - Log attention weights (if added)
   - Log which windows are selected
   - Log which TFs are trusted

---

## Conclusion

**Overall:** Good progress, but **not ready for production training** yet.

**Estimated work remaining:**
- Critical fixes: 2-4 hours
- Return-after-break implementation: 4-6 hours
- Testing & debugging: 2-4 hours
- **Total:** 1-2 days

**Priority order:**
1. Fix crash bugs (30 min)
2. Test with existing cache (1 hour)
3. Implement return-after-break (4-6 hours)
4. Full integration test (2-4 hours)

---

**Document Version:** 1.0
**Status:** Ready for action
