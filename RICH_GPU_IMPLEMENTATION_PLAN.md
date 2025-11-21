# RICH Progress for GPU Sequential Mode - Implementation Plan

**Status:** Ready to implement in next session
**Complexity:** Medium (requires code restructuring)
**Lines:** ~40 added (via helper function pattern)

---

## Current Issue

GPU sequential mode (lines 811-927) uses basic tqdm, while CPU parallel uses beautiful RICH.
Direct wrapping causes code duplication (100+ lines duplicated for RICH vs tqdm paths).

---

## Clean Solution: Callback Pattern

**Extract processing logic to helper, wrap with different progress displays:**

```python
def _process_timeframe_sequential(self, symbol, tf_name, tf_rule, df, timeframes, ...):
    """Helper: Process one timeframe, return features dict"""
    # All the processing code (lines 820-915)
    # Returns: features_dict for this timeframe

# Then in main code:
if use_rich:
    with Progress(...) as progress:
        for symbol, tf in ...List:
            features = self._process_timeframe_sequential(...)
            progress.update(advance=1)
else:
    with tqdm(...) as pbar:
        for symbol, tf in timeframes:
            features = self._process_timeframe_sequential(...)
            pbar.update(1)
```

---

## Implementation Steps

1. Extract lines 820-915 to `_process_timeframe_sequential()` helper
2. Wrap main loop with RICH (try/except)
3. Fallback to tqdm if RICH unavailable
4. Both call same helper (no duplication!)

---

## Estimated Changes

**File:** src/ml/features.py
**Lines added:** ~40 (helper function signature + RICH wrapper)
**Lines modified:** ~10 (extract to helper call)
**Complexity:** Medium
**Time:** 30 minutes

---

## Benefits

✅ RICH progress for GPU (both NVIDIA and Mac MPS)
✅ No code duplication
✅ Clean fallback to tqdm
✅ Maintainable

---

**Recommend implementing in next session with fresh context.**
